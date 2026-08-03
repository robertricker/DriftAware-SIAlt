# DriftAware-SIAlt

## Installation

Install the project in editable mode from the repository root:

```bash
python -m pip install -e .
```

This provides the `driftaware` command and makes the `src/driftaware_sialt`
package importable while developing.

The main processing code is organized by stage:

```text
src/driftaware_sialt/
├── cli.py
├── products/
│   ├── selection.py
│   ├── sea_ice_concentration.py
│   ├── sea_ice_drift.py
│   └── sea_ice_thickness.py
├── filters/
│   └── marginal_ice_zone.py
├── stacking/
│   ├── pipeline.py
│   └── point_density_correction.py
└── volume/
    ├── pipeline.py
    ├── calculations.py
    ├── density.py
    ├── interpolation.py
    ├── output.py
    └── netcdf_config.yaml
```

## Configuration

Configuration is split by responsibility:

```text
config/
└── cci/
    ├── common.yaml          # repositories, products, sensor, target, hemisphere
    ├── stacking.yaml        # drift-aware trajectory processing
    ├── gridding.yaml        # trajectory-to-NetCDF processing
    └── visualization.yaml   # plots and optional GIF generation
```

Each stage file inherits `common.yaml` through its relative `extends` entry. The
processing selector is a top-level `stage` field, and the settings for that stage
are in a sibling block:

```yaml
extends: common.yaml

stage: stacking

stacking:
  mode: fr
  t_window: 16
```

The shared `options` block in `common.yaml` now contains only scientific options
used across stages; it no longer contains `proc_step` or `proc_step_options`.

### Configuration reference

All relative input, output, auxiliary, and logging paths are resolved from
`base_dir`. Absolute paths are accepted. `{version}`, `{sensor}`, and
`{hemisphere}` placeholders in paths are expanded while loading the
configuration; multiple sensors are joined with underscores.

Supported sensor and target-variable combinations are:

- `envisat`, `cryosat2`, `sentinel3a`, `sentinel3b`: `sea_ice_thickness`,
  `sea_ice_freeboard`, or `radar_freeboard`.
- `icesat2`: `total_freeboard` or `sea_ice_thickness`.

Use hemisphere `nh` with `EPSG:6931`, and `sh` with `EPSG:6932`. Altimetry input
paths must point to the corresponding hemisphere. Additional source variables,
such as `snow_depth`, are listed under `options.add_variable`; use `[]` when none
are required.

Sea-ice concentration and drift products are listed in priority order. The first
exact match is selected. When no exact match exists, the closest file from any
configured product is used if it is no more than seven days away. Supported
concentration readers are `osi450`, `osi430`, and `c3s`; supported drift readers
are `osi455`, `osi435`, and `osi405`.

The separate `driftaware syncdata` command inventories every remote file in the
date span implied by `t_start`, `t_length`, and `mode`, then downloads files
whose names are absent from the corresponding local repository. Synchronization
does not run automatically as part of stacking. Concentration and drift
synchronization is limited to the products selected in their priority lists.
Altimetry synchronization is limited to `options.sensor`. Products without a
remote URL remain local-only. For example:

```yaml
remote_dir:
  altimetry:
    cryosat2: "ftp://ftp.awi.de/sea_ice/projects/cci/crdp/v4p0/l2p_release/{hemisphere}/cryosat2"
    sentinel3a: "ftp://ftp.awi.de/sea_ice/projects/cci/crdp/v4p0/l2p_release/{hemisphere}/sentinel3a"
    sentinel3b: "ftp://ftp.awi.de/sea_ice/projects/cci/crdp/v4p0/l2p_release/{hemisphere}/sentinel3b"
    envisat: "ftp://ftp.awi.de/sea_ice/projects/cci/crdp/v4p0/l2p_release/{hemisphere}/envisat"
  ice_conc:
    osi450: "ftp://osisaf.met.no/reprocessed/ice/conc/v3p1"
    osi430: "ftp://osisaf.met.no/reprocessed/ice/conc-cont-reproc/v3p0"
  ice_drift:
    osi455: "ftp://osisaf.met.no/reprocessed/ice/drift_lr/v1/merged"
    osi405: "ftp://osisaf.met.no/archive/ice/drift_lr/merged"
```

This is a download-only sync: extra local files are retained. Downloads are
placed in `YYYY/MM` subdirectories and become visible to the existing recursive
readers. A partial transfer is never exposed as an input file.

Stacking direction `f` means forward, `r` means reverse, and `fr` runs and merges
both directions. `t_length` accepts an integer number of days, `season`, or
`all`. A season ends on 1 May in the Northern Hemisphere and 1 November in the
Southern Hemisphere; `all` processes that season for every year available from
any configured sensor, starting with the year in `t_start`. Grid bounds use
`[xmin, ymin, xmax, ymax]` in metres in `options.out_epsg`; `dim` is the number
of cells along each axis.

Filled coastal drift can be reduced linearly between the last valid drift cell
and adjacent product land cells. Enable or disable this independently of the
selected drift product:

```yaml
coastal_drift_taper:
  enabled: true
```

Setting `enabled: false` restores the previous untapered spatial filling. The
taper changes displacement only; it does not reduce drift uncertainty.

The thermodynamic correction currently supports `winter_2layers` with ERA5 air
temperature and a constant ocean heat flux in W m-2. It requires `snow_depth`.
Ocean-heat-flux reanalysis input is not implemented.

Gridding mode `da` places values at their advected target location; `cv` places
them at the original acquisition location. `gridding.csv_dir` accepts a
run-directory name below `output_dir.trajectories`, an absolute directory, or
`all` for recursive input. With weighting disabled, ordinary means are used.
Enable count weighting with:

```yaml
weighting:
  enabled: true
  var_to_weight_with: counts
```

Count weighting is evaluated separately in every output grid cell. When
CryoSat-2 and Sentinel-3 observations are both available, half of the nominal
weight is assigned to CryoSat-2 and half to the Sentinel-3 family. The
Sentinel-3 share is divided equally between Sentinel-3A and Sentinel-3B when
both are present. Missing missions are ignored and the available shares are
renormalized. Individual trajectory parcels are weighted by the sum of these
mission shares multiplied by the square root of their corresponding observation
counts.

Time-distance weighting is available as an alternative:

```yaml
weighting:
  enabled: true
  var_to_weight_with: dt_days
```

It applies the symmetric weight `1 / (1 + abs(dt_days))**2`, giving the greatest
weight to observations acquired on the target date. Both `dt_days` and
`[dt_days]` are accepted; the string form is recommended. An empty
`var_to_weight_with` also defaults to `dt_days`, although setting it explicitly
makes the processing choice clearer. `counts` and `dt_days` are alternative
weighting modes.

Visualization presets use the current gridded NetCDF variable names. They cover
sea-ice thickness and its uncertainty/change fields, model thickness and
thermodynamic tendencies, radar and sea-ice freeboard, snow depth, sea-ice
concentration, acquisition distance and time offset, displacement uncertainty,
deformation, shear, divergence, model air temperature, and model ocean heat
flux. The selected variable must exist in the input NetCDF files. GIF generation
requires the ImageMagick `convert` command.

Synchronize the date-dependent inputs separately, then run the stages:

```bash
driftaware syncdata config/cci/stacking.yaml
driftaware stacking config/cci/stacking.yaml
driftaware gridding config/cci/gridding.yaml
driftaware visualization config/cci/visualization.yaml
driftaware volume config/cci/volume.yaml
```

The equivalent module synchronization invocation is:

```bash
python -m driftaware_sialt syncdata config/cci/stacking.yaml
```

Volume calculation operates on an existing gridded run. Select its directory
below `output_dir.gridded_data` in `volume.yaml`:

```yaml
volume:
  gridded_data_dir: "sea_ice_thickness-nh-16fr-epsg6931_250_15-..."
  ice_conc_product:
  sea_ice_density: ice_fons_2022
  snow_density: snow_fons_2022
  interp_missing_sit: true
  sic_interp_threshold: 15
```

An empty `ice_conc_product` uses the concentration already stored in the
gridded files. Set it to a configured product ID such as `osi430` to read
concentration externally. Each run creates a timestamped
`sea_ice_volume-{hemisphere}-...` directory below `output_dir.volume`.
Grid-cell area is calculated directly from the gridded product's `xc` and `yc`
coordinate spacing in metres. The volume summary CSV keeps the NetCDF SI units:
area and extent in m2, volume in m3, and mass in kg.

Trajectory CSV files are self-describing. Their first line is a JSON metadata
header containing `format_version`, CRS, target variable, stack mode and window,
and histogram settings. Gridding reads these values from the selected CSV files,
so they are not repeated in `gridding.yaml`. Files combined in one gridding run
must have compatible metadata.
Trajectory geometry is stored as WKT with projected coordinates rounded to the
nearest metre. Floating-point columns are rounded to four decimal places, and
sensor observation-count columns ending in `_cnt` are stored as integers.

Product filenames place the processing mode immediately after the product level:

```text
ESACCI-SEAICE-L2P-DA-SITHICK-SIRAL_CRYOSAT2-NH-20211001-fv1.1.csv
ESACCI-SEAICE-L3C-DA-SITHICK-SIRAL_CRYOSAT2-NH_25KM_EASE2-20211001-fv1.1.nc
ESACCI-SEAICE-L4-DA-SIVOL-SIRAL_CRYOSAT2-NH_25KM_EASE2-20211001-fv1.1.nc
```

Optional processing is controlled explicitly. For example:

```yaml
stacking:
  climatology:
    sit:
      enabled: false
      path:

  thermo_change:
    enabled: false
```

A disabled feature does not require its optional input paths.

Growth interpolation can correct latitude-band point density for land area. The
global vector mask is configured in `common.yaml`:

```yaml
auxiliary:
  land_mask: "/path/to/ne_50m_land.shp"
```

Enable or disable its use in `stacking.yaml`:

```yaml
stacking:
  growth_estimation:
    land_area_correction:
      enabled: true
```

## Operational scripts

SLURM submission templates are stored under `scripts/slurm/`.
