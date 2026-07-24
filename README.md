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
`base_dir`. Absolute paths are accepted. `{version}` and `{sensor}` placeholders
in paths are expanded while loading the configuration; multiple sensors are
joined with underscores.

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

Stacking direction `f` means forward, `r` means reverse, and `fr` runs and merges
both directions. `t_length` accepts an integer number of days, `season`, or
`all`. A season ends on 1 May in the Northern Hemisphere and 1 November in the
Southern Hemisphere; `all` processes that season for every year available from
any configured sensor, starting with the year in `t_start`. Grid bounds use
`[xmin, ymin, xmax, ymax]` in metres in `options.out_epsg`; `dim` is the number
of cells along each axis.

The thermodynamic correction currently supports `winter_2layers` with ERA5 air
temperature and a constant ocean heat flux in W m-2. It requires `snow_depth`.
Ocean-heat-flux reanalysis input is not implemented.

Gridding mode `da` places values at their advected target location; `cv` places
them at the original acquisition location. `gridding.csv_dir` accepts a
run-directory name below `output_dir.trajectories`, an absolute directory, or
`all` for recursive input. With weighting disabled, ordinary means are used.
Weighting can use `counts` for the combined CryoSat-2/Sentinel-3A/Sentinel-3B
case, or a list of numeric trajectory columns such as `[dt_days]`.

Visualization has plotting presets for thickness, freeboard, total freeboard,
snow depth, their uncertainty/change fields, acquisition distance and time
offset, shear, and divergence. The selected variable must exist in the input
NetCDF files. GIF generation requires the ImageMagick `convert` command.

Run the stages in order:

```bash
driftaware config/cci/stacking.yaml
driftaware config/cci/gridding.yaml
driftaware config/cci/visualization.yaml
```

The equivalent module invocation is:

```bash
python -m driftaware_sialt config/cci/stacking.yaml
```

Trajectory CSV files are self-describing. Their first line is a JSON metadata
header containing `format_version`, CRS, target variable, stack mode and window,
and histogram settings. Gridding reads these values from the selected CSV files,
so they are not repeated in `gridding.yaml`. Files combined in one gridding run
must have compatible metadata.

Product filenames place the processing mode immediately after the product level:

```text
ESACCI-SEAICE-L2P-DA-SITHICK-SIRAL_CRYOSAT2-NH-20211001-fv1.1.csv
ESACCI-SEAICE-L3C-DA-SITHICK-SIRAL_CRYOSAT2-NH_25KM_EASE2-20211001-fv1.1.nc
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
