# DriftAware-SIAlt

## Configuration

Configuration is split by responsibility:

```text
config/
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

Run the stages in order:

```bash
python main.py config/stacking.yaml
python main.py config/gridding.yaml
python main.py config/visualization.yaml
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
