from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from driftaware_sialt.config_loader import load_config
from driftaware_sialt.volume.calculations import (
    calculate_volume_and_mass,
    infer_grid_cell_area,
)
from driftaware_sialt.volume import pipeline
from driftaware_sialt.volume import interpolation
from driftaware_sialt.volume.output import (
    build_regional_summary,
    write_volume_dataset,
)


ROOT = Path(__file__).parents[1]


def test_volume_configuration_resolves_required_settings():
    config = load_config(ROOT / "config/cci/volume.yaml")

    assert config["stage"] == "volume"
    assert Path(config["output_dir"]["volume"]).is_absolute()
    assert config["volume"]["gridded_data_dir"]
    assert "resolution" not in config["volume"]
    assert config["volume"]["sea_ice_density"] == "ice_fons_2022"
    assert config["volume"]["snow_density"] == "snow_fons_2022"


def test_volume_calculation_accepts_time_dimension_and_percent_threshold():
    data = xr.Dataset(
        {
            "sea_ice_thickness": (
                ("time", "yc", "xc"),
                np.ones((1, 2, 2)),
            ),
            "snow_depth": (
                ("time", "yc", "xc"),
                np.full((1, 2, 2), 0.2),
            ),
        },
        coords={
            "time": [0],
            "yc": [0, 25000],
            "xc": [0, 25000],
        },
    )
    concentration_percent = xr.DataArray(
        np.array([[[10.0, 20.0], [50.0, 100.0]]]),
        dims=("time", "yc", "xc"),
    )

    result = calculate_volume_and_mass(
        data,
        concentration_percent,
        "sea_ice_thickness",
        900,
        300,
        interp_missing_sit=False,
        sic_threshold=15,
    )

    cell_area = (25 * 1000) ** 2
    np.testing.assert_allclose(
        result["sea_ice_area"].values,
        np.array([[[0.1, 0.2], [0.5, 1.0]]]) * cell_area,
    )
    np.testing.assert_allclose(
        result["sea_ice_extent"].values,
        np.array([[[0, 1], [1, 1]]]) * cell_area,
    )
    assert result["sea_ice_area"].dims == ("time", "yc", "xc")
    assert result["sea_ice_extent"].dims == ("time", "yc", "xc")


def test_volume_pipeline_writes_gridded_output_and_summary(tmp_path):
    run_name = "gridded-run"
    input_dir = tmp_path / "l3c" / run_name / "2022" / "01"
    input_dir.mkdir(parents=True)
    filename = (
        "ESACCI-SEAICE-L3C-DA-SITHICK-SIRAL_CRYOSAT2-NH_"
        "25KM_EASE2-20220115-fv1.1.nc"
    )
    timestamp = np.datetime64("2022-01-15T12:00:00").astype(
        "datetime64[s]").astype(int)
    dataset = xr.Dataset(
        {
            "sea_ice_thickness": (
                ("time", "yc", "xc"),
                np.ones((1, 2, 2)),
            ),
            "snow_depth": (
                ("time", "yc", "xc"),
                np.full((1, 2, 2), 0.2),
            ),
            "sea_ice_concentration": (
                ("time", "yc", "xc"),
                np.array([[[10.0, 20.0], [50.0, 100.0]]]),
            ),
            "region_code": (
                ("yc", "xc"),
                np.array([[0, 0], [1, 1]]),
            ),
            "longitude": (
                ("yc", "xc"),
                np.zeros((2, 2)),
            ),
            "latitude": (
                ("yc", "xc"),
                np.full((2, 2), 80.0),
            ),
            "time_bnds": (
                ("time", "nv"),
                np.array([[timestamp - 43200, timestamp + 43200]]),
            ),
            "crs": np.int32(-2147483648),
        },
        coords={
            "time": [timestamp],
            "yc": [0, 25000],
            "xc": [0, 25000],
        },
    )
    dataset["region_code"].attrs["flag_values"] = [0, 1]
    dataset.to_netcdf(input_dir / filename)

    config = {
        "options": {
            "hemisphere": "nh",
            "out_epsg": "EPSG:6931",
            "target_variable": "sea_ice_thickness",
        },
        "output_dir": {
            "gridded_data": str(tmp_path / "l3c"),
            "volume": str(tmp_path / "volume"),
        },
        "auxiliary": {"ice_conc": {}},
        "volume": {
            "gridded_data_dir": run_name,
            "ice_conc_product": None,
            "sea_ice_density": 900,
            "snow_density": 300,
            "interp_missing_sit": False,
            "sic_interp_threshold": 15,
        },
    }

    pipeline.run(config)

    output_runs = list(
        (tmp_path / "volume").glob("sea_ice_volume-nh-*"))
    assert len(output_runs) == 1
    output_root = output_runs[0]
    volume_filename = filename.replace(
        "-L3C-DA-SITHICK-",
        "-L4-DA-SIVOL-",
    )
    volume_file = output_root / "2022" / "01" / volume_filename
    assert volume_file.is_file()

    summary_filename = volume_filename.replace(
        "-20220115-",
        "-20220115-20220115-",
    ).replace(".nc", ".csv")
    assert (output_root / summary_filename).is_file()

    with xr.open_dataset(volume_file, decode_times=False) as output:
        assert output.attrs["processing_level"] == "L4"
        assert output.attrs["title"] == "Gridded sea ice volume and mass"
        assert "sea_ice_thickness" not in output
        assert "snow_depth" not in output
        assert "sea_ice_concentration" not in output
        assert {
            "time",
            "time_bnds",
            "xc",
            "yc",
            "longitude",
            "latitude",
            "crs",
            "region_code",
            "sea_ice_volume",
            "sea_ice_mass",
            "snow_volume",
            "snow_mass",
            "sea_ice_extent",
            "sea_ice_area",
            "sea_ice_extent_total",
            "sea_ice_area_total",
            "sea_ice_volume_total",
            "sea_ice_mass_total",
        } <= set(output.variables)


def test_grid_cell_area_is_inferred_from_rectangular_coordinate_spacing():
    data = xr.Dataset(
        coords={
            "xc": ("xc", [0, 20000, 40000], {"units": "m"}),
            "yc": ("yc", [60000, 30000, 0], {"units": "m"}),
        },
    )

    assert infer_grid_cell_area(data) == 20000 * 30000


def test_irregular_grid_spacing_is_rejected():
    data = xr.Dataset(
        coords={
            "xc": ("xc", [0, 20000, 50000], {"units": "m"}),
            "yc": ("yc", [0, 25000, 50000], {"units": "m"}),
        },
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "xc and yc must have regular",
    ):
        infer_grid_cell_area(data)


def test_volume_writer_discards_conflicting_source_encoding(tmp_path):
    dataset = xr.Dataset(
        {
            "sea_ice_thickness": (
                ("yc", "xc"),
                np.ones((2, 2)),
                {"coordinates": "longitude latitude"},
            ),
            "longitude": (("yc", "xc"), np.zeros((2, 2))),
            "latitude": (("yc", "xc"), np.zeros((2, 2))),
        },
        coords={"yc": [0, 25000], "xc": [0, 25000]},
    )
    dataset["sea_ice_thickness"].encoding["coordinates"] = (
        "longitude latitude")
    outfile = tmp_path / "volume.nc"

    write_volume_dataset(dataset, str(outfile))

    assert outfile.is_file()
    with xr.open_dataset(outfile) as written:
        assert "sea_ice_thickness" in written


def test_volume_calculation_rejects_non_thickness_target():
    data = xr.Dataset(
        {
            "sea_ice_freeboard": (
                ("time", "yc", "xc"),
                np.ones((1, 2, 2)),
            ),
            "snow_depth": (
                ("time", "yc", "xc"),
                np.ones((1, 2, 2)),
            ),
        },
        coords={"time": [0], "yc": [0, 25000], "xc": [0, 25000]},
    )

    with pytest.raises(ValueError, match="requires target_variable"):
        calculate_volume_and_mass(
            data,
            np.ones((2, 2)),
            "sea_ice_freeboard",
            900,
            300,
        )


def test_interpolation_uses_threshold_and_inferred_grid_spacing(monkeypatch):
    captured = {}

    class FakeInterpolator:
        def __init__(self, points, values, **kwargs):
            captured.update(kwargs)

        def __call__(self, points):
            captured["target_points"] = points
            return np.zeros(len(points))

    monkeypatch.setattr(
        interpolation, "RBFInterpolator", FakeInterpolator)
    result = interpolation._interpolate_2d_rbf(
        np.ones((2, 2)),
        np.array([[0.2, 0.6], [0.9, 0.1]]),
        np.array([0, 20000]),
        np.array([0, 30000]),
        sic_threshold=0.8,
    )

    assert captured["epsilon"] == pytest.approx(
        1.8 / np.sqrt(20000 * 30000))
    assert captured["target_points"].shape == (1, 2)
    assert np.isfinite(result).sum() == 1


def test_regional_summary_uses_si_units_and_excludes_global_totals():
    cell_area = 100.0
    concentration = np.array([[[0.1, 0.2], [0.5, 1.0]]])
    dataset = xr.Dataset(
        {
            "sea_ice_area": (
                ("time", "yc", "xc"),
                concentration * cell_area,
            ),
            "sea_ice_extent": (
                ("time", "yc", "xc"),
                (concentration > 0.15) * cell_area,
            ),
            "sea_ice_volume": (
                ("time", "yc", "xc"),
                concentration * cell_area,
            ),
            "sea_ice_mass": (
                ("time", "yc", "xc"),
                concentration * cell_area * 900,
            ),
            "snow_volume": (
                ("time", "yc", "xc"),
                concentration * cell_area * 0.2,
            ),
            "snow_mass": (
                ("time", "yc", "xc"),
                concentration * cell_area * 0.2 * 300,
            ),
            "sea_ice_area_total": (
                ("time",),
                [concentration.sum() * cell_area],
            ),
            "sea_ice_extent_total": (
                ("time",),
                [(concentration > 0.15).sum() * cell_area],
            ),
            "sea_ice_volume_total": (
                ("time",),
                [concentration.sum() * cell_area],
            ),
            "sea_ice_mass_total": (
                ("time",),
                [concentration.sum() * cell_area * 900],
            ),
            "region_code": (
                ("time", "yc", "xc"),
                np.array([[[0, 0], [1, 1]]]),
            ),
        },
        coords={"time": [0], "yc": [0, 1], "xc": [0, 1]},
    )
    dataset["region_code"].attrs["flag_values"] = [0, 1]

    summary = build_regional_summary(dataset)

    assert summary["sea_ice_area_0"].iloc[0] == pytest.approx(
        0.3 * cell_area)
    assert summary["sea_ice_area_1"].iloc[0] == pytest.approx(
        1.5 * cell_area)
    assert summary["sea_ice_area_total"].iloc[0] == pytest.approx(
        1.8 * cell_area)
    assert "sea_ice_area_total_0" not in summary
