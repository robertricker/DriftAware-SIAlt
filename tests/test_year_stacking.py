import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import driftaware_sialt.stacking.pipeline as pipeline
from driftaware_sialt.io_tools import read_dasit_csv
from driftaware_sialt.stacking.periods import RunPeriod


def _config(tmp_path):
    return {
        "stage": "stacking",
        "version": "2.0",
        "options": {
            "sensor": ["cryosat2"], "hemisphere": "sh", "out_epsg": "EPSG:6932",
            "target_variable": "sea_ice_freeboard", "add_variable": [],
            "ice_conc_products": ["concentration"], "ice_drift_products": ["drift"],
        },
        "input_dir": {},
        "auxiliary": {"ice_conc": {}, "ice_drift": {}},
        "output_dir": {"trajectories": str(tmp_path)},
        "run": RunPeriod(dt.date(2020, 12, 30), dt.date(2020, 12, 31),
                         dt.date(2020, 12, 26), dt.date(2021, 1, 4)).to_dict(),
        "stacking": {
            "t_window": 5, "mode": "fr", "filename_prefix": "ESACCI-SEAICE",
            "multiproc": False, "num_cpus": 2,
            "parcel_grid": {"bounds": [-1, -1, 1, 1], "dim": 1},
            "hist": {"n_bins": 40, "range": {"freeboard": [0, 1]}},
            "thermo_change": {"enabled": False},
            "growth_estimation": {
                "growth_grid": {"bounds": [-1, -1, 1, 1], "dim": 1},
                "growth_range": {"freeboard": [-0.25, 0.25]},
                "min_n_tiepoints": 99,
            },
        },
    }


def _stub_grid_and_logging(monkeypatch):
    monkeypatch.setattr(pipeline, "init_logger", lambda config: None)
    monkeypatch.setattr(pipeline.gridding_lib, "define_grid",
                        lambda *args, **kwargs: (object(), 1000))


@pytest.mark.parametrize("sensor", ["cryosat2", "icesat2"])
@pytest.mark.parametrize("missing_buffer", [False, True])
def test_year_boundary_csvs_have_full_lags_and_no_buffer_outputs(
    tmp_path, monkeypatch, sensor, missing_buffer,
):
    _stub_grid_and_logging(monkeypatch)
    config = _config(tmp_path)
    config["options"]["sensor"] = [sensor]
    if sensor == "icesat2":
        config["options"]["target_variable"] = "total_freeboard"
    original = copy.deepcopy(config)
    visited = []

    class Altimetry:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def get_file_list(self, directories):
            pass

        def get_file_dates(self):
            pass

        def get_target_files(self, start, end):
            self.date = start
            visited.append(start)
            self.target_files = {sensor: [] if missing_buffer and start < dt.datetime(2020, 12, 30) else ["input.nc"]}

        def get_product(self, sensors):
            self.product = pd.DataFrame({"total_freeboard_quality_flag": [0], "sensor": [sensor]})

    class Concentration:
        product_id = "concentration"
        target_files = "concentration.nc"

        def get_ice_concentration(self, file):
            return object()

        def interp_ice_concentration(self, data, x, y):
            return np.ones(len(x))

    class Drift:
        product_id = "drift"
        target_files = "drift.nc"
        ice_drift = {"time_bnds": np.array([0.0])}

        def get_ice_drift(self, file, concentration):
            pass

        def drift_correction(self, x, y):
            return np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))

        def deformation(self, x, y):
            return np.zeros(len(x)), np.zeros(len(x))

    def baseline(processor, *args, **kwargs):
        data = pd.DataFrame({
            processor.target_var: [0.5],
            processor.target_var + "_drift_unc": [0.0],
            "xu": [0.0], "yu": [0.0], "geometry": [[Point(0, 0)]],
            "t0": [0.0], "dt_days": [0], "divergence": [[0]], "shear": [[0]],
            "source_date": [processor.parent.date.strftime("%Y-%m-%d")],
            "_ice_conc_products": ["concentration"], "_ice_drift_products": [""],
        })
        if sensor == "icesat2":
            processor.master["gt1l"][processor.i][0] = data
        else:
            processor.master[processor.i][0] = data

    monkeypatch.setattr(pipeline, "SeaIceThicknessMultiProducts", Altimetry)
    monkeypatch.setattr(pipeline.SeaIceConcentrationProducts, "load_products", lambda *args, **kwargs: [Concentration()])
    monkeypatch.setattr(pipeline.SeaIceDriftProducts, "load_products", lambda *args, **kwargs: [Drift()])
    monkeypatch.setattr(pipeline, "select_aux_product", lambda products, *args: products[0])
    monkeypatch.setattr(pipeline.DriftAwareProcessor, "baseline_proc", baseline)

    pipeline.run(config)

    files = sorted(Path(tmp_path).rglob("*.csv"))
    assert len(files) == 2
    assert "20201230" in files[0].name
    assert "20201231" in files[1].name
    for file in files:
        data = read_dasit_csv(file)
        expected_lags = range(0 if missing_buffer else -4, 5)
        if missing_buffer and "20201231" in file.name:
            expected_lags = range(-1, 5)
        assert sorted(data.dt_days) == list(expected_lags)
        target = dt.datetime.strptime("20201230" if "20201230" in file.name else "20201231", "%Y%m%d")
        assert sorted(data.source_date) == [
            (target + dt.timedelta(days=lag)).strftime("%Y-%m-%d") for lag in expected_lags]
    assert min(visited) == dt.datetime(2020, 12, 26)
    assert max(visited) == dt.datetime(2021, 1, 4)
    assert config == original
