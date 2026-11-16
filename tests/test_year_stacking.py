import copy
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

import driftaware_sialt.input_data as input_data
import driftaware_sialt.stacking.pipeline as pipeline
from driftaware_sialt.io_tools import read_dasit_csv
from driftaware_sialt.stacking.periods import year_stacking_period


def _config(tmp_path, start=dt.datetime(2020, 12, 30), window=5):
    return {
        "stage": "stacking",
        "version": "4p0",
        "options": {
            "sensor": ["cryosat2"],
            "hemisphere": "sh",
            "out_epsg": "EPSG:6932",
            "target_variable": "sea_ice_freeboard",
            "add_variable": [],
            "ice_conc_products": ["concentration"],
            "ice_drift_products": ["drift"],
        },
        "input_dir": {},
        "auxiliary": {"ice_conc": {}, "ice_drift": {}},
        "output_dir": {"trajectories": str(tmp_path)},
        "stacking": {
            "t_start": start,
            "t_length": "year",
            "t_window": window,
            "mode": "fr",
            "filename_prefix": "ESACCI-SEAICE",
            "multiproc": False,
            "num_cpus": 2,
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


@pytest.mark.parametrize("year, days", [(2019, 365), (2020, 366)])
@pytest.mark.parametrize("window", [1, 5, 16])
def test_year_period_and_input_dates_include_both_buffers(tmp_path, year, days, window):
    start = dt.datetime(year, 1, 1)
    config = _config(tmp_path, start, window)
    original = copy.deepcopy(config)
    padding = dt.timedelta(days=window - 1)
    period = year_stacking_period(start, window, "sh")

    assert period.start == start - padding
    assert period.length == days + 2 * (window - 1)
    assert period.output_days == range(window - 1, window - 1 + days)
    required = input_data.required_input_dates(config)
    assert required["altimetry"] == [
        period.start + dt.timedelta(days=i) for i in range(period.length)]
    last = dt.datetime(year, 12, 31) + padding
    assert required["ice_conc"][0] == period.start - dt.timedelta(days=1)
    assert required["ice_conc"][-1] == last + dt.timedelta(days=1)
    assert required["ice_drift"][0] == period.start
    assert required["ice_drift"][-1] == last + dt.timedelta(days=1)
    assert config == original


@pytest.mark.parametrize("window", [0, -1, True, 2.5])
def test_year_rejects_invalid_window(window):
    with pytest.raises(ValueError, match="t_window must be a positive integer"):
        year_stacking_period(dt.datetime(2020, 1, 1), window, "sh")


def test_year_is_sh_only(tmp_path):
    config = _config(tmp_path)
    config["options"]["hemisphere"] = "nh"
    with pytest.raises(ValueError, match="requires hemisphere 'sh'"):
        input_data.required_input_dates(config)
    with pytest.raises(ValueError, match="requires hemisphere 'sh'"):
        pipeline.run(config)


def _stub_grid_and_logging(monkeypatch):
    monkeypatch.setattr(pipeline, "init_logger", lambda config: None)
    monkeypatch.setattr(
        pipeline.gridding_lib, "define_grid",
        lambda *args, **kwargs: (object(), 1000))


class _InlinePool:
    """Exercise the multiprocessing dispatch without spawning workers."""

    def __init__(self, workers):
        pass

    def apply_async(self, function, args):
        function(*args)

    def close(self):
        pass

    def join(self):
        pass


@pytest.mark.parametrize("multiproc", [False, True])
@pytest.mark.parametrize("mode", ["f", "r", "fr"])
def test_run_keeps_target_dates_separate_from_processing(tmp_path, monkeypatch, multiproc, mode):
    _stub_grid_and_logging(monkeypatch)
    config = _config(tmp_path, dt.datetime(2020, 7, 1))
    config["stacking"].update(multiproc=multiproc, mode=mode)
    original = copy.deepcopy(config)
    processed = []
    merged = []
    monkeypatch.setattr(pipeline.mp, "Pool", _InlinePool)
    monkeypatch.setattr(
        pipeline, "process_direction",
        lambda cfg, direction, grid, output_days: processed.append(
            (cfg["stacking"]["t_start"], cfg["stacking"]["t_length"], direction, output_days)))
    monkeypatch.setattr(
        pipeline, "merge_direction_stacks",
        lambda cfg, grid, growth_width, width, files_f, files_r, j: merged.append(
            cfg["stacking"]["t_start"] + dt.timedelta(days=j)))

    pipeline.run(config)

    assert processed == [
        (dt.datetime(2020, 6, 27), 192, direction, range(4, 188))
        for direction in mode]
    assert merged == [dt.datetime(2020, 7, 1) + dt.timedelta(days=i) for i in range(184)]
    assert config == original


@pytest.mark.parametrize("hemisphere", ["nh", "sh"])
@pytest.mark.parametrize("length", ["season", "all", 12])
def test_existing_period_modes_remain_unpadded(tmp_path, monkeypatch, hemisphere, length):
    _stub_grid_and_logging(monkeypatch)
    config = _config(tmp_path, dt.datetime(2020, 10, 1))
    config["options"]["hemisphere"] = hemisphere
    config["stacking"]["t_length"] = length
    index = SimpleNamespace(
        file_dates={"cryosat2": [dt.datetime(y, 10, 1) for y in (2019, 2020, 2022)]},
        get_file_list=lambda directories: None,
        get_file_dates=lambda: None)
    for module in (pipeline, input_data):
        monkeypatch.setattr(module, "SeaIceThicknessMultiProducts", lambda **kwargs: index)
    starts = [dt.datetime(y, 10, 1) for y in ([2020, 2022] if length == "all" else [2020])]
    periods = []
    for start in starts:
        end = (dt.datetime(start.year + 1, 5, 1) if hemisphere == "nh"
               else dt.datetime(start.year, 11, 1))
        periods.append((start, 12 if length == 12 else (end - start).days))
    assert input_data._stacking_periods(config) == periods
    processed = []
    merged = []
    monkeypatch.setattr(pipeline, "create_out_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(
        pipeline, "process_direction",
        lambda cfg, direction, grid, output_days: processed.append(
            (cfg["stacking"]["t_start"], cfg["stacking"]["t_length"], output_days)))
    monkeypatch.setattr(
        pipeline, "merge_direction_stacks",
        lambda cfg, grid, growth_width, width, files_f, files_r, j: merged.append(j))

    pipeline.run(config)

    assert processed == [(start, days, None) for start, days in periods for _ in "fr"]
    assert merged == [j for _, days in periods for j in range(days)]


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
