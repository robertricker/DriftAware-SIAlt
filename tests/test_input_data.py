import datetime as dt

import driftaware_sialt.input_data as input_data


def _config(tmp_path, *, mode="f", length=1):
    return {
        "stage": "stacking",
        "input_dir": {
            "cryosat2": str(tmp_path / "local" / "cryosat2"),
        },
        "auxiliary": {
            "ice_conc": {
                "osi430": str(tmp_path / "local" / "osi430"),
            },
            "ice_drift": {
                "osi405": str(tmp_path / "local" / "osi405"),
            },
        },
        "remote_dir": {
            "altimetry": {
                "cryosat2": "ftp://example.test/altimetry/cryosat2",
            },
            "ice_conc": {
                "osi430": "ftp://example.test/conc/osi430",
            },
            "ice_drift": {
                "osi405": "ftp://example.test/drift/osi405",
            },
        },
        "input_data": {"timeout_seconds": 1},
        "options": {
            "sensor": ["cryosat2"],
            "target_variable": "sea_ice_thickness",
            "hemisphere": "nh",
            "ice_conc_products": ["osi430"],
            "ice_drift_products": ["osi405"],
        },
        "stacking": {
            "t_start": dt.datetime(2021, 10, 1),
            "t_length": length,
            "mode": mode,
        },
    }


def test_required_input_dates_follow_stacking_direction(tmp_path):
    config = _config(tmp_path, mode="fr", length=2)

    result = input_data.required_input_dates(config)

    assert result["altimetry"] == [
        dt.datetime(2021, 10, 1),
        dt.datetime(2021, 10, 2),
    ]
    assert result["ice_conc"] == [
        dt.datetime(2021, 9, 30),
        dt.datetime(2021, 10, 1),
        dt.datetime(2021, 10, 2),
        dt.datetime(2021, 10, 3),
    ]
    assert result["ice_drift"] == [
        dt.datetime(2021, 10, 1),
        dt.datetime(2021, 10, 2),
        dt.datetime(2021, 10, 3),
    ]


def test_every_missing_remote_file_in_date_range_is_downloaded(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    remote_files = {
        "altimetry": [
            "ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-NH-"
            "20211001-fv4p0.nc",
            "ESACCI-SEAICE-L2P-SITHICK-SIRAL_CRYOSAT2-NH-"
            "20211002-fv4p0.nc",
        ],
        "conc": [
            "ice_conc_nh_ease2-250_icdr-v3p0_202110011200.nc",
            "ice_conc_nh_ease2-250_icdr-v3p0_202110021200.nc",
            "ice_conc_nh_ease2-250_icdr-v3p0_202110101200.nc",
        ],
        "drift": [
            "ice_drift_nh_polstere-625_multi-oi_"
            "202110011200-202110031200.nc",
            "ice_drift_nh_polstere-625_multi-oi_"
            "202110091200-202110111200.nc",
        ],
    }

    def fake_listing(directory_url, timeout):
        if "/altimetry/" in directory_url:
            family = "altimetry"
        elif "/conc/" in directory_url:
            family = "conc"
        else:
            family = "drift"
        return [
            directory_url + "/" + filename
            for filename in remote_files[family]
        ]

    def fake_download(source_url, destination, timeout):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fixture")

    monkeypatch.setattr(input_data, "_list_ftp_directory", fake_listing)
    monkeypatch.setattr(input_data, "_download_file", fake_download)

    report = input_data.sync_required_input_data(config)

    assert len(report.downloaded) == 4
    assert all(path.is_file() for path in report.downloaded)
    assert report.products["altimetry.cryosat2"].remote_files == 1
    assert report.products["ice_conc.osi430"].remote_files == 2
    assert report.products["ice_drift.osi405"].remote_files == 1

    second_report = input_data.sync_required_input_data(config)

    assert second_report.downloaded == []
    assert second_report.products[
        "altimetry.cryosat2"].already_local == 1
    assert second_report.products["ice_conc.osi430"].already_local == 2
    assert second_report.products["ice_drift.osi405"].already_local == 1


def test_only_selected_auxiliary_products_are_synchronized(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    config["auxiliary"]["ice_conc"]["osi450"] = str(
        tmp_path / "local" / "osi450")
    config["remote_dir"]["ice_conc"]["osi450"] = (
        "ftp://example.test/conc/osi450")
    listed = []

    def fake_listing(directory_url, timeout):
        listed.append(directory_url)
        return []

    monkeypatch.setattr(input_data, "_list_ftp_directory", fake_listing)

    report = input_data.sync_required_input_data(config)

    assert "ice_conc.osi430" in report.products
    assert "ice_conc.osi450" not in report.products
    assert not any("osi450" in url for url in listed)


def test_empty_remote_date_range_is_not_an_error(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        input_data,
        "_list_ftp_directory",
        lambda directory_url, timeout: [],
    )
    messages = []
    sink_id = input_data.logger.add(
        messages.append,
        format="{message}",
    )

    try:
        report = input_data.sync_required_input_data(config)
    finally:
        input_data.logger.remove(sink_id)

    assert report.downloaded == []
    assert all(
        product.remote_files == 0
        for product in report.products.values()
    )
    log_text = "".join(messages)
    assert "ice_conc.osi430: synchronization requested" in log_text
    assert "inspecting remote month" in log_text
    assert "comparison complete" in log_text
    assert "synchronization finished" in log_text


def test_adjacent_archive_month_is_scanned_for_osi405(monkeypatch):
    spec = input_data.ProductFileSpec(
        product_id="osi405",
        hemisphere_token="_nh_",
        date_str="{12}",
        date_pattern="%Y%m%d%H%M",
        date_offset=dt.timedelta(days=1),
    )
    listed = []
    filename = (
        "ice_drift_nh_polstere-625_multi-oi_"
        "202110301200-202111011200.nc"
    )

    def fake_listing(directory_url, timeout):
        listed.append(directory_url)
        if directory_url.endswith("/2021/11"):
            return [directory_url + "/" + filename]
        return []

    monkeypatch.setattr(input_data, "_list_ftp_directory", fake_listing)

    files = input_data._inventory_remote(
        "ftp://example.test/drift",
        [dt.datetime(2021, 10, 31)],
        spec,
        1,
    )

    assert [file.date.date() for file in files] == [dt.date(2021, 10, 31)]
    assert any(url.endswith("/2021/11") for url in listed)
