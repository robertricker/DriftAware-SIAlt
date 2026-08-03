"""Download-only synchronization of date-dependent stacking inputs."""

from __future__ import annotations

import datetime as dt
import ftplib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import urlopen

from loguru import logger

from driftaware_sialt.products.sea_ice_concentration import (
    SeaIceConcentrationProducts,
)
from driftaware_sialt.products.sea_ice_drift import SeaIceDriftProducts
from driftaware_sialt.products.sea_ice_thickness import (
    SeaIceThicknessMultiProducts,
)


ONE_DAY = dt.timedelta(days=1)
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ProductFileSpec:
    """Filename metadata needed to inventory one remote product."""

    product_id: str
    hemisphere_token: str
    date_str: str
    date_pattern: str
    date_offset: dt.timedelta = dt.timedelta(0)


@dataclass(frozen=True)
class RemoteFile:
    """One dated file found in a remote repository."""

    date: dt.datetime
    url: str


@dataclass
class ProductSyncReport:
    """Download-only synchronization counts for one product."""

    requested_dates: int
    remote_files: int = 0
    already_local: int = 0
    downloaded: int = 0
    remote_configured: bool = True


@dataclass
class InputDataReport:
    """Summary returned by :func:`sync_required_input_data`."""

    products: Dict[str, ProductSyncReport] = field(default_factory=dict)
    downloaded: List[Path] = field(default_factory=list)


def _as_datetime(value) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    return dt.datetime.fromisoformat(str(value))


def _season_length(start: dt.datetime, hemisphere: str) -> int:
    if hemisphere == "nh":
        end = dt.datetime(start.year + 1, 5, 1)
    elif hemisphere == "sh":
        end = dt.datetime(start.year, 11, 1)
    else:
        raise ValueError(f"unsupported hemisphere: {hemisphere}")
    length = (end - start).days
    if length <= 0:
        raise ValueError(
            f"stacking season starting at {start:%Y-%m-%d} has no days")
    return length


def _stacking_periods(config: Mapping) -> List[Tuple[dt.datetime, int]]:
    """Return stacking start dates and lengths without mutating the config."""
    stacking = config["stacking"]
    start = _as_datetime(stacking["t_start"])
    length = stacking["t_length"]
    hemisphere = config["options"]["hemisphere"]

    if isinstance(length, int) and not isinstance(length, bool):
        if length <= 0:
            raise ValueError("stacking.t_length must be greater than zero")
        return [(start, length)]

    if length == "season":
        return [(start, _season_length(start, hemisphere))]

    if length != "all":
        raise ValueError(
            "stacking.t_length must be an integer, 'season', or 'all'")

    sensors = config["options"]["sensor"]
    file_index = SeaIceThicknessMultiProducts(
        hem=hemisphere,
        sensor=sensors,
        target_var=config["options"]["target_variable"],
    )
    file_index.get_file_list(config["input_dir"])
    file_index.get_file_dates()
    years = sorted({
        date.year
        for sensor in sensors
        for date in file_index.file_dates[sensor]
        if date.year >= start.year
    })
    if not years:
        raise FileNotFoundError(
            "no altimetry files found for stacking.t_length=all")

    periods = []
    for year in years:
        period_start = start.replace(year=year)
        periods.append(
            (period_start, _season_length(period_start, hemisphere)))
    return periods


def required_input_dates(config: Mapping) -> Dict[str, List[dt.datetime]]:
    """Calculate the dates requested from each input family by stacking."""
    mode = str(config["stacking"]["mode"]).lower()
    directions = set(mode)
    if not directions or not directions <= {"f", "r"}:
        raise ValueError("stacking.mode must be 'f', 'r', or 'fr'")

    altimetry = set()
    concentration = set()
    drift = set()
    for start, length in _stacking_periods(config):
        run_dates = {start + index * ONE_DAY for index in range(length)}
        altimetry.update(run_dates)
        concentration.update(run_dates)
        if "f" in directions:
            concentration.update(date + ONE_DAY for date in run_dates)
            drift.update(date + ONE_DAY for date in run_dates)
        if "r" in directions:
            concentration.update(date - ONE_DAY for date in run_dates)
            drift.update(run_dates)

    return {
        "altimetry": sorted(altimetry),
        "ice_conc": sorted(concentration),
        "ice_drift": sorted(drift),
    }


def required_auxiliary_dates(config: Mapping) -> Dict[str, List[dt.datetime]]:
    """Backward-compatible view of required concentration and drift dates."""
    dates = required_input_dates(config)
    return {
        "ice_conc": dates["ice_conc"],
        "ice_drift": dates["ice_drift"],
    }


def _parse_product_date(
    spec: ProductFileSpec,
    filename: str,
) -> dt.datetime | None:
    match = re.search(r"\d" + spec.date_str, os.path.basename(filename))
    if match is None:
        return None
    try:
        parsed = dt.datetime.strptime(match.group(), spec.date_pattern)
    except ValueError:
        return None
    return parsed + spec.date_offset


def _shift_month(month: dt.datetime, offset: int) -> dt.datetime:
    month_index = month.year * 12 + month.month - 1 + offset
    year, zero_based_month = divmod(month_index, 12)
    return dt.datetime(year, zero_based_month + 1, 1)


def _archive_months(dates: Sequence[dt.datetime]) -> Iterable[dt.datetime]:
    """Yield requested and adjacent archive months, once each."""
    months = set()
    for date in dates:
        month = date.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0)
        # OSI405 files can be stored in the month following their parsed
        # processing date. Adjacent scans are harmless for the other archives.
        months.update(_shift_month(month, offset) for offset in (-1, 0, 1))
    return sorted(months)


def _archive_month_url(base_url: str, month: dt.datetime) -> str:
    return (
        base_url.rstrip("/")
        + f"/{month.year:04d}/{month.month:02d}"
    )


def _list_ftp_directory(directory_url: str, timeout: float) -> List[str]:
    """List NetCDF file URLs in one anonymous FTP directory."""
    parsed = urlsplit(directory_url)
    if parsed.scheme.lower() != "ftp":
        raise ValueError(
            f"unsupported remote repository scheme {parsed.scheme!r}: "
            f"{directory_url}")

    username = unquote(parsed.username) if parsed.username else "anonymous"
    password = unquote(parsed.password) if parsed.password else "anonymous@"
    host = parsed.hostname
    if host is None:
        raise ValueError(f"remote repository URL has no host: {directory_url}")

    ftp = ftplib.FTP()
    try:
        ftp.connect(host, parsed.port or 21, timeout=timeout)
        ftp.login(username, password)
        directory = unquote(parsed.path) or "/"
        try:
            names = ftp.nlst(directory)
        except ftplib.error_perm as error:
            if str(error).startswith("550"):
                return []
            raise
    finally:
        try:
            ftp.quit()
        except (AttributeError, EOFError, OSError, ftplib.Error):
            ftp.close()

    urls = []
    directory_path = parsed.path.rstrip("/")
    for name in names:
        basename = os.path.basename(name.rstrip("/"))
        if not basename.lower().endswith(".nc"):
            continue
        path = f"{directory_path}/{quote(basename)}"
        urls.append(urlunsplit(("ftp", parsed.netloc, path, "", "")))
    return urls


def _download_file(
    source_url: str,
    destination: Path,
    timeout: float,
) -> None:
    """Download one file atomically into the local product repository."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(
        f"{destination.name}.part-{os.getpid()}")
    try:
        with urlopen(source_url, timeout=timeout) as source:
            with partial.open("wb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _inventory_remote(
    remote_directory: str,
    dates: Sequence[dt.datetime],
    spec: ProductFileSpec,
    timeout: float,
    log_label: str | None = None,
) -> List[RemoteFile]:
    requested_days = {date.date() for date in dates}
    files = {}
    months = list(_archive_months(dates))
    for index, month in enumerate(months, start=1):
        month_url = _archive_month_url(remote_directory, month)
        if log_label:
            logger.info(
                "{}: inspecting remote month {}/{} ({:%Y-%m})",
                log_label,
                index,
                len(months),
                month,
            )
        for url in _list_ftp_directory(month_url, timeout):
            basename = unquote(os.path.basename(urlsplit(url).path))
            if spec.hemisphere_token.lower() not in basename.lower():
                continue
            file_date = _parse_product_date(spec, basename)
            if file_date is None or file_date.date() not in requested_days:
                continue
            files[basename] = RemoteFile(date=file_date, url=url)
    return sorted(files.values(), key=lambda item: (item.date, item.url))


def _local_netcdf_basenames(directory: Path) -> Set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.name
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() == ".nc"
    }


def _sync_product(
    family: str,
    product_id: str,
    local_directory,
    remote_directory,
    dates: Sequence[dt.datetime],
    spec: ProductFileSpec,
    timeout: float,
    report: InputDataReport,
) -> None:
    report_key = f"{family}.{product_id}"
    product_report = ProductSyncReport(requested_dates=len(dates))
    report.products[report_key] = product_report
    logger.info(
        "{}: synchronization requested for {} dates, {:%Y-%m-%d} through "
        "{:%Y-%m-%d}",
        report_key,
        len(dates),
        dates[0],
        dates[-1],
    )
    if not remote_directory:
        product_report.remote_configured = False
        logger.info(
            "no remote repository configured for {}; keeping it local-only",
            report_key,
        )
        return

    local_root = Path(local_directory)
    logger.info(
        "{}: indexing local NetCDF files in {}",
        report_key,
        local_root,
    )
    local_files = _local_netcdf_basenames(local_root)
    logger.info(
        "{}: found {} local NetCDF files; inventorying {}",
        report_key,
        len(local_files),
        remote_directory,
    )
    remote_files = _inventory_remote(
        remote_directory,
        dates,
        spec,
        timeout,
        log_label=report_key,
    )
    product_report.remote_files = len(remote_files)

    missing_files = []
    for remote_file in remote_files:
        filename = unquote(os.path.basename(urlsplit(remote_file.url).path))
        if filename in local_files:
            product_report.already_local += 1
            continue
        missing_files.append(remote_file)

    logger.info(
        "{}: comparison complete: {} remote files in range, {} already "
        "local, {} to download",
        report_key,
        product_report.remote_files,
        product_report.already_local,
        len(missing_files),
    )
    for index, remote_file in enumerate(missing_files, start=1):
        filename = unquote(os.path.basename(urlsplit(remote_file.url).path))
        destination = (
            local_root
            / f"{remote_file.date.year:04d}"
            / f"{remote_file.date.month:02d}"
            / filename
        )
        logger.info(
            "{}: downloading file {}/{} for {:%Y-%m-%d}: {}",
            report_key,
            index,
            len(missing_files),
            remote_file.date,
            filename,
        )
        _download_file(remote_file.url, destination, timeout)
        local_files.add(filename)
        report.downloaded.append(destination)
        product_report.downloaded += 1

    logger.info(
        "{}: synchronization finished: {} downloaded, {} already local",
        report_key,
        product_report.downloaded,
        product_report.already_local,
    )


def _auxiliary_spec(
    product_class,
    product_id: str,
    hemisphere: str,
) -> ProductFileSpec:
    product = product_class(product_id=product_id, hem=hemisphere)
    settings = product.config[product_id]
    return ProductFileSpec(
        product_id=product_id,
        hemisphere_token=settings["hem_" + hemisphere],
        date_str=settings["date_str"],
        date_pattern=settings["date_pt"],
        date_offset=settings["date_offset"],
    )


def _altimetry_spec(
    sensor: str,
    target_variable: str,
    hemisphere: str,
) -> ProductFileSpec:
    product = SeaIceThicknessMultiProducts(
        hem=hemisphere,
        sensor=[sensor],
        target_var=target_variable,
    )
    settings = product.config[sensor][target_variable]
    return ProductFileSpec(
        product_id=sensor,
        hemisphere_token=settings["hem_" + hemisphere],
        date_str=settings["date_str"],
        date_pattern=settings["date_pt"],
    )


def sync_required_input_data(config: Mapping) -> InputDataReport:
    """Download every missing remote file in the stacking date range.

    This is a download-only sync: local files that are not present remotely are
    retained. Auxiliary products are limited to the configured priority lists,
    and altimetry is limited to ``options.sensor``.
    """
    if config.get("stage") != "stacking":
        return InputDataReport()

    dates = required_input_dates(config)
    options = config["options"]
    auxiliary = config["auxiliary"]
    remote = config.get("remote_dir", {})
    timeout = TIMEOUT_SECONDS
    hemisphere = options["hemisphere"]
    report = InputDataReport()

    auxiliary_families = (
        (
            "ice_conc",
            SeaIceConcentrationProducts,
            options["ice_conc_products"],
        ),
        (
            "ice_drift",
            SeaIceDriftProducts,
            options["ice_drift_products"],
        ),
    )
    for family, product_class, product_ids in auxiliary_families:
        for product_id in product_ids:
            if product_id not in auxiliary[family]:
                raise KeyError(
                    f"no local {family} directory configured for {product_id}")
            _sync_product(
                family,
                product_id,
                auxiliary[family][product_id],
                remote.get(family, {}).get(product_id),
                dates[family],
                _auxiliary_spec(
                    product_class, product_id, hemisphere),
                timeout,
                report,
            )

    altimetry_remote = remote.get("altimetry", {})
    for sensor in options["sensor"]:
        local_directory = config["input_dir"][sensor]
        if isinstance(local_directory, Mapping):
            # No plain FTP repository is currently configured for ICESat-2.
            report.products[f"altimetry.{sensor}"] = ProductSyncReport(
                requested_dates=len(dates["altimetry"]),
                remote_configured=False,
            )
            logger.info(
                "structured altimetry input {} remains local-only", sensor)
            continue
        _sync_product(
            "altimetry",
            sensor,
            local_directory,
            altimetry_remote.get(sensor),
            dates["altimetry"],
            _altimetry_spec(
                sensor,
                options["target_variable"],
                hemisphere,
            ),
            timeout,
            report,
        )

    logger.info(
        "input repository sync complete: {} files downloaded",
        len(report.downloaded),
    )
    return report
