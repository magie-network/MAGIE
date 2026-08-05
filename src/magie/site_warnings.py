"""
magnetometer_email_monitor.py

Monitor magnetometer data files and send email warnings when a site has stopped
producing valid measurements. Also sends a restored email when valid data returns.

Expected data layout
--------------------
Data are organised under a root directory like:

    /YYYY/MM/DD/

Persistent IAGA files are expected at, for example:

    /YYYY/MM/DD/iaga2002/dunYYYYMMDDpsec.sec
    /YYYY/MM/DD/iaga2002/dunYYYYMMDDpmin.min

The IAGA filename type code may vary, for example ``psec`` for provisional or
``vsec`` for variation data.

The monitor reads persistent IAGA-2002 files only.

A site is considered active only if at least one of x, y, z contains a finite
value at a timestamp. A timestamp with only NaNs is treated as missing data.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from string import Template

import pandas as pd
import os
import numpy as np
from magpy.stream import read

from magie.email_utils import (
    load_email_config,
    load_recipients,
    send_html_email,
)
from magie.utils import enforce_types, get_site_metadata, tqdm_joblib

@dataclass
class SiteConfig:
    """
    Configuration for one magnetometer site.

    Parameters
    ----------
    code:
        Site code or known alias, for example ``"dun"`` or ``"valentia"``.
        Known aliases are resolved through ``magie.utils.get_site_metadata``.
    name:
        Optional human-readable site name for emails. When omitted, the station
        name from site metadata is used.
    data_root:
        Optional data root for this site. When omitted, the monitor uses the
        shared ``MonitorConfig.data_root`` value.
    assumed_data_frequency:
        Expected data cadence used for availability output when no file exists
        or cadence cannot be inferred. Known sites derive this from metadata
        when omitted.
    active:
        Whether this site should currently be monitored.
    permanently_off:
        Whether this site is intentionally retired. Permanently-off sites are
        skipped and will not generate outage or restored emails.
    """

    code: str
    name: str = ""
    null_data_value: float = 999999.00
    data_root: Path | None = None
    assumed_data_frequency: str | None = None
    active: bool = True
    permanently_off: bool = False

    def __post_init__(self) -> None:
        if self.data_root is not None:
            self.data_root = Path(self.data_root)

        metadata = get_site_metadata(self.code)

        if metadata is None:
            if not self.name:
                self.name = self.code
            return

        self.code = metadata["site_code"]
        if not self.name:
            self.name = metadata["station_name"]
        if self.assumed_data_frequency is None:
            interval_type = metadata.get("data_interval_type")
            if interval_type == "1-second":
                self.assumed_data_frequency = "sec"
            elif interval_type == "1-minute":
                self.assumed_data_frequency = "min"


@dataclass
class DayAvailability:
    """
    Availability summary for one site on one UTC day.
    """

    has_data: bool
    latest: datetime | None
    valid_samples: int
    expected_samples: int | None
    coverage_percent: float | None
    data_frequency: str | None
    source_file: str | None


@enforce_types(value=(datetime, pd.Timestamp, str))
def as_utc_datetime(value) -> datetime:
    """
    Return ``value`` as a timezone-aware UTC ``datetime``.
    """

    timestamp = utc_timestamp(value)
    if timestamp is pd.NaT:
        return timestamp
    return timestamp.to_pydatetime()


@enforce_types(value=(datetime, pd.Timestamp, str))
def utc_archive_day(value) -> datetime:
    """
    Return midnight UTC for the UTC archive day containing ``value``.
    """

    timestamp = utc_timestamp(value)
    if timestamp is pd.NaT:
        return timestamp
    timestamp = timestamp.normalize()
    return timestamp.to_pydatetime()


@enforce_types(value=(str, type(None)))
def parse_lookup_datetime(value: str | None) -> datetime | None:
    """
    Parse an ISO datetime from the availability lookup as UTC.
    """

    if not value:
        return None

    try:
        return as_utc_datetime(value)
    except (TypeError, ValueError):
        return None


@enforce_types(site=SiteConfig, default_data_root=(str, Path, Mapping))
def data_root_for_site(
    site: SiteConfig,
    default_data_root: Path | str | Mapping[str, Path | str],
) -> Path:
    """
    Return the archive root to use for one site.
    """

    if site.data_root is not None:
        return Path(site.data_root)

    if isinstance(default_data_root, Mapping):
        try:
            return Path(default_data_root[site.code])
        except KeyError as exc:
            raise KeyError(
                f"No data root configured for site {site.code!r}"
            ) from exc

    return Path(default_data_root)


@enforce_types(data_root=(str, Path), site_code=str, day=(datetime, pd.Timestamp, str))
def candidate_files_for_day(data_root: Path, site_code: str, day) -> list[Path]:
    """
    Return possible IAGA files for one site on one UTC archive day.
    """

    data_root = Path(data_root)
    day = utc_archive_day(day)
    ymd = day.strftime("%Y%m%d")
    iaga_dir = data_root / day.strftime("%Y/%m/%d") / "iaga2002"

    iaga_files = [
        iaga_dir / f"{site_code}{ymd}psec.sec",
        iaga_dir / f"{site_code}{ymd}pmin.min",
    ]
    iaga_files.extend(sorted(iaga_dir.glob(f"{site_code}{ymd}*sec.sec")))
    iaga_files.extend(sorted(iaga_dir.glob(f"{site_code}{ymd}*min.min")))
    return list(dict.fromkeys(iaga_files))


@enforce_types(seconds=(int, float, np.number, type(None)))
def frequency_from_seconds(seconds: float | None) -> tuple[str | None, int | None]:
    """
    Return a frequency label and expected daily sample count.
    """

    if seconds is None:
        return None, None

    if seconds <= 1.5:
        return "sec", 24 * 60 * 60

    if seconds <= 90:
        return "min", 24 * 60

    return f"{seconds:g}s", round((24 * 60 * 60) / seconds)


@enforce_types(path=(str, Path))
def frequency_from_path(path: Path) -> tuple[str | None, int | None]:
    """
    Infer cadence from a known magnetometer filename.
    """

    path = Path(path)
    name = path.name.lower()
    suffix = path.suffix.lower()

    if "psec" in name or suffix == ".sec":
        return "sec", 24 * 60 * 60

    if "pmin" in name or suffix == ".min":
        return "min", 24 * 60

    return None, None


@enforce_types(frequency=(str, type(None)))
def expected_samples_from_frequency(frequency: str | None) -> int | None:
    """
    Return expected daily sample count for a configured frequency label.
    """

    if frequency is None:
        return None

    label = frequency.strip().lower()

    if label in {"sec", "second", "seconds", "1-sec", "1-second"}:
        return 24 * 60 * 60

    if label in {"min", "minute", "minutes", "1-min", "1-minute"}:
        return 24 * 60

    return None


@enforce_types(path=(str, Path), cadence_path=(str, Path, type(None)))
def stream_availability(stream, path: Path, cadence_path: Path | None = None) -> DayAvailability:
    """
    Return valid-sample count, latest timestamp, and coverage for one stream.
    """

    path = Path(path)
    cadence_path = Path(cadence_path) if cadence_path is not None else path

    try:
        times = np.asarray(stream["time"], dtype=object)
        x = np.asarray(stream["x"], dtype=float)
        y = np.asarray(stream["y"], dtype=float)
        z = np.asarray(stream["z"], dtype=float)
    except Exception:
        return DayAvailability(False, None, 0, None, None, None, str(path))

    if len(times) == 0:
        return DayAvailability(False, None, 0, None, None, None, str(path))

    valid = np.isfinite(x) | np.isfinite(y) | np.isfinite(z)
    if not np.any(valid):
        return DayAvailability(False, None, 0, None, None, None, str(path))

    valid_times = []
    for value in times[valid]:
        try:
            timestamp = utc_timestamp(value)
        except (TypeError, ValueError):
            continue
        if timestamp is not pd.NaT:
            valid_times.append(timestamp.to_pydatetime())

    if not valid_times:
        return DayAvailability(False, None, 0, None, None, None, str(path))

    valid_times = sorted(set(valid_times))
    latest = valid_times[-1]
    valid_samples = len(valid_times)
    cadence_seconds = None

    if len(valid_times) > 1:
        deltas = np.diff([value.timestamp() for value in valid_times])
        positive_deltas = deltas[deltas > 0]
        if len(positive_deltas) > 0:
            cadence_seconds = float(np.median(positive_deltas))

    data_frequency, expected_samples = frequency_from_seconds(cadence_seconds)
    if expected_samples is None:
        data_frequency, expected_samples = frequency_from_path(cadence_path)

    coverage_percent = None
    if expected_samples:
        coverage_percent = round(
            min(100.0, (valid_samples / expected_samples) * 100),
            3,
        )

    return DayAvailability(
        has_data=True,
        latest=latest,
        valid_samples=valid_samples,
        expected_samples=expected_samples,
        coverage_percent=coverage_percent,
        data_frequency=data_frequency,
        source_file=str(path),
    )


@enforce_types(candidate=DayAvailability, current=DayAvailability)
def is_better_availability(
    candidate: DayAvailability,
    current: DayAvailability,
) -> bool:
    """
    Return whether a candidate file is a better daily availability source.
    """

    if not candidate.has_data:
        return False

    if not current.has_data:
        return True

    if candidate.latest is not None and current.latest is None:
        return True

    if candidate.latest is None:
        return False

    if current.latest is not None:
        if candidate.latest > current.latest:
            return True
        if candidate.latest < current.latest:
            return False

    candidate_coverage = candidate.coverage_percent
    current_coverage = current.coverage_percent
    if candidate_coverage is not None and current_coverage is None:
        return True
    if candidate_coverage is None:
        return candidate.valid_samples > current.valid_samples

    return candidate_coverage > current_coverage


@enforce_types(site_code=str, day=(datetime, pd.Timestamp, str), data_root=(str, Path))
def day_data_availability(site_code: str, day, data_root: Path) -> DayAvailability:
    """
    Return valid data availability for one site on one UTC day.
    """

    best = DayAvailability(False, None, 0, None, None, None, None)

    for path in candidate_files_for_day(data_root, site_code, day):
        if not path.exists():
            continue

        try:
            stream = read(str(path))
            availability = stream_availability(stream, path, cadence_path=path)
        except Exception:
            continue

        if is_better_availability(availability, best):
            best = availability

    return best


@enforce_types(site_code=str, data_root=(str, Path), day=(datetime, pd.Timestamp, str))
def availability_job(site_code: str, data_root: Path, day) -> tuple[str, DayAvailability]:
    """
    Run one site/day availability check for parallel execution.
    """

    day = utc_archive_day(day)
    return day.strftime("%Y-%m-%d"), day_data_availability(site_code, day, data_root)


@enforce_types(site=SiteConfig, site_data_root=(str, Path), day=(datetime, pd.Timestamp, str))
def site_availability_job(
    site: SiteConfig,
    site_data_root: Path,
    day,
) -> tuple[SiteConfig, str, DayAvailability]:
    """
    Run one site/day availability check and include the site in the result.
    """

    date_key, availability = availability_job(site.code, site_data_root, day)
    return site, date_key, availability


@enforce_types(
    sites=list,
    data_root=(str, Path, Mapping),
    start_date=(datetime, pd.Timestamp, str),
    end_date=(datetime, pd.Timestamp, str),
    output_path=(str, Path),
    parallel_jobs=int,
    show_progress=bool,
    update_existing=bool,
)
def build_availability_lookup(
    sites,
    data_root,
    start_date,
    end_date,
    output_path,
    parallel_jobs=1,
    show_progress=True,
    update_existing=False,
):
    """
    Write daily availability JSON for a set of sites.

    ``data_root`` may be one shared archive root or a dictionary mapping
    normalised site codes to archive roots. A site's own ``data_root`` field
    takes precedence over either form.
    """

    from joblib import Parallel, delayed

    output_path = Path(output_path)
    start_date = utc_archive_day(start_date)
    end_date = utc_archive_day(end_date)

    if update_existing and output_path.exists():
        lookup = json.loads(output_path.read_text(encoding="utf-8"))
        lookup.setdefault("stations", {})
    else:
        lookup = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "stations": {},
        }

    lookup["generated_at"] = datetime.now(timezone.utc).isoformat()

    days = []
    day = start_date
    while day <= end_date:
        days.append(day)
        day += timedelta(days=1)

    jobs = [
        (site, data_root_for_site(site, data_root), day)
        for site in sites
        for day in days
    ]

    if parallel_jobs == 1:
        job_results = [
            (site, *availability_job(site.code, site_data_root, day))
            for site, site_data_root, day in jobs
        ]
    elif jobs:
        with tqdm_joblib(
            total=len(jobs),
            desc_prefix="Checking magnetometer availability",
            unit="day",
            enabled=show_progress,
        ):
            job_results = Parallel(
                n_jobs=parallel_jobs,
                prefer="processes",
                backend="loky",
            )(
                delayed(site_availability_job)(site, site_data_root, day)
                for site, site_data_root, day in jobs
            )
    else:
        job_results = []

    results_by_site = {}
    for site, date_key, availability in job_results:
        results_by_site.setdefault(site.code, {"site": site, "dates": {}})
        results_by_site[site.code]["dates"][date_key] = availability

    for site in sites:
        site_data_root = data_root_for_site(site, data_root)
        station = lookup["stations"].get(site.code, {}) if update_existing else {}
        available_dates = dict(station.get("available_dates", {}))
        coverage_percent_by_date = dict(station.get("coverage_percent_by_date", {}))
        data_frequency_by_date = dict(station.get("data_frequency_by_date", {}))
        valid_samples_by_date = dict(station.get("valid_samples_by_date", {}))
        expected_samples_by_date = dict(station.get("expected_samples_by_date", {}))
        latest_valid_measurement_by_date = dict(
            station.get("latest_valid_measurement_by_date", {})
        )
        source_file_by_date = dict(station.get("source_file_by_date", {}))
        last_available_date = None
        last_valid_measurement = None
        date_results = results_by_site.get(site.code, {}).get("dates", {})

        for day in days:
            date_key = day.strftime("%Y-%m-%d")
            availability = date_results.get(
                date_key,
                DayAvailability(False, None, 0, None, None, None, None),
            )
            previous_had_data = bool(available_dates.get(date_key))
            previous_latest = parse_lookup_datetime(
                latest_valid_measurement_by_date.get(date_key)
            )
            if update_existing and previous_had_data:
                if availability.latest is None:
                    continue
                if previous_latest is not None and availability.latest < previous_latest:
                    continue

            data_frequency = availability.data_frequency or site.assumed_data_frequency
            expected_samples = (
                availability.expected_samples
                or expected_samples_from_frequency(data_frequency)
            )
            coverage_percent = availability.coverage_percent
            if coverage_percent is None and availability.has_data and expected_samples:
                coverage_percent = round(
                    min(100.0, (availability.valid_samples / expected_samples) * 100),
                    3,
                )
            elif not availability.has_data:
                coverage_percent = 0.0

            available_dates[date_key] = availability.has_data
            coverage_percent_by_date[date_key] = coverage_percent
            data_frequency_by_date[date_key] = data_frequency
            valid_samples_by_date[date_key] = availability.valid_samples
            expected_samples_by_date[date_key] = expected_samples
            latest_valid_measurement_by_date[date_key] = (
                availability.latest.isoformat() if availability.latest else None
            )
            source_file_by_date[date_key] = availability.source_file

        for date_key in sorted(available_dates):
            if not available_dates[date_key]:
                continue
            last_available_date = date_key
            last_valid_measurement = latest_valid_measurement_by_date.get(date_key)
            if (
                last_valid_measurement is None
                and date_key == station.get("last_available_date")
            ):
                last_valid_measurement = station.get("last_valid_measurement")

        if last_available_date is None:
            last_available_date = station.get("last_available_date")
            last_valid_measurement = station.get("last_valid_measurement")

        lookup["stations"][site.code] = {
            "name": site.name,
            "data_root": str(site_data_root),
            "active": site.active,
            "permanently_off": site.permanently_off,
            "last_available_date": last_available_date,
            "last_valid_measurement": last_valid_measurement,
            "available_dates": available_dates,
            "coverage_percent_by_date": coverage_percent_by_date,
            "data_frequency_by_date": data_frequency_by_date,
            "valid_samples_by_date": valid_samples_by_date,
            "expected_samples_by_date": expected_samples_by_date,
            "latest_valid_measurement_by_date": latest_valid_measurement_by_date,
            "source_file_by_date": source_file_by_date,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(lookup, indent=2), encoding="utf-8")






@enforce_types(value=int)
def number_word(value: int) -> str:
    """
    Return a lowercase English word for small non-negative counts.
    """

    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(value, str(value))
@enforce_types(value=(datetime, pd.Timestamp, str))
def format_ut_datetime(value: datetime) -> str:
    """
    Format a datetime as a timezone-free UT timestamp for email text.
    """

    return pd.Timestamp(value).round("s").strftime("%Y-%m-%d %H:%M:%S")

def format_event_datetime(value) -> str:
    """
    Format event timestamps for email tables at whole-second precision.
    """

    timestamp = pd.Timestamp(value)
    if timestamp is pd.NaT:
        return str(pd.NaT)
    return timestamp.round("s").strftime("%Y-%m-%d %H:%M:%S")

def format_event_duration(value) -> str:
    """
    Format event durations for email tables at whole-second precision.
    """

    duration = pd.Timedelta(value)
    if duration is pd.NaT:
        return str(pd.NaT)
    return str(duration.round("s"))

def utc_timestamp(value=None):
    """
    Return ``value`` as a timezone-aware UTC pandas timestamp.

    ``None`` means the current UTC time. Naive inputs are interpreted as UTC so
    the archive day logic is stable across Irish summer/winter local time.
    """

    timestamp = pd.Timestamp.now(tz='UTC') if value is None else pd.Timestamp(value)
    if timestamp is pd.NaT:
        return timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize('UTC')
    return timestamp.tz_convert('UTC')

def clamp_future_timestamp(timestamp, latest_allowed=None, source="measurement"):
    """
    Clamp future measurement timestamps to the monitor's current time.

    Future timestamps can appear when a generated availability lookup or IAGA
    file contains values beyond the monitor run time. The monitor must never
    save a future ``last_measurement`` because that creates negative outage
    durations. A warning is emitted when the future offset is large enough to
    suggest an upstream bug.
    """

    timestamp = utc_timestamp(timestamp)
    if latest_allowed is None or timestamp is pd.NaT:
        return timestamp

    latest_allowed = utc_timestamp(latest_allowed)
    if latest_allowed is pd.NaT or timestamp <= latest_allowed:
        return timestamp

    future_offset = timestamp - latest_allowed
    if future_offset > pd.Timedelta(minutes=10):
        warnings.warn(
            (
                f"{source} timestamp {timestamp.isoformat()} is "
                f"{future_offset} after monitor time {latest_allowed.isoformat()}; "
                "clamping to monitor time."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return latest_allowed

@enforce_types(site_config=SiteConfig)
def get_files(date, site_config):
    """
    Return candidate IAGA files for one site on one UTC archive day.
    """

    iaga_dir = site_config.data_root / date.strftime("%Y/%m/%d") / "iaga2002"
    return sorted(iaga_dir.glob(f"{site_config.code}{date:%Y%m%d}*"))

@enforce_types(path=(str, Path), null_data_value=(int, float))
def latest_valid_time_from_file(path, null_data_value, latest_allowed=None):
    """
    Read an IAGA file with MagPy and return the latest valid x/y/z timestamp.

    MagPy's ``read`` currently needs a string path for these files; passing a
    ``Path`` object can return an empty stream. A timestamp is valid when at
    least one of x, y, or z is finite and not the configured null value.
    """

    # MagPy can return an empty stream for Path objects here; use a string path.
    file = read(str(path))
    times = np.asarray(file['time'], dtype=object)
    if len(times) == 0:
        raise RuntimeError(f"MagPy read returned no timestamps for {path}")
    valid = np.zeros(len(times), dtype=bool)
    for component in ('x', 'y', 'z'):
        values = np.asarray(file[component], dtype=float)
        valid |= np.isfinite(values) & (values != null_data_value)
    if not np.any(valid):
        raise RuntimeError(f"MagPy read returned no valid x/y/z data for {path}")
    latest_time = pd.Timestamp(times[valid].max())
    return clamp_future_timestamp(
        latest_time,
        latest_allowed,
        source=f"MagPy file {path}",
    )

@enforce_types(site=SiteConfig, mag_availability=dict)
def latest_measurement_from_availability(site, mag_availability, latest_allowed=None):
    """
    Return the newest availability timestamp for a site, clamped to run time.
    """

    measurements = (
        mag_availability
        .get('stations', {})
        .get(site.code, {})
        .get('latest_valid_measurement_by_date', {})
    )
    if latest_allowed is not None:
        latest_allowed = utc_timestamp(latest_allowed)
    for key in list(measurements.keys())[::-1]:
        if measurements[key] is None or not measurements[key]:
            continue
        measurement_time = clamp_future_timestamp(
            measurements[key],
            latest_allowed,
            source=f"availability lookup {site.code} {key}",
        )
        return measurement_time.isoformat()
    return str(pd.NaT)

@enforce_types(site=SiteConfig, mag_availability=(dict, type(None)))
def default_station_status(site, mag_availability=None, latest_allowed=None):
    """
    Build the default status entry for a site.
    """

    latest = (
        latest_measurement_from_availability(site, mag_availability, latest_allowed)
        if mag_availability is not None
        else str(pd.NaT)
    )
    return {
        'status': 'permanently_off' if site.permanently_off else 'offline',
        'last_measurement': latest,
        'time_since_last': str(pd.NaT),
        'status_change': False,
    }

@enforce_types(site_configs=list, mag_availability=dict)
def create_monitor_status(site_configs, mag_availability, latest_allowed=None):
    """
    Create a fresh monitor status JSON structure for configured sites.
    """

    status= {'generated_at': pd.NaT, 'stations': {}}
    for site in site_configs:
        status['stations'][site.code] = default_station_status(
            site,
            mag_availability,
            latest_allowed,
        )
    return status

@enforce_types(site_configs=list, monitor_status=dict, mag_availability=(dict, type(None)))
def refresh_monitor_status_from_site_configs(site_configs, monitor_status, mag_availability=None, latest_allowed=None):
    """
    Synchronize an existing monitor status file with current site configs.

    This adds newly configured stations, fills missing fields from older status
    files, and keeps permanently-off site state aligned with ``SiteConfig``.
    """

    monitor_status.setdefault('stations', {})
    for site in site_configs:
        station_status = monitor_status['stations'].setdefault(
            site.code,
            default_station_status(site, mag_availability, latest_allowed),
        )
        station_status.setdefault('last_measurement', str(pd.NaT))
        station_status.setdefault('time_since_last', str(pd.NaT))
        station_status.setdefault('status_change', False)
        station_status.setdefault('status', 'offline')
        if site.permanently_off:
            station_status['status'] = 'permanently_off'
        elif station_status['status'] == 'permanently_off':
            station_status['status'] = 'offline'
    return monitor_status
@enforce_types(events=list)
def event_rows_html(events: list) -> str:
    """
    Render escaped table rows for a batch of site status events.
    """
    rows = []

    for site, event in events:
        site_code = site.code if isinstance(site, SiteConfig) else str(site)
        site_name = site.name if isinstance(site, SiteConfig) and site.name else site_code
        rows.append(
            "<tr>"
            f"<td>{escape(site_name)} ({escape(site_code)})</td>"
            f"<td>{escape(format_event_datetime(event['last_measurement']))}</td>"
            f"<td>{escape(format_event_duration(event['time_since_last']))}</td>"
            "</tr>"
        )

    return "\n".join(rows)
@enforce_types(
    subject=str,
    template_path=(str, Path),
    config=(str, Path),
    recipients=(str, Path),
    events=list,
    threshold_minutes=(int, type(None)),
)
def send_batched_status_email(
    subject: str,
    template_path: Path,
    config: Path,
    recipients: Path,
    checked_at: datetime,
    events: list,
    threshold_minutes: int | None = None,
) -> None:
    """
    Render and send one status email containing all events in a monitor run.
    """

    site_rows = event_rows_html(events)
    values = {
        "site_count": number_word(len(events)),
        "site_label": "site" if len(events) == 1 else "sites",
        "site_verb": "is" if len(events) == 1 else "are",
        "threshold_minutes": "" if threshold_minutes is None else threshold_minutes,
    }

    email_cfg = load_email_config(config)
    recipients = load_recipients(recipients)

    template_text = Path(template_path).read_text(encoding="utf-8")

    safe_values = {
        key: escape(str(value)) for key, value in values.items()
    }
    safe_values["site_rows"] = site_rows
    safe_values["checked_at"] = format_ut_datetime(checked_at)
    html = Template(template_text).safe_substitute(safe_values)

    send_html_email(
        smtp_host=email_cfg["smtp_host"],
        smtp_port=email_cfg["smtp_port"],
        username=email_cfg["username"],
        password=email_cfg["password"],
        from_addr=email_cfg["from_addr"],
        to_addrs=recipients,
        subject=subject,
        html_content=html,
        use_starttls=email_cfg["use_starttls"],
    )


class Mag_Monitor():
    """
    Stateful magnetometer activity monitor.

    The monitor keeps one JSON status file with each station's current state,
    last valid measurement, and whether a status transition has happened since
    the previous run. Emails are sent only for transitions so repeated cron runs
    do not resend the same outage/restored notification.

    Parameters
    ----------
    site_configs:
        List of ``SiteConfig`` objects to monitor.
    mag_availability_path:
        Availability lookup JSON used to seed a new monitor status file.
    monitor_status_path:
        Persistent monitor-owned JSON status file.
    alert_threshold:
        Maximum allowed age of the latest valid measurement.
    reset_monitor_status:
        When true, ignore any existing status file and create a fresh one.
    now:
        Optional fixed monitor time for tests. Defaults to current UTC time.
    """

    @enforce_types(
        site_configs=list,
        mag_availability_path=(str, Path),
        monitor_status_path=(str, Path),
        alert_threshold=pd.Timedelta,
        reset_monitor_status=bool,
        email_config_path=(str, Path),
        recipients_path=(str, Path),
        restored_email_template=(str, Path, type(None)),
        outage_email_template=(str, Path, type(None)),
    )
    def __init__(self, site_configs, mag_availability_path=Path('magnetometer_availability.json'),
                 monitor_status_path='magnetometer_monitor_status.json', alert_threshold=pd.Timedelta(hours=2),
                 reset_monitor_status=False, email_config_path=Path('email_config.json'), recipients_path=Path('recipients.txt'),
                 restored_email_template=None,
                 outage_email_template=None,
                 now=None):
        self.site_configs = site_configs
        self.now = utc_timestamp(now)
        self.mag_availability_path = Path(mag_availability_path)
        with open(self.mag_availability_path, 'r') as f:
            self.mag_availability = json.loads(f.read())
        template_dir = Path(__file__).parent / "assets" / "email_templates"
        self.restored_email_template = (
            Path(restored_email_template)
            if restored_email_template is not None
            else template_dir / "magnetometer_restored.html"
        )
        self.outage_email_template = (
            Path(outage_email_template)
            if outage_email_template is not None
            else template_dir / "magnetometer_outage.html"
        )
        self.email_config_path = Path(email_config_path)
        self.recipients_path = Path(recipients_path)
        self.alert_threshold = alert_threshold
        self.monitor_status_path = Path(monitor_status_path)
        if os.path.isfile(self.monitor_status_path) and not reset_monitor_status:
            with open(self.monitor_status_path, 'r') as f:
                self.monitor_status = json.loads(f.read())
        else:
            self.monitor_status = create_monitor_status(
                site_configs,
                self.mag_availability,
                latest_allowed=self.now,
            )
        self.refresh_monitor_status_from_site_configs()

    def refresh_monitor_status_from_site_configs(self):
        """
        Reconcile the loaded status JSON with current site configuration.
        """

        self.monitor_status = refresh_monitor_status_from_site_configs(
            self.site_configs,
            self.monitor_status,
            self.mag_availability,
            self.now,
        )

    def get_date(self, date=None, now=None):
        """
        Return a copy of monitor status updated with one archive day's files.

        ``now`` is used to clamp future measurements and calculate
        ``time_since_last``. Passing it explicitly makes historical tests
        deterministic.
        """

        if date is None:
            date = self.now.date()
        if now is None:
            now = self.now
        else:
            now = utc_timestamp(now)
        monitor_status = self.monitor_status.copy()
        for site in self.site_configs:
            if site.permanently_off:
                continue
            site_files = get_files(date, site)
            latest_time = None
            for site_file in site_files:
                file_latest_time = latest_valid_time_from_file(
                    site_file,
                    site.null_data_value,
                    latest_allowed=now,
                )
                if latest_time is None or file_latest_time > latest_time:
                    latest_time = file_latest_time

            if latest_time is not None:
                if monitor_status['stations'][site.code]['last_measurement']==str(pd.NaT) or latest_time > pd.Timestamp(monitor_status['stations'][site.code]['last_measurement']):
                    monitor_status['stations'][site.code]['last_measurement'] = str(latest_time)
                    monitor_status['stations'][site.code]['time_since_last'] = str(now-latest_time)
        return monitor_status
    
    def update_monitor_status(self):
        """
        Scan the recent archive window and update ``self.monitor_status``.
        """

        for date in pd.date_range(start=self.now-pd.Timedelta(days=2), end=self.now, freq='D'):
            self.monitor_status.update(self.get_date(date, self.now))
        self.monitor_status['generated_at'] = self.now.isoformat()
    
    def check_status(self):
        """
        Mark sites online/offline according to ``alert_threshold``.

        ``status_change`` is set only when a site crosses a status boundary.
        Email sending clears that flag after the corresponding notification is
        queued.
        """

        checked_at = utc_timestamp(self.monitor_status.get('generated_at', self.now))
        if checked_at is pd.NaT:
            checked_at = self.now

        for site in self.site_configs:
            if site.permanently_off:
                continue
            last_measurement = pd.Timestamp(self.monitor_status['stations'][site.code]['last_measurement'])
            if last_measurement is pd.NaT:
                time_since_last = pd.NaT
            else:
                if last_measurement.tzinfo is None:
                    last_measurement = last_measurement.tz_localize('UTC')
                else:
                    last_measurement = last_measurement.tz_convert('UTC')
                time_since_last = checked_at - last_measurement
                self.monitor_status['stations'][site.code]['time_since_last'] = str(time_since_last)

            if time_since_last is pd.NaT or time_since_last > self.alert_threshold:
                if self.monitor_status['stations'][site.code]['status'] != 'offline':
                    self.monitor_status['stations'][site.code]['status_change'] = True
                    self.monitor_status['stations'][site.code]['status'] = 'offline'

            else:
                if self.monitor_status['stations'][site.code]['status'] != 'online':
                    self.monitor_status['stations'][site.code]['status_change'] = True
                    self.monitor_status['stations'][site.code]['status'] = 'online'

    def email_alert(self):
        """
        Send batched outage and restored emails for pending status changes.
        """

        outage_events = []
        restored_events = []
        for site in self.site_configs:
            if site.permanently_off:
                continue
            # Only queue transition emails. Persistent outages are deduplicated
            # by leaving status_change false after the first notification.
            elif self.monitor_status['stations'][site.code]['status_change'] and self.monitor_status['stations'][site.code]['status'] == 'offline':
                outage_events.append([site, self.monitor_status['stations'][site.code]])
                self.monitor_status['stations'][site.code]['status_change']= False
            elif self.monitor_status['stations'][site.code]['status_change'] and self.monitor_status['stations'][site.code]['status'] == 'online':
                restored_events.append([site, self.monitor_status['stations'][site.code]])
                self.monitor_status['stations'][site.code]['status_change']= False
        if outage_events:
            site_count = len(outage_events)
            subject = (
                f"Magnetometer warning: {outage_events[0][0].code} is not sending data"
                if site_count == 1
                else f"Magnetometer warning: {number_word(site_count)} sites are not sending data"
            )
            send_batched_status_email(
                subject=subject,
                template_path=self.outage_email_template,
                config=self.email_config_path,
                recipients=self.recipients_path,
                checked_at= pd.Timestamp(self.monitor_status['generated_at'], tz='UTC'),
                events=outage_events,
                threshold_minutes=round(self.alert_threshold.total_seconds() / 60),
            )

        if restored_events:
            site_count = len(restored_events)
            subject = (
                f"Magnetometer restored: {restored_events[0][0].code} is sending data again"
                if site_count == 1
                else f"Magnetometer restored: {number_word(site_count)} sites are sending data again"
            )
            send_batched_status_email(
                subject=subject,
                template_path=self.restored_email_template,
                config=self.email_config_path,
                recipients=self.recipients_path,
                checked_at= pd.Timestamp(self.monitor_status['generated_at'], tz='UTC'),
                events=restored_events,
            )

    def save_monitor_status(self):
        """
        Persist ``self.monitor_status`` as pretty-printed JSON.
        """

        with open(self.monitor_status_path, 'w') as f:
            f.write(json.dumps(self.monitor_status, indent=2))
    
    def run_monitor(self):
        """
        Run one full monitor cycle: update, classify, notify, and save.
        """

        self.update_monitor_status()
        self.check_status()
        self.email_alert()
        self.save_monitor_status()
