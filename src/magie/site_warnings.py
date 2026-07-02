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

The monitor reads persistent IAGA-2002 files only. Legacy TXT files are
expected to be converted by the live IAGA updater before monitoring runs.

A site is considered active only if at least one of x, y, z contains a finite
value at a timestamp. A timestamp with only NaNs is treated as missing data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from string import Template

import numpy as np
from magpy.stream import DataStream, read

from magie.email_utils import (
    load_email_config,
    load_recipients,
    render_html_template,
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
class MonitorConfig:
    """
    Runtime configuration for the monitor.

    Parameters
    ----------
    data_root:
        Root directory containing ``YYYY/MM/DD`` data folders, or a mapping
        from normalised site code to root directory. This allows mixed archive
        roots, for example ``{"dun": Path("/archive"), "flo": Path("/flo")}``.
    monitor_status_path:
        Monitor-owned JSON file used to remember last measurements and whether
        each site was already offline. If it does not exist, it is created on
        the first monitor run. This prevents repeated emails every time the
        monitor runs.
    email_config_path:
        TOML file consumed by ``load_email_config``.
    recipients_path:
        Text file with one recipient email address per line.
    outage_email_template:
        HTML template for outage warnings.
    restored_email_template:
        HTML template for restored notifications.
    max_inactivity:
        Maximum allowed time since the last valid measurement.
    availability_lookup_path:
        JSON availability lookup written by ``build_availability_lookup``.
        When present, the monitor uses this for last-seen checks instead of
        rereading archive files for every site.
    availability_lookup_max_age:
        Optional maximum age of the lookup file's ``generated_at`` timestamp
        before the monitor ignores it and falls back to direct archive reads.
        ``None`` trusts the lookup regardless of age.
    reset_monitor_status:
        When ``True``, ignore any existing monitor status JSON and write a fresh
        one at the end of the run. Defaults to ``False`` so alert deduplication
        persists across normal monitor runs.
    force_restored_email:
        When ``True``, send a restored email for each site that currently has
        valid data, even if the previous monitor status did not mark it
        offline. Intended only for template/email testing.
    """

    data_root: Path | str | Mapping[str, Path | str]
    monitor_status_path: Path
    email_config_path: Path
    recipients_path: Path
    outage_email_template: Path
    restored_email_template: Path
    max_inactivity: timedelta
    availability_lookup_path: Path
    availability_lookup_max_age: timedelta | None = None
    reset_monitor_status: bool = False
    force_restored_email: bool = False


@dataclass
class SiteStatusEvent:
    """
    Status transition for one monitored site during a monitor run.
    """

    site_code: str
    site_name: str
    last_seen_at: str
    time_since_last: str


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


@dataclass
class MagnetometerReadResult:
    """
    Loaded magnetometer file and metadata derived while reading it.

    Attributes
    ----------
    stream:
        MagPy data stream loaded from the source file.
    source_path:
        Original file path that was read.
    cadence_path:
        Path used to infer cadence from the IAGA filename, for example
        ``psec`` or ``pmin``.
    """

    stream: DataStream
    source_path: Path
    cadence_path: Path


class SiteCheckError(RuntimeError):
    """Raised when a site alert check cannot determine a last-valid timestamp."""


@enforce_types(path=(str, Path))
def load_monitor_status(path: Path) -> dict:
    """
    Load monitor status from disk, creating an empty structure when missing.

    The status records each site's last measurement and whether it was offline
    during the previous check.
    The ``offline`` flag is reserved for detected outages. Sites that are
    intentionally disabled are recorded with ``status`` values such as
    ``"inactive"`` or ``"permanently_off"`` instead.
    """

    path = Path(path)

    if not path.exists():
        return {"generated_at": None, "stations": {}}

    status = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(status, dict):
        return {"generated_at": None, "stations": {}}

    if isinstance(status.get("stations"), dict):
        status.setdefault("generated_at", None)
        return status

    # Backwards compatibility for the previous flat state shape keyed by site.
    stations = {
        site_code: site_state
        for site_code, site_state in status.items()
        if isinstance(site_state, dict)
    }
    for site_state in stations.values():
        if "last_measurement" not in site_state and "last_seen_at" in site_state:
            site_state["last_measurement"] = site_state.get("last_seen_at")
    return {"generated_at": status.get("generated_at"), "stations": stations}


@enforce_types(path=(str, Path), status=dict, generated_at=datetime)
def save_monitor_status(path: Path, status: dict, generated_at: datetime) -> None:
    """
    Save monitor status to disk as pretty-printed JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status["generated_at"] = generated_at.isoformat()
    status.setdefault("stations", {})
    path.write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@enforce_types(value=datetime)
def format_ut_datetime(value: datetime) -> str:
    """
    Format a datetime as a timezone-free UT timestamp for email text.
    """

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@enforce_types(delta=timedelta)
def format_duration(delta: timedelta) -> str:
    """
    Format a duration using years, days, hours, minutes, and seconds.
    """

    total_seconds = max(0, int(round(delta.total_seconds())))
    units = [
        ("year", 365 * 24 * 60 * 60),
        ("day", 24 * 60 * 60),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1),
    ]
    parts = []

    for label, unit_seconds in units:
        value, total_seconds = divmod(total_seconds, unit_seconds)
        if value:
            suffix = "" if value == 1 else "s"
            parts.append(f"{value} {label}{suffix}")

    return ", ".join(parts) if parts else "0 seconds"


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


@enforce_types(path=(str, Path, type(None)))
def load_availability_lookup(path: Path | None) -> dict | None:
    """
    Load a build_availability_lookup JSON file if it is configured and present.

    The monitor treats this file as an optimisation/cache. Missing, malformed,
    or incomplete lookup data should not stop monitoring because the archive
    scan path can still calculate last-seen times directly from data files.
    """

    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        return None

    try:
        lookup = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(lookup, dict):
        return None

    stations = lookup.get("stations")
    if not isinstance(stations, dict):
        return None

    return lookup


@enforce_types(value=(str, type(None)))
def parse_lookup_datetime(value: str | None) -> datetime | None:
    """
    Parse an ISO datetime from the availability lookup as UTC.
    """

    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


@enforce_types(site_code=str, availability_lookup=(dict, type(None)))
def latest_available_day_from_lookup(
    site_code: str,
    availability_lookup: dict | None,
) -> datetime | None:
    """
    Return the latest UTC day marked available for a site in the lookup.
    """

    if availability_lookup is None:
        return None

    station = availability_lookup.get("stations", {}).get(site_code)
    if not isinstance(station, dict):
        return None

    available_dates = station.get("available_dates", {})
    if not isinstance(available_dates, dict):
        return None

    true_dates = sorted(
        date_key
        for date_key, has_data in available_dates.items()
        if has_data is True
    )
    if not true_dates:
        return None

    try:
        latest = datetime.fromisoformat(true_dates[-1])
    except ValueError:
        return None

    if latest.tzinfo is None:
        return latest.replace(tzinfo=timezone.utc)

    return latest.astimezone(timezone.utc)


@enforce_types(site_code=str, availability_lookup=(dict, type(None)))
def last_valid_measurement_from_lookup(
    site_code: str,
    availability_lookup: dict | None,
) -> datetime | None:
    """
    Return the last valid measurement timestamp stored in the availability lookup.

    This is a fallback for alert checks when the latest daily IAGA file cannot
    be read. If the lookup does not contain a parseable timestamp, the monitor
    should treat the check as inconclusive rather than alerting with an unknown
    last-seen time.
    """

    if availability_lookup is None:
        return None

    station = availability_lookup.get("stations", {}).get(site_code)
    if not isinstance(station, dict):
        return None

    last_valid = parse_lookup_datetime(station.get("last_valid_measurement"))
    if last_valid is not None:
        return last_valid

    dated_measurements = station.get("latest_valid_measurement_by_date", {})
    if not isinstance(dated_measurements, dict):
        return None

    for date_key in sorted(dated_measurements, reverse=True):
        last_valid = parse_lookup_datetime(dated_measurements.get(date_key))
        if last_valid is not None:
            return last_valid

    return None


@enforce_types(
    availability_lookup=(dict, type(None)),
    now=datetime,
    max_age=(timedelta, type(None)),
)
def availability_lookup_is_fresh(
    availability_lookup: dict | None,
    now: datetime,
    max_age: timedelta | None,
) -> bool:
    """
    Return whether a loaded availability lookup is recent enough to trust.
    """

    if availability_lookup is None:
        return False

    if max_age is None:
        return True

    generated_at = parse_lookup_datetime(availability_lookup.get("generated_at"))
    if generated_at is None:
        return False

    return now - generated_at <= max_age


@enforce_types(site=SiteConfig, default_data_root=(str, Path, Mapping))
def data_root_for_site(
    site: SiteConfig,
    default_data_root: Path | str | Mapping[str, Path | str],
) -> Path:
    """
    Return the archive root to use for one site.

    Per-site ``SiteConfig.data_root`` takes precedence. Otherwise,
    ``default_data_root`` may be either one shared root path or a mapping keyed
    by normalised site code. Mapping values may be strings or ``Path`` objects.
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


@enforce_types(
    data_root=(str, Path),
    site_code=str,
    day=datetime,
    today=datetime,
)
def candidate_files_for_day(
    data_root: Path,
    site_code: str,
    day: datetime,
    today: datetime,
) -> list[Path]:
    """
    Return possible files for one site on one day, in preferred read order.

    For today's date, the live TXT files are tried before IAGA files because the
    IAGA files may not exist yet or may be stale.

    For older dates, IAGA files are tried first because they are the converted
    end-of-day products and can be read directly by MagPy.

    Parameters
    ----------
    data_root:
        Root data directory.
    site_code:
        Magnetometer site code.
    day:
        Date to search.
    today:
        Current date/time, used to decide whether ``day`` is today.

    Returns
    -------
    list[pathlib.Path]
        Candidate file paths. These may or may not exist.
    """

    data_root = Path(data_root)

    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    dd = day.strftime("%d")
    ymd = day.strftime("%Y%m%d")
    doy = day.strftime("%j")

    day_dir = data_root / yyyy / mm / dd

    iaga_dir = day_dir / "iaga2002"
    iaga_files = [
        iaga_dir / f"{site_code}{ymd}psec.sec",
        iaga_dir / f"{site_code}{ymd}pmin.min",
    ]
    iaga_files.extend(sorted(iaga_dir.glob(f"{site_code}{ymd}*sec.sec")))
    iaga_files.extend(sorted(iaga_dir.glob(f"{site_code}{ymd}*min.min")))
    return list(dict.fromkeys(iaga_files))


@enforce_types(path=(str, Path))
def read_magnetometer_file_with_metadata(path: Path) -> MagnetometerReadResult:
    """
    Read a magnetometer file and return the stream plus cadence metadata.

    IAGA ``.sec`` and ``.min`` files are read directly with
    ``magpy.stream.read``. Legacy TXT files are not supported here; they must
    be converted to persistent IAGA-2002 files before this monitor runs.

    Parameters
    ----------
    path:
        Path to the data file.

    Returns
    -------
    MagnetometerReadResult
        Loaded stream, original source path, and cadence-informative path.

    Raises
    ------
    ValueError
        If the file type is unsupported.
    """

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".sec", ".min"}:
        return MagnetometerReadResult(
            stream=read(str(path)),
            source_path=path,
            cadence_path=path,
        )

    raise ValueError(f"Unsupported magnetometer file type: {path}")


@enforce_types(path=(str, Path))
def read_magnetometer_file(path: Path) -> DataStream:
    """
    Read either an IAGA file or a legacy TXT file as a MagPy DataStream.

    This compatibility wrapper returns only the stream. Use
    ``read_magnetometer_file_with_metadata`` when the caller also needs the
    converted IAGA filename for cadence inference.
    """

    return read_magnetometer_file_with_metadata(path).stream

@enforce_types(stream=DataStream)
def latest_valid_time(stream: DataStream) -> datetime | None:
    """
    Find the latest timestamp where x, y, or z contains real data.

    MagPy streams may contain a full time grid even when data are missing.
    Therefore the latest timestamp alone is not enough. We only accept a
    timestamp if at least one of ``x``, ``y`` or ``z`` is finite.

    Parameters
    ----------
    stream:
        MagPy DataStream object supporting dictionary-style access, e.g.
        ``stream["time"]``, ``stream["x"]``, ``stream["y"]``, ``stream["z"]``.

    Returns
    -------
    datetime.datetime or None
        Latest valid measurement time, or ``None`` if no valid x/y/z value is
        found.
    """

    try:
        times = np.asarray(stream["time"], dtype=object)
        x = np.asarray(stream["x"], dtype=float)
        y = np.asarray(stream["y"], dtype=float)
        z = np.asarray(stream["z"], dtype=float)
    except Exception:
        return None

    if len(times) == 0:
        return None

    valid = np.isfinite(x) | np.isfinite(y) | np.isfinite(z)

    if not np.any(valid):
        return None

    last_index = np.where(valid)[0][-1]
    last_time = times[last_index]

    if isinstance(last_time, datetime):
        if last_time.tzinfo is None:
            return last_time.replace(tzinfo=timezone.utc)
        return last_time.astimezone(timezone.utc)

    return None


@enforce_types(site_code=str, day=datetime, data_root=(str, Path))
def get_last_measurement_for_day(
    site_code: str,
    day: datetime,
    data_root: Path,
) -> datetime | None:
    """
    Find the latest valid measurement for a site on one UTC day.

    This is used with the availability lookup: the lookup says which days have
    data, and this function reads that day's IAGA files to get the actual latest
    timestamp.
    """

    latest: datetime | None = None

    for path in candidate_files_for_day(data_root, site_code, day, today=day):
        if not path.exists():
            continue

        try:
            stream = read(str(path))
            last_seen = latest_valid_time(stream)
        except Exception:
            continue

        if last_seen is None:
            continue

        if latest is None or last_seen > latest:
            latest = last_seen

    return latest


@enforce_types(
    subject=str,
    template_path=(str, Path),
    cfg=MonitorConfig,
    values=dict,
)
def send_status_email(
    subject: str,
    template_path: Path,
    cfg: MonitorConfig,
    values: dict[str, str | int | float],
) -> None:
    """
    Render and send one HTML status email.

    This uses your existing email utilities:
        - load_email_config
        - load_recipients
        - render_html_template
        - send_html_email
    """

    template_path = Path(template_path)
    email_cfg = load_email_config(cfg.email_config_path)
    recipients = load_recipients(cfg.recipients_path)

    html = render_html_template(str(template_path), values)

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


@enforce_types(events=list)
def event_rows_html(events: list[SiteStatusEvent]) -> str:
    """
    Render escaped table rows for a batch of site status events.
    """

    rows = []

    for event in events:
        rows.append(
            "<tr>"
            f"<td>{escape(event.site_name)} ({escape(event.site_code)})</td>"
            f"<td>{escape(event.last_seen_at)}</td>"
            f"<td>{escape(event.time_since_last)}</td>"
            "</tr>"
        )

    return "\n".join(rows)


@enforce_types(
    subject=str,
    template_path=(str, Path),
    cfg=MonitorConfig,
    values=dict,
    events=list,
)
def send_batched_status_email(
    subject: str,
    template_path: Path,
    cfg: MonitorConfig,
    values: dict[str, str | int | float],
    events: list[SiteStatusEvent],
) -> None:
    """
    Render and send one status email containing all events in a monitor run.
    """

    template_path = Path(template_path)
    site_rows = event_rows_html(events)
    values = {
        **values,
        "site_count": number_word(len(events)),
        "site_label": "site" if len(events) == 1 else "sites",
        "site_verb": "is" if len(events) == 1 else "are",
    }

    email_cfg = load_email_config(cfg.email_config_path)
    recipients = load_recipients(cfg.recipients_path)

    template_text = template_path.read_text(encoding="utf-8")
    if "checked_at" in values:
        checked_at = parse_lookup_datetime(str(values["checked_at"]))
        if checked_at is not None:
            values["checked_at"] = format_ut_datetime(checked_at)

    safe_values = {
        key: escape(str(value)) for key, value in values.items()
    }
    safe_values["site_rows"] = site_rows
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


@enforce_types(
    site=SiteConfig,
    cfg=MonitorConfig,
    state=dict,
    now=datetime,
    availability_lookup=(dict, type(None)),
)
def check_site(
    site: SiteConfig,
    cfg: MonitorConfig,
    state: dict,
    now: datetime,
    availability_lookup: dict | None = None,
) -> tuple[str, SiteStatusEvent] | None:
    """
    Check one magnetometer site and return an outage/restored event if needed.

    Email behaviour
    ---------------
    Was online, now offline:
        Return an outage event.

    Still offline:
        Send nothing.

    Was offline, now online:
        Return a restored event.

    Still online:
        Send nothing.

    The ``state`` dictionary is updated in-place. Emails are sent by
    ``run_monitor`` after all sites have been checked, so multiple site events
    from one run can be batched into one message.
    """

    site_state = state.setdefault(
        site.code,
        {
            "offline": False,
            "status": "unknown",
            "last_measurement": None,
            "last_alert_sent_at": None,
            "last_restored_sent_at": None,
        },
    )

    if site.permanently_off:
        site_state["offline"] = False
        site_state["status"] = "permanently_off"
        return None

    if not site.active:
        site_state["offline"] = False
        site_state["status"] = "inactive"
        return None

    site_data_root = data_root_for_site(site, cfg.data_root)
    latest_available_day = latest_available_day_from_lookup(
        site.code,
        availability_lookup,
    )
    last_seen = None

    if latest_available_day is not None:
        last_seen = get_last_measurement_for_day(
            site_code=site.code,
            day=latest_available_day,
            data_root=site_data_root,
        )

    if last_seen is None:
        last_seen = last_valid_measurement_from_lookup(
            site.code,
            availability_lookup,
        )

    if last_seen is None:
        site_state["status"] = "check_failed"
        raise SiteCheckError(
            f"Could not determine last valid measurement for site {site.code!r}. "
            "Suppressing alert email rather than sending an unknown last-seen time."
        )

    else:
        age = now - last_seen
        is_offline = age > cfg.max_inactivity
        time_since_last = format_duration(age)
        last_seen_text = format_ut_datetime(last_seen)
        site_state["last_measurement"] = last_seen.isoformat()

    was_offline = bool(site_state.get("offline", False))

    if is_offline and not was_offline:
        site_state["offline"] = True
        site_state["status"] = "offline"
        site_state["last_alert_sent_at"] = now.isoformat()
        return (
            "outage",
            SiteStatusEvent(
                site_code=site.code,
                site_name=site.name,
                last_seen_at=last_seen_text,
                time_since_last=time_since_last,
            ),
        )

    elif not is_offline and (was_offline or cfg.force_restored_email):
        site_state["offline"] = False
        site_state["status"] = "online"
        site_state["last_restored_sent_at"] = now.isoformat()
        return (
            "restored",
            SiteStatusEvent(
                site_code=site.code,
                site_name=site.name,
                last_seen_at=last_seen_text,
                time_since_last=time_since_last,
            ),
        )

    else:
        site_state["offline"] = is_offline
        site_state["status"] = "offline" if is_offline else "online"
        return None


@enforce_types(sites=list, cfg=MonitorConfig)
def run_monitor(sites: list[SiteConfig], cfg: MonitorConfig) -> None:
    """
    Run the monitor once for all configured sites.

    This function is intended to be called periodically from cron or systemd,
    for example every 5 or 10 minutes.

    It loads the previous state, checks all sites, sends any required emails,
    and writes the updated state back to disk.
    """

    now = datetime.now(timezone.utc)
    if cfg.reset_monitor_status:
        monitor_status = {"generated_at": None, "stations": {}}
    else:
        monitor_status = load_monitor_status(cfg.monitor_status_path)
    state = monitor_status.setdefault("stations", {})
    availability_lookup = load_availability_lookup(cfg.availability_lookup_path)
    if not availability_lookup_is_fresh(
        availability_lookup,
        now,
        cfg.availability_lookup_max_age,
    ):
        availability_lookup = None
    outage_events: list[SiteStatusEvent] = []
    restored_events: list[SiteStatusEvent] = []

    for site in sites:
        event = check_site(
            site,
            cfg,
            state,
            now,
            availability_lookup,
        )

        if event is None:
            continue

        event_type, site_event = event

        if event_type == "outage":
            outage_events.append(site_event)
        elif event_type == "restored":
            restored_events.append(site_event)

    if outage_events:
        site_count = len(outage_events)
        subject = (
            f"Magnetometer warning: {outage_events[0].site_code} is not sending data"
            if site_count == 1
            else f"Magnetometer warning: {number_word(site_count)} sites are not sending data"
        )
        send_batched_status_email(
            subject=subject,
            template_path=cfg.outage_email_template,
            cfg=cfg,
            values={
                "threshold_minutes": round(cfg.max_inactivity.total_seconds() / 60, 1),
                "checked_at": format_ut_datetime(now),
            },
            events=outage_events,
        )

    if restored_events:
        site_count = len(restored_events)
        subject = (
            f"Magnetometer restored: {restored_events[0].site_code} is sending data again"
            if site_count == 1
            else f"Magnetometer restored: {number_word(site_count)} sites are sending data again"
        )
        send_batched_status_email(
            subject=subject,
            template_path=cfg.restored_email_template,
            cfg=cfg,
            values={
                "checked_at": format_ut_datetime(now),
            },
            events=restored_events,
        )

    save_monitor_status(cfg.monitor_status_path, monitor_status, now)

@enforce_types(seconds=(int, float, np.number, type(None)))
def frequency_from_seconds(seconds: float | None) -> tuple[str | None, int | None]:
    """
    Return a frequency label and expected daily sample count.

    Parameters
    ----------
    seconds:
        Estimated cadence between valid samples.

    Returns
    -------
    tuple
        ``("sec", 86400)`` for one-second cadence, ``("min", 1440)`` for
        one-minute cadence, or a generic seconds label and expected count for
        other cadences. ``(None, None)`` is returned when cadence is unknown.
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
    Infer cadence from a known magnetometer filename when stream timing is sparse.

    Parameters
    ----------
    path:
        Candidate magnetometer file path.

    Returns
    -------
    tuple
        Frequency label and expected daily sample count inferred from ``psec``,
        ``pmin``, ``.sec``, or ``.min`` naming.
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

    Parameters
    ----------
    frequency:
        Frequency label such as ``"sec"``, ``"min"``, ``"1-second"`` or
        ``"1-minute"``.

    Returns
    -------
    int or None
        Expected samples in a complete UTC day, or ``None`` when the label is
        unknown.
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
def stream_availability(
    stream,
    path: Path,
    cadence_path: Path | None = None,
) -> DayAvailability:
    """
    Return valid-sample count, latest timestamp, and coverage for one stream.

    Parameters
    ----------
    stream:
        MagPy stream containing ``time``, ``x``, ``y`` and ``z`` arrays.
    path:
        Source file path used for reporting.
    cadence_path:
        Optional path or filename used for cadence inference. For converted TXT
        files this should be the converter's suggested IAGA filename.

    Returns
    -------
    DayAvailability
        Coverage summary for this one file. Timestamps where all x/y/z values
        are non-finite do not count as valid samples.
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
    for t in times[valid]:
        if not isinstance(t, datetime):
            continue
        if t.tzinfo is None:
            valid_times.append(t.replace(tzinfo=timezone.utc))
        else:
            valid_times.append(t)
    valid_times = [t.astimezone(timezone.utc) for t in valid_times]

    if not valid_times:
        return DayAvailability(False, None, 0, None, None, None, str(path))

    valid_times = sorted(set(valid_times))
    latest = valid_times[-1]
    valid_samples = len(valid_times)
    cadence_seconds = None

    if len(valid_times) > 1:
        deltas = np.diff([t.timestamp() for t in valid_times])
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


@enforce_types(site_code=str, day=datetime, data_root=(str, Path))
def day_data_availability(site_code: str, day: datetime, data_root: Path) -> DayAvailability:
    """
    Return valid data availability for one site on one UTC day.

    The monitor may have multiple candidate files for a site/day, for example
    live TXT data plus converted IAGA files. This function evaluates every
    existing candidate file and returns the best available coverage summary.

    Parameters
    ----------
    site_code:
        Normalised site code used in filenames.
    day:
        UTC day to inspect.
    data_root:
        Archive root containing ``YYYY/MM/DD`` folders.

    Returns
    -------
    DayAvailability
        Best coverage summary found for the day, or an empty summary when no
        valid data are present.
    """

    data_root = Path(data_root)
    best = DayAvailability(False, None, 0, None, None, None, None)

    for path in candidate_files_for_day(data_root, site_code, day, today=day):
        if not path.exists():
            continue

        try:
            read_result = read_magnetometer_file_with_metadata(path)
            availability = stream_availability(
                read_result.stream,
                read_result.source_path,
                cadence_path=read_result.cadence_path,
            )
        except Exception:
            continue

        if not availability.has_data:
            continue

        if availability.coverage_percent is None:
            if not best.has_data or availability.valid_samples > best.valid_samples:
                best = availability
            continue

        if (
            best.coverage_percent is None
            or availability.coverage_percent > best.coverage_percent
        ):
            best = availability

    return best


@enforce_types(site_code=str, day=datetime, data_root=(str, Path))
def day_has_valid_data(site_code: str, day: datetime, data_root: Path) -> tuple[bool, datetime | None]:
    """
    Return whether a site has any valid x/y/z data on a given day.

    Also returns the latest valid timestamp from that day, if present.
    """

    availability = day_data_availability(site_code, day, data_root)
    return availability.has_data, availability.latest


@enforce_types(site_code=str, data_root=(str, Path), day=datetime)
def availability_job(site_code: str, data_root: Path, day: datetime) -> tuple[str, DayAvailability]:
    """
    Run one site/day availability check for parallel execution.
    """

    return day.strftime("%Y-%m-%d"), day_data_availability(site_code, day, data_root)


@enforce_types(site=SiteConfig, site_data_root=(str, Path), day=datetime)
def site_availability_job(
    site: SiteConfig,
    site_data_root: Path,
    day: datetime,
) -> tuple[SiteConfig, str, DayAvailability]:
    """
    Run one site/day availability check and include the site in the result.
    """

    date_key, availability = availability_job(site.code, site_data_root, day)
    return site, date_key, availability


@enforce_types(
    sites=list,
    data_root=(str, Path, Mapping),
    start_date=datetime,
    end_date=datetime,
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

    ``parallel_jobs`` controls site/day parallelism. Use ``1`` for serial
    execution, or a larger value to use joblib with a tqdm progress bar.

    When ``update_existing`` is true, an existing output file is loaded and
    dates in the requested range are recalculated and overwritten while dates
    outside the range are preserved.

    Parameters
    ----------
    sites:
        List of ``SiteConfig`` objects to include in the output.
    data_root:
        One shared archive root, or a mapping from site code to archive root.
    start_date, end_date:
        Inclusive UTC date range to calculate.
    output_path:
        JSON file to write.
    parallel_jobs:
        Number of joblib workers. ``1`` runs serially.
    show_progress:
        Whether to show the joblib-backed tqdm progress bar.
    update_existing:
        If true, merge into an existing file while overwriting the requested
        date range. If false, rebuild the output from only the requested range.
    """

    from joblib import Parallel, delayed

    output_path = Path(output_path)

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
