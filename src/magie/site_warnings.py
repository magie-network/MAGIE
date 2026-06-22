"""
magnetometer_email_monitor.py

Monitor magnetometer data files and send email warnings when a site has stopped
producing valid measurements. Also sends a restored email when valid data returns.

Expected data layout
--------------------
Data are organised under a root directory like:

    /YYYY/MM/DD/

For live/current data, legacy TXT files may live at:

    /YYYY/MM/DD/txt/dunYYYYDDD.txt
    /YYYY/MM/DD/txt/dunYYYYMMDD.txt

where:
    dun  = site code
    YYYY = year
    DDD  = day of year

For completed days, converted IAGA files may exist, for example:

    /YYYY/MM/DD/iaga2002/dunYYYYMMDDpsec.sec
    /YYYY/MM/DD/iaga2002/dunYYYYMMDDpmin.min

The monitor prefers:
    - today's live TXT file first
    - previous days' IAGA files first
    - TXT fallback if IAGA files do not exist

A site is considered active only if at least one of x, y, z contains a finite
value at a timestamp. A timestamp with only NaNs is treated as missing data.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from string import Template
from typing import Iterable

import numpy as np
from magpy.stream import DataStream, read

from magie.email_utils import (
    load_email_config,
    load_recipients,
    render_html_template,
    send_html_email,
)
from magie.utils import enforce_types, get_site_metadata, tqdm_joblib

# Adjust this import to match wherever your converter actually lives.
from magie.file_conversions import magie_legacy2iaga2002


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
        from site code to root directory.
    state_path:
        JSON file used to remember whether each site was already offline.
        This prevents repeated emails every time the monitor runs.
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
    lookback_days:
        Number of days to search backwards when a site has been inactive across
        one or more day boundaries.
    """

    data_root: Path | Mapping[str, Path]
    state_path: Path
    email_config_path: Path
    recipients_path: Path
    outage_email_template: Path
    restored_email_template: Path
    max_inactivity: timedelta
    lookback_days: int = 7


@dataclass
class SiteStatusEvent:
    """
    Status transition for one monitored site during a monitor run.
    """

    site_code: str
    site_name: str
    last_seen_at: str
    minutes_since_last: str | float


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
        Path or filename that best describes cadence. For TXT files this is
        the IAGA filename suggested by ``magie_legacy2iaga2002`` because it
        includes ``psec`` or ``pmin``.
    """

    stream: DataStream
    source_path: Path
    cadence_path: Path


@enforce_types(path=(str, Path))
def load_state(path: Path) -> dict:
    """
    Load monitor state from disk.

    The state records whether each site was offline during the previous check.
    The ``offline`` flag is reserved for detected outages. Sites that are
    intentionally disabled are recorded with ``status`` values such as
    ``"inactive"`` or ``"permanently_off"`` instead.
    If the file does not exist, an empty state is returned.

    Returns
    -------
    dict
        State dictionary keyed by site code.
    """

    path = Path(path)

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


@enforce_types(path=(str, Path), state=dict)
def save_state(path: Path, state: dict) -> None:
    """
    Save monitor state to disk as pretty-printed JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


@enforce_types(site=SiteConfig, default_data_root=(str, Path, Mapping))
def data_root_for_site(
    site: SiteConfig,
    default_data_root: Path | Mapping[str, Path],
) -> Path:
    """
    Return the archive root to use for one site.

    Per-site ``SiteConfig.data_root`` takes precedence. Otherwise,
    ``default_data_root`` may be either one shared root path or a mapping keyed
    by normalised site code.
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

    txt_files = [
        day_dir / "txt" / f"{site_code}{yyyy}{doy}.txt",
        day_dir / "txt" / f"{site_code}{ymd}.txt",
    ]

    iaga_dir = day_dir / "iaga2002"
    iaga_files = [
        iaga_dir / f"{site_code}{ymd}psec.sec",
        iaga_dir / f"{site_code}{ymd}pmin.min",
    ]

    if day.date() == today.date():
        return txt_files + iaga_files

    return iaga_files + txt_files


@enforce_types(
    data_root=(str, Path),
    site_code=str,
    now=datetime,
    lookback_days=int,
)
def candidate_files(
    data_root: Path,
    site_code: str,
    now: datetime,
    lookback_days: int,
) -> Iterable[Path]:
    """
    Yield existing candidate files for a site, newest day first.

    This allows the monitor to handle inactivity across midnight or across
    several days. For example, if today's file contains only NaNs, the monitor
    can look back to yesterday's IAGA or TXT file to find the true last valid
    measurement.

    Parameters
    ----------
    data_root:
        Root data directory.
    site_code:
        Magnetometer site code.
    now:
        Current datetime.
    lookback_days:
        Number of days to search backwards.

    Yields
    ------
    pathlib.Path
        Existing files only, in preferred read order.
    """

    for offset in range(lookback_days):
        day = now - timedelta(days=offset)

        for path in candidate_files_for_day(data_root, site_code, day, now):
            if path.exists():
                yield path


@enforce_types(path=(str, Path))
def read_magnetometer_file_with_metadata(path: Path) -> MagnetometerReadResult:
    """
    Read a magnetometer file and return the stream plus cadence metadata.

    IAGA ``.sec`` and ``.min`` files are read directly with
    ``magpy.stream.read``.

    Legacy ``.txt`` files are first converted using ``magie_legacy2iaga2002``.
    The converted IAGA text is written to a temporary file, then read by MagPy.
    The converter's suggested IAGA filename is retained as ``cadence_path`` so
    downstream code can infer whether the TXT file represented second or minute
    data.

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

    if suffix == ".txt":
        iaga_text, suggested_filename = magie_legacy2iaga2002(str(path))

        cadence_path = Path(suggested_filename)
        suffix = cadence_path.suffix or ".sec"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(iaga_text)
            tmp_path = Path(tmp.name)

        return MagnetometerReadResult(
            stream=read(str(tmp_path)),
            source_path=path,
            cadence_path=cadence_path,
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


@enforce_types(
    site_code=str,
    now=datetime,
    data_root=(str, Path),
    lookback_days=int,
)
def get_last_measurement(
    site_code: str,
    now: datetime,
    data_root: Path,
    lookback_days: int,
) -> datetime | None:
    """
    Find the latest valid measurement time for a site.

    This is the main file-reading function used by the monitor. It searches
    recent files in preferred order, loads each file, and returns the newest
    timestamp with valid x/y/z data.

    Parameters
    ----------
    site_code:
        Magnetometer site code.
    now:
        Current datetime.
    data_root:
        Root data directory.
    lookback_days:
        Number of days to search backwards.

    Returns
    -------
    datetime.datetime or None
        Latest valid measurement time, or ``None`` if no valid data are found
        in the search window.
    """

    for path in candidate_files(data_root, site_code, now, lookback_days):
        stream = read_magnetometer_file(path)
        last_seen = latest_valid_time(stream)

        if last_seen is not None:
            return last_seen

    return None


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
            f"<td>{escape(str(event.minutes_since_last))}</td>"
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
        "site_count": len(events),
        "site_label": "site" if len(events) == 1 else "sites",
        "site_verb": "is" if len(events) == 1 else "are",
    }

    email_cfg = load_email_config(cfg.email_config_path)
    recipients = load_recipients(cfg.recipients_path)

    template_text = template_path.read_text(encoding="utf-8")
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


@enforce_types(site=SiteConfig, cfg=MonitorConfig, state=dict, now=datetime)
def check_site(
    site: SiteConfig,
    cfg: MonitorConfig,
    state: dict,
    now: datetime,
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
            "last_seen_at": None,
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

    last_seen = get_last_measurement(
        site_code=site.code,
        now=now,
        data_root=data_root_for_site(site, cfg.data_root),
        lookback_days=cfg.lookback_days,
    )

    if last_seen is None:
        age = None
        is_offline = True
        minutes_since_last = "unknown"
        last_seen_text = "No valid data found"
    else:
        age = now - last_seen
        is_offline = age > cfg.max_inactivity
        minutes_since_last = round(age.total_seconds() / 60, 1)
        last_seen_text = last_seen.isoformat()
        site_state["last_seen_at"] = last_seen.isoformat()

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
                minutes_since_last=minutes_since_last,
            ),
        )

    elif not is_offline and was_offline:
        site_state["offline"] = False
        site_state["status"] = "online"
        site_state["last_restored_sent_at"] = now.isoformat()
        return (
            "restored",
            SiteStatusEvent(
                site_code=site.code,
                site_name=site.name,
                last_seen_at=last_seen_text,
                minutes_since_last=minutes_since_last,
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
    state = load_state(cfg.state_path)
    outage_events: list[SiteStatusEvent] = []
    restored_events: list[SiteStatusEvent] = []

    for site in sites:
        event = check_site(site, cfg, state, now)

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
            else f"Magnetometer warning: {site_count} sites are not sending data"
        )
        send_batched_status_email(
            subject=subject,
            template_path=cfg.outage_email_template,
            cfg=cfg,
            values={
                "threshold_minutes": round(cfg.max_inactivity.total_seconds() / 60, 1),
                "checked_at": now.isoformat(),
            },
            events=outage_events,
        )

    if restored_events:
        site_count = len(restored_events)
        subject = (
            f"Magnetometer restored: {restored_events[0].site_code} is sending data again"
            if site_count == 1
            else f"Magnetometer restored: {site_count} sites are sending data again"
        )
        send_batched_status_email(
            subject=subject,
            template_path=cfg.restored_email_template,
            cfg=cfg,
            values={
                "checked_at": now.isoformat(),
            },
            events=restored_events,
        )

    save_state(cfg.state_path, state)

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

if __name__ == "__main__":
    sites = [
        SiteConfig(code="dun"),
        SiteConfig(code="val"),
        SiteConfig(code='bir'),
        SiteConfig(code="arm", permanently_off=True),
    ]

    cfg = MonitorConfig(
        data_root=Path("/home/simon/Documents/magnetometer_archive/"),
        state_path=Path("./magnetometer_monitor_state.json"),
        email_config_path=Path("/home/simon/gits/MAGIE/live_scripts/alert_config/email_config.toml"),
        recipients_path=Path("/home/simon/gits/MAGIE/live_scripts/alert_config/recipients.txt"),
        outage_email_template=Path("/home/simon/gits/MAGIE/src/magie/assets/email_templates/magnetometer_outage.html"),
        restored_email_template=Path("/home/simon/gits/MAGIE/src/magie/assets/email_templates/magnetometer_restored.html"),
        max_inactivity=timedelta(hours=1),
        lookback_days=7,
    )

    run_monitor(sites, cfg)
