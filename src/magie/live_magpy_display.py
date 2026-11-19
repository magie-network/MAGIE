"""Local live LEMI dashboard backed by MagPy and an optional SSH copy.

Run with ``python -m magie.live_magpy_display --help``. The HTTP server is intended
for a trusted local workstation, not for exposure to the public internet.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import math
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np
import pandas as pd
from matplotlib.dates import num2date

from magie.utils import enforce_types

__all__ = [
    "SampleStream", "DisplaySample", "KInterval", "make_samples", "corrected_copy",
    "Monitor", "calculate_live_k", "run_k_monitor", "main",
]

ASSETS = Path(__file__).parent / "assets" / "live_dashboard"

_PATH_TYPES = (str, os.PathLike)
_NUMBER_TYPES = (int, float, np.integer, np.floating)


@runtime_checkable
class SampleStream(Protocol):
    """Minimal MagPy column access required by :func:`make_samples`."""

    def _get_column(self, key: str) -> Sequence[object] | np.ndarray:
        """Return the values in a MagPy column."""
        ...


class DisplaySample(TypedDict):
    """One UTC observation, with magnetic components in nT and D in degrees."""

    time: str
    x: float
    y: float
    z: float
    h: float
    d: float | None
    temperature: float | None
    temperature2: float | None
    latitude: float | None
    longitude: float | None


class KInterval(TypedDict):
    """One UTC three-hour FMI interval and its sample coverage fraction."""

    start: str
    end: str
    value: int | None
    in_progress: bool
    coverage: float


@enforce_types(stream=SampleStream, scale=_NUMBER_TYPES, temperature_key=str, limit=int)


def make_samples(stream: SampleStream, scale: float, temperature_key: str, limit: int) -> list[DisplaySample]:
    """
    Convert MagPy columns to timestamp-sorted dashboard observations.

    Parameters
    ----------
    stream : SampleStream
        MagPy stream providing ``time``, ``x``, ``y``, ``z``, the selected
        temperature column, ``t2``, ``var3`` and ``var4`` through
        ``_get_column``. Numeric MagPy times are Matplotlib date numbers;
        datetime-like values are interpreted as UTC.
    scale : int or float
        Positive multiplier applied to X, Y and Z; results are in nT.
    temperature_key : str
        MagPy column for the first temperature in degrees Celsius. The second
        temperature is ``t2``; ``var3`` and ``var4`` are latitude and longitude
        in degrees for the LEMI files supported by this display.
    limit : int
        Maximum number of recent samples to return; must be positive.

    Returns
    -------
    list of DisplaySample
        Observations sorted by UTC ISO timestamp. Invalid timestamps and XYZ
        values are skipped; invalid optional values are represented by None.
        H is hypot(X, Y) in nT and D is atan2(Y, X) in degrees; D is None
        when H is zero. Geographic interpretation requires geographic axes.

    Raises
    ------
    TypeError
        An argument has an unsupported type.
    ValueError
        The scale or limit is not positive, or the temperature key is empty.

    Notes
    -----
    The input stream is not modified. This function assumes the file's XYZ
    units are nT after any header correction; it does not rotate the sensor.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not temperature_key.strip():
        raise ValueError("temperature_key must not be empty")
    times = stream._get_column("time")
    keys = ("x", "y", "z", temperature_key, "t2", "var3", "var4")
    columns = {key: stream._get_column(key) for key in set(keys)}
    rows: list[DisplaySample] = []
    for i in range(len(times)):
        try:
            stamp = times[i]
            if isinstance(stamp, (float, int, np.number)):
                dt = num2date(float(stamp), tz=timezone.utc)
            else:
                dt = np.datetime64(stamp, "us").astype(datetime).replace(tzinfo=timezone.utc)
            vals = [float(columns[k][i]) * scale for k in ("x", "y", "z")]
            if not all(math.isfinite(v) for v in vals):
                continue
            x, y, z = vals
            def optional(key: str) -> float | None:
                """Keep absent or non-finite auxiliary measurements as None."""
                try:
                    value = float(columns[key][i])
                    return value if math.isfinite(value) else None
                except (IndexError, TypeError, ValueError):
                    return None
            h = math.hypot(x, y)
            rows.append(dict(time=dt.isoformat(), x=x, y=y, z=z, h=h,
                             d=math.degrees(math.atan2(y, x)) if h else None,
                             temperature=optional(temperature_key),
                             temperature2=optional("t2"), latitude=optional("var3"),
                             longitude=optional("var4")))
        except (IndexError, TypeError, ValueError, OverflowError):
            continue
    rows.sort(key=lambda row: row["time"])
    return rows[-limit:]


@enforce_types(source=_PATH_TYPES, destination=_PATH_TYPES, correct_xyz=bool)
def corrected_copy(source: str | os.PathLike[str], destination: str | os.PathLike[str],
                   correct_xyz: bool = True) -> None:
    """
    Copy a MagPyBin v1.1 file, optionally correcting XYZ header divisors.

    Parameters
    ----------
    source, destination : str or os.PathLike
        Distinct source and output files. The output is overwritten only after
        the source header is validated; its parent directory must already exist.
    correct_xyz : bool, optional
        Set the X, Y and Z header divisors to one when true (the default).
        Other divisors and the binary records are preserved. When false all
        divisors retain their original values.

    Raises
    ------
    TypeError
        An argument has an unsupported type.
    ValueError
        The source and destination resolve to the same file, or the source is
        not a supported MagPyBin v1.1 header with XYZ keys.
    UnicodeDecodeError
        The header is not ASCII.
    OSError
        A file cannot be opened or copied. A copy failure can leave a partial
        destination. The source file is never intentionally modified.

    Notes
    -----
    The dashboard uses a temporary output because it must correct the header
    before MagPy reads a live file; it never rewrites the acquisition file.
    """
    source, destination = Path(source), Path(destination)
    if (source.resolve() == destination.resolve()
            or (destination.exists() and source.exists() and source.samefile(destination))):
        raise ValueError("source and destination must be different files")
    with source.open("rb") as src:
        header = src.readline()
        parts = header.decode("ascii").split()
        if len(parts) != 9 or parts[:2] != ["#", "MagPyBin"]:
            raise ValueError("Expected a MagPyBin v1.1 header with XYZ keys")
        keys = parts[3].strip("[]").split(",")
        divisors = parts[6].strip("[]").split(",")
        if len(keys) != len(divisors) or not all(k in keys for k in ("x", "y", "z")):
            raise ValueError("Invalid MagPyBin XYZ divisor list")
        if correct_xyz:
            for key in ("x", "y", "z"):
                divisors[keys.index(key)] = "1"
        parts[6] = "[" + ",".join(divisors) + "]"
        with destination.open("wb") as dst:
            dst.write((" ".join(parts) + "\n").encode("ascii"))
            shutil.copyfileobj(src, dst)


class Monitor:
    """
    Poll a local or SSH MagPyBin source and retain recent dashboard state.

    Parameters
    ----------
    source : str or os.PathLike
        Local file/directory or ``host:/absolute/path`` when ``remote`` is true.
        A ``{date}`` token expands to a UTC ``YYYY-MM-DD`` date.
    remote : bool
        Copy the selected remote file through non-interactive ``scp``.
    interval : int or float
        Positive target interval between polling starts in seconds. Polls do
        not overlap; a slow copy or read delays the next poll.
    scale : int or float
        Positive additional multiplier applied to XYZ after MagPy reads them.
    temperature_key : str
        MagPy column used for the first temperature.
    limit : int
        Positive maximum number of observations retained for the display.
    filename : str or None, optional
        Filename inside a source directory, optionally containing ``{date}``.
        Local directories support a glob; SSH expects an exact filename. None
        uses the local directory's lexicographically last ``*.bin`` file, or
        treats the remote source as an explicit file.
    station : str, optional
        Label shown in the dashboard, default ``"LEMI magnetometer"``.
    correct_xyz : bool, optional
        Correct XYZ header divisors to one in a temporary copy, default true.
    k_site : str or None, optional
        Station metadata alias used by the independent K worker. None disables
        K calculation until an alias is provided.

    Attributes
    ----------
    state : dict
        Most recent observations, status, source, refresh time and K results.
        Use ``lock`` when accessing or updating state across threads. Samples
        are UTC; XYZ/H are nT, D and position are degrees, temperatures are °C.
    lock : threading.Lock
        Lock protecting ``state``; file-history caches are managed by the
        polling and K workers.

    Raises
    ------
    TypeError
        An argument has an unsupported type.
    ValueError
        Required text, remote source or filename is invalid, or the interval,
        scale or limit is not positive.

    Notes
    -----
    Construction has no filesystem or network side effects. ``update`` reads
    one snapshot, and ``run`` repeats indefinitely. Dated history is cached
    until it falls outside the rolling window; absent historical files retry
    after 60 seconds. Errors are reported in state without replacing a prior
    successful observation with an empty list. Source files are not modified.
    """

    @enforce_types(source=_PATH_TYPES, remote=bool, interval=_NUMBER_TYPES,
                   scale=_NUMBER_TYPES, temperature_key=str, limit=int,
                   filename=(str, type(None)), station=str, correct_xyz=bool,
                   k_site=(str, type(None)))
    def __init__(self, source: str | os.PathLike[str], remote: bool, interval: float, scale: float,
                 temperature_key: str, limit: int, filename: str | None = None,
                 station: str = "LEMI magnetometer", correct_xyz: bool = True,
                 k_site: str | None = None) -> None:
        source = os.fsdecode(source)
        if not source.strip() or not temperature_key.strip() or not station.strip():
            raise ValueError("source, temperature_key and station must not be empty")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if filename and ("/" in filename or filename in (".", "..")):
            raise ValueError("filename must be a filename or local glob, not a path")
        if remote:
            if source.startswith("-") or ":" not in source or not source.split(":", 1)[1].startswith("/"):
                raise ValueError("remote source must be host:/absolute/path")
            if source.endswith("/") and not filename:
                raise ValueError("a remote directory requires filename")
            if filename and any(char in filename for char in "*?[]"):
                raise ValueError("remote filename must be exact, not a glob")
        self.source, self.remote, self.interval = source, remote, interval
        self.scale, self.temperature_key, self.limit = scale, temperature_key, limit
        self.filename, self.correct_xyz = filename, correct_xyz
        self.k_site = k_site
        self.k_history: dict[str, list[DisplaySample]] = {}
        self.dated = "{date}" in source or (filename is not None and "{date}" in filename)
        self.history: dict[str, list[DisplaySample]] = {}
        self.history_retry: dict[str, float] = {}
        self.today_samples: list[DisplaySample] = []
        self.today_date: str | None = None
        self.temperature_labels = ["Temperature 1", "Temperature 2"]
        self.lock = threading.Lock()
        self.state = {"samples": [], "error": "Waiting for first update", "updated": None,
                      "scale": scale, "source": source, "interval": interval,
                      "station": station, "correct_xyz": correct_xyz, "history_warnings": [],
                      "k": {"values": [], "status": "Waiting for K calculation", "updated": None}}

    @enforce_types(date=str)
    def _read_day(self, date: str) -> list[DisplaySample]:
        """Read a dated source snapshot, correcting only its temporary copy."""
        datetime.strptime(date, "%Y-%m-%d")
        from magpy.stream import read
        with tempfile.TemporaryDirectory(prefix="magie-lemi-") as tmp:
            source = self.source.replace("{date}", date)
            filename = self.filename.replace("{date}", date) if self.filename else None
            if self.remote:
                if filename:
                    source = source.rstrip("/") + "/" + filename
                path = Path(tmp) / "source.bin"
                subprocess.run(["scp", "-q", "-oBatchMode=yes", "-oConnectTimeout=10",
                                source, str(path)], check=True, timeout=60,
                               capture_output=True)
            else:
                path = Path(source)
                if path.is_dir():
                    files = sorted(p for p in path.glob(filename or "*.bin") if p.is_file())
                    if not files:
                        raise ValueError("No matching binary files in local directory")
                    path = files[-1]
            corrected = Path(tmp) / "corrected.bin"
            corrected_copy(path, corrected, self.correct_xyz)
            stream = read(str(corrected), datecheck=False)
            if date == self.today_date:
                labels = [stream.header.get(f"col-{key}", "") for key in (self.temperature_key, "t2")]
                # The physical meanings are inferred from these channel names;
                # the LEMI-018 documentation found does not define TE and TF.
                descriptions = {"TE": "Electronics (TE)", "TF": "Sensor (TF)"}
                self.temperature_labels = [descriptions.get(str(label).upper(), f"Temperature {i + 1}")
                                           for i, label in enumerate(labels)]
            samples = make_samples(stream, self.scale, self.temperature_key, self.limit)
        if not samples:
            raise ValueError("No valid XYZ samples found in source")
        return samples

    def update(self) -> None:
        """
        Refresh the latest data and the rolling history once.

        Notes
        -----
        File, transfer and reader failures are recorded in ``state["error"]``;
        missing historical dates are recorded in ``state["history_warnings"]``
        and retried after 60 seconds. A failed refresh retains previous samples
        when no samples are available. Network and temporary-file work
        occurs here, never at construction or import.
        """
        now = datetime.now(timezone.utc)
        today = now.date().isoformat()
        if self.today_date != today:
            if self.today_date and self.today_samples:
                self.history[self.today_date] = self.today_samples
            self.today_date, self.today_samples = today, []
        error = None
        try:
            self.today_samples = self._read_day(today)
        except Exception as exc:
            error = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                error = exc.stderr.decode(errors="replace").strip() or error

        warnings = []
        # A rolling 72-hour window can touch four UTC calendar dates.
        days = [(now.date() - timedelta(days=i)).isoformat() for i in range(1, 4)] if self.dated else []
        self.history = {day: rows for day, rows in self.history.items() if day in days}
        self.history_retry = {day: retry for day, retry in self.history_retry.items() if day in days}
        for day in days:
            if day not in self.history and time.monotonic() >= self.history_retry.get(day, 0):
                try:
                    self.history[day] = self._read_day(day)
                    self.history_retry.pop(day, None)
                except Exception:
                    # Retry absent or unreadable historical files without repeatedly
                    # copying or reconnecting on every live update.
                    self.history_retry[day] = time.monotonic() + 60
            if day not in self.history:
                warnings.append(f"{day}: historical file unavailable")
        samples = [row for day in reversed(days) for row in self.history.get(day, [])]
        samples.extend(self.today_samples)
        samples.sort(key=lambda row: row["time"])
        if self.dated:
            cutoff = (now - timedelta(days=3)).isoformat()
            samples = [row for row in samples if row["time"] >= cutoff]
        samples = samples[-self.limit:]
        with self.lock:
            # Keep the previous state when the current snapshot has no samples.
            if samples:
                self.state["samples"] = samples
            self.state.update(error=error, history_warnings=warnings,
                              temperature_labels=self.temperature_labels)
            if error is None:
                self.state["updated"] = now.isoformat()

    def run(self) -> None:
        """Poll indefinitely at the configured target interval without overlap."""
        while True:
            started = time.monotonic()
            self.update()
            # Do not overlap copies/reads if they take longer than the interval.
            time.sleep(max(0, self.interval - (time.monotonic() - started)))


@enforce_types(samples=list, site=str, now=datetime)
def calculate_live_k(samples: list[DisplaySample], site: str, now: datetime) -> list[KInterval]:
    """
    Calculate provisional live FMI K from LEMI observations.

    Parameters
    ----------
    samples : list of dict
        ISO timestamp strings under ``time`` and XYZ values in nT under ``x``,
        ``y`` and ``z``. Naive timestamps are interpreted as UTC; aware times
        are converted to UTC. Native cadence is one second or one minute.
        Missing timestamps are placed on a regular grid as NaNs; duplicate
        timestamps keep the last observation. The input list is not modified.
    site : str
        MAGIE station alias with numeric K9 threshold and longitude metadata.
    now : datetime.datetime
        Timezone-aware acquisition cutoff; converted to UTC. Observations after
        this time are excluded and missing future values are padded with NaNs.

    Returns
    -------
    list of KInterval
        UTC three-hour intervals in the previous three days. ``value`` is an
        integer 0–9 or None, ``coverage`` is the observed fraction and
        ``in_progress`` identifies the interval containing the cutoff.

    Raises
    ------
    TypeError
        An argument or sample field has an unsupported type.
    ValueError
        Samples, cadence, timezone, coverage or station metadata are invalid,
        or there is not enough recent history for the live calculation.

    Notes
    -----
    This applies MagPy's live filter and FMI method with the station's K9 and
    longitude; it does not rotate axes or run experimental anomaly removal.
    At least 45 hours and 80% horizontal coverage are required. Individual
    intervals need 80% coverage, and the current interval needs 30 minutes of
    observations. FMI may still return None. All results are provisional.
    """
    if not site.strip():
        raise ValueError("site must not be empty")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    for sample in samples:
        if not isinstance(sample, dict):
            raise TypeError("each sample must be a dictionary")
        if not isinstance(sample.get("time"), str):
            raise TypeError("each sample time must be a UTC ISO timestamp string")
        for key in ("x", "y", "z"):
            if not isinstance(sample.get(key), _NUMBER_TYPES):
                raise TypeError(f"each sample {key} must be numeric")
    from magpy.core import activity
    from magpy.stream import DataStream, join_streams
    from magie.k_index_magpy import _empty_padding_stream
    from magie.utils import get_site_metadata

    metadata = get_site_metadata(site)
    if not metadata or not isinstance(metadata.get("k9_threshold"), (int, float)):
        raise ValueError(f"No numeric K9 threshold and location for {site!r}")
    if not samples:
        raise ValueError("No LEMI samples available")
    frame = pd.DataFrame(samples, columns=["time", "x", "y", "z"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    cutoff = pd.Timestamp(now).tz_convert("UTC").tz_localize(None).floor("s")
    frame = frame[frame.time <= cutoff].drop_duplicates("time", keep="last").set_index("time").sort_index()
    if frame.empty:
        raise ValueError("No observations before the live cutoff")
    deltas = frame.index.to_series().diff().dt.total_seconds().dropna()
    cadence = float(deltas.median()) if len(deltas) else 0
    if cadence not in (1, 60):
        raise ValueError("K calculation requires a regular 1-second or 1-minute cadence")
    step = int(cadence)
    start = cutoff.floor("D") - pd.Timedelta(days=4)
    frame = frame.loc[start:cutoff]
    if frame.empty or frame.index[-1] < cutoff - pd.Timedelta(minutes=2):
        raise ValueError("Latest LEMI sample is delayed; K is unavailable")
    if frame.index[0] > cutoff - pd.Timedelta(hours=45):
        raise ValueError("At least 45 hours of preceding data are needed for live FMI K")
    if any(frame.index.second % step) or any(frame.index.microsecond):
        raise ValueError("K input timestamps must align to their cadence")
    grid = pd.date_range(start, cutoff.floor(f"{step}s"), freq=f"{step}s")
    frame = frame.reindex(grid)
    for key in ("x", "y", "z"):
        frame[key] = pd.to_numeric(frame[key], errors="coerce").replace([99999, 999999, 99999.99], np.nan)
    valid = frame[["x", "y"]].notna().all(axis=1)
    if valid.loc[cutoff - pd.Timedelta(hours=45):cutoff].mean() < .8:
        raise ValueError("Less than 80% coverage in the 45-hour FMI history")
    arr = [np.asarray([]) for _ in DataStream().KEYLIST]
    arr[0] = grid.to_numpy(dtype="datetime64[us]").astype(object)
    for key in ("x", "y", "z"):
        arr[DataStream().KEYLIST.index(key)] = frame[key].to_numpy(dtype=float)
    stream = DataStream(header={"DataSamplingRate": cadence, "col-x": "X", "col-y": "Y", "col-z": "Z"},
                        ndarray=np.asarray(arr, dtype=object))
    # Match operational live_k: pad unobserved future with NaNs, then filter
    # before the FMI calculation. Never introduce future magnetic values.
    stream = join_streams(stream, _empty_padding_stream(cutoff + pd.Timedelta(seconds=step), site,
                                                        sampling_step_seconds=step))
    stream = stream.filter()
    result = activity.K_fmi(stream, K9_limit=metadata["k9_threshold"],
                            longitude=metadata["geodetic_longitude"], step_size=60)
    values: list[KInterval] = []
    for stamp, value in zip(result["time"], result["var1"]):
        middle = pd.Timestamp(stamp)
        if middle.tzinfo is not None:
            middle = middle.tz_convert("UTC").tz_localize(None)
        begin = middle.floor("3h")
        if begin > cutoff or begin < cutoff - pd.Timedelta(days=3):
            continue
        finish = begin + pd.Timedelta(hours=3)
        observed_end = min(finish, cutoff + pd.Timedelta(seconds=step))
        expected = max(1, int((observed_end - begin).total_seconds() / step))
        count = int(valid.loc[(valid.index >= begin) & (valid.index < observed_end)].sum())
        elapsed = (observed_end - begin).total_seconds()
        good = count / expected >= .8 and (finish <= cutoff or elapsed >= 1800)
        value = float(value)
        values.append({"start": begin.isoformat() + "+00:00", "end": finish.isoformat() + "+00:00",
                       "value": int(value) if good and np.isfinite(value) and 0 <= value <= 9 else None,
                       "in_progress": bool(finish > cutoff), "coverage": round(min(1, count / expected), 3)})
    return values


@enforce_types(monitor=Monitor)
def run_k_monitor(monitor: Monitor) -> None:
    """
    Refresh live FMI K in a worker, caching completed UTC days.

    Parameters
    ----------
    monitor : Monitor
        Source and shared state. This function runs indefinitely, refreshing
        approximately once per minute without delaying the main polling loop.

    Raises
    ------
    TypeError
        ``monitor`` is not a Monitor.

    Notes
    -----
    Results and recoverable errors are published to ``monitor.state["k"]``
    under its lock. Dated sources may require four previous UTC days of files;
    unreadable days are omitted and retried in later worker cycles.
    """
    while True:
        started = time.monotonic()
        now = datetime.now(timezone.utc)
        try:
            if not monitor.k_site:
                raise ValueError("Set --k-site to a station with K9 metadata to enable K")
            today = now.date().isoformat()
            with monitor.lock:
                samples = list(monitor.state["samples"])
            if monitor.dated:
                # K's operational live workflow needs four preceding UTC days;
                # keep these outside the dashboard's rolling three-day window.
                days = [(now.date() - timedelta(days=i)).isoformat() for i in range(1, 5)]
                monitor.k_history = {d: rows for d, rows in monitor.k_history.items() if d in days}
                known = dict(monitor.history)
                for day in days:
                    if day in known:
                        monitor.k_history[day] = known[day]
                    elif day not in monitor.k_history:
                        try:
                            monitor.k_history[day] = monitor._read_day(day)
                        except Exception:
                            continue
                samples = [row for day in reversed(days) for row in monitor.k_history.get(day, [])] + [
                    row for row in samples if row["time"][:10] == today]
            values = calculate_live_k(samples, monitor.k_site, now)
            if not values or not any(row["value"] is not None for row in values):
                raise ValueError("Not enough valid observations to calculate K")
            result = {"values": values, "status": None, "updated": now.isoformat(), "site": monitor.k_site}
        except Exception as exc:
            result = {"values": [], "status": str(exc), "updated": now.isoformat(), "site": monitor.k_site}
        with monitor.lock:
            monitor.state["k"] = result
        time.sleep(max(1, 60 - (time.monotonic() - started)))


@enforce_types()
def main() -> None:
    """
    Parse CLI options and serve the dashboard on localhost until interrupted.

    Notes
    -----
    The HTTP server is intended for a trusted local workstation. Invalid CLI
    arguments exit through argparse; address binding and other startup errors
    propagate. See ``python -m magie.live_magpy_display --help`` for options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="Local path, or SSH source with --remote (legacy syntax)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", help="Local file or directory")
    group.add_argument("--ssh", help="SSH source as host:/absolute/file or host:/absolute/directory/ with --filename")
    parser.add_argument("--remote", action="store_true", help="Treat the positional source as an SSH source")
    parser.add_argument("--filename", help="Filename for an SSH directory, or glob for a local directory; {date} means current UTC date")
    parser.add_argument("--station", default="LEMI magnetometer", help="Station name displayed on the page")
    parser.add_argument("--k-site", help="Station metadata alias for live FMI K (defaults to --station when recognised)")
    parser.add_argument("--preserve-xyz-divisors", action="store_true", help="Use the XYZ divisors recorded in the file header")
    parser.add_argument("--interval", type=float, default=1, help="Target refresh interval in seconds (default: 1)")
    parser.add_argument("--xyz-scale", type=float, default=1, help="Additional multiplier applied after reading (default: 1)")
    parser.add_argument("--temperature-key", default="t1", help="MagPy temperature column (default: t1)")
    parser.add_argument("--limit", type=int, default=259200, help="Maximum recent samples shown (default: 259200)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.source and (args.local or args.ssh):
        parser.error("use either a positional source or --local/--ssh")
    if args.remote and (not args.source or args.local or args.ssh):
        parser.error("--remote requires a positional source; use --ssh instead")
    source = args.ssh or args.local or args.source
    remote = bool(args.ssh or args.remote)
    if not source:
        parser.error("provide --local PATH or --ssh HOST:/PATH")
    if (not math.isfinite(args.interval) or args.interval <= 0 or args.limit <= 0
            or not math.isfinite(args.xyz_scale) or args.xyz_scale <= 0):
        parser.error("interval, limit and xyz-scale must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.filename and ("/" in args.filename or args.filename in (".", "..")):
        parser.error("--filename must be a filename or local glob, not a path")
    if remote:
        if source.startswith("-") or ":" not in source or not source.split(":", 1)[1].startswith("/"):
            parser.error("SSH source must be host:/absolute/path")
        if source.endswith("/") and not args.filename:
            parser.error("an SSH directory requires --filename")
        if args.filename and any(char in args.filename for char in "*?[]"):
            parser.error("--filename must be an exact filename for SSH sources")
    elif args.filename and not Path(source.replace("{date}", "2000-01-01")).is_dir():
        parser.error("--filename requires a local directory")
    from magie.utils import get_site_metadata
    k_site = args.k_site or (args.station if get_site_metadata(args.station) else None)
    if args.k_site and not get_site_metadata(args.k_site):
        parser.error("--k-site is not a recognised station alias")
    monitor = Monitor(source, remote, args.interval, args.xyz_scale, args.temperature_key,
                      args.limit, args.filename, args.station, not args.preserve_xyz_divisors, k_site)

    class Handler(BaseHTTPRequestHandler):
        """Handle requests for the local dashboard and latest readings."""

        def do_GET(self) -> None:
            """Serve only the dashboard, its logo and the current JSON state."""
            if self.path == "/api/data":
                with monitor.lock:
                    version = repr((monitor.state["updated"], monitor.state["error"], monitor.state["history_warnings"],
                                    monitor.state["k"]["updated"]))
                    etag = '"' + hashlib.sha256(version.encode()).hexdigest() + '"'
                    if self.headers.get("If-None-Match") == etag:
                        self.send_response(304)
                        self.send_header("ETag", etag)
                        self.end_headers()
                        return
                    body = json.dumps(monitor.state, allow_nan=False).encode()
                kind = "application/json"
            elif self.path in ("/", "/index.html"):
                body, kind = (ASSETS / "index.html").read_bytes(), "text/html; charset=utf-8"
            elif self.path == "/magie-logo-white.png":
                body, kind = (ASSETS / "magie-logo-white.png").read_bytes(), "image/png"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            if self.path == "/api/data":
                self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    threading.Thread(target=monitor.run, daemon=True).start()
    threading.Thread(target=run_k_monitor, args=(monitor,), daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"LEMI dashboard: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
