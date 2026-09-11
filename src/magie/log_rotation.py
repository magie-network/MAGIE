"""
Log rotation utilities for MagIE operational scripts.

The rotation workflow is intentionally small and filesystem based:

1. Collect one or more plain-text log files.
2. Store their current contents in a zip archive named for the rotation period.
3. Truncate the original log files so the live scripts can keep appending to
   the same paths.

Archive names include the configured frequency and the first date in the
covered period. For example, a weekly rotation run on Friday 2026-09-11 creates
``logs_weekly_2026-09-07.zip`` because that week starts on Monday 2026-09-07.

Old zipped archives can be deleted separately with ``delete_old_zipped_logs``.
That cleanup uses each archive's modification time, which makes it independent
of the naming convention and suitable for cron-style maintenance jobs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from magie.utils import enforce_types


SUPPORTED_FREQUENCIES = {"daily", "weekly", "monthly"}


@enforce_types(now=(datetime, type(None)))
def _normalise_now(now: datetime | None) -> datetime:
    """
    Return an aware timestamp for naming and age comparisons.

    Naive datetimes are treated as UTC. This matches the rest of the operational
    code, where archive days and live monitoring decisions are normalised to UTC
    unless a caller explicitly provides timezone-aware values.
    """
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now


@enforce_types(now=(datetime, type(None)), frequency=str)
def period_start(now: datetime | None = None, frequency: str = "daily") -> datetime:
    """
    Return the start timestamp for a log-rotation period.

    Parameters
    ----------
    now : datetime.datetime or None, optional
        Reference time used to choose the period. Defaults to the current UTC
        time. Naive datetimes are treated as UTC.
    frequency : str, optional
        Rotation period. Supported values are ``"daily"``, ``"weekly"``, and
        ``"monthly"``.

    Returns
    -------
    datetime.datetime
        Start of the selected period. Weekly periods start on Monday. Monthly
        periods start on the first day of the month. The returned timestamp
        keeps the timezone of ``now``.

    Raises
    ------
    ValueError
        If ``frequency`` is not supported.
    """
    now = _normalise_now(now)
    frequency = frequency.lower()

    if frequency == "daily":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if frequency == "weekly":
        start = now - timedelta(days=now.weekday())
        return start.replace(hour=0, minute=0, second=0, microsecond=0)
    if frequency == "monthly":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    supported = ", ".join(sorted(SUPPORTED_FREQUENCIES))
    raise ValueError(f"Unsupported frequency {frequency!r}. Expected one of: {supported}.")


@enforce_types(frequency=str, now=(datetime, type(None)), archive_prefix=str)
def rotation_archive_name(
    frequency: str,
    now: datetime | None = None,
    archive_prefix: str = "logs",
) -> str:
    """
    Build a period-stamped zip filename for the requested rotation frequency.

    Parameters
    ----------
    frequency : str
        Rotation period. Supported values are ``"daily"``, ``"weekly"``, and
        ``"monthly"``.
    now : datetime.datetime or None, optional
        Reference time used to choose the period. Defaults to the current UTC
        time. Naive datetimes are treated as UTC.
    archive_prefix : str, optional
        Prefix used before the frequency and period date.

    Returns
    -------
    str
        Zip filename such as ``logs_daily_2026-09-11.zip``,
        ``logs_weekly_2026-09-07.zip``, or ``logs_monthly_2026-09-01.zip``.

    Raises
    ------
    ValueError
        If ``frequency`` is not supported.
    """
    frequency = frequency.lower()
    start = period_start(now=now, frequency=frequency)
    return f"{archive_prefix}_{frequency}_{start:%Y-%m-%d}.zip"


@enforce_types(
    log_paths=list,
    frequency=str,
    archive_dir=(str, Path, type(None)),
    now=(datetime, type(None)),
    archive_prefix=str,
    missing_ok=bool,
    overwrite=bool,
)
def rotate_logs(
    log_paths: list[str | Path],
    frequency: str,
    archive_dir: str | Path | None = None,
    now: datetime | None = None,
    archive_prefix: str = "logs",
    missing_ok: bool = True,
    overwrite: bool = False,
) -> Path | None:
    """
    Zip the requested log files into a period-named archive, then truncate them.

    The function writes the zip archive before modifying any source log file.
    If an archive for the same period already exists and ``overwrite`` is false,
    the function raises before truncating the logs. This protects against data
    loss when a scheduled rotation job is accidentally run more than once in the
    same period.

    Parameters
    ----------
    log_paths : list[str | pathlib.Path]
        Log files to rotate. Files are stored in the zip archive using their
        base filenames, not their full absolute paths.
    frequency : str
        One of ``"daily"``, ``"weekly"``, or ``"monthly"``.
    archive_dir : str or pathlib.Path or None, optional
        Destination directory for zip archives. Defaults to the parent directory
        of the first log file.
    now : datetime.datetime or None, optional
        Reference time used for the archive name. Naive datetimes are treated as
        UTC.
    archive_prefix : str, optional
        Prefix used in the archive filename.
    missing_ok : bool, optional
        Skip missing log files when true. Raise ``FileNotFoundError`` when false.
    overwrite : bool, optional
        Replace an existing archive for the same period. When false, an existing
        archive raises before any log is truncated.

    Returns
    -------
    pathlib.Path or None
        The created zip archive, or ``None`` when there are no existing log
        files to rotate.

    Raises
    ------
    FileExistsError
        If the period archive already exists and ``overwrite`` is false.
    FileNotFoundError
        If a requested log file is missing and ``missing_ok`` is false.
    ValueError
        If a requested log path exists but is not a file, or if ``frequency`` is
        not supported.
    """
    if not log_paths:
        return None

    # Resolve the set of actual files first so errors happen before creating or
    # truncating anything.
    logs: list[Path] = []
    for log_path in log_paths:
        path = Path(log_path)
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Log path is not a file: {path}")
            logs.append(path)
        elif not missing_ok:
            raise FileNotFoundError(path)

    if not logs:
        return None

    if archive_dir is None:
        archive_root = logs[0].parent
    else:
        archive_root = Path(archive_dir)
    archive_root.mkdir(parents=True, exist_ok=True)

    archive_path = archive_root / rotation_archive_name(
        frequency=frequency,
        now=now,
        archive_prefix=archive_prefix,
    )

    if archive_path.exists() and not overwrite:
        raise FileExistsError(
            f"Archive already exists for this period: {archive_path}. "
            "Pass overwrite=True to replace it."
        )

    # The archive is complete before the source logs are cleared.
    with ZipFile(archive_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in logs:
            archive.write(path, arcname=path.name)

    for path in logs:
        path.write_text("", encoding="utf-8")

    return archive_path


@enforce_types(
    archive_dir=(str, Path),
    age=timedelta,
    now=(datetime, type(None)),
    pattern=str,
)
def delete_old_zipped_logs(
    archive_dir: str | Path,
    age: timedelta,
    now: datetime | None = None,
    pattern: str = "*.zip",
) -> list[Path]:
    """
    Delete zipped logs older than ``now - age`` based on file modification time.

    This is kept separate from ``rotate_logs`` so a site can rotate frequently
    but clean up archives on a different schedule. Only files ending in
    ``.zip`` are removed, even when ``pattern`` matches other files.

    Parameters
    ----------
    archive_dir : str or pathlib.Path
        Directory containing zipped log archives.
    age : datetime.timedelta
        Retention age. Archives with modification times earlier than
        ``now - age`` are deleted.
    now : datetime.datetime or None, optional
        Reference time for age calculation. Naive datetimes are treated as UTC.
    pattern : str, optional
        Glob pattern for candidate zip files. Defaults to ``"*.zip"``.

    Returns
    -------
    list[pathlib.Path]
        Deleted archive paths.

    Raises
    ------
    ValueError
        If ``age`` is negative, or if ``archive_dir`` exists but is not a
        directory.
    """
    if age < timedelta(0):
        raise ValueError("age must not be negative")

    archive_root = Path(archive_dir)
    if not archive_root.exists():
        return []
    if not archive_root.is_dir():
        raise ValueError(f"Archive path is not a directory: {archive_root}")

    cutoff = _normalise_now(now) - age
    cutoff_timestamp = cutoff.timestamp()
    deleted: list[Path] = []

    for archive_path in archive_root.glob(pattern):
        if not archive_path.is_file() or archive_path.suffix != ".zip":
            continue
        if archive_path.stat().st_mtime < cutoff_timestamp:
            archive_path.unlink()
            deleted.append(archive_path)

    return deleted
