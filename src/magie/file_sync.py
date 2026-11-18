"""Incremental, one-source-to-one-destination file processing.

Callers supply discovery, destination mapping and processing functions. This
module shares metadata checks, SQLite state, direct writes and retry behaviour
across archive conversions and other operational jobs. All discovered paths
are mapped on every run; only changed or unrecorded pairs are processed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import sqlite3
import time

from magie.utils import enforce_types


__all__ = ["SyncResult", "sync_files"]

PathInput = str | os.PathLike[str]
SourceFinder = Callable[[], Iterable[PathInput]]
DestinationMapper = Callable[[Path], PathInput]
FileProcessor = Callable[[Path, Path], object]
_PATH_TYPES = (str, os.PathLike)


@dataclass
class SyncResult:
    """
    Counts and elapsed time from an incremental file-processing run.

    Attributes
    ----------
    found : int
        Number of unique source/destination pairs discovered.
    updated : int
        Successfully processed files, or pending updates in a dry run.
    skipped : int
        Pairs whose source and destination metadata match saved state.
    failed : int
        Files whose metadata check or processing failed. These retry next run.
    elapsed : float
        Total elapsed time in seconds, including discovery and mapping.
    """

    found: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed: float = 0.0


@enforce_types(path=_PATH_TYPES)
def _normalise_path(path: PathInput) -> Path:
    """Validate a path value and return an absolute, resolved Path."""
    return Path(path).resolve()


@enforce_types(path=Path)
def _fingerprint(path: Path) -> tuple[int, int]:
    """Return file size and nanosecond modification time without reading data."""
    info = path.stat()
    return info.st_size, info.st_mtime_ns


@contextmanager
@enforce_types(path=Path, dry_run=bool)
def _state(path: Path, dry_run: bool) -> Iterator[sqlite3.Connection]:
    """Open locked persistent state, or an in-memory copy for a dry run."""
    lock = None
    db = None
    try:
        if dry_run:
            # Copy saved records without creating or updating files on disk.
            db = sqlite3.connect(":memory:")
            if path.is_file():
                with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as saved:
                    saved.backup(db)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = path.with_suffix(path.suffix + ".lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another sync is using {path}") from exc
            db = sqlite3.connect(path)
        db.execute(
            "CREATE TABLE IF NOT EXISTS file_sync_records ("
            "job TEXT, source TEXT, destination TEXT, size INTEGER, mtime INTEGER, "
            "output_size INTEGER, output_mtime INTEGER, "
            "PRIMARY KEY (job, source, destination))"
        )
        yield db
    finally:
        if db is not None:
            db.close()
        if lock is not None:
            lock.close()


@enforce_types(process=Callable, source=Path, destination=Path)
def _process_file(
    process: FileProcessor,
    source: Path,
    destination: Path,
) -> None:
    """Write directly to the mapped destination and validate completion."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    process(source, destination)
    if not destination.is_file():
        raise RuntimeError("Processor did not create its destination file")


@enforce_types(source_files=Callable, destination_for=Callable, state_path=Path)
def _prepare_pairs(
    source_files: SourceFinder,
    destination_for: DestinationMapper,
    state_path: Path,
) -> list[tuple[Path, Path]]:
    """Validate all mappings before opening state or writing any outputs."""
    discovered = source_files()
    if isinstance(discovered, (str, bytes, os.PathLike)) or not isinstance(discovered, Iterable):
        raise TypeError("source_files() must return an iterable of file paths")
    pairs: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    sources: set[Path] = set()
    for value in discovered:
        source = _normalise_path(value)
        if source in sources:
            continue
        destination = _normalise_path(destination_for(source))
        if destination in destinations:
            raise ValueError(f"Multiple sources map to {destination}")
        sources.add(source)
        destinations.add(destination)
        pairs.append((source, destination))
    if sources & destinations:
        raise ValueError("Destinations must not overwrite any source files")
    protected = {state_path, state_path.with_suffix(state_path.suffix + ".lock")}
    if (sources | destinations) & protected:
        raise ValueError("Source/destination paths must not use the sync state or lock file")
    return pairs


@enforce_types(
    source_files=Callable,
    destination_for=Callable,
    process=Callable,
    state_path=_PATH_TYPES,
    job=str,
    dry_run=bool,
    log=(Callable, type(None)),
)
def sync_files(
    source_files: SourceFinder,
    destination_for: DestinationMapper,
    process: FileProcessor,
    *,
    state_path: PathInput,
    job: str = "default",
    dry_run: bool = False,
    log: Callable[[str], object] | None = print,
) -> SyncResult:
    """
    Discover, map and incrementally process one output file per source.

    Parameters
    ----------
    source_files : callable
        Called without arguments each run. Return an iterable of source paths
        as strings or ``os.PathLike`` objects. Duplicate sources are ignored.
    destination_for : callable
        Called with each absolute source ``pathlib.Path``. Return its final
        destination path as a string or ``os.PathLike`` object. All sources
        are mapped every run, including those subsequently skipped.
    process : callable
        Called as ``process(source, destination)`` with ``pathlib.Path``
        arguments. Write directly to the final destination file or raise an
        exception. Its parent already exists. No temporary output is created.
        The return value is ignored. A failure can leave a partial destination;
        the pair remains unrecorded so it is retried on the next run.
    state_path : str or os.PathLike
        SQLite state file. Created with its parent directories on normal runs.
        Use the same state file for writers targeting the same outputs.
    job : str, optional
        Record namespace, default ``"default"``. Change this value to rebuild
        outputs after changing processor code or settings.
    dry_run : bool, optional
        Report pending work without calling ``process`` or writing state or
        outputs. Defaults to false. Discovery, mapping and logging still run.
    log : callable or None, optional
        Receives progress and per-file error messages as strings. Defaults to
        ``print``. Use ``None`` to suppress messages. Logger exceptions propagate.

    Returns
    -------
    SyncResult
        Discovered, updated, skipped and failed counts, plus elapsed seconds.
        In a dry run, ``updated`` counts pending updates. An empty discovery
        returns zero counts; the caller decides whether that is an error.

    Raises
    ------
    TypeError
        An argument, discovery result or callback path has an invalid type.
    ValueError
        Mappings collide, overwrite a source, or use the state or lock path.
    RuntimeError
        Another normal run holds the state file's lock.
    OSError or sqlite3.Error
        State cannot be opened or initialized. Per-file metadata and processing
        errors are instead logged and counted in ``failed``.

    Notes
    -----
    Source and destination sizes and nanosecond modification times are checked
    against saved state. Matching pairs skip processing without reading data.
    Existing outputs without a record are processed once. Changes preserving
    both size and modification time cannot be detected.

    Live sources may change during processing without failing the run. Records
    keep the source metadata captured before processing and the destination
    metadata after completion. Changes to that source metadata are therefore
    picked up on the next run. The processor must still raise if it cannot
    produce a usable output from the live input.

    Discovery and mapping errors propagate before state or outputs are written.
    Processing writes directly to the mapped destination, so readers may see
    incomplete data while it is being written. Existing outputs are not backed
    up or restored on failure. The previous success record is removed before
    writing; only successful processing creates a new record. Failed or
    interrupted writes therefore retry on the next run, even if their metadata
    happens to match the old record. Empty output files are allowed; validate
    formats in your processor. Removed sources do not delete outputs. Locking
    uses ``fcntl`` on POSIX systems.

    Examples
    --------
    >>> from pathlib import Path
    >>> result = sync_files(  # doctest: +SKIP
    ...     lambda: Path("inputs").glob("*.txt"),
    ...     lambda source: Path("outputs") / source.name,
    ...     lambda source, output: output.write_text(source.read_text().upper()),
    ...     state_path="outputs/.sync.sqlite3",
    ... )
    """
    started = time.monotonic()
    result = SyncResult()
    path = _normalise_path(state_path)
    pairs = _prepare_pairs(source_files, destination_for, path)
    result.found = len(pairs)
    with _state(path, dry_run) as db:
        for source, destination in pairs:
            message: str | None = None
            try:
                # Keep the starting metadata so live updates during processing
                # remain visible as a change on the next run.
                before = _fingerprint(source)
                previous = db.execute(
                    "SELECT size, mtime, output_size, output_mtime FROM file_sync_records "
                    "WHERE job=? AND source=? AND destination=?",
                    (job, str(source), str(destination)),
                ).fetchone()
                if (previous and tuple(previous[:2]) == before
                        and destination.is_file()
                        and tuple(previous[2:]) == _fingerprint(destination)):
                    result.skipped += 1
                    continue
                if not dry_run:
                    # Invalidate old success before writing, so even an interrupted
                    # overwrite with unchanged metadata must retry on the next run.
                    db.execute(
                        "DELETE FROM file_sync_records WHERE job=? AND source=? AND destination=?",
                        (job, str(source), str(destination)),
                    )
                    db.commit()
                    _process_file(process, source, destination)
                    db.execute(
                        "INSERT OR REPLACE INTO file_sync_records VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (job, str(source), str(destination), *before, *_fingerprint(destination)),
                    )
                    # Preserve successes even if a later file fails or the run stops.
                    db.commit()
                result.updated += 1
                action = "Would update" if dry_run else "Updated"
                message = f"{action} {source} -> {destination}"
            except Exception as exc:
                result.failed += 1
                message = f"Failed {source}: {exc}"
            if log is not None and message is not None:
                log(message)
    result.elapsed = time.monotonic() - started
    return result
