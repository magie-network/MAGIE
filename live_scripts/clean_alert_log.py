import json
from datetime import date, datetime, timedelta
from pathlib import Path


DEFAULT_LOG_PATH = Path(__file__).with_name("alert_log.json")


def load_log(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_entry_date(key: str) -> date:
    try:
        _, timestamp = key.rsplit("|", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid log key format: {key!r}") from exc

    try:
        return datetime.fromisoformat(timestamp).date()
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp in log key: {key!r}") from exc


def prune_log_entries(logs: dict[str, int], today: date) -> tuple[dict[str, int], list[str]]:
    cutoff_date = today - timedelta(days=2)
    kept_logs: dict[str, int] = {}
    invalid_keys: list[str] = []

    for key, value in logs.items():
        try:
            entry_date = extract_entry_date(key)
        except ValueError:
            invalid_keys.append(key)
            kept_logs[key] = value
            continue

        if entry_date > cutoff_date:
            kept_logs[key] = value

    return kept_logs, invalid_keys


def clean_alert_log(log_path: str | Path = DEFAULT_LOG_PATH, today: date | None = None) -> int:
    log_path = Path(log_path)
    today = today or date.today()
    logs = load_log(log_path)
    pruned_logs, invalid_keys = prune_log_entries(logs, today)

    log_path.write_text(json.dumps(pruned_logs, sort_keys=True), encoding="utf-8")

    removed_count = len(logs) - len(pruned_logs)

    return 0


if __name__ == "__main__":
    raise SystemExit(clean_alert_log())
