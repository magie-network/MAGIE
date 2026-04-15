from magie import plot_k, live_k
import pandas as pd
from magie.utils import enforce_types, get_site_metadata
from magie.email_utils import render_html_template, send_html_email, load_email_config, load_recipients, load_mastodon_config
import json
from pathlib import Path
import tempfile
import re
from datetime import date, datetime, timedelta


@enforce_types(dt=pd.Timestamp)
def format_date_as_string(dt: pd.Timestamp) -> str:
    """
    Format a timestamp as a human-readable date for alert copy.

    Parameters
    ----------
    dt : pandas.Timestamp
        Timestamp to render.

    Returns
    -------
    str
        Date string such as ``"12th of December 2025"``.
    """
    day = dt.day

    # Handle 11th, 12th, 13th correctly
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    return f"{day}{suffix} of {dt.strftime('%B %Y')}"


@enforce_types(kp=(int, float))
def classify_storm(kp: float) -> str:
    """
    Map a K index value to the corresponding qualitative storm label.

    Parameters
    ----------
    kp : float
        K index value in the inclusive range 0 to 9.

    Returns
    -------
    str
        Storm classification label.

    Raises
    ------
    ValueError
        If ``kp`` falls outside the supported K index range.
    """
    if not 0 <= kp <= 9:
        raise ValueError("Kp must be between 0 and 9")

    thresholds = [
        (1, "Quiet"),
        (3, "Unsettled"),
        (4, "Active"),
        (5, "Minor Storm"),
        (7, "Major Storm"),
        (9, "Severe Storm"),
    ]

    for max_kp, label in thresholds:
        if kp <= max_kp:
            return label


@enforce_types(path=(str, Path))
def load_log(path="alert_log.json") -> dict[str, int]:
    """
    Load the alert deduplication log from disk.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        JSON file storing previously sent alerts.

    Returns
    -------
    dict[str, int]
        Mapping of ``"site|timestamp"`` keys to sent K values. Returns an
        empty dictionary when the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@enforce_types(site=str, timestamp=(str, pd.Timestamp, datetime))
def make_log_key(site: str, timestamp: str) -> str:
    """
    Build the composite key used in the alert log.

    Parameters
    ----------
    site : str
        Station name or site label.
    timestamp : str or pandas.Timestamp or datetime.datetime
        Timestamp associated with the alert window.

    Returns
    -------
    str
        Combined ``"site|timestamp"`` key.
    """
    return f"{site}|{timestamp}"


@enforce_types(
    timestamp=(str, pd.Timestamp, datetime),
    site=str,
    path=(str, Path),
)
def check_logs(timestamp, site, path="alert_log.json"):
    """
    Check whether an alert for a site and timestamp has already been sent.

    Parameters
    ----------
    timestamp : str or pandas.Timestamp or datetime.datetime
        Timestamp identifying the alert interval.
    site : str
        Station name or site label.
    path : str or pathlib.Path, optional
        Alert log file to inspect.

    Returns
    -------
    str
        ``"no alert"`` when the alert already exists in the log, otherwise
        ``"new alert"``.
    """
    logs = load_log(path)
    if make_log_key(site, timestamp) in logs:
        return "no alert"
    return "new alert"


@enforce_types(
    timestamp=(str, pd.Timestamp, datetime),
    k_value=int,
    site=str,
    path=(str, Path),
)
def save_log(timestamp, k_value, site, path="alert_log.json"):
    """
    Persist a sent alert to the deduplication log.

    Parameters
    ----------
    timestamp : str or pandas.Timestamp or datetime.datetime
        Timestamp identifying the alert interval.
    k_value : int
        K index value that triggered the alert.
    site : str
        Station name or site label.
    path : str or pathlib.Path, optional
        Alert log file to update.
    """
    logs = load_log(path)
    logs[make_log_key(site, timestamp)] = k_value
    Path(path).write_text(json.dumps(logs), encoding="utf-8")


@enforce_types(template_text=str)
def extract_site_block(template_text: str) -> str:
    """
    Extract the repeatable site-specific HTML block from an email template.

    Parameters
    ----------
    template_text : str
        Full HTML template content containing ``SITE_BLOCK`` markers.

    Returns
    -------
    str
        Template fragment between the begin and end markers.

    Raises
    ------
    RuntimeError
        If the expected markers are missing.
    """
    m = re.search(
        r"<!--\s*BEGIN SITE_BLOCK\s*-->(.*?)<!--\s*END SITE_BLOCK\s*-->",
        template_text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Could not find SITE_BLOCK markers in template.")
    return m.group(1)


@enforce_types(template_text=str, rendered_blocks_html=str)
def replace_site_block(template_text: str, rendered_blocks_html: str) -> str:
    """
    Replace the marked site block in an email template with rendered content.

    Parameters
    ----------
    template_text : str
        Original HTML template content.
    rendered_blocks_html : str
        Final HTML to insert in place of the marked site block.

    Returns
    -------
    str
        Updated template content.
    """
    return re.sub(
        r"<!--\s*BEGIN SITE_BLOCK\s*-->(.*?)<!--\s*END SITE_BLOCK\s*-->",
        rendered_blocks_html,
        template_text,
        flags=re.S,
    )


@enforce_types(dt_utc=pd.Timestamp)
def build_archive_url(dt_utc: pd.Timestamp) -> str:
    """
    Build the public archive URL for the PNG products of a UTC day.

    Parameters
    ----------
    dt_utc : pandas.Timestamp
        Timestamp whose UTC calendar date should be used.

    Returns
    -------
    str
        Daily archive URL for PNG products.
    """
    d = dt_utc.date()
    return f"https://data.magie.ie/{d.year:04d}/{d.month:02d}/{d.day:02d}/png/"


@enforce_types(
    template=(str, Path),
    email_config=(str, Path),
    recipients=(str, Path),
    png_save_path=(str, Path),
    alert_log_path=(str, Path),
    mastodon_config=(str, Path, type(None)),
)
def validate_alert_paths(
    template: str | Path,
    email_config: str | Path,
    recipients: str | Path,
    png_save_path: str | Path,
    alert_log_path: str | Path,
    mastodon_config: str | Path | None = None,
) -> dict[str, Path]:
    """
    Validate alert input and output paths.

    Required input files must already exist. Output directories must already
    exist.

    Returns
    -------
    dict[str, pathlib.Path]
        Normalized paths used by ``alert``.
    """
    resolved_paths = {
        "template": Path(template),
        "email_config": Path(email_config),
        "recipients": Path(recipients),
        "png_save_path": Path(png_save_path),
        "alert_log_path": Path(alert_log_path),
    }
    if mastodon_config is not None:
        resolved_paths["mastodon_config"] = Path(mastodon_config)

    for name in ("template", "email_config", "recipients"):
        path = resolved_paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"{name} file not found: {path}")

    if mastodon_config is not None and not resolved_paths["mastodon_config"].is_file():
        raise FileNotFoundError(f"mastodon_config file not found: {resolved_paths['mastodon_config']}")

    png_dir = resolved_paths["png_save_path"]
    if not png_dir.exists():
        raise FileNotFoundError(f"png_save_path directory not found: {png_dir}")
    if not png_dir.is_dir():
        raise NotADirectoryError(f"png_save_path is not a directory: {png_dir}")

    alert_log_parent = resolved_paths["alert_log_path"].parent
    if not alert_log_parent.exists():
        raise FileNotFoundError(f"alert_log_path parent directory not found: {alert_log_parent}")
    if not alert_log_parent.is_dir():
        raise NotADirectoryError(f"alert_log_path parent is not a directory: {alert_log_parent}")

    return resolved_paths


@enforce_types(
    template=str,
    email_config=str,
    recipients=str,
    png_save_path=str,
    sites=list,
    alert_threshold=int,
    mastodon_config=(str, Path, type(None)),
    alert_log_path=str,
    verbose=bool,
    path_prefix=str,
)
def alert(template: str = './email_templates/legacy_template.html',
          email_config: str = './email_config.toml', recipients: str = './recipients.txt',
          png_save_path: str = './magnetometer_live/', sites: list[str] = ['dun', 'val'],
          alert_threshold: int = 6, mastodon_config: str | Path | None = None, alert_log_path: str = "./alert_log.json",
          verbose: bool = True, path_prefix: str = 'https://data.magie.ie/') -> None:
    """
    Generate and dispatch legacy geomagnetic alert notifications.

    The function computes current K values for the requested sites, writes the
    associated plots, renders the legacy email template, and optionally posts
    the alert to Mastodon when configured.

    Parameters
    ----------
    template : str, optional
        Path to the HTML email template.
    email_config : str, optional
        Path to the TOML email configuration file.
    recipients : str, optional
        Path to the newline-delimited recipient list.
    png_save_path : str, optional
        Output directory prefix for generated PNG plots.
    sites : list[str], optional
        Site codes to evaluate for alert conditions.
    alert_threshold : int, optional
        Minimum K index required before an alert is sent.
    mastodon_config : str or pathlib.Path or None, optional
        TOML config for Mastodon posting. When ``None``, no social post is made.
    alert_log_path : str, optional
        JSON file used to suppress duplicate alerts.
    verbose : bool, optional
        When ``True``, print progress messages when an alert condition is met.
    path_prefix : str, optional
        Data source prefix passed to ``live_k``. The default HTTPS prefix
        downloads archived data, while a local folder prefix reads from local
        ``YYYY/MM/DD/txt/`` directories instead.

    Returns
    -------
    None
        This function is called for its side effects.
    """
    paths = validate_alert_paths(
        template=template,
        email_config=email_config,
        recipients=recipients,
        png_save_path=png_save_path,
        alert_log_path=alert_log_path,
        mastodon_config=mastodon_config,
    )
    template_path = paths["template"]
    email_config_path = paths["email_config"]
    recipients_path = paths["recipients"]
    png_dir = paths["png_save_path"]
    alert_log_path = paths["alert_log_path"]
    mastodon_config_path = paths.get("mastodon_config")

    raw_template = template_path.read_text(encoding="utf-8")
    site_block_template_text = extract_site_block(raw_template)
    site_blocks_rendered: list[str] = []
    now_time= pd.Timestamp.now()
    with tempfile.TemporaryDirectory(prefix="magie_alert_") as tmpdir:
        tmpdir = Path(tmpdir)
        Ks = []
        for site in sites:
            kvals = live_k(now_time, site_code=site, path_prefix=path_prefix)
            fig, ax, _ = plot_k(kvals)
            met = get_site_metadata(site)
            fig.suptitle(f"{met['station_name']} 3-Day Local K Index", fontsize=80)

            ax.set_ylabel('K Index (0-9)', size=30)
            fig.savefig(png_dir / f"{site}_kindex.png", dpi=300, bbox_inches='tight')

            kvals = pd.DataFrame({'K_index' : kvals['var1']}, index= kvals['time'])
            kvals = kvals.dropna().iloc[-1]
            if (
                kvals.name < now_time - pd.Timedelta(6.5, 'h')
                or kvals['K_index'] < alert_threshold
                or check_logs(kvals.name, met['station_name'], path=alert_log_path) == 'no alert'
            ):
                continue
            Ks.append(int(kvals['K_index']))
            if verbose:
                print("Alert condition met, preparing email...")
            html_inputs = {}

            site_block_tmp = tmpdir / "_site_block.html"
            site_block_tmp.write_text(site_block_template_text, encoding="utf-8")

            html_inputs.update({'k_value': int(kvals['K_index']),
                        'site': met['station_name'],
                        'start_ut': str(kvals.name.time())[:-3],
                        'end_ut': str((kvals.name + pd.Timedelta(3, 'h')).time())[:-3],
                        'date': format_date_as_string(kvals.name),
                        'storm_class': classify_storm(kvals['K_index'])})

            rendered_block = render_html_template(str(site_block_tmp), html_inputs)
            site_blocks_rendered.append(rendered_block)

            archive_url = build_archive_url(kvals.name)
            save_log(kvals.name, int(kvals['K_index']), met['station_name'], path=alert_log_path)

        if site_blocks_rendered:
            combined_blocks_html = "\n".join(site_blocks_rendered)

            stitched_template = replace_site_block(raw_template, combined_blocks_html)

            stitched_tmp = tmpdir / "_stitched.html"
            stitched_tmp.write_text(stitched_template, encoding="utf-8")
            html_email = render_html_template(str(stitched_tmp),
                                            {"archive_url": archive_url},
                                            )
            cfg = load_email_config(email_config_path)
            recipients = load_recipients(recipients_path)
            send_html_email(
                smtp_host=cfg["smtp_host"],
                smtp_port=cfg["smtp_port"],
                username=cfg["username"],
                password=cfg["password"],
                from_addr=cfg["from_addr"],
                to_addrs=recipients,
                subject="MagIE Aurora Alert",
                html_content=html_email,
                use_starttls=cfg["use_starttls"],
            )

            if mastodon_config_path is not None:
                from mastodon import Mastodon
                # Initialize Mastodon client
                mastodon = Mastodon(**load_mastodon_config(mastodon_config_path))

                # Paths to your images
                image_paths = [
                        png_dir / "dun_kindex.png",
                        png_dir / "val_kindex.png",
                ]

                # Upload images
                media_ids = []
                for path, site in zip(image_paths, sites):
                    media = mastodon.media_post(path, focus=(0.9, .3),
                                                description = f"K index  {get_site_metadata(site)['station_name']}")
                    media_ids.append(media["id"])

                # Post status with images
                status = 'MagIE Geomagnetic Alert: K-'+str(max(Ks))+f' ({classify_storm(max(Ks))})'+' Check out https://magie.ie/data for more information #Alert #Aurora #AuroraBorealis #NorthernLights #MagIE #Ireland #MastoDaoine'
                mastodon.status_post(status, media_ids=media_ids)

@enforce_types(key=str)
def extract_entry_date(key: str) -> date:
    """
    Parse the date portion of a stored alert-log key.

    Parameters
    ----------
    key : str
        Alert log key in ``"site|timestamp"`` format.

    Returns
    -------
    datetime.date
        Date extracted from the timestamp portion of the key.

    Raises
    ------
    ValueError
        If the key does not contain a valid timestamp payload.
    """
    try:
        _, timestamp = key.rsplit("|", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid log key format: {key!r}") from exc

    try:
        return datetime.fromisoformat(timestamp).date()
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp in log key: {key!r}") from exc


@enforce_types(logs=dict, today=date)
def prune_log_entries(logs: dict[str, int], today: date) -> tuple[dict[str, int], list[str]]:
    """
    Remove alert-log entries older than the retention window.

    Parameters
    ----------
    logs : dict[str, int]
        Existing alert log mapping.
    today : datetime.date
        Reference date used to calculate the retention cutoff.

    Returns
    -------
    tuple[dict[str, int], list[str]]
        Pruned log mapping and a list of keys whose timestamps could not be
        parsed and were therefore retained.
    """
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


@enforce_types(log_path=(str, Path), today=(date, type(None)))
def clean_alert_log(log_path: str | Path = './alert_log.json', today: date | None = None) -> int:
    """
    Rewrite the alert log after dropping entries older than two days.

    Parameters
    ----------
    log_path : str or pathlib.Path, optional
        JSON alert log to clean.
    today : datetime.date or None, optional
        Reference date for pruning. Defaults to the current local date.

    Returns
    -------
    int
        Number of invalid keys encountered during pruning.
    """
    log_path = Path(log_path)
    today = today or date.today()
    logs = load_log(log_path)
    pruned_logs, invalid_keys = prune_log_entries(logs, today)

    log_path.write_text(json.dumps(pruned_logs, sort_keys=True), encoding="utf-8")

    return len(invalid_keys)


if __name__ == "__main__":
    alert()
