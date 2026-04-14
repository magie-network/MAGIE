from magie import plot_k, live_k
import pandas as pd
from magie.utils import get_site_metadata
from magie.email_utils import render_html_template, send_html_email, load_email_config, load_recipients, load_mastodon_config
import json
from pathlib import Path
import tempfile
import re



def format_date_as_string(dt: pd.Timestamp) -> str:
    """
    Format date like: '12th of December 2025'
    """
    day = dt.day

    # Handle 11th, 12th, 13th correctly
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    return f"{day}{suffix} of {dt.strftime('%B %Y')}"

def classify_storm(kp: float) -> str:
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
def load_log(path="alert_log.json") -> dict:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())
def make_log_key(site: str, timestamp: str) -> str:
    return f"{site}|{timestamp}"

def check_logs(datetime, site, path="alert_log.json"):
    logs= load_log(path)
    if make_log_key(site, datetime) in logs:
            return 'no alert'
    else:
        return 'new alert'
def save_log(datetime, k_value, site, path="alert_log.json"):
    logs= load_log(path)
    logs[make_log_key(site, datetime)]= k_value
    Path(path).write_text(json.dumps(logs))

def extract_site_block(template_text: str) -> str:
    m = re.search(
        r"<!--\s*BEGIN SITE_BLOCK\s*-->(.*?)<!--\s*END SITE_BLOCK\s*-->",
        template_text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Could not find SITE_BLOCK markers in template.")
    return m.group(1)


def replace_site_block(template_text: str, rendered_blocks_html: str) -> str:
    return re.sub(
        r"<!--\s*BEGIN SITE_BLOCK\s*-->(.*?)<!--\s*END SITE_BLOCK\s*-->",
        rendered_blocks_html,
        template_text,
        flags=re.S,
    )


def build_archive_url(dt_utc: pd.Timestamp) -> str:
    d = dt_utc.date()
    return f"https://data.magie.ie/{d.year:04d}/{d.month:02d}/{d.day:02d}/png/"

def alert(template: str = './email_templates/legacy_template.html',
          email_config: str = './email_config.toml', recipients: str = './recipients.txt',
          png_save_path: str = './magnetometer_live/', sites: list[str] = ['dun', 'val'],
          alert_threshold: int = 6, mastodon_config = None, alert_log_path: str = "./alert_log.json") -> None:
    raw_template = Path(template).read_text(encoding="utf-8")
    site_block_template_text = extract_site_block(raw_template)
    site_blocks_rendered: list[str] = []
    now_time= pd.Timestamp.now()
    with tempfile.TemporaryDirectory(prefix="magie_alert_") as tmpdir:
        tmpdir = Path(tmpdir)
        Ks = []
        for site in sites:
            kvals = live_k(now_time, site_code=site)
            fig, ax, _ = plot_k(kvals)
            met = get_site_metadata(site)
            fig.suptitle(f"{met['station_name']} 3-Day Local K Index", fontsize=80)

            ax.set_ylabel('K Index (0-9)', size=30)
            fig.savefig(png_save_path + f"{site}_kindex.png", dpi=300, bbox_inches='tight')

            kvals = pd.DataFrame({'K_index' : kvals['var1']}, index= kvals['time'])
            kvals = kvals.dropna().iloc[-1]
            if (
                kvals.name < now_time - pd.Timedelta(6.5, 'h')
                or kvals['K_index'] < alert_threshold
                or check_logs(kvals.name, met['station_name'], path=alert_log_path) == 'no alert'
            ):
                continue
            Ks.append(int(kvals['K_index']))
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
            cfg = load_email_config(email_config)
            recipients = load_recipients(recipients)
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

            if mastodon_config is not None:
                from mastodon import Mastodon
                # Initialize Mastodon client
                mastodon = Mastodon(**load_mastodon_config(mastodon_config))

                # Paths to your images
                image_paths = [
                        png_save_path + "dun_kindex.png",
                        png_save_path + "val_kindex.png",
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
                print(status)

                print("Posted successfully!")

if __name__ == "__main__":
    alert()
