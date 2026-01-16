from magie import live_k, plot_k
from magie.k_index import plot_k_plotly
import pandas as pd
from magie.email_utils import render_html_template, send_html_email, load_email_config, load_recipients
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import tempfile



def to_ireland_time(dt: datetime) -> datetime:
    """
    Convert a datetime to Ireland local time (DST-aware).
    Assumes UTC if dt is naive.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    return dt.astimezone(ZoneInfo("Europe/Dublin"))

def format_date_as_string(dt: datetime) -> str:
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
        return {"version": 1, "alerts": {}}
    return json.loads(Path(path).read_text())
def make_log_key(site: str, timestamp: str) -> str:
    return f"{site}|{timestamp}"

def check_logs(datetime, k_value, site, path="alert_log.json"):
    logs= load_log(path)
    if make_log_key(site, datetime) in logs:
        last_log= logs[make_log_key(site, datetime)]
        if k_value!=last_log['k_value']:
            return 'update', last_log['k_value']
        else:
            return 'no alert', None
    else:
        return 'new alert', None


alert_template="../email_templates/alert_template.html"
update_template="../email_templates/update_template.html"


df= live_k(pd.Timestamp.now(), 'dun')

fig= plot_k_plotly(df)
fig.update_layout(
    title=dict(
        text="Dunsink K Index",
        x=0.5,          # center title
        xanchor="center",
        font=dict(size=40)),
    
    margin=dict(
        t=90, )
)


# fig2, ax, _= plot_k(df)
# fig2.suptitle('Dunsink K Index')
# ax.set_ylabel('K Index (0-9)')
# ax.set_xlabel('Universal Time')

with tempfile.TemporaryDirectory(prefix="magie_alert_") as tmpdir:
    tmpdir = Path(tmpdir)

    png_path = tmpdir / "k_index_dunsink.png"
    png_path2 = tmpdir / "kp_plot2.png"

    html_path = tmpdir / "k_index_dunsink.html"

    fig.write_image(png_path, scale=2)
    fig.write_html(html_path, include_plotlyjs="cdn")
    # fig2.savefig(png_path2)


    alert_threshold=3
    if (df.K_index>alert_threshold).any():
        # Load config + recipients
        cfg = load_email_config("./email_config.toml")
        recipients = load_recipients("./recipients.txt")
        for index, row in df[df.K_index>alert_threshold].iterrows():
            k_value= int(row['K_index'])
            decision, old_k= check_logs('dun', index, k_value)
            html_inputs={}
            if decision=='no alert':
                continue
            elif decision=='update':
                template= update_template
                html_inputs.update({'old_k':old_k, 'direction':'downgraded' if k_value<old_k else 'upgraded'})
            else:
                template= alert_template


            irish_time= to_ireland_time(index)
            html_inputs.update({'k_value':k_value,
                        'site':'Dunsink',
                        'time_range':f"{str(irish_time.time())[:-3]} to {str((irish_time+pd.Timedelta(3, 'h')).time())[:-3]}",
                        'date':format_date_as_string(irish_time),
                        'storm_class': classify_storm(k_value)})


            # Render your HTML earlier (template replacement step)
            html_email = render_html_template(template, html_inputs)

            send_html_email(
                smtp_host=cfg["smtp_host"],
                smtp_port=cfg["smtp_port"],
                username=cfg["username"],
                password=cfg["password"],
                from_addr=cfg["from_addr"],
                to_addrs=recipients,
                subject="MagIE Aurora Alert",
                html_content=html_email,
                inline_images={
                    "magie_logo": "../logos/MagIE-logo.png",
                    "logo_dias": "../logos/DIAS.png",
                    "logo_gsi": "../logos/GSI.png",
                    "logo_met": "../logos/met_eireann.jpg",
                    "k_plot": png_path
                },
            attachments=[html_path])

