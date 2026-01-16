import smtplib
from email.message import EmailMessage
from pathlib import Path
import mimetypes
from string import Template
from html import escape
import os
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 and below


def load_email_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    smtp = cfg["smtp"]
    email = cfg["email"]

    # password = os.getenv(smtp["password_env"])
    password = smtp['password']
    if not password:
        raise RuntimeError(
            f"Environment variable {smtp['password_env']} is not set"
        )

    return {
        "smtp_host": smtp["host"],
        "smtp_port": int(smtp["port"]),
        "username": smtp["username"],
        "password": password,
        "from_addr": email["from"],
    }


def load_recipients(path: str | Path) -> list[str]:
    recipients = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        recipients.append(line)
    return recipients


def send_html_email(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    from_addr: str,
    to_addrs: list[str],
    subject: str,
    html_path: str | None = None,
    html_content: str | None = None,
    inline_images: dict[str, str] | None = None,
    attachments: list[str] | None = None,

):
    if not to_addrs:
        raise ValueError("Recipient list is empty")

    if html_content is None:
        if html_path is None:
            raise ValueError("Provide either html_path or html_content")
        html_content = Path(html_path).read_text(encoding="utf-8")

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject

    # Plain-text fallback
    msg.set_content("This email contains HTML content.")

    # HTML part
    msg.add_alternative(html_content, subtype="html")
    html_part = msg.get_payload()[-1]

    # Inline images (CID)
    if inline_images:
        for cid, path in inline_images.items():
            path = Path(path)
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "application/octet-stream"
            maintype, subtype = mime_type.split("/", 1)

            html_part.add_related(
                path.read_bytes(),
                maintype=maintype,
                subtype=subtype,
                cid=f"<{cid}>",
                filename=path.name,
                disposition="inline",
            )
    if attachments:
        for path in attachments:
            path = Path(path)
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "application/octet-stream"
            maintype, subtype = mime_type.split("/", 1)

            msg.add_attachment(
                path.read_bytes(),
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )
    # Send
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)


def render_html_template(
    template_path: str,
    values: dict[str, str | int | float],
) -> str:
    """
    Renders an HTML email template using string.Template.

    - template_path: path to HTML file
    - values: dictionary of placeholder -> value

    All values are HTML-escaped for safety.
    """

    template_text = Path(template_path).read_text(encoding="utf-8")

    # HTML-escape all values to avoid breaking markup
    safe_values = {
        key: escape(str(value)) for key, value in values.items()
    }

    template = Template(template_text)
    return template.safe_substitute(safe_values)



if __name__ =='__main__':
    send_html_email('server', 'port', 'username',
                    'password', 'from_addr',
                    ['to_addr1', 'to_addr2'],
                    'MagIE Aurora Alert', '../../email_template.html',
                        inline_images={
        "magie_logo": "../../logos/MagIE-logo.png",
        "logo_dias": "../../logos/DIAS.png",
        "logo_gsi": "../../logos/GSI.png",
        "logo_met": "../../logos/met_eireann.jpg",
    })
