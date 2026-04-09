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

    use_starttls = bool(smtp.get("use_starttls", True))
    username = smtp.get("username", "") or ""
    password = smtp.get("password", "") or ""

    # Only require password if we’re actually authenticating
    if username and not password:
        raise RuntimeError("SMTP username provided but password is empty")

    return {
        "smtp_host": smtp["host"],
        "smtp_port": int(smtp["port"]),
        "username": username,
        "password": password,
        "use_starttls": use_starttls,
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
    username: str | None,
    password: str | None,
    from_addr: str,
    to_addrs: list[str],
    subject: str,
    html_path: str | None = None,
    html_content: str | None = None,
    inline_images: dict[str, str] | None = None,
    attachments: list[str] | None = None,
    use_starttls: bool = True,
    starttls_context=None,  # optional ssl.SSLContext
    timeout: float = 30.0,
):
    """
    Send an HTML email with optional inline images (CID) and attachments.

    Supports two modes:
      - Local relay (no auth, no TLS): use_starttls=False, username/password empty
      - Authenticated submission (STARTTLS + AUTH): use_starttls=True, username/password set

    Args:
      smtp_host/smtp_port: SMTP server
      username/password: optional; if provided, AUTH is attempted
      use_starttls: if True, attempt STARTTLS (and require it to be available)
      starttls_context: optional ssl.SSLContext passed to starttls()
      timeout: socket timeout in seconds
    """
    import smtplib
    from email.message import EmailMessage
    from pathlib import Path
    import mimetypes

    if not to_addrs:
        raise ValueError("Recipient list is empty")

    # Normalize credentials
    username = (username or "").strip()
    password = (password or "").strip()
    if (username and not password) or (password and not username):
        raise ValueError("SMTP username/password must either both be set or both be empty")

    # Load HTML
    if html_content is None:
        if html_path is None:
            raise ValueError("Provide either html_path or html_content")
        html_content = Path(html_path).read_text(encoding="utf-8")

    # Build message
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
            if not path.exists():
                raise FileNotFoundError(f"Inline image not found: {path}")

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

    # Attachments
    if attachments:
        for apath in attachments:
            path = Path(apath)
            if not path.exists():
                raise FileNotFoundError(f"Attachment not found: {path}")

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
    with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
        server.ehlo()

        if use_starttls:
            # If STARTTLS is requested, require the server to advertise it
            if not server.has_extn("STARTTLS"):
                raise RuntimeError(f"Server {smtp_host}:{smtp_port} does not support STARTTLS")
            server.starttls(context=starttls_context)
            server.ehlo()

        if username and password:
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
