import smtplib
from email.message import EmailMessage
from pathlib import Path
import mimetypes
from string import Template
from html import escape
import os
from magie.utils import enforce_types, get_asset_bytes
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # Python 3.10 and below


@enforce_types(path=(str, Path))
def load_email_config(path: str | Path) -> dict:
    """
    Load SMTP and sender settings from a TOML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the email configuration file.

    Returns
    -------
    dict
        Normalized SMTP settings and sender address.

    Raises
    ------
    RuntimeError
        If a username is provided without a password.
    """
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


@enforce_types(path=(str, Path))
def load_mastodon_config(path: str | Path) -> dict:
    """
    Load Mastodon API credentials from a TOML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Mastodon configuration file.

    Returns
    -------
    dict
        Access token and API base URL.
    """
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    access_token = cfg["access_token"]
    api_base_url = cfg["api_base_url"]

    return {
        "access_token": access_token,
        "api_base_url": api_base_url}


@enforce_types(path=(str, Path))
def load_recipients(path: str | Path) -> list[str]:
    """
    Load recipient email addresses from a newline-delimited text file.

    Parameters
    ----------
    path : str or pathlib.Path
        File containing one recipient per line.

    Returns
    -------
    list[str]
        Recipient addresses with blank lines and comments removed.
    """
    recipients = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        recipients.append(line)
    return recipients


@enforce_types(source=(str, Path))
def _read_inline_image(source: str | Path) -> tuple[bytes, str]:
    """
    Read inline image content from disk or packaged assets.

    Parameters
    ----------
    source : str or pathlib.Path
        Filesystem path or packaged asset name.

    Returns
    -------
    tuple[bytes, str]
        Image bytes and the resolved filename.
    """
    path = Path(source)
    if path.exists():
        return path.read_bytes(), path.name

    return get_asset_bytes(path.name), path.name


@enforce_types(
    smtp_host=str,
    smtp_port=int,
    username=(str, type(None)),
    password=(str, type(None)),
    from_addr=str,
    to_addrs=list,
    subject=str,
    html_path=(str, type(None)),
    html_content=(str, type(None)),
    inline_images=(dict, type(None)),
    attachments=(list, type(None)),
    use_starttls=bool,
    timeout=(int, float),
)
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

    Supports unauthenticated local relays and authenticated STARTTLS
    submission depending on the supplied credentials and flags.

    Parameters
    ----------
    smtp_host : str
        SMTP server hostname.
    smtp_port : int
        SMTP server port.
    username : str or None
        Username for SMTP authentication.
    password : str or None
        Password for SMTP authentication.
    from_addr : str
        Sender address.
    to_addrs : list[str]
        Recipient addresses.
    subject : str
        Email subject line.
    html_path : str or None, optional
        Path to an HTML file to load when ``html_content`` is not supplied.
    html_content : str or None, optional
        HTML payload to send directly.
    inline_images : dict[str, str] or None, optional
        Mapping of CID values to image paths or packaged asset names.
    attachments : list[str] or None, optional
        Attachment file paths.
    use_starttls : bool, optional
        Whether to require STARTTLS before sending.
    starttls_context : object, optional
        SSL context passed through to ``server.starttls()``.
    timeout : float, optional
        Socket timeout in seconds.

    Raises
    ------
    ValueError
        If the recipient list is empty, credentials are incomplete, or no HTML
        body source is provided.
    FileNotFoundError
        If any attachment path does not exist.
    RuntimeError
        If STARTTLS is requested but the server does not advertise support.
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
        for cid, image_source in inline_images.items():
            image_bytes, filename = _read_inline_image(image_source)
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type is None:
                mime_type = "application/octet-stream"
            maintype, subtype = mime_type.split("/", 1)

            html_part.add_related(
                image_bytes,
                maintype=maintype,
                subtype=subtype,
                cid=f"<{cid}>",
                filename=filename,
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


@enforce_types(template_path=str, values=dict)
def render_html_template(
    template_path: str,
    values: dict[str, str | int | float],
) -> str:
    """
    Render an HTML template using ``string.Template`` substitution.

    Parameters
    ----------
    template_path : str
        Path to the HTML template file.
    values : dict[str, str | int | float]
        Mapping of placeholders to replacement values.

    Returns
    -------
    str
        Rendered HTML with all replacement values HTML-escaped for safety.
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
        "magie_logo": "MagIE-logo.png",
        "logo_dias": "DIAS.png",
        "logo_gsi": "GSI.png",
        "logo_met": "met_eireann.jpg",
    })
