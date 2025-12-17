import smtplib
from email.message import EmailMessage
from pathlib import Path
import mimetypes


def send_html_email(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    from_addr: str,
    to_addrs: list[str],
    subject: str,
    html_path: str,
    inline_images: dict[str, str] | None = None,
):
    """
    inline_images example:
    {
        "logo": "logos/MagIE-logo.png",
        "logo_dias": "logos/dias.png"
    }
    Referenced in HTML as: <img src="cid:logo">
    """

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject

    # Plain text fallback (important for deliverability)
    msg.set_content("This email contains HTML content.")

    # Load HTML
    html_content = Path(html_path).read_text(encoding="utf-8")
    msg.add_alternative(html_content, subtype="html")

    html_part = msg.get_payload()[-1]

    # Attach inline images (optional)
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

    # Send
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(msg)

