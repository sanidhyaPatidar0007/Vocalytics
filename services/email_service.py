"""Outgoing e-mail over SMTP (anomaly alerts and the administrator test message).

Credentials come only from the environment (``config.settings``); the password
is never returned, logged or stored. Delivery problems never raise: ``send_email``
returns ``{"status": "failed", "error": <user-safe text>}`` so a mail outage
cannot break the caller (for example an anomaly check). Invalid input, such as
a malformed address or a header-injection attempt, is a caller bug and raises
``ValidationError`` before anything is sent.
"""

from __future__ import annotations

import logging
import re
import smtplib
import socket
import ssl
from datetime import timedelta
from email import policy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid, parseaddr
from html import escape
from typing import Any

from sqlalchemy import func, select

from auth.authorization import AuthContext, authorize
from config.settings import get_settings
from core.constants import Feature, Role
from core.errors import PlanLimitExceededError, ValidationError
from core.timeutil import utcnow
from database.connection import session_scope
from database.models import AuditLog
from services.audit_service import record_audit

logger = logging.getLogger(__name__)

MAX_RECIPIENTS = 50
MAX_ADDRESS_LENGTH = 254
MAX_SUBJECT_LENGTH = 300
# Test messages go out through the shared platform account: limit them per organization.
TEST_EMAIL_HOURLY_LIMIT = 5
_LOCAL_OR_DOMAIN = r"[^@\s<>,;:\"'()\[\]\\\x00-\x1f\x7f]+"
_ADDRESS = re.compile(rf"^{_LOCAL_OR_DOMAIN}@{_LOCAL_OR_DOMAIN}\.{_LOCAL_OR_DOMAIN}$")
_LINE_BREAKS = ("\r", "\n")


def smtp_status() -> dict[str, Any]:
    """Describe the SMTP configuration without ever exposing the password."""
    settings = get_settings()
    return {
        "configured": settings.smtp_configured,
        "host": settings.smtp_host,
        "port": settings.smtp_port,
        "use_tls": settings.smtp_use_tls,
        "use_ssl": settings.smtp_use_ssl,
        "from_address": settings.smtp_from,
        "username_set": bool(settings.smtp_username),
    }


def validate_address(address: object) -> str:
    """Return a trimmed, validated e-mail address or raise ``ValidationError``.

    The check is deliberately simple (one ``@``, a dot in the domain, no
    whitespace, quotes, brackets, commas or control characters). Line breaks are
    rejected explicitly because they enable header injection.
    """
    if not isinstance(address, str):
        raise ValidationError("E-mail addresses must be text.")
    if any(ch in address for ch in _LINE_BREAKS):
        raise ValidationError("E-mail addresses must not contain line breaks.")
    cleaned = address.strip()
    if not cleaned or len(cleaned) > MAX_ADDRESS_LENGTH or not _ADDRESS.match(cleaned):
        raise ValidationError(f"Invalid e-mail address: {cleaned[:60]!r}")
    return cleaned


def normalize_recipients(recipients: object) -> list[str]:
    """Validate a list of addresses, drop case-insensitive duplicates and keep order."""
    if isinstance(recipients, (str, bytes)) or not isinstance(recipients, (list, tuple)):
        raise ValidationError("Recipients must be a list of e-mail addresses.")
    result: list[str] = []
    seen: set[str] = set()
    for item in recipients:
        address = validate_address(item)
        if address.casefold() not in seen:
            seen.add(address.casefold())
            result.append(address)
    if len(result) > MAX_RECIPIENTS:
        raise ValidationError(f"At most {MAX_RECIPIENTS} recipients are allowed.")
    return result


def _validate_subject(subject: object) -> str:
    if not isinstance(subject, str):
        raise ValidationError("The subject must be text.")
    if any(ch in subject for ch in _LINE_BREAKS):
        raise ValidationError("The subject must not contain line breaks.")
    cleaned = subject.strip()
    if not cleaned or len(cleaned) > MAX_SUBJECT_LENGTH:
        raise ValidationError(f"The subject must be 1 to {MAX_SUBJECT_LENGTH} characters.")
    return cleaned


def _build_message(
    sender: str, recipients: list[str], subject: str, body_text: str, body_html: str | None
) -> MIMEMultipart:
    """Assemble a multipart/alternative message (plain text first, optional HTML second)."""
    message = MIMEMultipart("alternative", policy=policy.SMTP)
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Date"] = formatdate(localtime=False)
    message["Message-ID"] = make_msgid(domain=parseaddr(sender)[1].rpartition("@")[2] or "localhost")
    message.attach(MIMEText(body_text, "plain", "utf-8", policy=policy.SMTP))
    if body_html:
        message.attach(MIMEText(body_html, "html", "utf-8", policy=policy.SMTP))
    return message


def _failure_message(exc: BaseException) -> str:
    """Map a delivery exception to a fixed, credential-free sentence."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return "The mail server rejected the sign-in. Check SMTP_USERNAME and SMTP_PASSWORD."
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return "The mail server refused the recipient address(es)."
    if isinstance(exc, smtplib.SMTPSenderRefused):
        return "The mail server refused the sender address. Check SMTP_FROM."
    if isinstance(exc, ssl.SSLError):
        return (
            "A secure connection to the mail server could not be set up. "
            "Check SMTP_USE_TLS, SMTP_USE_SSL and SMTP_PORT."
        )
    if isinstance(exc, (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError)):
        return "The mail server closed the connection. Check SMTP_HOST, SMTP_PORT and the TLS/SSL settings."
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "The mail server did not answer in time. Check SMTP_HOST and SMTP_PORT."
    if isinstance(exc, smtplib.SMTPException):
        return "The mail server reported an error and the message was not delivered."
    return "The mail server could not be reached. Check SMTP_HOST, SMTP_PORT and the network connection."


def _deliver(message: MIMEMultipart, sender: str, recipients: list[str]) -> None:
    """Open one SMTP session and send ``message`` (raises smtplib/OS errors)."""
    settings = get_settings()
    host, port, timeout = settings.smtp_host, settings.smtp_port, settings.smtp_timeout_seconds
    if settings.smtp_use_ssl:
        client = smtplib.SMTP_SSL(host, port, timeout=timeout, context=ssl.create_default_context())
    else:
        client = smtplib.SMTP(host, port, timeout=timeout)
    with client as server:
        if settings.smtp_use_tls and not settings.smtp_use_ssl:
            server.starttls(context=ssl.create_default_context())
        if settings.smtp_username:
            server.login(settings.smtp_username, settings.smtp_password or "")
        server.send_message(message, from_addr=sender, to_addrs=recipients)


def send_email(
    *, to: list[str], subject: str, body_text: str, body_html: str | None = None
) -> dict[str, str | None]:
    """Send one message and report ``{"status": "sent"|"skipped"|"failed", "error": str | None}``.

    Input is validated first (``ValidationError`` for bad addresses, line breaks
    in headers or an empty body), so a malformed request is rejected even when
    SMTP is not configured. When SMTP is not configured the result is
    ``skipped``. Delivery errors (connection, TLS, authentication, refused
    recipients, timeouts) are returned as ``failed`` with a fixed user-safe
    sentence; the log line contains only the exception type, never credentials.
    """
    recipients = normalize_recipients(to)
    if not recipients:
        raise ValidationError("At least one recipient is required.")
    clean_subject = _validate_subject(subject)
    if not isinstance(body_text, str) or not body_text.strip():
        raise ValidationError("The message body must not be empty.")
    settings = get_settings()
    if not settings.smtp_configured:
        return {
            "status": "skipped",
            "error": "E-mail is not configured. Set SMTP_HOST and SMTP_FROM in the environment to enable it.",
        }
    sender = settings.smtp_from or ""
    if any(ch in sender for ch in _LINE_BREAKS):
        return {"status": "failed", "error": "The configured sender address (SMTP_FROM) is not valid."}
    try:
        message = _build_message(sender, recipients, clean_subject, body_text, body_html)
    except ValueError as exc:
        raise ValidationError("The message could not be built from the given text.") from exc
    try:
        _deliver(message, sender, recipients)
    except (smtplib.SMTPException, OSError) as exc:
        logger.warning(
            "E-mail delivery failed (%s) via %s:%s",
            type(exc).__name__,
            settings.smtp_host,
            settings.smtp_port,
        )
        return {"status": "failed", "error": _failure_message(exc)}
    logger.info("E-mail sent to %d recipient(s) via %s", len(recipients), settings.smtp_host)
    return {"status": "sent", "error": None}


def send_test_email(ctx: AuthContext, to: str) -> dict[str, str | None]:
    """Send a short test message to one address (Organization Admin only).

    Lets an administrator verify the SMTP settings. Needs the EMAIL_ALERTS plan
    feature, and each organization can send at most TEST_EMAIL_HOURLY_LIMIT test
    messages per hour (counted from the ``email.test`` audit entries). Records the
    audit entry ``email.test`` with the recipient address and the outcome, never
    the password.
    """
    with session_scope() as db:
        org_id = authorize(db, ctx, roles=(Role.ORG_ADMIN,), feature=Feature.EMAIL_ALERTS)
        recent = db.execute(
            select(func.count(AuditLog.id)).where(
                AuditLog.organization_id == org_id,
                AuditLog.action == "email.test",
                AuditLog.created_at >= utcnow() - timedelta(hours=1),
            )
        ).scalar_one()
    if recent >= TEST_EMAIL_HOURLY_LIMIT:
        raise PlanLimitExceededError(
            f"Your organization can send at most {TEST_EMAIL_HOURLY_LIMIT} test e-mails per hour. "
            "Try again later."
        )
    address = validate_address(to)
    app_name = get_settings().app_name
    result = send_email(
        to=[address],
        subject=f"{app_name}: test message",
        body_text=(
            f"This is a test message from {app_name}.\n"
            "If you can read it, e-mail alerts can be delivered to this address."
        ),
        body_html=(
            f"<p>This is a test message from {escape(app_name)}.</p>"
            "<p>If you can read it, e-mail alerts can be delivered to this address.</p>"
        ),
    )
    with session_scope() as db:
        record_audit(
            db,
            ctx,
            "email.test",
            target_type="email",
            details={"recipient": address, "status": result["status"]},
            organization_id=ctx.organization_id,
        )
    return result
