"""E-mail service: SMTP selection, failure mapping, header-injection defence, test message."""

from __future__ import annotations

import smtplib
import ssl
from datetime import timedelta

import pytest
from sqlalchemy import select

from config.settings import set_settings
from core.errors import (
    FeatureNotAvailableError,
    PermissionDeniedError,
    PlanLimitExceededError,
    ValidationError,
)
from core.timeutil import utcnow
from database import session_scope
from database.models import AuditLog
from services import email_service

PASSWORD = "s3cret-smtp-pw"


class FakeSMTP:
    """Records connections and messages; behaviour is steered by class attributes."""

    instances: list[FakeSMTP] = []
    fail_on: str | None = None
    error: BaseException | None = None

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
        self.started_tls = False
        self.logged_in: tuple[str, str] | None = None
        self.sent: list[tuple] = []
        FakeSMTP.instances.append(self)
        if FakeSMTP.fail_on == "connect":
            raise FakeSMTP.error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self, context=None):
        self._maybe_fail("starttls")
        self.started_tls = True

    def login(self, user, password):
        self._maybe_fail("login")
        self.logged_in = (user, password)

    def send_message(self, message, from_addr=None, to_addrs=None):
        self._maybe_fail("send")
        self.sent.append((message, from_addr, to_addrs))

    def _maybe_fail(self, step):
        if FakeSMTP.fail_on == step:
            raise FakeSMTP.error


class FakeSMTPSSL(FakeSMTP):
    pass


@pytest.fixture
def smtp(monkeypatch, app_settings):
    FakeSMTP.instances = []
    FakeSMTP.fail_on = None
    FakeSMTP.error = None
    FakeSMTPSSL.instances = FakeSMTP.instances
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTPSSL)

    def configure(**overrides):
        values = {
            "smtp_host": "smtp.example.test",
            "smtp_from": "alerts@example.test",
            "smtp_username": "mailer",
            "smtp_password": PASSWORD,
        }
        values.update(overrides)
        set_settings(app_settings.with_overrides(**values))

    configure()
    return configure


def _send(**kwargs):
    base = {"to": ["a@example.com"], "subject": "Hello", "body_text": "Body"}
    base.update(kwargs)
    return email_service.send_email(**base)


def test_status_never_contains_the_password(smtp):
    status = email_service.smtp_status()
    assert status == {
        "configured": True,
        "host": "smtp.example.test",
        "port": 587,
        "use_tls": True,
        "use_ssl": False,
        "from_address": "alerts@example.test",
        "username_set": True,
    }
    assert PASSWORD not in repr(status)


def test_status_when_not_configured():
    status = email_service.smtp_status()
    assert status["configured"] is False and status["host"] is None and status["username_set"] is False


def test_not_configured_is_skipped_with_reason():
    result = _send()
    assert result["status"] == "skipped"
    assert "SMTP_HOST" in result["error"]


def test_success_builds_multipart_alternative(smtp):
    result = _send(to=["a@example.com", "b@example.com"], subject="Ünïcode subject", body_html="<p>Hi</p>")
    assert result == {"status": "sent", "error": None}
    server = FakeSMTP.instances[0]
    message, from_addr, to_addrs = server.sent[0]
    assert from_addr == "alerts@example.test" and to_addrs == ["a@example.com", "b@example.com"]
    assert message.get_content_type() == "multipart/alternative"
    assert [part.get_content_type() for part in message.get_payload()] == ["text/plain", "text/html"]
    assert message["To"] == "a@example.com, b@example.com"
    assert message["Subject"] == "Ünïcode subject"
    assert message["Date"] and message["Message-ID"].endswith(
        "@example.test>"
    )  # sender domain, no host lookup
    assert server.timeout == 20
    assert server.logged_in == ("mailer", PASSWORD)


def test_plain_text_only_message_has_one_part(smtp):
    _send()
    message = FakeSMTP.instances[0].sent[0][0]
    assert [part.get_content_type() for part in message.get_payload()] == ["text/plain"]


def test_starttls_is_used_by_default(smtp):
    _send()
    server = FakeSMTP.instances[0]
    assert type(server) is FakeSMTP and server.started_tls is True


def test_ssl_selected_when_configured(smtp):
    smtp(smtp_use_ssl=True, smtp_use_tls=False, smtp_port=465)
    _send()
    server = FakeSMTP.instances[0]
    assert isinstance(server, FakeSMTPSSL) and server.port == 465
    assert server.started_tls is False and isinstance(server.context, ssl.SSLContext)


def test_plain_smtp_without_tls_and_without_login(smtp):
    smtp(smtp_use_tls=False, smtp_username=None, smtp_password=None)
    _send()
    server = FakeSMTP.instances[0]
    assert server.started_tls is False and server.logged_in is None and len(server.sent) == 1


@pytest.mark.parametrize(
    ("step", "error", "expected"),
    [
        ("connect", ConnectionRefusedError("refused"), "could not be reached"),
        ("connect", TimeoutError("timed out"), "did not answer in time"),
        ("starttls", ssl.SSLError("bad handshake"), "secure connection"),
        (
            "login",
            smtplib.SMTPAuthenticationError(535, b"bad credentials " + PASSWORD.encode()),
            "rejected the sign-in",
        ),
        ("send", smtplib.SMTPRecipientsRefused({"a@example.com": (550, b"no")}), "refused the recipient"),
        ("send", smtplib.SMTPSenderRefused(550, b"no", "alerts@example.test"), "refused the sender"),
        ("send", smtplib.SMTPServerDisconnected("gone"), "closed the connection"),
        ("send", smtplib.SMTPDataError(554, b"rejected"), "reported an error"),
    ],
)
def test_delivery_failures_return_failed_without_secrets(smtp, caplog, step, error, expected):
    FakeSMTP.fail_on, FakeSMTP.error = step, error
    with caplog.at_level("DEBUG"):
        result = _send()
    assert result["status"] == "failed"
    assert expected in result["error"]
    assert PASSWORD not in result["error"] and "535" not in result["error"]
    assert PASSWORD not in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"to": ["a@example.com\r\nBcc: evil@example.com"]},
        {"to": ["a@example.com\nBcc: evil@example.com"]},
        {"subject": "Hi\r\nBcc: evil@example.com"},
        {"subject": "Hi\nX-Injected: 1"},
        {"to": ["not-an-address"]},
        {"to": ["a@b"]},
        {"to": ["a b@example.com"]},
        {"to": ["<a@example.com>"]},
        {"to": ["a@example.com,b@example.com"]},
        {"to": []},
        {"to": "a@example.com"},
        {"to": [42]},
        {"subject": ""},
        {"subject": "x" * 301},
        {"body_text": "   "},
    ],
)
def test_invalid_input_is_rejected_even_when_not_configured(kwargs):
    with pytest.raises(ValidationError):
        _send(**kwargs)


def test_invalid_input_is_rejected_when_configured_and_nothing_is_sent(smtp):
    with pytest.raises(ValidationError):
        _send(subject="Hi\r\nBcc: evil@example.com")
    assert FakeSMTP.instances == []


def test_sender_with_line_break_is_reported_as_failed(smtp):
    smtp(smtp_from="alerts@example.test\r\nBcc: x@example.com")
    result = _send()
    assert result["status"] == "failed" and "SMTP_FROM" in result["error"]
    assert FakeSMTP.instances == []


def test_recipients_are_deduplicated_and_limited():
    assert email_service.normalize_recipients(["A@x.io", "a@X.io", " b@x.io "]) == ["A@x.io", "b@x.io"]
    many = [f"u{i}@example.com" for i in range(email_service.MAX_RECIPIENTS + 1)]
    with pytest.raises(ValidationError):
        email_service.normalize_recipients(many)
    assert len(email_service.normalize_recipients(many[:-1])) == email_service.MAX_RECIPIENTS


def test_test_email_success_audited_without_password(smtp, admin_ctx):
    result = email_service.send_test_email(admin_ctx, "ops@example.com")
    assert result["status"] == "sent"
    assert FakeSMTP.instances[0].sent[0][2] == ["ops@example.com"]
    with session_scope() as db:
        row = db.execute(select(AuditLog).where(AuditLog.action == "email.test")).scalar_one()
        assert row.organization_id == admin_ctx.organization_id
        assert "ops@example.com" in row.details_json and "sent" in row.details_json
        assert PASSWORD not in row.details_json


def test_test_email_not_configured_is_audited_as_skipped(admin_ctx):
    result = email_service.send_test_email(admin_ctx, "ops@example.com")
    assert result["status"] == "skipped"
    with session_scope() as db:
        assert (
            "skipped"
            in db.execute(select(AuditLog.details_json).where(AuditLog.action == "email.test")).scalar_one()
        )


def test_test_email_rejects_bad_address(admin_ctx):
    with pytest.raises(ValidationError):
        email_service.send_test_email(admin_ctx, "bad\r\naddress@example.com")


def test_test_email_is_org_admin_only(analyst_ctx, factory):
    with pytest.raises(PermissionDeniedError):
        email_service.send_test_email(analyst_ctx, "ops@example.com")
    with pytest.raises(PermissionDeniedError):
        email_service.send_test_email(factory.super_admin(), "ops@example.com")


def test_test_email_requires_the_email_alerts_feature(smtp, factory):
    basic_admin = factory.org_admin(factory.organization(plan="basic"))
    with pytest.raises(FeatureNotAvailableError):
        email_service.send_test_email(basic_admin, "someone@example.com")
    assert FakeSMTP.instances == []
    with session_scope() as db:
        assert db.execute(select(AuditLog).where(AuditLog.action == "email.test")).first() is None


def test_test_email_is_rate_limited_per_organization(smtp, admin_ctx, factory):
    limit = email_service.TEST_EMAIL_HOURLY_LIMIT
    for i in range(limit):
        assert email_service.send_test_email(admin_ctx, f"ops{i}@example.com")["status"] == "sent"
    with pytest.raises(PlanLimitExceededError, match="per hour"):
        email_service.send_test_email(admin_ctx, "one-more@example.com")
    assert sum(len(server.sent) for server in FakeSMTP.instances) == limit
    # Another organization has its own allowance.
    other_admin = factory.org_admin(factory.organization())
    assert email_service.send_test_email(other_admin, "ops@example.com")["status"] == "sent"
    # Sends older than an hour no longer count.
    with session_scope() as db:
        for row in db.execute(
            select(AuditLog).where(
                AuditLog.action == "email.test", AuditLog.organization_id == admin_ctx.organization_id
            )
        ).scalars():
            row.created_at = utcnow() - timedelta(hours=2)
    assert email_service.send_test_email(admin_ctx, "later@example.com")["status"] == "sent"
