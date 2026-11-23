"""Authentication: hashing, login, lockout, sessions, logout and password changes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from auth import authentication, password_manager
from auth.authentication import (
    GENERIC_LOGIN_FAILURE,
    active_sessions,
    change_password,
    login,
    logout,
    purge_old_sessions,
    revoke_user_sessions,
    validate_session,
)
from auth.authorization import AuthContext
from config.settings import set_settings
from core.constants import Role
from core.errors import (
    AccountLockedError,
    AuthenticationError,
    PermissionDeniedError,
    SessionExpiredError,
    ValidationError,
)
from database import session_scope
from database.models import AuditLog, LoginAttempt, Organization, User, UserSession
from tests.conftest import DEFAULT_PASSWORD

NEW_PASSWORD = "N3w!Secure#Pass"
START = datetime(2026, 1, 10, 9, 0, 0)


class FakeClock:
    """Mutable clock patched onto ``core.timeutil.utcnow``."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock(START)
    monkeypatch.setattr("core.timeutil.utcnow", fake)
    return fake


@pytest.fixture
def lockout_settings(app_settings):
    """Three failures within 15 minutes lock the account for 15 minutes."""
    settings = app_settings.with_overrides(
        login_max_failed_attempts=3, login_attempt_window_minutes=15, login_lockout_minutes=15
    )
    set_settings(settings)
    return settings


def make_user(factory, org_id, *, role: str = "analyst", username: str = "alice", **kwargs) -> int:
    return factory.user(org_id, role, username=username, **kwargs)


def session_row(token: str) -> dict:
    with session_scope() as db:
        row = db.execute(
            select(UserSession).where(UserSession.token_hash == hashlib.sha256(token.encode()).hexdigest())
        ).scalar_one()
        return {
            "id": row.id,
            "revoked_at": row.revoked_at,
            "revoke_reason": row.revoke_reason,
            "last_seen_at": row.last_seen_at,
            "expires_at": row.expires_at,
            "user_agent": row.user_agent,
        }


def attempts(username: str) -> list[tuple[bool, str | None]]:
    with session_scope() as db:
        rows = db.execute(
            select(LoginAttempt).where(LoginAttempt.username == username).order_by(LoginAttempt.id)
        ).scalars()
        return [(row.success, row.reason) for row in rows]


def audit_rows(action: str) -> list[AuditLog]:
    with session_scope() as db:
        return list(
            db.execute(select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.id)).scalars()
        )


def set_user(user_id: int, **values: object) -> None:
    with session_scope() as db:
        user = db.get(User, user_id)
        for key, value in values.items():
            setattr(user, key, value)


def set_org_status(org_id: int, status: str) -> None:
    with session_scope() as db:
        db.get(Organization, org_id).status = status


# ---------------------------------------------------------------------------
# Password hashing and policy
# ---------------------------------------------------------------------------


def test_argon2_hashes_are_salted_and_verify():
    first = password_manager.hash_password(DEFAULT_PASSWORD)
    second = password_manager.hash_password(DEFAULT_PASSWORD)
    assert first != second
    assert first.startswith("$argon2id$")
    assert password_manager.verify_password(first, DEFAULT_PASSWORD)
    assert password_manager.verify_password(second, DEFAULT_PASSWORD)


def test_wrong_password_does_not_verify():
    stored = password_manager.hash_password(DEFAULT_PASSWORD)
    assert not password_manager.verify_password(stored, DEFAULT_PASSWORD + "x")
    assert not password_manager.verify_password(stored, "")
    assert not password_manager.verify_password("not-a-hash", DEFAULT_PASSWORD)


def test_needs_rehash_after_parameter_change_and_login_upgrades_hash(app_settings, factory, org_id, clock):
    user_id = make_user(factory, org_id)
    with session_scope() as db:
        old_hash = db.get(User, user_id).password_hash
    assert not password_manager.needs_rehash(old_hash)
    set_settings(app_settings.with_overrides(password_hash_time_cost=2))
    assert password_manager.needs_rehash(old_hash)

    login("alice", DEFAULT_PASSWORD)
    with session_scope() as db:
        new_hash = db.get(User, user_id).password_hash
    assert new_hash != old_hash
    assert not password_manager.needs_rehash(new_hash)
    assert password_manager.verify_password(new_hash, DEFAULT_PASSWORD)


@pytest.mark.parametrize(
    ("password", "fragment"),
    [
        ("Sh0rt!", "at least"),
        ("alllowercaseletters", "three of"),
        ("Password123", "too common"),
        ("Xy!alice-2024", "username"),
        ("Aa1!" * 65, "at most 256"),
    ],
)
def test_password_policy_errors(password, fragment):
    with pytest.raises(ValidationError) as excinfo:
        password_manager.validate_password_strength(password, username="alice")
    assert fragment in excinfo.value.user_message


def test_strong_password_passes_policy():
    password_manager.validate_password_strength(NEW_PASSWORD, username="alice")


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def test_login_success_creates_session_storing_only_token_hash(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    result = login("  ALICE ", DEFAULT_PASSWORD, user_agent="UA/" + "x" * 400)

    assert result.ctx == AuthContext(
        user_id=user_id,
        role=Role.ANALYST,
        organization_id=org_id,
        username="alice",
        session_id=result.ctx.session_id,
    )
    assert result.ctx.session_id is not None
    assert result.expires_at == START + timedelta(hours=12)
    assert result.must_change_password is False
    assert len(result.token) >= 40

    with session_scope() as db:
        rows = db.execute(select(UserSession)).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.token_hash == hashlib.sha256(result.token.encode()).hexdigest()
        assert row.token_hash != result.token
        assert result.token not in (row.token_hash, row.user_agent)
        assert row.created_at == row.last_seen_at == START
        assert len(row.user_agent) == 300
        assert row.organization_id == org_id
        user = db.get(User, user_id)
        assert user.last_login_at == START
        assert user.failed_login_count == 0
    assert attempts("alice") == [(True, None)]


def test_login_reports_must_change_password(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    set_user(user_id, must_change_password=True)
    assert login("alice", DEFAULT_PASSWORD).must_change_password is True


def test_each_login_gets_a_distinct_token(factory, org_id, clock):
    make_user(factory, org_id)
    first = login("alice", DEFAULT_PASSWORD)
    second = login("alice", DEFAULT_PASSWORD)
    assert first.token != second.token
    assert first.ctx.session_id != second.ctx.session_id


def test_unknown_user_and_wrong_password_give_identical_message(factory, org_id, clock):
    make_user(factory, org_id)
    with pytest.raises(AuthenticationError) as unknown:
        login("nobody", DEFAULT_PASSWORD)
    with pytest.raises(AuthenticationError) as wrong:
        login("alice", "Wr0ng!Password")
    assert type(unknown.value) is type(wrong.value) is AuthenticationError
    assert unknown.value.user_message == wrong.value.user_message == GENERIC_LOGIN_FAILURE
    assert attempts("nobody") == [(False, "unknown_user")]
    assert attempts("alice") == [(False, "bad_password")]
    with session_scope() as db:
        assert db.execute(select(UserSession)).scalars().first() is None


@pytest.mark.parametrize(("username", "password"), [("", DEFAULT_PASSWORD), ("alice", ""), (None, None)])
def test_empty_credentials_fail_generically(factory, org_id, clock, username, password):
    make_user(factory, org_id)
    with pytest.raises(AuthenticationError) as excinfo:
        login(username, password)
    assert excinfo.value.user_message == GENERIC_LOGIN_FAILURE


def test_inactive_user_gets_generic_message(factory, org_id, clock):
    make_user(factory, org_id, active=False)
    with pytest.raises(AuthenticationError) as excinfo:
        login("alice", DEFAULT_PASSWORD)
    assert type(excinfo.value) is AuthenticationError
    assert excinfo.value.user_message == GENERIC_LOGIN_FAILURE
    assert attempts("alice") == [(False, "inactive")]


@pytest.mark.parametrize("status", ["suspended", "deactivated"])
def test_inactive_org_blocked_even_with_correct_password(factory, clock, status):
    org = factory.organization(status=status)
    make_user(factory, org, role="org_admin")
    with pytest.raises(PermissionDeniedError) as excinfo:
        login("alice", DEFAULT_PASSWORD)
    assert "organization is not active" in excinfo.value.user_message
    assert attempts("alice") == [(False, "org_inactive")]
    with session_scope() as db:
        assert db.execute(select(UserSession)).scalars().first() is None


def test_inactive_org_with_wrong_password_reveals_nothing(factory, clock):
    org = factory.organization(status="suspended")
    make_user(factory, org)
    with pytest.raises(AuthenticationError) as excinfo:
        login("alice", "Wr0ng!Password")
    assert excinfo.value.user_message == GENERIC_LOGIN_FAILURE


def test_super_admin_logs_in_without_organization(factory, clock):
    user_id = factory.user(None, "super_admin", username="root")
    result = login("root", DEFAULT_PASSWORD)
    assert result.ctx.role == Role.SUPER_ADMIN
    assert result.ctx.organization_id is None
    ctx = validate_session(result.token)
    assert ctx.user_id == user_id and ctx.is_super_admin


def test_unknown_user_runs_dummy_hash_verification(factory, org_id, clock, monkeypatch):
    make_user(factory, org_id)
    calls: list[str] = []
    real_verify = password_manager.verify_password

    def counting_verify(password_hash: str, password: str) -> bool:
        calls.append(password_hash)
        return real_verify(password_hash, password)

    monkeypatch.setattr(password_manager, "verify_password", counting_verify)
    with pytest.raises(AuthenticationError):
        login("ghost", DEFAULT_PASSWORD)
    assert len(calls) == 1
    assert calls[0].startswith("$argon2id$")
    with session_scope() as db:
        assert calls[0] not in db.execute(select(User.password_hash)).scalars().all()

    with pytest.raises(AuthenticationError):
        login("alice", "Wr0ng!Password")
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Lockout
# ---------------------------------------------------------------------------


def fail(username: str, times: int) -> None:
    for _ in range(times):
        with pytest.raises(AuthenticationError) as excinfo:
            login(username, "Wr0ng!Password")
        assert type(excinfo.value) is AuthenticationError


def test_lockout_after_max_failures_rejects_correct_password(lockout_settings, factory, org_id, clock):
    user_id = make_user(factory, org_id)
    fail("alice", 3)
    with session_scope() as db:
        user = db.get(User, user_id)
        assert user.failed_login_count == 3
        assert user.locked_until == START + timedelta(minutes=15)

    with pytest.raises(AccountLockedError) as excinfo:
        login("alice", DEFAULT_PASSWORD)
    assert "Try again in 15 minutes" in excinfo.value.user_message
    assert attempts("alice")[-1] == (False, "locked")


def test_attempt_before_threshold_is_still_generic(lockout_settings, factory, org_id, clock):
    make_user(factory, org_id)
    fail("alice", 2)
    assert login("alice", DEFAULT_PASSWORD).token


def test_lockout_applies_to_unknown_usernames(lockout_settings, clock):
    fail("ghost", 3)
    with pytest.raises(AccountLockedError):
        login("GHOST", "Wr0ng!Password")
    assert attempts("ghost") == [(False, "unknown_user")] * 3 + [(False, "locked")]


def test_lockout_expires_after_lockout_minutes(lockout_settings, factory, org_id, clock):
    user_id = make_user(factory, org_id)
    fail("alice", 3)
    clock.advance(minutes=14, seconds=59)
    with pytest.raises(AccountLockedError) as excinfo:
        login("alice", DEFAULT_PASSWORD)
    assert "Try again in 1 minute." in excinfo.value.user_message

    clock.advance(seconds=1)
    result = login("alice", DEFAULT_PASSWORD)
    assert result.ctx.user_id == user_id
    with session_scope() as db:
        user = db.get(User, user_id)
        assert user.failed_login_count == 0
        assert user.locked_until is None


def test_attempts_while_locked_do_not_extend_lock(lockout_settings, factory, org_id, clock):
    make_user(factory, org_id)
    fail("alice", 3)
    clock.advance(minutes=10)
    with pytest.raises(AccountLockedError):
        login("alice", "Wr0ng!Password")
    clock.advance(minutes=5)
    assert login("alice", DEFAULT_PASSWORD).token


def test_failures_spread_beyond_window_do_not_lock(lockout_settings, factory, org_id, clock):
    user_id = make_user(factory, org_id)
    fail("alice", 2)
    clock.advance(minutes=16)
    fail("alice", 1)
    with session_scope() as db:
        assert db.get(User, user_id).failed_login_count == 1
    assert login("alice", DEFAULT_PASSWORD).token


def test_success_resets_failure_counters(lockout_settings, factory, org_id, clock):
    user_id = make_user(factory, org_id)
    fail("alice", 2)
    with session_scope() as db:
        assert db.get(User, user_id).failed_login_count == 2
    login("alice", DEFAULT_PASSWORD)
    with session_scope() as db:
        assert db.get(User, user_id).failed_login_count == 0
    fail("alice", 2)
    assert login("alice", DEFAULT_PASSWORD).token
    fail("alice", 3)
    with pytest.raises(AccountLockedError):
        login("alice", DEFAULT_PASSWORD)


def test_admin_set_locked_until_is_honoured(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    set_user(user_id, locked_until=START + timedelta(minutes=5))
    with pytest.raises(AccountLockedError) as excinfo:
        login("alice", DEFAULT_PASSWORD)
    assert "Try again in 5 minutes" in excinfo.value.user_message


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_validate_session_returns_fresh_context(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    result = login("alice", DEFAULT_PASSWORD)
    ctx = validate_session(result.token)
    assert ctx == result.ctx
    assert ctx.user_id == user_id and ctx.session_id == result.ctx.session_id


def test_touch_is_throttled_to_once_per_minute(factory, org_id, clock):
    make_user(factory, org_id)
    token = login("alice", DEFAULT_PASSWORD).token
    clock.advance(seconds=59)
    validate_session(token)
    assert session_row(token)["last_seen_at"] == START
    clock.advance(seconds=1)
    validate_session(token, touch=False)
    assert session_row(token)["last_seen_at"] == START
    validate_session(token)
    assert session_row(token)["last_seen_at"] == START + timedelta(seconds=60)


def test_idle_timeout_boundary_and_expiry(factory, org_id, clock):
    make_user(factory, org_id)
    token = login("alice", DEFAULT_PASSWORD).token
    clock.advance(minutes=30)
    validate_session(token)
    clock.advance(minutes=30, seconds=1)
    with pytest.raises(SessionExpiredError):
        validate_session(token)
    row = session_row(token)
    assert row["revoked_at"] == clock.now
    assert row["revoke_reason"] == "idle"
    clock.advance(seconds=1)
    with pytest.raises(SessionExpiredError):
        validate_session(token)
    assert session_row(token)["revoke_reason"] == "idle"


def test_absolute_timeout_ends_an_active_session(factory, org_id, clock):
    make_user(factory, org_id)
    result = login("alice", DEFAULT_PASSWORD)
    while clock.now + timedelta(minutes=20) < result.expires_at:
        clock.advance(minutes=20)
        validate_session(result.token)
    clock.now = result.expires_at
    with pytest.raises(SessionExpiredError):
        validate_session(result.token)
    assert session_row(result.token)["revoke_reason"] == "expired"


@pytest.mark.parametrize("token", ["", None, "not-a-token", "x" * 5000, 12345])
def test_unknown_or_malformed_tokens_are_rejected(factory, org_id, clock, token):
    make_user(factory, org_id)
    login("alice", DEFAULT_PASSWORD)
    with pytest.raises(SessionExpiredError):
        validate_session(token)


def test_tampered_token_is_rejected(factory, org_id, clock):
    make_user(factory, org_id)
    token = login("alice", DEFAULT_PASSWORD).token
    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    with pytest.raises(SessionExpiredError):
        validate_session(tampered)
    with pytest.raises(SessionExpiredError):
        validate_session(hashlib.sha256(token.encode()).hexdigest())


def test_logout_revokes_and_is_idempotent(factory, org_id, clock):
    make_user(factory, org_id)
    token = login("alice", DEFAULT_PASSWORD).token
    logout(token)
    assert session_row(token)["revoke_reason"] == "logout"
    with pytest.raises(SessionExpiredError):
        validate_session(token)
    logout(token)
    logout("unknown-token")
    logout("")
    assert len(audit_rows("auth.logout")) == 1


def test_role_change_after_login_is_reflected(factory, org_id, clock):
    user_id = make_user(factory, org_id, role="org_admin")
    token = login("alice", DEFAULT_PASSWORD).token
    set_user(user_id, role="analyst")
    ctx = validate_session(token)
    assert ctx.role == Role.ANALYST
    assert not ctx.is_org_admin


def test_user_deactivation_ends_session(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    token = login("alice", DEFAULT_PASSWORD).token
    set_user(user_id, is_active=False)
    with pytest.raises(AuthenticationError) as excinfo:
        validate_session(token)
    assert type(excinfo.value) is AuthenticationError
    assert excinfo.value.user_message == "Your account is not active."
    assert session_row(token)["revoke_reason"] == "user_inactive"
    set_user(user_id, is_active=True)
    with pytest.raises(SessionExpiredError):
        validate_session(token)


def test_org_suspension_ends_session(factory, org_id, clock):
    make_user(factory, org_id)
    token = login("alice", DEFAULT_PASSWORD).token
    set_org_status(org_id, "suspended")
    with pytest.raises(PermissionDeniedError):
        validate_session(token)
    assert session_row(token)["revoke_reason"] == "org_inactive"
    set_org_status(org_id, "active")
    with pytest.raises(SessionExpiredError):
        validate_session(token)


def test_sessions_of_other_tenants_are_independent(factory, clock):
    org_a, org_b = factory.organization(), factory.organization()
    make_user(factory, org_a, username="anna")
    make_user(factory, org_b, username="bob")
    token_a = login("anna", DEFAULT_PASSWORD).token
    token_b = login("bob", DEFAULT_PASSWORD).token
    set_org_status(org_b, "suspended")
    assert validate_session(token_a).organization_id == org_a
    with pytest.raises(PermissionDeniedError):
        validate_session(token_b)


# ---------------------------------------------------------------------------
# Password change, session listing and maintenance
# ---------------------------------------------------------------------------


def test_password_change_revokes_other_sessions_only(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    set_user(user_id, must_change_password=True)
    current = login("alice", DEFAULT_PASSWORD, user_agent="laptop")
    other = login("alice", DEFAULT_PASSWORD, user_agent="phone")
    clock.advance(minutes=1)

    change_password(current.ctx, DEFAULT_PASSWORD, NEW_PASSWORD)

    assert validate_session(current.token).user_id == user_id
    with pytest.raises(SessionExpiredError):
        validate_session(other.token)
    assert session_row(other.token)["revoke_reason"] == "password_changed"
    with session_scope() as db:
        user = db.get(User, user_id)
        assert user.must_change_password is False
        assert user.password_changed_at == clock.now
    with pytest.raises(AuthenticationError):
        login("alice", DEFAULT_PASSWORD)
    assert login("alice", NEW_PASSWORD).token
    [entry] = audit_rows("auth.password_change")
    assert json.loads(entry.details_json)["other_sessions_revoked"] == 1


def test_password_change_validation(factory, org_id, clock):
    make_user(factory, org_id)
    ctx = login("alice", DEFAULT_PASSWORD).ctx
    with pytest.raises(AuthenticationError) as wrong:
        change_password(ctx, "Wr0ng!Password", NEW_PASSWORD)
    assert "current password is incorrect" in wrong.value.user_message
    with pytest.raises(ValidationError):
        change_password(ctx, DEFAULT_PASSWORD, "weak")
    with pytest.raises(ValidationError) as same:
        change_password(ctx, DEFAULT_PASSWORD, DEFAULT_PASSWORD)
    assert "different" in same.value.user_message
    with pytest.raises(ValidationError):
        change_password(ctx, DEFAULT_PASSWORD, "Alice!2024xyz")
    assert login("alice", DEFAULT_PASSWORD).token


def test_password_change_rejects_forged_context(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    other_org = factory.organization()
    forged = AuthContext(user_id=user_id, role=Role.ORG_ADMIN, organization_id=other_org, username="alice")
    with pytest.raises(AuthenticationError):
        change_password(forged, DEFAULT_PASSWORD, NEW_PASSWORD)
    with pytest.raises(AuthenticationError):
        active_sessions(forged)
    with pytest.raises(AuthenticationError):
        change_password(AuthContext.system(org_id), DEFAULT_PASSWORD, NEW_PASSWORD)


def test_active_sessions_lists_only_live_sessions_of_caller(factory, org_id, clock):
    make_user(factory, org_id)
    make_user(factory, org_id, username="bob")
    first = login("alice", DEFAULT_PASSWORD, user_agent="laptop")
    clock.advance(seconds=1)
    second = login("alice", DEFAULT_PASSWORD, user_agent="phone")
    gone = login("alice", DEFAULT_PASSWORD)
    login("bob", DEFAULT_PASSWORD)
    logout(gone.token)

    listed = active_sessions(second.ctx)
    assert [item["id"] for item in listed] == [second.ctx.session_id, first.ctx.session_id]
    assert [item["is_current"] for item in listed] == [True, False]
    assert listed[0]["user_agent"] == "phone"
    assert set(listed[0]) == {"id", "created_at", "last_seen_at", "expires_at", "user_agent", "is_current"}

    clock.advance(minutes=31)
    assert active_sessions(second.ctx) == []


def test_revoke_user_sessions_helper(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    keep = login("alice", DEFAULT_PASSWORD)
    login("alice", DEFAULT_PASSWORD)
    login("alice", DEFAULT_PASSWORD)
    with session_scope() as db:
        assert revoke_user_sessions(db, user_id, "admin_reset", except_session_id=keep.ctx.session_id) == 2
    with session_scope() as db:
        assert revoke_user_sessions(db, user_id, "admin_reset", except_session_id=keep.ctx.session_id) == 0
        assert revoke_user_sessions(db, user_id, "user_deactivated") == 1
    with pytest.raises(SessionExpiredError):
        validate_session(keep.token)
    assert session_row(keep.token)["revoke_reason"] == "user_deactivated"


def test_purge_old_sessions_deletes_only_dead_old_rows(factory, org_id, clock):
    make_user(factory, org_id)
    old_revoked = login("alice", DEFAULT_PASSWORD).token
    logout(old_revoked)
    old_expired = login("alice", DEFAULT_PASSWORD).token
    with pytest.raises(AuthenticationError):
        login("alice", "Wr0ng!Password")
    clock.advance(days=31)
    fresh = login("alice", DEFAULT_PASSWORD).token

    counts = purge_old_sessions(older_than_days=30)

    assert counts == {"sessions_deleted": 2, "login_attempts_deleted": 3}
    assert validate_session(fresh).username == "alice"
    with session_scope() as db:
        remaining = db.execute(select(UserSession.token_hash)).scalars().all()
    assert remaining == [hashlib.sha256(fresh.encode()).hexdigest()]
    assert hashlib.sha256(old_expired.encode()).hexdigest() not in remaining
    assert purge_old_sessions() == {"sessions_deleted": 0, "login_attempts_deleted": 0}


@pytest.mark.parametrize("days", [0, -1, True, "30"])
def test_purge_old_sessions_rejects_invalid_retention(days):
    with pytest.raises(ValidationError):
        purge_old_sessions(older_than_days=days)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def test_audit_rows_for_login_failure_and_logout_never_contain_secrets(factory, org_id, clock):
    user_id = make_user(factory, org_id)
    with pytest.raises(AuthenticationError):
        login("alice", "Wr0ng!Password")
    with pytest.raises(AuthenticationError):
        login("ghost", "Gh0st!Secret")
    result = login("alice", DEFAULT_PASSWORD)
    logout(result.token)

    [login_row] = audit_rows("auth.login")
    assert login_row.actor_user_id == user_id and login_row.organization_id == org_id
    assert json.loads(login_row.details_json)["username"] == "alice"

    failed = [json.loads(row.details_json) for row in audit_rows("auth.login_failed")]
    assert failed == [
        {"username": "alice", "reason": "bad_password"},
        {"username": "ghost", "reason": "unknown_user"},
    ]
    assert audit_rows("auth.login_failed")[0].organization_id == org_id
    [logout_row] = audit_rows("auth.logout")
    assert logout_row.actor_user_id == user_id

    with session_scope() as db:
        everything = " ".join(db.execute(select(AuditLog.details_json)).scalars())
    for secret in (DEFAULT_PASSWORD, "Wr0ng!Password", "Gh0st!Secret", result.token):
        assert secret not in everything
    assert hashlib.sha256(result.token.encode()).hexdigest() not in everything


@pytest.mark.parametrize(("raw", "expected"), [("  Alice ", "alice"), (None, ""), ("A" * 100, "a" * 64)])
def test_normalize_username(raw, expected):
    assert authentication.normalize_username(raw) == expected


def test_hash_token_is_sha256_hex():
    assert authentication.hash_token("abc") == hashlib.sha256(b"abc").hexdigest()


def test_dummy_hash_is_cached_per_parameter_set(app_settings):
    first = authentication._dummy_hash()
    assert authentication._dummy_hash() == first
    set_settings(app_settings.with_overrides(password_hash_time_cost=2))
    second = authentication._dummy_hash()
    assert second != first
    assert not password_manager.needs_rehash(second)
