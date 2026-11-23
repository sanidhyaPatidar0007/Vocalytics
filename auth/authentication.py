"""Authentication: login with lockout, server-side sessions, logout and password changes.

Design notes:

* Only the SHA-256 hash of a session token is stored. The raw token is
  returned once by ``login`` and never persisted or logged.
* Login failures share one generic message so the response never reveals
  whether a username exists. Unknown usernames still pay for an Argon2
  verification against a dummy hash so timing stays similar.
* Lockout is derived from ``login_attempts`` rows (so it also applies to
  unknown usernames) and mirrored on ``users.failed_login_count`` /
  ``users.locked_until`` for existing accounts.
* ``session_scope`` rolls back when an exception escapes it, so every
  "record, then fail" path decides the outcome inside the transaction, lets
  it commit and raises afterwards.
* The clock is always read through ``timeutil.utcnow()`` so tests can patch it.
"""

from __future__ import annotations

import functools
import hashlib
import math
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import Select, delete, func, or_, select, update
from sqlalchemy.orm import Session

from auth import password_manager
from auth.authorization import AuthContext
from config.settings import get_settings
from core import timeutil
from core.constants import OrgStatus, Role
from core.errors import (
    AccountLockedError,
    AppError,
    AuthenticationError,
    PermissionDeniedError,
    SessionExpiredError,
    ValidationError,
)
from database.connection import session_scope
from database.models import LoginAttempt, Organization, User, UserSession
from services.audit_service import record_audit

GENERIC_LOGIN_FAILURE = "Invalid username or password."
ORG_NOT_ACTIVE_MESSAGE = "Your organization is not active. Contact the platform administrator."
ACCOUNT_NOT_ACTIVE_MESSAGE = "Your account is not active."

#: Minimum interval between ``last_seen_at`` writes for one session.
TOUCH_INTERVAL = timedelta(seconds=60)

USERNAME_MAX_LENGTH = 64
USER_AGENT_MAX_LENGTH = 300
TOKEN_MAX_LENGTH = 512

REASON_BAD_PASSWORD = "bad_password"  # noqa: S105 - a login-attempt reason code, not a secret
REASON_UNKNOWN_USER = "unknown_user"
REASON_INACTIVE = "inactive"
REASON_LOCKED = "locked"
REASON_ORG_INACTIVE = "org_inactive"

#: Attempt reasons that count toward the lockout threshold. Attempts rejected
#: because the account was already locked do not extend the lock.
_COUNTED_FAILURES = (REASON_BAD_PASSWORD, REASON_UNKNOWN_USER, REASON_INACTIVE)


@dataclass(frozen=True)
class LoginResult:
    """Outcome of a successful login. ``token`` is shown to the caller once and never stored."""

    token: str
    ctx: AuthContext
    expires_at: datetime
    must_change_password: bool


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def hash_token(token: str) -> str:
    """Return the SHA-256 hex digest under which a session token is stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def normalize_username(username: object) -> str:
    """Strip and lowercase a username; non-strings normalize to an empty string."""
    if not isinstance(username, str):
        return ""
    return username.strip().lower()[:USERNAME_MAX_LENGTH]


@functools.lru_cache(maxsize=8)
def _dummy_hash_for(time_cost: int, memory_kib: int, parallelism: int) -> str:
    """Argon2 hash of a random secret, computed once per parameter set."""
    return password_manager.hash_password(secrets.token_urlsafe(24))


def _dummy_hash() -> str:
    """Dummy hash with the current Argon2 cost, so unknown users take similar time."""
    s = get_settings()
    return _dummy_hash_for(s.password_hash_time_cost, s.password_hash_memory_kib, s.password_hash_parallelism)


def _context_for(user: User, session_id: int | None) -> AuthContext:
    return AuthContext(
        user_id=user.id,
        role=Role(user.role),
        organization_id=user.organization_id,
        username=user.username,
        session_id=session_id,
    )


def _org_is_active(db: Session, organization_id: int | None) -> bool:
    """Super Admins (no organization) are never blocked by organization status."""
    if organization_id is None:
        return True
    org = db.get(Organization, organization_id)
    return org is not None and org.status == OrgStatus.ACTIVE.value


def _minutes_left(until: datetime, now: datetime) -> int:
    return max(1, math.ceil((until - now).total_seconds() / 60))


# ---------------------------------------------------------------------------
# Lockout
# ---------------------------------------------------------------------------


def _counted_failures_query(db: Session, username: str, since: datetime) -> Select:
    """Counted failures for ``username`` after ``since`` and after the last successful login.

    "After the last success" compares row ids, not timestamps, so attempts that
    share a timestamp with the success are still ordered correctly.
    """
    last_success_id = db.execute(
        select(func.max(LoginAttempt.id)).where(
            LoginAttempt.username == username, LoginAttempt.success.is_(True)
        )
    ).scalar_one_or_none()
    query = select(LoginAttempt.created_at).where(
        LoginAttempt.username == username,
        LoginAttempt.success.is_(False),
        LoginAttempt.reason.in_(_COUNTED_FAILURES),
        LoginAttempt.created_at >= since,
    )
    if last_success_id is not None:
        query = query.where(LoginAttempt.id > last_success_id)
    return query


def _lock_expiry(db: Session, username: str, now: datetime) -> datetime | None:
    """Return when the lock on ``username`` ends, or None when it is not locked.

    Failures before the most recent successful login are ignored. The lock is
    anchored on the latest counted failure F: it applies when at least
    LOGIN_MAX_FAILED_ATTEMPTS failures fall in ``[F - window, F]`` and lasts
    until ``F + lockout``.
    """
    s = get_settings()
    window = timedelta(minutes=s.login_attempt_window_minutes)
    lockout = timedelta(minutes=s.login_lockout_minutes)
    query = _counted_failures_query(db, username, now - window - lockout)
    recent = (
        db.execute(query.order_by(LoginAttempt.created_at.desc()).limit(s.login_max_failed_attempts))
        .scalars()
        .all()
    )
    if len(recent) < s.login_max_failed_attempts:
        return None
    newest, oldest = recent[0], recent[-1]
    if newest - oldest > window:
        return None
    until = newest + lockout
    return until if now < until else None


def _recent_failure_count(db: Session, username: str, now: datetime) -> int:
    """Counted failures inside the attempt window since the last success."""
    window_start = now - timedelta(minutes=get_settings().login_attempt_window_minutes)
    return len(db.execute(_counted_failures_query(db, username, window_start)).scalars().all())


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def _record_failure(db: Session, username: str, user: User | None, reason: str, now: datetime) -> None:
    """Store a failed attempt, update the user's counters and audit it (never the password)."""
    db.add(
        LoginAttempt(
            username=username, user_id=user.id if user else None, success=False, reason=reason, created_at=now
        )
    )
    db.flush()
    if user is not None and reason in _COUNTED_FAILURES:
        user.failed_login_count = _recent_failure_count(db, username, now)
        user.locked_until = _lock_expiry(db, username, now)
    record_audit(
        db,
        None,
        "auth.login_failed",
        target_type="user" if user else None,
        target_id=user.id if user else None,
        details={"username": username, "reason": reason},
        organization_id=user.organization_id if user else None,
    )


def _create_session(
    db: Session, user: User, user_agent: str | None, now: datetime
) -> tuple[str, UserSession]:
    """Create a session row and return the raw token with it (only the hash is stored)."""
    token = secrets.token_urlsafe(32)
    agent = user_agent[:USER_AGENT_MAX_LENGTH] if isinstance(user_agent, str) and user_agent else None
    row = UserSession(
        user_id=user.id,
        organization_id=user.organization_id,
        token_hash=hash_token(token),
        created_at=now,
        last_seen_at=now,
        expires_at=now + timedelta(hours=get_settings().session_absolute_timeout_hours),
        user_agent=agent,
    )
    db.add(row)
    db.flush()
    return token, row


def _complete_login(
    db: Session, user: User, username: str, password: str, user_agent: str | None, now: datetime
) -> LoginResult:
    """Success path: session, counter reset, optional rehash, attempt row and audit."""
    token, row = _create_session(db, user, user_agent, now)
    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = now
    if password_manager.needs_rehash(user.password_hash):
        user.password_hash = password_manager.hash_password(password)
    db.add(LoginAttempt(username=username, user_id=user.id, success=True, reason=None, created_at=now))
    ctx = _context_for(user, row.id)
    record_audit(
        db,
        ctx,
        "auth.login",
        target_type="user",
        target_id=user.id,
        details={"username": username, "session_id": row.id},
    )
    return LoginResult(
        token=token, ctx=ctx, expires_at=row.expires_at, must_change_password=bool(user.must_change_password)
    )


def login(username: str, password: str, *, user_agent: str | None = None) -> LoginResult:
    """Authenticate a user and open a new session.

    Raises ``AuthenticationError`` with one generic message for an unknown
    user, a wrong password or an inactive account; ``AccountLockedError``
    while the username is locked out (even with the correct password); and
    ``PermissionDeniedError`` when the password is correct but the user's
    organization is not active. Every attempt is recorded and audited.
    """
    name = normalize_username(username)
    secret = password if isinstance(password, str) else ""
    error: AppError | None = None
    result: LoginResult | None = None
    with session_scope() as db:
        now = timeutil.utcnow()
        user = db.execute(select(User).where(User.username == name)).scalar_one_or_none() if name else None
        lock_until = _lock_expiry(db, name, now)
        if user is not None and user.locked_until is not None and user.locked_until > now:
            lock_until = max(lock_until or user.locked_until, user.locked_until)
        if lock_until is not None:
            _record_failure(db, name, user, REASON_LOCKED, now)
            minutes = _minutes_left(lock_until, now)
            error = AccountLockedError(
                f"Too many failed sign-in attempts. Try again in {minutes} minute{'' if minutes == 1 else 's'}."
            )
        elif user is None:
            password_manager.verify_password(_dummy_hash(), secret)
            _record_failure(db, name, None, REASON_UNKNOWN_USER, now)
            error = AuthenticationError(GENERIC_LOGIN_FAILURE)
        elif not password_manager.verify_password(user.password_hash, secret):
            _record_failure(db, name, user, REASON_BAD_PASSWORD, now)
            error = AuthenticationError(GENERIC_LOGIN_FAILURE)
        elif not user.is_active:
            _record_failure(db, name, user, REASON_INACTIVE, now)
            error = AuthenticationError(GENERIC_LOGIN_FAILURE)
        elif not _org_is_active(db, user.organization_id):
            _record_failure(db, name, user, REASON_ORG_INACTIVE, now)
            error = PermissionDeniedError(ORG_NOT_ACTIVE_MESSAGE)
        else:
            result = _complete_login(db, user, name, secret, user_agent, now)
    if error is not None:
        raise error
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def _revoke(row: UserSession, reason: str, now: datetime) -> None:
    row.revoked_at = now
    row.revoke_reason = reason


def _find_session(db: Session, token: object) -> UserSession | None:
    if not isinstance(token, str) or not token or len(token) > TOKEN_MAX_LENGTH:
        return None
    return db.execute(
        select(UserSession).where(UserSession.token_hash == hash_token(token))
    ).scalar_one_or_none()


def _check_session(
    db: Session, row: UserSession | None, now: datetime
) -> tuple[AppError | None, User | None]:
    """Decide whether a session is usable, revoking it when it is not."""
    if row is None or row.revoked_at is not None:
        return SessionExpiredError(), None
    if now >= row.expires_at:
        _revoke(row, "expired", now)
        return SessionExpiredError(), None
    if now - row.last_seen_at > timedelta(minutes=get_settings().session_idle_timeout_minutes):
        _revoke(row, "idle", now)
        return SessionExpiredError(), None
    user = db.get(User, row.user_id)
    if user is None or not user.is_active:
        _revoke(row, "user_inactive", now)
        return AuthenticationError(ACCOUNT_NOT_ACTIVE_MESSAGE), None
    if not _org_is_active(db, user.organization_id):
        _revoke(row, "org_inactive", now)
        return PermissionDeniedError(ORG_NOT_ACTIVE_MESSAGE), None
    return None, user


def validate_session(token: str, *, touch: bool = True) -> AuthContext:
    """Resolve a session token to a fresh ``AuthContext`` built from the current user row.

    Unknown, empty, revoked, expired (absolute) and idle sessions raise
    ``SessionExpiredError``; a deactivated user raises ``AuthenticationError``
    and a non-active organization raises ``PermissionDeniedError``. Dead
    sessions are revoked before the error is raised. With ``touch`` the
    session's ``last_seen_at`` is refreshed at most once per minute.
    """
    error: AppError | None = None
    ctx: AuthContext | None = None
    with session_scope() as db:
        now = timeutil.utcnow()
        row = _find_session(db, token)
        error, user = _check_session(db, row, now)
        if error is None and user is not None and row is not None:
            if touch and now - row.last_seen_at >= TOUCH_INTERVAL:
                row.last_seen_at = now
            ctx = _context_for(user, row.id)
    if error is not None:
        raise error
    assert ctx is not None
    return ctx


def logout(token: str) -> None:
    """Revoke the session behind ``token``. Silent and idempotent for unknown or ended sessions."""
    with session_scope() as db:
        row = _find_session(db, token)
        if row is None or row.revoked_at is not None:
            return
        _revoke(row, "logout", timeutil.utcnow())
        user = db.get(User, row.user_id)
        actor = _context_for(user, row.id) if user is not None else None
        record_audit(
            db,
            actor,
            "auth.logout",
            target_type="session",
            target_id=row.id,
            details={"username": user.username if user else None},
            organization_id=row.organization_id,
        )


def revoke_user_sessions(
    db: Session, user_id: int, reason: str, *, except_session_id: int | None = None
) -> int:
    """Revoke every active session of a user inside the caller's transaction; returns the count."""
    statement = (
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
        .values(revoked_at=timeutil.utcnow(), revoke_reason=str(reason)[:50])
        .execution_options(synchronize_session=False)
    )
    if except_session_id is not None:
        statement = statement.where(UserSession.id != except_session_id)
    return int(db.execute(statement).rowcount or 0)


def _active_user(db: Session, ctx: AuthContext) -> User:
    """Return the caller's user row if it is active and still matches the context."""
    if not isinstance(ctx, AuthContext) or ctx.user_id is None:
        raise AuthenticationError("Please sign in.")
    user = db.get(User, ctx.user_id)
    if user is None or not user.is_active:
        raise AuthenticationError(ACCOUNT_NOT_ACTIVE_MESSAGE)
    if user.role != ctx.role.value or user.organization_id != ctx.organization_id:
        raise AuthenticationError("Your session is no longer valid. Please sign in again.")
    if not _org_is_active(db, user.organization_id):
        raise PermissionDeniedError(ORG_NOT_ACTIVE_MESSAGE)
    return user


def change_password(ctx: AuthContext, current_password: str, new_password: str) -> None:
    """Change the caller's password after verifying the current one.

    The new password must satisfy the password policy and differ from the
    current one. All of the user's other sessions are revoked; the session in
    ``ctx.session_id`` (if any) stays valid.
    """
    with session_scope() as db:
        user = _active_user(db, ctx)
        if not password_manager.verify_password(user.password_hash, current_password):
            raise AuthenticationError("The current password is incorrect.")
        password_manager.validate_password_strength(new_password, username=user.username)
        if new_password == current_password:
            raise ValidationError("The new password must be different from the current password.")
        now = timeutil.utcnow()
        user.password_hash = password_manager.hash_password(new_password)
        user.password_changed_at = now
        user.must_change_password = False
        revoked = revoke_user_sessions(db, user.id, "password_changed", except_session_id=ctx.session_id)
        record_audit(
            db,
            ctx,
            "auth.password_change",
            target_type="user",
            target_id=user.id,
            details={"username": user.username, "other_sessions_revoked": revoked},
        )


def active_sessions(ctx: AuthContext) -> list[dict]:
    """List the caller's live sessions (not revoked, not expired, not idle), newest activity first."""
    with session_scope() as db:
        user = _active_user(db, ctx)
        now = timeutil.utcnow()
        idle_cutoff = now - timedelta(minutes=get_settings().session_idle_timeout_minutes)
        rows = (
            db.execute(
                select(UserSession)
                .where(
                    UserSession.user_id == user.id,
                    UserSession.revoked_at.is_(None),
                    UserSession.expires_at > now,
                    UserSession.last_seen_at >= idle_cutoff,
                )
                .order_by(UserSession.last_seen_at.desc(), UserSession.id.desc())
            )
            .scalars()
            .all()
        )
        return [
            {
                "id": row.id,
                "created_at": row.created_at,
                "last_seen_at": row.last_seen_at,
                "expires_at": row.expires_at,
                "user_agent": row.user_agent,
                "is_current": row.id == ctx.session_id,
            }
            for row in rows
        ]


def purge_old_sessions(*, older_than_days: int = 30) -> dict:
    """Delete long-dead sessions and old login attempts (maintenance task).

    A session is deleted when it was revoked, expired or last seen before the
    cutoff. Login attempts older than the cutoff are deleted too. Returns
    ``{"sessions_deleted": n, "login_attempts_deleted": m}``.
    """
    if not isinstance(older_than_days, int) or isinstance(older_than_days, bool) or older_than_days < 1:
        raise ValidationError("The retention period must be at least one day.")
    cutoff = timeutil.utcnow() - timedelta(days=older_than_days)
    with session_scope() as db:
        sessions = db.execute(
            delete(UserSession)
            .where(
                or_(
                    UserSession.revoked_at < cutoff,
                    UserSession.expires_at < cutoff,
                    UserSession.last_seen_at < cutoff,
                )
            )
            .execution_options(synchronize_session=False)
        ).rowcount
        attempts = db.execute(
            delete(LoginAttempt)
            .where(LoginAttempt.created_at < cutoff)
            .execution_options(synchronize_session=False)
        ).rowcount
    return {"sessions_deleted": int(sessions or 0), "login_attempts_deleted": int(attempts or 0)}
