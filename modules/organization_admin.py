"""Organization Admin workflows (specification section 6).

Every public function takes the caller's ``AuthContext`` first, opens its own
``session_scope()`` and calls ``authorize(db, ctx, roles=(Role.ORG_ADMIN,))``
before touching any row. The organization always comes from the authorized
context, and every lookup of a user, dataset or history row is filtered by
that organization, so an id that belongs to another tenant raises
``NotFoundError`` exactly like an id that does not exist.

The user helpers at the top of the module (username/e-mail validation, user
creation and session revocation) are shared with ``modules.super_admin``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from auth.authorization import AuthContext, authorize, authorize_dataset
from auth.password_manager import hash_password, validate_password_strength
from core import timeutil
from core.constants import (
    ANALYST_DEFAULT_PERMISSIONS,
    DatasetStatus,
    Permission,
    Role,
    UsageMetric,
)
from core.errors import ConflictError, NotFoundError, ValidationError
from database.connection import session_scope
from database.models import (
    AnalysisHistory,
    AnomalyEvent,
    CleaningOperation,
    Dataset,
    DatasetAccess,
    DatasetVersion,
    Organization,
    UsageRecord,
    User,
    UserPermission,
    UserSession,
)
from modules import org_settings, subscriptions, usage_tracking
from services.audit_service import record_audit

USERNAME_PATTERN = re.compile(r"^[a-z0-9._-]{3,64}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_EMAIL_LENGTH = 254
MAX_FULL_NAME_LENGTH = 200
ALL_PERMISSIONS = tuple(sorted(p.value for p in Permission))
RECENT_OPERATIONS_LIMIT = 10
ANOMALY_WINDOW_DAYS = 7
MAX_USAGE_MONTHS = 36
MAX_HISTORY_LIMIT = 1000


# ---------------------------------------------------------------------------
# Shared user helpers (also used by modules.super_admin)
# ---------------------------------------------------------------------------


def normalize_username(username: str) -> str:
    """Lowercase and validate a username: 3 to 64 characters of ``[a-z0-9._-]``."""
    if not isinstance(username, str):
        raise ValidationError("A username is required.")
    clean = username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(clean):
        raise ValidationError(
            "Usernames must be 3 to 64 characters using letters, digits, dots, underscores or hyphens."
        )
    return clean


def normalize_email(email: str) -> str:
    """Strip and validate an e-mail address with a basic ``name@domain.tld`` check."""
    if not isinstance(email, str):
        raise ValidationError("An e-mail address is required.")
    clean = email.strip()
    if len(clean) > MAX_EMAIL_LENGTH or not EMAIL_PATTERN.fullmatch(clean):
        raise ValidationError("Enter a valid e-mail address.")
    return clean


def normalize_full_name(full_name: str | None) -> str:
    """Strip a display name and enforce its maximum length."""
    clean = (full_name or "").strip()
    if len(clean) > MAX_FULL_NAME_LENGTH:
        raise ValidationError(f"The full name must be at most {MAX_FULL_NAME_LENGTH} characters.")
    return clean


def validate_permissions(permissions: set[str] | frozenset[str] | list[str] | tuple[str, ...]) -> set[str]:
    """Return the permission keys as plain strings, rejecting unknown keys."""
    if isinstance(permissions, (str, bytes)) or not isinstance(permissions, (set, frozenset, list, tuple)):
        raise ValidationError("Permissions must be a set of permission names.")
    keys = {p.value if isinstance(p, Permission) else str(p) for p in permissions}
    unknown = keys - set(ALL_PERMISSIONS)
    if unknown:
        raise ValidationError(f"Unknown permission(s): {', '.join(sorted(unknown))}.")
    return keys


def ensure_username_available(db: Session, username: str) -> None:
    """Usernames are unique across the whole platform."""
    taken = db.execute(select(User.id).where(func.lower(User.username) == username)).first()
    if taken:
        raise ConflictError("This username is already taken.")


def create_user_row(
    db: Session,
    *,
    organization_id: int,
    role: Role,
    username: str,
    email: str,
    password: str,
    full_name: str,
    created_by_id: int | None,
) -> User:
    """Validate and insert a user of ``organization_id`` (quota enforced).

    Validation order: username format and uniqueness, e-mail, name, password
    strength (the password may not contain the username), then the plan's user
    quota. The row is flushed so its id is available to the caller.

    The password was chosen by an administrator, so the account must change it
    at first sign-in (``must_change_password``); until then the UI offers only
    the Account page.
    """
    clean_username = normalize_username(username)
    ensure_username_available(db, clean_username)
    clean_email = normalize_email(email)
    clean_name = normalize_full_name(full_name)
    validate_password_strength(password, username=clean_username)
    subscriptions.enforce_user_quota(db, organization_id)
    user = User(
        organization_id=organization_id,
        username=clean_username,
        email=clean_email,
        full_name=clean_name,
        role=role.value,
        password_hash=hash_password(password),
        is_active=True,
        must_change_password=True,
        created_by_id=created_by_id,
    )
    db.add(user)
    db.flush()
    return user


def revoke_sessions(db: Session, user_ids: list[int], reason: str) -> int:
    """Revoke every open session of ``user_ids``. Returns how many were revoked."""
    if not user_ids:
        return 0
    result = db.execute(
        update(UserSession)
        .where(UserSession.user_id.in_(user_ids), UserSession.revoked_at.is_(None))
        .values(revoked_at=timeutil.utcnow(), revoke_reason=reason[:50])
    )
    return int(result.rowcount or 0)


def user_to_dict(user: User, permissions: list[str]) -> dict[str, Any]:
    """Plain representation of a user row (never the password hash)."""
    return {
        "id": user.id,
        "organization_id": user.organization_id,
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": user.is_active,
        "permissions": permissions,
        "must_change_password": user.must_change_password,
        "last_login_at": user.last_login_at,
        "created_at": user.created_at,
    }


# ---------------------------------------------------------------------------
# Internal lookups (always scoped to the caller's organization)
# ---------------------------------------------------------------------------


def _authorize_admin(db: Session, ctx: AuthContext) -> int:
    return authorize(db, ctx, roles=(Role.ORG_ADMIN,))


def _validate_id(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"Invalid {label}.")
    return value


def _org_user(db: Session, org_id: int, user_id: int) -> User:
    _validate_id(user_id, "user")
    user = db.execute(
        select(User).where(User.id == user_id, User.organization_id == org_id)
    ).scalar_one_or_none()
    if user is None:
        raise NotFoundError("User not found.")
    return user


def _permissions_of(db: Session, user: User) -> list[str]:
    """Stored grants for analysts; Org Admins implicitly hold every permission."""
    if user.role == Role.ORG_ADMIN.value:
        return list(ALL_PERMISSIONS)
    rows = db.execute(select(UserPermission.permission).where(UserPermission.user_id == user.id)).scalars()
    return sorted(rows)


def _org_dataset(db: Session, ctx: AuthContext, dataset_id: int) -> Dataset:
    return authorize_dataset(db, ctx, dataset_id, roles=(Role.ORG_ADMIN,), include_archived=True)


def _org_analyst(db: Session, org_id: int, user_id: int) -> User:
    user = _org_user(db, org_id, user_id)
    if user.role != Role.ANALYST.value:
        raise ValidationError(
            "Dataset access is only managed for analysts. Organization Admins see every dataset."
        )
    return user


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def _user_counts(db: Session, org_id: int) -> dict[str, int]:
    rows = db.execute(
        select(User.role, User.is_active, func.count(User.id))
        .where(User.organization_id == org_id)
        .group_by(User.role, User.is_active)
    ).all()
    counts = {"admins": 0, "analysts": 0, "active": 0, "inactive": 0, "total": 0}
    for role, is_active, count in rows:
        if role == Role.ORG_ADMIN.value:
            counts["admins"] += count
        elif role == Role.ANALYST.value:
            counts["analysts"] += count
        counts["active" if is_active else "inactive"] += count
        counts["total"] += count
    return counts


def _dataset_counts(db: Session, org_id: int) -> dict[str, int]:
    rows = db.execute(
        select(Dataset.status, func.count(Dataset.id))
        .where(Dataset.organization_id == org_id)
        .group_by(Dataset.status)
    ).all()
    by_status = dict(rows)
    active = int(by_status.get(DatasetStatus.ACTIVE.value, 0))
    archived = int(by_status.get(DatasetStatus.ARCHIVED.value, 0))
    return {"active": active, "archived": archived, "total": active + archived}


def _average_quality(db: Session, org_id: int) -> float | None:
    """Mean quality score of the current version of every active dataset (metadata only)."""
    value = db.execute(
        select(func.avg(DatasetVersion.quality_score))
        .select_from(Dataset)
        .join(DatasetVersion, DatasetVersion.id == Dataset.current_version_id)
        .where(
            Dataset.organization_id == org_id,
            Dataset.status == DatasetStatus.ACTIVE.value,
            DatasetVersion.quality_score.is_not(None),
        )
    ).scalar_one()
    return None if value is None else float(value)


def _recent_operations(db: Session, org_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        select(CleaningOperation, Dataset.name, User.username)
        .join(Dataset, Dataset.id == CleaningOperation.dataset_id)
        .outerjoin(User, User.id == CleaningOperation.user_id)
        .where(CleaningOperation.organization_id == org_id)
        .order_by(CleaningOperation.created_at.desc(), CleaningOperation.id.desc())
        .limit(RECENT_OPERATIONS_LIMIT)
    ).all()
    return [
        {
            "operation_id": op.id,
            "dataset_id": op.dataset_id,
            "dataset_name": dataset_name,
            "operation_type": op.operation_type,
            "status": op.status,
            "user": username or "system",
            "created_at": op.created_at,
        }
        for op, dataset_name, username in rows
    ]


def org_dashboard(ctx: AuthContext) -> dict[str, Any]:
    """Organization overview built from database metadata only (no dataset file is read)."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        org = db.get(Organization, org_id)
        limits = subscriptions.get_limits(db, org_id)
        storage_used = subscriptions.org_storage_bytes(db, org_id)
        analyses_used = usage_tracking.monthly_total(db, org_id, UsageMetric.ANALYSIS_QUERY)
        anomaly_since = timeutil.utcnow() - timedelta(days=ANOMALY_WINDOW_DAYS)
        anomaly_count = db.execute(
            select(func.count(AnomalyEvent.id)).where(
                AnomalyEvent.organization_id == org_id, AnomalyEvent.detected_at >= anomaly_since
            )
        ).scalar_one()
        return {
            "organization_id": org_id,
            "organization_name": org.name,
            "plan_code": limits.plan_code,
            "plan_name": limits.plan_name,
            "features": sorted(subscriptions.enabled_features(db, org_id)),
            "users": _user_counts(db, org_id),
            "max_users": limits.max_users,
            "datasets": _dataset_counts(db, org_id),
            "max_datasets": limits.max_datasets,
            "storage_used_bytes": storage_used,
            "storage_quota_bytes": limits.max_storage_bytes,
            "storage_used_fraction": storage_used / limits.max_storage_bytes,
            "analyses_this_month": analyses_used,
            "max_monthly_analyses": limits.max_monthly_analyses,
            "average_quality_score": _average_quality(db, org_id),
            "recent_cleaning_operations": _recent_operations(db, org_id),
            "recent_anomaly_events": int(anomaly_count),
            "anomaly_window_days": ANOMALY_WINDOW_DAYS,
        }


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------


def list_users(ctx: AuthContext) -> list[dict[str, Any]]:
    """Every user of the caller's organization, admins first, then by username."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        users = (
            db.execute(
                select(User).where(User.organization_id == org_id).order_by(User.role.desc(), User.username)
            )
            .scalars()
            .all()
        )
        grants: dict[int, list[str]] = {}
        for user_id, permission in db.execute(
            select(UserPermission.user_id, UserPermission.permission).where(
                UserPermission.organization_id == org_id
            )
        ).all():
            grants.setdefault(user_id, []).append(permission)
        return [
            user_to_dict(
                user,
                list(ALL_PERMISSIONS)
                if user.role == Role.ORG_ADMIN.value
                else sorted(grants.get(user.id, [])),
            )
            for user in users
        ]


def create_analyst(
    ctx: AuthContext,
    *,
    username: str,
    email: str,
    full_name: str,
    password: str,
    permissions: set[str] | None = None,
) -> dict[str, Any]:
    """Create an analyst in the caller's organization.

    Permissions default to ``ANALYST_DEFAULT_PERMISSIONS``; unknown keys raise
    ``ValidationError``. The plan's active-user quota is enforced.
    """
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        granted = (
            set(ANALYST_DEFAULT_PERMISSIONS) if permissions is None else validate_permissions(permissions)
        )
        user = create_user_row(
            db,
            organization_id=org_id,
            role=Role.ANALYST,
            username=username,
            email=email,
            password=password,
            full_name=full_name,
            created_by_id=ctx.user_id,
        )
        for key in sorted(granted):
            db.add(
                UserPermission(
                    organization_id=org_id, user_id=user.id, permission=key, granted_by_id=ctx.user_id
                )
            )
        db.flush()
        record_audit(
            db,
            ctx,
            "user.create",
            target_type="user",
            target_id=user.id,
            details={"username": user.username, "role": user.role, "permissions": sorted(granted)},
        )
        return user_to_dict(user, sorted(granted))


def update_user(
    ctx: AuthContext, user_id: int, *, email: str | None = None, full_name: str | None = None
) -> dict[str, Any]:
    """Change a user's e-mail address and/or display name."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        user = _org_user(db, org_id, user_id)
        changed: list[str] = []
        if email is not None:
            user.email = normalize_email(email)
            changed.append("email")
        if full_name is not None:
            user.full_name = normalize_full_name(full_name)
            changed.append("full_name")
        if not changed:
            raise ValidationError("Nothing to update.")
        record_audit(
            db, ctx, "user.update", target_type="user", target_id=user.id, details={"fields": changed}
        )
        return user_to_dict(user, _permissions_of(db, user))


def set_user_active(ctx: AuthContext, user_id: int, active: bool) -> dict[str, Any]:
    """Activate or deactivate a user of the caller's organization.

    Admins cannot deactivate themselves. Deactivation revokes the user's
    sessions; reactivation enforces the plan's user quota. Setting the current
    state again changes nothing and writes no audit entry.
    """
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        if not isinstance(active, bool):
            raise ValidationError("The active flag must be true or false.")
        user = _org_user(db, org_id, user_id)
        if not active and user.id == ctx.user_id:
            raise ValidationError("You cannot deactivate your own account.")
        revoked = 0
        if user.is_active != active:
            if active:
                subscriptions.enforce_user_quota(db, org_id)
            else:
                revoked = revoke_sessions(db, [user.id], "user_deactivated")
            user.is_active = active
            record_audit(
                db,
                ctx,
                "user.activate" if active else "user.deactivate",
                target_type="user",
                target_id=user.id,
                details={"sessions_revoked": revoked},
            )
        result = user_to_dict(user, _permissions_of(db, user))
        result["sessions_revoked"] = revoked
        return result


def reset_user_password(ctx: AuthContext, user_id: int, new_password: str) -> dict[str, Any]:
    """Set a new password, require a change at next sign-in and revoke all sessions."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        user = _org_user(db, org_id, user_id)
        validate_password_strength(new_password, username=user.username)
        user.password_hash = hash_password(new_password)
        user.password_changed_at = timeutil.utcnow()
        user.must_change_password = True
        user.failed_login_count = 0
        user.locked_until = None
        revoked = revoke_sessions(db, [user.id], "password_reset")
        record_audit(
            db,
            ctx,
            "user.reset_password",
            target_type="user",
            target_id=user.id,
            details={"sessions_revoked": revoked},
        )
        result = user_to_dict(user, _permissions_of(db, user))
        result["sessions_revoked"] = revoked
        return result


def set_user_permissions(ctx: AuthContext, user_id: int, permissions: set[str]) -> dict[str, Any]:
    """Replace an analyst's permission set (Org Admins hold every permission implicitly)."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        user = _org_user(db, org_id, user_id)
        if user.role != Role.ANALYST.value:
            raise ValidationError("Permissions can only be configured for analysts.")
        wanted = validate_permissions(permissions)
        current = set(_permissions_of(db, user))
        added, removed = sorted(wanted - current), sorted(current - wanted)
        if removed:
            db.execute(
                delete(UserPermission).where(
                    UserPermission.user_id == user.id, UserPermission.permission.in_(removed)
                )
            )
        for key in added:
            db.add(
                UserPermission(
                    organization_id=org_id, user_id=user.id, permission=key, granted_by_id=ctx.user_id
                )
            )
        db.flush()
        record_audit(
            db,
            ctx,
            "user.permissions",
            target_type="user",
            target_id=user.id,
            details={"added": added, "removed": removed},
        )
        return user_to_dict(user, sorted(wanted))


# ---------------------------------------------------------------------------
# Dataset grants
# ---------------------------------------------------------------------------


def _access_row(db: Session, dataset_id: int, user_id: int) -> DatasetAccess | None:
    return db.execute(
        select(DatasetAccess).where(DatasetAccess.dataset_id == dataset_id, DatasetAccess.user_id == user_id)
    ).scalar_one_or_none()


def grant_dataset_access(ctx: AuthContext, dataset_id: int, user_id: int) -> dict[str, Any]:
    """Let an analyst of the organization see a dataset (idempotent)."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        dataset = _org_dataset(db, ctx, dataset_id)
        user = _org_analyst(db, org_id, user_id)
        created = _access_row(db, dataset.id, user.id) is None
        if created:
            db.add(
                DatasetAccess(
                    organization_id=org_id, dataset_id=dataset.id, user_id=user.id, granted_by_id=ctx.user_id
                )
            )
            record_audit(
                db,
                ctx,
                "dataset.grant",
                target_type="dataset",
                target_id=dataset.id,
                details={"user_id": user.id, "username": user.username},
            )
        return {"dataset_id": dataset.id, "user_id": user.id, "has_access": True, "changed": created}


def revoke_dataset_access(ctx: AuthContext, dataset_id: int, user_id: int) -> dict[str, Any]:
    """Remove an analyst's access to a dataset (idempotent)."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        dataset = _org_dataset(db, ctx, dataset_id)
        user = _org_analyst(db, org_id, user_id)
        row = _access_row(db, dataset.id, user.id)
        if row is not None:
            db.delete(row)
            record_audit(
                db,
                ctx,
                "dataset.revoke",
                target_type="dataset",
                target_id=dataset.id,
                details={"user_id": user.id, "username": user.username},
            )
        return {"dataset_id": dataset.id, "user_id": user.id, "has_access": False, "changed": row is not None}


def list_dataset_access(ctx: AuthContext, dataset_id: int) -> list[dict[str, Any]]:
    """Analysts holding an explicit grant on the dataset, ordered by username."""
    with session_scope() as db:
        _authorize_admin(db, ctx)
        dataset = _org_dataset(db, ctx, dataset_id)
        rows = db.execute(
            select(DatasetAccess, User)
            .join(User, User.id == DatasetAccess.user_id)
            .where(DatasetAccess.dataset_id == dataset.id, User.organization_id == dataset.organization_id)
            .order_by(User.username)
        ).all()
        return [
            {
                "user_id": user.id,
                "username": user.username,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "granted_at": access.created_at,
                "granted_by_id": access.granted_by_id,
            }
            for access, user in rows
        ]


# ---------------------------------------------------------------------------
# Organization settings
# ---------------------------------------------------------------------------


def get_settings_view(ctx: AuthContext) -> dict[str, Any]:
    """Effective organization settings (stored values merged over global defaults)."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        return org_settings.get_org_settings(db, org_id)


def update_org_settings(ctx: AuthContext, changes: dict[str, Any]) -> dict[str, Any]:
    """Validate and merge setting changes into ``organizations.settings_json``."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        if not isinstance(changes, dict) or not changes:
            raise ValidationError("No settings were changed.")
        clean = org_settings.validate_changes(changes)
        org = db.get(Organization, org_id)
        stored = json.loads(org.settings_json or "{}")
        stored = {k: v for k, v in stored.items() if k in org_settings.ALLOWED_KEYS}
        stored.update(clean)
        org.settings_json = json.dumps(stored)
        record_audit(
            db,
            ctx,
            "org.settings.update",
            target_type="organization",
            target_id=org_id,
            details={"keys": sorted(clean)},
        )
        db.flush()
        return org_settings.get_org_settings(db, org_id)


# ---------------------------------------------------------------------------
# Usage and analysis history
# ---------------------------------------------------------------------------


def _month_starts(now: datetime, months: int) -> list[datetime]:
    """First instant of the last ``months`` calendar months, oldest first."""
    year, month = now.year, now.month
    starts: list[datetime] = []
    for _ in range(months):
        starts.append(datetime(year, month, 1))
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return list(reversed(starts))


def usage_report(ctx: AuthContext, *, months: int = 6) -> list[dict[str, Any]]:
    """Usage totals per calendar month and metric, oldest month first.

    Every month in the window is listed and every known metric appears with a
    zero when nothing was recorded. Each row also carries the plan's current
    limits so the UI can compare usage against them.
    """
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        if not isinstance(months, int) or isinstance(months, bool) or not 1 <= months <= MAX_USAGE_MONTHS:
            raise ValidationError(f"Months must be between 1 and {MAX_USAGE_MONTHS}.")
        limits = subscriptions.get_limits(db, org_id)
        starts = _month_starts(timeutil.utcnow(), months)
        month_key = func.strftime("%Y-%m", UsageRecord.created_at)
        rows = db.execute(
            select(month_key, UsageRecord.metric, func.sum(UsageRecord.quantity))
            .where(UsageRecord.organization_id == org_id, UsageRecord.created_at >= starts[0])
            .group_by(month_key, UsageRecord.metric)
        ).all()
        totals: dict[tuple[str, str], float] = {
            (key, metric): float(total or 0) for key, metric, total in rows
        }
        metrics = sorted({m.value for m in UsageMetric} | {metric for _key, metric in totals})
        limit_view = {
            "max_monthly_analyses": limits.max_monthly_analyses,
            "max_users": limits.max_users,
            "max_storage_mb": limits.max_storage_mb,
            "max_upload_mb": limits.max_upload_mb,
            "max_datasets": limits.max_datasets,
        }
        report = []
        for start in starts:
            key = start.strftime("%Y-%m")
            month_totals = {metric: totals.get((key, metric), 0.0) for metric in metrics}
            report.append(
                {
                    "month": key,
                    "month_start": start,
                    "totals": month_totals,
                    "analysis_quota_used_fraction": month_totals[UsageMetric.ANALYSIS_QUERY.value]
                    / limits.max_monthly_analyses,
                    "limits": dict(limit_view),
                }
            )
        return report


def org_analysis_history(
    ctx: AuthContext, *, limit: int = 200, user_id: int | None = None, dataset_id: int | None = None
) -> list[dict[str, Any]]:
    """Organization-wide analysis history, newest first, optionally filtered.

    A ``user_id`` or ``dataset_id`` filter that does not belong to the
    caller's organization raises ``NotFoundError``.
    """
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_HISTORY_LIMIT:
            raise ValidationError(f"The limit must be between 1 and {MAX_HISTORY_LIMIT}.")
        query = (
            select(AnalysisHistory, User.username, Dataset.name)
            .outerjoin(User, User.id == AnalysisHistory.user_id)
            .outerjoin(Dataset, Dataset.id == AnalysisHistory.dataset_id)
            .where(AnalysisHistory.organization_id == org_id)
        )
        if user_id is not None:
            query = query.where(AnalysisHistory.user_id == _org_user(db, org_id, user_id).id)
        if dataset_id is not None:
            query = query.where(AnalysisHistory.dataset_id == _org_dataset(db, ctx, dataset_id).id)
        rows = db.execute(
            query.order_by(AnalysisHistory.created_at.desc(), AnalysisHistory.id.desc()).limit(limit)
        ).all()
        return [
            {
                "id": entry.id,
                "question": entry.question,
                "input_mode": entry.input_mode,
                "user_id": entry.user_id,
                "username": username or "system",
                "dataset_id": entry.dataset_id,
                "dataset_name": dataset_name,
                "status": entry.status,
                "planner": entry.planner,
                "chart_type": entry.chart_type,
                "duration_ms": entry.duration_ms,
                "created_at": entry.created_at,
            }
            for entry, username, dataset_name in rows
        ]
