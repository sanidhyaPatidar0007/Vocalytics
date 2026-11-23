"""Super Admin (platform operator) workflows (specification sections 4, 5 and 18).

Every public function takes the caller's ``AuthContext`` first, opens its own
``session_scope()`` and calls ``authorize_super_admin`` before reading or
writing anything, so organization members always get ``PermissionDeniedError``.
Every change is written to the audit log with an action prefixed
``platform.`` and, when an organization is targeted, that organization's id,
which gives the explicit and auditable trail required by section 4.

Nothing here ever deletes tenant data: suspending or deactivating an
organization only blocks access (``auth.authorization.ensure_org_active``)
and revokes open sessions. Plan changes end the current subscription and
start a new one, so the subscription history is preserved.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from auth.authorization import AuthContext, authorize_super_admin
from config.settings import get_settings
from core import storage, timeutil
from core.constants import ALL_FEATURES, USER_ROLES, DatasetStatus, OrgStatus, Role
from core.errors import ConflictError, NotFoundError, ValidationError
from database.connection import session_scope
from database.models import (
    AnalysisHistory,
    CleaningOperation,
    Dataset,
    ErrorEvent,
    Organization,
    PlanFeature,
    Subscription,
    SubscriptionPlan,
    UsageRecord,
    User,
)
from modules import subscriptions
from modules.organization_admin import create_user_row, normalize_email, revoke_sessions
from services.audit_service import record_audit

PLAN_CODE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,49}$")
CURRENCY_PATTERN = re.compile(r"^[A-Za-z]{3}$")
BILLING_PERIODS = ("monthly", "quarterly", "yearly")
PLAN_LIMIT_FIELDS = (
    "max_users",
    "max_storage_mb",
    "max_upload_mb",
    "max_datasets",
    "max_monthly_analyses",
    "max_versions_per_dataset",
)
PLAN_UPDATABLE_FIELDS = (
    "name",
    "description",
    "price_cents",
    "currency",
    "billing_period",
    *PLAN_LIMIT_FIELDS,
)
MAX_ORG_NAME_LENGTH = 200
MAX_SLUG_LENGTH = 100
MAX_PLAN_NAME_LENGTH = 100
MAX_TEXT_LENGTH = 2000
MAX_BILLING_REFERENCE_LENGTH = 100
MAX_ERROR_EVENTS = 1000
METRICS_WINDOW_DAYS = 30
ERROR_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_id(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"Invalid {label}.")
    return value


def _clean_text(value: object, label: str, max_length: int, *, required: bool) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"The {label} must be text.")
    clean = value.strip()
    if required and not clean:
        raise ValidationError(f"The {label} is required.")
    if len(clean) > max_length:
        raise ValidationError(f"The {label} must be at most {max_length} characters.")
    return clean


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"{label} must be a positive whole number.")
    return value


def _optional_email(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return normalize_email(value)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (slug or "org")[: MAX_SLUG_LENGTH - 10].strip("-") or "org"


def _unique_slug(db: Session, name: str) -> str:
    """Slug from the name, with ``-2``, ``-3``... appended until it is unused."""
    base = _slugify(name)
    candidate, counter = base, 1
    while db.execute(select(Organization.id).where(Organization.slug == candidate)).first():
        counter += 1
        candidate = f"{base}-{counter}"
    return candidate


def _ensure_org_name_free(db: Session, name: str, *, exclude_id: int | None = None) -> None:
    query = select(Organization.id).where(func.lower(Organization.name) == name.lower())
    if exclude_id is not None:
        query = query.where(Organization.id != exclude_id)
    if db.execute(query).first():
        raise ConflictError("An organization with this name already exists.")


def _get_org(db: Session, org_id: int) -> Organization:
    _validate_id(org_id, "organization")
    org = db.get(Organization, org_id)
    if org is None:
        raise NotFoundError("Organization not found.")
    return org


def _get_plan(db: Session, plan_id: int) -> SubscriptionPlan:
    _validate_id(plan_id, "plan")
    plan = db.get(SubscriptionPlan, plan_id)
    if plan is None:
        raise NotFoundError("Plan not found.")
    return plan


def _plan_by_code(db: Session, plan_code: str) -> SubscriptionPlan:
    if not isinstance(plan_code, str):
        raise ValidationError("A plan code is required.")
    plan = db.execute(
        select(SubscriptionPlan).where(SubscriptionPlan.code == plan_code.strip().lower())
    ).scalar_one_or_none()
    if plan is None:
        raise NotFoundError("Plan not found.")
    if not plan.is_active:
        raise ValidationError("This plan is inactive and cannot be assigned.")
    return plan


def _validate_plan_fields(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize plan attributes (any subset of the updatable fields)."""
    clean: dict[str, Any] = {}
    for key, value in values.items():
        if key == "name":
            clean[key] = _clean_text(value, "plan name", MAX_PLAN_NAME_LENGTH, required=True)
        elif key == "description":
            clean[key] = _clean_text(value, "description", MAX_TEXT_LENGTH, required=False)
        elif key == "price_cents":
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationError("The price must be zero or a positive whole number of cents.")
            clean[key] = value
        elif key == "currency":
            if not isinstance(value, str) or not CURRENCY_PATTERN.fullmatch(value.strip()):
                raise ValidationError("The currency must be a three-letter code such as USD.")
            clean[key] = value.strip().upper()
        elif key == "billing_period":
            if value not in BILLING_PERIODS:
                raise ValidationError("The billing period must be monthly, quarterly or yearly.")
            clean[key] = value
        elif key in PLAN_LIMIT_FIELDS:
            clean[key] = _positive_int(value, key.replace("_", " ").capitalize())
    ceiling = get_settings().absolute_upload_ceiling_mb
    if clean.get("max_upload_mb", 0) > ceiling:
        raise ValidationError(f"The upload limit cannot exceed the platform ceiling of {ceiling} MB.")
    if "max_versions_per_dataset" in clean and clean["max_versions_per_dataset"] < 2:
        raise ValidationError("Plans must keep at least 2 versions per dataset.")
    return clean


def _validate_feature_keys(features: set[str] | frozenset[str] | None) -> set[str]:
    if features is None:
        return set()
    if isinstance(features, (str, bytes)) or not isinstance(features, (set, frozenset, list, tuple)):
        raise ValidationError("Features must be a set of feature names.")
    keys = {str(getattr(f, "value", f)) for f in features}
    unknown = keys - set(ALL_FEATURES)
    if unknown:
        raise ValidationError(f"Unknown feature(s): {', '.join(sorted(unknown))}.")
    return keys


def _validate_feature_key(feature_key: str) -> str:
    key = str(getattr(feature_key, "value", feature_key))
    if key not in ALL_FEATURES:
        raise ValidationError("Unknown feature.")
    return key


# ---------------------------------------------------------------------------
# Plain-data views
# ---------------------------------------------------------------------------


def _org_dict(db: Session, org: Organization) -> dict[str, Any]:
    subscription = subscriptions.get_active_subscription(db, org.id)
    plan = db.get(SubscriptionPlan, subscription.plan_id) if subscription else None
    dataset_rows = dict(
        db.execute(
            select(Dataset.status, func.count(Dataset.id))
            .where(Dataset.organization_id == org.id)
            .group_by(Dataset.status)
        ).all()
    )
    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "status": org.status,
        "status_reason": org.status_reason,
        "status_changed_at": org.status_changed_at,
        "contact_email": org.contact_email,
        "plan_code": plan.code if plan else None,
        "plan_name": plan.name if plan else None,
        "renewal_date": subscription.renewal_date if subscription else None,
        "active_users": subscriptions.active_user_count(db, org.id),
        "active_datasets": int(dataset_rows.get(DatasetStatus.ACTIVE.value, 0)),
        "archived_datasets": int(dataset_rows.get(DatasetStatus.ARCHIVED.value, 0)),
        "storage_bytes": subscriptions.org_storage_bytes(db, org.id),
        "created_at": org.created_at,
    }


def _plan_dict(db: Session, plan: SubscriptionPlan) -> dict[str, Any]:
    stored = {row.feature_key: row.enabled for row in plan.features}
    organizations = db.execute(
        select(func.count(Subscription.id)).where(
            Subscription.plan_id == plan.id, Subscription.status == "active"
        )
    ).scalar_one()
    return {
        "id": plan.id,
        "code": plan.code,
        "name": plan.name,
        "description": plan.description,
        "price_cents": plan.price_cents,
        "currency": plan.currency,
        "billing_period": plan.billing_period,
        **{key: getattr(plan, key) for key in PLAN_LIMIT_FIELDS},
        "is_active": plan.is_active,
        "organizations": int(organizations),
        "features": {key: bool(stored.get(key, False)) for key in ALL_FEATURES},
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
    }


def _subscription_dict(
    subscription: Subscription, plan: SubscriptionPlan, org_name: str, assigned_by: str | None
) -> dict[str, Any]:
    return {
        "id": subscription.id,
        "organization_id": subscription.organization_id,
        "organization_name": org_name,
        "plan_id": plan.id,
        "plan_code": plan.code,
        "plan_name": plan.name,
        "status": subscription.status,
        "started_at": subscription.started_at,
        "ended_at": subscription.ended_at,
        "renewal_date": subscription.renewal_date,
        "auto_renew": subscription.auto_renew,
        "billing_reference": subscription.billing_reference,
        "notes": subscription.notes,
        "assigned_by": assigned_by,
    }


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


def list_organizations(ctx: AuthContext) -> list[dict[str, Any]]:
    """Every organization with its plan, renewal date, user/dataset counts and storage."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        orgs = db.execute(select(Organization).order_by(Organization.name)).scalars().all()
        return [_org_dict(db, org) for org in orgs]


def _start_subscription(
    db: Session,
    ctx: AuthContext,
    org_id: int,
    plan: SubscriptionPlan,
    *,
    renewal_date: datetime | None = None,
    auto_renew: bool = True,
    billing_reference: str | None = None,
    notes: str = "",
) -> Subscription:
    subscription = Subscription(
        organization_id=org_id,
        plan_id=plan.id,
        status="active",
        started_at=timeutil.utcnow(),
        renewal_date=renewal_date,
        auto_renew=auto_renew,
        billing_reference=billing_reference,
        notes=notes,
        assigned_by_id=ctx.user_id,
    )
    db.add(subscription)
    db.flush()
    return subscription


def _new_org_admin(
    db: Session,
    ctx: AuthContext,
    org_id: int,
    *,
    username: str,
    email: str,
    password: str,
    full_name: str,
) -> User:
    user = create_user_row(
        db,
        organization_id=org_id,
        role=Role.ORG_ADMIN,
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        created_by_id=ctx.user_id,
    )
    record_audit(
        db,
        ctx,
        "platform.org_admin.create",
        target_type="user",
        target_id=user.id,
        details={"username": user.username},
        organization_id=org_id,
    )
    return user


def create_organization(
    ctx: AuthContext,
    *,
    name: str,
    contact_email: str | None = None,
    plan_code: str = "basic",
    admin_username: str | None = None,
    admin_email: str | None = None,
    admin_password: str | None = None,
    admin_full_name: str = "",
) -> dict[str, Any]:
    """Create an organization, subscribe it to an active plan and optionally add its first Org Admin.

    Names are unique (case-insensitive, ``ConflictError``). The slug is
    generated from the name and made unique. When ``admin_username`` is given,
    ``admin_email`` and ``admin_password`` are required and the password policy
    applies. Everything happens in one transaction.
    """
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        clean_name = _clean_text(name, "organization name", MAX_ORG_NAME_LENGTH, required=True)
        _ensure_org_name_free(db, clean_name)
        email = _optional_email(contact_email)
        plan = _plan_by_code(db, plan_code)
        wants_admin = any(v is not None for v in (admin_username, admin_email, admin_password))
        if wants_admin and not (admin_username and admin_email and admin_password):
            raise ValidationError(
                "To create the first Organization Admin, enter a username, e-mail and password."
            )
        org = Organization(
            name=clean_name,
            slug=_unique_slug(db, clean_name),
            status=OrgStatus.ACTIVE.value,
            contact_email=email,
        )
        db.add(org)
        db.flush()
        _start_subscription(db, ctx, org.id, plan)
        record_audit(
            db,
            ctx,
            "platform.organization.create",
            target_type="organization",
            target_id=org.id,
            details={"name": org.name, "plan_code": plan.code},
            organization_id=org.id,
        )
        admin_id = None
        if wants_admin:
            admin = _new_org_admin(
                db,
                ctx,
                org.id,
                username=admin_username or "",
                email=admin_email or "",
                password=admin_password or "",
                full_name=admin_full_name,
            )
            admin_id = admin.id
        result = _org_dict(db, org)
        result["admin_user_id"] = admin_id
        return result


def update_organization(
    ctx: AuthContext, org_id: int, *, name: str | None = None, contact_email: str | None = None
) -> dict[str, Any]:
    """Rename an organization and/or change its contact e-mail (an empty e-mail clears it)."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        org = _get_org(db, org_id)
        changes: dict[str, Any] = {}
        if name is not None:
            clean_name = _clean_text(name, "organization name", MAX_ORG_NAME_LENGTH, required=True)
            _ensure_org_name_free(db, clean_name, exclude_id=org.id)
            changes["name"] = [org.name, clean_name]
            org.name = clean_name
        if contact_email is not None:
            org.contact_email = _optional_email(contact_email)
            changes["contact_email"] = "updated"
        if not changes:
            raise ValidationError("Nothing to update.")
        record_audit(
            db,
            ctx,
            "platform.organization.update",
            target_type="organization",
            target_id=org.id,
            details=changes,
            organization_id=org.id,
        )
        db.flush()
        return _org_dict(db, org)


def set_organization_status(
    ctx: AuthContext, org_id: int, status: str, *, reason: str = ""
) -> dict[str, Any]:
    """Activate, suspend or deactivate an organization without deleting any data.

    Records the reason and time. When the new status is not ``active`` every
    open session of the organization's users is revoked, and ``authorize``
    blocks the organization's members on their next call.
    """
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        org = _get_org(db, org_id)
        valid = {s.value for s in OrgStatus}
        if status not in valid:
            raise ValidationError("The status must be active, suspended or deactivated.")
        clean_reason = _clean_text(reason, "reason", MAX_TEXT_LENGTH, required=False)
        previous = org.status
        org.status = status
        org.status_reason = clean_reason or None
        org.status_changed_at = timeutil.utcnow()
        revoked = 0
        if status != OrgStatus.ACTIVE.value:
            user_ids = list(db.execute(select(User.id).where(User.organization_id == org.id)).scalars())
            revoked = revoke_sessions(db, user_ids, f"organization_{status}")
        record_audit(
            db,
            ctx,
            f"platform.organization.{status}",
            target_type="organization",
            target_id=org.id,
            details={
                "previous_status": previous,
                "status": status,
                "reason": clean_reason,
                "sessions_revoked": revoked,
            },
            organization_id=org.id,
        )
        db.flush()
        result = _org_dict(db, org)
        result["sessions_revoked"] = revoked
        return result


def create_org_admin(
    ctx: AuthContext, org_id: int, *, username: str, email: str, password: str, full_name: str = ""
) -> dict[str, Any]:
    """Add an Organization Admin to an organization (plan user quota enforced)."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        org = _get_org(db, org_id)
        user = _new_org_admin(
            db, ctx, org.id, username=username, email=email, password=password, full_name=full_name
        )
        return {
            "id": user.id,
            "organization_id": org.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "is_active": user.is_active,
            "created_at": user.created_at,
        }


# ---------------------------------------------------------------------------
# Plans and features
# ---------------------------------------------------------------------------


def list_plans(ctx: AuthContext) -> list[dict[str, Any]]:
    """Every plan with limits, pricing, status, usage count and the full feature matrix."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        plans = (
            db.execute(select(SubscriptionPlan).order_by(SubscriptionPlan.price_cents, SubscriptionPlan.code))
            .scalars()
            .all()
        )
        return [_plan_dict(db, plan) for plan in plans]


def create_plan(
    ctx: AuthContext,
    *,
    code: str,
    name: str,
    description: str = "",
    price_cents: int = 0,
    currency: str = "USD",
    billing_period: str = "monthly",
    max_users: int,
    max_storage_mb: int,
    max_upload_mb: int,
    max_datasets: int,
    max_monthly_analyses: int,
    max_versions_per_dataset: int,
    features: set[str] | None = None,
) -> dict[str, Any]:
    """Create a plan with a ``plan_features`` row for every known feature.

    ``features`` lists the enabled keys; every other feature is stored as
    disabled. Codes are lowercase slugs and unique (``ConflictError``).
    """
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        clean_code = code.strip().lower() if isinstance(code, str) else ""
        if not PLAN_CODE_PATTERN.fullmatch(clean_code):
            raise ValidationError(
                "The plan code must be 2 to 50 lowercase letters, digits, hyphens or underscores."
            )
        values = _validate_plan_fields(
            {
                "name": name,
                "description": description,
                "price_cents": price_cents,
                "currency": currency,
                "billing_period": billing_period,
                "max_users": max_users,
                "max_storage_mb": max_storage_mb,
                "max_upload_mb": max_upload_mb,
                "max_datasets": max_datasets,
                "max_monthly_analyses": max_monthly_analyses,
                "max_versions_per_dataset": max_versions_per_dataset,
            }
        )
        enabled = _validate_feature_keys(features)
        if db.execute(select(SubscriptionPlan.id).where(SubscriptionPlan.code == clean_code)).first():
            raise ConflictError("A plan with this code already exists.")
        plan = SubscriptionPlan(code=clean_code, is_active=True, **values)
        plan.features = [PlanFeature(feature_key=key, enabled=key in enabled) for key in ALL_FEATURES]
        db.add(plan)
        db.flush()
        record_audit(
            db,
            ctx,
            "platform.plan.create",
            target_type="plan",
            target_id=plan.id,
            details={"code": plan.code, "features": sorted(enabled)},
        )
        return _plan_dict(db, plan)


def update_plan(ctx: AuthContext, plan_id: int, **changes: Any) -> dict[str, Any]:
    """Change whitelisted plan attributes (name, description, pricing, billing, limits).

    The code is fixed once created. Limit changes take effect immediately for
    every organization on the plan.
    """
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        plan = _get_plan(db, plan_id)
        if not changes:
            raise ValidationError("Nothing to update.")
        unknown = set(changes) - set(PLAN_UPDATABLE_FIELDS)
        if unknown:
            raise ValidationError(f"These plan fields cannot be changed: {', '.join(sorted(unknown))}.")
        clean = _validate_plan_fields(changes)
        before = {key: getattr(plan, key) for key in clean}
        for key, value in clean.items():
            setattr(plan, key, value)
        record_audit(
            db,
            ctx,
            "platform.plan.update",
            target_type="plan",
            target_id=plan.id,
            details={"code": plan.code, "before": before, "after": clean},
        )
        db.flush()
        return _plan_dict(db, plan)


def set_plan_feature(ctx: AuthContext, plan_id: int, feature_key: str, enabled: bool) -> dict[str, Any]:
    """Enable or disable one feature of a plan; it applies on the very next authorized call."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        if not isinstance(enabled, bool):
            raise ValidationError("The enabled flag must be true or false.")
        plan = _get_plan(db, plan_id)
        key = _validate_feature_key(feature_key)
        row = next((f for f in plan.features if f.feature_key == key), None)
        if row is None:
            plan.features.append(PlanFeature(feature_key=key, enabled=enabled))
        else:
            row.enabled = enabled
        record_audit(
            db,
            ctx,
            "platform.plan.feature",
            target_type="plan",
            target_id=plan.id,
            details={"code": plan.code, "feature": key, "enabled": enabled},
        )
        db.flush()
        return _plan_dict(db, plan)


def set_plan_active(ctx: AuthContext, plan_id: int, active: bool) -> dict[str, Any]:
    """Make a plan (un)available for new assignments; existing subscriptions keep working."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        if not isinstance(active, bool):
            raise ValidationError("The active flag must be true or false.")
        plan = _get_plan(db, plan_id)
        plan.is_active = active
        record_audit(
            db,
            ctx,
            "platform.plan.activate" if active else "platform.plan.deactivate",
            target_type="plan",
            target_id=plan.id,
            details={"code": plan.code},
        )
        db.flush()
        return _plan_dict(db, plan)


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------


def assign_plan(
    ctx: AuthContext,
    org_id: int,
    plan_code: str,
    *,
    renewal_date: datetime | None = None,
    auto_renew: bool = True,
    billing_reference: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Move an organization to a plan, keeping the subscription history.

    In one transaction the current active subscription is ended (status
    ``ended`` with ``ended_at``) and a new active one is created. Inactive
    plans are rejected.
    """
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        org = _get_org(db, org_id)
        plan = _plan_by_code(db, plan_code)
        if renewal_date is not None and not isinstance(renewal_date, datetime):
            raise ValidationError("The renewal date must be a date and time.")
        if not isinstance(auto_renew, bool):
            raise ValidationError("Auto-renew must be true or false.")
        reference = None
        if billing_reference is not None:
            reference = (
                _clean_text(
                    billing_reference, "billing reference", MAX_BILLING_REFERENCE_LENGTH, required=False
                )
                or None
            )
        clean_notes = _clean_text(notes, "notes", MAX_TEXT_LENGTH, required=False)
        previous = subscriptions.get_active_subscription(db, org.id)
        previous_code = None
        if previous is not None:
            previous_code = db.get(SubscriptionPlan, previous.plan_id).code
            previous.status = "ended"
            previous.ended_at = timeutil.utcnow()
            db.flush()  # the partial unique index allows one active row per organization
        subscription = _start_subscription(
            db,
            ctx,
            org.id,
            plan,
            renewal_date=renewal_date,
            auto_renew=auto_renew,
            billing_reference=reference,
            notes=clean_notes,
        )
        record_audit(
            db,
            ctx,
            "platform.subscription.assign",
            target_type="subscription",
            target_id=subscription.id,
            details={"previous_plan": previous_code, "plan": plan.code, "renewal_date": renewal_date},
            organization_id=org.id,
        )
        assigned_by = db.get(User, ctx.user_id).username if ctx.user_id else None
        return _subscription_dict(subscription, plan, org.name, assigned_by)


def list_subscriptions(
    ctx: AuthContext, *, organization_id: int | None = None, include_ended: bool = True
) -> list[dict[str, Any]]:
    """Subscriptions with renewal metadata, newest first (optionally one organization)."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        query = (
            select(Subscription, SubscriptionPlan, Organization.name, User.username)
            .join(SubscriptionPlan, SubscriptionPlan.id == Subscription.plan_id)
            .join(Organization, Organization.id == Subscription.organization_id)
            .outerjoin(User, User.id == Subscription.assigned_by_id)
        )
        if organization_id is not None:
            query = query.where(Subscription.organization_id == _get_org(db, organization_id).id)
        if not include_ended:
            query = query.where(Subscription.status == "active")
        rows = db.execute(query.order_by(Subscription.started_at.desc(), Subscription.id.desc())).all()
        return [_subscription_dict(sub, plan, org_name, username) for sub, plan, org_name, username in rows]


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


def _percentile_nearest_rank(values: list[int], percentile: float) -> int:
    """Nearest-rank percentile: the smallest value with at least ``percentile`` % of values at or below it."""
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def _latency(db: Session, column: Any, since: datetime) -> dict[str, Any]:
    """Exact count, mean and nearest-rank p95 of a ``duration_ms`` column since ``since``."""
    table = column.class_
    durations = [
        int(value)
        for value in db.execute(
            select(column).where(column.is_not(None), table.created_at >= since)
        ).scalars()
    ]
    if not durations:
        return {"count": 0, "average_ms": None, "p95_ms": None}
    return {
        "count": len(durations),
        "average_ms": sum(durations) / len(durations),
        "p95_ms": _percentile_nearest_rank(durations, 95),
    }


def _users_by_role(db: Session) -> dict[str, dict[str, int]]:
    result = {role: {"active": 0, "total": 0} for role in USER_ROLES}
    for role, is_active, count in db.execute(
        select(User.role, User.is_active, func.count(User.id)).group_by(User.role, User.is_active)
    ).all():
        bucket = result.setdefault(role, {"active": 0, "total": 0})
        bucket["total"] += count
        if is_active:
            bucket["active"] += count
    return result


def _analysis_jobs(db: Session, since: datetime) -> dict[str, Any]:
    total = db.execute(select(func.count(AnalysisHistory.id))).scalar_one()
    recent = dict(
        db.execute(
            select(AnalysisHistory.status, func.count(AnalysisHistory.id))
            .where(AnalysisHistory.created_at >= since)
            .group_by(AnalysisHistory.status)
        ).all()
    )
    return {
        "total": int(total),
        "last_30_days": int(sum(recent.values())),
        "last_30_days_success": int(recent.get("success", 0)),
        "last_30_days_error": int(recent.get("error", 0)),
        "last_30_days_by_status": {k: int(v) for k, v in recent.items()},
    }


def _db_storage_bytes(db: Session) -> int:
    """Platform-wide bytes, counted exactly like each organization's storage quota
    (originals, live versions and purged files that could not be deleted yet)."""
    org_ids = db.execute(select(Organization.id)).scalars().all()
    return sum(subscriptions.org_storage_bytes(db, org_id) for org_id in org_ids)


def platform_metrics(ctx: AuthContext) -> dict[str, Any]:
    """Platform-wide totals, storage, errors, latency and usage.

    Latency figures are exact: the mean of every recorded ``duration_ms`` in
    the last 30 days and the 95th percentile by the nearest-rank method
    (``None`` when nothing was recorded).
    """
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        now = timeutil.utcnow()
        since_30 = now - timedelta(days=METRICS_WINDOW_DAYS)
        since_7 = now - timedelta(days=ERROR_WINDOW_DAYS)
        org_status = {s.value: 0 for s in OrgStatus}
        org_status.update(
            dict(
                db.execute(
                    select(Organization.status, func.count(Organization.id)).group_by(Organization.status)
                ).all()
            )
        )
        dataset_status = dict(
            db.execute(select(Dataset.status, func.count(Dataset.id)).group_by(Dataset.status)).all()
        )
        errors_7 = db.execute(
            select(func.count(ErrorEvent.id)).where(ErrorEvent.created_at >= since_7)
        ).scalar_one()
        usage = {
            metric: float(total or 0)
            for metric, total in db.execute(
                select(UsageRecord.metric, func.sum(UsageRecord.quantity))
                .where(UsageRecord.created_at >= since_30)
                .group_by(UsageRecord.metric)
            ).all()
        }
        return {
            "generated_at": now,
            "organizations": {"total": int(sum(org_status.values())), "by_status": org_status},
            "users_by_role": _users_by_role(db),
            "datasets": {
                "active": int(dataset_status.get(DatasetStatus.ACTIVE.value, 0)),
                "archived": int(dataset_status.get(DatasetStatus.ARCHIVED.value, 0)),
            },
            "analysis_jobs": _analysis_jobs(db, since_30),
            "storage": {
                "recorded_bytes": _db_storage_bytes(db),
                "disk_bytes": storage.directory_size_bytes(get_settings().storage_root),
            },
            "errors_last_7_days": int(errors_7),
            "latency_last_30_days": {
                "analysis": _latency(db, AnalysisHistory.duration_ms, since_30),
                "cleaning": _latency(db, CleaningOperation.duration_ms, since_30),
            },
            "usage_last_30_days": usage,
        }


def list_error_events(ctx: AuthContext, *, limit: int = 100) -> list[dict[str, Any]]:
    """Most recent application errors (already redacted; no stack traces are stored)."""
    with session_scope() as db:
        authorize_super_admin(db, ctx)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_ERROR_EVENTS:
            raise ValidationError(f"The limit must be between 1 and {MAX_ERROR_EVENTS}.")
        rows = db.execute(
            select(ErrorEvent, Organization.name)
            .outerjoin(Organization, Organization.id == ErrorEvent.organization_id)
            .order_by(ErrorEvent.created_at.desc(), ErrorEvent.id.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": event.id,
                "created_at": event.created_at,
                "organization_id": event.organization_id,
                "organization_name": org_name,
                "user_id": event.user_id,
                "component": event.component,
                "error_type": event.error_type,
                "message": event.message,
                "reference": event.reference,
            }
            for event, org_name in rows
        ]
