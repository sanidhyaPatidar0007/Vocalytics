"""Plan, feature and limit lookups plus backend enforcement of plan limits.

Plan administration (creating plans, toggling features, assigning plans) lives
in ``modules.super_admin``. This module only answers "what does this
organization's active plan allow?" and raises when a limit would be exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config.settings import get_settings
from core import storage
from core.constants import FEATURE_LABELS, DatasetStatus, Feature
from core.errors import (
    FeatureNotAvailableError,
    PermissionDeniedError,
    PlanLimitExceededError,
    StorageQuotaExceededError,
)
from database.models import Dataset, DatasetVersion, PlanFeature, Subscription, SubscriptionPlan, User

MB = 1024 * 1024


@dataclass(frozen=True)
class PlanLimits:
    plan_id: int
    plan_code: str
    plan_name: str
    max_users: int
    max_storage_mb: int
    max_upload_mb: int
    max_datasets: int
    max_monthly_analyses: int
    max_versions_per_dataset: int

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * MB

    @property
    def max_storage_bytes(self) -> int:
        return self.max_storage_mb * MB


def get_active_subscription(db: Session, organization_id: int) -> Subscription | None:
    return db.execute(
        select(Subscription).where(
            Subscription.organization_id == organization_id, Subscription.status == "active"
        )
    ).scalar_one_or_none()


def get_active_plan(db: Session, organization_id: int) -> SubscriptionPlan:
    subscription = get_active_subscription(db, organization_id)
    if subscription is None:
        raise PermissionDeniedError(
            "Your organization has no active subscription. Contact the platform administrator."
        )
    plan = db.get(SubscriptionPlan, subscription.plan_id)
    if plan is None:
        raise PermissionDeniedError("Your organization's plan is not available.")
    return plan


def enabled_features(db: Session, organization_id: int) -> frozenset[str]:
    subscription = get_active_subscription(db, organization_id)
    if subscription is None:
        return frozenset()
    rows = db.execute(
        select(PlanFeature.feature_key).where(
            PlanFeature.plan_id == subscription.plan_id, PlanFeature.enabled.is_(True)
        )
    ).scalars()
    return frozenset(rows)


def org_has_feature(db: Session, organization_id: int, feature: Feature | str) -> bool:
    key = feature.value if isinstance(feature, Feature) else str(feature)
    return key in enabled_features(db, organization_id)


def require_feature(db: Session, organization_id: int, feature: Feature | str) -> None:
    key = feature.value if isinstance(feature, Feature) else str(feature)
    if not org_has_feature(db, organization_id, key):
        label = FEATURE_LABELS.get(key, key)
        raise FeatureNotAvailableError(f"'{label}' is not included in your organization's plan.")


def get_limits(db: Session, organization_id: int) -> PlanLimits:
    plan = get_active_plan(db, organization_id)
    ceiling = get_settings().absolute_upload_ceiling_mb
    return PlanLimits(
        plan_id=plan.id,
        plan_code=plan.code,
        plan_name=plan.name,
        max_users=plan.max_users,
        max_storage_mb=plan.max_storage_mb,
        max_upload_mb=min(plan.max_upload_mb, ceiling),
        max_datasets=plan.max_datasets,
        max_monthly_analyses=plan.max_monthly_analyses,
        max_versions_per_dataset=plan.max_versions_per_dataset,
    )


def org_storage_bytes(db: Session, organization_id: int) -> int:
    """Bytes held by original uploads, every non-purged version file, and purged
    version files that could not be deleted yet (for example, still open on Windows)."""
    originals = db.execute(
        select(func.coalesce(func.sum(Dataset.original_size_bytes), 0)).where(
            Dataset.organization_id == organization_id
        )
    ).scalar_one()
    versions = db.execute(
        select(func.coalesce(func.sum(DatasetVersion.size_bytes), 0)).where(
            DatasetVersion.organization_id == organization_id, DatasetVersion.is_purged.is_(False)
        )
    ).scalar_one()
    pending = db.execute(
        select(DatasetVersion.file_path, DatasetVersion.size_bytes).where(
            DatasetVersion.organization_id == organization_id,
            DatasetVersion.is_purged.is_(True),
            DatasetVersion.file_pending_delete.is_(True),
        )
    ).all()
    leftover = sum(size for path, size in pending if _file_exists(path))
    return int(originals) + int(versions) + int(leftover)


def _file_exists(relative_path: str) -> bool:
    try:
        return storage.resolve_relative(relative_path).exists()
    except (PermissionDeniedError, OSError):
        return False


def enforce_upload_size(db: Session, organization_id: int, size_bytes: int) -> None:
    limits = get_limits(db, organization_id)
    if size_bytes > limits.max_upload_bytes:
        raise PlanLimitExceededError(
            f"The file is {size_bytes / MB:.1f} MB; your plan allows uploads up to {limits.max_upload_mb} MB."
        )


def enforce_storage_quota(
    db: Session, organization_id: int, additional_bytes: int, *, freed_bytes: int = 0
) -> None:
    limits = get_limits(db, organization_id)
    projected = org_storage_bytes(db, organization_id) + max(additional_bytes, 0) - max(freed_bytes, 0)
    if projected > limits.max_storage_bytes:
        raise StorageQuotaExceededError(
            f"This would use {projected / MB:.1f} MB of storage; your plan allows {limits.max_storage_mb} MB. "
            "Archive or clean up datasets, or ask for a larger plan."
        )


def active_dataset_count(db: Session, organization_id: int) -> int:
    return int(
        db.execute(
            select(func.count(Dataset.id)).where(
                Dataset.organization_id == organization_id, Dataset.status == DatasetStatus.ACTIVE.value
            )
        ).scalar_one()
    )


def enforce_dataset_quota(db: Session, organization_id: int) -> None:
    limits = get_limits(db, organization_id)
    if active_dataset_count(db, organization_id) >= limits.max_datasets:
        raise PlanLimitExceededError(
            f"Your plan allows {limits.max_datasets} active datasets. Archive a dataset or upgrade the plan."
        )


def active_user_count(db: Session, organization_id: int) -> int:
    return int(
        db.execute(
            select(func.count(User.id)).where(
                User.organization_id == organization_id, User.is_active.is_(True)
            )
        ).scalar_one()
    )


def enforce_user_quota(db: Session, organization_id: int) -> None:
    limits = get_limits(db, organization_id)
    if active_user_count(db, organization_id) >= limits.max_users:
        raise PlanLimitExceededError(f"Your plan allows {limits.max_users} active users.")
