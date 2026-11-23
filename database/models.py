"""SQLAlchemy ORM models (SQLite).

Every organization-owned table carries ``organization_id``. Services must
always filter by the organization taken from the authenticated server-side
context (see ``auth.authorization``), never by an id supplied by the browser.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from core import timeutil


def _now() -> datetime:
    return timeutil.utcnow()


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now, nullable=False)


# ---------------------------------------------------------------------------
# Tenancy, plans and subscriptions
# ---------------------------------------------------------------------------


class Organization(TimestampMixin, Base):
    __tablename__ = "organizations"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'suspended', 'deactivated')", name="ck_organizations_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    contact_email: Mapped[str | None] = mapped_column(String(254))
    settings_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status_reason: Mapped[str | None] = mapped_column(Text)
    status_changed_at: Mapped[datetime | None] = mapped_column(DateTime)


class SubscriptionPlan(TimestampMixin, Base):
    __tablename__ = "subscription_plans"
    __table_args__ = (
        CheckConstraint("price_cents >= 0", name="ck_plans_price"),
        CheckConstraint("billing_period IN ('monthly', 'quarterly', 'yearly')", name="ck_plans_period"),
        CheckConstraint(
            "max_users > 0 AND max_storage_mb > 0 AND max_upload_mb > 0 AND max_datasets > 0 "
            "AND max_monthly_analyses > 0 AND max_versions_per_dataset >= 2",
            name="ck_plans_limits",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    billing_period: Mapped[str] = mapped_column(String(20), nullable=False, default="monthly")
    max_users: Mapped[int] = mapped_column(Integer, nullable=False)
    max_storage_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    max_upload_mb: Mapped[int] = mapped_column(Integer, nullable=False)
    max_datasets: Mapped[int] = mapped_column(Integer, nullable=False)
    max_monthly_analyses: Mapped[int] = mapped_column(Integer, nullable=False)
    max_versions_per_dataset: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    features: Mapped[list[PlanFeature]] = relationship(back_populates="plan", cascade="all, delete-orphan")


class PlanFeature(Base):
    __tablename__ = "plan_features"
    __table_args__ = (UniqueConstraint("plan_id", "feature_key", name="uq_plan_features_plan_feature"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        ForeignKey("subscription_plans.id", ondelete="CASCADE"), nullable=False
    )
    feature_key: Mapped[str] = mapped_column(String(50), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now, nullable=False)

    plan: Mapped[SubscriptionPlan] = relationship(back_populates="features")


class Subscription(TimestampMixin, Base):
    __tablename__ = "subscriptions"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'ended')", name="ck_subscriptions_status"),
        # At most one active subscription per organization.
        Index(
            "uq_subscriptions_one_active",
            "organization_id",
            unique=True,
            sqlite_where=text("status = 'active'"),
        ),
        Index("ix_subscriptions_org", "organization_id", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    plan_id: Mapped[int] = mapped_column(ForeignKey("subscription_plans.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    renewal_date: Mapped[datetime | None] = mapped_column(DateTime)
    auto_renew: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    billing_reference: Mapped[str | None] = mapped_column(String(100))
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    assigned_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))

    plan: Mapped[SubscriptionPlan] = relationship()


# ---------------------------------------------------------------------------
# Users, sessions and permissions
# ---------------------------------------------------------------------------


class User(TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('super_admin', 'org_admin', 'analyst')", name="ck_users_role"),
        CheckConstraint(
            "(role = 'super_admin' AND organization_id IS NULL) "
            "OR (role <> 'super_admin' AND organization_id IS NOT NULL)",
            name="ck_users_role_org",
        ),
        CheckConstraint("failed_login_count >= 0", name="ck_users_failed_count"),
        Index("ix_users_org_role", "organization_id", "role"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int | None] = mapped_column(ForeignKey("organizations.id"))
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    failed_login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)
    password_changed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))


class UserSession(Base):
    __tablename__ = "user_sessions"
    __table_args__ = (Index("ix_user_sessions_user", "user_id", "revoked_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    organization_id: Mapped[int | None] = mapped_column(ForeignKey("organizations.id"))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime)
    revoke_reason: Mapped[str | None] = mapped_column(String(50))
    user_agent: Mapped[str | None] = mapped_column(String(300))


class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    __table_args__ = (Index("ix_login_attempts_username_time", "username", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class UserPermission(Base):
    __tablename__ = "user_permissions"
    __table_args__ = (UniqueConstraint("user_id", "permission", name="uq_user_permissions_user_perm"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    permission: Mapped[str] = mapped_column(String(50), nullable=False)
    granted_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


# ---------------------------------------------------------------------------
# Datasets, versions and operations
# ---------------------------------------------------------------------------


class Dataset(TimestampMixin, Base):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_datasets_org_name"),
        CheckConstraint("source_type IN ('csv', 'xlsx', 'xls', 'derived')", name="ck_datasets_source"),
        CheckConstraint("status IN ('active', 'archived')", name="ck_datasets_status"),
        Index("ix_datasets_org_status", "organization_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_type: Mapped[str] = mapped_column(String(10), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_file_path: Mapped[str | None] = mapped_column(String(500))
    original_sha256: Mapped[str | None] = mapped_column(String(64))
    original_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sheet_name: Mapped[str | None] = mapped_column(String(255))
    import_options_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    import_report_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    current_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id", use_alter=True, name="fk_datasets_current_version")
    )
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version_number", name="uq_dataset_versions_number"),
        CheckConstraint("version_number >= 1", name="ck_dataset_versions_number"),
        CheckConstraint(
            "row_count >= 0 AND column_count >= 0 AND size_bytes >= 0", name="ck_dataset_versions_counts"
        ),
        Index("ix_dataset_versions_org_dataset", "organization_id", "dataset_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_json: Mapped[str] = mapped_column(Text, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float)
    quality_json: Mapped[str | None] = mapped_column(Text)
    operation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    is_pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_purged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    purged_at: Mapped[datetime | None] = mapped_column(DateTime)
    # Set when a version is purged; cleared once its file is confirmed deleted. While set and the
    # file still exists, its bytes count toward the storage quota and the deletion is retried.
    file_pending_delete: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class CleaningOperation(Base):
    __tablename__ = "cleaning_operations"
    __table_args__ = (
        CheckConstraint("status IN ('applied', 'failed', 'cancelled')", name="ck_cleaning_operations_status"),
        Index("ix_cleaning_operations_org_dataset", "organization_id", "dataset_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    input_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    output_version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    operation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="applied")
    rows_before: Mapped[int | None] = mapped_column(Integer)
    rows_after: Mapped[int | None] = mapped_column(Integer)
    columns_before: Mapped[int | None] = mapped_column(Integer)
    columns_after: Mapped[int | None] = mapped_column(Integer)
    rows_added: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_removed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_affected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cells_changed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quality_before: Mapped[float | None] = mapped_column(Float)
    quality_after: Mapped[float | None] = mapped_column(Float)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class DatasetAccess(Base):
    """Explicit dataset grants for analysts (org admins see every dataset)."""

    __tablename__ = "dataset_access"
    __table_args__ = (UniqueConstraint("dataset_id", "user_id", name="uq_dataset_access_dataset_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    granted_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


# ---------------------------------------------------------------------------
# Analysis, reports and usage
# ---------------------------------------------------------------------------


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    __table_args__ = (
        CheckConstraint("input_mode IN ('text', 'voice', 'builder')", name="ck_analysis_input_mode"),
        CheckConstraint(
            "status IN ('success', 'clarification', 'error', 'rejected')", name="ck_analysis_status"
        ),
        Index("ix_analysis_history_org_time", "organization_id", "created_at"),
        Index("ix_analysis_history_user_time", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"))
    version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    question: Mapped[str] = mapped_column(Text, nullable=False)
    input_mode: Mapped[str] = mapped_column(String(20), nullable=False, default="text")
    planner: Mapped[str | None] = mapped_column(String(20))
    plan_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text)
    result_preview_json: Mapped[str | None] = mapped_column(Text)
    chart_type: Mapped[str | None] = mapped_column(String(30))
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    model_name: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class SavedReport(TimestampMixin, Base):
    __tablename__ = "saved_reports"
    __table_args__ = (Index("ix_saved_reports_org", "organization_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id"))
    analysis_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_history.id"))
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False, default="")
    plan_json: Mapped[str | None] = mapped_column(Text)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    result_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    chart_json: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")


class UsageRecord(Base):
    __tablename__ = "usage_records"
    __table_args__ = (Index("ix_usage_records_org_metric_time", "organization_id", "metric", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    metric: Mapped[str] = mapped_column(String(50), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


# ---------------------------------------------------------------------------
# Anomaly detection and notifications
# ---------------------------------------------------------------------------


class AnomalyRule(TimestampMixin, Base):
    __tablename__ = "anomaly_rules"
    __table_args__ = (
        UniqueConstraint("organization_id", "name", name="uq_anomaly_rules_org_name"),
        CheckConstraint(
            "method IN ('iqr', 'zscore', 'period_change', 'metric_drop', 'baseline_deviation', 'missing_increase')",
            name="ck_anomaly_rules_method",
        ),
        CheckConstraint(
            "severity IN ('low', 'medium', 'high', 'critical')", name="ck_anomaly_rules_severity"
        ),
        CheckConstraint(
            "schedule IN ('manual', 'hourly', 'daily', 'weekly')", name="ck_anomaly_rules_schedule"
        ),
        CheckConstraint("cooldown_minutes >= 0", name="ck_anomaly_rules_cooldown"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    method: Mapped[str] = mapped_column(String(30), nullable=False)
    metric_column: Mapped[str | None] = mapped_column(String(255))
    aggregation: Mapped[str] = mapped_column(String(20), nullable=False, default="sum")
    date_column: Mapped[str | None] = mapped_column(String(255))
    frequency: Mapped[str | None] = mapped_column(String(10))
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    recipients_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    schedule: Mapped[str] = mapped_column(String(20), nullable=False, default="manual")
    cooldown_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    __table_args__ = (
        CheckConstraint(
            "status IN ('new', 'notified', 'suppressed', 'acknowledged')", name="ck_anomaly_events_status"
        ),
        # One event per finding: repeated checks of the same rule never duplicate an alert.
        UniqueConstraint("rule_id", "fingerprint", name="uq_anomaly_events_rule_fingerprint"),
        Index("ix_anomaly_events_rule_fp_time", "organization_id", "rule_id", "fingerprint", "detected_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    rule_id: Mapped[int] = mapped_column(ForeignKey("anomaly_rules.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    version_id: Mapped[int | None] = mapped_column(ForeignKey("dataset_versions.id"))
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    observed_value: Mapped[float | None] = mapped_column(Float)
    expected_value: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="new")
    suppressed_reason: Mapped[str | None] = mapped_column(String(100))
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
    acknowledged_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime)


class NotificationLog(Base):
    __tablename__ = "notification_logs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('sent', 'failed', 'skipped', 'suppressed')", name="ck_notification_logs_status"
        ),
        Index("ix_notification_logs_org_time", "organization_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("anomaly_events.id"))
    rule_id: Mapped[int | None] = mapped_column(ForeignKey("anomaly_rules.id"))
    channel: Mapped[str] = mapped_column(String(20), nullable=False, default="email")
    recipient: Mapped[str] = mapped_column(String(254), nullable=False)
    subject: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


# ---------------------------------------------------------------------------
# Audit, monitoring and schema bookkeeping
# ---------------------------------------------------------------------------


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_org_time", "organization_id", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int | None] = mapped_column(ForeignKey("organizations.id"))
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    actor_role: Mapped[str | None] = mapped_column(String(20))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    target_type: Mapped[str | None] = mapped_column(String(50))
    target_id: Mapped[str | None] = mapped_column(String(64))
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class ErrorEvent(Base):
    __tablename__ = "error_events"
    __table_args__ = (Index("ix_error_events_time", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int | None] = mapped_column(ForeignKey("organizations.id"))
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    component: Mapped[str] = mapped_column(String(100), nullable=False)
    error_type: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    reference: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    applied_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_now)
