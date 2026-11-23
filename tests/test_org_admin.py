"""Organization Admin workflows (modules/organization_admin.py)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

from auth.authorization import has_permission, load_context
from auth.password_manager import verify_password
from core import timeutil
from core.constants import ANALYST_DEFAULT_PERMISSIONS, Permission
from core.errors import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    PlanLimitExceededError,
    ValidationError,
)
from database import session_scope
from database.models import (
    AnalysisHistory,
    AnomalyEvent,
    AnomalyRule,
    AuditLog,
    CleaningOperation,
    Dataset,
    DatasetVersion,
    Organization,
    UsageRecord,
    User,
    UserSession,
)
from modules import organization_admin as org_admin
from services import dataset_service, versioning_service

STRONG = "Str0ng!Passw0rd"
NEW_PASSWORD = "N3w!Secure#Pass"


def _open_session(user_id: int, org_id: int) -> int:
    with session_scope() as db:
        row = UserSession(
            user_id=user_id,
            organization_id=org_id,
            token_hash=uuid.uuid4().hex + uuid.uuid4().hex,
            expires_at=timeutil.utcnow() + timedelta(hours=1),
        )
        db.add(row)
        db.flush()
        return row.id


def _revoked(session_id: int) -> tuple[bool, str | None]:
    with session_scope() as db:
        row = db.get(UserSession, session_id)
        return row.revoked_at is not None, row.revoke_reason


def _audit_count() -> int:
    with session_scope() as db:
        return db.execute(select(func.count(AuditLog.id))).scalar_one()


def _audit(action: str) -> list[AuditLog]:
    with session_scope() as db:
        return list(db.execute(select(AuditLog).where(AuditLog.action == action)).scalars())


@pytest.fixture
def other(factory, sales_frame):
    """A second organization with its own admin, analyst and dataset."""
    org = factory.organization("Other Co")
    admin = factory.org_admin(org)
    analyst = factory.analyst(org)
    dataset = factory.dataset(admin, sales_frame, "other sales")
    return {"org": org, "admin": admin, "analyst": analyst, "dataset": dataset}


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

ORG_ADMIN_CALLS = {
    "org_dashboard": lambda ctx: org_admin.org_dashboard(ctx),
    "list_users": lambda ctx: org_admin.list_users(ctx),
    "create_analyst": lambda ctx: org_admin.create_analyst(
        ctx, username="sneaky", email="s@example.com", full_name="S", password=STRONG
    ),
    "update_user": lambda ctx: org_admin.update_user(ctx, ctx.user_id or 1, full_name="Me"),
    "set_user_active": lambda ctx: org_admin.set_user_active(ctx, 1, False),
    "reset_user_password": lambda ctx: org_admin.reset_user_password(ctx, 1, NEW_PASSWORD),
    "set_user_permissions": lambda ctx: org_admin.set_user_permissions(ctx, ctx.user_id or 1, {"clean_data"}),
    "grant_dataset_access": lambda ctx: org_admin.grant_dataset_access(ctx, 1, ctx.user_id or 1),
    "revoke_dataset_access": lambda ctx: org_admin.revoke_dataset_access(ctx, 1, 1),
    "list_dataset_access": lambda ctx: org_admin.list_dataset_access(ctx, 1),
    "get_settings_view": lambda ctx: org_admin.get_settings_view(ctx),
    "update_org_settings": lambda ctx: org_admin.update_org_settings(ctx, {"date_dayfirst": True}),
    "usage_report": lambda ctx: org_admin.usage_report(ctx),
    "org_analysis_history": lambda ctx: org_admin.org_analysis_history(ctx),
}


@pytest.mark.parametrize("call", list(ORG_ADMIN_CALLS), ids=list(ORG_ADMIN_CALLS))
@pytest.mark.parametrize("role", ["analyst", "super_admin"])
def test_only_org_admins_may_call(factory, admin_ctx, sales_frame, call, role):
    factory.dataset(admin_ctx, sales_frame)
    ctx = factory.analyst(admin_ctx.organization_id, permissions=set(p.value for p in Permission))
    if role == "super_admin":
        ctx = factory.super_admin()
    before = _audit_count()
    with pytest.raises(PermissionDeniedError):
        ORG_ADMIN_CALLS[call](ctx)
    assert _audit_count() == before


def test_suspended_org_admin_is_blocked(admin_ctx, org_id):
    with session_scope() as db:
        db.get(Organization, org_id).status = "suspended"
    with pytest.raises(PermissionDeniedError, match="not active"):
        org_admin.list_users(admin_ctx)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def test_dashboard_reports_metadata_without_loading_files(factory, sales_frame, monkeypatch):
    org = factory.organization("Basic Shop", plan="basic")
    admin = factory.org_admin(org)
    analyst_id = factory.user(org, "analyst")
    factory.user(org, "analyst", active=False)
    first = factory.dataset(admin, sales_frame, "first")
    second = factory.dataset(admin, sales_frame, "second")
    archived = factory.dataset(admin, sales_frame, "archived")
    with session_scope() as db:
        for dataset_id, score in ((first, 80.0), (second, 90.0), (archived, 10.0)):
            dataset = db.get(Dataset, dataset_id)
            db.get(DatasetVersion, dataset.current_version_id).quality_score = score
        db.get(Dataset, archived).status = "archived"
        now = timeutil.utcnow()
        for minutes in range(12):
            db.add(
                CleaningOperation(
                    organization_id=org,
                    dataset_id=first,
                    user_id=analyst_id,
                    operation_type=f"op{minutes}",
                    status="applied" if minutes else "failed",
                    created_at=now + timedelta(minutes=minutes),
                )
            )
        rule = AnomalyRule(organization_id=org, dataset_id=first, name="r", method="iqr", threshold=1.5)
        db.add(rule)
        db.flush()
        for days in (1, 6, 8):
            db.add(
                AnomalyEvent(
                    organization_id=org,
                    rule_id=rule.id,
                    dataset_id=first,
                    fingerprint=f"fp{days}",
                    severity="low",
                    title="t",
                    detected_at=now - timedelta(days=days),
                )
            )
        for created in (now, now - timedelta(days=40)):
            db.add(UsageRecord(organization_id=org, metric="analysis_query", quantity=2, created_at=created))

    def no_files(*_args, **_kwargs):
        raise AssertionError("the dashboard must not read dataset files")

    monkeypatch.setattr(versioning_service, "read_parquet", no_files)
    monkeypatch.setattr(versioning_service, "iter_parquet", no_files)
    dashboard = org_admin.org_dashboard(admin)

    assert dashboard["plan_code"] == "basic" and dashboard["plan_name"] == "Basic"
    assert "voice_analysis" not in dashboard["features"] and "data_profiling" in dashboard["features"]
    assert dashboard["users"] == {"admins": 1, "analysts": 2, "active": 2, "inactive": 1, "total": 3}
    assert dashboard["datasets"] == {"active": 2, "archived": 1, "total": 3}
    assert dashboard["average_quality_score"] == 85.0
    assert dashboard["storage_used_bytes"] > 0 and dashboard["storage_quota_bytes"] == 2048 * 1024 * 1024
    assert dashboard["analyses_this_month"] == 2.0 and dashboard["max_monthly_analyses"] == 500
    recent = dashboard["recent_cleaning_operations"]
    assert len(recent) == 10 and recent[0]["operation_type"] == "op11"
    assert recent[0]["dataset_name"] == "first" and recent[0]["user"].startswith("analyst")
    assert dashboard["recent_anomaly_events"] == 2


def test_dashboard_for_empty_org(factory):
    admin = factory.org_admin(factory.organization())
    dashboard = org_admin.org_dashboard(admin)
    assert dashboard["average_quality_score"] is None
    assert dashboard["recent_cleaning_operations"] == [] and dashboard["datasets"]["total"] == 0


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def test_list_users_only_own_org(admin_ctx, analyst_ctx, other):
    users = org_admin.list_users(admin_ctx)
    assert [u["id"] for u in users] == [admin_ctx.user_id, analyst_ctx.user_id]
    admin_row, analyst_row = users
    assert admin_row["role"] == "org_admin" and admin_row["permissions"] == sorted(
        p.value for p in Permission
    )
    assert analyst_row["permissions"] == sorted(ANALYST_DEFAULT_PERMISSIONS)
    assert {"username", "email", "full_name", "is_active", "last_login_at", "created_at"} <= set(analyst_row)
    assert "password_hash" not in analyst_row


def test_create_analyst_defaults_and_audit(admin_ctx, org_id):
    user = org_admin.create_analyst(
        admin_ctx, username="  Jane.Doe ", email=" jane@example.com ", full_name=" Jane Doe ", password=STRONG
    )
    assert (
        user["username"] == "jane.doe"
        and user["email"] == "jane@example.com"
        and user["full_name"] == "Jane Doe"
    )
    assert user["role"] == "analyst" and user["organization_id"] == org_id
    assert user["permissions"] == sorted(ANALYST_DEFAULT_PERMISSIONS)
    with session_scope() as db:
        row = db.get(User, user["id"])
        assert verify_password(row.password_hash, STRONG) and row.created_by_id == admin_ctx.user_id
        ctx = load_context(db, user["id"])
        assert has_permission(db, ctx, Permission.EXPORT_DATA)
        assert not has_permission(db, ctx, Permission.CLEAN_DATA)
    audit = _audit("user.create")
    assert len(audit) == 1 and audit[0].organization_id == org_id and audit[0].target_id == str(user["id"])


def test_create_analyst_with_custom_permissions(admin_ctx):
    user = org_admin.create_analyst(
        admin_ctx,
        username="cleaner",
        email="c@example.com",
        full_name="C",
        password=STRONG,
        permissions={Permission.CLEAN_DATA.value, Permission.UPLOAD_DATASETS.value},
    )
    assert user["permissions"] == ["clean_data", "upload_datasets"]
    assert (
        org_admin.create_analyst(
            admin_ctx,
            username="viewer",
            email="v@example.com",
            full_name="V",
            password=STRONG,
            permissions=set(),
        )["permissions"]
        == []
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"username": "ab"},
        {"username": "x" * 65},
        {"username": "bad name"},
        {"username": "bad/name"},
        {"email": "no-at-sign"},
        {"email": "a@b"},
        {"email": "x" * 250 + "@a.com"},
        {"full_name": "n" * 201},
        {"password": "short1!"},
        {"password": "alllowercaseletters"},
        {"password": "Password123"},
        {"username": "maria", "password": "Maria!2024xyz"},
        {"permissions": {"admin_everything"}},
        {"permissions": "clean_data"},
    ],
)
def test_create_analyst_validation(admin_ctx, overrides):
    values = dict(username="maria", email="m@example.com", full_name="Maria", password=STRONG)
    values.update(overrides)
    with pytest.raises(ValidationError):
        org_admin.create_analyst(admin_ctx, **values)
    assert len(org_admin.list_users(admin_ctx)) == 1


def test_usernames_are_globally_unique(admin_ctx, other):
    org_admin.create_analyst(
        admin_ctx, username="shared", email="a@example.com", full_name="A", password=STRONG
    )
    with pytest.raises(ConflictError):
        org_admin.create_analyst(
            other["admin"], username="SHARED", email="b@example.com", full_name="B", password=STRONG
        )


def test_user_quota_enforced_on_create_and_reactivate(factory):
    org = factory.organization("Tiny", plan="basic")  # Basic allows 5 active users
    admin = factory.org_admin(org)
    created = [
        org_admin.create_analyst(
            admin, username=f"user{i}", email=f"u{i}@example.com", full_name="", password=STRONG
        )
        for i in range(4)
    ]
    with pytest.raises(PlanLimitExceededError):
        org_admin.create_analyst(
            admin, username="user9", email="u9@example.com", full_name="", password=STRONG
        )
    org_admin.set_user_active(admin, created[0]["id"], False)
    extra = org_admin.create_analyst(
        admin, username="user9", email="u9@example.com", full_name="", password=STRONG
    )
    with pytest.raises(PlanLimitExceededError):
        org_admin.set_user_active(admin, created[0]["id"], True)
    org_admin.set_user_active(admin, extra["id"], False)
    assert org_admin.set_user_active(admin, created[0]["id"], True)["is_active"] is True


def test_update_user(admin_ctx, analyst_ctx, other):
    updated = org_admin.update_user(
        admin_ctx, analyst_ctx.user_id, email="new@example.com", full_name="New Name"
    )
    assert updated["email"] == "new@example.com" and updated["full_name"] == "New Name"
    assert org_admin.update_user(admin_ctx, admin_ctx.user_id, full_name="Boss")["full_name"] == "Boss"
    with pytest.raises(ValidationError):
        org_admin.update_user(admin_ctx, analyst_ctx.user_id)
    with pytest.raises(ValidationError):
        org_admin.update_user(admin_ctx, analyst_ctx.user_id, email="broken")
    with pytest.raises(NotFoundError):
        org_admin.update_user(admin_ctx, other["analyst"].user_id, full_name="Hijack")
    with pytest.raises(NotFoundError):
        org_admin.update_user(admin_ctx, 99999, full_name="Ghost")
    with pytest.raises(ValidationError):
        org_admin.update_user(admin_ctx, True, full_name="Bool id")
    with session_scope() as db:
        assert db.get(User, other["analyst"].user_id).full_name != "Hijack"
    assert len(_audit("user.update")) == 2


def test_set_user_active_revokes_sessions_and_blocks_self(admin_ctx, analyst_ctx, org_id, other):
    session_id = _open_session(analyst_ctx.user_id, org_id)
    admin_session = _open_session(admin_ctx.user_id, org_id)
    with pytest.raises(ValidationError, match="your own account"):
        org_admin.set_user_active(admin_ctx, admin_ctx.user_id, False)
    result = org_admin.set_user_active(admin_ctx, analyst_ctx.user_id, False)
    assert result["is_active"] is False and result["sessions_revoked"] == 1
    assert _revoked(session_id) == (True, "user_deactivated")
    assert _revoked(admin_session) == (False, None)
    with pytest.raises(AuthenticationError):
        org_admin.list_users(analyst_ctx)
    with session_scope() as db, pytest.raises(AuthenticationError):
        load_context(db, analyst_ctx.user_id)
    # Repeating the current state is a no-op (no extra audit entry).
    assert org_admin.set_user_active(admin_ctx, analyst_ctx.user_id, False)["sessions_revoked"] == 0
    assert len(_audit("user.deactivate")) == 1
    assert org_admin.set_user_active(admin_ctx, analyst_ctx.user_id, True)["is_active"] is True
    assert len(_audit("user.activate")) == 1
    with pytest.raises(ValidationError):
        org_admin.set_user_active(admin_ctx, analyst_ctx.user_id, "no")
    with pytest.raises(NotFoundError):
        org_admin.set_user_active(admin_ctx, other["analyst"].user_id, False)


def test_admin_can_deactivate_another_admin(factory, admin_ctx, org_id):
    second = factory.org_admin(org_id)
    assert org_admin.set_user_active(admin_ctx, second.user_id, False)["is_active"] is False


def test_reset_user_password(admin_ctx, analyst_ctx, org_id, other):
    session_id = _open_session(analyst_ctx.user_id, org_id)
    with session_scope() as db:
        db.get(User, analyst_ctx.user_id).failed_login_count = 3
    result = org_admin.reset_user_password(admin_ctx, analyst_ctx.user_id, NEW_PASSWORD)
    assert result["must_change_password"] is True and result["sessions_revoked"] == 1
    assert _revoked(session_id) == (True, "password_reset")
    with session_scope() as db:
        user = db.get(User, analyst_ctx.user_id)
        assert verify_password(user.password_hash, NEW_PASSWORD) and not verify_password(
            user.password_hash, STRONG
        )
        assert user.must_change_password and user.failed_login_count == 0
    audit = _audit("user.reset_password")
    assert len(audit) == 1 and NEW_PASSWORD not in audit[0].details_json
    with pytest.raises(ValidationError):
        org_admin.reset_user_password(admin_ctx, analyst_ctx.user_id, "weak")
    with pytest.raises(NotFoundError):
        org_admin.reset_user_password(admin_ctx, other["admin"].user_id, NEW_PASSWORD)
    with session_scope() as db:
        assert verify_password(db.get(User, other["admin"].user_id).password_hash, STRONG)


def test_set_user_permissions(admin_ctx, analyst_ctx, other):
    result = org_admin.set_user_permissions(admin_ctx, analyst_ctx.user_id, {"clean_data", "export_data"})
    assert result["permissions"] == ["clean_data", "export_data"]
    with session_scope() as db:
        assert has_permission(db, analyst_ctx, Permission.CLEAN_DATA)
        assert not has_permission(db, analyst_ctx, Permission.USE_VOICE)
    details = json.loads(_audit("user.permissions")[0].details_json)
    assert details == {"added": ["clean_data"], "removed": ["save_reports", "use_voice"]}
    # Re-adding a removed permission in a later call works (no unique-constraint clash).
    again = org_admin.set_user_permissions(admin_ctx, analyst_ctx.user_id, {Permission.USE_VOICE})
    assert again["permissions"] == ["use_voice"]
    assert org_admin.set_user_permissions(admin_ctx, analyst_ctx.user_id, set())["permissions"] == []
    with pytest.raises(ValidationError):
        org_admin.set_user_permissions(admin_ctx, analyst_ctx.user_id, {"root"})
    with pytest.raises(ValidationError, match="analysts"):
        org_admin.set_user_permissions(admin_ctx, admin_ctx.user_id, {"clean_data"})
    with pytest.raises(NotFoundError):
        org_admin.set_user_permissions(admin_ctx, other["analyst"].user_id, {"clean_data"})


# ---------------------------------------------------------------------------
# Dataset grants
# ---------------------------------------------------------------------------


def test_dataset_grants_control_analyst_visibility(admin_ctx, analyst_ctx, factory, sales_frame, org_id):
    dataset_id = factory.dataset(admin_ctx, sales_frame, "sales")
    assert dataset_service.list_datasets(analyst_ctx) == []
    granted = org_admin.grant_dataset_access(admin_ctx, dataset_id, analyst_ctx.user_id)
    assert granted == {
        "dataset_id": dataset_id,
        "user_id": analyst_ctx.user_id,
        "has_access": True,
        "changed": True,
    }
    assert [d["dataset_id"] for d in dataset_service.list_datasets(analyst_ctx)] == [dataset_id]
    assert org_admin.grant_dataset_access(admin_ctx, dataset_id, analyst_ctx.user_id)["changed"] is False
    access = org_admin.list_dataset_access(admin_ctx, dataset_id)
    assert [a["user_id"] for a in access] == [analyst_ctx.user_id] and access[0][
        "granted_by_id"
    ] == admin_ctx.user_id

    revoked = org_admin.revoke_dataset_access(admin_ctx, dataset_id, analyst_ctx.user_id)
    assert revoked["has_access"] is False and revoked["changed"] is True
    assert dataset_service.list_datasets(analyst_ctx) == []
    assert org_admin.list_dataset_access(admin_ctx, dataset_id) == []
    assert org_admin.revoke_dataset_access(admin_ctx, dataset_id, analyst_ctx.user_id)["changed"] is False
    # Granting again after a revoke works.
    assert org_admin.grant_dataset_access(admin_ctx, dataset_id, analyst_ctx.user_id)["changed"] is True
    grants, revokes = _audit("dataset.grant"), _audit("dataset.revoke")
    assert len(grants) == 2 and len(revokes) == 1 and grants[0].organization_id == org_id


def test_dataset_grants_are_tenant_scoped(admin_ctx, analyst_ctx, factory, sales_frame, other):
    dataset_id = factory.dataset(admin_ctx, sales_frame, "mine")
    with pytest.raises(NotFoundError):
        org_admin.grant_dataset_access(admin_ctx, other["dataset"], analyst_ctx.user_id)
    with pytest.raises(NotFoundError):
        org_admin.grant_dataset_access(admin_ctx, dataset_id, other["analyst"].user_id)
    with pytest.raises(NotFoundError):
        org_admin.revoke_dataset_access(admin_ctx, other["dataset"], analyst_ctx.user_id)
    with pytest.raises(NotFoundError):
        org_admin.revoke_dataset_access(admin_ctx, dataset_id, other["analyst"].user_id)
    with pytest.raises(NotFoundError):
        org_admin.list_dataset_access(admin_ctx, other["dataset"])
    with pytest.raises(ValidationError):
        org_admin.grant_dataset_access(admin_ctx, dataset_id, admin_ctx.user_id)
    with pytest.raises(ValidationError):
        org_admin.list_dataset_access(admin_ctx, -1)
    assert dataset_service.list_datasets(other["analyst"]) == []


def test_archived_datasets_can_be_granted(admin_ctx, analyst_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame, "archived")
    dataset_service.set_dataset_status(admin_ctx, dataset_id, archived=True)
    assert org_admin.grant_dataset_access(admin_ctx, dataset_id, analyst_ctx.user_id)["has_access"] is True
    assert len(org_admin.list_dataset_access(admin_ctx, dataset_id)) == 1


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_view_and_update(admin_ctx, org_id, other):
    view = org_admin.get_settings_view(admin_ctx)
    assert view["date_dayfirst"] is False and view["text_missing_placeholder"] == "Not Provided"
    updated = org_admin.update_org_settings(
        admin_ctx, {"date_dayfirst": True, "missing_placeholders": [" NA ", "-", "NA", ""]}
    )
    assert updated["date_dayfirst"] is True and updated["missing_placeholders"] == ["NA", "-"]
    later = org_admin.update_org_settings(admin_ctx, {"alert_default_recipients": ["ops@example.com"]})
    assert later["date_dayfirst"] is True and later["alert_default_recipients"] == ["ops@example.com"]
    with session_scope() as db:
        stored = json.loads(db.get(Organization, org_id).settings_json)
        assert stored == {
            "date_dayfirst": True,
            "missing_placeholders": ["NA", "-"],
            "alert_default_recipients": ["ops@example.com"],
        }
    assert org_admin.get_settings_view(other["admin"])["date_dayfirst"] is False
    audit = _audit("org.settings.update")
    assert len(audit) == 2 and audit[0].organization_id == org_id
    assert json.loads(audit[0].details_json)["keys"] == ["date_dayfirst", "missing_placeholders"]


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"theme": "dark"},
        {"date_dayfirst": "yes"},
        {"text_missing_placeholder": ""},
        {"alert_default_recipients": ["nope"]},
    ],
)
def test_settings_validation(admin_ctx, org_id, changes):
    with pytest.raises(ValidationError):
        org_admin.update_org_settings(admin_ctx, changes)
    with session_scope() as db:
        assert json.loads(db.get(Organization, org_id).settings_json or "{}") == {}


# ---------------------------------------------------------------------------
# Usage and analysis history
# ---------------------------------------------------------------------------


def test_usage_report_groups_by_calendar_month(admin_ctx, org_id, other, monkeypatch):
    monkeypatch.setattr(timeutil, "utcnow", lambda: datetime(2026, 2, 10, 12, 0))
    rows = [
        (org_id, "analysis_query", 1, datetime(2026, 2, 1, 0, 0)),
        (org_id, "analysis_query", 2, datetime(2026, 2, 9)),
        (org_id, "upload_bytes", 4096, datetime(2026, 1, 31, 23, 59)),
        (org_id, "analysis_query", 5, datetime(2025, 9, 1)),
        (org_id, "analysis_query", 7, datetime(2025, 8, 31, 23, 59)),  # outside a 6-month window
        (other["org"], "analysis_query", 100, datetime(2026, 2, 5)),
    ]
    with session_scope() as db:
        for org, metric, quantity, created in rows:
            db.add(UsageRecord(organization_id=org, metric=metric, quantity=quantity, created_at=created))
    report = org_admin.usage_report(admin_ctx)
    assert [r["month"] for r in report] == ["2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
    by_month = {r["month"]: r["totals"] for r in report}
    assert by_month["2026-02"]["analysis_query"] == 3.0
    assert by_month["2026-01"]["upload_bytes"] == 4096.0 and by_month["2026-01"]["analysis_query"] == 0.0
    assert by_month["2025-09"]["analysis_query"] == 5.0
    assert by_month["2025-10"]["export"] == 0.0 and "voice_query" in by_month["2025-10"]
    assert report[-1]["limits"]["max_monthly_analyses"] == 10000
    assert report[-1]["analysis_quota_used_fraction"] == 3 / 10000
    single = org_admin.usage_report(admin_ctx, months=1)
    assert len(single) == 1 and single[0]["month_start"] == datetime(2026, 2, 1)
    for bad in (0, 37, "6", True):
        with pytest.raises(ValidationError):
            org_admin.usage_report(admin_ctx, months=bad)


def _add_history(org: int, user_id: int | None, dataset_id: int | None, question: str, minutes: int) -> None:
    with session_scope() as db:
        db.add(
            AnalysisHistory(
                organization_id=org,
                user_id=user_id,
                dataset_id=dataset_id,
                question=question,
                status="success",
                planner="llm",
                duration_ms=minutes * 10,
                created_at=timeutil.utcnow() + timedelta(minutes=minutes),
            )
        )


def test_org_analysis_history_filters_and_isolation(admin_ctx, analyst_ctx, factory, sales_frame, other):
    dataset_id = factory.dataset(admin_ctx, sales_frame, "sales")
    second_id = factory.dataset(admin_ctx, sales_frame, "returns")
    _add_history(admin_ctx.organization_id, analyst_ctx.user_id, dataset_id, "total revenue", 1)
    _add_history(admin_ctx.organization_id, admin_ctx.user_id, second_id, "top products", 2)
    _add_history(admin_ctx.organization_id, None, None, "system check", 3)
    _add_history(other["org"], other["analyst"].user_id, other["dataset"], "secret question", 4)

    history = org_admin.org_analysis_history(admin_ctx)
    assert [h["question"] for h in history] == ["system check", "top products", "total revenue"]
    newest = history[-1]
    assert newest["dataset_name"] == "sales" and newest["username"] == analyst_ctx.username
    assert newest["planner"] == "llm" and newest["duration_ms"] == 10 and newest["status"] == "success"
    assert history[0]["username"] == "system" and history[0]["dataset_name"] is None
    assert [
        h["question"] for h in org_admin.org_analysis_history(admin_ctx, user_id=analyst_ctx.user_id)
    ] == ["total revenue"]
    assert [h["question"] for h in org_admin.org_analysis_history(admin_ctx, dataset_id=second_id)] == [
        "top products"
    ]
    assert len(org_admin.org_analysis_history(admin_ctx, limit=1)) == 1
    with pytest.raises(NotFoundError):
        org_admin.org_analysis_history(admin_ctx, user_id=other["analyst"].user_id)
    with pytest.raises(NotFoundError):
        org_admin.org_analysis_history(admin_ctx, dataset_id=other["dataset"])
    for bad in (0, 1001, "10"):
        with pytest.raises(ValidationError):
            org_admin.org_analysis_history(admin_ctx, limit=bad)
    assert [h["question"] for h in org_admin.org_analysis_history(other["admin"])] == ["secret question"]


def test_import_defaults_are_readable_by_analysts_only_within_their_org(factory, org_id, admin_ctx):
    """Analysts prefill the import form from the organization's placeholders and date order,
    without access to the rest of the admin-only settings."""
    from core.errors import PermissionDeniedError as _Denied
    from modules import org_settings

    org_admin.update_org_settings(
        admin_ctx,
        {org_settings.MISSING_PLACEHOLDERS: ["n/a", "-"], org_settings.DATE_DAYFIRST: True,
         org_settings.ALERT_DEFAULT_RECIPIENTS: ["ops@example.com"]},
    )
    analyst = factory.analyst(org_id)
    defaults = org_settings.get_import_defaults(analyst)
    assert defaults == {
        org_settings.MISSING_PLACEHOLDERS: ["n/a", "-"],
        org_settings.DATE_DAYFIRST: True,
        org_settings.TEXT_PLACEHOLDER: "Not Provided",
    }
    other = factory.analyst(factory.organization("Other Co"))
    assert org_settings.get_import_defaults(other)[org_settings.MISSING_PLACEHOLDERS] != ["n/a", "-"]
    with pytest.raises(_Denied):
        org_settings.get_import_defaults(factory.super_admin())
