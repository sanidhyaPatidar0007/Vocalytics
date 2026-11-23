"""Super Admin workflows (modules/super_admin.py)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select, update

from auth.authorization import authorize
from config.settings import set_settings
from core import storage, timeutil
from core.constants import ALL_FEATURES, BASIC_FEATURES, Feature
from core.errors import (
    ConflictError,
    FeatureNotAvailableError,
    NotFoundError,
    PermissionDeniedError,
    PlanLimitExceededError,
    ValidationError,
)
from database import session_scope
from database.models import (
    AnalysisHistory,
    AuditLog,
    CleaningOperation,
    Dataset,
    DatasetVersion,
    ErrorEvent,
    Organization,
    PlanFeature,
    Subscription,
    SubscriptionPlan,
    UsageRecord,
    User,
    UserSession,
)
from modules import subscriptions, super_admin
from services.audit_service import record_error

STRONG = "Str0ng!Passw0rd"


@pytest.fixture
def root_ctx(factory):
    return factory.super_admin()


def _plan_id(code: str) -> int:
    with session_scope() as db:
        return db.execute(select(SubscriptionPlan.id).where(SubscriptionPlan.code == code)).scalar_one()


def _open_session(user_id: int, org_id: int | None) -> int:
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


def _session_state(session_id: int) -> tuple[datetime | None, str | None]:
    with session_scope() as db:
        row = db.get(UserSession, session_id)
        return row.revoked_at, row.revoke_reason


def _audit(action: str) -> list[AuditLog]:
    with session_scope() as db:
        return list(db.execute(select(AuditLog).where(AuditLog.action == action)).scalars())


def _plan_kwargs(**overrides):
    values = dict(
        code="team",
        name="Team",
        description="Mid tier",
        price_cents=9900,
        currency="eur",
        billing_period="yearly",
        max_users=10,
        max_storage_mb=4096,
        max_upload_mb=100,
        max_datasets=40,
        max_monthly_analyses=1000,
        max_versions_per_dataset=5,
        features={Feature.CSV_XLSX_UPLOAD.value, Feature.VOICE_ANALYSIS.value},
    )
    values.update(overrides)
    return values


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

SUPER_ADMIN_CALLS = {
    "list_organizations": lambda ctx: super_admin.list_organizations(ctx),
    "create_organization": lambda ctx: super_admin.create_organization(ctx, name="New Org"),
    "update_organization": lambda ctx: super_admin.update_organization(ctx, 1, name="Renamed"),
    "set_organization_status": lambda ctx: super_admin.set_organization_status(ctx, 1, "suspended"),
    "list_plans": lambda ctx: super_admin.list_plans(ctx),
    "create_plan": lambda ctx: super_admin.create_plan(ctx, **_plan_kwargs()),
    "update_plan": lambda ctx: super_admin.update_plan(ctx, 1, name="X"),
    "set_plan_feature": lambda ctx: super_admin.set_plan_feature(ctx, 1, Feature.VOICE_ANALYSIS.value, False),
    "set_plan_active": lambda ctx: super_admin.set_plan_active(ctx, 1, False),
    "assign_plan": lambda ctx: super_admin.assign_plan(ctx, 1, "basic"),
    "list_subscriptions": lambda ctx: super_admin.list_subscriptions(ctx),
    "create_org_admin": lambda ctx: super_admin.create_org_admin(
        ctx, 1, username="intruder", email="i@example.com", password=STRONG
    ),
    "platform_metrics": lambda ctx: super_admin.platform_metrics(ctx),
    "list_error_events": lambda ctx: super_admin.list_error_events(ctx),
}


@pytest.mark.parametrize("call", list(SUPER_ADMIN_CALLS), ids=list(SUPER_ADMIN_CALLS))
@pytest.mark.parametrize("role", ["org_admin", "analyst"])
def test_non_super_admins_are_denied_everything(factory, admin_ctx, analyst_ctx, call, role):
    ctx = admin_ctx if role == "org_admin" else analyst_ctx
    with pytest.raises(PermissionDeniedError):
        SUPER_ADMIN_CALLS[call](ctx)
    with session_scope() as db:
        assert (
            db.execute(
                select(func.count(AuditLog.id)).where(AuditLog.action.startswith("platform."))
            ).scalar_one()
            == 0
        )


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


def test_list_organizations_reports_plan_counts_and_storage(
    root_ctx, factory, admin_ctx, org_id, sales_frame
):
    factory.analyst(org_id)
    factory.user(org_id, "analyst", active=False)
    factory.dataset(admin_ctx, sales_frame, "kept")
    archived = factory.dataset(admin_ctx, sales_frame, "old")
    with session_scope() as db:
        db.get(Dataset, archived).status = "archived"
    other = factory.organization("Beta", plan="basic")

    rows = {row["id"]: row for row in super_admin.list_organizations(root_ctx)}
    acme = rows[org_id]
    assert acme["name"] == "Acme Retail" and acme["plan_code"] == "premium" and acme["plan_name"] == "Premium"
    assert acme["status"] == "active" and acme["active_users"] == 2
    assert acme["active_datasets"] == 1 and acme["archived_datasets"] == 1
    with session_scope() as db:
        assert acme["storage_bytes"] == subscriptions.org_storage_bytes(db, org_id) > 0
    assert rows[other]["plan_code"] == "basic" and rows[other]["active_users"] == 0
    assert rows[other]["storage_bytes"] == 0 and "renewal_date" in rows[other] and rows[other]["created_at"]


def test_create_organization_with_first_admin(root_ctx):
    result = super_admin.create_organization(
        root_ctx,
        name="  Globex Corp  ",
        contact_email="ops@globex.com",
        plan_code="premium",
        admin_username="Globex.Admin",
        admin_email="admin@globex.com",
        admin_password=STRONG,
        admin_full_name="Gina Admin",
    )
    assert result["name"] == "Globex Corp" and result["slug"] == "globex-corp"
    assert result["plan_code"] == "premium" and result["active_users"] == 1
    with session_scope() as db:
        admin = db.get(User, result["admin_user_id"])
        assert admin.username == "globex.admin" and admin.role == "org_admin"
        assert admin.organization_id == result["id"] and admin.created_by_id == root_ctx.user_id
        sub = db.execute(
            select(Subscription).where(Subscription.organization_id == result["id"])
        ).scalar_one()
        assert sub.status == "active" and sub.assigned_by_id == root_ctx.user_id
    created = _audit("platform.organization.create")
    assert len(created) == 1 and created[0].organization_id == result["id"]
    assert created[0].actor_role == "super_admin" and created[0].actor_user_id == root_ctx.user_id
    assert _audit("platform.org_admin.create")[0].organization_id == result["id"]


def test_create_organization_generates_unique_slugs(root_ctx, factory):
    seed = factory.organization("Seed")  # the factory assigns slugs such as "org-2"
    with session_scope() as db:
        taken = db.get(Organization, seed).slug
    number = taken.split("-")[1]
    first = super_admin.create_organization(root_ctx, name=f"Org {number}")
    second = super_admin.create_organization(root_ctx, name=f"ORG-{number}!")
    third = super_admin.create_organization(root_ctx, name="!!!")
    assert first["slug"] == f"{taken}-2"
    assert second["slug"] == f"{taken}-3"
    assert third["slug"] == "org"
    assert first["plan_code"] == "basic"


def test_create_organization_rejects_duplicates_and_bad_input(root_ctx, org_id, factory):
    with pytest.raises(ConflictError):
        super_admin.create_organization(root_ctx, name="acme retail")
    with pytest.raises(ValidationError):
        super_admin.create_organization(root_ctx, name="   ")
    with pytest.raises(ValidationError):
        super_admin.create_organization(root_ctx, name="x" * 201)
    with pytest.raises(ValidationError):
        super_admin.create_organization(root_ctx, name="Bad Mail", contact_email="not-an-email")
    with pytest.raises(NotFoundError):
        super_admin.create_organization(root_ctx, name="No Plan", plan_code="gold")
    with pytest.raises(ValidationError, match="username, e-mail and password"):
        super_admin.create_organization(root_ctx, name="Half Admin", admin_username="half")
    factory.user(org_id, "analyst", username="taken")
    with pytest.raises(ConflictError):
        super_admin.create_organization(
            root_ctx, name="Clash", admin_username="TAKEN", admin_email="a@b.com", admin_password=STRONG
        )
    with pytest.raises(ValidationError):
        super_admin.create_organization(
            root_ctx, name="Weak", admin_username="weakadmin", admin_email="a@b.com", admin_password="short"
        )
    # Every failure rolled back completely: only the fixture organization exists.
    with session_scope() as db:
        assert db.execute(select(func.count(Organization.id))).scalar_one() == 1
        assert not _audit("platform.organization.create")


def test_update_organization(root_ctx, org_id, factory):
    factory.organization("Beta")
    result = super_admin.update_organization(
        root_ctx, org_id, name="Acme Stores", contact_email="hq@acme.com"
    )
    assert result["name"] == "Acme Stores" and result["contact_email"] == "hq@acme.com"
    assert super_admin.update_organization(root_ctx, org_id, contact_email="")["contact_email"] is None
    assert super_admin.update_organization(root_ctx, org_id, name="ACME STORES")["name"] == "ACME STORES"
    with pytest.raises(ConflictError):
        super_admin.update_organization(root_ctx, org_id, name="beta")
    with pytest.raises(ValidationError):
        super_admin.update_organization(root_ctx, org_id)
    with pytest.raises(NotFoundError):
        super_admin.update_organization(root_ctx, 9999, name="Ghost")
    with pytest.raises(ValidationError):
        super_admin.update_organization(root_ctx, 0, name="Ghost")
    audit = _audit("platform.organization.update")
    assert len(audit) == 3 and {a.organization_id for a in audit} == {org_id}


@pytest.mark.parametrize("status", ["suspended", "deactivated"])
def test_suspension_keeps_data_blocks_access_and_revokes_sessions(
    root_ctx, factory, admin_ctx, analyst_ctx, org_id, sales_frame, status
):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    admin_session = _open_session(admin_ctx.user_id, org_id)
    analyst_session = _open_session(analyst_ctx.user_id, org_id)
    other_org = factory.organization("Other")
    other_ctx = factory.org_admin(other_org)
    other_session = _open_session(other_ctx.user_id, other_org)

    result = super_admin.set_organization_status(root_ctx, org_id, status, reason="Unpaid invoice")
    assert result["status"] == status and result["status_reason"] == "Unpaid invoice"
    assert result["status_changed_at"] is not None and result["sessions_revoked"] == 2

    for session_id in (admin_session, analyst_session):
        revoked_at, reason = _session_state(session_id)
        assert revoked_at is not None and reason == f"organization_{status}"
    assert _session_state(other_session) == (None, None)
    with session_scope() as db:
        assert db.get(Dataset, dataset_id) is not None
        assert db.execute(select(func.count(User.id)).where(User.organization_id == org_id)).scalar_one() == 2
        for ctx in (admin_ctx, analyst_ctx):
            with pytest.raises(PermissionDeniedError, match="not active"):
                authorize(db, ctx)
        assert authorize(db, other_ctx) == other_org
    audit = _audit(f"platform.organization.{status}")
    assert len(audit) == 1 and audit[0].organization_id == org_id
    assert json.loads(audit[0].details_json)["previous_status"] == "active"

    reactivated = super_admin.set_organization_status(root_ctx, org_id, "active")
    assert reactivated["status"] == "active" and reactivated["status_reason"] is None
    assert reactivated["sessions_revoked"] == 0
    with session_scope() as db:
        assert authorize(db, admin_ctx) == org_id


def test_set_organization_status_validation(root_ctx, org_id):
    with pytest.raises(ValidationError):
        super_admin.set_organization_status(root_ctx, org_id, "deleted")
    with pytest.raises(NotFoundError):
        super_admin.set_organization_status(root_ctx, 4242, "suspended")
    with pytest.raises(ValidationError):
        super_admin.set_organization_status(root_ctx, org_id, "suspended", reason="r" * 2001)


def test_create_org_admin_enforces_quota_and_uniqueness(root_ctx, factory, org_id):
    other = factory.organization("Small", plan="basic")
    created = super_admin.create_org_admin(
        root_ctx, other, username="Small.Admin", email="boss@small.com", password=STRONG, full_name="Boss"
    )
    assert created["username"] == "small.admin" and created["role"] == "org_admin"
    assert created["organization_id"] == other
    audit = _audit("platform.org_admin.create")
    assert len(audit) == 1 and audit[0].organization_id == other and audit[0].target_id == str(created["id"])

    with pytest.raises(ConflictError):
        super_admin.create_org_admin(
            root_ctx, org_id, username="small.admin", email="x@y.com", password=STRONG
        )
    with pytest.raises(ValidationError):
        super_admin.create_org_admin(root_ctx, org_id, username="ab", email="x@y.com", password=STRONG)
    with pytest.raises(ValidationError):
        super_admin.create_org_admin(
            root_ctx, org_id, username="weakling", email="x@y.com", password="weakling123"
        )
    with pytest.raises(NotFoundError):
        super_admin.create_org_admin(root_ctx, 777, username="ghost", email="x@y.com", password=STRONG)

    super_admin.update_plan(root_ctx, _plan_id("basic"), max_users=1)
    with pytest.raises(PlanLimitExceededError):
        super_admin.create_org_admin(root_ctx, other, username="second", email="s@small.com", password=STRONG)


# ---------------------------------------------------------------------------
# Plans and features
# ---------------------------------------------------------------------------


def test_list_plans_reports_full_feature_matrix(root_ctx, factory):
    factory.organization(plan="basic")
    factory.organization(plan="basic")
    factory.organization(plan="premium")
    plans = {p["code"]: p for p in super_admin.list_plans(root_ctx)}
    assert set(plans) == {"basic", "premium"}
    basic = plans["basic"]
    assert set(basic["features"]) == set(ALL_FEATURES)
    assert {k for k, v in basic["features"].items() if v} == set(BASIC_FEATURES)
    assert all(plans["premium"]["features"].values())
    assert basic["organizations"] == 2 and plans["premium"]["organizations"] == 1
    assert basic["price_cents"] == 4900 and basic["billing_period"] == "monthly" and basic["is_active"]
    assert basic["max_users"] == 5 and basic["max_versions_per_dataset"] == 10


def test_create_plan_stores_every_feature_row(root_ctx):
    plan = super_admin.create_plan(root_ctx, **_plan_kwargs(code="Team"))
    assert plan["code"] == "team" and plan["currency"] == "EUR" and plan["billing_period"] == "yearly"
    assert plan["features"][Feature.VOICE_ANALYSIS.value] is True
    assert sum(plan["features"].values()) == 2 and plan["organizations"] == 0
    with session_scope() as db:
        rows = db.execute(select(PlanFeature).where(PlanFeature.plan_id == plan["id"])).scalars().all()
        assert sorted(r.feature_key for r in rows) == sorted(ALL_FEATURES)
    assert _audit("platform.plan.create")[0].target_id == str(plan["id"])
    with pytest.raises(ConflictError):
        super_admin.create_plan(root_ctx, **_plan_kwargs())


def test_create_plan_without_features_disables_all(root_ctx):
    plan = super_admin.create_plan(root_ctx, **_plan_kwargs(features=None))
    assert not any(plan["features"].values())


@pytest.mark.parametrize(
    "overrides",
    [
        {"code": "Bad Code"},
        {"code": "x"},
        {"name": " "},
        {"price_cents": -1},
        {"price_cents": 1.5},
        {"currency": "EURO"},
        {"currency": "U1D"},
        {"billing_period": "weekly"},
        {"max_users": 0},
        {"max_storage_mb": -5},
        {"max_datasets": True},
        {"max_monthly_analyses": "10"},
        {"max_versions_per_dataset": 1},
        {"max_upload_mb": 1025},
        {"features": {"teleportation"}},
        {"features": "voice_analysis"},
    ],
)
def test_create_plan_validation(root_ctx, overrides):
    with pytest.raises(ValidationError):
        super_admin.create_plan(root_ctx, **_plan_kwargs(**overrides))
    assert len(super_admin.list_plans(root_ctx)) == 2


def test_upload_limit_respects_configured_ceiling(root_ctx, app_settings):
    set_settings(app_settings.with_overrides(absolute_upload_ceiling_mb=64))
    with pytest.raises(ValidationError, match="64 MB"):
        super_admin.create_plan(root_ctx, **_plan_kwargs(max_upload_mb=65))
    assert super_admin.create_plan(root_ctx, **_plan_kwargs(max_upload_mb=64))["max_upload_mb"] == 64


def test_update_plan_whitelist_and_validation(root_ctx):
    plan_id = _plan_id("basic")
    updated = super_admin.update_plan(
        root_ctx,
        plan_id,
        name="Basic Plus",
        price_cents=5900,
        currency="inr",
        billing_period="quarterly",
        max_users=7,
    )
    assert updated["name"] == "Basic Plus" and updated["price_cents"] == 5900
    assert (
        updated["currency"] == "INR"
        and updated["billing_period"] == "quarterly"
        and updated["max_users"] == 7
    )
    audit = _audit("platform.plan.update")
    assert json.loads(audit[0].details_json)["before"]["max_users"] == 5
    for bad in ({"code": "other"}, {"is_active": False}, {"features": set()}, {}):
        with pytest.raises(ValidationError):
            super_admin.update_plan(root_ctx, plan_id, **bad)
    for bad in (
        {"max_users": 0},
        {"max_versions_per_dataset": 1},
        {"max_upload_mb": 5000},
        {"billing_period": "daily"},
    ):
        with pytest.raises(ValidationError):
            super_admin.update_plan(root_ctx, plan_id, **bad)
    with pytest.raises(NotFoundError):
        super_admin.update_plan(root_ctx, 999, name="Ghost")
    with session_scope() as db:
        assert db.get(SubscriptionPlan, plan_id).max_users == 7


def test_plan_limit_changes_apply_immediately(root_ctx, factory):
    org = factory.organization(plan="basic")
    super_admin.update_plan(root_ctx, _plan_id("basic"), max_datasets=3)
    with session_scope() as db:
        assert subscriptions.get_limits(db, org).max_datasets == 3


def test_feature_toggle_takes_effect_immediately(root_ctx, factory):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    with session_scope() as db, pytest.raises(FeatureNotAvailableError):
        authorize(db, ctx, feature=Feature.VOICE_ANALYSIS)
    plan = super_admin.set_plan_feature(root_ctx, _plan_id("basic"), Feature.VOICE_ANALYSIS.value, True)
    assert plan["features"][Feature.VOICE_ANALYSIS.value] is True
    with session_scope() as db:
        assert authorize(db, ctx, feature=Feature.VOICE_ANALYSIS) == org
    super_admin.set_plan_feature(root_ctx, _plan_id("basic"), Feature.DATA_PROFILING.value, False)
    with session_scope() as db, pytest.raises(FeatureNotAvailableError):
        authorize(db, ctx, feature=Feature.DATA_PROFILING)
    audit = _audit("platform.plan.feature")
    assert [json.loads(a.details_json)["feature"] for a in audit] == ["voice_analysis", "data_profiling"]


def test_set_plan_feature_recreates_missing_row_and_validates(root_ctx):
    plan_id = _plan_id("basic")
    with session_scope() as db:
        db.execute(
            PlanFeature.__table__.delete().where(
                PlanFeature.plan_id == plan_id, PlanFeature.feature_key == Feature.EMAIL_ALERTS.value
            )
        )
    assert super_admin.list_plans(root_ctx)[0]["features"][Feature.EMAIL_ALERTS.value] is False
    plan = super_admin.set_plan_feature(root_ctx, plan_id, Feature.EMAIL_ALERTS.value, True)
    assert plan["features"][Feature.EMAIL_ALERTS.value] is True
    with pytest.raises(ValidationError):
        super_admin.set_plan_feature(root_ctx, plan_id, "time_travel", True)
    with pytest.raises(ValidationError):
        super_admin.set_plan_feature(root_ctx, plan_id, Feature.EMAIL_ALERTS.value, "yes")
    with pytest.raises(NotFoundError):
        super_admin.set_plan_feature(root_ctx, 4040, Feature.EMAIL_ALERTS.value, True)


def test_inactive_plan_cannot_be_assigned_but_existing_subscriptions_work(root_ctx, factory):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    other = factory.organization(plan="premium")
    result = super_admin.set_plan_active(root_ctx, _plan_id("basic"), False)
    assert result["is_active"] is False and result["organizations"] == 1
    with session_scope() as db:
        assert authorize(db, ctx, feature=Feature.DATA_PROFILING) == org
    with pytest.raises(ValidationError, match="inactive"):
        super_admin.assign_plan(root_ctx, other, "basic")
    with pytest.raises(ValidationError, match="inactive"):
        super_admin.create_organization(root_ctx, name="Late Comer", plan_code="basic")
    with pytest.raises(ValidationError):
        super_admin.set_plan_active(root_ctx, _plan_id("basic"), 1)
    assert super_admin.set_plan_active(root_ctx, _plan_id("basic"), True)["is_active"] is True
    super_admin.assign_plan(root_ctx, other, "basic")
    assert [a.action for a in _audit("platform.plan.deactivate")] == ["platform.plan.deactivate"]
    with pytest.raises(NotFoundError):
        super_admin.set_plan_active(root_ctx, 31337, True)


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------


def test_assign_plan_keeps_history_and_one_active_subscription(root_ctx, factory):
    org = factory.organization("Gamma", plan="basic")
    ctx = factory.org_admin(org)
    renewal = datetime(2027, 1, 1)
    sub = super_admin.assign_plan(
        root_ctx,
        org,
        "premium",
        renewal_date=renewal,
        auto_renew=False,
        billing_reference=" INV-42 ",
        notes="Upgrade",
    )
    assert sub["plan_code"] == "premium" and sub["status"] == "active"
    assert sub["renewal_date"] == renewal and sub["auto_renew"] is False
    assert sub["billing_reference"] == "INV-42" and sub["notes"] == "Upgrade"
    with session_scope() as db:
        assert authorize(db, ctx, feature=Feature.APPEND_MERGE) == org
    super_admin.assign_plan(root_ctx, org, "basic")

    history = super_admin.list_subscriptions(root_ctx, organization_id=org)
    assert [h["plan_code"] for h in history] == ["basic", "premium", "basic"]
    assert [h["status"] for h in history] == ["active", "ended", "ended"]
    assert all(h["ended_at"] is not None for h in history[1:])
    assert history[0]["assigned_by"] == root_ctx.username and history[2]["assigned_by"] is None
    active = super_admin.list_subscriptions(root_ctx, organization_id=org, include_ended=False)
    assert len(active) == 1 and active[0]["organization_name"] == "Gamma"
    with session_scope() as db, pytest.raises(FeatureNotAvailableError):
        authorize(db, ctx, feature=Feature.APPEND_MERGE)
    audit = _audit("platform.subscription.assign")
    assert len(audit) == 2 and {a.organization_id for a in audit} == {org}
    assert json.loads(audit[0].details_json)["previous_plan"] == "basic"


def test_assign_plan_validation(root_ctx, org_id):
    with pytest.raises(NotFoundError):
        super_admin.assign_plan(root_ctx, 555, "basic")
    with pytest.raises(NotFoundError):
        super_admin.assign_plan(root_ctx, org_id, "platinum")
    with pytest.raises(ValidationError):
        super_admin.assign_plan(root_ctx, org_id, "basic", renewal_date="2027-01-01")
    with pytest.raises(ValidationError):
        super_admin.assign_plan(root_ctx, org_id, "basic", billing_reference="x" * 101)
    with pytest.raises(ValidationError):
        super_admin.assign_plan(root_ctx, org_id, "basic", auto_renew="no")
    with session_scope() as db:
        subs = db.execute(select(Subscription).where(Subscription.organization_id == org_id)).scalars().all()
        assert len(subs) == 1 and subs[0].status == "active"


def test_list_subscriptions_all_orgs(root_ctx, factory):
    a = factory.organization(plan="basic")
    b = factory.organization(plan="premium")
    rows = super_admin.list_subscriptions(root_ctx)
    assert {r["organization_id"] for r in rows} == {a, b}
    assert {"renewal_date", "auto_renew", "billing_reference", "notes", "started_at"} <= set(rows[0])
    with pytest.raises(NotFoundError):
        super_admin.list_subscriptions(root_ctx, organization_id=999)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------


def _seed_metrics(org_a: int, org_b: int, analyst_id: int, dataset_id: int) -> None:
    now = timeutil.utcnow()
    with session_scope() as db:
        db.execute(update(CleaningOperation).values(duration_ms=None))
        # 20 analyses in the window with durations 1..20 ms (status mix), plus two older ones.
        for i in range(1, 21):
            db.add(
                AnalysisHistory(
                    organization_id=org_a if i % 2 else org_b,
                    user_id=analyst_id,
                    question=f"q{i}",
                    status="error" if i <= 3 else ("clarification" if i == 4 else "success"),
                    duration_ms=i,
                    created_at=now - timedelta(days=1, minutes=i),
                )
            )
        db.add(
            AnalysisHistory(
                organization_id=org_a, question="q-null", status="rejected", duration_ms=None, created_at=now
            )
        )
        for days in (31, 400):
            db.add(
                AnalysisHistory(
                    organization_id=org_a,
                    question="old",
                    status="success",
                    duration_ms=99999,
                    created_at=now - timedelta(days=days),
                )
            )
        for duration in (100, 200, 300, 400):
            db.add(
                CleaningOperation(
                    organization_id=org_a,
                    dataset_id=dataset_id,
                    operation_type="trim_whitespace",
                    status="applied",
                    duration_ms=duration,
                    created_at=now - timedelta(days=2),
                )
            )
        db.add(
            CleaningOperation(
                organization_id=org_a,
                dataset_id=dataset_id,
                operation_type="merge",
                status="failed",
                duration_ms=50000,
                created_at=now - timedelta(days=45),
            )
        )
        for days, error_type in ((1, "ValueError"), (6, "KeyError"), (8, "OldError")):
            db.add(
                ErrorEvent(
                    component="test",
                    error_type=error_type,
                    message="m",
                    reference=uuid.uuid4().hex[:12],
                    created_at=now - timedelta(days=days),
                )
            )
        for metric, quantity, days in (
            ("analysis_query", 1, 1),
            ("analysis_query", 1, 29),
            ("analysis_query", 1, 40),
            ("upload_bytes", 2048, 3),
            ("export", 1, 5),
        ):
            db.add(
                UsageRecord(
                    organization_id=org_b if metric == "export" else org_a,
                    metric=metric,
                    quantity=quantity,
                    created_at=now - timedelta(days=days),
                )
            )


def test_platform_metrics_match_seeded_data_exactly(
    root_ctx, factory, admin_ctx, org_id, sales_frame, app_settings
):
    analyst_id = factory.user(org_id, "analyst")
    factory.user(org_id, "analyst", active=False)
    org_b = factory.organization("Beta", plan="basic")
    factory.organization("Gamma", status="suspended")
    factory.organization("Delta", status="deactivated")
    dataset_id = factory.dataset(admin_ctx, sales_frame, "sales")
    archived = factory.dataset(admin_ctx, sales_frame, "old")
    with session_scope() as db:
        db.get(Dataset, archived).status = "archived"
    _seed_metrics(org_id, org_b, analyst_id, dataset_id)

    metrics = super_admin.platform_metrics(root_ctx)
    assert metrics["organizations"] == {
        "total": 4,
        "by_status": {"active": 2, "suspended": 1, "deactivated": 1},
    }
    assert metrics["users_by_role"] == {
        "super_admin": {"active": 1, "total": 1},
        "org_admin": {"active": 1, "total": 1},
        "analyst": {"active": 1, "total": 2},
    }
    assert metrics["datasets"] == {"active": 1, "archived": 1}
    jobs = metrics["analysis_jobs"]
    assert jobs["total"] == 23 and jobs["last_30_days"] == 21
    assert jobs["last_30_days_success"] == 16 and jobs["last_30_days_error"] == 3
    assert jobs["last_30_days_by_status"] == {"success": 16, "error": 3, "clarification": 1, "rejected": 1}
    assert metrics["errors_last_7_days"] == 2

    # Durations 1..20: mean 10.5; nearest-rank p95 is the 19th value.
    analysis = metrics["latency_last_30_days"]["analysis"]
    assert analysis == {"count": 20, "average_ms": 10.5, "p95_ms": 19}
    # Durations 100, 200, 300, 400: mean 250; p95 rank ceil(3.8) = 4.
    cleaning = metrics["latency_last_30_days"]["cleaning"]
    assert cleaning == {"count": 4, "average_ms": 250.0, "p95_ms": 400}
    assert metrics["usage_last_30_days"] == {"analysis_query": 2.0, "upload_bytes": 2048.0, "export": 1.0}

    with session_scope() as db:
        originals = db.execute(select(func.sum(Dataset.original_size_bytes))).scalar_one() or 0
        versions = db.execute(select(func.sum(DatasetVersion.size_bytes))).scalar_one()
    assert metrics["storage"]["recorded_bytes"] == originals + versions > 0
    assert metrics["storage"]["disk_bytes"] == storage.directory_size_bytes(app_settings.storage_root) > 0


def test_platform_metrics_empty_latency_is_none(root_ctx):
    metrics = super_admin.platform_metrics(root_ctx)
    assert metrics["latency_last_30_days"]["analysis"] == {"count": 0, "average_ms": None, "p95_ms": None}
    assert metrics["analysis_jobs"]["total"] == 0 and metrics["usage_last_30_days"] == {}


@pytest.mark.parametrize(
    ("values", "expected"),
    [([5], 5), ([1, 2], 2), (list(range(1, 101)), 95), (list(range(100, 0, -1)), 95), ([7, 7, 7, 1], 7)],
)
def test_percentile_nearest_rank(values, expected):
    assert super_admin._percentile_nearest_rank(values, 95) == expected


def test_list_error_events_is_redacted_and_bounded(root_ctx, admin_ctx, org_id):
    record_error("analytics", RuntimeError("boom password=hunter2 happened"), ctx=admin_ctx)
    record_error("voice", ValueError("second"))
    events = super_admin.list_error_events(root_ctx)
    assert [e["component"] for e in events] == ["voice", "analytics"]
    first = events[1]
    assert "hunter2" not in first["message"] and "password=***" in first["message"]
    assert first["organization_id"] == org_id and first["organization_name"] == "Acme Retail"
    assert first["error_type"] == "RuntimeError" and len(first["reference"]) == 12
    assert "traceback" not in json.dumps(events, default=str).lower()
    assert len(super_admin.list_error_events(root_ctx, limit=1)) == 1
    for bad in (0, 1001, "5", True):
        with pytest.raises(ValidationError):
            super_admin.list_error_events(root_ctx, limit=bad)


def test_platform_storage_counts_purged_files_still_on_disk(root_ctx, factory, admin_ctx, org_id, sales_frame):
    """Regression (SR3 follow-up): the platform figure matches the organizations' storage quotas,
    including purged version files that could not be deleted yet."""
    from services import cleaning_service

    dataset_id = factory.dataset(admin_ctx, sales_frame, "sales")
    cleaning_service.apply_cleaning(admin_ctx, dataset_id, "trim_whitespace", {"columns": ["region"]})
    with session_scope() as db:
        first = db.execute(
            select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id, DatasetVersion.version_number == 1)
        ).scalar_one()
        assert storage.resolve_relative(first.file_path).exists()
        first.is_purged, first.file_pending_delete = True, True
        first_size = first.size_bytes
    with session_scope() as db:
        expected = subscriptions.org_storage_bytes(db, org_id)
    recorded = super_admin.platform_metrics(root_ctx)["storage"]["recorded_bytes"]
    assert recorded == expected and first_size > 0
