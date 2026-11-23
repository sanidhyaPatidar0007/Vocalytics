"""Server-side authorization guards (auth/authorization.py)."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from auth.authorization import AuthContext, authorize, authorize_dataset, authorize_super_admin
from core.constants import Feature, Permission, Role
from core.errors import AuthenticationError, FeatureNotAvailableError, NotFoundError, PermissionDeniedError
from database import session_scope
from database.models import Organization, PlanFeature, Subscription, User


def test_org_member_is_authorized_for_own_org(admin_ctx, org_id):
    with session_scope() as db:
        assert authorize(db, admin_ctx) == org_id


def test_forged_context_with_other_org_is_rejected(admin_ctx, factory):
    other_org = factory.organization()
    forged = AuthContext(user_id=admin_ctx.user_id, role=Role.ORG_ADMIN, organization_id=other_org)
    with session_scope() as db, pytest.raises(AuthenticationError):
        authorize(db, forged)


def test_forged_role_escalation_is_rejected(analyst_ctx):
    forged = AuthContext(
        user_id=analyst_ctx.user_id, role=Role.ORG_ADMIN, organization_id=analyst_ctx.organization_id
    )
    with session_scope() as db, pytest.raises(AuthenticationError):
        authorize(db, forged)


def test_inactive_user_is_rejected(admin_ctx):
    with session_scope() as db:
        db.get(User, admin_ctx.user_id).is_active = False
    with session_scope() as db, pytest.raises(AuthenticationError):
        authorize(db, admin_ctx)


@pytest.mark.parametrize("status", ["suspended", "deactivated"])
def test_suspended_org_is_blocked(admin_ctx, org_id, status):
    with session_scope() as db:
        db.get(Organization, org_id).status = status
    with session_scope() as db, pytest.raises(PermissionDeniedError, match="not active"):
        authorize(db, admin_ctx)


def test_super_admin_cannot_act_on_tenant_data(factory):
    super_ctx = factory.super_admin()
    with session_scope() as db, pytest.raises(PermissionDeniedError):
        authorize(db, super_ctx)
    with session_scope() as db:
        assert authorize_super_admin(db, super_ctx).role == "super_admin"


def test_non_super_admin_cannot_use_platform_actions(admin_ctx):
    with session_scope() as db, pytest.raises(PermissionDeniedError):
        authorize_super_admin(db, admin_ctx)


def test_role_restriction(analyst_ctx):
    with session_scope() as db, pytest.raises(PermissionDeniedError):
        authorize(db, analyst_ctx, roles=(Role.ORG_ADMIN,))


def test_analyst_permissions_are_explicit(factory, org_id):
    limited = factory.analyst(org_id, permissions=set())
    with session_scope() as db:
        with pytest.raises(PermissionDeniedError):
            authorize(db, limited, permission=Permission.EXPORT_DATA)
    allowed = factory.analyst(org_id, permissions={Permission.CLEAN_DATA.value})
    with session_scope() as db:
        assert authorize(db, allowed, permission=Permission.CLEAN_DATA)


def test_plan_features_are_enforced_and_read_fresh(factory):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    with session_scope() as db, pytest.raises(FeatureNotAvailableError):
        authorize(db, ctx, feature=Feature.VOICE_ANALYSIS)
    with session_scope() as db:
        plan_id = db.execute(
            select(Subscription.plan_id).where(Subscription.organization_id == org)
        ).scalar_one()
        db.execute(
            select(PlanFeature).where(
                PlanFeature.plan_id == plan_id, PlanFeature.feature_key == "voice_analysis"
            )
        ).scalar_one().enabled = True
    with session_scope() as db:
        assert authorize(db, ctx, feature=Feature.VOICE_ANALYSIS) == org


def test_org_without_subscription_has_no_features(factory):
    org = factory.organization()
    ctx = factory.org_admin(org)
    with session_scope() as db:
        db.execute(
            select(Subscription).where(Subscription.organization_id == org)
        ).scalar_one().status = "ended"
    with session_scope() as db, pytest.raises(FeatureNotAvailableError):
        authorize(db, ctx, feature=Feature.DATA_PROFILING)


def test_cross_tenant_dataset_access_looks_like_not_found(factory, sales_frame):
    org_a, org_b = factory.organization(), factory.organization()
    admin_a, admin_b = factory.org_admin(org_a), factory.org_admin(org_b)
    dataset_b = factory.dataset(admin_b, sales_frame, "secret")
    with session_scope() as db, pytest.raises(NotFoundError) as excinfo:
        authorize_dataset(db, admin_a, dataset_b)
    with session_scope() as db, pytest.raises(NotFoundError) as missing:
        authorize_dataset(db, admin_a, 999_999)
    assert excinfo.value.user_message == missing.value.user_message  # no existence leak


def test_analyst_sees_only_granted_datasets(factory, org_id, admin_ctx, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    analyst_id = factory.user(org_id, "analyst")
    analyst = factory.ctx(analyst_id)
    with session_scope() as db, pytest.raises(NotFoundError):
        authorize_dataset(db, analyst, dataset_id)
    factory.grant(dataset_id, analyst_id)
    with session_scope() as db:
        assert authorize_dataset(db, analyst, dataset_id).id == dataset_id


@pytest.mark.parametrize("bad_id", [0, -1, True, "1", None, 1.5])
def test_invalid_dataset_ids_are_rejected(admin_ctx, bad_id):
    from core.errors import ValidationError

    with session_scope() as db, pytest.raises(ValidationError):
        authorize_dataset(db, admin_ctx, bad_id)


def test_system_context_respects_suspension(org_id):
    with session_scope() as db:
        db.get(Organization, org_id).status = "suspended"
    with session_scope() as db, pytest.raises(PermissionDeniedError):
        authorize(db, AuthContext.system(org_id))
