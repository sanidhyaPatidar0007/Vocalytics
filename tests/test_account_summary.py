"""modules.account: the summary navigation is built from (the UI no longer queries the DB)."""

from __future__ import annotations

from sqlalchemy import update

from core.constants import BASIC_FEATURES, PREMIUM_FEATURES, Permission
from database.connection import session_scope
from database.models import User
from modules.account import account_summary


def test_super_admin_has_no_organization_features_or_permissions(factory):
    summary = account_summary(factory.super_admin())
    assert (summary.features, summary.permissions, summary.org_name, summary.plan_name) == (
        frozenset(),
        frozenset(),
        None,
        None,
    )
    assert summary.must_change_password is False


def test_org_admin_gets_every_permission_and_the_plan_features(factory):
    org_id = factory.organization("Summary Org", plan="basic")
    summary = account_summary(factory.org_admin(org_id))
    assert summary.permissions == frozenset(p.value for p in Permission)
    assert summary.features == frozenset(BASIC_FEATURES)
    assert (summary.org_name, summary.plan_name.lower()) == ("Summary Org", "basic")


def test_analyst_gets_only_granted_permissions_and_the_password_flag(factory, org_id):
    analyst = factory.analyst(org_id, permissions={"export_data"})
    with session_scope() as db:
        db.execute(update(User).where(User.id == analyst.user_id).values(must_change_password=True))
    summary = account_summary(analyst)
    assert summary.permissions == frozenset({"export_data"})
    assert summary.features == frozenset(PREMIUM_FEATURES)
    assert summary.must_change_password is True
