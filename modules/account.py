"""What the signed-in user may see: the account summary the navigation is built from.

The UI calls this module instead of querying the database itself. The result is
only a hint for building menus; every service call still authorizes itself with
a fresh database check.
"""

from __future__ import annotations

from dataclasses import dataclass

from auth.authorization import AuthContext, user_permissions
from core.constants import Permission, Role
from database.connection import session_scope
from database.models import Organization, User
from modules import subscriptions


@dataclass(frozen=True)
class AccountSummary:
    must_change_password: bool
    features: frozenset[str]
    permissions: frozenset[str]
    org_name: str | None
    plan_name: str | None


def account_summary(ctx: AuthContext) -> AccountSummary:
    """Plan features, permissions and display names for ``ctx`` (Super Admins have no organization)."""
    with session_scope() as db:
        user = db.get(User, ctx.user_id) if ctx.user_id is not None else None
        must_change = bool(user and user.must_change_password)
        if ctx.organization_id is None:
            return AccountSummary(must_change, frozenset(), frozenset(), None, None)
        org = db.get(Organization, ctx.organization_id)
        subscription = subscriptions.get_active_subscription(db, ctx.organization_id)
        permissions = (
            frozenset(p.value for p in Permission)
            if ctx.role == Role.ORG_ADMIN
            else frozenset(user_permissions(db, ctx.user_id or 0))
        )
        return AccountSummary(
            must_change_password=must_change,
            features=frozenset(subscriptions.enabled_features(db, ctx.organization_id)),
            permissions=permissions,
            org_name=org.name if org else None,
            plan_name=subscription.plan.name if subscription is not None else None,
        )
