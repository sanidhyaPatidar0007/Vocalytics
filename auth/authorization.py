"""Server-side authorization, the single source of truth for "who may do what".

Every service function receives an ``AuthContext`` that was built from the
database (``auth.authentication.validate_session`` or ``load_context``), never
from browser input, and calls ``authorize`` / ``authorize_dataset`` before it
touches tenant data. All checks read fresh state from the database, so a
suspended organization, a deactivated user or a downgraded plan takes effect
on the very next call.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from core.constants import DatasetStatus, Feature, OrgStatus, Permission, Role
from core.errors import AuthenticationError, NotFoundError, PermissionDeniedError, ValidationError
from database.models import Dataset, DatasetAccess, Organization, User, UserPermission
from modules import subscriptions


@dataclass(frozen=True)
class AuthContext:
    """Identity of the caller, resolved on the server."""

    user_id: int | None
    role: Role
    organization_id: int | None
    username: str = ""
    session_id: int | None = None

    @classmethod
    def system(cls, organization_id: int) -> AuthContext:
        """Context for the background worker acting inside one organization."""
        return cls(user_id=None, role=Role.SYSTEM, organization_id=organization_id, username="system")

    @property
    def is_super_admin(self) -> bool:
        return self.role == Role.SUPER_ADMIN

    @property
    def is_org_admin(self) -> bool:
        return self.role == Role.ORG_ADMIN

    @property
    def is_analyst(self) -> bool:
        return self.role == Role.ANALYST


def load_context(db: Session, user_id: int, session_id: int | None = None) -> AuthContext:
    """Build a context for an existing, active user (tests and authentication)."""
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Your account is not active.")
    return AuthContext(
        user_id=user.id,
        role=Role(user.role),
        organization_id=user.organization_id,
        username=user.username,
        session_id=session_id,
    )


def _current_user(db: Session, ctx: AuthContext) -> User | None:
    if not isinstance(ctx, AuthContext):
        raise AuthenticationError("Please sign in.")
    if ctx.role == Role.SYSTEM:
        return None
    if ctx.user_id is None:
        raise AuthenticationError("Please sign in.")
    user = db.get(User, ctx.user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Your account is not active.")
    # A role or organization change after sign-in invalidates the context.
    if user.role != ctx.role.value or user.organization_id != ctx.organization_id:
        raise AuthenticationError("Your session is no longer valid. Please sign in again.")
    return user


def ensure_org_active(db: Session, organization_id: int) -> Organization:
    org = db.get(Organization, organization_id)
    if org is None:
        raise NotFoundError("Organization not found.")
    if org.status != OrgStatus.ACTIVE.value:
        raise PermissionDeniedError("Your organization is not active. Contact the platform administrator.")
    return org


def user_permissions(db: Session, user_id: int) -> frozenset[str]:
    rows = db.execute(select(UserPermission.permission).where(UserPermission.user_id == user_id)).scalars()
    return frozenset(rows)


def has_permission(db: Session, ctx: AuthContext, permission: Permission | str) -> bool:
    key = permission.value if isinstance(permission, Permission) else str(permission)
    if ctx.role in (Role.ORG_ADMIN, Role.SYSTEM):
        return True
    if ctx.role == Role.ANALYST and ctx.user_id is not None:
        return key in user_permissions(db, ctx.user_id)
    return False


def authorize(
    db: Session,
    ctx: AuthContext,
    *,
    roles: Iterable[Role] | None = None,
    permission: Permission | None = None,
    feature: Feature | None = None,
) -> int:
    """Authorize an organization-scoped action and return the organization id.

    Checks, in order: the caller is an active user whose role/org still match,
    the caller belongs to an organization, the role is allowed, the
    organization is active, the plan includes ``feature`` and the caller holds
    ``permission``. Super Admins are platform operators and do not act on
    tenant data through this function.
    """
    _current_user(db, ctx)
    if ctx.role == Role.SUPER_ADMIN or ctx.organization_id is None:
        raise PermissionDeniedError("This action is only available to members of an organization.")
    if roles is not None:
        allowed = set(roles)
        if Role.ORG_ADMIN in allowed:
            allowed.add(Role.SYSTEM)
        if ctx.role not in allowed:
            raise PermissionDeniedError()
    ensure_org_active(db, ctx.organization_id)
    if feature is not None:
        subscriptions.require_feature(db, ctx.organization_id, feature)
    if permission is not None and not has_permission(db, ctx, permission):
        raise PermissionDeniedError()
    return ctx.organization_id


def authorize_super_admin(db: Session, ctx: AuthContext) -> User:
    user = _current_user(db, ctx)
    if ctx.role != Role.SUPER_ADMIN or user is None:
        raise PermissionDeniedError()
    return user


def _validate_id(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"Invalid {label}.")
    return value


def visible_datasets_query(ctx: AuthContext, *, include_archived: bool = False) -> Select:
    """Datasets the caller may see: own organization; analysts only with a grant."""
    query = select(Dataset).where(Dataset.organization_id == ctx.organization_id)
    if not include_archived:
        query = query.where(Dataset.status == DatasetStatus.ACTIVE.value)
    if ctx.role == Role.ANALYST:
        granted = select(DatasetAccess.dataset_id).where(DatasetAccess.user_id == ctx.user_id)
        query = query.where(Dataset.id.in_(granted))
    return query


def get_dataset_for_user(
    db: Session, ctx: AuthContext, dataset_id: int, *, include_archived: bool = False
) -> Dataset:
    """Return a dataset the caller may access, else NotFoundError (no existence leak)."""
    _validate_id(dataset_id, "dataset")
    dataset = db.execute(
        visible_datasets_query(ctx, include_archived=include_archived).where(Dataset.id == dataset_id)
    ).scalar_one_or_none()
    if dataset is None:
        raise NotFoundError("Dataset not found.")
    return dataset


def authorize_dataset(
    db: Session,
    ctx: AuthContext,
    dataset_id: int,
    *,
    permission: Permission | None = None,
    feature: Feature | None = None,
    roles: Iterable[Role] | None = None,
    include_archived: bool = False,
) -> Dataset:
    """``authorize`` plus dataset visibility in one call."""
    authorize(db, ctx, roles=roles, permission=permission, feature=feature)
    return get_dataset_for_user(db, ctx, dataset_id, include_archived=include_archived)
