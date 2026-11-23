"""Shared pytest fixtures.

Every test runs against its own SQLite database and storage folder inside
pytest's ``tmp_path``; the real ``data/`` and ``storage/`` folders are never
touched. Memory and disk readings are faked with generous defaults so results
do not depend on the machine; resource tests override them explicitly.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from auth.authorization import AuthContext, load_context
from auth.password_manager import hash_password
from config.settings import Settings, reset_settings, set_settings
from core import timeutil
from core.constants import ANALYST_DEFAULT_PERMISSIONS, Role
from database import init_database, reset_engine, session_scope
from database.models import DatasetAccess, Organization, Subscription, SubscriptionPlan, User, UserPermission
from services import resource_manager

GB = 1024**3
DEFAULT_PASSWORD = "Str0ng!Passw0rd"


@pytest.fixture(autouse=True)
def app_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings.for_testing(tmp_path)
    set_settings(settings)
    reset_engine()
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: 8 * GB)
    monkeypatch.setattr(resource_manager, "memory_total_provider", lambda: 16 * GB)
    monkeypatch.setattr(resource_manager, "disk_free_provider", lambda path: 200 * GB)
    init_database()
    assert settings.database_path.resolve().is_relative_to(tmp_path.resolve())
    yield settings
    reset_engine()
    reset_settings()


class Factory:
    """Creates tenants, users, contexts and datasets directly in the test database."""

    def __init__(self) -> None:
        self._seq = itertools.count(1)

    def _n(self) -> int:
        return next(self._seq)

    def organization(self, name: str | None = None, *, plan: str = "premium", status: str = "active") -> int:
        n = self._n()
        with session_scope() as db:
            org = Organization(name=name or f"Org {n}", slug=f"org-{n}", status=status)
            db.add(org)
            db.flush()
            plan_row = db.execute(select(SubscriptionPlan).where(SubscriptionPlan.code == plan)).scalar_one()
            db.add(
                Subscription(
                    organization_id=org.id, plan_id=plan_row.id, status="active", started_at=timeutil.utcnow()
                )
            )
            return org.id

    def user(
        self,
        org_id: int | None,
        role: str = "analyst",
        *,
        username: str | None = None,
        password: str = DEFAULT_PASSWORD,
        permissions: set[str] | frozenset[str] | None = None,
        active: bool = True,
    ) -> int:
        n = self._n()
        with session_scope() as db:
            user = User(
                organization_id=None if role == Role.SUPER_ADMIN.value else org_id,
                username=username or f"{role}{n}",
                email=f"{role}{n}@example.com",
                full_name=f"{role.title()} {n}",
                role=role,
                password_hash=hash_password(password),
                is_active=active,
            )
            db.add(user)
            db.flush()
            if role == Role.ANALYST.value:
                granted = ANALYST_DEFAULT_PERMISSIONS if permissions is None else permissions
                for permission in granted:
                    db.add(UserPermission(organization_id=org_id, user_id=user.id, permission=permission))
            return user.id

    def ctx(self, user_id: int) -> AuthContext:
        with session_scope() as db:
            return load_context(db, user_id)

    def org_admin(self, org_id: int) -> AuthContext:
        return self.ctx(self.user(org_id, Role.ORG_ADMIN.value))

    def analyst(self, org_id: int, permissions: set[str] | frozenset[str] | None = None) -> AuthContext:
        return self.ctx(self.user(org_id, Role.ANALYST.value, permissions=permissions))

    def super_admin(self) -> AuthContext:
        return self.ctx(self.user(None, Role.SUPER_ADMIN.value))

    def dataset(self, ctx: AuthContext, frame: pd.DataFrame, name: str | None = None) -> int:
        """Register a dataset from an in-memory frame (bypasses file parsing)."""
        from services.versioning_service import create_dataset_with_initial_version

        label = name or f"dataset {self._n()}"
        with session_scope() as db:
            dataset, _ = create_dataset_with_initial_version(
                db, ctx, name=label, source_type="derived", original_filename=f"{label}.csv", frame=frame
            )
            return dataset.id

    def grant(self, dataset_id: int, user_id: int) -> None:
        with session_scope() as db:
            user = db.get(User, user_id)
            db.add(
                DatasetAccess(organization_id=user.organization_id, dataset_id=dataset_id, user_id=user_id)
            )


@pytest.fixture
def factory() -> Factory:
    return Factory()


@pytest.fixture
def org_id(factory: Factory) -> int:
    return factory.organization("Acme Retail", plan="premium")


@pytest.fixture
def admin_ctx(factory: Factory, org_id: int) -> AuthContext:
    return factory.org_admin(org_id)


@pytest.fixture
def analyst_ctx(factory: Factory, org_id: int) -> AuthContext:
    return factory.analyst(org_id)


@pytest.fixture
def sales_frame() -> pd.DataFrame:
    """Typed business data used by many service tests."""
    return pd.DataFrame(
        {
            "order_id": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
            "region": pd.array(["North", "South", "North", "East", "South", "North"], dtype="string"),
            "product": pd.array(["A", "B", "A", "C", "B", "C"], dtype="string"),
            "revenue": [1200.0, 980.5, 1500.0, 300.25, 760.0, 1100.0],
            "quantity": pd.array([3, 2, 5, 1, 2, 4], dtype="Int64"),
            "order_date": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-02-03", "2024-02-17", "2024-03-02", "2024-03-15"]
            ),
        }
    )
