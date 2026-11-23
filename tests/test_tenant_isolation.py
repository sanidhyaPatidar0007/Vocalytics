"""Cross-tenant isolation for every organization-owned resource (spec sections 4, 18, 22 and the addendum).

Two organizations, A and B, each get an Org Admin, an analyst, an imported dataset with two versions,
a cleaning history, analysis history, a saved report, an anomaly rule with an event, a dataset grant,
organization settings and usage records. Then every public service/module function that takes an
organization-owned id is called as A's admin and as A's analyst with B's ids (and with mixed A/B id
pairs). Each call must fail with ``NotFoundError`` (or ``PermissionDeniedError`` where the function is
documented as role-restricted), must never run a supplied transform, and must leave every row and file
of organization B (and the shared plan catalogue) byte-for-byte unchanged.

The two-tenant world is built once for this module: the conftest ``app_settings`` and ``factory``
fixtures are overridden here with module-scoped equivalents (same settings factory, same fake memory
and disk readings) so the full table runs against one database instead of rebuilding it per case.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import itertools
import pkgutil
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import select

import modules
import services
from auth.authentication import login, validate_session
from auth.authorization import AuthContext, load_context
from auth.password_manager import hash_password
from config.settings import Settings, reset_settings, set_settings
from core import storage, timeutil
from core.constants import Permission, Role
from core.errors import AuthenticationError, NotFoundError, PermissionDeniedError, ValidationError
from database import init_database, reset_engine, session_scope
from database.models import (
    AuditLog,
    Base,
    DatasetAccess,
    Organization,
    PlanFeature,
    Subscription,
    SubscriptionPlan,
    User,
    UserPermission,
)
from modules import analyst, organization_admin, super_admin
from services import (
    analytics_engine,
    anomaly_service,
    append_service,
    audit_service,
    cleaning_service,
    dataset_service,
    duplicate_manager,
    merge_service,
    missing_values,
    profiling_service,
    resource_manager,
    versioning_service,
)

ROOT = Path(__file__).resolve().parent.parent
GB = 1024**3
PASSWORD = "Tenant#Pass123"
QUESTION = "What is the total revenue?"
ALL_PERMISSIONS = frozenset(p.value for p in Permission)

ORDERS_CSV = (
    "order_id,region,product,revenue,quantity,order_date\n"
    "1,North,widget,1200.00,3,2024-01-05\n"
    "2,South,gadget,980.50,2,2024-01-20\n"
    "3,North,widget,1500.00,5,2024-02-03\n"
    "4,East,gizmo,1010.00,1,2024-02-17\n"
    "5,South,gadget,760.00,2,2024-03-02\n"
    "6,West,gizmo,1100.00,4,2024-03-15\n"
    "7,North,widget,1250.00,4,2024-04-01\n"
    "8,South,gadget,890.00,3,2024-04-12\n"
    "9,East,widget,98000.00,2,2024-05-02\n"
    "10,West,gizmo,1010.00,6,2024-05-18\n"
    "11,South,widget,1180.00,3,2024-06-01\n"
    "12,East,gadget,940.00,2,2024-06-15\n"
)


# ---------------------------------------------------------------------------
# Module-scoped database and two-tenant world
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_settings(tmp_path_factory):
    """Module-scoped twin of the conftest fixture: one isolated database for this file."""
    root = tmp_path_factory.mktemp("tenant_isolation")
    patch = pytest.MonkeyPatch()
    settings = Settings.for_testing(root)
    set_settings(settings)
    reset_engine()
    patch.setattr(resource_manager, "memory_available_provider", lambda: 8 * GB)
    patch.setattr(resource_manager, "memory_total_provider", lambda: 16 * GB)
    patch.setattr(resource_manager, "disk_free_provider", lambda path: 200 * GB)
    init_database()
    assert settings.database_path.resolve().is_relative_to(root.resolve())
    yield settings
    reset_engine()
    reset_settings()
    patch.undo()


class Builder:
    """Creates organizations and users directly in the database (unique names per module)."""

    def __init__(self) -> None:
        self._seq = itertools.count(1)

    def organization(self, label: str, *, plan: str = "premium") -> int:
        n = next(self._seq)
        with session_scope() as db:
            org = Organization(name=f"{label} {n}", slug=f"iso-{n}", status="active")
            db.add(org)
            db.flush()
            plan_row = db.execute(select(SubscriptionPlan).where(SubscriptionPlan.code == plan)).scalar_one()
            db.add(
                Subscription(
                    organization_id=org.id, plan_id=plan_row.id, status="active", started_at=timeutil.utcnow()
                )
            )
            return org.id

    def user(self, org_id: int | None, role: Role, *, permissions: frozenset[str] = frozenset()) -> int:
        n = next(self._seq)
        with session_scope() as db:
            user = User(
                organization_id=None if role == Role.SUPER_ADMIN else org_id,
                username=f"iso_{role.value}{n}",
                email=f"iso{n}@example.com",
                full_name=f"Isolation {n}",
                role=role.value,
                password_hash=hash_password(PASSWORD),
                is_active=True,
            )
            db.add(user)
            db.flush()
            if role == Role.ANALYST:
                for permission in permissions:
                    db.add(UserPermission(organization_id=org_id, user_id=user.id, permission=permission))
            return user.id

    @staticmethod
    def ctx(user_id: int) -> AuthContext:
        with session_scope() as db:
            return load_context(db, user_id)

    @staticmethod
    def grant(org_id: int, dataset_id: int, user_id: int) -> None:
        with session_scope() as db:
            db.add(DatasetAccess(organization_id=org_id, dataset_id=dataset_id, user_id=user_id))


@pytest.fixture(scope="module")
def builder(app_settings) -> Builder:
    return Builder()


def build_tenant(builder: Builder, label: str) -> SimpleNamespace:
    """One organization with every kind of organization-owned record."""
    org = builder.organization(label)
    admin_id = builder.user(org, Role.ORG_ADMIN)
    analyst_id = builder.user(org, Role.ANALYST, permissions=ALL_PERMISSIONS)
    admin = builder.ctx(admin_id)
    analyst_ctx = builder.ctx(analyst_id)
    imported = dataset_service.import_dataset(
        admin, file_name=f"{label}.csv", data=ORDERS_CSV.encode(), dataset_name=f"{label} orders"
    )
    ds = imported["dataset_id"]
    # Version 2 with a cleaning-operation record.
    cleaning_service.apply_cleaning(admin, ds, "normalize_case", {"columns": ["product"], "mode": "title"})
    versions = versioning_service.list_versions(admin, ds)
    assert [v["version_number"] for v in versions] == [2, 1]
    organization_admin.grant_dataset_access(admin, ds, analyst_id)
    admin_answer = analytics_engine.answer_question(admin, ds, QUESTION)
    analyst_answer = analytics_engine.answer_question(analyst_ctx, ds, QUESTION)
    assert admin_answer["status"] == analyst_answer["status"] == "success"
    report = analyst.save_report(admin, history_id=admin_answer["history_id"], title=f"{label} revenue")
    analyst_report = analyst.save_report(
        analyst_ctx, history_id=analyst_answer["history_id"], title=f"{label} analyst revenue"
    )
    rule = anomaly_service.create_rule(
        admin, dataset_id=ds, name=f"{label} outliers", method="iqr", threshold=1.5, metric_column="revenue"
    )
    run = anomaly_service.run_rule(admin, rule["id"])
    assert run["new_events"] >= 1
    organization_admin.update_org_settings(admin, {"text_missing_placeholder": f"Not given {label}"})
    with session_scope() as db:
        premium_id = db.execute(
            select(SubscriptionPlan.id).where(SubscriptionPlan.code == "premium")
        ).scalar_one()
    return SimpleNamespace(
        label=label,
        org=org,
        admin_id=admin_id,
        analyst_id=analyst_id,
        admin=admin,
        analyst=analyst_ctx,
        ds=ds,
        v1=versions[1]["version_id"],
        v2=versions[0]["version_id"],
        history=admin_answer["history_id"],
        analyst_history=analyst_answer["history_id"],
        report=report["report_id"],
        analyst_report=analyst_report["report_id"],
        rule=rule["id"],
        event=run["events"][0]["id"],
        plan_id=premium_id,
    )


@pytest.fixture(scope="module")
def world(builder) -> SimpleNamespace:
    a = build_tenant(builder, "Alpha")
    b = build_tenant(builder, "Bravo")
    return SimpleNamespace(a=a, b=b, builder=builder)


# ---------------------------------------------------------------------------
# Snapshots: every row and file owned by one organization (plus the plan catalogue)
# ---------------------------------------------------------------------------


def _org_tables():
    return [t for t in Base.metadata.sorted_tables if "organization_id" in t.columns]


def snapshot(org_id: int) -> dict[str, Any]:
    state: dict[str, Any] = {}
    with session_scope() as db:
        for table in _org_tables():
            rows = db.execute(
                select(table).where(table.c.organization_id == org_id).order_by(*table.primary_key)
            ).all()
            state[table.name] = [tuple(row) for row in rows]
        org_table = Organization.__table__
        state["organizations"] = [
            tuple(r) for r in db.execute(select(org_table).where(org_table.c.id == org_id)).all()
        ]
        for model in (SubscriptionPlan, PlanFeature):
            table = model.__table__
            state[table.name] = [
                tuple(r) for r in db.execute(select(table).order_by(*table.primary_key)).all()
            ]
    folder = storage.storage_root() / "orgs" / str(org_id)
    state["files"] = sorted(
        (p.relative_to(folder).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
        for p in folder.rglob("*")
        if p.is_file()
    )
    return state


def assert_unchanged(before: dict[str, Any], after: dict[str, Any]) -> None:
    changed = [key for key in before if before[key] != after.get(key)]
    assert changed == [], f"organization B data changed in: {changed}"


# ---------------------------------------------------------------------------
# The cross-tenant table
# ---------------------------------------------------------------------------

NOT_FOUND = NotFoundError
DENIED = PermissionDeniedError
NOOP = "noop"  # documented "never raises" helpers: must silently do nothing for foreign ids
SCOPED = "scoped"  # the foreign id is ignored and only the caller's own organization is returned


class Spy:
    """A transform that records whether it was ever invoked."""

    def __init__(self) -> None:
        self.calls = 0

    def frame(self, df: pd.DataFrame):
        self.calls += 1
        return df, {}, {}

    def batch(self, df: pd.DataFrame, offset: int):
        self.calls += 1
        return df, {}


@dataclass(frozen=True)
class Case:
    case_id: str
    function: str  # module.function as introspected
    call: Callable[[AuthContext, SimpleNamespace, SimpleNamespace, Spy], Any]  # (ctx, A, B, spy)
    admin: Any = NOT_FOUND
    analyst: Any = NOT_FOUND


AGG_PLAN = {"operation": "aggregate", "metrics": [{"column": "revenue", "agg": "sum"}]}
BAD_ERROR = ValidationError("simulated failure")


def _resolve_in_session(ctx, dataset_id, version_id=None):
    with session_scope() as db:
        return versioning_service.resolve_version(db, ctx, dataset_id, version_id)


# fmt: off
CASES: list[Case] = [
    # services.analytics_engine
    Case("answer_question", "services.analytics_engine.answer_question",
         lambda c, a, b, s: analytics_engine.answer_question(c, b.ds, QUESTION)),
    Case("answer_question-own-ds-foreign-version", "services.analytics_engine.answer_question",
         lambda c, a, b, s: analytics_engine.answer_question(c, a.ds, QUESTION, version_id=b.v1)),
    Case("get_schema_summary", "services.analytics_engine.get_schema_summary",
         lambda c, a, b, s: analytics_engine.get_schema_summary(c, b.ds)),
    Case("get_schema_summary-own-ds-foreign-version", "services.analytics_engine.get_schema_summary",
         lambda c, a, b, s: analytics_engine.get_schema_summary(c, a.ds, version_id=b.v2)),
    Case("run_plan", "services.analytics_engine.run_plan",
         lambda c, a, b, s: analytics_engine.run_plan(c, b.ds, AGG_PLAN)),
    # services.anomaly_service (Org Admin only)
    Case("create_rule", "services.anomaly_service.create_rule",
         lambda c, a, b, s: anomaly_service.create_rule(
             c, dataset_id=b.ds, name="stolen", method="iqr", threshold=1.5, metric_column="revenue"),
         analyst=DENIED),
    Case("update_rule", "services.anomaly_service.update_rule",
         lambda c, a, b, s: anomaly_service.update_rule(c, b.rule, threshold=9.0), analyst=DENIED),
    Case("set_rule_active", "services.anomaly_service.set_rule_active",
         lambda c, a, b, s: anomaly_service.set_rule_active(c, b.rule, False), analyst=DENIED),
    Case("delete_rule", "services.anomaly_service.delete_rule",
         lambda c, a, b, s: anomaly_service.delete_rule(c, b.rule), analyst=DENIED),
    Case("run_rule", "services.anomaly_service.run_rule",
         lambda c, a, b, s: anomaly_service.run_rule(c, b.rule), analyst=DENIED),
    Case("list_events-rule-filter", "services.anomaly_service.list_events",
         lambda c, a, b, s: anomaly_service.list_events(c, rule_id=b.rule), analyst=DENIED),
    Case("acknowledge_event", "services.anomaly_service.acknowledge_event",
         lambda c, a, b, s: anomaly_service.acknowledge_event(c, b.event), analyst=DENIED),
    # services.append_service (mixed pairs are the exfiltration path)
    Case("analyze_append-own-target-foreign-source", "services.append_service.analyze_append",
         lambda c, a, b, s: append_service.analyze_append(c, a.ds, b.ds)),
    Case("analyze_append-foreign-target", "services.append_service.analyze_append",
         lambda c, a, b, s: append_service.analyze_append(c, b.ds, a.ds)),
    Case("preview_append", "services.append_service.preview_append",
         lambda c, a, b, s: append_service.preview_append(c, a.ds, b.ds)),
    Case("apply_append-into-own", "services.append_service.apply_append",
         lambda c, a, b, s: append_service.apply_append(c, a.ds, b.ds, confirm=True)),
    Case("apply_append-into-foreign", "services.append_service.apply_append",
         lambda c, a, b, s: append_service.apply_append(c, b.ds, a.ds, confirm=True)),
    Case("apply_append-new-dataset", "services.append_service.apply_append",
         lambda c, a, b, s: append_service.apply_append(
             c, a.ds, b.ds, output="new_dataset", new_dataset_name="copied", confirm=True)),
    Case("resolve_pair", "services.append_service.resolve_pair",
         lambda c, a, b, s: append_service.resolve_pair(c, a.ds, b.ds, permission=None)),
    Case("record_operation_failure", "services.append_service.record_operation_failure",
         lambda c, a, b, s: append_service.record_operation_failure(c, b.ds, "append", {}, BAD_ERROR),
         admin=NOOP, analyst=NOOP),
    # services.audit_service
    Case("list_audit_logs-org-filter", "services.audit_service.list_audit_logs",
         lambda c, a, b, s: audit_service.list_audit_logs(c, organization_id=b.org), admin=SCOPED, analyst=DENIED),
    # services.cleaning_service
    Case("preview_cleaning", "services.cleaning_service.preview_cleaning",
         lambda c, a, b, s: cleaning_service.preview_cleaning(c, b.ds, "trim_whitespace", {"columns": ["region"]})),
    Case("apply_cleaning", "services.cleaning_service.apply_cleaning",
         lambda c, a, b, s: cleaning_service.apply_cleaning(c, b.ds, "trim_whitespace", {"columns": ["region"]})),
    Case("detect_casing_issues", "services.cleaning_service.detect_casing_issues",
         lambda c, a, b, s: cleaning_service.detect_casing_issues(c, b.ds)),
    Case("identify_redundant_columns", "services.cleaning_service.identify_redundant_columns",
         lambda c, a, b, s: cleaning_service.identify_redundant_columns(c, b.ds)),
    Case("outlier_report", "services.cleaning_service.outlier_report",
         lambda c, a, b, s: cleaning_service.outlier_report(c, b.ds, "revenue")),
    # services.dataset_service
    Case("get_dataset", "services.dataset_service.get_dataset",
         lambda c, a, b, s: dataset_service.get_dataset(c, b.ds)),
    Case("export_dataset", "services.dataset_service.export_dataset",
         lambda c, a, b, s: dataset_service.export_dataset(c, b.ds)),
    Case("export_dataset-own-ds-foreign-version", "services.dataset_service.export_dataset",
         lambda c, a, b, s: dataset_service.export_dataset(c, a.ds, version_id=b.v1, file_format="xlsx")),
    Case("download_original", "services.dataset_service.download_original",
         lambda c, a, b, s: dataset_service.download_original(c, b.ds)),
    Case("set_dataset_status", "services.dataset_service.set_dataset_status",
         lambda c, a, b, s: dataset_service.set_dataset_status(c, b.ds, archived=True), analyst=DENIED),
    Case("update_dataset_details", "services.dataset_service.update_dataset_details",
         lambda c, a, b, s: dataset_service.update_dataset_details(c, b.ds, name="renamed"), analyst=DENIED),
    # services.duplicate_manager
    Case("find_exact_duplicates", "services.duplicate_manager.find_exact_duplicates",
         lambda c, a, b, s: duplicate_manager.find_exact_duplicates(c, b.ds)),
    Case("preview_remove_duplicates", "services.duplicate_manager.preview_remove_duplicates",
         lambda c, a, b, s: duplicate_manager.preview_remove_duplicates(c, b.ds)),
    Case("remove_duplicates", "services.duplicate_manager.remove_duplicates",
         lambda c, a, b, s: duplicate_manager.remove_duplicates(c, b.ds, confirm=True)),
    Case("find_value_duplicates", "services.duplicate_manager.find_value_duplicates",
         lambda c, a, b, s: duplicate_manager.find_value_duplicates(c, b.ds, ["region"])),
    Case("validate_identifier", "services.duplicate_manager.validate_identifier",
         lambda c, a, b, s: duplicate_manager.validate_identifier(c, b.ds, "order_id")),
    Case("preview_generate_ids", "services.duplicate_manager.preview_generate_ids",
         lambda c, a, b, s: duplicate_manager.preview_generate_ids(c, b.ds, "order_id")),
    Case("generate_ids", "services.duplicate_manager.generate_ids",
         lambda c, a, b, s: duplicate_manager.generate_ids(c, b.ds, "order_id", confirm=True)),
    Case("find_composite_key_conflicts", "services.duplicate_manager.find_composite_key_conflicts",
         lambda c, a, b, s: duplicate_manager.find_composite_key_conflicts(c, b.ds, ["region", "product"])),
    Case("resolve_composite_key_duplicates", "services.duplicate_manager.resolve_composite_key_duplicates",
         lambda c, a, b, s: duplicate_manager.resolve_composite_key_duplicates(
             c, b.ds, ["region", "product"], policy="keep_first", confirm=True)),
    # services.merge_service
    Case("analyze_merge-own-left-foreign-right", "services.merge_service.analyze_merge",
         lambda c, a, b, s: merge_service.analyze_merge(
             c, a.ds, b.ds, left_on=["region"], right_on=["region"], how="inner")),
    Case("analyze_merge-foreign-left", "services.merge_service.analyze_merge",
         lambda c, a, b, s: merge_service.analyze_merge(
             c, b.ds, a.ds, left_on=["region"], right_on=["region"], how="left")),
    Case("preview_merge", "services.merge_service.preview_merge",
         lambda c, a, b, s: merge_service.preview_merge(
             c, a.ds, b.ds, left_on=["order_id"], right_on=["order_id"], how="left")),
    Case("apply_merge", "services.merge_service.apply_merge",
         lambda c, a, b, s: merge_service.apply_merge(
             c, a.ds, b.ds, left_on=["order_id"], right_on=["order_id"], how="left",
             new_dataset_name="joined", confirm=True, confirm_many_to_many=True)),
    # services.missing_values
    Case("analyze_missing", "services.missing_values.analyze_missing",
         lambda c, a, b, s: missing_values.analyze_missing(c, b.ds)),
    Case("recommend_plan", "services.missing_values.recommend_plan",
         lambda c, a, b, s: missing_values.recommend_plan(c, b.ds)),
    Case("preview_missing_treatment", "services.missing_values.preview_missing_treatment",
         lambda c, a, b, s: missing_values.preview_missing_treatment(
             c, b.ds, [{"column": "revenue", "strategy": "median"}])),
    Case("apply_missing_treatment", "services.missing_values.apply_missing_treatment",
         lambda c, a, b, s: missing_values.apply_missing_treatment(
             c, b.ds, [{"column": "revenue", "strategy": "median"}])),
    Case("detect_placeholders", "services.missing_values.detect_placeholders",
         lambda c, a, b, s: missing_values.detect_placeholders(c, b.ds)),
    Case("standardize_placeholders", "services.missing_values.standardize_placeholders",
         lambda c, a, b, s: missing_values.standardize_placeholders(c, b.ds, ["N/A"])),
    # services.profiling_service
    Case("profile_dataset", "services.profiling_service.profile_dataset",
         lambda c, a, b, s: profiling_service.profile_dataset(c, b.ds)),
    Case("profile_dataset-own-ds-foreign-version", "services.profiling_service.profile_dataset",
         lambda c, a, b, s: profiling_service.profile_dataset(c, a.ds, version_id=b.v1)),
    Case("health_check", "services.profiling_service.health_check",
         lambda c, a, b, s: profiling_service.health_check(c, b.ds)),
    Case("column_summary_for_ai", "services.profiling_service.column_summary_for_ai",
         lambda c, a, b, s: profiling_service.column_summary_for_ai(c, b.ds)),
    Case("repeated_values", "services.profiling_service.repeated_values",
         lambda c, a, b, s: profiling_service.repeated_values(c, b.ds, ["region"])),
    # services.versioning_service
    Case("get_version_ref", "services.versioning_service.get_version_ref",
         lambda c, a, b, s: versioning_service.get_version_ref(c, b.ds)),
    Case("get_version_ref-own-ds-foreign-version", "services.versioning_service.get_version_ref",
         lambda c, a, b, s: versioning_service.get_version_ref(c, a.ds, b.v1)),
    Case("resolve_version", "services.versioning_service.resolve_version",
         lambda c, a, b, s: _resolve_in_session(c, b.ds)),
    Case("resolve_version-own-ds-foreign-version", "services.versioning_service.resolve_version",
         lambda c, a, b, s: _resolve_in_session(c, a.ds, b.v2)),
    Case("load_frame", "services.versioning_service.load_frame",
         lambda c, a, b, s: versioning_service.load_frame(c, b.ds)),
    Case("load_frame-own-ds-foreign-version", "services.versioning_service.load_frame",
         lambda c, a, b, s: versioning_service.load_frame(c, a.ds, version_id=b.v1)),
    Case("load_frame_with_ref", "services.versioning_service.load_frame_with_ref",
         lambda c, a, b, s: versioning_service.load_frame_with_ref(c, b.ds, version_id=b.v2)),
    Case("load_preview", "services.versioning_service.load_preview",
         lambda c, a, b, s: versioning_service.load_preview(c, b.ds)),
    Case("load_preview-own-ds-foreign-version", "services.versioning_service.load_preview",
         lambda c, a, b, s: versioning_service.load_preview(c, a.ds, version_id=b.v2)),
    Case("iter_frame_batches", "services.versioning_service.iter_frame_batches",
         lambda c, a, b, s: list(versioning_service.iter_frame_batches(c, b.ds))),
    Case("iter_frame_batches-own-ds-foreign-version", "services.versioning_service.iter_frame_batches",
         lambda c, a, b, s: list(versioning_service.iter_frame_batches(c, a.ds, version_id=b.v1))),
    Case("load_frame_with_ref-own-ds-foreign-version", "services.versioning_service.load_frame_with_ref",
         lambda c, a, b, s: versioning_service.load_frame_with_ref(c, a.ds, version_id=b.v1)),
    Case("list_versions", "services.versioning_service.list_versions",
         lambda c, a, b, s: versioning_service.list_versions(c, b.ds)),
    Case("list_operations", "services.versioning_service.list_operations",
         lambda c, a, b, s: versioning_service.list_operations(c, b.ds)),
    Case("verify_original", "services.versioning_service.verify_original",
         lambda c, a, b, s: versioning_service.verify_original(c, b.ds)),
    Case("restore_version", "services.versioning_service.restore_version",
         lambda c, a, b, s: versioning_service.restore_version(c, b.ds, b.v1)),
    Case("restore_version-own-ds-foreign-version", "services.versioning_service.restore_version",
         lambda c, a, b, s: versioning_service.restore_version(c, a.ds, b.v1)),
    Case("apply_frame_operation", "services.versioning_service.apply_frame_operation",
         lambda c, a, b, s: versioning_service.apply_frame_operation(
             c, b.ds, operation_type="trim_whitespace", parameters={}, transform=s.frame)),
    Case("apply_streaming_operation", "services.versioning_service.apply_streaming_operation",
         lambda c, a, b, s: versioning_service.apply_streaming_operation(
             c, b.ds, operation_type="trim_whitespace", parameters={}, transform_batch=s.batch)),
    Case("record_failed_operation", "services.versioning_service.record_failed_operation",
         lambda c, a, b, s: versioning_service.record_failed_operation(c, b.ds, "trim_whitespace", {}, BAD_ERROR),
         admin=NOOP, analyst=NOOP),
    Case("record_failure", "services.versioning_service.record_failure",
         lambda c, a, b, s: versioning_service.record_failure(c, b.ds, "trim_whitespace", {}, BAD_ERROR),
         admin=NOOP, analyst=NOOP),
    # modules.analyst
    Case("get_history_item", "modules.analyst.get_history_item",
         lambda c, a, b, s: analyst.get_history_item(c, b.history)),
    Case("get_history_item-analyst-row", "modules.analyst.get_history_item",
         lambda c, a, b, s: analyst.get_history_item(c, b.analyst_history)),
    Case("list_my_history-dataset-filter", "modules.analyst.list_my_history",
         lambda c, a, b, s: analyst.list_my_history(c, dataset_id=b.ds)),
    Case("export_history_result", "modules.analyst.export_history_result",
         lambda c, a, b, s: analyst.export_history_result(c, b.history)),
    Case("save_report", "modules.analyst.save_report",
         lambda c, a, b, s: analyst.save_report(c, history_id=b.history, title="copied")),
    Case("list_reports-dataset-filter", "modules.analyst.list_reports",
         lambda c, a, b, s: analyst.list_reports(c, dataset_id=b.ds)),
    Case("get_report", "modules.analyst.get_report",
         lambda c, a, b, s: analyst.get_report(c, b.report)),
    Case("get_report-analyst-report", "modules.analyst.get_report",
         lambda c, a, b, s: analyst.get_report(c, b.analyst_report)),
    Case("update_report", "modules.analyst.update_report",
         lambda c, a, b, s: analyst.update_report(c, b.report, title="defaced")),
    Case("delete_report", "modules.analyst.delete_report",
         lambda c, a, b, s: analyst.delete_report(c, b.report)),
    Case("export_report", "modules.analyst.export_report",
         lambda c, a, b, s: analyst.export_report(c, b.report, file_format="csv")),
    # modules.organization_admin (Org Admin only)
    Case("update_user", "modules.organization_admin.update_user",
         lambda c, a, b, s: organization_admin.update_user(c, b.analyst_id, full_name="Renamed"), analyst=DENIED),
    Case("set_user_active", "modules.organization_admin.set_user_active",
         lambda c, a, b, s: organization_admin.set_user_active(c, b.analyst_id, False), analyst=DENIED),
    Case("reset_user_password", "modules.organization_admin.reset_user_password",
         lambda c, a, b, s: organization_admin.reset_user_password(c, b.admin_id, "Hijack#Pass123"),
         analyst=DENIED),
    Case("set_user_permissions", "modules.organization_admin.set_user_permissions",
         lambda c, a, b, s: organization_admin.set_user_permissions(c, b.analyst_id, set(ALL_PERMISSIONS)),
         analyst=DENIED),
    Case("grant_dataset_access-foreign-dataset", "modules.organization_admin.grant_dataset_access",
         lambda c, a, b, s: organization_admin.grant_dataset_access(c, b.ds, a.analyst_id), analyst=DENIED),
    Case("grant_dataset_access-foreign-user", "modules.organization_admin.grant_dataset_access",
         lambda c, a, b, s: organization_admin.grant_dataset_access(c, a.ds, b.analyst_id), analyst=DENIED),
    Case("revoke_dataset_access", "modules.organization_admin.revoke_dataset_access",
         lambda c, a, b, s: organization_admin.revoke_dataset_access(c, b.ds, b.analyst_id), analyst=DENIED),
    Case("revoke_dataset_access-foreign-user", "modules.organization_admin.revoke_dataset_access",
         lambda c, a, b, s: organization_admin.revoke_dataset_access(c, a.ds, b.analyst_id), analyst=DENIED),
    Case("list_dataset_access", "modules.organization_admin.list_dataset_access",
         lambda c, a, b, s: organization_admin.list_dataset_access(c, b.ds), analyst=DENIED),
    Case("org_analysis_history-user-filter", "modules.organization_admin.org_analysis_history",
         lambda c, a, b, s: organization_admin.org_analysis_history(c, user_id=b.analyst_id), analyst=DENIED),
    Case("org_analysis_history-dataset-filter", "modules.organization_admin.org_analysis_history",
         lambda c, a, b, s: organization_admin.org_analysis_history(c, dataset_id=b.ds), analyst=DENIED),
    # modules.super_admin (platform operators only)
    Case("update_organization", "modules.super_admin.update_organization",
         lambda c, a, b, s: super_admin.update_organization(c, b.org, name="Taken over"), admin=DENIED, analyst=DENIED),
    Case("set_organization_status", "modules.super_admin.set_organization_status",
         lambda c, a, b, s: super_admin.set_organization_status(c, b.org, "suspended"), admin=DENIED, analyst=DENIED),
    Case("create_org_admin", "modules.super_admin.create_org_admin",
         lambda c, a, b, s: super_admin.create_org_admin(
             c, b.org, username="intruder", email="intruder@example.com", password="Intruder#Pass1"),
         admin=DENIED, analyst=DENIED),
    Case("assign_plan", "modules.super_admin.assign_plan",
         lambda c, a, b, s: super_admin.assign_plan(c, b.org, "basic"), admin=DENIED, analyst=DENIED),
    Case("list_subscriptions-org-filter", "modules.super_admin.list_subscriptions",
         lambda c, a, b, s: super_admin.list_subscriptions(c, organization_id=b.org), admin=DENIED, analyst=DENIED),
    Case("update_plan", "modules.super_admin.update_plan",
         lambda c, a, b, s: super_admin.update_plan(c, b.plan_id, price_cents=1), admin=DENIED, analyst=DENIED),
    Case("set_plan_feature", "modules.super_admin.set_plan_feature",
         lambda c, a, b, s: super_admin.set_plan_feature(c, b.plan_id, "voice_analysis", False),
         admin=DENIED, analyst=DENIED),
    Case("set_plan_active", "modules.super_admin.set_plan_active",
         lambda c, a, b, s: super_admin.set_plan_active(c, b.plan_id, False), admin=DENIED, analyst=DENIED),
]
# fmt: on

# Functions with id-like parameters that are deliberately not in CASES, with the reason. Only helpers
# whose first parameter is an open ``db`` session (they run inside a transaction their caller already
# authorized and receive the organization id resolved from the AuthContext) or pure functions without a
# ``ctx`` may be exempt.
EXEMPT = {
    "services.anomaly_service.make_fingerprint": "pure hash of a finding; reads no data",
    "services.audit_service.record_audit": "writes inside the caller's authorized transaction",
    "modules.org_settings.get_org_settings": "db helper; callers pass the authorized organization id",
    "modules.organization_admin.create_user_row": "db helper used after authorize() inside the same transaction",
    "modules.subscriptions.active_dataset_count": "db helper for plan checks after authorize()",
    "modules.subscriptions.active_user_count": "db helper for plan checks after authorize()",
    "modules.subscriptions.enabled_features": "db helper for plan checks after authorize()",
    "modules.subscriptions.enforce_dataset_quota": "db helper for plan checks after authorize()",
    "modules.subscriptions.enforce_storage_quota": "db helper for plan checks after authorize()",
    "modules.subscriptions.enforce_upload_size": "db helper for plan checks after authorize()",
    "modules.subscriptions.enforce_user_quota": "db helper for plan checks after authorize()",
    "modules.subscriptions.get_active_plan": "db helper for plan checks after authorize()",
    "modules.subscriptions.get_active_subscription": "db helper for plan checks after authorize()",
    "modules.subscriptions.get_limits": "db helper for plan checks after authorize()",
    "modules.subscriptions.org_has_feature": "db helper for plan checks after authorize()",
    "modules.subscriptions.org_storage_bytes": "db helper for plan checks after authorize()",
    "modules.subscriptions.require_feature": "db helper called by authorize() itself",
    "modules.usage_tracking.enforce_monthly_analysis_quota": "db helper called after authorize()",
    "modules.usage_tracking.monthly_total": "db helper called after authorize()",
    "modules.usage_tracking.record_usage": "db helper writing inside the caller's authorized transaction",
    "modules.usage_tracking.total_since": "db helper called after authorize()",
    "modules.usage_tracking.usage_summary": "db helper called after authorize()",
}

ID_PARAM = re.compile(r"(^id$|_id$)")


def _public_functions_with_ids() -> dict[str, inspect.Signature]:
    found: dict[str, inspect.Signature] = {}
    for package in (services, modules):
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{info.name}")
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_") or obj.__module__ != module.__name__:
                    continue
                signature = inspect.signature(obj)
                if any(ID_PARAM.search(p) for p in signature.parameters):
                    found[f"{module.__name__}.{name}"] = signature
    return found


def test_every_function_taking_an_owned_id_is_in_the_table():
    discovered = _public_functions_with_ids()
    covered = {case.function for case in CASES}
    missing = sorted(set(discovered) - covered - set(EXEMPT))
    assert missing == [], f"add these functions to CASES (or justify them in EXEMPT): {missing}"
    unknown = sorted((covered | set(EXEMPT)) - set(discovered))
    assert unknown == [], f"table entries that do not match a discovered function: {unknown}"
    # An exemption is only acceptable for helpers that run inside a caller's transaction (first
    # parameter ``db``) or pure functions; a ctx-first entry point must be in the table.
    entry_points = sorted(name for name in EXEMPT if next(iter(discovered[name].parameters)) == "ctx")
    assert entry_points == [], f"ctx-first functions must be tested, not exempted: {entry_points}"
    for name in EXEMPT:
        params = list(discovered[name].parameters)
        assert params[0] == "db" or "ctx" not in params, name
    assert len({case.case_id for case in CASES}) == len(CASES)


ROLE_CASES = [
    pytest.param(case, role, id=f"{case.case_id}[{role}]") for case in CASES for role in ("admin", "analyst")
]


@pytest.mark.parametrize(("case", "role"), ROLE_CASES)
def test_foreign_ids_are_rejected_and_leave_b_unchanged(world, case: Case, role: str):
    a, b = world.a, world.b
    ctx = a.admin if role == "admin" else a.analyst
    expected = case.admin if role == "admin" else case.analyst
    spy = Spy()
    before_b = snapshot(b.org)
    if expected is NOOP:
        assert case.call(ctx, a, b, spy) is None
    elif expected is SCOPED:
        rows = case.call(ctx, a, b, spy)
        assert rows and all(row["organization_id"] == a.org for row in rows)
    else:
        with pytest.raises(expected) as info:
            case.call(ctx, a, b, spy)
        if expected is NotFoundError:
            assert type(info.value) is NotFoundError
        assert b.label not in info.value.user_message  # nothing about B is echoed back
    assert spy.calls == 0, "a transform ran on another organization's data"
    assert_unchanged(before_b, snapshot(b.org))


def test_b_still_works_for_its_own_members(world):
    """Sanity check that the table above is meaningful: B's ids are valid for B's own users."""
    b = world.b
    assert dataset_service.get_dataset(b.admin, b.ds)["dataset_id"] == b.ds
    assert len(versioning_service.load_frame(b.analyst, b.ds, version_id=b.v1)) == 12
    assert analyst.get_report(b.admin, b.report)["report_id"] == b.report
    assert analyst.get_history_item(b.analyst, b.analyst_history)["history_id"] == b.analyst_history
    assert [e["id"] for e in anomaly_service.list_events(b.admin, rule_id=b.rule)]
    assert {u["user_id"] for u in organization_admin.list_dataset_access(b.admin, b.ds)} == {b.analyst_id}
    assert dataset_service.download_original(b.admin, b.ds)[0] == ORDERS_CSV.encode()


def test_listings_never_include_other_organizations(world):
    a, b = world.a, world.b
    for ctx in (a.admin, a.analyst):
        assert {d["dataset_id"] for d in dataset_service.list_datasets(ctx)} == {a.ds}
        assert all(r["report_id"] != b.report for r in analyst.list_reports(ctx))
        assert all(
            h["history_id"] not in (b.history, b.analyst_history) for h in analyst.list_my_history(ctx)
        )
    assert {r["id"] for r in anomaly_service.list_rules(a.admin)} == {a.rule}
    assert {e["rule_id"] for e in anomaly_service.list_events(a.admin)} == {a.rule}
    assert all(n["rule_id"] != b.rule for n in anomaly_service.list_notifications(a.admin))
    users = {u["id"] if "id" in u else u["user_id"] for u in organization_admin.list_users(a.admin)}
    assert users == {a.admin_id, a.analyst_id}
    assert {h["id"] for h in organization_admin.org_analysis_history(a.admin)} == {
        a.history,
        a.analyst_history,
    }
    assert all(row["organization_id"] == a.org for row in audit_service.list_audit_logs(a.admin))
    settings = organization_admin.get_settings_view(a.admin)
    assert "Bravo" not in repr(settings) and "Alpha" in repr(settings)
    dashboard = organization_admin.org_dashboard(a.admin)
    assert "Bravo" not in repr(dashboard)
    usage = organization_admin.usage_report(a.admin, months=1)
    with session_scope() as db:
        from database.models import UsageRecord

        own = (
            db.execute(select(UsageRecord.metric).where(UsageRecord.organization_id == a.org)).scalars().all()
        )
    analysis_total = sum(month["totals"].get("analysis_query", 0) for month in usage)
    assert analysis_total == own.count("analysis_query")
    storage_info = dataset_service.storage_summary(a.admin)
    assert storage_info["active_datasets"] == 1


# ---------------------------------------------------------------------------
# Super Admin, missing grants, suspension and forged contexts
# ---------------------------------------------------------------------------

TENANT_READS: dict[str, Callable[[AuthContext, SimpleNamespace], Any]] = {
    "list_datasets": lambda c, b: dataset_service.list_datasets(c),
    "get_dataset": lambda c, b: dataset_service.get_dataset(c, b.ds),
    "load_frame": lambda c, b: versioning_service.load_frame(c, b.ds),
    "load_preview": lambda c, b: versioning_service.load_preview(c, b.ds),
    "list_versions": lambda c, b: versioning_service.list_versions(c, b.ds),
    "list_operations": lambda c, b: versioning_service.list_operations(c, b.ds),
    "profile_dataset": lambda c, b: profiling_service.profile_dataset(c, b.ds),
    "health_check": lambda c, b: profiling_service.health_check(c, b.ds),
    "analyze_missing": lambda c, b: missing_values.analyze_missing(c, b.ds),
    "find_exact_duplicates": lambda c, b: duplicate_manager.find_exact_duplicates(c, b.ds),
    "export_dataset": lambda c, b: dataset_service.export_dataset(c, b.ds),
    "download_original": lambda c, b: dataset_service.download_original(c, b.ds),
    "answer_question": lambda c, b: analytics_engine.answer_question(c, b.ds, QUESTION),
    "get_schema_summary": lambda c, b: analytics_engine.get_schema_summary(c, b.ds),
    "list_my_history": lambda c, b: analyst.list_my_history(c),
    "get_history_item": lambda c, b: analyst.get_history_item(c, b.history),
    "list_reports": lambda c, b: analyst.list_reports(c),
    "get_report": lambda c, b: analyst.get_report(c, b.report),
    "export_report": lambda c, b: analyst.export_report(c, b.report),
    "list_rules": lambda c, b: anomaly_service.list_rules(c),
    "list_events": lambda c, b: anomaly_service.list_events(c),
    "list_users": lambda c, b: organization_admin.list_users(c),
    "org_dashboard": lambda c, b: organization_admin.org_dashboard(c),
    "org_analysis_history": lambda c, b: organization_admin.org_analysis_history(c),
    "usage_report": lambda c, b: organization_admin.usage_report(c),
    "get_settings_view": lambda c, b: organization_admin.get_settings_view(c),
    "storage_summary": lambda c, b: dataset_service.storage_summary(c),
}


@pytest.fixture(scope="module")
def platform(world) -> AuthContext:
    return world.builder.ctx(world.builder.user(None, Role.SUPER_ADMIN))


@pytest.mark.parametrize("name", sorted(TENANT_READS))
def test_super_admin_cannot_read_tenant_data_through_tenant_functions(world, platform, name):
    before = snapshot(world.b.org)
    with pytest.raises(PermissionDeniedError):
        TENANT_READS[name](platform, world.b)
    assert_unchanged(before, snapshot(world.b.org))


def test_super_admin_audit_access_is_explicit_and_audited(world, platform):
    rows = audit_service.list_audit_logs(platform, organization_id=world.b.org)
    assert rows and all(r["organization_id"] == world.b.org for r in rows)


DATASET_READS = [
    "get_dataset",
    "load_frame",
    "load_preview",
    "list_versions",
    "list_operations",
    "profile_dataset",
    "health_check",
    "analyze_missing",
    "find_exact_duplicates",
    "export_dataset",
    "download_original",
    "answer_question",
    "get_schema_summary",
]


@pytest.fixture(scope="module")
def ungranted(world) -> AuthContext:
    """An analyst of B with every permission but no dataset grant."""
    return world.builder.ctx(world.builder.user(world.b.org, Role.ANALYST, permissions=ALL_PERMISSIONS))


@pytest.mark.parametrize("name", DATASET_READS)
def test_analyst_without_grant_gets_not_found(world, ungranted, name):
    with pytest.raises(NotFoundError):
        TENANT_READS[name](ungranted, world.b)


def test_analyst_without_grant_sees_nothing_in_listings(world, ungranted):
    b = world.b
    assert dataset_service.list_datasets(ungranted) == []
    assert analyst.list_reports(ungranted) == []
    with pytest.raises(NotFoundError):
        analyst.get_report(ungranted, b.report)
    with pytest.raises(NotFoundError):
        analyst.get_history_item(ungranted, b.analyst_history)  # not theirs, and they are not an admin
    with pytest.raises(NotFoundError):
        analyst.list_reports(ungranted, dataset_id=b.ds)


def test_revoking_a_grant_takes_effect_on_the_next_call(world):
    b = world.b
    viewer_id = world.builder.user(b.org, Role.ANALYST, permissions=ALL_PERMISSIONS)
    viewer = world.builder.ctx(viewer_id)
    organization_admin.grant_dataset_access(b.admin, b.ds, viewer_id)
    assert len(versioning_service.load_preview(viewer, b.ds)) == 12
    organization_admin.revoke_dataset_access(b.admin, b.ds, viewer_id)
    with pytest.raises(NotFoundError):
        versioning_service.load_preview(viewer, b.ds)


def test_suspending_b_blocks_b_immediately_and_leaves_a_untouched(world, platform):
    a, b = world.a, world.b
    with session_scope() as db:
        username = db.get(User, b.analyst_id).username
    token = login(username, PASSWORD).token
    assert validate_session(token).user_id == b.analyst_id
    before_a = snapshot(a.org)
    a_frame = versioning_service.load_frame(a.admin, a.ds)
    super_admin.set_organization_status(platform, b.org, "suspended", reason="isolation test")
    try:
        for ctx in (b.admin, b.analyst):
            for name in ("list_datasets", "get_dataset", "load_frame", "get_report", "answer_question"):
                with pytest.raises(PermissionDeniedError):
                    TENANT_READS[name](ctx, b)
        with pytest.raises((PermissionDeniedError, AuthenticationError)):
            validate_session(token)
        with pytest.raises(PermissionDeniedError):
            login(username, PASSWORD)
        # Organization A is not affected at all.
        assert {d["dataset_id"] for d in dataset_service.list_datasets(a.admin)} == {a.ds}
        pd.testing.assert_frame_equal(versioning_service.load_frame(a.analyst, a.ds), a_frame)
        assert analyst.get_report(a.admin, a.report)["report_id"] == a.report
        assert_unchanged(before_a, snapshot(a.org))
        with session_scope() as db:
            assert db.get(Organization, a.org).status == "active"
    finally:
        super_admin.set_organization_status(platform, b.org, "active")
    assert dataset_service.get_dataset(b.admin, b.ds)["dataset_id"] == b.ds  # historical data kept


def _forged_contexts(world) -> dict[str, Any]:
    a, b = world.a, world.b
    return {
        "own-user-other-org": AuthContext(user_id=a.admin_id, role=Role.ORG_ADMIN, organization_id=b.org),
        "analyst-claims-admin": AuthContext(user_id=a.analyst_id, role=Role.ORG_ADMIN, organization_id=a.org),
        "analyst-claims-other-org": AuthContext(
            user_id=a.analyst_id, role=Role.ANALYST, organization_id=b.org
        ),
        "admin-claims-super-admin": AuthContext(
            user_id=a.admin_id, role=Role.SUPER_ADMIN, organization_id=None
        ),
        "b-user-id-with-a-org": AuthContext(user_id=b.admin_id, role=Role.ORG_ADMIN, organization_id=a.org),
        "no-user": AuthContext(user_id=None, role=Role.ORG_ADMIN, organization_id=b.org),
        "unknown-user": AuthContext(user_id=987654, role=Role.ORG_ADMIN, organization_id=b.org),
        "plain-object": SimpleNamespace(user_id=b.admin_id, role=Role.ORG_ADMIN, organization_id=b.org),
    }


@pytest.mark.parametrize(
    "forgery",
    [
        "own-user-other-org",
        "analyst-claims-admin",
        "analyst-claims-other-org",
        "admin-claims-super-admin",
        "b-user-id-with-a-org",
        "no-user",
        "unknown-user",
        "plain-object",
    ],
)
def test_forged_auth_context_is_rejected(world, forgery):
    b = world.b
    ctx = _forged_contexts(world)[forgery]
    before = snapshot(b.org)
    calls = [
        lambda: dataset_service.list_datasets(ctx),
        lambda: dataset_service.get_dataset(ctx, b.ds),
        lambda: versioning_service.load_frame(ctx, b.ds),
        lambda: analyst.get_report(ctx, b.report),
        lambda: organization_admin.list_users(ctx),
        lambda: organization_admin.reset_user_password(ctx, b.analyst_id, "Forged#Pass123"),
        lambda: super_admin.list_organizations(ctx),
        lambda: anomaly_service.list_events(ctx),
    ]
    for call in calls:
        with pytest.raises((AuthenticationError, PermissionDeniedError)) as info:
            call()
        if forgery != "admin-claims-super-admin":
            # Identity forgeries are caught by the fresh database check, before any role logic.
            assert isinstance(info.value, AuthenticationError), (forgery, type(info.value))
    assert_unchanged(before, snapshot(b.org))


def test_system_context_is_only_constructed_by_the_anomaly_worker():
    """``AuthContext.system`` bypasses the user check by design; keep its use confined."""
    users: list[str] = []
    for folder in (
        "app.py",
        "auth",
        "core",
        "database",
        "modules",
        "services",
        "components",
        "views",
        "config",
    ):
        path = ROOT / folder
        files = [path] if path.is_file() else sorted(path.rglob("*.py")) if path.is_dir() else []
        for file in files:
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "system"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "AuthContext"
                ):
                    users.append(file.relative_to(ROOT).as_posix())
    assert set(users) == {"services/anomaly_service.py"}


def test_system_context_cannot_act_on_another_organizations_rule(world):
    a, b = world.a, world.b
    before = snapshot(b.org)
    with pytest.raises(NotFoundError):
        anomaly_service.run_rule(AuthContext.system(a.org), b.rule)
    assert_unchanged(before, snapshot(b.org))


def test_audit_log_records_no_cross_tenant_entries(world):
    """No audit row of organization B was written by organization A's users during the run."""
    a, b = world.a, world.b
    with session_scope() as db:
        rows = db.execute(
            select(AuditLog.actor_user_id).where(
                AuditLog.organization_id == b.org, AuditLog.actor_user_id.in_([a.admin_id, a.analyst_id])
            )
        ).all()
    assert rows == []


def test_snapshot_detects_changes_to_b(world):
    """Guards against a vacuous "unchanged" check: a real change by B's own admin is detected."""
    b = world.b
    before = snapshot(b.org)
    original = analyst.get_report(b.admin, b.report)["title"]
    analyst.update_report(b.admin, b.report, title="Temporary title")
    try:
        with pytest.raises(AssertionError):
            assert_unchanged(before, snapshot(b.org))
    finally:
        analyst.update_report(b.admin, b.report, title=original)


@pytest.fixture(scope="module")
def private_a_dataset(world) -> int:
    """A second dataset of organization A that A's analyst was never granted."""
    imported = dataset_service.import_dataset(
        world.a.admin, file_name="private.csv", data=ORDERS_CSV.encode(), dataset_name="Alpha private"
    )
    return imported["dataset_id"]


PAIR_CALLS: dict[str, Callable[[AuthContext, int, int], Any]] = {
    "analyze_append": lambda c, x, y: append_service.analyze_append(c, x, y),
    "preview_append": lambda c, x, y: append_service.preview_append(c, x, y),
    "apply_append": lambda c, x, y: append_service.apply_append(c, x, y, confirm=True),
    "analyze_merge": lambda c, x, y: merge_service.analyze_merge(
        c, x, y, left_on=["order_id"], right_on=["order_id"], how="inner"
    ),
    "preview_merge": lambda c, x, y: merge_service.preview_merge(
        c, x, y, left_on=["order_id"], right_on=["order_id"], how="inner"
    ),
    "apply_merge": lambda c, x, y: merge_service.apply_merge(
        c,
        x,
        y,
        left_on=["order_id"],
        right_on=["order_id"],
        how="inner",
        new_dataset_name="mixed",
        confirm=True,
    ),
}


@pytest.mark.parametrize("name", sorted(PAIR_CALLS))
@pytest.mark.parametrize("order", ["granted-first", "ungranted-first"])
def test_analyst_cannot_combine_a_granted_dataset_with_an_ungranted_one(
    world, private_a_dataset, name, order
):
    a = world.a
    pair = (a.ds, private_a_dataset) if order == "granted-first" else (private_a_dataset, a.ds)
    before = snapshot(a.org)
    with pytest.raises(NotFoundError):
        PAIR_CALLS[name](a.analyst, *pair)
    assert_unchanged(before, snapshot(a.org))
