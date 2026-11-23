"""Analyst workspace: history visibility, saved reports and downloads."""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta

import openpyxl
import pandas as pd
import pytest
from sqlalchemy import delete, select

from config.settings import set_settings
from core.constants import Feature, Permission
from core.errors import FeatureNotAvailableError, NotFoundError, PermissionDeniedError, ValidationError
from database import session_scope
from database.models import AnalysisHistory, AuditLog, DatasetAccess, SavedReport, UsageRecord, UserPermission
from modules import analyst
from services import dataset_service

SCRIPT = "<script>alert(1)</script>"
PLAN = {
    "operation": "group_aggregate",
    "group_by": ["region"],
    "metrics": [{"column": "revenue", "agg": "sum"}],
}
CHART = {
    "type": "bar",
    "x": "region",
    "y": "revenue",
    "color": None,
    "title": "Revenue by region",
    "stacked": False,
}
BASE_TIME = datetime(2024, 6, 1, 12, 0, 0)


def preview(
    rows=None, *, columns=None, truncated=False, total_rows=None, chart=CHART, facts=None, warnings=None
):
    rows = [["North", 3800.0, "2024-01-05"], ["South", 1740.5, None]] if rows is None else rows
    return {
        "columns": columns or ["region", "revenue", "when"],
        "rows": rows,
        "truncated": truncated,
        "total_rows": len(rows) if total_rows is None else total_rows,
        "chart": chart,
        "facts": [{"label": "Total", "value": "5,540.5"}] if facts is None else facts,
        "warnings": warnings or [],
    }


def add_history(
    ctx,
    dataset_id,
    *,
    question="Revenue by region",
    status="success",
    payload=None,
    minutes=0,
    answer="North leads.",
) -> int:
    with session_scope() as db:
        row = AnalysisHistory(
            organization_id=ctx.organization_id,
            user_id=ctx.user_id,
            dataset_id=dataset_id,
            question=question,
            input_mode="text",
            planner="rules",
            plan_json=json.dumps(PLAN),
            status=status,
            answer_text=answer if status == "success" else None,
            result_preview_json=json.dumps(preview() if payload is None else payload),
            chart_type="bar",
            error_message=None if status == "success" else "It failed.",
            duration_ms=42,
            created_at=BASE_TIME + timedelta(minutes=minutes),
        )
        db.add(row)
        db.flush()
        return row.id


def audit_actions(action: str) -> list[AuditLog]:
    with session_scope() as db:
        rows = (
            db.execute(select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.id))
            .scalars()
            .all()
        )
        db.expunge_all()
        return list(rows)


class World:
    """One Premium organization with an admin, two analysts and a granted dataset."""

    def __init__(self, factory, org_id, admin_ctx, sales_frame):
        self.factory, self.org_id, self.admin = factory, org_id, admin_ctx
        self.alice = factory.analyst(org_id)
        self.bob = factory.analyst(org_id)
        self.dataset_id = factory.dataset(admin_ctx, sales_frame, "Sales 2024")
        factory.grant(self.dataset_id, self.alice.user_id)
        factory.grant(self.dataset_id, self.bob.user_id)

    def revoke(self, ctx) -> None:
        with session_scope() as db:
            db.execute(
                delete(DatasetAccess).where(
                    DatasetAccess.dataset_id == self.dataset_id, DatasetAccess.user_id == ctx.user_id
                )
            )


@pytest.fixture
def world(factory, org_id, admin_ctx, sales_frame) -> World:
    return World(factory, org_id, admin_ctx, sales_frame)


@pytest.fixture
def outsider(factory):
    """Admin and analyst of a different Premium organization."""
    other = factory.organization("Other Co", plan="premium")
    return factory.org_admin(other), factory.analyst(other), other


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_list_my_history_returns_only_own_rows_newest_first(world):
    first = add_history(world.alice, world.dataset_id, question="first", minutes=0)
    second = add_history(world.alice, world.dataset_id, question="second", minutes=5)
    add_history(world.bob, world.dataset_id, question="bobs")
    add_history(world.admin, world.dataset_id, question="admins")
    rows = analyst.list_my_history(world.alice)
    assert [r["history_id"] for r in rows] == [second, first]
    top = rows[0]
    assert (
        top["question"] == "second"
        and top["dataset_name"] == "Sales 2024"
        and top["dataset_id"] == world.dataset_id
    )
    assert (top["status"], top["planner"], top["input_mode"], top["duration_ms"]) == (
        "success",
        "rules",
        "text",
        42,
    )
    assert top["created_at"] == BASE_TIME + timedelta(minutes=5)
    assert [r["question"] for r in analyst.list_my_history(world.admin)] == ["admins"]


def test_list_my_history_filters_by_dataset_and_limit(world, factory, sales_frame):
    other_dataset = factory.dataset(world.admin, sales_frame, "Other")
    factory.grant(other_dataset, world.alice.user_id)
    add_history(world.alice, world.dataset_id, question="a", minutes=0)
    add_history(world.alice, other_dataset, question="b", minutes=1)
    add_history(world.alice, world.dataset_id, question="c", minutes=2)
    assert [r["question"] for r in analyst.list_my_history(world.alice, dataset_id=world.dataset_id)] == [
        "c",
        "a",
    ]
    assert [r["question"] for r in analyst.list_my_history(world.alice, limit=1)] == ["c"]


@pytest.mark.parametrize("limit", [0, -1, 501, True, "5"])
def test_list_my_history_rejects_bad_limits(world, limit):
    with pytest.raises(ValidationError):
        analyst.list_my_history(world.alice, limit=limit)


def test_list_my_history_dataset_filter_is_tenant_and_access_safe(world, factory, sales_frame, outsider):
    other_admin, _, _ = outsider
    foreign = factory.dataset(other_admin, sales_frame, "Foreign")
    with pytest.raises(NotFoundError):
        analyst.list_my_history(world.alice, dataset_id=foreign)
    ungranted = factory.dataset(world.admin, sales_frame, "Ungranted")
    with pytest.raises(NotFoundError):
        analyst.list_my_history(world.alice, dataset_id=ungranted)
    with pytest.raises(ValidationError):
        analyst.list_my_history(world.alice, dataset_id=0)


def test_get_history_item_returns_table_facts_chart_and_plan(world):
    history_id = add_history(world.alice, world.dataset_id)
    item = analyst.get_history_item(world.alice, history_id)
    assert item["history_id"] == history_id and item["dataset_name"] == "Sales 2024"
    assert item["answer_text"] == "North leads." and item["plan"] == PLAN and item["chart"] == CHART
    assert item["facts"] == [{"label": "Total", "value": "5,540.5"}]
    table = item["result"]
    assert list(table.columns) == ["region", "revenue", "when"] and len(table) == 2
    assert pd.api.types.is_datetime64_any_dtype(table["when"])
    assert table["when"].iloc[0] == pd.Timestamp("2024-01-05") and pd.isna(table["when"].iloc[1])
    assert item["truncated"] is False and item["total_rows"] == 2 and item["warnings"] == []


def test_get_history_item_keeps_text_columns_that_are_not_dates(world):
    payload = preview(rows=[["2024-01-05", "x"], ["not a date", "y"]], columns=["a", "b"], chart=None)
    table = analyst.get_history_item(
        world.alice, add_history(world.alice, world.dataset_id, payload=payload)
    )["result"]
    assert table["a"].tolist() == ["2024-01-05", "not a date"]


def test_get_history_item_reports_truncation(world):
    payload = preview(rows=[["a", 1.0, None]], truncated=True, total_rows=900)
    item = analyst.get_history_item(world.alice, add_history(world.alice, world.dataset_id, payload=payload))
    assert item["truncated"] is True and item["total_rows"] == 900 and len(item["result"]) == 1


def test_get_history_item_survives_damaged_or_missing_results(world):
    history_id = add_history(world.alice, world.dataset_id, status="error")
    with session_scope() as db:
        db.get(AnalysisHistory, history_id).result_preview_json = "{not json"
        db.get(AnalysisHistory, history_id).plan_json = None
    item = analyst.get_history_item(world.alice, history_id)
    assert item["result"].empty and item["plan"] is None and item["chart"] is None
    assert item["status"] == "error" and item["error_message"] == "It failed."


def test_get_history_item_visibility(world, outsider):
    other_admin, other_analyst, _ = outsider
    history_id = add_history(world.alice, world.dataset_id)
    assert analyst.get_history_item(world.alice, history_id)["history_id"] == history_id
    assert analyst.get_history_item(world.admin, history_id)["history_id"] == history_id
    for stranger in (world.bob, other_admin, other_analyst):
        with pytest.raises(NotFoundError):
            analyst.get_history_item(stranger, history_id)
    with pytest.raises(NotFoundError):
        analyst.get_history_item(world.alice, history_id + 999)
    with pytest.raises(ValidationError):
        analyst.get_history_item(world.alice, 0)


def test_revoking_dataset_access_hides_own_history(world):
    """SEC2: list, open and export of an analyst's own history need current dataset access."""
    history_id = add_history(world.alice, world.dataset_id)
    world.revoke(world.alice)
    assert analyst.list_my_history(world.alice) == []
    with pytest.raises(NotFoundError):
        analyst.get_history_item(world.alice, history_id)
    with pytest.raises(NotFoundError):
        analyst.export_history_result(world.alice, history_id, file_format="csv")
    assert analyst.get_history_item(world.admin, history_id)["history_id"] == history_id
    assert analyst.export_history_result(world.admin, history_id)[1] == f"analysis_{history_id}.csv"
    world.factory.grant(world.dataset_id, world.alice.user_id)
    assert [h["history_id"] for h in analyst.list_my_history(world.alice)] == [history_id]


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


def test_save_report_copies_the_analysis(world):
    history_id = add_history(world.alice, world.dataset_id)
    report = analyst.save_report(
        world.alice, history_id=history_id, title="  Q1 revenue  ", notes=" check later "
    )
    assert report["title"] == "Q1 revenue" and report["notes"] == "check later"
    assert report["question"] == "Revenue by region" and report["dataset_name"] == "Sales 2024"
    assert report["history_id"] == history_id and report["created_by_id"] == world.alice.user_id
    assert report["is_owner"] is True and report["created_by"] == world.alice.username
    full = analyst.get_report(world.alice, report["report_id"])
    assert full["answer_text"] == "North leads." and full["plan"] == PLAN and full["chart"] == CHART
    assert list(full["result"].columns) == ["region", "revenue", "when"] and len(full["result"]) == 2
    assert full["facts"] == [{"label": "Total", "value": "5,540.5"}]
    entry = audit_actions("report.save")[-1]
    assert entry.actor_user_id == world.alice.user_id and entry.target_id == str(report["report_id"])
    assert entry.organization_id == world.org_id


def test_save_report_bounds_the_stored_rows(world, app_settings):
    set_settings(app_settings.with_overrides(result_max_rows=3))
    rows = [[f"r{i}", float(i), None] for i in range(10)]
    history_id = add_history(world.alice, world.dataset_id, payload=preview(rows=rows, total_rows=10))
    report = analyst.save_report(world.alice, history_id=history_id, title="Big")
    full = analyst.get_report(world.alice, report["report_id"])
    assert len(full["result"]) == 3 and full["truncated"] is True and full["total_rows"] == 10


def test_save_report_accepts_a_result_without_chart(world):
    history_id = add_history(world.alice, world.dataset_id, payload=preview(chart=None))
    report = analyst.save_report(world.alice, history_id=history_id, title="No chart")
    assert analyst.get_report(world.alice, report["report_id"])["chart"] is None


def test_org_admin_can_save_any_organization_analysis(world):
    history_id = add_history(world.alice, world.dataset_id)
    report = analyst.save_report(world.admin, history_id=history_id, title="By admin")
    assert report["created_by_id"] == world.admin.user_id


@pytest.mark.parametrize("title", ["", "   ", None, 5, "t" * 201])
def test_save_report_validates_the_title(world, title):
    history_id = add_history(world.alice, world.dataset_id)
    with pytest.raises(ValidationError):
        analyst.save_report(world.alice, history_id=history_id, title=title)


def test_save_report_title_and_notes_boundaries(world):
    history_id = add_history(world.alice, world.dataset_id)
    assert len(analyst.save_report(world.alice, history_id=history_id, title="t" * 200)["title"]) == 200
    assert (
        analyst.save_report(world.alice, history_id=history_id, title="x", notes="n" * 5000)["notes"]
        == "n" * 5000
    )
    with pytest.raises(ValidationError):
        analyst.save_report(world.alice, history_id=history_id, title="x", notes="n" * 5001)
    with pytest.raises(ValidationError):
        analyst.save_report(world.alice, history_id=history_id, title="x", notes=7)


@pytest.mark.parametrize("status", ["error", "rejected", "clarification"])
def test_save_report_requires_a_successful_analysis(world, status):
    history_id = add_history(world.alice, world.dataset_id, status=status)
    with pytest.raises(ValidationError, match="successful"):
        analyst.save_report(world.alice, history_id=history_id, title="x")


def test_save_report_needs_the_permission(world, factory):
    limited = factory.analyst(world.org_id, permissions={Permission.EXPORT_DATA.value})
    factory.grant(world.dataset_id, limited.user_id)
    history_id = add_history(limited, world.dataset_id)
    with pytest.raises(PermissionDeniedError):
        analyst.save_report(limited, history_id=history_id, title="x")


def test_save_report_needs_the_premium_feature(factory, sales_frame):
    basic = factory.organization("Basic Co", plan="basic")
    admin = factory.org_admin(basic)
    member = factory.analyst(basic)
    dataset_id = factory.dataset(admin, sales_frame, "Basic data")
    factory.grant(dataset_id, member.user_id)
    history_id = add_history(member, dataset_id)
    with pytest.raises(FeatureNotAvailableError):
        analyst.save_report(member, history_id=history_id, title="x")
    with pytest.raises(FeatureNotAvailableError):
        analyst.save_report(admin, history_id=add_history(admin, dataset_id), title="x")
    with pytest.raises(FeatureNotAvailableError):
        analyst.list_reports(admin)


def test_save_report_cannot_use_other_peoples_or_other_tenants_analyses(world, outsider):
    other_admin, other_analyst, _ = outsider
    history_id = add_history(world.alice, world.dataset_id)
    for stranger in (world.bob, other_admin, other_analyst):
        with pytest.raises(NotFoundError):
            analyst.save_report(stranger, history_id=history_id, title="x")
    with pytest.raises(NotFoundError):
        analyst.save_report(world.alice, history_id=history_id + 500, title="x")


def test_save_report_requires_current_dataset_access_for_analysts(world):
    history_id = add_history(world.alice, world.dataset_id)
    world.revoke(world.alice)
    with pytest.raises(NotFoundError):
        analyst.save_report(world.alice, history_id=history_id, title="x")


# ---------------------------------------------------------------------------
# list_reports / get_report
# ---------------------------------------------------------------------------


def save(ctx, dataset_id, title="Report", **kwargs) -> int:
    return analyst.save_report(ctx, history_id=add_history(ctx, dataset_id, **kwargs), title=title)[
        "report_id"
    ]


def test_org_admin_sees_every_report_of_the_organization(world, outsider):
    other_admin, _, _ = outsider
    a = save(world.alice, world.dataset_id, "Alice")
    b = save(world.bob, world.dataset_id, "Bob")
    assert {r["report_id"] for r in analyst.list_reports(world.admin)} == {a, b}
    assert analyst.list_reports(other_admin) == []


def test_analysts_see_own_reports_plus_reports_on_accessible_datasets(world):
    a = save(world.alice, world.dataset_id, "Alice")
    b = save(world.bob, world.dataset_id, "Bob")
    shared = analyst.list_reports(world.alice)
    assert {r["report_id"] for r in shared} == {a, b}
    assert {r["report_id"]: r["is_owner"] for r in shared} == {a: True, b: False}
    assert {r["created_by"] for r in shared} == {world.alice.username, world.bob.username}


def test_revoking_dataset_access_hides_every_report_on_it_including_own(world):
    """SEC2 (lead decision): reports follow current dataset access for everyone but Org Admins."""
    a = save(world.alice, world.dataset_id, "Alice")
    b = save(world.bob, world.dataset_id, "Bob")
    world.revoke(world.alice)
    assert analyst.list_reports(world.alice) == []
    for report_id in (a, b):
        with pytest.raises(NotFoundError):
            analyst.get_report(world.alice, report_id)
        with pytest.raises(NotFoundError):
            analyst.export_report(world.alice, report_id, file_format="csv")
        with pytest.raises(NotFoundError):
            analyst.update_report(world.alice, report_id, title="x")
        with pytest.raises(NotFoundError):
            analyst.delete_report(world.alice, report_id)
    assert {r["report_id"] for r in analyst.list_reports(world.admin)} == {a, b}
    assert analyst.get_report(world.admin, a)["title"] == "Alice"
    assert analyst.export_report(world.admin, a, file_format="csv")[1].endswith(".csv")
    assert analyst.update_report(world.admin, a, title="Kept")["title"] == "Kept"
    world.factory.grant(world.dataset_id, world.alice.user_id)
    assert {r["report_id"] for r in analyst.list_reports(world.alice)} == {a, b}


def test_archiving_the_dataset_hides_its_reports_and_history_from_analysts(world):
    history_id = add_history(world.alice, world.dataset_id)
    report_id = analyst.save_report(world.alice, history_id=history_id, title="Mine")["report_id"]
    dataset_service.set_dataset_status(world.admin, world.dataset_id, archived=True)
    assert analyst.list_reports(world.alice) == [] and analyst.list_my_history(world.alice) == []
    with pytest.raises(NotFoundError):
        analyst.get_report(world.alice, report_id)
    with pytest.raises(NotFoundError):
        analyst.get_history_item(world.alice, history_id)
    with pytest.raises(NotFoundError):
        analyst.export_history_result(world.alice, history_id)
    assert [r["report_id"] for r in analyst.list_reports(world.admin)] == [report_id]
    assert analyst.get_history_item(world.admin, history_id)["history_id"] == history_id


def test_list_reports_orders_newest_first_and_filters_by_dataset(world, factory, sales_frame):
    second_dataset = factory.dataset(world.admin, sales_frame, "Second")
    factory.grant(second_dataset, world.alice.user_id)
    first = save(world.alice, world.dataset_id, "one")
    other = save(world.alice, second_dataset, "two")
    third = save(world.alice, world.dataset_id, "three")
    assert [r["report_id"] for r in analyst.list_reports(world.alice)] == [third, other, first]
    assert [r["report_id"] for r in analyst.list_reports(world.alice, dataset_id=world.dataset_id)] == [
        third,
        first,
    ]
    assert [r["report_id"] for r in analyst.list_reports(world.alice, limit=1)] == [third]
    with pytest.raises(ValidationError):
        analyst.list_reports(world.alice, limit=0)
    with pytest.raises(ValidationError):
        analyst.list_reports(world.alice, limit=1001)


def test_list_reports_dataset_filter_rejects_foreign_and_ungranted_datasets(
    world, factory, sales_frame, outsider
):
    other_admin, _, _ = outsider
    foreign = factory.dataset(other_admin, sales_frame, "Foreign")
    ungranted = factory.dataset(world.admin, sales_frame, "Ungranted")
    for dataset_id in (foreign, ungranted):
        with pytest.raises(NotFoundError):
            analyst.list_reports(world.alice, dataset_id=dataset_id)
    with pytest.raises(NotFoundError):
        analyst.list_reports(world.admin, dataset_id=foreign)


def test_reports_without_a_dataset_link_stay_visible_to_org_admins_only(world):
    report_id = save(world.alice, world.dataset_id)
    with session_scope() as db:
        db.get(SavedReport, report_id).dataset_id = None
    assert analyst.get_report(world.admin, report_id)["dataset_name"] is None
    for member in (world.alice, world.bob):  # no dataset means no current access (fails closed)
        with pytest.raises(NotFoundError):
            analyst.get_report(member, report_id)
    assert analyst.list_reports(world.alice) == []


def test_get_report_is_tenant_isolated(world, outsider):
    other_admin, other_analyst, _ = outsider
    report_id = save(world.alice, world.dataset_id)
    for stranger in (other_admin, other_analyst):
        with pytest.raises(NotFoundError):
            analyst.get_report(stranger, report_id)
    with pytest.raises(NotFoundError):
        analyst.get_report(world.alice, report_id + 100)
    with pytest.raises(ValidationError):
        analyst.get_report(world.alice, -3)


# ---------------------------------------------------------------------------
# update_report / delete_report
# ---------------------------------------------------------------------------


def test_owner_updates_title_and_notes(world):
    report_id = save(world.alice, world.dataset_id, "Old")
    updated = analyst.update_report(world.alice, report_id, title=" New ", notes="n")
    assert updated["title"] == "New" and updated["notes"] == "n"
    assert analyst.update_report(world.alice, report_id, notes="")["notes"] == ""
    assert analyst.get_report(world.alice, report_id)["title"] == "New"
    assert audit_actions("report.update")[0].target_id == str(report_id)


def test_update_validates_input(world):
    report_id = save(world.alice, world.dataset_id)
    with pytest.raises(ValidationError):
        analyst.update_report(world.alice, report_id)
    with pytest.raises(ValidationError):
        analyst.update_report(world.alice, report_id, title="  ")
    with pytest.raises(ValidationError):
        analyst.update_report(world.alice, report_id, title="t" * 201)
    with pytest.raises(ValidationError):
        analyst.update_report(world.alice, report_id, notes="n" * 5001)
    assert analyst.get_report(world.alice, report_id)["title"] == "Report"


def test_update_and_delete_are_limited_to_owner_and_org_admin(world):
    report_id = save(world.alice, world.dataset_id, "Mine")
    with pytest.raises(PermissionDeniedError):
        analyst.update_report(world.bob, report_id, title="Hijacked")
    with pytest.raises(PermissionDeniedError):
        analyst.delete_report(world.bob, report_id)
    assert analyst.get_report(world.alice, report_id)["title"] == "Mine"
    assert (
        analyst.update_report(world.admin, report_id, title="Edited by admin")["title"] == "Edited by admin"
    )
    assert analyst.delete_report(world.admin, report_id)["deleted"] is True


def test_update_and_delete_need_the_save_permission(world):
    report_id = save(world.alice, world.dataset_id)
    with session_scope() as db:
        db.execute(
            delete(UserPermission).where(
                UserPermission.user_id == world.alice.user_id,
                UserPermission.permission == Permission.SAVE_REPORTS.value,
            )
        )
    with pytest.raises(PermissionDeniedError):
        analyst.update_report(world.alice, report_id, title="x")
    with pytest.raises(PermissionDeniedError):
        analyst.delete_report(world.alice, report_id)


def test_delete_removes_the_report_and_keeps_the_history(world):
    history_id = add_history(world.alice, world.dataset_id)
    report_id = analyst.save_report(world.alice, history_id=history_id, title="Gone soon")["report_id"]
    result = analyst.delete_report(world.alice, report_id)
    assert result == {"report_id": report_id, "title": "Gone soon", "deleted": True}
    with pytest.raises(NotFoundError):
        analyst.get_report(world.alice, report_id)
    assert analyst.get_history_item(world.alice, history_id)["history_id"] == history_id
    assert audit_actions("report.delete")[0].target_id == str(report_id)
    with pytest.raises(NotFoundError):
        analyst.delete_report(world.alice, report_id)


def test_update_and_delete_are_tenant_isolated(world, outsider):
    other_admin, other_analyst, _ = outsider
    report_id = save(world.alice, world.dataset_id)
    for stranger in (other_admin, other_analyst):
        with pytest.raises(NotFoundError):
            analyst.update_report(stranger, report_id, title="x")
        with pytest.raises(NotFoundError):
            analyst.delete_report(stranger, report_id)
    assert analyst.get_report(world.alice, report_id)["title"] == "Report"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

EVIL_PAYLOAD = {
    "columns": ["region", "revenue", "when"],
    "rows": [['=HYPERLINK("http://evil")', 10.0, "2024-01-05"], [SCRIPT, 20.0, None]],
    "truncated": False,
    "total_rows": 2,
    "chart": CHART,
    "facts": [{"label": "Total", "value": "30"}],
    "warnings": [],
}


@pytest.fixture
def evil_report(world) -> int:
    history_id = add_history(
        world.alice,
        world.dataset_id,
        question=f"What is {SCRIPT}?",
        payload=EVIL_PAYLOAD,
        answer=f"See {SCRIPT}",
    )
    return analyst.save_report(world.alice, history_id=history_id, title="Q1: revenue/region?")["report_id"]


def test_export_report_html_is_escaped_and_offline(world, evil_report):
    content, name, mime = analyst.export_report(world.alice, evil_report)
    page = content.decode("utf-8")
    assert (name, mime) == ("Q1_revenue_region.html", "text/html")
    assert SCRIPT not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "plotly-graph-div" in page and "<script src" not in page
    assert "Sales 2024" in page and "Q1: revenue/region?" in page


def test_export_report_csv_escapes_formulas(world, evil_report):
    content, name, mime = analyst.export_report(world.alice, evil_report, file_format="csv")
    assert (name, mime) == ("Q1_revenue_region.csv", "text/csv")
    frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    assert frame["region"].iloc[0].startswith("'=HYPERLINK") and list(frame.columns) == [
        "region",
        "revenue",
        "when",
    ]
    assert frame["revenue"].tolist() == [10.0, 20.0]


def test_export_report_xlsx_escapes_formulas(world, evil_report):
    content, name, mime = analyst.export_report(world.alice, evil_report, file_format="xlsx")
    assert (
        name.endswith(".xlsx") and mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    sheet = openpyxl.load_workbook(io.BytesIO(content)).active
    assert sheet["A1"].value == "region" and sheet["A2"].value.startswith("'=HYPERLINK")
    assert sheet["B2"].value == 10.0 and sheet["C2"].value == datetime(2024, 1, 5)


def test_export_report_json_has_the_full_result(world, evil_report):
    content, name, mime = analyst.export_report(world.alice, evil_report, file_format="JSON")
    data = json.loads(content)
    assert (name, mime) == ("Q1_revenue_region.json", "application/json")
    assert data["columns"] == ["region", "revenue", "when"] and data["rows"][1] == [SCRIPT, 20.0, None]
    assert data["rows"][0][2] == "2024-01-05T00:00:00"
    assert data["question"] == f"What is {SCRIPT}?" and data["dataset"] == "Sales 2024"
    assert data["chart"] == CHART and data["facts"] == EVIL_PAYLOAD["facts"] and data["truncated"] is False


def test_export_report_rejects_unknown_formats(world, evil_report):
    for bad in ("pdf", "", None, "../x"):
        with pytest.raises(ValidationError):
            analyst.export_report(world.alice, evil_report, file_format=bad)


def test_export_report_records_audit_and_usage(world, evil_report):
    analyst.export_report(world.alice, evil_report, file_format="csv")
    entry = audit_actions("report.export")[0]
    assert entry.target_id == str(evil_report) and json.loads(entry.details_json)["format"] == "csv"
    with session_scope() as db:
        usage = db.execute(select(UsageRecord).where(UsageRecord.metric == "export")).scalars().all()
        assert len(usage) == 1 and usage[0].user_id == world.alice.user_id


def test_export_report_needs_export_permission(world, factory):
    limited = factory.analyst(world.org_id, permissions={Permission.SAVE_REPORTS.value})
    factory.grant(world.dataset_id, limited.user_id)
    report_id = save(limited, world.dataset_id)
    with pytest.raises(PermissionDeniedError):
        analyst.export_report(limited, report_id)
    with pytest.raises(PermissionDeniedError):
        analyst.export_history_result(limited, add_history(limited, world.dataset_id))
    assert analyst.get_report(limited, report_id)["report_id"] == report_id


def test_export_report_needs_the_premium_feature(factory, sales_frame):
    basic = factory.organization("Basic Co", plan="basic")
    admin = factory.org_admin(basic)
    with pytest.raises(FeatureNotAvailableError):
        analyst.export_report(admin, 1)


def test_export_report_visibility_and_tenant_isolation(world, evil_report, outsider):
    other_admin, other_analyst, _ = outsider
    assert analyst.export_report(world.bob, evil_report, file_format="csv")[1].endswith(".csv")
    assert analyst.export_report(world.admin, evil_report, file_format="csv")[1].endswith(".csv")
    for stranger in (other_admin, other_analyst):
        with pytest.raises(NotFoundError):
            analyst.export_report(stranger, evil_report)
    world.revoke(world.bob)
    with pytest.raises(NotFoundError):
        analyst.export_report(world.bob, evil_report)


def test_export_history_result_in_every_format(world):
    history_id = add_history(world.alice, world.dataset_id, question=f"Q {SCRIPT}", payload=EVIL_PAYLOAD)
    csv_bytes, csv_name, csv_mime = analyst.export_history_result(world.alice, history_id)
    assert (csv_name, csv_mime) == (f"analysis_{history_id}.csv", "text/csv")
    assert pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig")["region"].iloc[0].startswith("'=")
    xlsx = openpyxl.load_workbook(
        io.BytesIO(analyst.export_history_result(world.alice, history_id, file_format="xlsx")[0])
    )
    assert xlsx.active["A2"].value.startswith("'=")
    data = json.loads(analyst.export_history_result(world.alice, history_id, file_format="json")[0])
    assert data["rows"][1][0] == SCRIPT and data["question"] == f"Q {SCRIPT}"
    page = analyst.export_history_result(world.alice, history_id, file_format="html")[0].decode("utf-8")
    assert SCRIPT not in page and "plotly-graph-div" in page
    assert audit_actions("analysis.export")[0].target_id == str(history_id)


def test_export_history_html_notes_truncated_results(world):
    payload = preview(rows=[["a", 1.0, None]], truncated=True, total_rows=900, chart=None)
    history_id = add_history(world.alice, world.dataset_id, payload=payload)
    page = analyst.export_history_result(world.alice, history_id, file_format="html")[0].decode("utf-8")
    assert "Rows shown" in page and "1 of 900" in page
    assert (
        json.loads(analyst.export_history_result(world.alice, history_id, file_format="json")[0])[
            "total_rows"
        ]
        == 900
    )


def test_export_history_result_validations(world):
    failed = add_history(world.alice, world.dataset_id, status="error")
    with pytest.raises(ValidationError, match="successful"):
        analyst.export_history_result(world.alice, failed)
    empty = add_history(world.alice, world.dataset_id, payload={"columns": [], "rows": []})
    with pytest.raises(ValidationError, match="no table"):
        analyst.export_history_result(world.alice, empty, file_format="csv")
    with pytest.raises(ValidationError, match="no table"):
        analyst.export_history_result(world.alice, empty, file_format="xlsx")
    assert (
        b"The result table is empty."
        in analyst.export_history_result(world.alice, empty, file_format="html")[0]
    )
    with pytest.raises(ValidationError):
        analyst.export_history_result(
            world.alice, add_history(world.alice, world.dataset_id), file_format="zip"
        )


def test_export_history_result_visibility_and_tenant_isolation(world, outsider):
    other_admin, other_analyst, _ = outsider
    history_id = add_history(world.alice, world.dataset_id)
    assert analyst.export_history_result(world.admin, history_id)[1] == f"analysis_{history_id}.csv"
    for stranger in (world.bob, other_admin, other_analyst):
        with pytest.raises(NotFoundError):
            analyst.export_history_result(stranger, history_id)
    with pytest.raises(NotFoundError):
        analyst.export_history_result(world.alice, history_id + 77)


def _set_premium_feature(world, feature: Feature, enabled: bool) -> None:
    from database.models import SubscriptionPlan
    from modules import super_admin

    with session_scope() as db:
        plan_id = db.execute(select(SubscriptionPlan.id).where(SubscriptionPlan.code == "premium")).scalar_one()
    super_admin.set_plan_feature(world.factory.super_admin(), plan_id, feature.value, enabled)


def test_exports_drop_stored_charts_the_plan_no_longer_includes(world):
    """Exports follow the same plan rule as the screen (STANDARD_CHARTS, ADVANCED_ANALYTICS)."""
    history_id = add_history(world.alice, world.dataset_id)
    report_id = analyst.save_report(world.alice, history_id=history_id, title="Charted")["report_id"]
    exports = {
        "history": lambda fmt: analyst.export_history_result(world.alice, history_id, file_format=fmt)[0],
        "report": lambda fmt: analyst.export_report(world.alice, report_id, file_format=fmt)[0],
    }
    for export in exports.values():
        assert "plotly-graph-div" in export("html").decode("utf-8")
    _set_premium_feature(world, Feature.STANDARD_CHARTS, False)
    for name, export in exports.items():
        page = export("html").decode("utf-8")
        assert "plotly-graph-div" not in page, name
        assert "North" in page and "North leads." in page, name  # table and narrative are kept
        data = json.loads(export("json"))
        assert data["chart"] is None and data["rows"][0][0] == "North", name

    _set_premium_feature(world, Feature.STANDARD_CHARTS, True)
    heatmap = {**CHART, "type": "heatmap"}
    advanced_id = add_history(world.alice, world.dataset_id, payload=preview(chart=heatmap))
    assert json.loads(analyst.export_history_result(world.alice, advanced_id, file_format="json")[0])["chart"]
    _set_premium_feature(world, Feature.ADVANCED_ANALYTICS, False)
    data = json.loads(analyst.export_history_result(world.alice, advanced_id, file_format="json")[0])
    assert data["chart"] is None
    data = json.loads(analyst.export_history_result(world.alice, history_id, file_format="json")[0])
    assert data["chart"] == CHART  # standard charts are unaffected by the advanced feature


def test_export_history_result_works_without_the_premium_feature(factory, sales_frame):
    basic = factory.organization("Basic Co", plan="basic")
    admin = factory.org_admin(basic)
    dataset_id = factory.dataset(admin, sales_frame, "Basic data")
    content, name, _ = analyst.export_history_result(admin, add_history(admin, dataset_id))
    assert name.endswith(".csv") and content.startswith(b"\xef\xbb\xbfregion")
