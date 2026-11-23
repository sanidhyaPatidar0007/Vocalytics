"""Data Analyst workspace: analysis history, saved reports and result downloads.

History rows are written by the analytics engine. This module only reads them
(an analyst sees their own, an Organization Admin sees the whole organization)
and turns successful ones into saved reports and downloads. Every function
authorizes against the database first; ids that belong to another user or
another organization raise ``NotFoundError`` so their existence is not leaked.

Results follow current dataset access: for anyone who is not an Org Admin,
history items and saved reports (their own included) are listed, opened,
exported, changed or deleted only while the dataset is visible to them (active
and, for analysts, granted; see ``_dataset_visible``). Org Admins see every
item of their organization.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from sqlalchemy import select, true
from sqlalchemy.orm import Session

from auth.authorization import AuthContext, authorize, get_dataset_for_user, visible_datasets_query
from config.settings import get_settings
from core import timeutil
from core.constants import Feature, Permission, UsageMetric
from core.errors import NotFoundError, PermissionDeniedError, ValidationError
from database.connection import session_scope
from database.models import AnalysisHistory, Dataset, SavedReport, User
from modules import subscriptions, usage_tracking
from services import chart_service
from services.audit_service import record_audit
from services.dataset_service import frame_to_csv_bytes, frame_to_xlsx_bytes
from services.frame_utils import json_safe

MAX_TITLE_CHARS = 200
MAX_NOTES_CHARS = 5000
MAX_HISTORY_LIMIT = 500
MAX_REPORT_LIMIT = 1000
EXPORT_FORMATS = ("html", "csv", "xlsx", "json")
_MIME_TYPES = {
    "html": "text/html",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "json": "application/json",
}
# Chart types that need Feature.ADVANCED_ANALYTICS (same rule as components.chart_components).
ADVANCED_CHART_TYPES = frozenset({"heatmap", "correlation"})
_ISO_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?$")


# ---------------------------------------------------------------------------
# Validation and stored-result helpers
# ---------------------------------------------------------------------------


def _check_id(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"Invalid {label}.")
    return value


def _clean_title(title: object) -> str:
    text = title.strip() if isinstance(title, str) else ""
    if not text:
        raise ValidationError("Enter a title for the report.")
    if len(text) > MAX_TITLE_CHARS:
        raise ValidationError(f"The title can have at most {MAX_TITLE_CHARS} characters.")
    return text


def _clean_notes(notes: object) -> str:
    if notes is None:
        return ""
    if not isinstance(notes, str):
        raise ValidationError("The notes must be text.")
    if len(notes) > MAX_NOTES_CHARS:
        raise ValidationError(f"The notes can have at most {MAX_NOTES_CHARS} characters.")
    return notes.strip()


def _loads(text: str | None) -> Any:
    """Parse stored JSON; damaged or missing values read as None."""
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _stored_result(text: str | None) -> dict[str, Any]:
    """Normalise a stored result into columns, rows, truncated, total_rows, facts, warnings, chart."""
    data = _loads(text)
    data = data if isinstance(data, dict) else {}
    columns = [str(c) for c in data["columns"]] if isinstance(data.get("columns"), list) else []
    rows = data["rows"] if isinstance(data.get("rows"), list) else []
    total = data.get("total_rows")
    return {
        "columns": columns,
        "rows": rows,
        "truncated": bool(data.get("truncated", False)),
        "total_rows": total if isinstance(total, int) and not isinstance(total, bool) else len(rows),
        "facts": [f for f in data.get("facts") or [] if isinstance(f, dict)],
        "warnings": [str(w) for w in data.get("warnings") or []],
        "chart": data["chart"] if isinstance(data.get("chart"), dict) else None,
    }


def _restore_datetimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn text columns that hold only ISO dates (JSON has no date type) back into datetimes."""
    for name in [c for c, dtype in frame.dtypes.items() if dtype == "object"]:
        values = frame[name].dropna()
        if len(values) and all(isinstance(v, str) and _ISO_DATETIME.match(v) for v in values):
            frame[name] = pd.to_datetime(frame[name])
    return frame


def _frame(stored: dict[str, Any]) -> pd.DataFrame:
    try:
        frame = pd.DataFrame(stored["rows"], columns=stored["columns"])
    except (ValueError, TypeError) as exc:
        raise ValidationError("The stored result could not be read.") from exc
    return _restore_datetimes(frame)


def _bounded_result_json(history_json: str | None) -> tuple[str, str | None]:
    """Result payload and chart JSON for a report, with rows capped at the configured result size."""
    stored = _stored_result(history_json)
    limit = get_settings().result_max_rows
    rows = stored["rows"][:limit]
    result = {
        "columns": stored["columns"],
        "rows": rows,
        "truncated": stored["truncated"] or len(rows) < len(stored["rows"]),
        "total_rows": stored["total_rows"],
        "facts": stored["facts"],
        "warnings": stored["warnings"],
    }
    chart = json.dumps(stored["chart"]) if stored["chart"] else None
    return json.dumps(result), chart


def _file_stem(text: str, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")[:60] or fallback


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def _dataset_visible(ctx: AuthContext, dataset_column):
    """SQL condition: the row's dataset is currently visible to the caller (always true for Org Admins).

    Uses the same rule as every other dataset read (``visible_datasets_query``:
    active dataset plus, for analysts, a current grant). Rows without a dataset
    fail closed for everyone but Org Admins.
    """
    if ctx.is_org_admin:
        return true()
    return dataset_column.in_(visible_datasets_query(ctx).with_only_columns(Dataset.id))


def _history_summary(row: AnalysisHistory, dataset_name: str | None) -> dict[str, Any]:
    return {
        "history_id": row.id,
        "question": row.question,
        "dataset_id": row.dataset_id,
        "dataset_name": dataset_name,
        "status": row.status,
        "planner": row.planner,
        "input_mode": row.input_mode,
        "created_at": row.created_at,
        "duration_ms": row.duration_ms,
    }


def list_my_history(ctx: AuthContext, *, dataset_id: int | None = None, limit: int = 50) -> list[dict]:
    """The caller's own analyses in their organization, newest first."""
    limit = _check_id(limit, "limit")
    if limit > MAX_HISTORY_LIMIT:
        raise ValidationError(f"The limit can be at most {MAX_HISTORY_LIMIT}.")
    with session_scope() as db:
        org_id = authorize(db, ctx)
        query = (
            select(AnalysisHistory, Dataset.name)
            .outerjoin(Dataset, Dataset.id == AnalysisHistory.dataset_id)
            .where(
                AnalysisHistory.organization_id == org_id,
                AnalysisHistory.user_id == ctx.user_id,
                _dataset_visible(ctx, AnalysisHistory.dataset_id),
            )
        )
        if dataset_id is not None:
            _check_id(dataset_id, "dataset")
            get_dataset_for_user(db, ctx, dataset_id, include_archived=True)
            query = query.where(AnalysisHistory.dataset_id == dataset_id)
        rows = db.execute(
            query.order_by(AnalysisHistory.created_at.desc(), AnalysisHistory.id.desc()).limit(limit)
        ).all()
        return [_history_summary(row, name) for row, name in rows]


def _history_row(db: Session, ctx: AuthContext, history_id: int) -> tuple[AnalysisHistory, str | None]:
    """An analysis the caller may read: their own on a dataset they can see, or any for Org Admins."""
    _check_id(history_id, "analysis")
    found = db.execute(
        select(AnalysisHistory, Dataset.name)
        .outerjoin(Dataset, Dataset.id == AnalysisHistory.dataset_id)
        .where(
            AnalysisHistory.id == history_id,
            AnalysisHistory.organization_id == ctx.organization_id,
            _dataset_visible(ctx, AnalysisHistory.dataset_id),
        )
    ).first()
    if found is None or not (ctx.is_org_admin or found[0].user_id == ctx.user_id):
        raise NotFoundError("Analysis not found.")
    return found[0], found[1]


def get_history_item(ctx: AuthContext, history_id: int) -> dict:
    """One analysis with its stored result table (DataFrame), facts, chart spec and plan."""
    with session_scope() as db:
        authorize(db, ctx)
        row, dataset_name = _history_row(db, ctx, history_id)
        stored = _stored_result(row.result_preview_json)
        plan = _loads(row.plan_json)
        return {
            **_history_summary(row, dataset_name),
            "version_id": row.version_id,
            "answer_text": row.answer_text or "",
            "error_message": row.error_message,
            "plan": plan if isinstance(plan, dict) else None,
            "result": _frame(stored),
            "truncated": stored["truncated"],
            "total_rows": stored["total_rows"],
            "facts": stored["facts"],
            "warnings": stored["warnings"],
            "chart": stored["chart"],
        }


# ---------------------------------------------------------------------------
# Saved reports
# ---------------------------------------------------------------------------


def _report_summary(
    row: SavedReport, dataset_name: str | None, creator: str | None, ctx: AuthContext
) -> dict[str, Any]:
    return {
        "report_id": row.id,
        "title": row.title,
        "notes": row.notes,
        "question": row.question,
        "dataset_id": row.dataset_id,
        "dataset_name": dataset_name,
        "history_id": row.analysis_id,
        "created_by_id": row.created_by_id,
        "created_by": creator,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "is_owner": row.created_by_id is not None and row.created_by_id == ctx.user_id,
    }


def _report_query():
    return (
        select(SavedReport, Dataset.name, User.username)
        .outerjoin(Dataset, Dataset.id == SavedReport.dataset_id)
        .outerjoin(User, User.id == SavedReport.created_by_id)
    )


def _report_row(db: Session, ctx: AuthContext, report_id: int) -> tuple[SavedReport, str | None, str | None]:
    """A report the caller may see: any for Org Admins, else one on a dataset they can see."""
    _check_id(report_id, "report")
    found = db.execute(
        _report_query().where(
            SavedReport.id == report_id,
            SavedReport.organization_id == ctx.organization_id,
            _dataset_visible(ctx, SavedReport.dataset_id),
        )
    ).first()
    if found is None:
        raise NotFoundError("Report not found.")
    return found[0], found[1], found[2]


def _require_owner_or_admin(ctx: AuthContext, row: SavedReport) -> None:
    if not (ctx.is_org_admin or (row.created_by_id is not None and row.created_by_id == ctx.user_id)):
        raise PermissionDeniedError(
            "Only the person who saved a report or an organization admin can change it."
        )


def save_report(ctx: AuthContext, *, history_id: int, title: str, notes: str = "") -> dict:
    """Save a successful analysis as a report (question, plan, answer, bounded result and chart)."""
    title, notes = _clean_title(title), _clean_notes(notes)
    with session_scope() as db:
        org_id = authorize(db, ctx, permission=Permission.SAVE_REPORTS, feature=Feature.SAVED_REPORTS)
        history, dataset_name = _history_row(db, ctx, history_id)
        if history.status != "success":
            raise ValidationError("Only successful analyses can be saved as reports.")
        if history.dataset_id is not None and ctx.is_analyst:
            get_dataset_for_user(db, ctx, history.dataset_id, include_archived=True)
        result_json, chart_json = _bounded_result_json(history.result_preview_json)
        report = SavedReport(
            organization_id=org_id,
            dataset_id=history.dataset_id,
            analysis_id=history.id,
            created_by_id=ctx.user_id,
            title=title,
            question=history.question,
            plan_json=history.plan_json,
            answer_text=history.answer_text or "",
            result_json=result_json,
            chart_json=chart_json,
            notes=notes,
        )
        db.add(report)
        db.flush()
        record_audit(
            db,
            ctx,
            "report.save",
            target_type="report",
            target_id=report.id,
            details={"history_id": history.id, "dataset_id": history.dataset_id, "title": title},
        )
        return _report_summary(report, dataset_name, ctx.username, ctx)


def list_reports(ctx: AuthContext, *, dataset_id: int | None = None, limit: int = 200) -> list[dict]:
    """Reports the caller may see, newest first.

    Org Admins see every report of the organization. Everyone else sees the
    reports on datasets they currently have access to (their own reports on a
    dataset they lost access to are hidden until access is granted again).
    """
    limit = _check_id(limit, "limit")
    if limit > MAX_REPORT_LIMIT:
        raise ValidationError(f"The limit can be at most {MAX_REPORT_LIMIT}.")
    with session_scope() as db:
        org_id = authorize(db, ctx, feature=Feature.SAVED_REPORTS)
        query = _report_query().where(
            SavedReport.organization_id == org_id, _dataset_visible(ctx, SavedReport.dataset_id)
        )
        if dataset_id is not None:
            _check_id(dataset_id, "dataset")
            get_dataset_for_user(db, ctx, dataset_id, include_archived=True)
            query = query.where(SavedReport.dataset_id == dataset_id)
        rows = db.execute(
            query.order_by(SavedReport.created_at.desc(), SavedReport.id.desc()).limit(limit)
        ).all()
        return [_report_summary(row, name, creator, ctx) for row, name, creator in rows]


def _report_content(row: SavedReport) -> dict[str, Any]:
    stored = _stored_result(row.result_json)
    plan = _loads(row.plan_json)
    chart = _loads(row.chart_json)
    return {
        "answer_text": row.answer_text,
        "plan": plan if isinstance(plan, dict) else None,
        "result": _frame(stored),
        "truncated": stored["truncated"],
        "total_rows": stored["total_rows"],
        "facts": stored["facts"],
        "warnings": stored["warnings"],
        "chart": chart if isinstance(chart, dict) else None,
    }


def get_report(ctx: AuthContext, report_id: int) -> dict:
    """One report with its result table (DataFrame) and chart spec."""
    with session_scope() as db:
        authorize(db, ctx, feature=Feature.SAVED_REPORTS)
        row, dataset_name, creator = _report_row(db, ctx, report_id)
        return {**_report_summary(row, dataset_name, creator, ctx), **_report_content(row)}


def update_report(
    ctx: AuthContext, report_id: int, *, title: str | None = None, notes: str | None = None
) -> dict:
    """Change a report's title and/or notes (owner or Org Admin)."""
    if title is None and notes is None:
        raise ValidationError("Provide a new title or new notes.")
    new_title = _clean_title(title) if title is not None else None
    new_notes = _clean_notes(notes) if notes is not None else None
    with session_scope() as db:
        authorize(db, ctx, permission=Permission.SAVE_REPORTS, feature=Feature.SAVED_REPORTS)
        row, dataset_name, creator = _report_row(db, ctx, report_id)
        _require_owner_or_admin(ctx, row)
        changed = []
        if new_title is not None and new_title != row.title:
            row.title = new_title
            changed.append("title")
        if new_notes is not None and new_notes != row.notes:
            row.notes = new_notes
            changed.append("notes")
        if changed:
            row.updated_at = timeutil.utcnow()
            record_audit(
                db, ctx, "report.update", target_type="report", target_id=row.id, details={"changed": changed}
            )
        db.flush()
        return _report_summary(row, dataset_name, creator, ctx)


def delete_report(ctx: AuthContext, report_id: int) -> dict:
    """Delete a report (owner or Org Admin). The analysis it was saved from stays in the history."""
    with session_scope() as db:
        authorize(db, ctx, permission=Permission.SAVE_REPORTS, feature=Feature.SAVED_REPORTS)
        row, _, _ = _report_row(db, ctx, report_id)
        _require_owner_or_admin(ctx, row)
        summary = {"report_id": row.id, "title": row.title, "deleted": True}
        db.delete(row)
        record_audit(
            db,
            ctx,
            "report.delete",
            target_type="report",
            target_id=summary["report_id"],
            details={"title": summary["title"]},
        )
        return summary


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def _plan_chart(db: Session, organization_id: int, chart: dict | None) -> dict | None:
    """The stored chart spec if the organization's current plan still includes it, else None.

    Exports follow the same rule as the screen: no chart without standard
    charts, no heatmap or correlation without advanced analytics.
    """
    if not chart:
        return None
    features = subscriptions.enabled_features(db, organization_id)
    if Feature.STANDARD_CHARTS.value not in features:
        return None
    if chart.get("type") in ADVANCED_CHART_TYPES and Feature.ADVANCED_ANALYTICS.value not in features:
        return None
    return chart


def _check_format(file_format: str) -> str:
    key = file_format.strip().lower() if isinstance(file_format, str) else ""
    if key not in EXPORT_FORMATS:
        raise ValidationError(f"The download format must be one of: {', '.join(EXPORT_FORMATS)}.")
    return key


def _render(
    file_format: str,
    stem: str,
    *,
    title: str,
    question: str,
    answer_text: str,
    dataset_name: str,
    content: dict[str, Any],
) -> tuple[bytes, str, str]:
    """Render a stored result in one of the download formats as (bytes, file name, MIME type)."""
    table: pd.DataFrame = content["result"]
    generated_at = timeutil.utcnow()
    if file_format in ("csv", "xlsx") and table.shape[1] == 0:
        raise ValidationError("This result has no table to download.")
    if file_format == "csv":
        data = frame_to_csv_bytes(table)
    elif file_format == "xlsx":
        data = frame_to_xlsx_bytes(table, sheet_name="result")
    elif file_format == "json":
        document = {
            "title": title,
            "question": question,
            "answer": answer_text,
            "dataset": dataset_name,
            "generated_at": generated_at.isoformat(),
            "columns": [str(c) for c in table.columns],
            "rows": [[json_safe(v) for v in row] for row in table.itertuples(index=False, name=None)],
            "truncated": content["truncated"],
            "total_rows": content["total_rows"],
            "facts": content["facts"],
            "warnings": content["warnings"],
            "chart": content["chart"],
        }
        data = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
    else:
        facts = list(content["facts"])
        if content["truncated"]:
            facts.append({"label": "Rows shown", "value": f"{len(table)} of {content['total_rows']}"})
        data = chart_service.report_html(
            title=title,
            question=question,
            answer_text=answer_text,
            table=table,
            chart_spec=content["chart"],
            facts=facts,
            generated_at=generated_at,
            dataset_name=dataset_name,
        ).encode("utf-8")
    return data, f"{stem}.{file_format}", _MIME_TYPES[file_format]


def export_report(ctx: AuthContext, report_id: int, *, file_format: str = "html") -> tuple[bytes, str, str]:
    """Download a saved report as html, csv, xlsx or json: (content, file name, MIME type)."""
    with session_scope() as db:
        org_id = authorize(db, ctx, permission=Permission.EXPORT_DATA, feature=Feature.SAVED_REPORTS)
        key = _check_format(file_format)
        row, dataset_name, _ = _report_row(db, ctx, report_id)
        content = _report_content(row)
        content["chart"] = _plan_chart(db, org_id, content["chart"])
        result = _render(
            key,
            _file_stem(row.title, "report"),
            title=row.title,
            question=row.question,
            answer_text=row.answer_text,
            dataset_name=dataset_name or "",
            content=content,
        )
        usage_tracking.record_usage(
            db, org_id, ctx.user_id, UsageMetric.EXPORT, details={"report_id": row.id, "format": key}
        )
        record_audit(
            db, ctx, "report.export", target_type="report", target_id=row.id, details={"format": key}
        )
        return result


def export_history_result(
    ctx: AuthContext, history_id: int, *, file_format: str = "csv"
) -> tuple[bytes, str, str]:
    """Download the stored result of a successful analysis: (content, file name, MIME type)."""
    with session_scope() as db:
        org_id = authorize(db, ctx, permission=Permission.EXPORT_DATA)
        key = _check_format(file_format)
        history, dataset_name = _history_row(db, ctx, history_id)
        if history.status != "success":
            raise ValidationError("Only successful analyses have a result to download.")
        stored = _stored_result(history.result_preview_json)
        content = {**stored, "result": _frame(stored), "chart": _plan_chart(db, org_id, stored["chart"])}
        title = history.question if len(history.question) <= 120 else history.question[:117] + "..."
        result = _render(
            key,
            f"analysis_{history.id}",
            title=title,
            question=history.question,
            answer_text=history.answer_text or "",
            dataset_name=dataset_name or "",
            content=content,
        )
        usage_tracking.record_usage(
            db, org_id, ctx.user_id, UsageMetric.EXPORT, details={"history_id": history.id, "format": key}
        )
        record_audit(
            db, ctx, "analysis.export", target_type="analysis", target_id=history.id, details={"format": key}
        )
        return result
