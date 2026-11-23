"""Shared data display helpers: schema, previews, import reports, versions.

Every dynamic value here comes from uploaded data or user input, so it is
shown through ``st.dataframe`` / ``st.text`` or passed through ``md_escape``
before any Markdown-rendering call.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from components.feedback import md_escape

READABLE_DTYPES = {
    "Int64": "whole number",
    "float64": "decimal number",
    "boolean": "true/false",
    "datetime64[ns]": "date and time",
    "string": "text",
}


def readable_dtype(dtype: str | None) -> str:
    if not dtype:
        return ""
    return f"{READABLE_DTYPES.get(str(dtype), str(dtype))} ({dtype})"


def human_bytes(size: int | float | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return value


def safe_frame(
    rows: list[dict[str, Any]] | pd.DataFrame | None, columns: list[str] | None = None
) -> pd.DataFrame:
    """A DataFrame that Arrow can always serialize (mixed object columns become text)."""
    if rows is None:
        return pd.DataFrame(columns=columns or [])
    frame = (
        rows.copy()
        if isinstance(rows, pd.DataFrame)
        else pd.DataFrame([{k: _cell(v) for k, v in r.items()} for r in rows])
    )
    if columns is not None and not frame.empty:
        frame = frame[[c for c in columns if c in frame.columns]]
    elif columns is not None:
        frame = pd.DataFrame(columns=columns)
    for name in frame.columns:
        if frame[name].dtype == object:
            kinds = {type(v) for v in frame[name].dropna().tolist()}
            if len(kinds) > 1 or (kinds and kinds != {str}):
                frame[name] = frame[name].map(
                    lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(_cell(v))
                )
    return frame


def preview_table(
    df: pd.DataFrame | None, *, height: int | None = None, empty_message: str = "No rows to show."
) -> None:
    """Show dataset rows as a table without the index (values are never rendered as Markdown)."""
    if df is None or len(df.columns) == 0:
        st.info(empty_message)
        return
    kwargs: dict[str, Any] = {"hide_index": True, "width": "stretch"}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(safe_frame(df), **kwargs)
    if df.empty:
        st.caption(empty_message)


def records_table(
    rows: list[dict[str, Any]] | None,
    *,
    columns: list[str] | None = None,
    empty_message: str = "Nothing to show.",
) -> None:
    """Show a list of service records (dicts) as a table."""
    if not rows:
        st.info(empty_message)
        return
    st.dataframe(safe_frame(rows, columns), hide_index=True, width="stretch")


def schema_table(schema: list[dict[str, Any]] | None, import_types: dict[str, Any] | None = None) -> None:
    """Columns with logical type, semantic hint (from the import report) and readable storage type."""
    if not schema:
        st.info("No schema is available for this version.")
        return
    types = import_types or {}
    rows = []
    for column in schema:
        name = column.get("name")
        info = types.get(name) or {}
        rows.append(
            {
                "Column": name,
                "Logical type": column.get("logical_type"),
                "Semantic hint": info.get("semantic") or "",
                "Stored as": readable_dtype(column.get("dtype") or info.get("pandas_dtype")),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def with_labels(labels: dict[str, Any], values: dict[str, Any] | None) -> dict[str, Any]:
    """A table record that starts with fixed label cells (row number, copies) before dataset values.

    A dataset column that has the same name as a label cell is kept, renamed to
    ``<name> (column)``, so it can never replace the label.
    """
    record = dict(labels)
    for name, value in (values or {}).items():
        key = name
        if key in record:
            key, number = f"{name} (column)", 2
            while key in record or key in (values or {}):
                key, number = f"{name} (column {number})", number + 1
        record[key] = value
    return record


def before_after_table(
    sample: list[dict[str, Any]] | None, *, row_key: str = "row", empty_message: str = "No sample rows."
) -> None:
    """Before/after sample rows from a preview (one table per side, aligned by row number)."""
    if not sample:
        st.info(empty_message)
        return
    before = []
    after = []
    for item in sample:
        number = item.get(row_key, item.get("row_index"))
        before.append(with_labels({"row": number}, item.get("before")))
        if item.get("after") is None:
            after.append({"row": number, "result": "row removed"})
        else:
            after.append(with_labels({"row": number}, item["after"]))
    left, right = st.columns(2)
    with left:
        st.caption("Before")
        st.dataframe(safe_frame(before), hide_index=True, width="stretch")
    with right:
        st.caption("After")
        st.dataframe(safe_frame(after), hide_index=True, width="stretch")


def import_report_view(report: dict[str, Any] | None) -> None:
    """Warnings, malformed rows, header changes, column type decisions and missing-value breakdown."""
    if not report:
        st.info(
            "This dataset was not imported from a file (it was derived from other data), so there is no import report."
        )
        return
    cols = st.columns(4)
    cols[0].metric("Rows imported", f"{int(report.get('rows_imported') or 0):,}")
    cols[1].metric("Columns", f"{int(report.get('columns') or 0):,}")
    cols[2].metric("Malformed rows skipped", f"{int(report.get('malformed_rows_skipped') or 0):,}")
    cols[3].metric("Blank rows skipped", f"{int(report.get('blank_rows_skipped') or 0):,}")
    details = {
        "File type": report.get("file_type"),
        "Size": human_bytes(report.get("file_size_bytes")),
        "Encoding": report.get("encoding"),
        "Delimiter": {"\t": "tab"}.get(report.get("delimiter"), report.get("delimiter")),
        "Decimal separator": report.get("decimal_separator"),
        "Thousands separator": report.get("thousands_separator") or "none",
        "Sheet": report.get("sheet_name"),
        "Dates read day first": "yes" if report.get("dayfirst") else "no",
        "Missing-value placeholders": ", ".join(report.get("missing_tokens") or []) or "none",
    }
    st.dataframe(
        pd.DataFrame(
            [
                {"Setting": k, "Value": "" if v is None else str(v)}
                for k, v in details.items()
                if v is not None
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    warnings = report.get("warnings") or []
    if warnings:
        st.markdown("**Warnings**")
        for warning in warnings:
            st.warning(md_escape(warning), icon=":material/warning:")
    else:
        st.success("No import warnings.", icon=":material/check_circle:")
    malformed = report.get("malformed_row_examples") or []
    if malformed:
        st.markdown("**Malformed rows (line numbers in the file)**")
        records_table(malformed)
    ignored = int(report.get("cells_outside_header_ignored") or 0)
    if ignored:
        st.caption(f"{ignored:,} non-empty cell(s) to the right of the last header column were ignored.")
    changes = report.get("header_changes") or []
    if changes:
        st.markdown("**Header changes**")
        records_table(changes)
    column_types = report.get("column_types") or {}
    if column_types:
        st.markdown("**Column type decisions**")
        rows = []
        breakdown = []
        for name, info in column_types.items():
            rows.append(
                {
                    "Column": name,
                    "Type": info.get("logical_type"),
                    "Semantic hint": info.get("semantic"),
                    "Missing": info.get("missing"),
                    "Date format": info.get("date_format") or "",
                    "Reasons": " ".join(info.get("reasons") or []),
                }
            )
            for kind, count in (info.get("missing_breakdown") or {}).items():
                breakdown.append({"Column": name, "Missing as": kind, "Count": count})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Missing values found at import**")
        if breakdown:
            st.dataframe(pd.DataFrame(breakdown), hide_index=True, width="stretch")
        else:
            st.caption("No missing values were found at import.")


def version_badge(
    number: int | None, *, is_current: bool = False, is_original: bool = False, is_purged: bool = False
) -> str:
    """Short plain-text label for a version, for example 'v3 (current)'."""
    if number is None:
        return "no version"
    tags = []
    if is_current:
        tags.append("current")
    if is_original:
        tags.append("original import")
    if is_purged:
        tags.append("data purged")
    return f"v{number}" + (f" ({', '.join(tags)})" if tags else "")


def _score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def applied_version_view(
    result: dict[str, Any] | None, *, quality_before: float | None = None, label: str = "Saved"
) -> None:
    """Outcome of an apply: the new version number and quality before/after."""
    if not result:
        return
    number = result.get("version_number")
    if number is None:
        st.info("Nothing needed to change, so no new version was created.", icon=":material/info:")
        return
    before = result.get("quality_before", quality_before)
    after = result.get("quality_after", result.get("quality_score"))
    with st.container(border=True):
        st.success(f"{label} as version {int(number)}.", icon=":material/check_circle:")
        cols = st.columns(3)
        cols[0].metric("New version", version_badge(int(number), is_current=True))
        delta = None
        if before is not None and after is not None:
            delta = round(float(after) - float(before), 1)
        cols[1].metric("Quality before", _score(before))
        cols[2].metric("Quality after", _score(after), delta=delta)


def flash(page: str, message: str) -> None:
    """Keep a success message for the next run (used before st.rerun)."""
    st.session_state[f"_vdas_page_{page}_flash"] = message


def show_flash(page: str) -> None:
    """Show and clear a message stored with ``flash`` (messages are built from escaped text)."""
    message = st.session_state.pop(f"_vdas_page_{page}_flash", None)
    if message:
        st.success(message, icon=":material/check_circle:")


__all__ = [
    "applied_version_view",
    "flash",
    "show_flash",
    "before_after_table",
    "human_bytes",
    "import_report_view",
    "preview_table",
    "readable_dtype",
    "records_table",
    "safe_frame",
    "schema_table",
    "version_badge",
]
