"""Saved reports: view, edit, delete and download the analyses you saved."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.chart_components import render_result
from components.feedback import confirm, feature_unavailable, md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Feature, Permission
from modules import analyst
from services import dataset_service

FLASH_KEY = "reports_flash"
ALL_DATASETS = 0
EXPORT_LABELS = {"html": "HTML report", "csv": "CSV", "xlsx": "Excel", "json": "JSON"}

ctx = require_ctx()
caps = capabilities()
page_header("Saved reports", "Analyses you saved: review them, update the notes and download them.")
if caps is None or not caps.has(Feature.SAVED_REPORTS):
    feature_unavailable("Saved reports are not included in your plan.")
    st.stop()

flash = st.session_state.pop(FLASH_KEY, None)
if flash:
    st.success(flash, icon=":material/check_circle:")

datasets = run_action(lambda: dataset_service.list_datasets(ctx), component="reports") or []
names = {d["dataset_id"]: d["name"] for d in datasets}
dataset_choice = st.selectbox(
    "Dataset",
    [ALL_DATASETS, *names],
    format_func=lambda value: "All datasets" if value == ALL_DATASETS else names[value],
    key="reports_dataset",
)
reports = (
    run_action(
        lambda: analyst.list_reports(ctx, dataset_id=None if dataset_choice == ALL_DATASETS else dataset_choice),
        component="reports",
    )
    or []
)
if not reports:
    st.info("No saved reports yet. Ask the analyst a question, then use Save as report under the answer.")
    st.stop()

show_table(
    pd.DataFrame(
        {
            "Title": [r["title"] for r in reports],
            "Dataset": [r["dataset_name"] or "" for r in reports],
            "Question": [r["question"] for r in reports],
            "Saved by": [r["created_by"] or "" for r in reports],
            "Saved (UTC)": [r["created_at"] for r in reports],
            "Updated (UTC)": [r["updated_at"] for r in reports],
        }
    ),
    height=240,
)

by_id = {r["report_id"]: r for r in reports}
report_id = st.selectbox(
    "Open a report",
    list(by_id),
    format_func=lambda value: f"{by_id[value]['title']} (#{value})",
    key=f"reports_item_{dataset_choice}",
)
report = run_action(lambda: analyst.get_report(ctx, report_id), component="reports")
if report is None:
    st.stop()

key = f"reports_{report_id}"
with st.container(border=True):
    st.subheader(md_escape(report["title"]))
    st.caption(
        md_escape(
            f"Dataset: {report['dataset_name'] or 'removed'} | Saved by {report['created_by'] or 'unknown'} "
            f"on {report['created_at']:%Y-%m-%d %H:%M} UTC | Last changed {report['updated_at']:%Y-%m-%d %H:%M} UTC"
        )
    )
    st.markdown("**Question**")
    st.text(report["question"])
    st.markdown(md_escape(report["answer_text"]))
    if report["notes"]:
        st.markdown("**Notes**")
        st.text(report["notes"])
    render_result(
        table=report["result"],
        chart_spec=report["chart"],
        facts=report["facts"],
        warnings=report["warnings"],
        key=key,
        truncated=report["truncated"],
        total_rows=report["total_rows"],
        caps=caps,
    )

st.subheader("Download")
if caps.can(Permission.EXPORT_DATA):
    for column, (file_format, label) in zip(st.columns(len(EXPORT_LABELS)), EXPORT_LABELS.items(), strict=True):
        slot = f"{key}_export_{file_format}"
        with column:
            if st.button(f"Prepare {label}", key=f"{key}_prepare_{file_format}"):
                prepared = run_action(
                    lambda f=file_format: analyst.export_report(ctx, report_id, file_format=f),
                    success="Download prepared.",
                    component="reports",
                )
                if prepared is not None:
                    st.session_state[slot] = prepared
            ready = st.session_state.get(slot)
            if ready is not None:
                data, file_name, mime = ready
                st.download_button(
                    md_escape(f"Download {file_name}"),
                    data=data,
                    file_name=file_name,
                    mime=mime,
                    key=f"{key}_download_{file_format}",
                )
else:
    feature_unavailable("Downloading reports needs the permission to export data.")

st.subheader("Edit or delete")
can_change = caps.can(Permission.SAVE_REPORTS) and (ctx.is_org_admin or report["is_owner"])
if not caps.can(Permission.SAVE_REPORTS):
    feature_unavailable("Changing reports needs the permission to save reports.")
elif not can_change:
    st.info("Only the person who saved this report or an organization admin can edit or delete it.")
else:
    with st.form(f"{key}_edit_form"):
        title = st.text_input("Title", value=report["title"], key=f"{key}_title", max_chars=200)
        notes = st.text_area("Notes", value=report["notes"], key=f"{key}_notes", max_chars=5000)
        saved = st.form_submit_button("Save changes")
    if saved:
        updated = run_action(
            lambda: analyst.update_report(ctx, report_id, title=title, notes=notes), component="reports"
        )
        if updated is not None:
            for file_format in EXPORT_LABELS:
                st.session_state.pop(f"{key}_export_{file_format}", None)  # prepared files hold the old text
            st.session_state[FLASH_KEY] = "Report updated."
            st.rerun()
    with st.expander("Delete this report"):
        st.write("Deleting removes the saved report. The analysis stays in your history.")
        sure = confirm("I understand that this deletes the report", key=f"{key}_confirm_delete")
        if st.button("Delete report", key=f"{key}_delete"):
            if not sure:
                st.error("Tick the confirmation box to delete this report.")
            else:
                deleted = run_action(lambda: analyst.delete_report(ctx, report_id), component="reports")
                if deleted is not None:
                    st.session_state[FLASH_KEY] = "Report deleted."
                    st.rerun()
