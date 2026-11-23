"""Analyst home: your datasets, recent questions and the next steps."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.feedback import feature_unavailable, md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Feature
from modules import analyst
from services import dataset_service

ctx = require_ctx()
caps = capabilities()
page_header("Home", "Your datasets, your recent questions and what to do next.")

datasets = run_action(lambda: dataset_service.list_datasets(ctx), component="analysis_home") or []
history = run_action(lambda: analyst.list_my_history(ctx, limit=10), component="analysis_home") or []

scores = [d["quality_score"] for d in datasets if d.get("quality_score") is not None]
col1, col2, col3 = st.columns(3)
col1.metric("Datasets available to you", f"{len(datasets):,}")
col2.metric("Rows across datasets", f"{sum(d['rows'] for d in datasets):,}")
col3.metric("Average quality score", f"{sum(scores) / len(scores):.0f}" if scores else "not available")

st.subheader("Your datasets")
if datasets:
    table = pd.DataFrame(
        {
            "Dataset": [d["name"] for d in datasets],
            "Version": [d["current_version"] for d in datasets],
            "Rows": [d["rows"] for d in datasets],
            "Columns": [d["columns"] for d in datasets],
            "Quality score": [d["quality_score"] for d in datasets],
            "Updated": [d["updated_at"] for d in datasets],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Quality score": st.column_config.ProgressColumn(
                "Quality score", min_value=0, max_value=100, format="%.0f"
            ),
            "Rows": st.column_config.NumberColumn("Rows", format="%d"),
        },
    )
else:
    st.info("No datasets yet. Upload one on the Datasets page, or ask an organization admin to share one with you.")

st.subheader("Recent questions")
if history:
    recent = pd.DataFrame(
        {
            "When": [h["created_at"] for h in history],
            "Dataset": [h["dataset_name"] or "" for h in history],
            "Question": [h["question"] for h in history],
            "Status": [h["status"] for h in history],
            "How it was asked": [h["input_mode"] for h in history],
        }
    )
    show_table(recent)
    st.caption("Open the Analysis history page to view a full answer, export it or save it as a report.")
else:
    st.info("You have not asked anything yet. Open Ask the analyst and try an example question.")

st.subheader("Next steps")
steps = st.container(border=True)
with steps:
    if not datasets:
        st.markdown("1. Get a dataset: upload one on the **Datasets** page or ask an organization admin for access.")
    else:
        st.markdown("1. Check **Data quality** first, so your answers rest on clean data.")
    if caps is not None and caps.has(Feature.TEXT_AI_ANALYSIS):
        st.markdown("2. Open **Ask the analyst** and ask a question in plain language, by text or by voice.")
    else:
        feature_unavailable("Asking questions is not included in your plan.")
    st.markdown("3. Open **Analysis history** to revisit, download or save any earlier answer.")
    for target, label in (("views/data/quality.py", "Data quality"), ("views/analysis/ask.py", "Ask the analyst")):
        try:
            st.page_link(target, label=md_escape(label))
        except Exception:  # noqa: BLE001, S112 - the page is not offered to this user or plan
            continue
