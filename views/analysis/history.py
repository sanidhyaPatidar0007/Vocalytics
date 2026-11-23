"""Analysis history: your earlier questions, their results, exports and saved reports."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.chart_components import render_result, result_actions
from components.feedback import md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from modules import analyst
from services import dataset_service

HISTORY_LIMIT = 100
ALL_DATASETS = 0
PLANNER_LABELS = {"llm": "Local AI model", "rules": "Rule-based interpreter", "builder": "Query builder"}
MODE_LABELS = {"text": "Typed", "voice": "Voice", "builder": "Query builder"}

ctx = require_ctx()
caps = capabilities()
page_header(
    "Analysis history",
    "Your earlier questions and answers. Open one to see its result, download it or save it as a report.",
)

datasets = run_action(lambda: dataset_service.list_datasets(ctx), component="history") or []
names = {d["dataset_id"]: d["name"] for d in datasets}
dataset_choice = st.selectbox(
    "Dataset",
    [ALL_DATASETS, *names],
    format_func=lambda value: "All datasets" if value == ALL_DATASETS else names[value],
    key="history_dataset",
)
items = (
    run_action(
        lambda: analyst.list_my_history(
            ctx, dataset_id=None if dataset_choice == ALL_DATASETS else dataset_choice, limit=HISTORY_LIMIT
        ),
        component="history",
    )
    or []
)

if not items:
    st.info("No analyses yet for this selection. Open Ask the analyst, ask a question and it will appear here.")
    st.stop()

st.caption(f"Showing your {len(items)} most recent analyses (times are UTC).")
show_table(
    pd.DataFrame(
        {
            "When": [i["created_at"] for i in items],
            "Dataset": [i["dataset_name"] or "" for i in items],
            "Question": [i["question"] for i in items],
            "Status": [i["status"] for i in items],
            "Planner": [PLANNER_LABELS.get(i["planner"], "none") for i in items],
            "Asked by": [MODE_LABELS.get(i["input_mode"], i["input_mode"]) for i in items],
        }
    ),
    height=280,
)

by_id = {i["history_id"]: i for i in items}
selected_id = st.selectbox(
    "Open an analysis",
    list(by_id),
    format_func=lambda value: f"#{value}: {by_id[value]['question'][:80]} ({by_id[value]['status']})",
    key=f"history_item_{dataset_choice}",
)

item = run_action(lambda: analyst.get_history_item(ctx, selected_id), component="history")
if item is None:
    st.stop()

key = f"history_{item['history_id']}"
with st.container(border=True):
    st.markdown("**Question**")
    st.text(item["question"])
    st.caption(
        md_escape(
            " | ".join(
                [
                    f"Dataset: {item['dataset_name'] or 'removed'}",
                    f"Status: {item['status']}",
                    f"Planner: {PLANNER_LABELS.get(item['planner'], 'none')}",
                    f"Asked by: {MODE_LABELS.get(item['input_mode'], item['input_mode'])}",
                    f"Took {item['duration_ms']} ms" if item["duration_ms"] is not None else "Duration not recorded",
                ]
            )
        )
    )
    if item["status"] == "success":
        st.markdown(md_escape(item["answer_text"]))
    elif item["status"] == "clarification":
        st.markdown("**The analyst needed more detail and asked:**")
        st.info(md_escape(item["answer_text"]), icon=":material/help:")
    else:
        st.error(md_escape(item["error_message"] or item["answer_text"] or "This question was not answered."))
    if item["plan"]:
        with st.expander("How this was calculated"):
            st.json(item["plan"])
    if item["status"] == "success":
        render_result(
            table=item["result"],
            chart_spec=item["chart"],
            facts=item["facts"],
            warnings=item["warnings"],
            key=key,
            truncated=item["truncated"],
            total_rows=item["total_rows"],
            caps=caps,
        )
        result_actions(ctx, caps, item["history_id"], default_title=item["question"], key=key)
