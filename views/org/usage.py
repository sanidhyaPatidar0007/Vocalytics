"""Organization Admin: monthly usage against plan limits and the organization's analysis history."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.dashboard_cards import require_role, usage_bar
from components.feedback import page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Role, UsageMetric
from modules import organization_admin
from services import dataset_service

PREFIX = "org_usage"
METRIC_LABELS = {
    UsageMetric.ANALYSIS_QUERY.value: "Text analyses",
    UsageMetric.VOICE_QUERY.value: "Voice analyses",
    UsageMetric.UPLOAD_BYTES.value: "Uploaded bytes",
    UsageMetric.EXPORT.value: "Exports",
    UsageMetric.CLEANING_OPERATION.value: "Cleaning operations",
    UsageMetric.LLM_CALL.value: "Language model calls",
    UsageMetric.ANOMALY_CHECK.value: "Anomaly checks",
}

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header(
    "Usage and history", "What your organization used each month, and every analysis run by your team."
)
require_role(ctx, Role.ORG_ADMIN, "This page is only available to Organization Admins.")

usage_tab, history_tab = st.tabs(["Monthly usage", "Analysis history"])

with usage_tab:
    months = st.selectbox("Months to show", [3, 6, 12, 24], index=1, key=f"{PREFIX}_months")
    report = run_action(lambda: organization_admin.usage_report(ctx, months=months), component=PREFIX)
    if report:
        latest = report[-1]
        limits = latest["limits"]
        analyses = latest["totals"].get(UsageMetric.ANALYSIS_QUERY.value, 0.0)
        with st.container(border=True):
            st.markdown(f"**Current month ({latest['month']})**")
            usage_bar("Analyses", analyses, limits["max_monthly_analyses"], "analyses")
        metrics = sorted(report[0]["totals"])
        table = pd.DataFrame(
            [
                {"Month": row["month"], **{METRIC_LABELS.get(m, m): row["totals"][m] for m in metrics}}
                for row in report
            ]
        )
        show_table(table)
        metric = st.selectbox(
            "Metric to chart",
            metrics,
            format_func=lambda key: METRIC_LABELS.get(key, key),
            key=f"{PREFIX}_metric",
        )
        figure = go.Figure(
            go.Bar(
                x=[row["month"] for row in report],
                y=[row["totals"][metric] for row in report],
                name=METRIC_LABELS.get(metric, metric),
            )
        )
        figure.update_layout(
            title=METRIC_LABELS.get(metric, metric),
            xaxis_title="Month",
            yaxis_title="Total",
            margin={"l": 10, "r": 10, "t": 50, "b": 10},
        )
        st.plotly_chart(figure, width="stretch", key=f"{PREFIX}_chart")

with history_tab:
    users = run_action(lambda: organization_admin.list_users(ctx), component=PREFIX) or []
    datasets = (
        run_action(lambda: dataset_service.list_datasets(ctx, include_archived=True), component=PREFIX) or []
    )
    user_names = {user["id"]: user["username"] for user in users}
    dataset_names = {dataset["dataset_id"]: dataset["name"] for dataset in datasets}
    col_user, col_dataset, col_limit = st.columns([2, 2, 1])
    user_filter = col_user.selectbox(
        "User",
        [None, *user_names],
        format_func=lambda uid: "All users" if uid is None else user_names[uid],
        key=f"{PREFIX}_history_user",
    )
    dataset_filter = col_dataset.selectbox(
        "Dataset",
        [None, *dataset_names],
        format_func=lambda did: "All datasets" if did is None else dataset_names[did],
        key=f"{PREFIX}_history_dataset",
    )
    limit = col_limit.number_input(
        "Rows to show", min_value=1, max_value=1000, value=200, step=50, key=f"{PREFIX}_history_limit"
    )
    entries = run_action(
        lambda: organization_admin.org_analysis_history(
            ctx, limit=int(limit), user_id=user_filter, dataset_id=dataset_filter
        ),
        component=PREFIX,
    )
    if entries is not None:
        frame = pd.DataFrame(
            [
                {
                    "When (UTC)": entry["created_at"],
                    "User": entry["username"],
                    "Dataset": entry["dataset_name"] or "",
                    "Question": entry["question"],
                    "Mode": entry["input_mode"],
                    "Status": entry["status"],
                    "Chart": entry["chart_type"] or "",
                    "Duration (ms)": entry["duration_ms"],
                }
                for entry in entries
            ]
        )
        st.caption(f"Showing {len(entries)} entries, newest first.")
        show_table(frame, empty_message="No analyses match these filters yet.")
