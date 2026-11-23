"""Organization dashboard: plan, users, datasets, storage and quota use, recent activity."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.dashboard_cards import feature_list, metric_row, require_role, usage_bar
from components.feedback import md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Feature, Role
from modules import organization_admin
from services.resource_manager import human_bytes

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header("Dashboard", "Your plan, people, datasets and usage at a glance.")
require_role(ctx, Role.ORG_ADMIN, "This page is only available to Organization Admins.")

data = run_action(lambda: organization_admin.org_dashboard(ctx), component="org_dashboard")
if data is None:
    st.stop()

st.subheader(md_escape(data["organization_name"]))
quality = data["average_quality_score"]
metric_row(
    [
        ("Plan", data["plan_name"], None),
        (
            "Users",
            f"{data['users']['active']} active",
            f"{data['users']['admins']} admin(s), {data['users']['analysts']} analyst(s), "
            f"{data['users']['inactive']} inactive.",
        ),
        ("Datasets", data["datasets"]["active"], f"{data['datasets']['archived']} archived."),
        (
            "Average quality",
            "n/a" if quality is None else f"{quality:.0f} of 100",
            "Mean quality score of the current version of each active dataset.",
        ),
    ]
)

left, right = st.columns(2)
with left, st.container(border=True):
    st.markdown("**Limits and usage**")
    usage_bar("Storage", data["storage_used_bytes"] / 1024**2, data["storage_quota_bytes"] / 1024**2, "MB")
    usage_bar("Analyses this month", data["analyses_this_month"], data["max_monthly_analyses"], "analyses")
    usage_bar("Active users", data["users"]["active"], data["max_users"], "users")
    usage_bar("Active datasets", data["datasets"]["active"], data["max_datasets"], "datasets")
    st.caption(
        f"Storage in use: {human_bytes(data['storage_used_bytes'])} of {human_bytes(data['storage_quota_bytes'])}."
    )
with right, st.container(border=True):
    st.markdown(f"**Features in the {md_escape(data['plan_name'])} plan**")
    feature_list(set(data["features"]))

st.subheader("Anomalies")
if Feature.ANOMALY_DETECTION.value in data["features"]:
    st.metric(f"Anomaly events, last {data['anomaly_window_days']} days", data["recent_anomaly_events"])
else:
    st.info(
        "Anomaly detection is not part of your plan, so no anomaly events are recorded.",
        icon=":material/lock:",
    )

st.subheader("Recent cleaning operations")
operations = data["recent_cleaning_operations"]
if operations:
    frame = pd.DataFrame(operations).rename(
        columns={
            "created_at": "When (UTC)",
            "dataset_name": "Dataset",
            "operation_type": "Operation",
            "status": "Status",
            "user": "User",
        }
    )[["When (UTC)", "Dataset", "Operation", "Status", "User"]]
    show_table(frame)
else:
    st.info("No cleaning operations yet. Upload a dataset on the Datasets page to get started.")
