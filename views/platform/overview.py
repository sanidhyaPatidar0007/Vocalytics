"""Super Admin overview: platform totals, storage, latency, usage and recent errors."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.dashboard_cards import metric_row, require_role
from components.feedback import page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Role
from modules import super_admin
from services.resource_manager import human_bytes

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header(
    "Platform overview",
    "Organizations, users, datasets, analysis jobs, storage, latency and errors across the platform.",
)
require_role(ctx, Role.SUPER_ADMIN, "This page is only available to the platform Super Admin.")

metrics = run_action(lambda: super_admin.platform_metrics(ctx), component="platform_overview")
if metrics is None:
    st.stop()


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


orgs = metrics["organizations"]
st.subheader("Organizations")
by_status = orgs["by_status"]
metric_row(
    [
        ("Organizations", orgs["total"], None),
        ("Active", by_status.get("active", 0), None),
        (
            "Suspended",
            by_status.get("suspended", 0),
            "Suspended organizations keep their data but cannot sign in.",
        ),
        ("Deactivated", by_status.get("deactivated", 0), None),
    ]
)

st.subheader("Users and datasets")
roles = metrics["users_by_role"]
datasets = metrics["datasets"]
metric_row(
    [
        (
            "Super admins",
            f"{roles['super_admin']['active']} of {roles['super_admin']['total']}",
            "Active of total.",
        ),
        (
            "Org admins",
            f"{roles['org_admin']['active']} of {roles['org_admin']['total']}",
            "Active of total.",
        ),
        ("Analysts", f"{roles['analyst']['active']} of {roles['analyst']['total']}", "Active of total."),
        ("Active datasets", datasets["active"], None),
        ("Archived datasets", datasets["archived"], None),
    ]
)

st.subheader("Analysis jobs")
jobs = metrics["analysis_jobs"]
metric_row(
    [
        ("Jobs, all time", jobs["total"], None),
        ("Jobs, last 30 days", jobs["last_30_days"], None),
        ("Succeeded", jobs["last_30_days_success"], "Last 30 days."),
        ("Failed", jobs["last_30_days_error"], "Last 30 days."),
        (
            "Errors, last 7 days",
            metrics["errors_last_7_days"],
            "Application errors recorded with a reference id.",
        ),
    ]
)

st.subheader("Storage")
storage = metrics["storage"]
metric_row(
    [
        (
            "Recorded in database",
            human_bytes(storage["recorded_bytes"]),
            "Original uploads plus kept versions, as recorded.",
        ),
        ("On disk", human_bytes(storage["disk_bytes"]), "Everything under the storage folder."),
    ]
)

st.subheader("Processing latency, last 30 days")
latency = metrics["latency_last_30_days"]
latency_rows = [
    {
        "Operation": label,
        "Samples": latency[key]["count"],
        "Average": _ms(latency[key]["average_ms"]),
        "95th percentile": _ms(latency[key]["p95_ms"]),
    }
    for label, key in (("Analyses", "analysis"), ("Cleaning operations", "cleaning"))
]
show_table(pd.DataFrame(latency_rows))

st.subheader("Usage by metric, last 30 days")
usage = metrics["usage_last_30_days"]
usage_frame = pd.DataFrame(
    [{"Metric": name.replace("_", " ").capitalize(), "Total": total} for name, total in sorted(usage.items())]
)
show_table(usage_frame, empty_message="No usage has been recorded in the last 30 days.")

st.subheader("Recent errors")
st.caption("Most recent application errors. Quote the reference when helping a user.")
events = run_action(lambda: super_admin.list_error_events(ctx, limit=50), component="platform_overview") or []
if events:
    error_frame = pd.DataFrame(events).rename(
        columns={
            "created_at": "When (UTC)",
            "organization_name": "Organization",
            "component": "Component",
            "error_type": "Type",
            "message": "Message",
            "reference": "Reference",
        }
    )[["When (UTC)", "Reference", "Organization", "Component", "Type", "Message"]]
    show_table(error_frame)
else:
    st.info("No application errors have been recorded.")
