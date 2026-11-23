"""Super Admin audit log and application error events."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from components.dashboard_cards import require_role
from components.feedback import page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Role
from modules import super_admin
from services import audit_service

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header(
    "Audit and errors", "Security-relevant actions across all organizations, and recent application errors."
)
require_role(ctx, Role.SUPER_ADMIN, "This page is only available to the platform Super Admin.")

organizations = run_action(lambda: super_admin.list_organizations(ctx), component="platform_audit") or []
org_names = {org["id"]: org["name"] for org in organizations}

audit_tab, errors_tab = st.tabs(["Audit log", "Error events"])

with audit_tab:
    col_org, col_prefix, col_limit = st.columns([2, 2, 1])
    org_choice = col_org.selectbox(
        "Organization",
        [None, *org_names],
        format_func=lambda value: "All organizations" if value is None else org_names[value],
        key="platform_audit_org",
    )
    prefix = col_prefix.text_input(
        "Action starts with", key="platform_audit_prefix", placeholder="platform.plan"
    )
    limit = col_limit.number_input(
        "Rows to show", min_value=1, max_value=1000, value=200, step=50, key="platform_audit_limit"
    )
    entries = run_action(
        lambda: audit_service.list_audit_logs(
            ctx, organization_id=org_choice, action_prefix=prefix.strip() or None, limit=int(limit)
        ),
        component="platform_audit",
    )
    if entries is not None:
        frame = pd.DataFrame(
            [
                {
                    "When (UTC)": entry["created_at"],
                    "Organization": org_names.get(
                        entry["organization_id"],
                        "Platform" if entry["organization_id"] is None else f"#{entry['organization_id']}",
                    ),
                    "Actor role": entry["actor_role"] or "system",
                    "Actor id": entry["actor_user_id"],
                    "Action": entry["action"],
                    "Target": f"{entry['target_type'] or ''} {entry['target_id'] or ''}".strip(),
                    "Details": json.dumps(entry["details"], default=str, sort_keys=True),
                }
                for entry in entries
            ]
        )
        st.caption(f"Showing {len(entries)} entries, newest first.")
        show_table(frame, empty_message="No audit entries match these filters.")

with errors_tab:
    error_limit = st.number_input(
        "Rows to show", min_value=1, max_value=1000, value=100, step=50, key="platform_audit_error_limit"
    )
    events = run_action(
        lambda: super_admin.list_error_events(ctx, limit=int(error_limit)), component="platform_audit"
    )
    if events is not None:
        frame = pd.DataFrame(events)
        if not frame.empty:
            frame = frame.rename(
                columns={
                    "created_at": "When (UTC)",
                    "reference": "Reference",
                    "organization_name": "Organization",
                    "component": "Component",
                    "error_type": "Type",
                    "message": "Message",
                }
            )[["When (UTC)", "Reference", "Organization", "Component", "Type", "Message"]]
        show_table(frame, empty_message="No application errors have been recorded.")
