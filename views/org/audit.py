"""Organization Admin: the organization's audit log."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from components.dashboard_cards import require_role
from components.feedback import page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Role
from modules import organization_admin
from services import audit_service

PREFIX = "org_audit"

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header("Audit log", "Security-relevant and data-changing actions in your organization, newest first.")
require_role(ctx, Role.ORG_ADMIN, "This page is only available to Organization Admins.")

users = run_action(lambda: organization_admin.list_users(ctx), component=PREFIX) or []
user_names = {user["id"]: user["username"] for user in users}

col_prefix, col_limit = st.columns([3, 1])
prefix = col_prefix.text_input(
    "Action starts with",
    key=f"{PREFIX}_prefix",
    placeholder="user.",
    help="For example user. for account changes, dataset. for sharing, org. for settings.",
)
limit = col_limit.number_input(
    "Rows to show", min_value=1, max_value=1000, value=200, step=50, key=f"{PREFIX}_limit"
)
entries = run_action(
    lambda: audit_service.list_audit_logs(ctx, action_prefix=prefix.strip() or None, limit=int(limit)),
    component=PREFIX,
)
if entries is not None:
    frame = pd.DataFrame(
        [
            {
                "When (UTC)": entry["created_at"],
                "Actor": user_names.get(
                    entry["actor_user_id"],
                    "system" if entry["actor_user_id"] is None else f"user #{entry['actor_user_id']}",
                ),
                "Action": entry["action"],
                "Target": f"{entry['target_type'] or ''} {entry['target_id'] or ''}".strip(),
                "Details": json.dumps(entry["details"], default=str, sort_keys=True),
            }
            for entry in entries
        ]
    )
    st.caption(f"Showing {len(entries)} entries.")
    show_table(frame, empty_message="No audit entries match this filter.")
