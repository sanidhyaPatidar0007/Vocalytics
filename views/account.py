"""Account: profile, password change and active sessions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from auth import authentication
from components.feedback import md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import NOTICE_KEY, pop_notice, require_ctx

ctx = require_ctx()
caps = capabilities()
page_header("Account", "Your profile, password and active sessions.")
notice = pop_notice()
if notice:
    st.info(notice)
if caps is not None and caps.must_change_password:
    st.warning(
        "Your password was set by an administrator. Choose your own password below to unlock the rest of the app.",
        icon=":material/lock_reset:",
    )

col1, col2, col3 = st.columns(3)
col1.metric("Username", md_escape(ctx.username))
col2.metric("Role", ctx.role.value.replace("_", " ").title())
col3.metric("Organization", md_escape(caps.org_name if caps and caps.org_name else "Platform"))

st.subheader("Change password")
with st.form("change_password_form", clear_on_submit=True):
    current = st.text_input("Current password", type="password", autocomplete="current-password")
    new = st.text_input("New password", type="password", autocomplete="new-password")
    repeat = st.text_input("Repeat new password", type="password", autocomplete="new-password")
    submitted = st.form_submit_button("Change password", type="primary")
if submitted:
    if new != repeat:
        st.error("The new passwords do not match.")
    else:
        changed = run_action(
            lambda: authentication.change_password(ctx, current, new) or True,
            component="account",
        )
        if changed:
            st.session_state[NOTICE_KEY] = "Password changed. Your other sessions were signed out."
            st.rerun()

st.subheader("Active sessions")
sessions = run_action(lambda: authentication.active_sessions(ctx), component="account") or []
table = pd.DataFrame(sessions)
if not table.empty:
    table = table.rename(
        columns={
            "created_at": "Signed in (UTC)",
            "last_seen_at": "Last active (UTC)",
            "expires_at": "Expires (UTC)",
            "user_agent": "Browser",
            "is_current": "This session",
        }
    ).drop(columns=["id"], errors="ignore")
show_table(table, empty_message="No active sessions.")
