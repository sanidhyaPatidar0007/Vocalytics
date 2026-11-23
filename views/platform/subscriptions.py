"""Super Admin: assign plans to organizations and review the subscription history."""

from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import streamlit as st

from components.dashboard_cards import commit, form_generation, require_role, show_flash
from components.feedback import feature_unavailable, md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Role
from modules import super_admin

PREFIX = "platform_subscriptions"

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header("Subscriptions", "Move organizations between plans and review every subscription they have had.")
require_role(ctx, Role.SUPER_ADMIN, "This page is only available to the platform Super Admin.")
show_flash()

organizations = run_action(lambda: super_admin.list_organizations(ctx), component=PREFIX) or []
plans = [
    plan
    for plan in (run_action(lambda: super_admin.list_plans(ctx), component=PREFIX) or [])
    if plan["is_active"]
]
orgs_by_id = {org["id"]: org for org in organizations}

assign_tab, history_tab = st.tabs(["Assign a plan", "Subscription history"])

with assign_tab:
    if not organizations:
        st.info("There are no organizations yet. Create one on the Organizations page first.")
    elif not plans:
        feature_unavailable(
            "There is no active plan to assign. Create or activate a plan on the Plans and features page."
        )
    else:
        org_id = st.selectbox(
            "Organization",
            list(orgs_by_id),
            format_func=lambda value: orgs_by_id[value]["name"],
            key=f"{PREFIX}_assign_org",
        )
        current = orgs_by_id[org_id]
        renewal = current["renewal_date"].date().isoformat() if current["renewal_date"] else "not set"
        st.caption(
            f"Current plan: {md_escape(current['plan_name'] or 'none')}. Renewal date: {md_escape(renewal)}. "
            "Assigning a plan ends the current subscription and keeps it in the history."
        )
        counter = f"{PREFIX}_assign_gen"
        gen = form_generation(counter)
        with st.form(f"{PREFIX}_assign_form"):
            plan_code = st.selectbox(
                "Plan",
                [plan["code"] for plan in plans],
                format_func=lambda code: next(
                    f"{p['name']} ({p['code']})" for p in plans if p["code"] == code
                ),
                key=f"{PREFIX}_assign_plan_{gen}",
            )
            renewal_date = st.date_input(
                "Renewal date (optional)", value=None, key=f"{PREFIX}_assign_renewal_{gen}"
            )
            auto_renew = st.checkbox(
                "Renew automatically", value=True, key=f"{PREFIX}_assign_auto_renew_{gen}"
            )
            billing_reference = st.text_input(
                "Billing reference (optional)", key=f"{PREFIX}_assign_reference_{gen}"
            )
            notes = st.text_area("Notes (optional)", key=f"{PREFIX}_assign_notes_{gen}")
            submitted = st.form_submit_button("Assign plan", type="primary")
        if submitted:
            commit(
                lambda: super_admin.assign_plan(
                    ctx,
                    org_id,
                    plan_code,
                    renewal_date=datetime.combine(renewal_date, time.min) if renewal_date else None,
                    auto_renew=auto_renew,
                    billing_reference=billing_reference or None,
                    notes=notes,
                ),
                success=f"Plan {md_escape(plan_code)} assigned to {md_escape(current['name'])}.",
                reset=counter,
            )

with history_tab:
    col_org, col_ended = st.columns([3, 1])
    org_filter = col_org.selectbox(
        "Organization",
        [None, *orgs_by_id],
        format_func=lambda value: "All organizations" if value is None else orgs_by_id[value]["name"],
        key=f"{PREFIX}_history_org",
    )
    include_ended = col_ended.checkbox(
        "Include ended subscriptions", value=True, key=f"{PREFIX}_history_ended"
    )
    rows = run_action(
        lambda: super_admin.list_subscriptions(ctx, organization_id=org_filter, include_ended=include_ended),
        component=PREFIX,
    )
    if rows is not None:
        frame = pd.DataFrame(
            [
                {
                    "Organization": row["organization_name"],
                    "Plan": f"{row['plan_name']} ({row['plan_code']})",
                    "Status": row["status"],
                    "Started (UTC)": row["started_at"],
                    "Ended (UTC)": row["ended_at"],
                    "Renewal date": row["renewal_date"],
                    "Auto renew": "yes" if row["auto_renew"] else "no",
                    "Billing reference": row["billing_reference"] or "",
                    "Assigned by": row["assigned_by"] or "system",
                    "Notes": row["notes"] or "",
                }
                for row in rows
            ]
        )
        show_table(frame, empty_message="No subscriptions match these filters.")
