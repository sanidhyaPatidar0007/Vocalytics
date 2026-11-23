"""Super Admin: organizations (create, edit, change status, add an Organization Admin)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.dashboard_cards import commit, form_generation, require_role, show_flash
from components.feedback import confirm, feature_unavailable, md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import OrgStatus, Role
from modules import super_admin
from services.resource_manager import human_bytes

PREFIX = "platform_orgs"

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header(
    "Organizations",
    "Create and manage the organizations that use the platform. Nothing here deletes an organization's data.",
)
require_role(ctx, Role.SUPER_ADMIN, "This page is only available to the platform Super Admin.")
show_flash()

organizations = run_action(lambda: super_admin.list_organizations(ctx), component=PREFIX) or []
plans = [
    plan
    for plan in (run_action(lambda: super_admin.list_plans(ctx), component=PREFIX) or [])
    if plan["is_active"]
]
by_id = {org["id"]: org for org in organizations}

if organizations:
    table = pd.DataFrame(
        [
            {
                "Name": org["name"],
                "Status": org["status"],
                "Plan": org["plan_name"] or "none",
                "Renewal date": org["renewal_date"],
                "Active users": org["active_users"],
                "Active datasets": org["active_datasets"],
                "Archived datasets": org["archived_datasets"],
                "Storage": human_bytes(org["storage_bytes"]),
                "Contact e-mail": org["contact_email"] or "",
                "Status reason": org["status_reason"] or "",
                "Created (UTC)": org["created_at"],
            }
            for org in organizations
        ]
    )
    show_table(table)
else:
    st.info("No organizations yet. Create the first one in the Create organization tab.")

selected_id = None
if organizations:
    selected_id = st.selectbox(
        "Organization to manage",
        list(by_id),
        format_func=lambda org_id: f"{by_id[org_id]['name']} ({by_id[org_id]['status']})",
        key=f"{PREFIX}_selected",
        help="Used by the Edit organization, Change status and Add organization admin tabs.",
    )
selected = by_id.get(selected_id)

create_tab, edit_tab, status_tab, admin_tab = st.tabs(
    ["Create organization", "Edit organization", "Change status", "Add organization admin"]
)

with create_tab:
    if not plans:
        feature_unavailable(
            "There is no active plan to assign. Create or activate a plan on the Plans and features page first."
        )
    else:
        counter = f"{PREFIX}_create_gen"
        gen = form_generation(counter)
        with st.form(f"{PREFIX}_create_form"):
            name = st.text_input("Organization name", key=f"{PREFIX}_create_name_{gen}")
            contact = st.text_input("Contact e-mail (optional)", key=f"{PREFIX}_create_contact_{gen}")
            plan_code = st.selectbox(
                "Plan",
                [plan["code"] for plan in plans],
                format_func=lambda code: next(
                    f"{p['name']} ({p['code']})" for p in plans if p["code"] == code
                ),
                key=f"{PREFIX}_create_plan_{gen}",
            )
            st.markdown("**First organization admin (optional)**")
            st.caption(
                "Leave these empty to add an admin later. If you fill one, fill all of username, e-mail and password."
            )
            admin_username = st.text_input("Admin username", key=f"{PREFIX}_create_admin_username_{gen}")
            admin_email = st.text_input("Admin e-mail", key=f"{PREFIX}_create_admin_email_{gen}")
            admin_full_name = st.text_input("Admin full name", key=f"{PREFIX}_create_admin_name_{gen}")
            admin_password = st.text_input(
                "Admin initial password", type="password", key=f"{PREFIX}_create_admin_password_{gen}"
            )
            submitted = st.form_submit_button("Create organization", type="primary")
        if submitted:
            wants_admin = bool(admin_username.strip() or admin_email.strip() or admin_password)
            commit(
                lambda: super_admin.create_organization(
                    ctx,
                    name=name,
                    contact_email=contact or None,
                    plan_code=plan_code,
                    admin_username=admin_username.strip() or None,
                    admin_email=admin_email.strip() or None,
                    admin_password=admin_password or None,
                    admin_full_name=admin_full_name,
                ),
                success=f"Organization {md_escape(name.strip())} created"
                + (f" with admin {md_escape(admin_username.strip().lower())}." if wants_admin else "."),
                reset=counter,
            )

with edit_tab:
    if selected is None:
        st.info("Create an organization first.")
    else:
        with st.form(f"{PREFIX}_edit_form_{selected['id']}"):
            new_name = st.text_input(
                "Organization name", value=selected["name"], key=f"{PREFIX}_edit_name_{selected['id']}"
            )
            new_contact = st.text_input(
                "Contact e-mail (empty clears it)",
                value=selected["contact_email"] or "",
                key=f"{PREFIX}_edit_contact_{selected['id']}",
            )
            saved = st.form_submit_button("Save changes", type="primary")
        if saved:
            commit(
                lambda: super_admin.update_organization(
                    ctx, selected["id"], name=new_name, contact_email=new_contact
                ),
                success=f"Organization {md_escape(new_name.strip())} saved.",
            )

with status_tab:
    if selected is None:
        st.info("Create an organization first.")
    else:
        st.markdown(f"Current status of **{md_escape(selected['name'])}**: `{selected['status']}`")
        if selected["status_reason"]:
            st.text(f"Reason on record: {selected['status_reason']}")
        options = [status.value for status in OrgStatus if status.value != selected["status"]]
        verbs = {
            OrgStatus.ACTIVE.value: "Activate",
            OrgStatus.SUSPENDED.value: "Suspend",
            OrgStatus.DEACTIVATED.value: "Deactivate",
        }
        new_status = st.selectbox(
            "New status",
            options,
            format_func=lambda value: verbs[value],
            key=f"{PREFIX}_status_choice_{selected['id']}",
        )
        reason = st.text_input(
            "Reason (recorded in the audit log)", key=f"{PREFIX}_status_reason_{selected['id']}_{new_status}"
        )
        needs_confirm = new_status != OrgStatus.ACTIVE.value
        if needs_confirm:
            st.info(
                "Members of this organization will be signed out and cannot sign in until it is activated again. "
                "All datasets, versions and history are kept; nothing is deleted."
            )
            confirmed = confirm(
                f"I understand that {md_escape(selected['name'])} will lose access until it is activated again",
                key=f"{PREFIX}_status_confirm_{selected['id']}_{new_status}",
            )
        else:
            st.caption("Activating restores access for every active member of this organization.")
            confirmed = True
        if st.button(
            f"{verbs[new_status]} organization",
            key=f"{PREFIX}_status_apply_{selected['id']}_{new_status}",
            type="primary",
        ):
            if not confirmed:
                st.error("Tick the confirmation box before changing the status.")
            else:
                commit(
                    lambda: super_admin.set_organization_status(
                        ctx, selected["id"], new_status, reason=reason
                    ),
                    success=f"Organization {md_escape(selected['name'])} is now {new_status}.",
                )

with admin_tab:
    if selected is None:
        st.info("Create an organization first.")
    else:
        st.caption(f"The new admin is added to {md_escape(selected['name'])}. The plan's user limit applies.")
        counter = f"{PREFIX}_admin_gen_{selected['id']}"
        gen = form_generation(counter)
        with st.form(f"{PREFIX}_admin_form_{selected['id']}"):
            username = st.text_input("Username", key=f"{PREFIX}_admin_username_{selected['id']}_{gen}")
            email = st.text_input("E-mail", key=f"{PREFIX}_admin_email_{selected['id']}_{gen}")
            full_name = st.text_input("Full name", key=f"{PREFIX}_admin_name_{selected['id']}_{gen}")
            password = st.text_input(
                "Initial password", type="password", key=f"{PREFIX}_admin_password_{selected['id']}_{gen}"
            )
            added = st.form_submit_button("Add organization admin", type="primary")
        if added:
            commit(
                lambda: super_admin.create_org_admin(
                    ctx,
                    selected["id"],
                    username=username,
                    email=email,
                    password=password,
                    full_name=full_name,
                ),
                success=(
                    f"Organization admin {md_escape(username.strip().lower())} added to "
                    f"{md_escape(selected['name'])}."
                ),
                reset=counter,
            )
