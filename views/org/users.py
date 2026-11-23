"""Organization Admin: users, analyst permissions and dataset access."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.dashboard_cards import commit, form_generation, require_role, show_flash
from components.feedback import confirm, md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import ANALYST_DEFAULT_PERMISSIONS, Permission, Role
from modules import organization_admin
from services import dataset_service

PREFIX = "org_users"
PERMISSION_LABELS = {
    Permission.UPLOAD_DATASETS.value: "Upload datasets",
    Permission.CLEAN_DATA.value: "Clean, combine and restore data",
    Permission.EXPORT_DATA.value: "Export data and results",
    Permission.USE_VOICE.value: "Use voice input",
    Permission.SAVE_REPORTS.value: "Save reports",
}
PERMISSION_KEYS = [permission.value for permission in Permission]

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header(
    "Users and access",
    "Manage the people in your organization, what analysts may do and which datasets they can see.",
)
require_role(ctx, Role.ORG_ADMIN, "This page is only available to Organization Admins.")
show_flash()


def _permission_label(key: str) -> str:
    return PERMISSION_LABELS.get(key, key)


users = run_action(lambda: organization_admin.list_users(ctx), component=PREFIX) or []
users_by_id = {user["id"]: user for user in users}
role_names = {Role.ORG_ADMIN.value: "Organization admin", Role.ANALYST.value: "Analyst"}

if users:
    show_table(
        pd.DataFrame(
            [
                {
                    "Username": user["username"],
                    "Full name": user["full_name"],
                    "E-mail": user["email"],
                    "Role": role_names.get(user["role"], user["role"]),
                    "Active": "yes" if user["is_active"] else "no",
                    "Permissions": "all"
                    if user["role"] == Role.ORG_ADMIN.value
                    else ", ".join(_permission_label(p) for p in user["permissions"]) or "none",
                    "Must change password": "yes" if user["must_change_password"] else "no",
                    "Last sign-in (UTC)": user["last_login_at"],
                }
                for user in users
            ]
        )
    )

selected_id = None
if users:
    selected_id = st.selectbox(
        "User to manage",
        list(users_by_id),
        format_func=lambda uid: (
            f"{users_by_id[uid]['username']} ({role_names.get(users_by_id[uid]['role'], users_by_id[uid]['role'])}"
            + ("" if users_by_id[uid]["is_active"] else ", inactive")
            + ")"
        ),
        key=f"{PREFIX}_selected",
        help="Used by the Edit user, Status and password and Permissions tabs.",
    )
selected = users_by_id.get(selected_id)

create_tab, edit_tab, status_tab, permissions_tab, access_tab = st.tabs(
    ["Create analyst", "Edit user", "Status and password", "Permissions", "Dataset access"]
)

with create_tab:
    st.caption(
        "Give the analyst the initial password securely; they must choose their own password at first sign-in. "
        "Your plan's user limit applies."
    )
    counter = f"{PREFIX}_create_gen"
    gen = form_generation(counter)
    with st.form(f"{PREFIX}_create_form"):
        col_user, col_email = st.columns(2)
        username = col_user.text_input(
            "Username",
            key=f"{PREFIX}_create_username_{gen}",
            help="3 to 64 letters, digits, dots, underscores or hyphens.",
        )
        email = col_email.text_input("E-mail", key=f"{PREFIX}_create_email_{gen}")
        full_name = st.text_input("Full name", key=f"{PREFIX}_create_name_{gen}")
        password = st.text_input("Initial password", type="password", key=f"{PREFIX}_create_password_{gen}")
        granted = st.multiselect(
            "Permissions",
            PERMISSION_KEYS,
            default=[key for key in PERMISSION_KEYS if key in ANALYST_DEFAULT_PERMISSIONS],
            format_func=_permission_label,
            key=f"{PREFIX}_create_permissions_{gen}",
        )
        created = st.form_submit_button("Create analyst", type="primary")
    if created:
        commit(
            lambda: organization_admin.create_analyst(
                ctx,
                username=username,
                email=email,
                full_name=full_name,
                password=password,
                permissions=set(granted),
            ),
            success=(
                f"Analyst {md_escape(username.strip().lower())} created. "
                "Share the initial password with them securely."
            ),
            reset=counter,
        )

with edit_tab:
    if selected is None:
        st.info("There are no users yet.")
    else:
        uid = selected["id"]
        with st.form(f"{PREFIX}_edit_form_{uid}"):
            new_email = st.text_input("E-mail", value=selected["email"], key=f"{PREFIX}_edit_email_{uid}")
            new_name = st.text_input(
                "Full name", value=selected["full_name"], key=f"{PREFIX}_edit_name_{uid}"
            )
            saved = st.form_submit_button("Save user", type="primary")
        if saved:
            commit(
                lambda: organization_admin.update_user(ctx, uid, email=new_email, full_name=new_name),
                success=f"User {md_escape(selected['username'])} saved.",
            )

with status_tab:
    if selected is None:
        st.info("There are no users yet.")
    else:
        uid = selected["id"]
        st.markdown(
            f"**{md_escape(selected['username'])}** is currently {'active' if selected['is_active'] else 'inactive'}."
        )
        if selected["is_active"]:
            st.caption(
                "Deactivating signs the user out everywhere and blocks sign-in. "
                "Their work and history are kept. You cannot deactivate yourself."
            )
            if st.button("Deactivate user", key=f"{PREFIX}_deactivate_{uid}"):
                commit(
                    lambda: organization_admin.set_user_active(ctx, uid, False),
                    success=f"User {md_escape(selected['username'])} deactivated.",
                )
        else:
            st.caption("Reactivating lets the user sign in again. Your plan's user limit applies.")
            if st.button("Reactivate user", key=f"{PREFIX}_activate_{uid}"):
                commit(
                    lambda: organization_admin.set_user_active(ctx, uid, True),
                    success=f"User {md_escape(selected['username'])} reactivated.",
                )
        st.divider()
        st.markdown("**Reset password**")
        st.caption(
            "Sets a new temporary password, signs the user out everywhere and requires a new password at next sign-in."
        )
        counter = f"{PREFIX}_reset_gen_{uid}"
        gen = form_generation(counter)
        new_password = st.text_input(
            "New temporary password", type="password", key=f"{PREFIX}_reset_password_{uid}_{gen}"
        )
        confirmed = confirm(
            f"I understand {md_escape(selected['username'])} will be signed out and must choose a new password",
            key=f"{PREFIX}_reset_confirm_{uid}_{gen}",
        )
        if st.button("Reset password", key=f"{PREFIX}_reset_apply_{uid}", type="primary"):
            if not confirmed:
                st.error("Tick the confirmation box before resetting the password.")
            else:
                commit(
                    lambda: organization_admin.reset_user_password(ctx, uid, new_password),
                    success=(
                        f"Password reset for {md_escape(selected['username'])}. "
                        "They must change it at next sign-in."
                    ),
                    reset=counter,
                )

with permissions_tab:
    if selected is None:
        st.info("There are no users yet.")
    elif selected["role"] != Role.ANALYST.value:
        st.info("Organization admins hold every permission. Select an analyst to change permissions.")
    else:
        uid = selected["id"]
        st.caption(
            "Permissions control what the analyst may do. Plan features still decide what is available at all."
        )
        wanted = st.multiselect(
            "Permissions",
            PERMISSION_KEYS,
            default=[key for key in PERMISSION_KEYS if key in selected["permissions"]],
            format_func=_permission_label,
            key=f"{PREFIX}_permissions_{uid}",
        )
        if st.button("Save permissions", key=f"{PREFIX}_permissions_save_{uid}", type="primary"):
            commit(
                lambda: organization_admin.set_user_permissions(ctx, uid, set(wanted)),
                success=f"Permissions saved for {md_escape(selected['username'])} ({len(wanted)} granted).",
            )

with access_tab:
    st.caption(
        "Analysts only see datasets you share with them. Analysts who upload a dataset get access to it automatically."
    )
    datasets = (
        run_action(lambda: dataset_service.list_datasets(ctx, include_archived=True), component=PREFIX) or []
    )
    analysts = {user["id"]: user for user in users if user["role"] == Role.ANALYST.value}
    if not datasets:
        st.info("No datasets yet. Upload one on the Datasets page, then share it here.")
    elif not analysts:
        st.info("There are no analysts yet. Create one in the Create analyst tab.")
    else:
        by_dataset = {dataset["dataset_id"]: dataset for dataset in datasets}
        dataset_id = st.selectbox(
            "Dataset",
            list(by_dataset),
            format_func=lambda did: (
                by_dataset[did]["name"] + (" (archived)" if by_dataset[did]["status"] == "archived" else "")
            ),
            key=f"{PREFIX}_access_dataset",
        )
        holders = (
            run_action(lambda: organization_admin.list_dataset_access(ctx, dataset_id), component=PREFIX)
            or []
        )
        st.markdown("**Analysts with access**")
        show_table(
            pd.DataFrame(
                [
                    {
                        "Username": holder["username"],
                        "Full name": holder["full_name"],
                        "Active": "yes" if holder["is_active"] else "no",
                        "Granted (UTC)": holder["granted_at"],
                    }
                    for holder in holders
                ]
            ),
            empty_message="No analyst has access to this dataset yet. Organization admins always see every dataset.",
        )
        holder_names = {holder["user_id"]: holder["username"] for holder in holders}
        holder_ids = set(holder_names)
        col_grant, col_revoke = st.columns(2)
        with col_grant, st.container(border=True):
            candidates = [uid for uid in analysts if uid not in holder_ids]
            if candidates:
                grantee = st.selectbox(
                    "Analyst to give access",
                    candidates,
                    format_func=lambda uid: analysts[uid]["username"],
                    key=f"{PREFIX}_grant_user_{dataset_id}",
                )
                if st.button("Grant access", key=f"{PREFIX}_grant_{dataset_id}", type="primary"):
                    commit(
                        lambda: organization_admin.grant_dataset_access(ctx, dataset_id, grantee),
                        success=(
                            f"{md_escape(analysts[grantee]['username'])} can now open "
                            f"{md_escape(by_dataset[dataset_id]['name'])}."
                        ),
                    )
            else:
                st.info("Every analyst already has access.")
        with col_revoke, st.container(border=True):
            if holder_ids:
                revokee = st.selectbox(
                    "Analyst to remove access from",
                    sorted(holder_ids),
                    format_func=lambda uid: holder_names[uid],
                    key=f"{PREFIX}_revoke_user_{dataset_id}",
                )
                if st.button("Revoke access", key=f"{PREFIX}_revoke_{dataset_id}"):
                    commit(
                        lambda: organization_admin.revoke_dataset_access(ctx, dataset_id, revokee),
                        success=(
                            f"{md_escape(holder_names[revokee])} can no longer open "
                            f"{md_escape(by_dataset[dataset_id]['name'])}."
                        ),
                    )
            else:
                st.info("No analyst has access, so there is nothing to revoke.")
