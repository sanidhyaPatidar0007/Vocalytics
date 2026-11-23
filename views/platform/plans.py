"""Super Admin: plans, limits, pricing metadata and per-plan feature toggles."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.dashboard_cards import commit, form_generation, require_role, show_flash
from components.feedback import md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import ALL_FEATURES, FEATURE_LABELS, Role
from modules import super_admin

PREFIX = "platform_plans"
LIMIT_LABELS = {
    "max_users": "Maximum users",
    "max_storage_mb": "Storage limit (MB)",
    "max_upload_mb": "Upload limit per file (MB)",
    "max_datasets": "Maximum datasets",
    "max_monthly_analyses": "Analyses per month",
    "max_versions_per_dataset": "Versions kept per dataset",
}
FALLBACK_LIMITS = {
    "max_users": 5,
    "max_storage_mb": 1024,
    "max_upload_mb": 50,
    "max_datasets": 20,
    "max_monthly_analyses": 200,
    "max_versions_per_dataset": 10,
}

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header(
    "Plans and features",
    "Define plans, their limits and prices, and choose which features each plan includes.",
)
require_role(ctx, Role.SUPER_ADMIN, "This page is only available to the platform Super Admin.")
show_flash()

plans = run_action(lambda: super_admin.list_plans(ctx), component=PREFIX) or []
by_id = {plan["id"]: plan for plan in plans}


if plans:
    rows = []
    for plan in plans:
        row = {
            "Code": plan["code"],
            "Name": plan["name"],
            "Price": f"{plan['price_cents'] / 100:,.2f} {plan['currency']}",
            "Billing": plan["billing_period"],
            "Active": "yes" if plan["is_active"] else "no",
            "Organizations": plan["organizations"],
            "Features": f"{sum(plan['features'].values())} of {len(plan['features'])}",
        }
        row.update({LIMIT_LABELS[key]: plan[key] for key in LIMIT_LABELS})
        rows.append(row)
    show_table(pd.DataFrame(rows))
else:
    st.info("No plans yet. Create one in the Create plan tab.")

selected_id = None
if plans:
    selected_id = st.selectbox(
        "Plan to manage",
        list(by_id),
        format_func=lambda plan_id: (
            f"{by_id[plan_id]['name']} ({by_id[plan_id]['code']})"
            + ("" if by_id[plan_id]["is_active"] else ", inactive")
        ),
        key=f"{PREFIX}_selected",
        help="Used by the Edit plan and Features tabs.",
    )
selected = by_id.get(selected_id)

create_tab, edit_tab, features_tab = st.tabs(["Create plan", "Edit plan", "Features"])

with create_tab:
    base = plans[0] if plans else None
    defaults = {key: (base[key] if base else FALLBACK_LIMITS[key]) for key in LIMIT_LABELS}
    counter = f"{PREFIX}_create_gen"
    gen = form_generation(counter)
    with st.form(f"{PREFIX}_create_form"):
        col_code, col_name = st.columns(2)
        code = col_code.text_input(
            "Plan code",
            key=f"{PREFIX}_create_code_{gen}",
            help="Lowercase letters, digits, hyphen or underscore. Fixed once created.",
        )
        name = col_name.text_input("Plan name", key=f"{PREFIX}_create_name_{gen}")
        description = st.text_area("Description", key=f"{PREFIX}_create_description_{gen}")
        col_price, col_currency, col_period = st.columns(3)
        price = col_price.number_input(
            "Price", min_value=0.0, value=0.0, step=1.0, format="%.2f", key=f"{PREFIX}_create_price_{gen}"
        )
        currency = col_currency.text_input(
            "Currency", value="USD", max_chars=3, key=f"{PREFIX}_create_currency_{gen}"
        )
        period = col_period.selectbox(
            "Billing period", list(super_admin.BILLING_PERIODS), key=f"{PREFIX}_create_period_{gen}"
        )
        limit_cols = st.columns(3)
        limits = {
            key: limit_cols[index % 3].number_input(
                LIMIT_LABELS[key],
                min_value=1,
                value=int(defaults[key]),
                step=1,
                key=f"{PREFIX}_create_{key}_{gen}",
            )
            for index, key in enumerate(LIMIT_LABELS)
        }
        features = st.multiselect(
            "Features included from the start",
            list(ALL_FEATURES),
            format_func=lambda key: FEATURE_LABELS[key],
            key=f"{PREFIX}_create_features_{gen}",
        )
        created = st.form_submit_button("Create plan", type="primary")
    if created:
        commit(
            lambda: super_admin.create_plan(
                ctx,
                code=code.strip(),
                name=name,
                description=description,
                price_cents=round(price * 100),
                currency=currency,
                billing_period=period,
                features=set(features),
                **{key: int(value) for key, value in limits.items()},
            ),
            success=f"Plan {md_escape(name.strip())} created with {len(features)} feature(s).",
            reset=counter,
        )

with edit_tab:
    if selected is None:
        st.info("Create a plan first.")
    else:
        pid = selected["id"]
        st.caption(
            f"Changes to limits apply immediately to the {selected['organizations']} organization(s) on this plan. "
            "The plan code cannot be changed."
        )
        with st.form(f"{PREFIX}_edit_form_{pid}"):
            col_name, col_period = st.columns(2)
            new_name = col_name.text_input(
                "Plan name", value=selected["name"], key=f"{PREFIX}_edit_name_{pid}"
            )
            new_period = col_period.selectbox(
                "Billing period",
                list(super_admin.BILLING_PERIODS),
                index=list(super_admin.BILLING_PERIODS).index(selected["billing_period"]),
                key=f"{PREFIX}_edit_period_{pid}",
            )
            new_description = st.text_area(
                "Description", value=selected["description"] or "", key=f"{PREFIX}_edit_description_{pid}"
            )
            col_price, col_currency = st.columns(2)
            new_price = col_price.number_input(
                "Price",
                min_value=0.0,
                value=selected["price_cents"] / 100,
                step=1.0,
                format="%.2f",
                key=f"{PREFIX}_edit_price_{pid}",
            )
            new_currency = col_currency.text_input(
                "Currency", value=selected["currency"], max_chars=3, key=f"{PREFIX}_edit_currency_{pid}"
            )
            limit_cols = st.columns(3)
            new_limits = {
                key: limit_cols[index % 3].number_input(
                    LIMIT_LABELS[key],
                    min_value=1,
                    value=int(selected[key]),
                    step=1,
                    key=f"{PREFIX}_edit_{key}_{pid}",
                )
                for index, key in enumerate(LIMIT_LABELS)
            }
            saved = st.form_submit_button("Save plan changes", type="primary")
        if saved:
            wanted = {
                "name": new_name,
                "description": new_description,
                "price_cents": round(new_price * 100),
                "currency": new_currency,
                "billing_period": new_period,
                **{key: int(value) for key, value in new_limits.items()},
            }
            current = {key: selected[key] for key in wanted}
            changes = {key: value for key, value in wanted.items() if value != current[key]}
            commit(
                lambda: super_admin.update_plan(ctx, pid, **changes),
                success=f"Plan {md_escape(new_name.strip())} saved ({len(changes)} field(s) changed).",
            )
        st.divider()
        if selected["is_active"]:
            st.caption("Deactivating a plan stops new assignments. Organizations already on it keep working.")
            if st.button("Deactivate plan", key=f"{PREFIX}_deactivate_{pid}"):
                commit(
                    lambda: super_admin.set_plan_active(ctx, pid, False),
                    success=f"Plan {md_escape(selected['name'])} deactivated.",
                )
        else:
            st.caption("This plan is inactive and cannot be assigned to organizations.")
            if st.button("Activate plan", key=f"{PREFIX}_activate_{pid}"):
                commit(
                    lambda: super_admin.set_plan_active(ctx, pid, True),
                    success=f"Plan {md_escape(selected['name'])} activated.",
                )

with features_tab:
    if selected is None:
        st.info("Create a plan first.")
    else:
        pid = selected["id"]
        st.caption(
            f"Features included in {md_escape(selected['name'])}. A change applies to the {selected['organizations']} "
            "organization(s) on this plan on their next action."
        )
        with st.container(border=True):
            wanted_features = {
                key: st.checkbox(
                    FEATURE_LABELS[key], value=selected["features"][key], key=f"{PREFIX}_feature_{pid}_{key}"
                )
                for key in ALL_FEATURES
            }
        changed = {key: value for key, value in wanted_features.items() if value != selected["features"][key]}
        if changed:
            st.caption(f"{len(changed)} unsaved change(s).")

        def _save_features() -> dict:
            for key, value in changed.items():
                super_admin.set_plan_feature(ctx, pid, key, value)
            return {"changed": len(changed)}

        if st.button("Save features", key=f"{PREFIX}_save_features_{pid}", type="primary"):
            if not changed:
                st.info("No feature was changed.")
            else:
                commit(
                    _save_features,
                    success=f"Saved {len(changed)} feature change(s) for plan {md_escape(selected['name'])}.",
                )
