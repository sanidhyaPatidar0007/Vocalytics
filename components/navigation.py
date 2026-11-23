"""Role-specific navigation and the sign-in screen.

Navigation only decides which pages are *offered*. Every page and service
still enforces role, permission and plan on the server, so hiding a page is
never the security boundary.

``PAGES`` is the single registry of page files. ``tests/ui`` asserts that every
registered file exists, so a missing page fails the test suite instead of
silently disappearing from the menu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from auth.authorization import AuthContext
from components import session
from components.feedback import md_escape, run_action
from core.constants import Feature, Permission, Role
from modules import account

CAPS_KEY = "_vdas_caps"
ROOT = Path(__file__).resolve().parent.parent

# key: (file relative to the project root, title, Material icon)
PAGES: dict[str, tuple[str, str, str]] = {
    "account": ("views/account.py", "Account", ":material/person:"),
    "platform_overview": ("views/platform/overview.py", "Overview", ":material/monitoring:"),
    "platform_orgs": ("views/platform/organizations.py", "Organizations", ":material/domain:"),
    "platform_plans": ("views/platform/plans.py", "Plans and features", ":material/tune:"),
    "platform_subscriptions": ("views/platform/subscriptions.py", "Subscriptions", ":material/receipt_long:"),
    "platform_audit": ("views/platform/audit.py", "Audit and errors", ":material/policy:"),
    "org_dashboard": ("views/org/dashboard.py", "Dashboard", ":material/space_dashboard:"),
    "org_users": ("views/org/users.py", "Users and access", ":material/group:"),
    "org_usage": ("views/org/usage.py", "Usage and history", ":material/query_stats:"),
    "org_settings": ("views/org/settings.py", "Settings", ":material/settings:"),
    "org_audit": ("views/org/audit.py", "Audit log", ":material/policy:"),
    "analyst_home": ("views/analysis/home.py", "Home", ":material/home:"),
    "datasets": ("views/data/datasets.py", "Datasets", ":material/table_view:"),
    "quality": ("views/data/quality.py", "Data quality", ":material/fact_check:"),
    "missing": ("views/data/missing_values.py", "Missing values", ":material/healing:"),
    "duplicates": ("views/data/duplicates.py", "Duplicates", ":material/content_copy:"),
    "cleaning": ("views/data/cleaning.py", "Cleaning", ":material/cleaning_services:"),
    "combine": ("views/data/combine.py", "Append and merge", ":material/merge:"),
    "versions": ("views/data/versions.py", "Versions and history", ":material/history:"),
    "ask": ("views/analysis/ask.py", "Ask the analyst", ":material/forum:"),
    "history": ("views/analysis/history.py", "Analysis history", ":material/manage_search:"),
    "reports": ("views/analysis/reports.py", "Saved reports", ":material/bookmarks:"),
    "alert_rules": ("views/alerts/rules.py", "Anomaly rules", ":material/notifications_active:"),
    "alert_events": ("views/alerts/events.py", "Anomalies and notifications", ":material/crisis_alert:"),
}


@dataclass(frozen=True)
class Capabilities:
    role: Role
    features: frozenset[str]
    permissions: frozenset[str]
    org_name: str | None
    plan_name: str | None
    must_change_password: bool = False

    def has(self, feature: Feature) -> bool:
        return feature.value in self.features

    def can(self, permission: Permission) -> bool:
        return self.role == Role.ORG_ADMIN or permission.value in self.permissions


def load_capabilities(ctx: AuthContext) -> Capabilities:
    summary = account.account_summary(ctx)
    return Capabilities(
        ctx.role,
        summary.features,
        summary.permissions,
        summary.org_name,
        summary.plan_name,
        summary.must_change_password,
    )


def capabilities() -> Capabilities | None:
    """Capabilities computed for the current run (UI hints only)."""
    return st.session_state.get(CAPS_KEY)


def page_keys_for(caps: Capabilities) -> dict[str, list[str]]:
    """Section name -> registry keys offered to this user (pure, unit-tested)."""
    if caps.must_change_password:
        # A password set by an administrator must be replaced before anything else.
        return {"Account": ["account"]}
    if caps.role == Role.SUPER_ADMIN:
        return {
            "Platform": [
                "platform_overview",
                "platform_orgs",
                "platform_plans",
                "platform_subscriptions",
                "platform_audit",
            ],
            "Account": ["account"],
        }
    is_admin = caps.role == Role.ORG_ADMIN
    sections: dict[str, list[str]] = {"Overview": ["org_dashboard" if is_admin else "analyst_home"]}
    data = ["datasets"]
    if caps.has(Feature.DATA_PROFILING):
        data.append("quality")
    if caps.can(Permission.CLEAN_DATA):
        if caps.has(Feature.MISSING_VALUES):
            data.append("missing")
        if caps.has(Feature.DUPLICATE_DETECTION):
            data.append("duplicates")
        if caps.has(Feature.STANDARD_CLEANING) or caps.has(Feature.ADVANCED_CLEANING):
            data.append("cleaning")
        if caps.has(Feature.APPEND_MERGE):
            data.append("combine")
    data.append("versions")
    sections["Data"] = data
    analysis = []
    if caps.has(Feature.TEXT_AI_ANALYSIS):
        analysis.append("ask")
    analysis.append("history")
    if caps.has(Feature.SAVED_REPORTS) and caps.can(Permission.SAVE_REPORTS):
        analysis.append("reports")
    sections["Analysis"] = analysis
    if is_admin and caps.has(Feature.ANOMALY_DETECTION):
        sections["Alerts"] = ["alert_rules", "alert_events"]
    if is_admin:
        sections["Organization"] = ["org_users", "org_usage", "org_settings", "org_audit"]
    sections["Account"] = ["account"]
    return sections


def _sign_in_page() -> None:
    st.title("Voice-Enabled Data Analyst")
    st.caption(
        "Upload business data, check its quality, clean it safely and ask questions in plain language."
    )
    notice = session.pop_notice()
    if notice:
        st.info(notice)
    left, _ = st.columns([1, 1])
    with left, st.form("sign_in_form"):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        ctx = run_action(lambda: session.sign_in(username, password), component="sign_in")
        if ctx is not None:
            st.rerun()


def build_navigation(ctx: AuthContext | None):  # noqa: ANN201 - returns st.navigation's page object
    if ctx is None:
        st.session_state.pop(CAPS_KEY, None)
        return st.navigation(
            [st.Page(_sign_in_page, title="Sign in", icon=":material/login:", url_path="sign-in")]
        )
    caps = load_capabilities(ctx)
    st.session_state[CAPS_KEY] = caps
    sections: dict[str, list[st.Page]] = {}
    first = True
    for section, keys in page_keys_for(caps).items():
        pages = []
        for key in keys:
            path, title, icon = PAGES[key]
            if not (ROOT / path).exists():
                continue  # only during development; tests/ui assert every registered page exists
            pages.append(st.Page(path, title=title, icon=icon, default=first))
            first = False
        if pages:
            sections[section] = pages
    _sidebar(ctx, caps)
    return st.navigation(sections)


_ROLE_LABELS = {
    Role.SUPER_ADMIN: "Super Admin",
    Role.ORG_ADMIN: "Organization Admin",
    Role.ANALYST: "Data Analyst",
}


def _sidebar(ctx: AuthContext, caps: Capabilities) -> None:
    with st.sidebar:
        st.markdown(f"**{md_escape(ctx.username)}**")
        st.caption(_ROLE_LABELS.get(ctx.role, ctx.role.value))
        if caps.org_name:
            st.caption(f"{md_escape(caps.org_name)} \\| {md_escape(caps.plan_name or 'no active plan')}")
        if st.button("Sign out", key="sidebar_sign_out", icon=":material/logout:"):
            session.sign_out()
            st.rerun()
