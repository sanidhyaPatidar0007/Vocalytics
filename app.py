"""Streamlit entry point: bootstrap, sign-in gate and role-based navigation.

Run from the project root (PowerShell):

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from components import session
from components.navigation import build_navigation, capabilities
from components.theme import apply_theme, hide_table_downloads
from config.settings import get_settings
from core.constants import Permission, Role
from core.logging_setup import configure_logging
from database import init_database
from services.audit_service import record_error

st.set_page_config(page_title="Voice Data Analyst", page_icon=":material/insights:", layout="wide")


@st.cache_resource(show_spinner=False)
def _bootstrap(database_path: str) -> bool:
    """One-time process setup per database (no tenant data is cached here)."""
    configure_logging()
    init_database(seed=True)
    return True


_bootstrap(str(get_settings().database_path))
apply_theme()
ctx = session.refresh_context()
page = build_navigation(ctx)
caps = capabilities()
if ctx is not None and ctx.role == Role.ANALYST and caps is not None and not caps.can(Permission.EXPORT_DATA):
    hide_table_downloads()
try:
    page.run()
except Exception as exc:  # noqa: BLE001 - st.stop/st.rerun are BaseException and pass through
    reference = record_error("page", exc, ctx=session.current_ctx())
    st.error(f"This page hit an unexpected problem and nothing was changed. Reference: {reference}")
