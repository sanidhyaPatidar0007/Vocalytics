"""Server-side session handling for the Streamlit UI.

The session token lives only in ``st.session_state`` (server memory for this
browser tab); it is never rendered or sent to the browser. ``app.py`` calls
``refresh_context`` on every rerun, which validates the token against the
database, so expiry, logout, deactivation and suspension take effect on the
next interaction. Pages call ``require_ctx`` to get the validated identity.
"""

from __future__ import annotations

import streamlit as st

from auth import authentication
from auth.authorization import AuthContext
from core.errors import AppError, AuthenticationError, PermissionDeniedError, SessionExpiredError

TOKEN_KEY = "_vdas_token"  # noqa: S105 - name of a session-state key, not a secret
CTX_KEY = "_vdas_ctx"
NOTICE_KEY = "_vdas_notice"


def _user_agent() -> str | None:
    try:
        headers = st.context.headers
        return headers.get("User-Agent") if headers else None
    except Exception:  # noqa: BLE001 - headers are unavailable in some runtimes (tests)
        return None


def refresh_context() -> AuthContext | None:
    """Validate the stored token and cache the fresh identity for this run."""
    token = st.session_state.get(TOKEN_KEY)
    if not token:
        st.session_state.pop(CTX_KEY, None)
        return None
    try:
        ctx = authentication.validate_session(token)
    except (SessionExpiredError, AuthenticationError, PermissionDeniedError) as exc:
        _reset_state()
        st.session_state[NOTICE_KEY] = exc.user_message
        return None
    st.session_state[CTX_KEY] = ctx
    return ctx


def sign_in(username: str, password: str) -> AuthContext:
    """Authenticate, then start from a clean session state (no data from a previous user)."""
    result = authentication.login(username, password, user_agent=_user_agent())
    _reset_state()
    st.session_state[TOKEN_KEY] = result.token
    st.session_state[CTX_KEY] = result.ctx
    if result.must_change_password:
        st.session_state[NOTICE_KEY] = "Please change your temporary password on the Account page."
    return result.ctx


def sign_out() -> None:
    token = st.session_state.get(TOKEN_KEY)
    if token:
        try:
            authentication.logout(token)
        except AppError:
            pass
    _reset_state()
    st.session_state[NOTICE_KEY] = "You have been signed out."


def _reset_state() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def current_ctx() -> AuthContext | None:
    return st.session_state.get(CTX_KEY)


def require_ctx() -> AuthContext:
    """Return the validated identity or stop the page (defense in depth)."""
    ctx = current_ctx()
    if ctx is None:
        st.warning("Please sign in to continue.")
        st.stop()
    return ctx


def pop_notice() -> str | None:
    return st.session_state.pop(NOTICE_KEY, None)


def page_state(page: str, key: str, default=None):  # noqa: ANN001, ANN201 - generic UI state helper
    """Per-page state stored under a namespaced key."""
    full = f"_vdas_page_{page}_{key}"
    if full not in st.session_state:
        st.session_state[full] = default
    return st.session_state[full]


def set_page_state(page: str, key: str, value) -> None:  # noqa: ANN001
    st.session_state[f"_vdas_page_{page}_{key}"] = value


def clear_page_state(page: str) -> None:
    prefix = f"_vdas_page_{page}_"
    for key in [k for k in st.session_state.keys() if str(k).startswith(prefix)]:
        del st.session_state[key]


__all__ = [
    "TOKEN_KEY",
    "clear_page_state",
    "current_ctx",
    "page_state",
    "pop_notice",
    "refresh_context",
    "require_ctx",
    "set_page_state",
    "sign_in",
    "sign_out",
]
