"""Small display and flow helpers shared by the administration pages.

* ``metric_row``, ``usage_bar`` and ``feature_list`` render summary cards with
  native Streamlit components. Dynamic text goes through ``md_escape``.
* ``require_role`` stops a page for the wrong role (UI hint only; every service
  still enforces the role on the server).
* ``commit`` runs a service call, then stores a one-time notice and reruns the
  page so tables and selectors show the new state together with the notice.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import streamlit as st

from auth.authorization import AuthContext
from components.feedback import feature_unavailable, md_escape, run_action
from core.constants import ALL_FEATURES, FEATURE_LABELS, Role

T = TypeVar("T")

FLASH_KEY = "_vdas_flash"


MAX_CARDS_PER_ROW = 4


def metric_row(items: list[tuple[str, object, str | None]]) -> None:
    """Render ``(label, value, help)`` tuples as ``st.metric`` cards.

    More than four cards are split into balanced rows (5 -> 3 + 2, 6 -> 3 + 3)
    so labels and values stay readable on laptop-width screens.
    """
    if not items:
        return
    rows = -(-len(items) // MAX_CARDS_PER_ROW)
    per_row = -(-len(items) // rows)
    for start in range(0, len(items), per_row):
        batch = items[start : start + per_row]
        for column, (label, value, help_text) in zip(st.columns(per_row), batch, strict=False):
            shown = md_escape(value) if isinstance(value, str) else ("" if value is None else str(value))
            column.metric(md_escape(label), shown, help=help_text)


def usage_bar(label: str, used: float, limit: float, unit: str) -> None:
    """Show ``used`` against ``limit`` as a progress bar with a caption."""
    limit_value = float(limit) if limit else 0.0
    used_value = float(used or 0)
    fraction = min(max(used_value / limit_value, 0.0), 1.0) if limit_value > 0 else 0.0
    st.progress(fraction, text=md_escape(f"{label}: {_number(used_value)} of {_number(limit_value)} {unit}"))
    if limit_value <= 0:
        st.caption("No limit is set for this item.")
    elif used_value > limit_value:
        st.caption(
            f":material/warning: Over the limit by {_number(used_value - limit_value)} {md_escape(unit)}."
        )
    else:
        st.caption(
            f"{fraction * 100:.0f}% used, {_number(limit_value - used_value)} {md_escape(unit)} remaining."
        )


def feature_list(enabled: set[str] | frozenset[str]) -> None:
    """List every plan feature, marking the enabled ones."""
    lines = []
    for key in ALL_FEATURES:
        label = md_escape(FEATURE_LABELS.get(key, key))
        if key in enabled:
            lines.append(f"- :material/check_circle: {label}")
        else:
            lines.append(f"- :material/block: {label} (not in this plan)")
    st.markdown("\n".join(lines))


def require_role(ctx: AuthContext, role: Role, message: str) -> None:
    """Stop the page with a short explanation when the signed-in role does not match."""
    if ctx.role != role:
        feature_unavailable(message)
        st.stop()


def form_generation(counter_key: str) -> int:
    """Current generation of a form; add it to widget keys so a successful submit empties the form."""
    return int(st.session_state.get(counter_key, 0))


def commit(action: Callable[[], T], *, success: str, reset: str | None = None) -> T | None:
    """Run a service call; on success keep ``success`` as a notice and rerun the page.

    ``reset`` names a counter (see ``form_generation``) that is increased on
    success, which gives the form fresh widgets. After a failure the entered
    values stay so the user can correct them.

    ``success`` is shown as Markdown, so any dynamic part must already be passed
    through ``md_escape`` by the caller. On failure nothing reruns and the safe
    error message from ``run_action`` stays visible.
    """
    result = run_action(action)
    if result is None:
        return None
    st.session_state[FLASH_KEY] = success
    if reset:
        st.session_state[reset] = form_generation(reset) + 1
    st.rerun()
    return result  # pragma: no cover - st.rerun raises


def show_flash() -> None:
    """Show (once) the notice stored by ``commit``."""
    message = st.session_state.pop(FLASH_KEY, None)
    if message:
        st.success(message, icon=":material/check_circle:")


def _number(value: float) -> str:
    return f"{value:,.0f}" if abs(value - round(value)) < 1e-9 else f"{value:,.1f}"
