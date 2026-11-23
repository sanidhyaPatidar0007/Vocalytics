"""Reusable result display for the analysis pages: charts, facts, tables and result actions.

Everything shown here comes from datasets, users or the local AI model, so
all dynamic text is passed through ``md_escape`` or shown with
``st.dataframe``. Charts are built by ``services.chart_service`` (validated
specs, escaped labels); nothing here evaluates anything.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from auth.authorization import AuthContext
from components.feedback import feature_unavailable, md_escape, run_action, show_table
from core.constants import Feature, Permission
from core.errors import AppError
from modules import analyst
from services import chart_service
from services.analytics_engine import format_number

FACTS_PER_ROW = 4
NO_CHART = "none"
# Chart types that need Feature.ADVANCED_ANALYTICS (ARCHITECTURE feature matrix: heatmaps, correlation).
ADVANCED_CHART_TYPES = frozenset({"heatmap", "correlation"})


def _is_table(table: object) -> bool:
    return isinstance(table, pd.DataFrame) and not table.empty and table.shape[1] > 0


def _allowed(spec: dict | None, advanced: bool) -> bool:
    return spec is not None and (advanced or spec.get("type") not in ADVANCED_CHART_TYPES)


def normalize_chart_spec(
    table: pd.DataFrame | None, spec: dict | None, *, advanced: bool = True
) -> dict | None:
    """A spec that is valid for the table (and allowed by the plan), or None.

    The analytics engine proposes a chart per operation; when ``chart_service``
    rejects it (for example a histogram whose x column holds range labels instead
    of numbers) a bar chart of the same columns is used, and failing that the
    automatic suggestion for the table. Valid specs are returned with defaults filled in.
    Without ``advanced`` (Feature.ADVANCED_ANALYTICS) heatmap and correlation specs
    are never returned.
    """
    if spec is None or not _is_table(table):
        return None
    try:
        valid = chart_service.validate_chart_spec(table, spec)
        if _allowed(valid, advanced):
            return valid
    except AppError:
        pass
    if spec.get("x") and spec.get("y"):
        try:
            return chart_service.validate_chart_spec(
                table, {"type": "bar", "x": spec["x"], "y": spec["y"], "color": spec.get("color")}
            )
        except AppError:
            pass
    suggestion = chart_service.suggest_chart(table)
    return suggestion if _allowed(suggestion, advanced) else None


def render_chart(table: pd.DataFrame | None, spec: dict | None, *, key: str) -> bool:
    """Show the chart for a result table. Returns True when a chart was drawn.

    Shows an information message when there is nothing to plot and a safe
    message when the chart settings are not valid for the table.
    """
    if spec is None or not _is_table(table):
        st.info("There is nothing to chart for this result.")
        return False
    outcome = run_action(lambda: (chart_service.build_figure(table, spec),), component="chart")
    if outcome is None:
        return False
    figure = outcome[0]
    if figure is None:
        st.info("There is nothing to chart for this result.")
        return False
    st.plotly_chart(figure, width="stretch", key=f"{key}_figure")
    return True


def _candidate_spec(table: pd.DataFrame, current: dict | None, chart_type: str) -> dict:
    """A validated spec of the chosen type, keeping the current axes when they still fit."""
    keep = {k: v for k, v in (current or {}).items() if k in {"x", "y", "color"} and v is not None}
    try:
        return chart_service.validate_chart_spec(table, {"type": chart_type, **keep})
    except AppError:
        return chart_service.validate_chart_spec(table, {"type": chart_type})


def chart_type_override(
    table: pd.DataFrame | None, current_spec: dict | None, *, key: str, advanced: bool = True
) -> dict | None:
    """Offer the chart types that are valid for the table and return the spec to draw.

    The returned spec is the current one until the user picks a different
    type; an invalid choice shows a safe message and keeps the current chart.
    Without ``advanced`` (Feature.ADVANCED_ANALYTICS) heatmap and correlation
    are not offered.
    """
    if not _is_table(table):
        return current_spec
    types = chart_service.available_chart_types(table)
    options = [NO_CHART, *(t for t in types if advanced or t not in ADVANCED_CHART_TYPES)]
    if len(options) == 1:
        return current_spec
    current_type = (current_spec or {}).get("type")
    index = options.index(current_type) if current_type in options else 0

    def _label(value: str) -> str:
        return "No chart" if value == NO_CHART else chart_service.humanize(value)

    choice = st.selectbox("Chart type", options, index=index, format_func=_label, key=f"{key}_type")
    if choice == NO_CHART:
        return None
    if choice == current_type and current_spec is not None:
        return current_spec
    spec = run_action(lambda: _candidate_spec(table, current_spec, choice), component="chart")
    return spec if spec is not None else current_spec


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def render_facts(facts: list[dict[str, Any]] | None) -> None:
    """Computed facts: numbers as metrics, text values (which can come from the data) in a table."""
    items = [f for f in facts or [] if isinstance(f, dict) and "label" in f]
    numbers = [f for f in items if _is_number(f.get("value"))]
    texts_ = [f for f in items if not _is_number(f.get("value"))]
    for start in range(0, len(numbers), FACTS_PER_ROW):
        batch = numbers[start : start + FACTS_PER_ROW]
        for column, fact in zip(st.columns(FACTS_PER_ROW), batch, strict=False):
            column.metric(md_escape(fact["label"]), format_number(fact["value"]))
    if texts_:
        st.dataframe(
            pd.DataFrame(
                {
                    "Fact": [str(f["label"]) for f in texts_],
                    "Value": ["" if f.get("value") is None else str(f["value"]) for f in texts_],
                }
            ),
            hide_index=True,
            width="stretch",
        )


def display_frame(table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Computed results for display: float noise such as 353358.62999999995 shows as 353358.63.

    Float columns are rounded to 15 significant digits (what a double holds
    reliably) for display only; exports and saved results keep full precision.
    """
    if not isinstance(table, pd.DataFrame):
        return table
    out = table.copy()
    for name in out.columns:
        if pd.api.types.is_float_dtype(out[name].dtype):
            out[name] = out[name].map(lambda value: float(f"{value:.15g}") if pd.notna(value) else value)
    return out


def render_result(
    *,
    table: pd.DataFrame | None,
    chart_spec: dict | None,
    facts: list[dict[str, Any]] | None,
    warnings: list[str] | None,
    key: str,
    truncated: bool = False,
    total_rows: int | None = None,
    caps: Any = None,
) -> None:
    """Facts, then the result table and the chart in tabs, then warnings.

    ``caps`` is ``components.navigation.capabilities()``. Charts are drawn only
    when the plan includes standard charts; without capabilities no chart is
    drawn. A stored chart spec is ignored when the feature is missing, and
    heatmap or correlation charts need the advanced analytics feature.
    """
    charts = caps is not None and caps.has(Feature.STANDARD_CHARTS)
    advanced = caps is not None and caps.has(Feature.ADVANCED_ANALYTICS)
    render_facts(facts)
    table_tab, chart_tab = st.tabs(["Result table", "Chart"])
    with table_tab:
        show_table(display_frame(table), height=320, empty_message="This result has no table.")
        if truncated and isinstance(table, pd.DataFrame):
            st.caption(f"Showing the first {len(table):,} of {total_rows or len(table):,} result rows.")
    with chart_tab:
        if not charts:
            feature_unavailable("Charts are not included in your plan. The result table has every value.")
        elif _is_table(table):
            current = normalize_chart_spec(table, chart_spec, advanced=advanced)
            spec = chart_type_override(table, current, key=key, advanced=advanced)
            render_chart(table, spec if _allowed(spec, advanced) else None, key=key)
        else:
            st.info("There is nothing to chart for this result.")
    for warning in warnings or []:
        st.warning(md_escape(warning), icon=":material/warning:")


def result_actions(
    ctx: AuthContext, caps: Any, history_id: int, *, default_title: str, key: str
) -> None:
    """Save as report and prepare CSV/Excel downloads for a successful analysis.

    ``caps`` is ``components.navigation.capabilities()``. Exports are only
    generated by the explicit "Prepare" buttons and kept in session state.
    """
    with st.expander("Save or download this result"):
        save_col, export_col = st.columns(2)
        with save_col:
            st.markdown("**Save as report**")
            if caps.has(Feature.SAVED_REPORTS) and caps.can(Permission.SAVE_REPORTS):
                with st.form(f"{key}_save_form", clear_on_submit=True):
                    title = st.text_input("Report title", value=default_title[:200], key=f"{key}_save_title")
                    notes = st.text_area("Notes (optional)", key=f"{key}_save_notes")
                    submitted = st.form_submit_button("Save as report")
                if submitted:
                    run_action(
                        lambda: analyst.save_report(ctx, history_id=history_id, title=title, notes=notes),
                        success="Report saved. Find it on the Saved reports page.",
                        component="report",
                    )
            else:
                feature_unavailable(
                    "Saving reports needs the saved reports feature in your plan and the permission to save reports."
                )
        with export_col:
            st.markdown("**Download the result table**")
            if caps.can(Permission.EXPORT_DATA):
                for label, file_format in (
                    ("Prepare CSV of this result", "csv"),
                    ("Prepare Excel of this result", "xlsx"),
                ):
                    slot = f"{key}_export_{file_format}"
                    if st.button(label, key=f"{key}_prepare_{file_format}"):
                        prepared = run_action(
                            lambda f=file_format: analyst.export_history_result(ctx, history_id, file_format=f),
                            success="Download prepared.",
                            component="export",
                        )
                        if prepared is not None:
                            st.session_state[slot] = prepared
                    ready = st.session_state.get(slot)
                    if ready is not None:
                        data, file_name, mime = ready
                        st.download_button(
                            md_escape(f"Download {file_name}"),
                            data=data,
                            file_name=file_name,
                            mime=mime,
                            key=f"{key}_download_{file_format}",
                        )
            else:
                feature_unavailable("Downloading results needs the permission to export data.")


__all__ = [
    "chart_type_override",
    "normalize_chart_spec",
    "render_chart",
    "render_facts",
    "render_result",
    "result_actions",
]
