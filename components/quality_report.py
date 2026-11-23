"""Data quality report: score header, separated issues, column profile and missing chart.

The quality score is the page's one bold element. Confirmed problems (solid
red edge) and suspected issues (dashed amber edge) use the static theme
classes from ``components.theme``; every dynamic value placed inside that HTML
is flattened to one line and passed through ``components.feedback.escape``
(see ``_html_text``). Column names in the missing-values chart are
HTML-escaped because Plotly renders a subset of HTML in labels.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.data_preview import safe_frame
from components.feedback import escape
from components.theme import COLORS, score_color
from services.quality_score import WEIGHTS

COMPONENT_LABELS = {
    "completeness": "Completeness (cells that are not missing)",
    "uniqueness": "Uniqueness (rows that are not exact duplicates)",
    "validity": "Validity (values that fit their column)",
    "consistency": "Consistency (text without stray spaces)",
}


def _html_text(value: Any) -> str:
    """Untrusted text for an HTML block: one line, then HTML-escaped.

    Under CommonMark an HTML block ends at the first blank line, so a value
    holding a blank line would turn the rest of the block into Markdown (images,
    links). Collapsing all whitespace keeps the whole block HTML.
    """
    return escape(" ".join(("" if value is None else str(value)).split()))


def _fraction(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def quality_header(quality: dict[str, Any] | None, *, status: str | None = None) -> None:
    """Large quality score colored by health, the formula components as bars, and the counts."""
    if not quality:
        st.info("No quality score is available yet. Run the profile to compute it.")
        return
    score = float(quality.get("score") or 0.0)
    color = score_color(score)
    left, right = st.columns([1, 2], gap="large")
    with left, st.container(border=True):
        status_html = f'<div class="vdas-score-label">Health: {_html_text(status)}</div>' if status else ""
        st.markdown(
            f'<div class="vdas-score-label">Quality score (0 to 100)</div>'
            f'<div class="vdas-score" style="color: {_html_text(color)};">{_html_text(f"{score:.1f}")}</div>'
            f"{status_html}",
            unsafe_allow_html=True,
        )
        st.caption(
            "Only confirmed problems lower the score. Suspected issues are listed for review but never penalized."
        )
    with right:
        for key, label in COMPONENT_LABELS.items():
            value = _fraction(quality.get(key))
            weight = int(round(WEIGHTS.get(key, 0) * 100))
            st.progress(value, text=f"{label}: {value * 100:.1f}% (weight {weight}%)")
    counts = st.columns(5)
    counts[0].metric("Rows", f"{int(quality.get('rows') or 0):,}")
    counts[1].metric("Columns", f"{int(quality.get('columns') or 0):,}")
    counts[2].metric("Missing cells", f"{int(quality.get('missing_cells') or 0):,}")
    counts[3].metric("Duplicate rows", f"{int(quality.get('duplicate_rows') or 0):,}")
    counts[4].metric("Invalid cells", f"{int(quality.get('invalid_cells') or 0):,}")


def _issue_html(issue: dict[str, Any], css_class: str) -> str:
    column = issue.get("column")
    head = f"<strong>{_html_text(column)}</strong> &middot; " if column else ""
    severity = issue.get("severity")
    pill = f' <span class="vdas-pill">{_html_text(severity)}</span>' if severity else ""
    source = ' <span class="vdas-pill">from the import</span>' if issue.get("source") == "import" else ""
    step = issue.get("recommended_step")
    step_html = f'<br><span class="vdas-score-label">Next step: {_html_text(step)}</span>' if step else ""
    return f'<div class="{css_class}">{head}{_html_text(issue.get("message", ""))}{pill}{source}{step_html}</div>'


def issues_view(confirmed: list[dict[str, Any]] | None, suspected: list[dict[str, Any]] | None) -> None:
    """Confirmed problems (solid edge) and suspected issues (dashed edge), never mixed."""
    confirmed = confirmed or []
    suspected = suspected or []
    st.markdown(
        '<div class="vdas-issue-confirmed"><strong>Confirmed problem</strong>: definitely wrong or missing data '
        "that lowers the score.</div>"
        '<div class="vdas-issue-suspected"><strong>Suspected issue</strong>: needs a human decision; it may be '
        "legitimate (a real outlier or a common category) and is not counted against the score.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(f"Confirmed problems ({len(confirmed)})")
        if not confirmed:
            st.success("No confirmed problems were found.", icon=":material/check_circle:")
        else:
            st.markdown(
                "".join(_issue_html(i, "vdas-issue-confirmed") for i in confirmed), unsafe_allow_html=True
            )
    with right:
        st.subheader(f"Suspected issues ({len(suspected)})")
        if not suspected:
            st.info("Nothing needs review.", icon=":material/info:")
        else:
            st.markdown(
                "".join(_issue_html(i, "vdas-issue-suspected") for i in suspected), unsafe_allow_html=True
            )


def _fmt(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def column_profile_frame(columns_profile: list[dict[str, Any]] | None) -> pd.DataFrame:
    rows = []
    for col in columns_profile or []:
        numeric = col.get("numeric") or {}
        dates = col.get("datetime") or {}
        text = col.get("text") or {}
        top = col.get("top_values") or []
        rows.append(
            {
                "Column": col.get("name"),
                "Type": col.get("logical_type"),
                "Semantic hint": col.get("semantic"),
                "Missing": col.get("missing"),
                "Missing %": col.get("missing_pct"),
                "Distinct": col.get("distinct"),
                "Distinct %": col.get("distinct_pct"),
                "Constant": bool(col.get("is_constant")),
                "Empty": bool(col.get("is_empty")),
                "Min": _fmt(numeric.get("min", dates.get("min"))),
                "Max": _fmt(numeric.get("max", dates.get("max"))),
                "Mean": _fmt(numeric.get("mean")),
                "Median": _fmt(numeric.get("median")),
                "Outlier candidates": numeric.get("outlier_candidates"),
                "Spaces around text": text.get("whitespace_count"),
                "Most common": ", ".join(f"{t.get('value')} ({t.get('count')})" for t in top[:3]),
            }
        )
    return safe_frame(pd.DataFrame(rows))


def column_profile_table(columns_profile: list[dict[str, Any]] | None) -> None:
    frame = column_profile_frame(columns_profile)
    if frame.empty:
        st.info("No columns to profile.")
        return
    st.dataframe(frame, hide_index=True, width="stretch")


def missing_chart(columns_profile: list[dict[str, Any]] | None, *, key: str) -> None:
    """Bar chart of the missing percentage per column."""
    cols = columns_profile or []
    if not cols:
        return
    # Plotly renders a subset of HTML (links, styles) in tick and hover labels.
    names = [html.escape(str(c.get("name")), quote=False) for c in cols]
    values = [float(c.get("missing_pct") or 0.0) for c in cols]
    if not any(values):
        st.success("No column has missing values.", icon=":material/check_circle:")
        return
    figure = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=[COLORS["confirmed"] if v >= 50 else COLORS["primary"] for v in values],
            hovertemplate="%{x}: %{y:.2f}% missing<extra></extra>",
        )
    )
    figure.update_layout(
        title="Missing values per column (%)",
        yaxis_title="Missing (%)",
        xaxis_title="Column",
        yaxis_range=[0, 100],
        margin={"l": 40, "r": 10, "t": 50, "b": 40},
        height=340,
    )
    st.plotly_chart(figure, key=key, width="stretch")
    st.caption("Bars at or above 50% are drawn in red: half or more of that column is missing.")


__all__ = ["column_profile_frame", "column_profile_table", "issues_view", "missing_chart", "quality_header"]
