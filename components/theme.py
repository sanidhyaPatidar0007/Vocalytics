"""Visual system: a small set of tokens applied once per page.

Palette (also in .streamlit/config.toml): ink #16212B, field #F7F9FB,
steel #E9EEF3, harbor blue #1F5F8B (primary), confirmed-issue red #B3261E,
suspected-issue amber #A15C00, healthy green #2E7D4F. Confirmed problems are
drawn with a solid edge, suspected ones with a dashed edge, so the difference
is visible without relying on colour alone.

The CSS below is a static constant. Never interpolate user or dataset text
into it.
"""

from __future__ import annotations

import streamlit as st

COLORS = {
    "ink": "#16212B",
    "field": "#F7F9FB",
    "steel": "#E9EEF3",
    "primary": "#1F5F8B",
    "confirmed": "#B3261E",
    "suspected": "#A15C00",
    "healthy": "#2E7D4F",
}

_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
h1 {letter-spacing: -0.01em;}
div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DCE3EA;
  border-radius: 8px; padding: 0.6rem 0.9rem;}
div[data-testid="stMetricLabel"] p {color: #4A5A68;}
.vdas-issue-confirmed {border-left: 4px solid #B3261E; padding: 0.35rem 0.7rem; margin: 0.25rem 0; background: #FFFFFF;}
.vdas-issue-suspected {border-left: 4px dashed #A15C00; padding: 0.35rem 0.7rem;
  margin: 0.25rem 0; background: #FFFFFF;}
.vdas-score {font-size: 3.2rem; font-weight: 700; line-height: 1; color: #16212B;}
.vdas-score-label {color: #4A5A68; font-size: 0.85rem;}
.vdas-pill {display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px;
  background: #E9EEF3; color: #16212B; font-size: 0.8rem;}
@media (prefers-reduced-motion: reduce) {* {transition: none !important; animation: none !important;}}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def score_color(score: float | None) -> str:
    if score is None:
        return COLORS["ink"]
    if score >= 85:
        return COLORS["healthy"]
    if score >= 65:
        return COLORS["suspected"]
    return COLORS["confirmed"]


# Streamlit adds a "Download as CSV" button to every table toolbar. Analysts
# without the export permission must not be offered file downloads, so the
# button is hidden for them (static CSS; exports are also enforced server-side).
_HIDE_TABLE_DOWNLOAD_CSS = """
<style>
[data-testid="stElementToolbar"] button[aria-label="Download as CSV"] {display: none !important;}
</style>
"""


def hide_table_downloads() -> None:
    st.markdown(_HIDE_TABLE_DOWNLOAD_CSS, unsafe_allow_html=True)

