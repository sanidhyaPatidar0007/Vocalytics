"""Dataset selector shared by data and analysis pages.

Only datasets returned by ``dataset_service.list_datasets(ctx)`` (already
scoped to the organization and, for analysts, to granted datasets) can be
selected, so a page can never be pointed at another tenant's dataset id.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from auth.authorization import AuthContext
from components.feedback import run_action
from services import dataset_service

SELECTED_KEY = "_vdas_selected_dataset"


def dataset_selector(
    ctx: AuthContext,
    *,
    key: str,
    label: str = "Dataset",
    help_text: str | None = None,
    shared: bool = True,
    avoid: int | None = None,
) -> dict[str, Any] | None:
    """Render a dataset picker and return the selected dataset row (or None).

    By default the choice is shared across pages, so moving between Quality,
    Missing values and Analysis keeps the same dataset selected. A second picker
    on the same page (append source, merge right side) passes ``shared=False``
    and ``avoid`` (the other side's dataset id) so it starts on a different
    dataset and does not overwrite the shared choice.
    """
    datasets = run_action(lambda: dataset_service.list_datasets(ctx), component="dataset_picker") or []
    if not datasets:
        st.info("No datasets yet. Upload one on the Datasets page (or ask an admin to share one with you).")
        return None
    by_id = {d["dataset_id"]: d for d in datasets}
    ids = list(by_id)
    if shared:
        remembered = st.session_state.get(SELECTED_KEY)
        index = ids.index(remembered) if remembered in by_id else 0
    else:
        index = next((i for i, dataset_id in enumerate(ids) if dataset_id != avoid), 0)

    def _format(dataset_id: int) -> str:
        row = by_id[dataset_id]
        quality = row.get("quality_score")
        score = f", quality {quality:.1f}" if quality is not None else ""
        return f"{row['name']} (v{row.get('current_version') or '-'}, {row['rows']:,} rows{score})"

    choice = st.selectbox(label, ids, index=index, format_func=_format, key=key, help=help_text)
    if shared:
        # Several shared pickers can sit on one page (append target, merge left). Only a picker
        # whose value changed since its last run may update the shared choice, so an untouched
        # picker never overwrites what the user just picked in another one.
        last_key = f"{SELECTED_KEY}:last:{key}"
        if st.session_state.get(last_key) != choice:
            st.session_state[SELECTED_KEY] = choice
            st.session_state[last_key] = choice
    return by_id.get(choice)
