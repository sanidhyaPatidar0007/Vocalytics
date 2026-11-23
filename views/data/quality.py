"""Data quality: profile, automated health check, separated issues and repeated values."""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.data_preview import preview_table
from components.dataset_picker import dataset_selector
from components.feedback import Progress, feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.quality_report import column_profile_table, issues_view, missing_chart, quality_header
from components.session import page_state, require_ctx, set_page_state
from core.constants import Feature
from services import dataset_service, profiling_service
from services.resource_manager import OperationGuard

PAGE = "quality"
STATUS_TEXT = {
    "healthy": "Healthy",
    "needs attention": "Needs attention",
    "critical": "Critical",
    "empty": "Empty (no rows)",
}

ctx = require_ctx()
caps = capabilities()
page_header(
    "Data quality",
    "Profile a dataset, run the automated health check and review confirmed and suspected issues.",
)


def _stored(name: str, dataset_id: int, version_id: int | None) -> Any:
    item = page_state(PAGE, name)
    if item and item["dataset_id"] == dataset_id and item["version_id"] == version_id:
        return item["data"]
    return None


def _store(name: str, dataset_id: int, version_id: int | None, data: Any) -> None:
    set_page_state(PAGE, name, {"dataset_id": dataset_id, "version_id": version_id, "data": data})


def _repeated_values(dataset_id: int, version_id: int | None, columns: list[str]) -> None:
    st.subheader("Repeated values")
    st.caption(
        "Values that occur more than once in the selected columns (or combinations of them). Repeated values are "
        "not necessarily duplicate records: cities, categories and names repeat legitimately. Nothing is changed."
    )
    picked = st.multiselect("Columns to check", columns, key=f"{PAGE}_rep_cols_{dataset_id}")
    c1, c2, c3 = st.columns(3)
    trim = c1.checkbox("Ignore spaces at the start and end", value=True, key=f"{PAGE}_rep_trim_{dataset_id}")
    ci = c2.checkbox("Ignore letter case", value=False, key=f"{PAGE}_rep_ci_{dataset_id}")
    top = c3.number_input(
        "Show at most", min_value=1, max_value=500, value=50, step=10, key=f"{PAGE}_rep_top_{dataset_id}"
    )
    if st.button("Find repeated values", key=f"{PAGE}_rep_run"):
        result = run_action(
            lambda: profiling_service.repeated_values(
                ctx, dataset_id, picked, trim=trim, case_insensitive=ci, top=int(top)
            ),
            component=PAGE,
        )
        if result is not None:
            _store("repeated", dataset_id, version_id, result)
    result = _stored("repeated", dataset_id, version_id)
    if result is not None:
        if result.empty:
            st.info("No value repeats in the selected columns.")
        else:
            st.caption(
                f"{len(result):,} repeated value(s), most frequent first. The "
                "count is the number of rows with that value."
            )
            preview_table(result)


selected = dataset_selector(ctx, key=f"{PAGE}_select")
if selected is not None:
    if not caps.has(Feature.DATA_PROFILING):
        feature_unavailable("Your plan does not include data profiling.")
        st.stop()
    dataset_id = selected["dataset_id"]
    version_id = selected.get("current_version_id")
    info = run_action(lambda: dataset_service.get_dataset(ctx, dataset_id), component=PAGE)
    if info is None:
        st.stop()
    c1, c2 = st.columns(2)
    run_profile = c1.button("Run profile", key=f"{PAGE}_profile", type="primary", icon=":material/analytics:")
    run_health = c2.button("Run health check", key=f"{PAGE}_health", icon=":material/health_and_safety:")
    if run_profile:
        progress = Progress("Profiling the dataset")
        profile = run_action(
            lambda: profiling_service.profile_dataset(
                ctx, dataset_id, guard=OperationGuard(progress=progress)
            ),
            component=PAGE,
        )
        progress.done()
        if profile is not None:
            _store("profile", dataset_id, profile["version_id"], profile)
    if run_health:
        with st.spinner("Running the health check"):
            health = run_action(lambda: profiling_service.health_check(ctx, dataset_id), component=PAGE)
        if health is not None:
            _store("health", dataset_id, health["version_id"], health)

    profile = _stored("profile", dataset_id, version_id)
    health = _stored("health", dataset_id, version_id)
    status = STATUS_TEXT.get(health["status"], health["status"]) if health else None
    quality_header(profile["quality"] if profile else info.get("quality"), status=status)

    if health:
        with st.container(border=True):
            st.subheader(f"Health check: {status}")
            st.caption(
                f"Version {health['version_number']}: {health['confirmed_issue_count']} confirmed problem(s) and "
                f"{health['suspected_issue_count']} suspected issue(s). "
                + " ".join(
                    f"{md_escape(k.capitalize())}: {md_escape(v)}." for k, v in health["thresholds"].items()
                )
            )
            issues_view(health["top_issues"], health["suspected_issues"])

    if profile:
        st.subheader("Issues found by the profile")
        issues_view(profile["confirmed_issues"], profile["suspected_issues"])
        st.subheader("Column profile")
        column_profile_table(profile["columns_profile"])
        missing_chart(profile["columns_profile"], key=f"{PAGE}_missing_chart")
    elif not health:
        st.info(
            "Run the profile for the full list of confirmed and suspected issues, or "
            "the health check for a quick status with next steps."
        )

    columns = [c["name"] for c in info.get("schema") or []]
    with st.container(border=True):
        _repeated_values(dataset_id, version_id, columns)
