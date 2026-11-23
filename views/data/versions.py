"""Versions and history: every version, the operation history, previews, restore and integrity."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from components.data_preview import applied_version_view, human_bytes, preview_table, version_badge
from components.dataset_picker import dataset_selector
from components.feedback import confirm, feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import page_state, require_ctx, set_page_state
from core.constants import Permission
from services import dataset_service, versioning_service

PAGE = "versions"

ctx = require_ctx()
caps = capabilities()
page_header(
    "Versions and history",
    "Every saved version of a dataset, what changed it, and restoring an earlier version.",
)


def _versions_table(versions: list[dict[str, Any]]) -> None:
    rows = [
        {
            "Version": version_badge(
                v["version_number"],
                is_current=v["is_current"],
                is_original=v["is_original_import"],
                is_purged=v["is_purged"],
            ),
            "Created": v["created_at"],
            "By": v["created_by"],
            "Operation": v["operation_type"],
            "Rows": v["rows"],
            "Columns": v["columns"],
            "Size": human_bytes(v["size_bytes"]),
            "Quality": None if v["quality_score"] is None else round(float(v["quality_score"]), 1),
            "Note": v.get("note") or "",
        }
        for v in versions
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Purged versions were removed by the retention limit; their history stays but their data cannot be restored."
    )


def _operations(dataset_id: int) -> None:
    st.subheader("Operation history")
    operations = run_action(lambda: versioning_service.list_operations(ctx, dataset_id), component=PAGE) or []
    if not operations:
        st.info("No operations have been recorded for this dataset yet.")
        return

    def score(value: Any) -> Any:
        return None if value is None else round(float(value), 1)

    table = pd.DataFrame(
        [
            {
                "Operation": op["operation_type"],
                "Status": op["status"],
                "User": op["user"],
                "Time": op["created_at"],
                "Version": f"v{op['input_version'] or '-'} to v{op['output_version'] or '-'}",
                "Rows before": op["rows_before"],
                "Rows after": op["rows_after"],
                "Rows removed": op["rows_removed"],
                "Cells changed": op["cells_changed"],
                "Quality before": score(op["quality_before"]),
                "Quality after": score(op["quality_after"]),
            }
            for op in operations
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    by_id = {op["operation_id"]: op for op in operations}
    with st.expander("Operation parameters and details"):
        choice = st.selectbox(
            "Operation",
            list(by_id),
            format_func=lambda i: (
                f"{by_id[i]['operation_type']} ({by_id[i]['status']}, {by_id[i]['created_at']:%Y-%m-%d %H:%M})"
                if hasattr(by_id[i]["created_at"], "strftime")
                else f"{by_id[i]['operation_type']} ({by_id[i]['status']})"
            ),
            key=f"{PAGE}_op_{dataset_id}",
        )
        op = by_id[choice]
        if op.get("error_message"):
            st.markdown("**Why it did not complete**")
            st.text(op["error_message"])
        st.markdown("**Parameters**")
        st.code(json.dumps(op["parameters"], indent=2, default=str, ensure_ascii=False), language="json")
        if op.get("summary"):
            st.markdown("**Summary**")
            st.code(
                json.dumps(op["summary"], indent=2, default=str, ensure_ascii=False)[:20000], language="json"
            )


def _version_tools(dataset: dict[str, Any], versions: list[dict[str, Any]]) -> None:
    dataset_id = dataset["dataset_id"]
    st.subheader("Look at a version")
    by_id = {v["version_id"]: v for v in versions}
    choice = st.selectbox(
        "Version",
        list(by_id),
        format_func=lambda i: (
            version_badge(
                by_id[i]["version_number"],
                is_current=by_id[i]["is_current"],
                is_original=by_id[i]["is_original_import"],
                is_purged=by_id[i]["is_purged"],
            )
            + f", {by_id[i]['operation_type']}"
        ),
        key=f"{PAGE}_version_{dataset_id}",
    )
    version = by_id[choice]
    if version["is_purged"]:
        st.warning(
            "This version's data was purged by the retention limit, so it cannot be previewed, exported or restored."
        )
        return
    preview = run_action(
        lambda: versioning_service.load_preview(ctx, dataset_id, version_id=choice, max_rows=50),
        component=PAGE,
    )
    if preview is not None:
        st.caption(
            f"First {len(preview):,} of {version['rows']:,} rows of version {version['version_number']}."
        )
        preview_table(preview)

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("**Restore**")
        if not caps.can(Permission.CLEAN_DATA):
            feature_unavailable("You can view versions, but restoring needs the clean data permission.")
        elif version["is_current"]:
            st.caption("This is already the current version.")
        else:
            st.caption(
                "Restoring copies this version forward as a new current version. No version is deleted."
            )
            ok = confirm(
                f"Make a copy of version {version['version_number']} the current data",
                key=f"{PAGE}_restore_confirm_{choice}",
            )
            if st.button(
                "Restore this version", key=f"{PAGE}_restore_{choice}", disabled=not ok, type="primary"
            ):
                result = run_action(
                    lambda: versioning_service.restore_version(ctx, dataset_id, choice), component=PAGE
                )
                if result is not None:
                    set_page_state(PAGE, "restored", {"dataset_id": dataset_id, "data": result.to_dict()})
                    st.rerun()
    with right, st.container(border=True):
        st.markdown("**Export this version**")
        if not caps.can(Permission.EXPORT_DATA):
            feature_unavailable("You do not have permission to export data.")
        else:
            exports = page_state(PAGE, "exports", {}) or {}
            if st.button("Prepare CSV of this version", key=f"{PAGE}_prepare_{choice}"):
                result = run_action(
                    lambda: dataset_service.export_dataset(
                        ctx, dataset_id, version_id=choice, file_format="csv"
                    ),
                    component=PAGE,
                )
                if result is not None:
                    exports[choice] = {"data": result[0], "name": result[1], "mime": result[2]}
                    set_page_state(PAGE, "exports", exports)
            prepared = exports.get(choice)
            if prepared:
                st.download_button(
                    f"Download {md_escape(prepared['name'])}",
                    data=prepared["data"],
                    file_name=prepared["name"],
                    mime=prepared["mime"],
                    key=f"{PAGE}_download_{choice}",
                    icon=":material/download:",
                )


def _integrity(dataset_id: int) -> None:
    st.subheader("Original upload")
    if st.button("Check original file integrity", key=f"{PAGE}_verify"):
        result = run_action(lambda: versioning_service.verify_original(ctx, dataset_id), component=PAGE)
        if result is not None:
            set_page_state(PAGE, "verify", {"dataset_id": dataset_id, "data": result})
    stored = page_state(PAGE, "verify")
    if stored and stored["dataset_id"] == dataset_id:
        result = stored["data"]
        if not result["has_original"]:
            st.info("This dataset was derived from other datasets, so there is no original upload to check.")
        elif result["intact"]:
            st.success(
                f"The original upload ({human_bytes(result.get('size_bytes'))}) "
                "matches its recorded checksum and is unchanged.",
                icon=":material/verified:",
            )
        else:
            st.error(
                "The stored original file does not match its recorded checksum. Contact your administrator."
            )
    else:
        st.caption("The original upload is stored unchanged with a checksum. Check it at any time.")


selected = dataset_selector(ctx, key=f"{PAGE}_select")
if selected is not None:
    dataset_id = selected["dataset_id"]
    restored = page_state(PAGE, "restored")
    if restored and restored["dataset_id"] == dataset_id:
        applied_version_view(restored["data"], label="Restored")
        set_page_state(PAGE, "restored", None)
    versions = run_action(lambda: versioning_service.list_versions(ctx, dataset_id), component=PAGE) or []
    st.markdown(f"**{md_escape(selected['name'])}**: {len(versions)} version(s)")
    if versions:
        _versions_table(versions)
        _version_tools(selected, versions)
    _operations(dataset_id)
    _integrity(dataset_id)
