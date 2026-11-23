"""Append and merge: combine two datasets after an exact analysis and a preview.

Nothing is written until the final confirmed button; not applying a preview is
how an append or merge is cancelled.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.data_preview import applied_version_view, human_bytes, records_table
from components.dataset_picker import dataset_selector
from components.feedback import Progress, confirm, feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import page_state, require_ctx, set_page_state
from core.constants import Feature, Permission
from services import append_service, dataset_service, merge_service
from services.resource_manager import OperationGuard

PAGE = "combine"
UNMATCHED_LABELS = {
    "union": "Keep every column (missing values where a side lacks it)",
    "intersection": "Keep only columns present in both",
    "strict": "Require identical columns",
}
TYPE_LABELS = {
    "error": "Stop when column types are incompatible",
    "cast_to_text": "Store incompatible columns as text",
}
OUTPUT_LABELS = {
    "new_version": "Save as a new version of the first dataset",
    "new_dataset": "Save as a new dataset",
}
JOIN_LABELS = {
    "inner": "Inner: only rows whose keys match on both sides",
    "left": "Left: every left row, matched right values where found",
    "right": "Right: every right row, matched left values where found",
    "outer": "Outer: every row from both sides",
}

ctx = require_ctx()
caps = capabilities()
page_header(
    "Append and merge",
    "Stack the rows of two datasets or join them on key columns, after checking the exact result.",
)


def _pair_state(name: str, ids: tuple[Any, ...]) -> dict[str, Any] | None:
    stored = page_state(PAGE, name)
    if stored and stored["ids"] == ids:
        return stored
    return None


def _put(name: str, ids: tuple[Any, ...], params: Any, data: Any) -> None:
    set_page_state(PAGE, name, None if data is None else {"ids": ids, "params": params, "data": data})


def _resource(check: dict[str, Any]) -> None:
    if check["ok"]:
        st.caption(md_escape(check["message"]))
    else:
        st.warning(md_escape(check["message"]), icon=":material/memory:")


def _warnings(items: list[str]) -> None:
    for item in items:
        st.warning(md_escape(item))


def _names(dataset_id: int) -> list[str]:
    info = run_action(lambda: dataset_service.get_dataset(ctx, dataset_id), component=PAGE)
    return [c["name"] for c in (info or {}).get("schema") or []]


def _output_choice(prefix: str, default: str) -> tuple[str, str | None]:
    options = list(OUTPUT_LABELS) if default == "new_version" else ["new_dataset", "new_version"]
    output = st.radio(
        "Save the result", options, format_func=OUTPUT_LABELS.get, key=f"{PAGE}_{prefix}_output"
    )
    name = None
    if output == "new_dataset":
        name = (
            st.text_input(
                "New dataset name (leave empty for a generated name)",
                key=f"{PAGE}_{prefix}_name",
                max_chars=200,
            ).strip()
            or None
        )
    return output, name


def _done(result: dict[str, Any], quality_before: float | None, label: str) -> None:
    for name in ("append_analysis", "append_preview", "merge_analysis", "merge_preview"):
        set_page_state(PAGE, name, None)
    set_page_state(PAGE, "applied", {"data": result, "quality_before": quality_before, "label": label})
    st.rerun()


# ---------------------------------------------------------------------------
# Append
# ---------------------------------------------------------------------------


def _append_analysis_view(analysis: dict[str, Any]) -> None:
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Target rows", f"{analysis['target']['rows']:,}")
        cols[1].metric("Source rows", f"{analysis['source']['rows']:,}")
        cols[2].metric("Rows after appending", f"{analysis['estimated_output_rows']:,}")
        cols[3].metric("Combined input size", human_bytes(analysis["combined_input_bytes"]))
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Matching columns**")
        c1.text("\n".join(analysis["matching_columns"]) or "none")
        c2.markdown("**Only in the target**")
        c2.text("\n".join(analysis["only_in_target"]) or "none")
        c3.markdown("**Only in the source**")
        c3.text("\n".join(analysis["only_in_source"]) or "none")
        if analysis["type_conflicts"]:
            st.markdown("**Type conflicts**")
            records_table(
                [
                    {
                        "Column": c["column"],
                        "Target type": c["target_type"],
                        "Source type": c["source_type"],
                        "Result when stopping on conflicts": c["resulting_type"]["error"] or "rejected",
                        "Result when storing as text": c["resulting_type"]["cast_to_text"],
                    }
                    for c in analysis["type_conflicts"]
                ]
            )
        if analysis["case_only_differences"]:
            st.markdown("**Names that differ only by letter case (never matched automatically)**")
            records_table(analysis["case_only_differences"])
        if analysis["schemas_identical"]:
            st.success("Both datasets have the same columns and types.")
        _warnings(analysis["warnings"])
        _resource(analysis["resource_check"])


def _append_preview_view(preview: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**Preview (nothing saved)**")
        cols = st.columns(3)
        cols[0].metric("Rows after appending", f"{preview['output_rows']:,}")
        cols[1].metric("Output columns", f"{len(preview['columns']):,}")
        dupes = preview["possible_duplicates"]
        cols[2].metric("Possible duplicate rows", "not checked" if dupes is None else f"{dupes:,}")
        records_table(
            [
                {
                    "Column": c["name"],
                    "Type": c["logical_type"],
                    "From target": c["from_target"],
                    "From source": c["from_source"],
                }
                for c in preview["columns"]
            ]
        )
        dropped = preview["dropped_columns"]
        if dropped["target"] or dropped["source"]:
            st.info(
                "Dropped columns: target "
                + (", ".join(md_escape(c) for c in dropped["target"]) or "none")
                + "; source "
                + (", ".join(md_escape(c) for c in dropped["source"]) or "none")
            )
        _warnings(preview["warnings"])
        left, right = st.columns(2)
        with left:
            st.caption("First target rows (aligned to the output)")
            records_table(preview["sample_target_rows"], empty_message="The target has no rows.")
        with right:
            st.caption("First source rows (aligned to the output)")
            records_table(preview["sample_source_rows"], empty_message="The source has no rows.")


def _append_tab() -> None:
    target = dataset_selector(ctx, key=f"{PAGE}_append_target", label="Target dataset (rows are added to it)")
    source = dataset_selector(
        ctx,
        key=f"{PAGE}_append_source",
        label="Source dataset (rows to add)",
        shared=False,
        avoid=target["dataset_id"] if target else None,
    )
    if not target or not source:
        return
    if target["dataset_id"] == source["dataset_id"]:
        st.info("Choose two different datasets: the source rows are appended to the target.")
        return
    ids = (
        target["dataset_id"],
        target.get("current_version_id"),
        source["dataset_id"],
        source.get("current_version_id"),
    )
    if st.button("Analyze append", key=f"{PAGE}_append_analyze"):
        result = run_action(
            lambda: append_service.analyze_append(ctx, target["dataset_id"], source["dataset_id"]),
            component=PAGE,
        )
        _put("append_analysis", ids, None, result)
    analysis = _pair_state("append_analysis", ids)
    if not analysis:
        st.caption("Analyze first to compare the columns and types of both datasets.")
        return
    _append_analysis_view(analysis["data"])
    st.markdown("**Append options**")
    c1, c2 = st.columns(2)
    unmatched = c1.radio(
        "Columns that exist on one side only",
        list(UNMATCHED_LABELS),
        format_func=UNMATCHED_LABELS.get,
        key=f"{PAGE}_append_unmatched",
    )
    conflicts = c2.radio(
        "Incompatible column types",
        list(TYPE_LABELS),
        format_func=TYPE_LABELS.get,
        key=f"{PAGE}_append_types",
    )
    label = (
        st.text_input(
            "Add a column with the source dataset name (optional column name)",
            key=f"{PAGE}_append_label",
            max_chars=128,
        ).strip()
        or None
    )
    params = {"unmatched_policy": unmatched, "type_conflict_policy": conflicts, "source_label_column": label}
    if st.button("Preview append", key=f"{PAGE}_append_preview", type="primary"):
        progress = Progress("Checking for duplicate rows")
        result = run_action(
            lambda: append_service.preview_append(
                ctx,
                target["dataset_id"],
                source["dataset_id"],
                **params,
                guard=OperationGuard(progress=progress),
            ),
            component=PAGE,
        )
        progress.done()
        _put("append_preview", ids, params, result)
    stored = _pair_state("append_preview", ids)
    if not stored:
        return
    _append_preview_view(stored["data"])
    if stored["params"] != params:
        st.info("The options changed after this preview. Preview again before appending.")
        return
    if not caps.can(Permission.CLEAN_DATA):
        feature_unavailable("Appending needs the clean data permission.")
        return
    output, new_name = _output_choice("append", "new_version")
    st.caption("To cancel, simply do not apply: nothing has been saved yet.")
    ok = confirm("Append these datasets as previewed", key=f"{PAGE}_append_confirm")
    if st.button("Append and save", key=f"{PAGE}_append_apply", disabled=not ok):
        saved = stored["params"]
        progress = Progress("Appending")
        result = run_action(
            lambda: append_service.apply_append(
                ctx,
                target["dataset_id"],
                source["dataset_id"],
                **saved,
                output=output,
                new_dataset_name=new_name,
                confirm=ok,
                guard=OperationGuard(progress=progress),
            ),
            component=PAGE,
        )
        progress.done()
        if result is not None:
            before = target.get("quality_score") if output == "new_version" else None
            _done(result, before, "Appended and saved")


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def _merge_analysis_view(analysis: dict[str, Any]) -> None:
    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Output rows (exact)", f"{analysis['output_rows']:,}")
        cols[1].metric("Growth", f"{analysis['growth_factor']:.2f}x of the larger input")
        cols[2].metric("Unmatched left rows", f"{analysis['left_unmatched_rows']:,}")
        cols[3].metric("Unmatched right rows", f"{analysis['right_unmatched_rows']:,}")
        relationship = analysis["relationship"]
        if analysis["many_to_many"]:
            st.error(
                f"Many-to-many relationship: {analysis['many_to_many_keys']:,} key value(s) repeat on both sides, so "
                "matching rows multiply. Applying needs an extra confirmation.",
                icon=":material/warning:",
            )
        else:
            st.caption(f"Relationship between the keys: {md_escape(relationship)}.")
        unmatched = analysis["unmatched"]
        records_table(
            [
                {
                    "Side": side,
                    "Rows": analysis[f"{side}_rows"],
                    "Matched rows": analysis[f"{side}_matched_rows"],
                    "Unmatched rows": unmatched[f"{side}_unmatched_rows"],
                    "Rows with a missing key": unmatched[f"{side}_null_key_rows"],
                    "Duplicate key values": analysis[f"{side}_duplicate_keys"],
                    "Unmatched kept": unmatched[f"{side}_unmatched_kept"],
                    "Unmatched left out": unmatched[f"{side}_rows_excluded"],
                }
                for side in ("left", "right")
            ]
        )
        st.caption(
            "Output rows for each join type (exact): "
            + ", ".join(f"{k} {v:,}" for k, v in analysis["output_rows_by_join"].items())
        )
        _warnings(analysis["warnings"])
        _resource(analysis["resource_check"])


def _merge_tab() -> None:
    left = dataset_selector(ctx, key=f"{PAGE}_merge_left", label="Left dataset")
    right = dataset_selector(
        ctx,
        key=f"{PAGE}_merge_right",
        label="Right dataset",
        shared=False,
        avoid=left["dataset_id"] if left else None,
    )
    if not left or not right:
        return
    if left["dataset_id"] == right["dataset_id"]:
        st.info("Choose two different datasets to merge.")
        return
    ids = (
        left["dataset_id"],
        left.get("current_version_id"),
        right["dataset_id"],
        right.get("current_version_id"),
    )
    c1, c2 = st.columns(2)
    left_on = c1.multiselect(
        "Left key columns", _names(left["dataset_id"]), key=f"{PAGE}_merge_left_on_{left['dataset_id']}"
    )
    right_on = c2.multiselect(
        "Right key columns", _names(right["dataset_id"]), key=f"{PAGE}_merge_right_on_{right['dataset_id']}"
    )
    how = st.selectbox("Join type", list(JOIN_LABELS), format_func=JOIN_LABELS.get, key=f"{PAGE}_merge_how")
    s1, s2 = st.columns(2)
    left_suffix = s1.text_input(
        "Suffix for left columns with the same name", value="_left", key=f"{PAGE}_merge_lsuffix"
    )
    right_suffix = s2.text_input(
        "Suffix for right columns with the same name", value="_right", key=f"{PAGE}_merge_rsuffix"
    )
    params = {
        "left_on": list(left_on),
        "right_on": list(right_on),
        "how": how,
        "suffixes": (left_suffix, right_suffix),
    }
    counts_ok = bool(left_on) and len(left_on) == len(right_on)
    if left_on and right_on and len(left_on) != len(right_on):
        st.warning("Choose the same number of key columns on both sides; they are paired in order.")
    b1, b2 = st.columns(2)
    if b1.button("Analyze merge", key=f"{PAGE}_merge_analyze", disabled=not counts_ok):
        result = run_action(
            lambda: merge_service.analyze_merge(ctx, left["dataset_id"], right["dataset_id"], **params),
            component=PAGE,
        )
        _put("merge_analysis", ids, params, result)
        set_page_state(PAGE, "merge_preview", None)
    if b2.button("Preview merge", key=f"{PAGE}_merge_preview", disabled=not counts_ok, type="primary"):
        with st.spinner("Merging a sample of keys"):
            result = run_action(
                lambda: merge_service.preview_merge(ctx, left["dataset_id"], right["dataset_id"], **params),
                component=PAGE,
            )
        _put("merge_preview", ids, params, result)
        _put("merge_analysis", ids, params, result)
    analysis = _pair_state("merge_analysis", ids)
    if not analysis:
        st.caption(
            "Choose the key columns and analyze the merge to see the exact number of output rows first."
        )
        return
    _merge_analysis_view(analysis["data"])
    preview = _pair_state("merge_preview", ids)
    if preview:
        st.markdown("**Sample of output rows (merged from a few whole keys; nothing saved)**")
        records_table(preview["data"]["sample_rows"], empty_message="No output rows for the sampled keys.")
    if analysis["params"] != params:
        st.info("The keys or options changed after this analysis. Analyze again before merging.")
        return
    if not caps.can(Permission.CLEAN_DATA):
        feature_unavailable("Merging needs the clean data permission.")
        return
    data = analysis["data"]
    output, new_name = _output_choice("merge", "new_dataset")
    st.caption("To cancel, simply do not apply: nothing has been saved yet.")
    ok = confirm(f"Merge into {data['output_rows']:,} rows as analyzed", key=f"{PAGE}_merge_confirm")
    many = True
    if data["many_to_many"]:
        many = confirm("I accept that many-to-many matching multiplies rows", key=f"{PAGE}_merge_confirm_m2m")
    if st.button("Merge and save", key=f"{PAGE}_merge_apply", disabled=not (ok and many)):
        saved = analysis["params"]
        progress = Progress("Merging")
        result = run_action(
            lambda: merge_service.apply_merge(
                ctx,
                left["dataset_id"],
                right["dataset_id"],
                **saved,
                output=output,
                new_dataset_name=new_name,
                confirm=ok,
                confirm_many_to_many=many and data["many_to_many"],
                guard=OperationGuard(progress=progress),
            ),
            component=PAGE,
        )
        progress.done()
        if result is not None:
            before = left.get("quality_score") if output == "new_version" else None
            _done(result, before, "Merged and saved")


if not caps.has(Feature.APPEND_MERGE):
    feature_unavailable("Appending and merging datasets is not included in your plan.")
    st.stop()
applied = page_state(PAGE, "applied")
if applied:
    result = applied["data"]
    if result.get("output") == "new_dataset":
        created = run_action(lambda: dataset_service.get_dataset(ctx, result["dataset_id"]), component=PAGE)
        name = md_escape(created["name"]) if created else "the new dataset"
        score = result.get("quality_score")
        with st.container(border=True):
            st.success(f"{applied['label']} as the new dataset {name}.", icon=":material/check_circle:")
            cols = st.columns(3)
            cols[0].metric("Rows", f"{result['rows']:,}")
            cols[1].metric("Columns", f"{result['columns']:,}")
            cols[2].metric("Quality score", "-" if score is None else f"{score:.1f}")
    else:
        applied_version_view(result, quality_before=applied["quality_before"], label=applied["label"])
    set_page_state(PAGE, "applied", None)
tabs = st.tabs(["Append rows", "Merge on keys"])
with tabs[0]:
    _append_tab()
with tabs[1]:
    _merge_tab()
