"""Missing values: per-column analysis, recommended and manual treatment plans, placeholders."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.data_preview import applied_version_view, before_after_table, records_table
from components.dataset_picker import dataset_selector
from components.feedback import Progress, confirm, feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import page_state, require_ctx, set_page_state
from core.constants import Feature, Permission
from services import missing_values
from services.resource_manager import OperationGuard

PAGE = "missing"

STRATEGY_LABELS = {
    "mean": "Mean (average)",
    "median": "Median (middle value)",
    "mode": "Most frequent value",
    "constant": "A value you enter",
    "date_value": "A date you enter",
    "placeholder": "The organization's placeholder text",
    "ffill": "Copy the previous row's value",
    "bfill": "Copy the next row's value",
    "leave": "Leave unchanged",
    "drop_rows": "Drop rows missing this value",
    "drop_column": "Drop the whole column",
}
FLAG_LABELS = {
    "identifier": "identifier",
    "sensitive": "sensitive",
    "target_label": "target or label",
    "all_null": "all missing",
    "high_missing": "50% or more missing",
    "single_non_null": "one value only",
    "small_dataset": "small dataset",
    "mixed_types": "mixed numbers and text",
}

ctx = require_ctx()
caps = capabilities()
page_header(
    "Missing values",
    "See what is missing in each column, preview a treatment plan, and save it as a new version.",
)


def _key(dataset_id: int, version_id: int | None, kind: str, column: str) -> str:
    """Editor widget key tied to the dataset, its version and the column it describes.

    A new version has a different column list, so a plan made for one version never
    lands on a different column of the next.
    """
    return f"{PAGE}_{kind}_{dataset_id}_{version_id}_{column}"


def _analysis(dataset_id: int, version_id: int | None) -> dict[str, Any] | None:
    stored = page_state(PAGE, "analysis")
    if stored and stored["dataset_id"] == dataset_id and stored["version_id"] == version_id:
        return stored["data"]
    result = run_action(lambda: missing_values.analyze_missing(ctx, dataset_id), component=PAGE)
    if result is not None:
        set_page_state(PAGE, "analysis", {"dataset_id": dataset_id, "version_id": version_id, "data": result})
    return result


def _analysis_view(analysis: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Missing cells", f"{analysis['total_missing_cells']:,}")
    cols[1].metric("Missing share", f"{analysis['missing_pct']:.2f}%")
    cols[2].metric("Rows with a missing value", f"{analysis['rows_with_missing']:,}")
    cols[3].metric("Rows", f"{analysis['rows']:,}")
    tokens = ", ".join(md_escape(t) for t in analysis.get("placeholders") or []) or "none"
    st.caption(
        f"Placeholders counted as missing in text columns: {tokens}. "
        + " ".join(md_escape(n) for n in analysis["notes"])
    )
    rows = []
    for c in analysis["columns"]:
        rows.append(
            {
                "Column": c["name"],
                "Type": c["logical_type"],
                "Kind": c["semantic"],
                "Missing": c["missing_count"],
                "Missing %": round(c["missing_pct"], 2),
                "Recommended": STRATEGY_LABELS.get(c["recommended_strategy"], c["recommended_strategy"]),
                "Reason": c["reason"],
                "Expected cells changed": c["expected_impact"]["cells_changed"],
                "Flags": ", ".join(FLAG_LABELS.get(k, k) for k, v in c["flags"].items() if v),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _fill_recommended(dataset_id: int, version_id: int | None, columns: list[dict[str, Any]]) -> None:
    """Button callback: put the recommended plan into the editor widgets (before they render)."""
    plan = run_action(lambda: missing_values.recommend_plan(ctx, dataset_id), component=PAGE)
    if plan is None:
        return
    wanted = {entry["column"]: entry["strategy"] for entry in plan}
    for column in columns:
        st.session_state[_key(dataset_id, version_id, "strategy", column["name"])] = wanted.get(
            column["name"], "leave"
        )
    st.session_state[f"{PAGE}_flash_plan"] = len(plan)


def _plan_editor(
    dataset_id: int, version_id: int | None, analysis: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    columns = analysis["columns"]
    st.subheader("Treatment plan")
    st.button(
        "Use the recommended plan",
        key=f"{PAGE}_recommend_{dataset_id}",
        on_click=_fill_recommended,
        args=(dataset_id, version_id, columns),
        icon=":material/auto_fix_high:",
    )
    filled = st.session_state.pop(f"{PAGE}_flash_plan", None)
    if filled is not None:
        st.success(f"The recommended plan was filled in for {filled} column(s). Review it, then preview.")
    show_all = st.toggle("Include columns without missing values", key=f"{PAGE}_show_all_{dataset_id}")
    plans: list[dict[str, Any]] = []
    shown = 0
    for column in columns:
        if not show_all and column["missing_count"] == 0:
            continue
        shown += 1
        allowed = list(missing_values.STRATEGIES[missing_values.type_group(column["logical_type"])])
        c1, c2, c3 = st.columns([2, 2, 2])
        c1.text(f"{column['name']}\n{column['logical_type']}, {column['missing_count']:,} missing")
        strategy_key = _key(dataset_id, version_id, "strategy", column["name"])
        st.session_state.setdefault(strategy_key, "leave")
        strategy = c2.selectbox(
            "Strategy",
            allowed,
            format_func=lambda s: STRATEGY_LABELS.get(s, s),
            key=strategy_key,
        )
        value = None
        if strategy in missing_values.VALUE_STRATEGIES:
            help_text = (
                "A date such as 2024-01-31" if strategy == "date_value" else "Must fit the column type"
            )
            value = c3.text_input(
                "Value", key=_key(dataset_id, version_id, "value", column["name"]), help=help_text
            )
        if strategy != "leave":
            plans.append({"column": column["name"], "strategy": strategy, "value": value})
    if not shown:
        st.info(
            "No column has missing values. Turn on 'Include columns without missing values' to drop a column."
        )
    st.markdown("**Options**")
    o1, o2 = st.columns(2)
    order = o1.checkbox(
        "Row order is meaningful (for example a time series)",
        key=f"{PAGE}_order_{dataset_id}_{version_id}",
    )
    sensitive = o1.checkbox(
        "Allow changes to sensitive columns", key=f"{PAGE}_sensitive_{dataset_id}_{version_id}"
    )
    target = o1.checkbox(
        "Allow changes to target or label columns", key=f"{PAGE}_target_{dataset_id}_{version_id}"
    )
    drop_columns = o2.checkbox("Confirm dropping columns", key=f"{PAGE}_dropcols_{dataset_id}_{version_id}")
    use_filter = o2.checkbox(
        "Compute fill statistics only from some rows (prevents leakage)",
        key=f"{PAGE}_fit_{dataset_id}_{version_id}",
        help="For example only the training split, so statistics do not leak from test rows.",
    )
    fit_filter = None
    if use_filter:
        names = [c["name"] for c in columns]
        fit_column = o2.selectbox("Split column", names, key=f"{PAGE}_fit_col_{dataset_id}_{version_id}")
        fit_values = o2.text_input(
            "Values to fit on (comma separated)", key=f"{PAGE}_fit_values_{dataset_id}_{version_id}"
        )
        fit_filter = {"column": fit_column, "values": [v.strip() for v in fit_values.split(",") if v.strip()]}
    options = {
        "row_order_meaningful": order,
        "fit_filter": fit_filter,
        "allow_sensitive": sensitive,
        "allow_target": target,
        "confirm_drop_columns": drop_columns,
    }
    return plans, options


def _preview_view(preview: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**Preview (nothing has been saved)**")
        cols = st.columns(5)
        cols[0].metric("Rows before", f"{preview['rows_before']:,}")
        cols[1].metric("Rows after", f"{preview['rows_after']:,}")
        cols[2].metric("Cells filled", f"{preview['cells_changed']:,}")
        cols[3].metric("Rows changed", f"{preview['rows_affected']:,}")
        cols[4].metric("Rows removed", f"{preview['rows_removed']:,}")
        table = [
            {
                "Column": c["column"],
                "Strategy": STRATEGY_LABELS.get(c["strategy"], c["strategy"]),
                "Missing before": c["missing_before"],
                "Missing after": c["missing_after"],
                "Fill value": "" if c["fill_value"] is None else str(c["fill_value"]),
                "Cells changed": c["cells_changed"],
                "Rows used for the statistic": c["fit_rows"],
            }
            for c in preview["columns"]
        ]
        records_table(table)
        if preview["columns_removed"]:
            st.warning(
                "Columns that will be removed: " + ", ".join(md_escape(c) for c in preview["columns_removed"])
            )
        for warning in preview["warnings"]:
            st.warning(md_escape(warning))
        st.caption("Sample of changed or removed rows (row numbers start at 0)")
        before_after_table(preview["sample"], row_key="row_index", empty_message="No rows change.")


def _treatment(selected: dict[str, Any], analysis: dict[str, Any]) -> None:
    dataset_id = selected["dataset_id"]
    version_id = selected.get("current_version_id")
    plans, options = _plan_editor(dataset_id, version_id, analysis)
    params = {"plans": plans, **options}
    if st.button("Preview changes", key=f"{PAGE}_preview_btn", type="primary"):
        preview = run_action(
            lambda: missing_values.preview_missing_treatment(ctx, dataset_id, plans, **options),
            component=PAGE,
        )
        set_page_state(
            PAGE,
            "preview",
            None
            if preview is None
            else {"dataset_id": dataset_id, "version_id": version_id, "params": params, "data": preview},
        )
    stored = page_state(PAGE, "preview")
    if not stored or stored["dataset_id"] != dataset_id or stored["version_id"] != version_id:
        st.caption("Preview the plan to see its exact effect before anything is saved.")
        return
    _preview_view(stored["data"])
    if stored["params"] != params:
        st.info("The plan or options changed after this preview. Preview again before applying.")
        return
    if not caps.can(Permission.CLEAN_DATA):
        feature_unavailable("Applying a treatment needs the clean data permission.")
        return
    preview = stored["data"]
    destructive = preview["rows_removed"] > 0 or bool(preview["columns_removed"])
    ok = True
    if destructive:
        ok = confirm(
            "I understand that rows or columns will be removed in the new version",
            key=f"{PAGE}_apply_confirm_{dataset_id}",
        )
    if st.button("Apply and save new version", key=f"{PAGE}_apply_btn", disabled=not ok):
        progress = Progress("Applying")
        saved = stored["params"]
        result = run_action(
            lambda: missing_values.apply_missing_treatment(
                ctx,
                dataset_id,
                saved["plans"],
                guard=OperationGuard(progress=progress),
                **{k: v for k, v in saved.items() if k != "plans"},
            ),
            component=PAGE,
        )
        progress.done()
        if result is not None:
            set_page_state(PAGE, "preview", None)
            set_page_state(PAGE, "applied", {"dataset_id": dataset_id, "data": result})
            st.rerun()


def _placeholders(selected: dict[str, Any], analysis: dict[str, Any]) -> None:
    dataset_id = selected["dataset_id"]
    version_id = selected.get("current_version_id")
    st.subheader("Placeholder values")
    st.caption(
        "Find text values such as N/A, empty strings or spaces that stand for missing data, and convert them to real "
        "missing values so every tool treats them the same way."
    )
    text = st.text_area(
        "Placeholders to look for (one per line)",
        value="\n".join(analysis.get("placeholders") or []),
        key=f"{PAGE}_tokens_{dataset_id}",
    )
    tokens = [line.strip() for line in text.splitlines() if line.strip()]
    if st.button("Detect placeholders", key=f"{PAGE}_detect"):
        result = run_action(
            lambda: missing_values.detect_placeholders(ctx, dataset_id, tokens), component=PAGE
        )
        set_page_state(
            PAGE,
            "detected",
            None
            if result is None
            else {"dataset_id": dataset_id, "version_id": version_id, "tokens": tokens, "data": result},
        )
    stored = page_state(PAGE, "detected")
    if not stored or stored["dataset_id"] != dataset_id or stored["version_id"] != version_id:
        return
    result = stored["data"]
    st.caption(md_escape(result["note"]))
    records_table(
        [
            {
                "Column": c["column"],
                "Empty text": c["empty"],
                "Spaces only": c["whitespace_only"],
                "Placeholder matches": ", ".join(f"{k}: {v}" for k, v in c["placeholder_matches"].items()),
                "Total to convert": c["total"],
                "Already missing": c["already_missing"],
            }
            for c in result["columns"]
        ],
        empty_message="This dataset has no text columns.",
    )
    if not result["total"]:
        st.success("No placeholder values were found.")
        return
    if stored["tokens"] != tokens:
        st.info("The placeholder list changed after detection. Detect again before converting.")
        return
    if not caps.can(Permission.CLEAN_DATA):
        feature_unavailable("Converting placeholders needs the clean data permission.")
        return
    if st.button(
        f"Convert {result['total']:,} value(s) to missing and save new version", key=f"{PAGE}_standardize"
    ):
        saved = stored["tokens"]
        outcome = run_action(
            lambda: missing_values.standardize_placeholders(ctx, dataset_id, saved), component=PAGE
        )
        if outcome is not None:
            set_page_state(PAGE, "detected", None)
            set_page_state(PAGE, "applied", {"dataset_id": dataset_id, "data": outcome})
            st.rerun()


selected = dataset_selector(ctx, key=f"{PAGE}_select")
if selected is not None:
    if not caps.has(Feature.MISSING_VALUES):
        feature_unavailable("Your plan does not include missing-value treatment.")
        st.stop()
    applied = page_state(PAGE, "applied")
    if applied and applied["dataset_id"] == selected["dataset_id"]:
        applied_version_view(applied["data"])
        set_page_state(PAGE, "applied", None)
    analysis = _analysis(selected["dataset_id"], selected.get("current_version_id"))
    if analysis is not None:
        _analysis_view(analysis)
        with st.container(border=True):
            _treatment(selected, analysis)
        with st.container(border=True):
            _placeholders(selected, analysis)
