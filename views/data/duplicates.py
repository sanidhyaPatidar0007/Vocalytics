"""Duplicates: exact duplicate rows, repeated values, identifier checks, composite keys, id generation."""

from __future__ import annotations

from typing import Any

import streamlit as st

from components.data_preview import applied_version_view, records_table, with_labels
from components.dataset_picker import dataset_selector
from components.feedback import Progress, confirm, feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import page_state, require_ctx, set_page_state
from core.constants import Feature, Permission
from services import dataset_service, duplicate_manager
from services.resource_manager import OperationGuard

PAGE = "duplicates"
KEEP_LABELS = {
    "first": "Keep the first row of each group",
    "last": "Keep the last row of each group",
    "none": "Remove every row of each duplicate group",
}
POLICY_LABELS = {
    "keep_first": "Keep the first row with each key",
    "keep_last": "Keep the last row with each key",
    "keep_most_complete": "Keep the row with the fewest missing values",
    "drop_all": "Remove every row that shares a key",
}
RULE_LABELS = {
    "none": "No normalization",
    "email": "E-mail address",
    "phone": "Phone number (digits only)",
    "identifier": "Identifier (ignore separators)",
}

ctx = require_ctx()
caps = capabilities()
page_header(
    "Duplicates", "Find duplicate rows, repeated values and identifier problems, then resolve them safely."
)


def _get(name: str, dataset_id: int, version_id: int | None) -> dict[str, Any] | None:
    stored = page_state(PAGE, name)
    if stored and stored["dataset_id"] == dataset_id and stored["version_id"] == version_id:
        return stored
    return None


def _put(name: str, dataset_id: int, version_id: int | None, params: Any, data: Any) -> None:
    set_page_state(
        PAGE,
        name,
        None
        if data is None
        else {"dataset_id": dataset_id, "version_id": version_id, "params": params, "data": data},
    )


def _applied(dataset_id: int, result: dict[str, Any]) -> None:
    for name in ("exact", "removal", "keys", "ids", "values", "ident"):
        set_page_state(PAGE, name, None)
    set_page_state(PAGE, "applied", {"dataset_id": dataset_id, "data": result})
    st.rerun()


def _can_clean() -> bool:
    if caps.can(Permission.CLEAN_DATA):
        return True
    feature_unavailable("Changing data needs the clean data permission.")
    return False


def _exact_tab(dataset_id: int, version_id: int | None, columns: list[str]) -> None:
    st.caption(
        "Exact duplicates are rows that are identical in every compared "
        "column. Detection is exact across the whole dataset."
    )
    if st.button("Find exact duplicates", key=f"{PAGE}_exact_find", type="primary"):
        progress = Progress("Scanning rows")
        result = run_action(
            lambda: duplicate_manager.find_exact_duplicates(
                ctx, dataset_id, guard=OperationGuard(progress=progress)
            ),
            component=PAGE,
        )
        progress.done()
        _put("exact", dataset_id, version_id, None, result)
    found = _get("exact", dataset_id, version_id)
    if found:
        result = found["data"]
        cols = st.columns(4)
        cols[0].metric("Duplicate rows", f"{result['duplicate_rows']:,}")
        cols[1].metric("Share of rows", f"{result['pct']:.2f}%")
        cols[2].metric("Duplicate groups", f"{result['group_count']:,}")
        cols[3].metric("Rows in groups", f"{result['rows_in_groups']:,}")
        st.caption(md_escape(result["method"]))
        if result["groups"]:
            records_table(
                [
                    with_labels(
                        {"Rows (0-based)": ", ".join(map(str, g["rows"])), "Copies": g["size"]}, g["values"]
                    )
                    for g in result["groups"]
                ]
            )
            if result["groups_truncated"]:
                st.caption("Only the first groups are shown.")
        else:
            st.success("No exact duplicate rows were found.")

    st.markdown("**Remove duplicate rows**")
    keep = st.radio(
        "Which rows to keep", list(KEEP_LABELS), format_func=KEEP_LABELS.get, key=f"{PAGE}_keep_{dataset_id}"
    )
    subset = st.multiselect(
        "Compare only these columns (leave empty to compare every column)",
        columns,
        key=f"{PAGE}_subset_{dataset_id}",
    )
    params = {"keep": keep, "subset": list(subset) or None}
    if st.button("Preview removal", key=f"{PAGE}_removal_preview"):
        result = run_action(
            lambda: duplicate_manager.preview_remove_duplicates(ctx, dataset_id, **params), component=PAGE
        )
        _put("removal", dataset_id, version_id, params, result)
    stored = _get("removal", dataset_id, version_id)
    if not stored:
        return
    preview = stored["data"]
    with st.container(border=True):
        cols = st.columns(3)
        cols[0].metric("Rows before", f"{preview['rows_before']:,}")
        cols[1].metric("Rows after", f"{preview['rows_after']:,}")
        cols[2].metric("Rows removed", f"{preview['rows_removed']:,}")
        st.caption("Sample of rows that would be removed (row numbers start at 0)")
        records_table(
            [with_labels({"row": s["row_index"]}, s["values"]) for s in preview["sample_removed"]],
            empty_message="No rows would be removed.",
        )
    if stored["params"] != params:
        st.info("The options changed after this preview. Preview again before removing rows.")
        return
    if not preview["rows_removed"] or not _can_clean():
        return
    ok = confirm(
        f"Remove {preview['rows_removed']:,} row(s) in a new version",
        key=f"{PAGE}_remove_confirm_{dataset_id}",
    )
    if st.button("Remove duplicates and save new version", key=f"{PAGE}_remove", disabled=not ok):
        saved = stored["params"]
        result = run_action(
            lambda: duplicate_manager.remove_duplicates(ctx, dataset_id, confirm=ok, **saved), component=PAGE
        )
        if result is not None:
            _applied(dataset_id, result)


def _values_tab(dataset_id: int, version_id: int | None, columns: list[str]) -> None:
    st.info(md_escape(duplicate_manager.REPEATED_VALUES_NOTE), icon=":material/info:")
    picked = st.multiselect("Columns", columns, key=f"{PAGE}_values_cols_{dataset_id}")
    c1, c2, c3 = st.columns(3)
    trim = c1.checkbox(
        "Ignore spaces at the start and end", value=True, key=f"{PAGE}_values_trim_{dataset_id}"
    )
    ci = c2.checkbox("Ignore letter case", value=True, key=f"{PAGE}_values_ci_{dataset_id}")
    top = c3.number_input("Values to show per column", 1, 1000, 100, key=f"{PAGE}_values_top_{dataset_id}")
    rules = {}
    for index, name in enumerate(picked):
        rule = st.selectbox(
            f"Comparison rule for {md_escape(name)}",
            list(RULE_LABELS),
            format_func=RULE_LABELS.get,
            key=f"{PAGE}_values_rule_{dataset_id}_{index}_{name}",
        )
        if rule != "none":
            rules[name] = rule
    if st.button("Find repeated values", key=f"{PAGE}_values_find", type="primary"):
        result = run_action(
            lambda: duplicate_manager.find_value_duplicates(
                ctx,
                dataset_id,
                list(picked),
                trim=trim,
                case_insensitive=ci,
                normalization=rules or None,
                top=int(top),
            ),
            component=PAGE,
        )
        _put("values", dataset_id, version_id, None, result)
    stored = _get("values", dataset_id, version_id)
    if not stored:
        return
    result = stored["data"]
    st.caption(md_escape(result["note"]))
    for column in result["columns"]:
        with st.container(border=True):
            st.markdown(
                f"**{md_escape(column['column'])}** ({md_escape(RULE_LABELS.get(column['rule'], column['rule']))})"
            )
            cols = st.columns(4)
            cols[0].metric("Distinct values", f"{column['distinct_values']:,}")
            cols[1].metric("Repeated values", f"{column['repeated_value_count']:,}")
            cols[2].metric("Rows with a repeated value", f"{column['rows_with_repeated_values']:,}")
            cols[3].metric("Missing", f"{column['missing']:,}")
            records_table(
                [
                    {
                        "Value (as compared)": v["value"],
                        "Rows": v["count"],
                        "Example rows (0-based)": ", ".join(map(str, v["example_rows"])),
                        "Spellings in the data": ", ".join(map(str, v["original_values"])),
                    }
                    for v in column["values"]
                ],
                empty_message="No value repeats in this column.",
            )


def _identifier_tab(dataset_id: int, version_id: int | None, columns: list[str]) -> None:
    st.caption(
        "Check whether a column is a valid unique identifier: no repeated values and no missing values."
    )
    column = st.selectbox("Identifier column", columns, key=f"{PAGE}_ident_col_{dataset_id}")
    c1, c2 = st.columns(2)
    trim = c1.checkbox(
        "Ignore spaces at the start and end", value=True, key=f"{PAGE}_ident_trim_{dataset_id}"
    )
    ci = c2.checkbox("Ignore letter case", value=False, key=f"{PAGE}_ident_ci_{dataset_id}")
    if st.button("Validate identifier", key=f"{PAGE}_ident_run", type="primary"):
        result = run_action(
            lambda: duplicate_manager.validate_identifier(
                ctx, dataset_id, column, trim=trim, case_insensitive=ci
            ),
            component=PAGE,
        )
        _put("ident", dataset_id, version_id, None, result)
    stored = _get("ident", dataset_id, version_id)
    if not stored:
        return
    result = stored["data"]
    if result["valid_identifier"]:
        st.success(f"{md_escape(result['column'])} is a valid identifier: every row has a unique value.")
    else:
        st.warning(f"{md_escape(result['column'])} is not a valid unique identifier.")
    cols = st.columns(4)
    cols[0].metric("Distinct values", f"{result['distinct_values']:,}")
    cols[1].metric("Repeated values", f"{result['duplicate_value_count']:,}")
    cols[2].metric("Rows involved", f"{result['rows_involved']:,}")
    cols[3].metric("Missing", f"{result['missing_count']:,}")
    st.caption(md_escape(result["explanation"]))
    records_table(
        [
            {
                "Value": v["value"],
                "Rows": v["count"],
                "Example rows (0-based)": ", ".join(map(str, v["example_rows"])),
            }
            for v in result["examples"]
        ],
        empty_message="No repeated values.",
    )


def _keys_tab(dataset_id: int, version_id: int | None, columns: list[str]) -> None:
    st.caption(
        "Rows that share a business key (for example customer_id and transaction_date) are conflicts when their other "
        "values differ. Resolving them needs an explicit policy."
    )
    keys = st.multiselect("Key columns", columns, key=f"{PAGE}_keys_cols_{dataset_id}")
    if st.button("Find key conflicts", key=f"{PAGE}_keys_find", type="primary"):
        result = run_action(
            lambda: duplicate_manager.find_composite_key_conflicts(ctx, dataset_id, list(keys)),
            component=PAGE,
        )
        _put("keys", dataset_id, version_id, list(keys), result)
    stored = _get("keys", dataset_id, version_id)
    if not stored:
        return
    result = stored["data"]
    cols = st.columns(4)
    cols[0].metric("Groups sharing a key", f"{result['group_count']:,}")
    cols[1].metric("Identical groups", f"{result['identical_groups']:,}")
    cols[2].metric("Conflicting groups", f"{result['conflicting_groups']:,}")
    cols[3].metric("Rows with a missing key part", f"{result['rows_with_missing_key']:,}")
    if not result["groups"]:
        st.success("Every key value is unique.")
        return
    records_table(
        [
            {
                "Key": ", ".join(f"{k}={v}" for k, v in g["key"].items()),
                "Rows (0-based)": ", ".join(map(str, g["rows"])),
                "Rows": g["size"],
                "Identical": g["identical"],
                "Columns that differ": ", ".join(g["differing_columns"]),
            }
            for g in result["groups"]
        ]
    )
    with st.expander("Conflicting records"):
        for group in result["groups"][:20]:
            records_table(group["records"])
    if stored["params"] != list(keys):
        st.info("The key columns changed. Find conflicts again before resolving.")
        return
    if not _can_clean():
        return
    policy = st.selectbox(
        "Resolution policy",
        list(POLICY_LABELS),
        index=None,
        placeholder="Choose how to resolve the conflicts",
        format_func=POLICY_LABELS.get,
        key=f"{PAGE}_keys_policy_{dataset_id}",
    )
    ok = confirm(
        "Remove the rows this policy discards in a new version", key=f"{PAGE}_keys_confirm_{dataset_id}"
    )
    if st.button("Resolve and save new version", key=f"{PAGE}_keys_resolve", disabled=not (ok and policy)):
        saved = stored["params"]
        outcome = run_action(
            lambda: duplicate_manager.resolve_composite_key_duplicates(
                ctx, dataset_id, saved, policy=policy, confirm=ok
            ),
            component=PAGE,
        )
        if outcome is not None:
            _applied(dataset_id, outcome)


def _ids_tab(dataset_id: int, version_id: int | None, columns: list[str]) -> None:
    if not caps.has(Feature.ADVANCED_CLEANING):
        feature_unavailable(
            "Generating identifiers is part of advanced cleaning, which your plan does not include."
        )
        return
    st.caption(
        "Fill missing or repeated identifiers, or add a new identifier column. Existing unique identifiers are never "
        "rewritten, generated values never collide, and the old to new mapping is kept in the history."
    )
    column = st.selectbox("Identifier column", columns, key=f"{PAGE}_ids_col_{dataset_id}")
    c1, c2 = st.columns(2)
    target = c1.radio(
        "What to generate",
        list(duplicate_manager.ID_TARGETS),
        format_func=lambda t: (
            "Fill missing and repeated values in this column"
            if t == "fill_missing_and_duplicates"
            else "A new identifier column"
        ),
        key=f"{PAGE}_ids_target_{dataset_id}",
    )
    mode = c2.radio(
        "Identifier style",
        list(duplicate_manager.ID_MODES),
        format_func=lambda m: "Random UUID" if m == "uuid4" else "Sequential number",
        key=f"{PAGE}_ids_mode_{dataset_id}",
    )
    c3, c4, c5 = st.columns(3)
    prefix = c3.text_input("Prefix (optional)", key=f"{PAGE}_ids_prefix_{dataset_id}", max_chars=50)
    width = c4.number_input("Digits for sequential numbers", 1, 20, 4, key=f"{PAGE}_ids_width_{dataset_id}")
    new_column = None
    if target == "new_column":
        new_column = c5.text_input("New column name", key=f"{PAGE}_ids_new_{dataset_id}", max_chars=128)
    params = {"mode": mode, "prefix": prefix, "target": target, "new_column": new_column, "width": int(width)}
    if st.button("Preview identifiers", key=f"{PAGE}_ids_preview", type="primary"):
        result = run_action(
            lambda: duplicate_manager.preview_generate_ids(ctx, dataset_id, column, **params), component=PAGE
        )
        _put("ids", dataset_id, version_id, {"column": column, **params}, result)
    stored = _get("ids", dataset_id, version_id)
    if not stored:
        return
    preview = stored["data"]
    cols = st.columns(3)
    cols[0].metric("Identifiers to generate", f"{preview['ids_to_generate']:,}")
    cols[1].metric("Missing identifiers", f"{preview['missing_ids']:,}")
    cols[2].metric("Repeated identifiers", f"{preview['repeated_ids']:,}")
    st.caption(md_escape(preview["note"]))
    records_table(
        [
            {"Row (0-based)": m["row_index"], "Old value": m["old_value"], "New value": m["new_value"]}
            for m in preview["mapping_sample"]
        ],
        empty_message="Nothing to generate.",
    )
    if stored["params"] != {"column": column, **params}:
        st.info("The options changed after this preview. Preview again before generating.")
        return
    if not preview["ids_to_generate"] or not _can_clean():
        return
    ok = confirm(
        f"Write {preview['ids_to_generate']:,} generated identifier(s) in a new version",
        key=f"{PAGE}_ids_confirm_{dataset_id}",
    )
    if st.button("Generate identifiers and save new version", key=f"{PAGE}_ids_apply", disabled=not ok):
        saved = dict(stored["params"])
        saved_column = saved.pop("column")
        outcome = run_action(
            lambda: duplicate_manager.generate_ids(ctx, dataset_id, saved_column, confirm=ok, **saved),
            component=PAGE,
        )
        if outcome is not None:
            set_page_state(PAGE, "mapping", {"dataset_id": dataset_id, "data": outcome})
            _applied(dataset_id, outcome)


selected = dataset_selector(ctx, key=f"{PAGE}_select")
if selected is not None:
    if not caps.has(Feature.DUPLICATE_DETECTION):
        feature_unavailable("Your plan does not include duplicate detection.")
        st.stop()
    dataset_id = selected["dataset_id"]
    version_id = selected.get("current_version_id")
    applied = page_state(PAGE, "applied")
    if applied and applied["dataset_id"] == dataset_id:
        applied_version_view(applied["data"])
        set_page_state(PAGE, "applied", None)
    mapping = page_state(PAGE, "mapping")
    if mapping and mapping["dataset_id"] == dataset_id:
        with st.expander("Old to new identifier mapping", expanded=True):
            data = mapping["data"]
            records_table(
                [
                    {
                        "Row (0-based)": m["row_index"],
                        "Old value": m["old_value"],
                        "New value": m["new_value"],
                    }
                    for m in data["mapping"]
                ]
            )
            if data.get("mapping_truncated"):
                st.caption(
                    f"Only the first {duplicate_manager.MAPPING_CAP:,} changed identifiers are listed here and "
                    "in the operation history. The previous version still holds the original values."
                )
        set_page_state(PAGE, "mapping", None)
    info = run_action(lambda: dataset_service.get_dataset(ctx, dataset_id), component=PAGE)
    names = [c["name"] for c in (info or {}).get("schema") or []]
    tabs = st.tabs(
        ["Exact duplicates", "Repeated values", "Identifier check", "Composite keys", "Generate identifiers"]
    )
    with tabs[0]:
        _exact_tab(dataset_id, version_id, names)
    with tabs[1]:
        _values_tab(dataset_id, version_id, names)
    with tabs[2]:
        _identifier_tab(dataset_id, version_id, names)
    with tabs[3]:
        _keys_tab(dataset_id, version_id, names)
    with tabs[4]:
        _ids_tab(dataset_id, version_id, names)
