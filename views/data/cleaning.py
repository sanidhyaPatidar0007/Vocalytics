"""Cleaning workspace: choose an operation, set its options, preview the exact effect, apply.

Also holds the read-only reports (letter-case variants, possibly redundant
columns, outliers) that never change data.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from components.data_preview import applied_version_view, before_after_table, records_table, with_labels
from components.dataset_picker import dataset_selector
from components.feedback import Progress, confirm, feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import page_state, require_ctx, set_page_state
from core.constants import Feature, Permission
from services import cleaning_service, dataset_service
from services.resource_manager import OperationGuard
from services.type_inference import DATE_FORMATS, ISO_FORMAT

PAGE = "cleaning"
TEXT, INTEGER, FLOAT, BOOLEAN, DATETIME = "text", "integer", "float", "boolean", "datetime"
NUMERIC = (INTEGER, FLOAT)
FAILURE_LABELS = {
    "abort": "Change nothing if any value fails",
    "set_missing": "Set values that fail to missing",
}
DAYFIRST = {"Organization default": None, "Day first (31/12/2024)": True, "Month first (12/31/2024)": False}

ctx = require_ctx()
caps = capabilities()
page_header(
    "Cleaning",
    "Choose a cleaning operation, preview exactly what it changes, then save the result as a new version.",
)


class FormError(ValueError):
    """An option the user typed could not be read (shown as a message, nothing is called)."""


def _k(op: str, field: str, dataset_id: int) -> str:
    return f"{PAGE}_{op}_{field}_{dataset_id}"


def _is_current(stored: dict[str, Any] | None, dataset_id: int, version_id: int | None) -> bool:
    """A read-only report is shown only for the dataset version it was computed on."""
    return bool(stored) and stored["dataset_id"] == dataset_id and stored.get("version_id") == version_id


def _of(types: dict[str, str], *wanted: str) -> list[str]:
    return [name for name, logical in types.items() if logical in wanted]


def _mapping(text: str, what: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if "=" not in line:
            raise FormError(f"Line {number} needs the form old=new for {what}.")
        old, new = line.split("=", 1)
        mapping[old.strip() if what == "column names" else old] = new.strip()
    if not mapping:
        raise FormError(f"Enter at least one old=new line for {what}.")
    return mapping


def _typed(value: str, logical: str, option: str) -> Any:
    text = value.strip()
    if logical in NUMERIC:
        try:
            number = float(text)
        except ValueError as exc:
            raise FormError(f"'{option}' must be a number.") from exc
        return int(number) if logical == INTEGER and number.is_integer() else number
    if logical == BOOLEAN:
        if text.lower() not in ("true", "false"):
            raise FormError(f"'{option}' must be true or false.")
        return text.lower() == "true"
    return text


def _text_columns_form(op: str, dataset_id: int, types: dict[str, str]) -> dict[str, Any]:
    picked = st.multiselect(
        "Text columns (leave empty for every text column)",
        _of(types, TEXT),
        key=_k(op, "columns", dataset_id),
    )
    return {"columns": list(picked)} if picked else {}


def _failure(op: str, dataset_id: int) -> str:
    return st.radio(
        "If a value cannot be converted",
        list(FAILURE_LABELS),
        format_func=FAILURE_LABELS.get,
        key=_k(op, "fail", dataset_id),
        horizontal=True,
    )


def _dayfirst(op: str, dataset_id: int) -> dict[str, Any]:
    choice = DAYFIRST[st.selectbox("Date order", list(DAYFIRST), key=_k(op, "dayfirst", dataset_id))]
    return {} if choice is None else {"dayfirst": choice}


def _constraint_rules(dataset_id: int, types: dict[str, str]) -> dict[str, Any]:
    rules: list[dict[str, Any]] = page_state(PAGE, f"rules_{dataset_id}", []) or []
    st.caption(
        "Add one or more rules, then preview to get the validation report. This report never changes data."
    )
    column = st.selectbox("Column", list(types), key=_k("rule", "column", dataset_id))
    logical = types.get(column, TEXT)
    c1, c2 = st.columns(2)
    not_null = c1.checkbox("Must not be missing", key=_k("rule", "notnull", dataset_id))
    unique = c2.checkbox("Must be unique", key=_k("rule", "unique", dataset_id))
    low = high = regex = ""
    if logical in NUMERIC or logical == DATETIME:
        hint = "2024-01-31" if logical == DATETIME else "a number"
        low = c1.text_input(f"Minimum ({hint}, optional)", key=_k("rule", "min", dataset_id))
        high = c2.text_input(f"Maximum ({hint}, optional)", key=_k("rule", "max", dataset_id))
    allowed = st.text_area("Allowed values (one per line, optional)", key=_k("rule", "allowed", dataset_id))
    if logical == TEXT:
        regex = st.text_input(
            "Pattern the whole value must match (regular expression, optional)",
            key=_k("rule", "regex", dataset_id),
        )
    b1, b2 = st.columns(2)
    if b1.button("Add rule", key=f"{PAGE}_rule_add"):
        try:
            rule: dict[str, Any] = {"column": column, "not_null": not_null, "unique": unique}
            if low.strip():
                rule["min"] = _typed(low, logical, "Minimum")
            if high.strip():
                rule["max"] = _typed(high, logical, "Maximum")
            values = [line for line in allowed.splitlines() if line.strip()]
            if values:
                rule["allowed_values"] = [_typed(v, logical, "Allowed values") for v in values]
            if regex.strip():
                rule["regex"] = regex
            rules = [*rules, rule]
            set_page_state(PAGE, f"rules_{dataset_id}", rules)
        except FormError as exc:
            st.error(md_escape(exc))
    if b2.button("Clear rules", key=f"{PAGE}_rule_clear", disabled=not rules):
        rules = []
        set_page_state(PAGE, f"rules_{dataset_id}", rules)
    if rules:
        records_table(rules)
    else:
        st.info("No rules yet. Choose a column and its checks, then add the rule.")
    if not rules:
        raise FormError("Add at least one rule before previewing.")
    return {"rules": rules}


def _params_form(op: str, dataset_id: int, types: dict[str, str]) -> dict[str, Any]:
    """Render the option widgets for ``op`` and return the service parameters."""
    if op in ("trim_whitespace", "remove_control_chars"):
        return _text_columns_form(op, dataset_id, types)
    if op == "normalize_case":
        params = _text_columns_form(op, dataset_id, types)
        params["mode"] = st.radio(
            "Letter case", ["lower", "upper", "title"], key=_k(op, "mode", dataset_id), horizontal=True
        )
        return params
    if op == "rename_columns":
        st.caption("Current columns: " + ", ".join(md_escape(n) for n in types))
        text = st.text_area("Renames, one per line as old=new", key=_k(op, "mapping", dataset_id))
        return {"mapping": _mapping(text, "column names")}
    if op == "remove_empty":
        c1, c2 = st.columns(2)
        return {
            "rows": c1.checkbox("Remove fully empty rows", value=True, key=_k(op, "rows", dataset_id)),
            "columns": c2.checkbox("Remove fully empty columns", value=True, key=_k(op, "cols", dataset_id)),
        }
    if op == "drop_columns":
        return {
            "columns": list(
                st.multiselect("Columns to remove", list(types), key=_k(op, "columns", dataset_id))
            )
        }
    if op == "map_categories":
        column = st.selectbox("Text column", _of(types, TEXT), key=_k(op, "column", dataset_id))
        text = st.text_area(
            "Mappings, one per line as old value=new value", key=_k(op, "mapping", dataset_id)
        )
        ci = st.checkbox("Match old values ignoring letter case", key=_k(op, "ci", dataset_id))
        return {"column": column, "mapping": _mapping(text, "category labels"), "case_insensitive": ci}
    if op == "standardize_dates":
        column = st.selectbox("Text column with dates", _of(types, TEXT), key=_k(op, "column", dataset_id))
        formats = ["auto", ISO_FORMAT, *DATE_FORMATS, "custom"]
        fmt = st.selectbox(
            "Input date format",
            formats,
            format_func=lambda f: {"auto": "Detect automatically", "custom": "Custom format"}.get(f, f),
            key=_k(op, "format", dataset_id),
        )
        if fmt == "custom":
            fmt = st.text_input("Custom format, for example %d.%m.%Y", key=_k(op, "custom", dataset_id))
        return {
            "column": column,
            "input_format": fmt,
            **_dayfirst(op, dataset_id),
            "on_failure": _failure(op, dataset_id),
        }
    if op == "convert_numeric":
        column = st.selectbox("Text column with numbers", _of(types, TEXT), key=_k(op, "column", dataset_id))
        c1, c2 = st.columns(2)
        decimal = c1.selectbox("Decimal separator", [".", ","], key=_k(op, "decimal", dataset_id))
        thousands_options = {
            "None": None,
            "Comma (,)": ",",
            "Period (.)": ".",
            "Space": " ",
            "Apostrophe (')": "'",
        }
        thousands = thousands_options[
            c2.selectbox("Thousands separator", list(thousands_options), key=_k(op, "thousands", dataset_id))
        ]
        c3, c4, c5 = st.columns(3)
        return {
            "column": column,
            "decimal_separator": decimal,
            "thousands_separator": thousands,
            "strip_currency": c3.checkbox("Remove currency symbols", key=_k(op, "currency", dataset_id)),
            "strip_percent": c4.checkbox("Remove percent signs", key=_k(op, "percent", dataset_id)),
            "percent_as_fraction": c5.checkbox("Store 25% as 0.25", key=_k(op, "fraction", dataset_id)),
            "on_failure": _failure(op, dataset_id),
        }
    if op == "convert_types":
        column = st.selectbox("Column", list(types), key=_k(op, "column", dataset_id))
        targets = [t for t in (INTEGER, FLOAT, BOOLEAN, DATETIME, TEXT) if t != types.get(column)]
        target = st.selectbox(
            f"Convert from {types.get(column)} to", targets, key=_k(op, "target", dataset_id)
        )
        params = {"column": column, "target": target, "on_failure": _failure(op, dataset_id)}
        if types.get(column) == TEXT and target == DATETIME:
            params.update(_dayfirst(op, dataset_id))
        return params
    if op == "handle_outliers":
        column = st.selectbox("Number column", _of(types, *NUMERIC), key=_k(op, "column", dataset_id))
        c1, c2, c3 = st.columns(3)
        method = c1.radio(
            "Method",
            ["iqr", "zscore"],
            format_func=lambda m: "IQR fences" if m == "iqr" else "Z-score",
            key=_k(op, "method", dataset_id),
        )
        threshold = c2.number_input(
            "Threshold (k for IQR, z for z-score)",
            min_value=0.1,
            max_value=100.0,
            value=1.5 if method == "iqr" else 3.0,
            step=0.1,
            key=_k(op, f"threshold_{method}", dataset_id),
        )
        action = c3.selectbox(
            "Action",
            ["retain", "cap", "exclude", "flag"],
            format_func={
                "retain": "Report only (retain)",
                "cap": "Cap to the bounds",
                "exclude": "Remove the rows",
                "flag": "Add a flag column",
            }.get,
            key=_k(op, "action", dataset_id),
        )
        return {"column": column, "method": method, "threshold": float(threshold), "action": action}
    if op == "fix_invalid_values":
        column = st.selectbox("Text column", _of(types, TEXT), key=_k(op, "column", dataset_id))
        pattern = st.radio(
            "Validate as",
            ["email", "regex"],
            format_func=lambda p: "E-mail address" if p == "email" else "Custom pattern",
            key=_k(op, "pattern", dataset_id),
            horizontal=True,
        )
        params: dict[str, Any] = {"column": column, "pattern": pattern}
        if pattern == "regex":
            params["regex"] = st.text_input(
                "Pattern the whole value must match", key=_k(op, "regex", dataset_id)
            )
        params["action"] = st.radio(
            "Invalid values",
            ["report", "set_missing"],
            format_func=lambda a: "Report only" if a == "report" else "Set them to missing",
            key=_k(op, "action", dataset_id),
            horizontal=True,
        )
        return params
    if op == "validate_constraints":
        return _constraint_rules(dataset_id, types)
    raise FormError("This operation has no options form.")


def _preview_view(preview: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(
            f"**Preview of {md_escape(preview['label'])} on version {preview['version_number']} (nothing saved)**"
        )
        if "constraints" in preview:
            if preview["passed"]:
                st.success("Every rule passed.")
            else:
                st.warning(
                    f"{preview['failures']['count']:,} violation(s) in {preview['rows_with_violations']:,} row(s)."
                )
            records_table(
                [
                    {
                        "Column": rule["column"],
                        "Rule": json.dumps(rule["rule"], default=str, ensure_ascii=False),
                        "Violations": rule["violations"],
                        "Checks": ", ".join(f"{c['check']}: {c['count']}" for c in rule["checks"]),
                        "Passed": rule["passed"],
                    }
                    for rule in preview["constraints"]
                ]
            )
            if preview["failures"]["examples"]:
                st.caption("Examples (row numbers start at 1)")
                records_table(preview["failures"]["examples"])
            return
        cols = st.columns(6)
        cols[0].metric("Rows before", f"{preview['rows_before']:,}")
        cols[1].metric("Rows after", f"{preview['rows_after']:,}")
        cols[2].metric("Cells changed", f"{preview['cells_changed']:,}")
        cols[3].metric("Rows affected", f"{preview['rows_affected']:,}")
        cols[4].metric(
            "Quality before", "-" if preview["quality_before"] is None else f"{preview['quality_before']:.1f}"
        )
        delta = None
        if preview["quality_before"] is not None and preview["quality_after"] is not None:
            delta = round(preview["quality_after"] - preview["quality_before"], 1)
        cols[5].metric(
            "Quality after",
            "-" if preview["quality_after"] is None else f"{preview['quality_after']:.1f}",
            delta=delta,
        )
        changes = []
        if preview["columns_removed"]:
            changes.append("removed: " + ", ".join(md_escape(c) for c in preview["columns_removed"]))
        if preview["columns_added"]:
            changes.append("added: " + ", ".join(md_escape(c) for c in preview["columns_added"]))
        if preview["columns_renamed"]:
            changes.append(
                "renamed: "
                + ", ".join(
                    f"{md_escape(a)} to {md_escape(b)}" for a, b in preview["columns_renamed"].items()
                )
            )
        if changes:
            st.info("Columns " + "; ".join(changes))
        failures = preview["failures"]
        if failures["count"]:
            st.warning(
                f"{failures['count']:,} value(s) could not be converted or are "
                "invalid. Examples (row numbers start at 1):"
            )
            records_table(failures["examples"])
        for warning in preview["warnings"]:
            st.warning(md_escape(warning))
        if preview["counts"] or preview["details"]:
            with st.expander("Operation details"):
                st.code(
                    json.dumps(
                        {"counts": preview["counts"], "details": preview["details"]},
                        indent=2,
                        default=str,
                        ensure_ascii=False,
                    ),
                    language="json",
                )
        st.caption("Sample of affected rows (row numbers start at 1)")
        before_after_table(preview["sample"], empty_message="No rows change.")


def _workspace(selected: dict[str, Any], types: dict[str, str]) -> None:
    dataset_id = selected["dataset_id"]
    version_id = selected.get("current_version_id")
    operations = run_action(cleaning_service.list_operations, component=PAGE) or []
    if not operations:
        return
    meta = {o["name"]: o for o in operations}
    locked = {
        o["name"]
        for o in operations
        if o["feature"] == "advanced" and not caps.has(Feature.ADVANCED_CLEANING)
    }
    locked |= {
        o["name"]
        for o in operations
        if o["feature"] == "standard" and not caps.has(Feature.STANDARD_CLEANING)
    }
    op = st.selectbox(
        "Operation",
        list(meta),
        format_func=lambda n: (
            f"{meta[n]['label']} (locked: not in your plan)" if n in locked else meta[n]["label"]
        ),
        key=f"{PAGE}_operation",
    )
    info = meta[op]
    if op in locked:
        tier = "Advanced cleaning" if info["feature"] == "advanced" else "Standard cleaning"
        feature_unavailable(f"{tier} is not included in your plan, so this operation is locked.")
        return
    tags = []
    if info["read_only"]:
        tags.append("read-only report")
    if info["destructive"]:
        tags.append("removes data, needs confirmation")
    elif info["confirm_when"]:
        tags.append(f"needs confirmation when {info['confirm_when']}")
    if tags:
        st.caption("This operation: " + "; ".join(md_escape(t) for t in tags) + ".")
    try:
        params = _params_form(op, dataset_id, types)
        form_error = None
    except FormError as exc:
        params, form_error = None, str(exc)
    if st.button("Preview changes", key=f"{PAGE}_preview_btn", type="primary"):
        if form_error:
            st.error(md_escape(form_error))
        else:
            progress = Progress("Previewing")
            preview = run_action(
                lambda: cleaning_service.preview_cleaning(
                    ctx, dataset_id, op, params, guard=OperationGuard(progress=progress)
                ),
                component=PAGE,
            )
            progress.done()
            set_page_state(
                PAGE,
                "preview",
                None
                if preview is None
                else {
                    "dataset_id": dataset_id,
                    "version_id": version_id,
                    "op": op,
                    "params": params,
                    "data": preview,
                },
            )
    stored = page_state(PAGE, "preview")
    if (
        not stored
        or stored["dataset_id"] != dataset_id
        or stored["version_id"] != version_id
        or stored["op"] != op
    ):
        st.caption("Preview the operation to see its exact effect before anything is saved.")
        return
    preview = stored["data"]
    _preview_view(preview)
    if stored["params"] != params:
        st.info("The options changed after this preview. Preview again before applying.")
        return
    if not preview["can_apply"]:
        if "constraints" not in preview:
            st.info("This configuration cannot be applied as it is. Read the warnings above.")
        return
    if not caps.can(Permission.CLEAN_DATA):
        feature_unavailable("Applying cleaning needs the clean data permission.")
        return
    ok = True
    if preview["requires_confirmation"]:
        ok = confirm(
            "I understand that this removes or blanks data in the new version",
            key=f"{PAGE}_confirm_{dataset_id}_{op}",
        )
    if st.button("Apply and save new version", key=f"{PAGE}_apply_btn", disabled=not ok):
        progress = Progress("Applying")
        saved = stored["params"]
        result = run_action(
            lambda: cleaning_service.apply_cleaning(
                ctx, dataset_id, op, saved, confirm=ok, guard=OperationGuard(progress=progress)
            ),
            component=PAGE,
        )
        progress.done()
        if result is not None:
            set_page_state(PAGE, "preview", None)
            set_page_state(PAGE, "applied", {"dataset_id": dataset_id, "data": result})
            st.rerun()


def _reports(dataset_id: int, version_id: int | None, types: dict[str, str]) -> None:
    st.subheader("Reports for review")
    if not caps.has(Feature.STANDARD_CLEANING):
        feature_unavailable(
            "The cleaning reports are part of standard cleaning, which your plan does not include."
        )
        return
    st.caption("These reports never change data. Use them to decide which operation to run.")
    tabs = st.tabs(["Letter case", "Redundant columns", "Outliers"])
    with tabs[0]:
        if st.button("Find letter-case variants", key=f"{PAGE}_casing"):
            result = run_action(
                lambda: cleaning_service.detect_casing_issues(ctx, dataset_id), component=PAGE
            )
            set_page_state(
                PAGE,
                "casing",
                None
                if result is None
                else {"dataset_id": dataset_id, "version_id": version_id, "data": result},
            )
        stored = page_state(PAGE, "casing")
        if _is_current(stored, dataset_id, version_id):
            records_table(
                [
                    {
                        "Column": f["column"],
                        "Variants": ", ".join(f"{v['value']} ({v['count']})" for v in f.get("variants", []))
                        or f.get("reason", ""),
                        "Rows": f.get("total"),
                        "Suggested spelling": f.get("suggested"),
                        "Status": f.get("status", f["kind"]),
                    }
                    for f in stored["data"]
                ],
                empty_message="No values differ only in letter case.",
            )
    with tabs[1]:
        if st.button("Find possibly redundant columns", key=f"{PAGE}_redundant"):
            result = run_action(
                lambda: cleaning_service.identify_redundant_columns(ctx, dataset_id), component=PAGE
            )
            set_page_state(
                PAGE,
                "redundant",
                None
                if result is None
                else {"dataset_id": dataset_id, "version_id": version_id, "data": result},
            )
        stored = page_state(PAGE, "redundant")
        if _is_current(stored, dataset_id, version_id):
            records_table(
                [
                    {
                        "Finding": f["kind"].replace("_", " "),
                        "Columns": ", ".join(f["columns"]),
                        "Detail": f["detail"],
                        "Correlation": f.get("correlation"),
                        "Status": f["status"],
                    }
                    for f in stored["data"]
                ],
                empty_message="No redundant columns were found.",
            )
    with tabs[2]:
        numeric = _of(types, *NUMERIC)
        if not numeric:
            st.info("This dataset has no number columns.")
            return
        column = st.selectbox("Number column", numeric, key=f"{PAGE}_outlier_col_{dataset_id}")
        method = st.radio(
            "Method",
            ["iqr", "zscore"],
            format_func=lambda m: "IQR fences" if m == "iqr" else "Z-score",
            key=f"{PAGE}_outlier_method_{dataset_id}",
            horizontal=True,
        )
        if st.button("Show outlier report", key=f"{PAGE}_outliers"):
            result = run_action(
                lambda: cleaning_service.outlier_report(ctx, dataset_id, column, method=method),
                component=PAGE,
            )
            set_page_state(
                PAGE,
                "outliers",
                None
                if result is None
                else {"dataset_id": dataset_id, "version_id": version_id, "data": result},
            )
        stored = page_state(PAGE, "outliers")
        if _is_current(stored, dataset_id, version_id):
            result = stored["data"]
            cols = st.columns(3)
            cols[0].metric("Possible outliers", f"{result['count']:,}")
            cols[1].metric("Share of values", f"{result['percent']:.2f}%")
            cols[2].metric("Values checked", f"{result['non_missing']:,}")
            st.caption("Possible outliers are not errors: review them before changing anything.")
            st.code(json.dumps(result["bounds"], indent=2, default=str), language="json")
            records_table(
                [with_labels({"row": s["row"]}, s["values"]) for s in result["sample"]],
                empty_message="No outliers.",
            )


selected = dataset_selector(ctx, key=f"{PAGE}_select")
if selected is not None:
    if not (caps.has(Feature.STANDARD_CLEANING) or caps.has(Feature.ADVANCED_CLEANING)):
        feature_unavailable("Your plan does not include data cleaning.")
        st.stop()
    dataset_id = selected["dataset_id"]
    applied = page_state(PAGE, "applied")
    if applied and applied["dataset_id"] == dataset_id:
        applied_version_view(applied["data"])
        result = applied["data"]
        if result.get("failures", {}).get("count"):
            st.warning(f"{result['failures']['count']:,} value(s) failed and were set to missing.")
        set_page_state(PAGE, "applied", None)
    info = run_action(lambda: dataset_service.get_dataset(ctx, dataset_id), component=PAGE)
    types = {c["name"]: c["logical_type"] for c in (info or {}).get("schema") or []}
    with st.container(border=True):
        _workspace(selected, types)
    _reports(dataset_id, selected.get("current_version_id"), types)
