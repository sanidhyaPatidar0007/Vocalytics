"""Anomaly rules: create, edit, activate, delete and run checks on demand."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.feedback import (
    confirm,
    escape,
    feature_unavailable,
    md_escape,
    page_header,
    run_action,
    show_table,
)
from components.navigation import capabilities
from components.session import require_ctx
from config.settings import get_settings
from core.constants import Feature, LogicalType
from modules import organization_admin
from services import anomaly_service, dataset_service

FLASH_KEY = "alert_rules_flash"
LAST_RUN_KEY = "alert_rules_last_run"
ALL_COLUMNS = "(all columns)"
NUMERIC_TYPES = {LogicalType.INTEGER.value, LogicalType.FLOAT.value}
METHOD_LABELS = {
    "iqr": "Outliers by IQR",
    "zscore": "Outliers by z-score",
    "period_change": "Change between periods",
    "metric_drop": "Drop in a metric",
    "baseline_deviation": "Deviation from the recent baseline",
    "missing_increase": "Missing or invalid values increase",
}
METHOD_HELP = {
    "iqr": "Flags values outside the usual range of a numeric column (quartiles widened by the multiplier).",
    "zscore": "Flags values that are many standard deviations from the average of a numeric column.",
    "period_change": "Flags a day, week or month whose total changed by more than the percentage, up or down.",
    "metric_drop": "Flags a period whose total fell by more than the percentage (for example a revenue drop).",
    "baseline_deviation": "Compares the latest period with the previous six periods and flags a large deviation.",
    "missing_increase": "Flags columns whose share of missing or invalid values grew since the last version.",
}
THRESHOLD_LABELS = {
    "iqr": "IQR multiplier (1.5 is typical)",
    "zscore": "Z-score threshold (3.0 is typical)",
    "period_change": "Flag changes larger than (percent)",
    "metric_drop": "Flag drops larger than (percent, at most 100)",
    "baseline_deviation": "Flag deviations larger than (standard deviations)",
    "missing_increase": "Flag increases larger than (percentage points, at most 100)",
}
THRESHOLD_DEFAULTS = {
    "iqr": 1.5,
    "zscore": 3.0,
    "period_change": 50.0,
    "metric_drop": 30.0,
    "baseline_deviation": 3.0,
    "missing_increase": 5.0,
}
PERCENT_METHODS = {"metric_drop", "missing_increase"}
FREQUENCY_LABELS = {"D": "Day", "W": "Week", "M": "Month"}
SCHEDULE_LABELS = {
    "manual": "Manual only",
    "hourly": "Hourly (by the worker)",
    "daily": "Daily (by the worker)",
    "weekly": "Weekly (by the worker)",
}
SEVERITY_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
WORKER_NOTE = (
    "Recurring checks are not run by this page, because a browser session is not a scheduler. "
    "Run scripts/run_anomaly_checks.py with Windows Task Scheduler to check rules with an hourly, "
    "daily or weekly schedule. The buttons below run a check right now."
)
SCHEDULER_COMMAND = (
    'schtasks /Create /SC HOURLY /TN "VoiceDataAnalyst Alerts" '
    '/TR "<project folder>\\.venv\\Scripts\\python.exe <project folder>\\scripts\\run_anomaly_checks.py"'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flash(message: str) -> None:
    st.session_state[FLASH_KEY] = message
    st.rerun()


def _rule_label(rule: dict[str, Any]) -> str:
    return f"{rule['name']} ({'active' if rule['is_active'] else 'inactive'})"


def _index(options: list[Any], value: Any) -> int:
    return options.index(value) if value in options else 0


def _show_events(events: list[dict[str, Any]]) -> None:
    """New events are suspected issues for review: dashed amber edge, escaped text."""
    for event in events:
        st.markdown(
            f'<div class="vdas-issue-suspected"><strong>{escape(event["title"])}</strong><br>'
            f"Severity: {escape(event['severity'])} | Status: {escape(event['status'])} | "
            f"Period: {escape(event.get('period'))}</div>",
            unsafe_allow_html=True,
        )


def _show_run(result: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Findings", f"{result['findings']:,}")
    cols[1].metric("New events", f"{result['new_events']:,}")
    cols[2].metric("Duplicates suppressed", f"{result['duplicate_suppressed']:,}")
    cols[3].metric("Held back by cooldown", f"{result['cooldown_suppressed']:,}")
    st.caption(f"Checked dataset version {result['version_number']}.")
    _show_events(result["events"])
    if result["notifications"]:
        show_table(
            pd.DataFrame(
                {
                    "Recipient": [n["recipient"] for n in result["notifications"]],
                    "Outcome": [n["status"] for n in result["notifications"]],
                    "Detail": [n["error"] or "" for n in result["notifications"]],
                }
            )
        )
    else:
        st.caption(
            "No e-mail was attempted: there was nothing new to report, no recipients are set, "
            "the plan has no e-mail alerts, or the rule is in its cooldown."
        )


def _show_last_run() -> None:
    last = st.session_state.get(LAST_RUN_KEY)
    if not last:
        return
    with st.container(border=True):
        st.subheader(md_escape(last["title"]))
        for result in last["results"]:
            if result.get("status") == "failed":
                st.error(md_escape(f"{result['rule_name']}: {result['error']}"), icon=":material/error:")
                continue
            if last["show_names"]:
                st.markdown(f"**{md_escape(result['rule_name'])}**")
            _show_run(result)


def _recipients_from(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _rule_fields(
    prefix: str,
    schema: list[dict[str, Any]],
    method: str,
    aggregation: str,
    current: dict[str, Any] | None,
    *,
    email_enabled: bool,
    default_recipients: list[str],
) -> dict[str, Any]:
    """Method-specific form fields (call inside a form); returns the values entered."""
    types = {c["name"]: c["logical_type"] for c in schema}
    numeric = [n for n, t in types.items() if t in NUMERIC_TYPES]
    dates = [n for n, t in types.items() if t == LogicalType.DATETIME.value]
    same_method = current is not None and current["method"] == method
    current = current or {}
    values: dict[str, Any] = {
        "method": method,
        "aggregation": "sum",
        "date_column": None,
        "frequency": None,
        "metric_column": None,
    }
    values["name"] = st.text_input("Rule name", value=current.get("name", ""), key=f"{prefix}_name", max_chars=200)
    if method in anomaly_service.VALUE_METHODS:
        if not numeric:
            st.warning("This dataset has no numeric column, so this rule type cannot be used on it.")
        values["metric_column"] = st.selectbox(
            "Numeric column to check",
            numeric,
            index=_index(numeric, current.get("metric_column")),
            key=f"{prefix}_metric",
        )
    elif method in anomaly_service.PERIOD_METHODS:
        values["aggregation"] = aggregation
        metric_options = [n for n in types if n not in dates] if aggregation == "count" else numeric
        if not dates:
            st.warning("This dataset has no date column, so this rule type cannot be used on it.")
        if not metric_options:
            st.warning("This dataset has no numeric column to add up, so this rule type cannot be used on it.")
        values["date_column"] = st.selectbox(
            "Date column", dates, index=_index(dates, current.get("date_column")), key=f"{prefix}_date"
        )
        values["metric_column"] = st.selectbox(
            "Metric column",
            metric_options,
            index=_index(metric_options, current.get("metric_column")),
            key=f"{prefix}_metric",
        )
        frequencies = list(FREQUENCY_LABELS)
        values["frequency"] = st.selectbox(
            "Compare by",
            frequencies,
            format_func=FREQUENCY_LABELS.get,
            index=_index(frequencies, current.get("frequency")),
            key=f"{prefix}_frequency",
        )
    else:
        options = [ALL_COLUMNS, *types]
        chosen = st.selectbox(
            "Column to watch (optional)",
            options,
            index=_index(options, current.get("metric_column")),
            key=f"{prefix}_metric",
        )
        values["metric_column"] = None if chosen == ALL_COLUMNS else chosen
    limit = 100.0 if method in PERCENT_METHODS else 100000.0
    default = float(current["threshold"]) if same_method else THRESHOLD_DEFAULTS[method]
    values["threshold"] = float(
        st.number_input(
            THRESHOLD_LABELS[method],
            min_value=0.01,
            max_value=limit,
            value=min(default, limit),
            step=0.1,
            key=f"{prefix}_threshold",
        )
    )
    severities, schedules = list(SEVERITY_LABELS), list(SCHEDULE_LABELS)
    values["severity"] = st.selectbox(
        "Severity",
        severities,
        format_func=SEVERITY_LABELS.get,
        index=_index(severities, current.get("severity", "medium")),
        key=f"{prefix}_severity",
    )
    values["schedule"] = st.selectbox(
        "Schedule",
        schedules,
        format_func=SCHEDULE_LABELS.get,
        index=_index(schedules, current.get("schedule", "manual")),
        key=f"{prefix}_schedule",
        help="Non-manual schedules only run when the worker script is scheduled (see the note above).",
    )
    values["cooldown_minutes"] = int(
        st.number_input(
            "Cooldown between e-mails (minutes)",
            min_value=0,
            max_value=anomaly_service.MAX_COOLDOWN_MINUTES,
            value=int(current.get("cooldown_minutes", get_settings().alert_default_cooldown_minutes)),
            step=5,
            key=f"{prefix}_cooldown",
        )
    )
    recipients_default = current.get("recipients", default_recipients)
    recipients_text = st.text_area(
        "E-mail recipients (one address per line)",
        value="\n".join(recipients_default),
        key=f"{prefix}_recipients",
        disabled=not email_enabled,
        help="Prefilled from the organization's default alert recipients.",
    )
    if not email_enabled:
        st.caption(
            "E-mail alerts are not included in your plan, so recipients cannot be set. "
            "Events are still recorded."
        )
    values["recipients"] = _recipients_from(recipients_text) if email_enabled else None
    return values


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

ctx = require_ctx()
caps = capabilities()
page_header(
    "Anomaly rules",
    "Rules that look for unusual values and sudden changes in a dataset. "
    "A flag is a prompt to look, not a confirmed problem.",
)
if not ctx.is_org_admin:
    feature_unavailable("Only organization admins can manage anomaly rules.")
    st.stop()
if caps is None or not caps.has(Feature.ANOMALY_DETECTION):
    feature_unavailable("Anomaly detection is not included in your plan.")
    st.stop()

flash = st.session_state.pop(FLASH_KEY, None)
if flash:
    st.success(flash, icon=":material/check_circle:")
st.info(WORKER_NOTE, icon=":material/schedule:")
with st.expander("Windows Task Scheduler command"):
    st.code(SCHEDULER_COMMAND, language="powershell")

email_enabled = caps.has(Feature.EMAIL_ALERTS)
datasets = run_action(lambda: dataset_service.list_datasets(ctx), component="alert_rules") or []
rules = run_action(lambda: anomaly_service.list_rules(ctx), component="alert_rules") or []
org_settings = run_action(lambda: organization_admin.get_settings_view(ctx), component="alert_rules") or {}
default_recipients = list(org_settings.get("alert_default_recipients", []))

result_slot = st.container()
if st.button("Run all checks", key="alert_rules_run_all", disabled=not rules, help="Runs every active rule now."):
    with st.spinner("Checking every active rule..."):
        results = run_action(lambda: anomaly_service.run_all_rules(ctx), component="alert_rules")
    if results is not None:
        st.session_state[LAST_RUN_KEY] = {
            "title": "Results of running all checks" if results else "No active rules to check",
            "results": results,
            "show_names": True,
        }
        st.rerun()

rules_tab, create_tab, edit_tab = st.tabs(["Rules", "Create a rule", "Edit the selected rule"])
by_id = {r["id"]: r for r in rules}
selected_id = None

with rules_tab:
    if not rules:
        st.info("No rules yet. Open Create a rule to set up the first one.")
    else:
        show_table(
            pd.DataFrame(
                {
                    "Name": [r["name"] for r in rules],
                    "Dataset": [r["dataset_name"] for r in rules],
                    "Method": [METHOD_LABELS.get(r["method"], r["method"]) for r in rules],
                    "Column": [r["metric_column"] or "" for r in rules],
                    "Date column": [r["date_column"] or "" for r in rules],
                    "Compare by": [FREQUENCY_LABELS.get(r["frequency"], "") for r in rules],
                    "Threshold": [r["threshold"] for r in rules],
                    "Severity": [r["severity"] for r in rules],
                    "Schedule": [r["schedule"] for r in rules],
                    "Cooldown (min)": [r["cooldown_minutes"] for r in rules],
                    "Active": ["yes" if r["is_active"] else "no" for r in rules],
                    "Recipients": [len(r["recipients"]) for r in rules],
                    "Events": [r["event_count"] for r in rules],
                    "Last checked (UTC)": [r["last_checked_at"] for r in rules],
                }
            ),
            height=260,
        )
        selected_id = st.selectbox(
            "Selected rule", list(by_id), format_func=lambda i: _rule_label(by_id[i]), key="alert_rules_selected"
        )
        rule = by_id[selected_id]
        st.caption(md_escape(METHOD_HELP.get(rule["method"], "")))
        a, b = st.columns(2)
        if a.button(
            "Run check now",
            key="alert_rules_run_one",
            help="Checks the selected rule now, even if it is inactive.",
        ):
            with st.spinner("Checking the rule..."):
                outcome = run_action(lambda: anomaly_service.run_rule(ctx, selected_id), component="alert_rules")
            if outcome is not None:
                st.session_state[LAST_RUN_KEY] = {
                    "title": f"Result of checking {rule['name']}",
                    "results": [{**outcome, "rule_name": rule["name"], "status": "ok"}],
                    "show_names": False,
                }
                st.rerun()
        toggle_label = "Deactivate rule" if rule["is_active"] else "Activate rule"
        if b.button(toggle_label, key="alert_rules_toggle"):
            changed = run_action(
                lambda: anomaly_service.set_rule_active(ctx, selected_id, not rule["is_active"]),
                component="alert_rules",
            )
            if changed is not None:
                _flash("Rule deactivated." if rule["is_active"] else "Rule activated.")
        with st.expander("Delete this rule"):
            st.write("A rule that has recorded events cannot be deleted. Deactivate it to keep its history.")
            sure = confirm("I understand that this deletes the rule", key="alert_rules_confirm_delete")
            if st.button("Delete rule", key="alert_rules_delete"):
                if not sure:
                    st.error("Tick the confirmation box to delete this rule.")
                else:
                    deleted = run_action(lambda: anomaly_service.delete_rule(ctx, selected_id), component="alert_rules")
                    if deleted is not None:
                        _flash("Rule deleted.")

with create_tab:
    if not datasets:
        st.info("No datasets are available yet. Upload a dataset first, then create a rule for it.")
    else:
        names = {d["dataset_id"]: d["name"] for d in datasets}
        dataset_id = st.selectbox("Dataset", list(names), format_func=names.get, key="alert_rules_create_dataset")
        methods = list(METHOD_LABELS)
        method = st.selectbox("Rule type", methods, format_func=METHOD_LABELS.get, key="alert_rules_create_method")
        st.caption(md_escape(METHOD_HELP[method]))
        aggregation = "sum"
        if method in anomaly_service.PERIOD_METHODS:
            aggregation = st.selectbox(
                "Aggregation per period", list(anomaly_service.AGGREGATIONS), key="alert_rules_create_aggregation"
            )
        dataset = run_action(lambda: dataset_service.get_dataset(ctx, dataset_id), component="alert_rules")
        if dataset is not None:
            prefix = f"alert_rules_create_{dataset_id}_{method}_{aggregation}"
            with st.form(f"{prefix}_form"):
                values = _rule_fields(
                    prefix,
                    dataset["schema"],
                    method,
                    aggregation,
                    None,
                    email_enabled=email_enabled,
                    default_recipients=default_recipients,
                )
                submitted = st.form_submit_button("Create rule")
            if submitted:
                created = run_action(
                    lambda: anomaly_service.create_rule(ctx, dataset_id=dataset_id, **values), component="alert_rules"
                )
                if created is not None:
                    _flash(f"Rule created: {md_escape(created['name'])}. Use Run check now to try it.")

with edit_tab:
    if selected_id is None:
        st.info("Select a rule on the Rules tab to edit it.")
    else:
        rule = by_id[selected_id]
        st.caption(md_escape(f"Editing {rule['name']} on {rule['dataset_name']}. The dataset cannot be changed."))
        methods = list(METHOD_LABELS)
        method = st.selectbox(
            "Rule type",
            methods,
            format_func=METHOD_LABELS.get,
            index=_index(methods, rule["method"]),
            key=f"alert_rules_edit_method_{selected_id}",
        )
        aggregation = rule["aggregation"] if rule["aggregation"] in anomaly_service.AGGREGATIONS else "sum"
        if method in anomaly_service.PERIOD_METHODS:
            aggregations = list(anomaly_service.AGGREGATIONS)
            aggregation = st.selectbox(
                "Aggregation per period",
                aggregations,
                index=_index(aggregations, aggregation),
                key=f"alert_rules_edit_aggregation_{selected_id}",
            )
        dataset = run_action(lambda: dataset_service.get_dataset(ctx, rule["dataset_id"]), component="alert_rules")
        if dataset is not None:
            prefix = f"alert_rules_edit_{selected_id}_{method}_{aggregation}"
            with st.form(f"{prefix}_form"):
                values = _rule_fields(
                    prefix,
                    dataset["schema"],
                    method,
                    aggregation,
                    rule,
                    email_enabled=email_enabled,
                    default_recipients=default_recipients,
                )
                submitted = st.form_submit_button("Save rule changes")
            if submitted:
                changes = {key: value for key, value in values.items() if not (key == "recipients" and value is None)}
                updated = run_action(
                    lambda: anomaly_service.update_rule(ctx, selected_id, **changes), component="alert_rules"
                )
                if updated is not None:
                    _flash(f"Rule saved: {md_escape(updated['name'])}.")

with result_slot:
    _show_last_run()
