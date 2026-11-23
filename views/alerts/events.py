"""Anomaly events: review what was flagged, acknowledge it and check e-mail delivery."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.feedback import escape, feature_unavailable, md_escape, page_header, run_action, show_table
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Feature
from services import anomaly_service, email_service

FLASH_KEY = "alert_events_flash"
ALL = "all"
EVENT_LIMIT = 200
NOTIFICATION_LIMIT = 200
METHOD_LABELS = {
    "iqr": "Outliers by IQR",
    "zscore": "Outliers by z-score",
    "period_change": "Change between periods",
    "metric_drop": "Drop in a metric",
    "baseline_deviation": "Deviation from the recent baseline",
    "missing_increase": "Missing or invalid values increase",
}
OPEN_STATUSES = {"new", "notified", "suppressed"}


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "" if value is None else str(value)
    number = float(value)
    return f"{number:,.2f}" if number.is_integer() or abs(number) >= 0.01 else f"{number:.4g}"


def _explanation(event: dict[str, Any]) -> pd.DataFrame:
    """The fields that explain why an event was flagged, as text rows (untrusted values stay plain text)."""
    d = event["details"]
    rows = [
        ("Method", METHOD_LABELS.get(d.get("method"), d.get("method"))),
        ("Column", d.get("column")),
        ("Period", d.get("period")),
        ("Direction", d.get("direction")),
        ("Observed value", d.get("observed")),
        ("Expected value", d.get("expected")),
        ("Lower bound", d.get("lower_bound")),
        ("Upper bound", d.get("upper_bound")),
        ("Threshold", d.get("threshold")),
        ("Aggregation", d.get("aggregation")),
        ("Dataset version", d.get("dataset_version")),
        ("Severity", event["severity"]),
        ("Status", event["status"]),
        ("Held back because", event["suppressed_reason"]),
        ("Detected (UTC)", event["detected_at"]),
        ("Acknowledged (UTC)", event["acknowledged_at"]),
    ]
    return pd.DataFrame(
        [{"Field": label, "Value": _fmt(value)} for label, value in rows if value not in (None, "")]
    )


ctx = require_ctx()
caps = capabilities()
page_header(
    "Anomalies and notifications",
    "Review what the rules flagged, acknowledge it once looked at and check whether e-mails went out.",
)
if not ctx.is_org_admin:
    feature_unavailable("Only organization admins can review anomalies.")
    st.stop()
if caps is None or not caps.has(Feature.ANOMALY_DETECTION):
    feature_unavailable("Anomaly detection is not included in your plan.")
    st.stop()

flash = st.session_state.pop(FLASH_KEY, None)
if flash:
    st.success(flash, icon=":material/check_circle:")

events_tab, notifications_tab = st.tabs(["Events", "Notifications and e-mail"])

with events_tab:
    rules = run_action(lambda: anomaly_service.list_rules(ctx), component="alert_events") or []
    rule_names = {r["id"]: r["name"] for r in rules}
    f1, f2 = st.columns(2)
    rule_choice = f1.selectbox(
        "Rule",
        [ALL, *rule_names],
        format_func=lambda v: "All rules" if v == ALL else rule_names[v],
        key="alert_events_rule",
    )
    status_choice = f2.selectbox(
        "Status",
        [ALL, *anomaly_service.EVENT_STATUSES],
        format_func=lambda v: "All statuses" if v == ALL else v,
        key="alert_events_status",
    )
    events = (
        run_action(
            lambda: anomaly_service.list_events(
                ctx,
                rule_id=None if rule_choice == ALL else rule_choice,
                status=None if status_choice == ALL else status_choice,
                limit=EVENT_LIMIT,
            ),
            component="alert_events",
        )
        or []
    )
    if not events:
        st.info("No events for this selection. Run a check on the Anomaly rules page to look for unusual values.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Events shown", f"{len(events):,}")
        c2.metric("Not yet acknowledged", f"{sum(e['status'] in OPEN_STATUSES for e in events):,}")
        show_table(
            pd.DataFrame(
                {
                    "Detected (UTC)": [e["detected_at"] for e in events],
                    "Rule": [e["rule_name"] for e in events],
                    "Severity": [e["severity"] for e in events],
                    "Status": [e["status"] for e in events],
                    "What was flagged": [e["title"] for e in events],
                    "Observed": [e["observed_value"] for e in events],
                    "Expected": [e["expected_value"] for e in events],
                }
            ),
            height=280,
        )
        by_id = {e["id"]: e for e in events}
        event_id = st.selectbox(
            "Event details",
            list(by_id),
            format_func=lambda i: f"#{i}: {by_id[i]['title'][:80]}",
            key=f"alert_events_item_{rule_choice}_{status_choice}",
        )
        event = by_id[event_id]
        with st.container(border=True):
            st.markdown(
                f'<div class="vdas-issue-suspected"><strong>{escape(event["title"])}</strong></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "A statistical flag for review. It does not mean the data is wrong, only that it is unusual "
                "compared with the rule's expectation."
            )
            st.dataframe(_explanation(event), hide_index=True, width="stretch")
            with st.expander("All recorded details"):
                st.json(event["details"])
            if event["status"] == "acknowledged":
                st.caption(f"Acknowledged on {event['acknowledged_at']:%Y-%m-%d %H:%M} UTC.")
            elif st.button("Acknowledge this event", key=f"alert_events_ack_{event_id}"):
                done = run_action(lambda: anomaly_service.acknowledge_event(ctx, event_id), component="alert_events")
                if done is not None:
                    st.session_state[FLASH_KEY] = "Event acknowledged."
                    st.rerun()

with notifications_tab:
    smtp = email_service.smtp_status()
    st.subheader("E-mail delivery")
    if smtp["configured"]:
        security = "SSL" if smtp["use_ssl"] else "STARTTLS" if smtp["use_tls"] else "no encryption"
        st.success(
            md_escape(
                f"SMTP is configured: {smtp['host']}:{smtp['port']}, sending as {smtp['from_address']} "
                f"({security}, {'sign-in set' if smtp['username_set'] else 'no sign-in'})."
            ),
            icon=":material/mark_email_read:",
        )
    else:
        st.info(
            "SMTP is not configured, so alert e-mails are logged as skipped and nothing is sent. "
            "Set SMTP_HOST and SMTP_FROM (and credentials if needed) in the .env file, then restart the app.",
            icon=":material/mail_lock:",
        )
    st.subheader("Notification log")
    notifications = (
        run_action(
            lambda: anomaly_service.list_notifications(ctx, limit=NOTIFICATION_LIMIT),
            component="alert_events",
        )
        or []
    )
    status_filter = st.selectbox(
        "Notification status",
        [ALL, "sent", "failed", "skipped"],
        format_func=lambda v: "All" if v == ALL else v,
        key="alert_events_notification_status",
    )
    shown = [n for n in notifications if status_filter == ALL or n["status"] == status_filter]
    counts = {name: sum(n["status"] == name for n in notifications) for name in ("sent", "failed", "skipped")}
    m1, m2, m3 = st.columns(3)
    m1.metric("Sent", f"{counts['sent']:,}")
    m2.metric("Failed", f"{counts['failed']:,}")
    m3.metric("Skipped", f"{counts['skipped']:,}")
    show_table(
        pd.DataFrame(
            {
                "Time (UTC)": [n["created_at"] for n in shown],
                "Rule": [n["rule_name"] or "" for n in shown],
                "Recipient": [n["recipient"] for n in shown],
                "Status": [n["status"] for n in shown],
                "Subject": [n["subject"] or "" for n in shown],
                "Detail": [n["error_message"] or "" for n in shown],
            }
        ),
        height=280,
        empty_message=(
            "No notifications recorded yet. "
            "A notification is logged for each recipient when a check finds something new."
        ),
    )
