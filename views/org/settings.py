"""Organization Admin: data-handling settings, alert recipients and e-mail delivery check."""

from __future__ import annotations

import streamlit as st

from components.dashboard_cards import commit, require_role, show_flash
from components.feedback import md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import require_ctx
from core.constants import Feature, Role
from modules import org_settings, organization_admin
from services import email_service

PREFIX = "org_settings"
DAY_FIRST = "Day first (for example 31/12/2025)"
MONTH_FIRST = "Month first (for example 12/31/2025)"

ctx = require_ctx()
caps = capabilities()  # UI hints only; services enforce role, permission and plan
page_header(
    "Settings", "How your datasets are read, who receives alerts, and whether e-mail can be delivered."
)
require_role(ctx, Role.ORG_ADMIN, "This page is only available to Organization Admins.")
show_flash()

current = run_action(lambda: organization_admin.get_settings_view(ctx), component=PREFIX)
if current is None:
    st.stop()

st.subheader("Data and alert settings")
with st.form(f"{PREFIX}_form"):
    placeholders = st.text_area(
        "Values treated as missing (one per line)",
        value="\n".join(current[org_settings.MISSING_PLACEHOLDERS]),
        key=f"{PREFIX}_placeholders",
        help=(
            "Text such as N/A or null is counted as missing when datasets are checked and cleaned. "
            "Zero, False and Unknown are never missing unless you list them."
        ),
    )
    date_order = st.radio(
        "Ambiguous dates are read as",
        [DAY_FIRST, MONTH_FIRST],
        index=0 if current[org_settings.DATE_DAYFIRST] else 1,
        key=f"{PREFIX}_date_order",
    )
    text_placeholder = st.text_input(
        "Replacement for missing text",
        value=current[org_settings.TEXT_PLACEHOLDER],
        key=f"{PREFIX}_text_placeholder",
        help="Used when you fill missing text values with a placeholder.",
    )
    recipients = st.text_area(
        "Default alert recipients (one e-mail address per line)",
        value="\n".join(current[org_settings.ALERT_DEFAULT_RECIPIENTS]),
        key=f"{PREFIX}_recipients",
    )
    saved = st.form_submit_button("Save settings", type="primary")
if saved:
    commit(
        lambda: organization_admin.update_org_settings(
            ctx,
            {
                org_settings.MISSING_PLACEHOLDERS: placeholders.splitlines(),
                org_settings.DATE_DAYFIRST: date_order == DAY_FIRST,
                org_settings.TEXT_PLACEHOLDER: text_placeholder,
                org_settings.ALERT_DEFAULT_RECIPIENTS: recipients.splitlines(),
            },
        ),
        success="Settings saved. They apply to datasets you open or clean from now on.",
    )

st.subheader("E-mail delivery")
status = run_action(email_service.smtp_status, component=PREFIX)
if status is None:
    st.stop()
with st.container(border=True):
    if status["configured"]:
        security = "SSL" if status["use_ssl"] else ("STARTTLS" if status["use_tls"] else "none")
        st.markdown(f"E-mail is configured on {md_escape(status['host'])}, port {status['port']}.")
        col_from, col_security, col_login = st.columns(3)
        col_from.metric("Sender address", md_escape(status["from_address"] or "not set"))
        col_security.metric("Connection security", security)
        col_login.metric("Sign-in to server", "set" if status["username_set"] else "not used")
    else:
        st.warning(
            "E-mail is not configured. Anomaly alerts are recorded but cannot be sent. "
            "The administrator who runs the server sets SMTP_HOST and SMTP_FROM in the environment.",
            icon=":material/mail_off:",
        )
    st.caption("The server password is never shown here.")

if caps is not None and not caps.has(Feature.EMAIL_ALERTS):
    st.info(
        "Test e-mails are available on plans that include automated e-mail alerts. "
        "Ask your platform administrator about upgrading the plan.",
        icon=":material/mail_off:",
    )
    st.stop()
st.caption(
    f"Your organization can send up to {email_service.TEST_EMAIL_HOURLY_LIMIT} test e-mails per hour."
)
with st.form(f"{PREFIX}_test_form"):
    recipient = st.text_input(
        "Send a test message to", key=f"{PREFIX}_test_to", placeholder="name@example.com"
    )
    sent = st.form_submit_button("Send test e-mail")
if sent:
    result = run_action(lambda: email_service.send_test_email(ctx, recipient), component=PREFIX)
    if result is not None:
        if result["status"] == "sent":
            st.success(
                f"Test e-mail sent to {md_escape(recipient.strip())}. Check that inbox to confirm delivery.",
                icon=":material/mark_email_read:",
            )
        elif result["status"] == "skipped":
            st.info(f"Skipped: no message was sent. {md_escape(result['error'])}", icon=":material/mail_off:")
        else:
            st.error(
                f"Failed: the message could not be delivered. {md_escape(result['error'])}",
                icon=":material/error:",
            )
