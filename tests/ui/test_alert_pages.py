"""AppTest coverage of the alert pages (anomaly rules, events and notifications)."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import update
from streamlit.testing.v1 import AppTest

from components.navigation import CAPS_KEY, load_capabilities
from components.session import CTX_KEY
from config.settings import set_settings
from database import session_scope
from database.models import PlanFeature
from modules import organization_admin
from services import anomaly_service, email_service
from tests.ui.conftest import ROOT, texts

RULES = "views/alerts/rules.py"
EVENTS = "views/alerts/events.py"


@pytest.fixture
def tenant(factory, org_id):
    admin = factory.org_admin(org_id)
    factory.user(org_id, "org_admin", username="dana")
    analyst_id = factory.user(org_id, "analyst", username="ana")
    frame = pd.DataFrame(
        {
            "amount": [10.0, 11.0, 12.0, 10.5, 11.5, 500.0],
            "units": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
            "day": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]
            ),
        }
    )
    dataset_id = factory.dataset(admin, frame, "Metrics")
    factory.grant(dataset_id, analyst_id)
    return {"org_id": org_id, "admin": admin, "dataset_id": dataset_id, "analyst_id": analyst_id}


def _join(elements) -> str:
    """Visible text without the Markdown escapes the pages add on purpose."""
    return " ".join(str(e.value) for e in elements).replace("\\", "")


def _metrics(app) -> dict[str, str]:
    return {m.label: m.value for m in app.metric}


def _frame_with(app, column: str) -> pd.DataFrame:
    return next(d.value for d in app.dataframe if column in d.value.columns)


def _click(app, label: str) -> None:
    [button] = [b for b in app.button if b.label == label]
    button.click().run()


def _iqr_rule(tenant, name: str = "Amount outliers", **overrides) -> dict:
    values = {
        "dataset_id": tenant["dataset_id"],
        "name": name,
        "method": "iqr",
        "threshold": 1.5,
        "metric_column": "amount",
        "recipients": ["ops@example.com"],
        "cooldown_minutes": 0,
    }
    values.update(overrides)
    return anomaly_service.create_rule(tenant["admin"], **values)


def _create_prefix(tenant, method: str = "iqr", aggregation: str = "sum") -> str:
    return f"alert_rules_create_{tenant['dataset_id']}_{method}_{aggregation}"


def _disable_feature(feature: str) -> None:
    with session_scope() as db:
        db.execute(update(PlanFeature).where(PlanFeature.feature_key == feature).values(enabled=False))


def _direct_page(username: str, path: str) -> AppTest:
    """Run a page file on its own with the session identity set (defense in depth check)."""
    from auth import authentication
    from tests.conftest import DEFAULT_PASSWORD

    result = authentication.login(username, DEFAULT_PASSWORD)
    app = AppTest.from_file(str(ROOT / path), default_timeout=120)
    app.session_state[CTX_KEY] = result.ctx
    app.session_state[CAPS_KEY] = load_capabilities(result.ctx)
    return app.run()


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------


def test_analyst_is_not_offered_the_alert_pages(make_app, tenant):
    app = make_app("ana")
    for page in (RULES, EVENTS):
        with pytest.raises(ValueError, match="Could not find a navigation page"):
            app.switch_page(page)
    app = make_app("dana")
    app.switch_page(RULES).run()
    assert not app.exception and app.title[0].value == "Anomaly rules"


@pytest.mark.parametrize("path", [RULES, EVENTS])
def test_analyst_opening_an_alert_page_directly_gets_no_controls(tenant, path):
    app = _direct_page("ana", path)
    assert not app.exception
    assert "Only organization admins" in _join(app.info)
    assert not app.button and not app.metric


@pytest.mark.parametrize("path", [RULES, EVENTS])
def test_admin_without_the_plan_feature_gets_an_explanation(factory, path):
    org = factory.organization("Basic Co", plan="basic")
    factory.user(org, "org_admin", username="basic_admin")
    app = _direct_page("basic_admin", path)
    assert not app.exception
    assert "not included in your plan" in _join(app.info)
    assert not app.button


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_rules_page_renders_for_admin_with_the_worker_note(make_app, tenant):
    app = make_app("dana", page=RULES)
    assert not app.exception
    assert app.title[0].value == "Anomaly rules"
    assert "scripts/run_anomaly_checks.py" in _join(app.info)
    assert "Windows Task Scheduler" in _join(app.info)
    assert "No rules yet" in _join(app.info)
    assert [b.label for b in app.button if b.label == "Run all checks"] == ["Run all checks"]
    assert app.button(key="alert_rules_run_all").disabled


def test_create_iqr_rule_through_the_form(make_app, tenant):
    app = make_app("dana", page=RULES)
    prefix = _create_prefix(tenant)
    app.text_input(key=f"{prefix}_name").input("Amount outliers")
    app.text_area(key=f"{prefix}_recipients").input("ops@example.com")
    _click(app, "Create rule")
    assert not app.exception
    assert "Rule created: Amount outliers" in _join(app.success)
    [rule] = anomaly_service.list_rules(tenant["admin"])
    assert (
        rule["method"],
        rule["metric_column"],
        rule["threshold"],
        rule["severity"],
    ) == ("iqr", "amount", 1.5, "medium")
    assert rule["recipients"] == ["ops@example.com"]
    assert list(_frame_with(app, "Method")["Name"]) == ["Amount outliers"]


def test_create_rule_with_period_fields(make_app, tenant):
    app = make_app("dana", page=RULES)
    app.selectbox(key="alert_rules_create_method").select("metric_drop").run()
    prefix = _create_prefix(tenant, "metric_drop")
    assert {"Date column", "Metric column", "Compare by"} <= {s.label for s in app.selectbox}
    app.text_input(key=f"{prefix}_name").input("Revenue drop")
    app.number_input(key=f"{prefix}_threshold").set_value(25.0)
    app.selectbox(key=f"{prefix}_frequency").select("W")
    _click(app, "Create rule")
    assert not app.exception
    [rule] = anomaly_service.list_rules(tenant["admin"])
    assert (rule["method"], rule["date_column"], rule["metric_column"], rule["frequency"], rule["threshold"]) == (
        "metric_drop",
        "day",
        "amount",
        "W",
        25.0,
    )


def test_create_missing_increase_rule_defaults_to_all_columns(make_app, tenant):
    app = make_app("dana", page=RULES)
    assert "Missing or invalid values increase" in app.selectbox(key="alert_rules_create_method").options
    app.selectbox(key="alert_rules_create_method").select("missing_increase").run()
    app.text_input(key=f"{_create_prefix(tenant, 'missing_increase')}_name").input("Missing watch")
    _click(app, "Create rule")
    assert not app.exception
    [rule] = anomaly_service.list_rules(tenant["admin"])
    assert (rule["method"], rule["metric_column"], rule["threshold"]) == ("missing_increase", None, 5.0)


def test_create_rule_shows_validation_errors(make_app, tenant):
    app = make_app("dana", page=RULES)
    prefix = _create_prefix(tenant)
    _click(app, "Create rule")
    assert "rule name must be 1 to 200 characters" in _join(app.error)
    app.text_input(key=f"{prefix}_name").input("Bad recipients")
    app.text_area(key=f"{prefix}_recipients").input("not-an-email")
    _click(app, "Create rule")
    assert "Invalid e-mail address" in _join(app.error)
    assert anomaly_service.list_rules(tenant["admin"]) == []
    _iqr_rule(tenant, "Taken")
    app = make_app("dana", page=RULES)
    app.text_input(key=f"{prefix}_name").input("taken")
    _click(app, "Create rule")
    assert "already exists" in _join(app.error)


def test_recipients_are_prefilled_from_org_settings(make_app, tenant):
    organization_admin.update_org_settings(
        tenant["admin"], {"alert_default_recipients": ["boss@example.com", "ops@example.com"]}
    )
    app = make_app("dana", page=RULES)
    field = app.text_area(key=f"{_create_prefix(tenant)}_recipients")
    assert field.value == "boss@example.com\nops@example.com"
    assert not field.disabled


def test_recipients_are_disabled_without_the_email_alerts_feature(make_app, tenant):
    _disable_feature("email_alerts")
    app = make_app("dana", page=RULES)
    prefix = _create_prefix(tenant)
    assert app.text_area(key=f"{prefix}_recipients").disabled
    assert "E-mail alerts are not included in your plan" in _join(app.caption)
    app.text_input(key=f"{prefix}_name").input("No mail rule")
    _click(app, "Create rule")
    assert not app.exception
    [rule] = anomaly_service.list_rules(tenant["admin"])
    assert rule["recipients"] == []


def test_run_check_now_shows_findings_events_and_skipped_notification(make_app, tenant):
    _iqr_rule(tenant)
    app = make_app("dana", page=RULES)
    app.button(key="alert_rules_run_one").click().run()
    assert not app.exception
    metrics = _metrics(app)
    assert (metrics["Findings"], metrics["New events"], metrics["Duplicates suppressed"]) == ("1", "1", "0")
    outcome = _frame_with(app, "Outcome")
    assert list(outcome["Recipient"]) == ["ops@example.com"]
    assert list(outcome["Outcome"]) == ["skipped"]
    assert "Possible" in _join(app.markdown) or "unusual" in _join(app.markdown).lower()
    app.button(key="alert_rules_run_one").click().run()
    metrics = _metrics(app)
    assert (metrics["Findings"], metrics["New events"], metrics["Duplicates suppressed"]) == ("1", "0", "1")
    assert len(anomaly_service.list_events(tenant["admin"])) == 1


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"status": "sent", "error": None}, ("sent", "")),
        (
            {"status": "failed", "error": "The mail server refused the message."},
            ("failed", "The mail server refused the message."),
        ),
    ],
)
def test_run_check_shows_sent_and_failed_notification_outcomes(make_app, tenant, monkeypatch, result, expected):
    monkeypatch.setattr(email_service, "send_email", lambda **kwargs: result)
    _iqr_rule(tenant)
    app = make_app("dana", page=RULES)
    app.button(key="alert_rules_run_one").click().run()
    assert not app.exception
    outcome = _frame_with(app, "Outcome")
    assert (outcome["Outcome"].iloc[0], outcome["Detail"].iloc[0]) == expected
    assert _metrics(app)["New events"] == "1"


def test_run_all_checks_reports_every_active_rule(make_app, tenant):
    _iqr_rule(tenant, "Active one")
    inactive = _iqr_rule(tenant, "Inactive one", metric_column="units", is_active=False)
    app = make_app("dana", page=RULES)
    app.button(key="alert_rules_run_all").click().run()
    assert not app.exception
    assert "Results of running all checks" in _join(app.subheader)
    assert "Active one" in _join(app.markdown)
    assert "Inactive one" not in _join(app.markdown)
    assert anomaly_service.list_rules(tenant["admin"])[0]["event_count"] == 1
    assert anomaly_service.list_events(tenant["admin"], rule_id=inactive["id"]) == []


def test_run_all_with_a_failing_rule_shows_its_message(make_app, tenant):
    _iqr_rule(tenant)
    with session_scope() as db:
        from database.models import Dataset

        db.execute(update(Dataset).where(Dataset.id == tenant["dataset_id"]).values(status="archived"))
    app = make_app("dana", page=RULES)
    app.button(key="alert_rules_run_all").click().run()
    assert not app.exception
    assert app.error


def test_deactivate_and_activate_a_rule(make_app, tenant):
    _iqr_rule(tenant)
    app = make_app("dana", page=RULES)
    app.button(key="alert_rules_toggle").click().run()
    assert "Rule deactivated" in _join(app.success)
    assert anomaly_service.list_rules(tenant["admin"])[0]["is_active"] is False
    assert app.button(key="alert_rules_toggle").label == "Activate rule"
    app.button(key="alert_rules_toggle").click().run()
    assert "Rule activated" in _join(app.success)
    assert anomaly_service.list_rules(tenant["admin"])[0]["is_active"] is True


def test_delete_rule_needs_confirmation(make_app, tenant):
    _iqr_rule(tenant)
    app = make_app("dana", page=RULES)
    app.button(key="alert_rules_delete").click().run()
    assert "Tick the confirmation box" in _join(app.error)
    assert len(anomaly_service.list_rules(tenant["admin"])) == 1
    app.checkbox(key="alert_rules_confirm_delete").check()
    app.button(key="alert_rules_delete").click().run()
    assert not app.exception
    assert "Rule deleted" in _join(app.success)
    assert anomaly_service.list_rules(tenant["admin"]) == []


def test_delete_rule_with_events_shows_the_service_message(make_app, tenant):
    rule = _iqr_rule(tenant)
    anomaly_service.run_rule(tenant["admin"], rule["id"])
    app = make_app("dana", page=RULES)
    app.checkbox(key="alert_rules_confirm_delete").check()
    app.button(key="alert_rules_delete").click().run()
    assert "already has recorded events" in _join(app.error)
    assert len(anomaly_service.list_rules(tenant["admin"])) == 1


def test_edit_the_selected_rule(make_app, tenant):
    rule = _iqr_rule(tenant)
    app = make_app("dana", page=RULES)
    prefix = f"alert_rules_edit_{rule['id']}_iqr_sum"
    assert app.text_input(key=f"{prefix}_name").value == "Amount outliers"
    app.text_input(key=f"{prefix}_name").input("Amount outliers v2")
    app.number_input(key=f"{prefix}_threshold").set_value(3.0)
    app.selectbox(key=f"{prefix}_severity").select("high")
    app.text_area(key=f"{prefix}_recipients").input("a@example.com\nb@example.com")
    _click(app, "Save rule changes")
    assert not app.exception
    assert "Rule saved: Amount outliers v2" in _join(app.success)
    [stored] = anomaly_service.list_rules(tenant["admin"])
    assert (stored["name"], stored["threshold"], stored["severity"], stored["recipients"]) == (
        "Amount outliers v2",
        3.0,
        "high",
        ["a@example.com", "b@example.com"],
    )


def test_edit_rule_shows_validation_error(make_app, tenant):
    rule = _iqr_rule(tenant)
    app = make_app("dana", page=RULES)
    app.text_input(key=f"alert_rules_edit_{rule['id']}_iqr_sum_name").input("   ")
    _click(app, "Save rule changes")
    assert "rule name must be 1 to 200 characters" in _join(app.error)
    assert anomaly_service.list_rules(tenant["admin"])[0]["name"] == "Amount outliers"


def test_edit_can_switch_the_method(make_app, tenant):
    rule = _iqr_rule(tenant)
    app = make_app("dana", page=RULES)
    app.selectbox(key=f"alert_rules_edit_method_{rule['id']}").select("period_change").run()
    prefix = f"alert_rules_edit_{rule['id']}_period_change_sum"
    app.number_input(key=f"{prefix}_threshold").set_value(40.0)
    _click(app, "Save rule changes")
    assert not app.exception
    [stored] = anomaly_service.list_rules(tenant["admin"])
    assert (stored["method"], stored["date_column"], stored["metric_column"], stored["threshold"]) == (
        "period_change",
        "day",
        "amount",
        40.0,
    )


# ---------------------------------------------------------------------------
# Events and notifications
# ---------------------------------------------------------------------------


def test_events_page_renders_empty_for_admin(make_app, tenant):
    app = make_app("dana", page=EVENTS)
    assert not app.exception
    assert app.title[0].value == "Anomalies and notifications"
    assert "No events for this selection" in _join(app.info)
    assert "SMTP is not configured" in _join(app.info)


def test_events_table_details_and_acknowledge(make_app, tenant):
    rule = _iqr_rule(tenant)
    anomaly_service.run_rule(tenant["admin"], rule["id"])
    app = make_app("dana", page=EVENTS)
    assert not app.exception
    events = _frame_with(app, "What was flagged")
    assert list(events["Rule"]) == ["Amount outliers"] and list(events["Status"]) == ["new"]
    details = _frame_with(app, "Field")
    shown = dict(zip(details["Field"], details["Value"], strict=True))
    assert shown["Method"] == "Outliers by IQR"
    assert shown["Column"] == "amount" and shown["Direction"] == "above"
    assert shown["Observed value"] == "500.00" and "Upper bound" in shown and "Threshold" in shown
    assert "A statistical flag for review" in _join(app.caption)
    event_id = anomaly_service.list_events(tenant["admin"])[0]["id"]
    app.button(key=f"alert_events_ack_{event_id}").click().run()
    assert not app.exception
    assert "Event acknowledged" in _join(app.success)
    assert anomaly_service.list_events(tenant["admin"])[0]["status"] == "acknowledged"
    assert list(_frame_with(app, "What was flagged")["Status"]) == ["acknowledged"]
    assert not [b for b in app.button if b.label == "Acknowledge this event"]
    assert "Acknowledged on" in _join(app.caption)


def test_events_filters_by_rule_and_status(make_app, tenant):
    first = _iqr_rule(tenant, "First")
    second = _iqr_rule(tenant, "Second", metric_column="units", threshold=0.5)
    anomaly_service.run_rule(tenant["admin"], first["id"])
    anomaly_service.run_rule(tenant["admin"], second["id"])
    events = anomaly_service.list_events(tenant["admin"])
    app = make_app("dana", page=EVENTS)
    assert len(_frame_with(app, "What was flagged")) == len(events)
    app.selectbox(key="alert_events_rule").select(first["id"]).run()
    assert set(_frame_with(app, "What was flagged")["Rule"]) == {"First"}
    app.selectbox(key="alert_events_rule").select("all").run()
    app.selectbox(key="alert_events_status").select("acknowledged").run()
    assert not app.exception
    assert "No events for this selection" in _join(app.info)


def test_notification_log_shows_skipped_when_smtp_is_not_configured(make_app, tenant):
    rule = _iqr_rule(tenant)
    anomaly_service.run_rule(tenant["admin"], rule["id"])
    app = make_app("dana", page=EVENTS)
    assert not app.exception
    log = _frame_with(app, "Recipient")
    assert list(log["Status"]) == ["skipped"] and list(log["Recipient"]) == ["ops@example.com"]
    assert "E-mail is not configured" in str(log["Detail"].iloc[0])
    metrics = _metrics(app)
    assert (metrics["Sent"], metrics["Failed"], metrics["Skipped"]) == ("0", "0", "1")
    app.selectbox(key="alert_events_notification_status").select("sent").run()
    assert "No notifications recorded yet" in _join(app.info)


def test_smtp_status_when_configured_never_shows_the_password(make_app, tenant, app_settings):
    set_settings(
        app_settings.with_overrides(
            smtp_host="smtp.example.com",
            smtp_from="alerts@example.com",
            smtp_username="user",
            smtp_password="s3cret-pass",
        )
    )
    app = make_app("dana", page=EVENTS)
    assert not app.exception
    text = _join(app.success)
    assert "SMTP is configured" in text and "smtp.example.com" in text and "alerts@example.com" in text
    assert "s3cret-pass" not in text + texts(app)


# ---------------------------------------------------------------------------
# Untrusted text and selection that changes under the user
# ---------------------------------------------------------------------------

PAYLOAD = "![x](http://example.com/p.png) **b**"


def test_rule_names_with_markdown_are_shown_literally(make_app, tenant):
    app = make_app("dana", page=RULES)
    app.text_input(key=f"{_create_prefix(tenant)}_name").input(PAYLOAD)
    _click(app, "Create rule")
    assert not app.exception
    assert not any(PAYLOAD in str(e.value) for e in [*app.success, *app.subheader, *app.markdown, *app.caption])
    assert PAYLOAD in _join(app.success)
    app.button(key="alert_rules_run_one").click().run()
    assert not app.exception
    assert not any(PAYLOAD in str(e.value) for e in [*app.subheader, *app.markdown])
    assert PAYLOAD in _join(app.subheader)
    rule_id = anomaly_service.list_rules(tenant["admin"])[0]["id"]
    app.text_input(key=f"alert_rules_edit_{rule_id}_iqr_sum_name").input(PAYLOAD + " v2")
    _click(app, "Save rule changes")
    assert not any(PAYLOAD in str(e.value) for e in app.success)
    assert PAYLOAD in _join(app.success)


def test_event_titles_with_markdown_are_escaped_in_html(make_app, tenant):
    rule = _iqr_rule(tenant, PAYLOAD)
    anomaly_service.run_rule(tenant["admin"], rule["id"])
    app = make_app("dana", page=EVENTS)
    assert not app.exception
    html_blocks = [str(m.value) for m in app.markdown if "vdas-issue-suspected" in str(m.value)]
    assert html_blocks and all("<img" not in block and "<b" not in block for block in html_blocks)


def test_deleting_the_selected_rule_of_two_shows_the_other(make_app, tenant):
    first = _iqr_rule(tenant, "First rule")
    second = _iqr_rule(tenant, "Second rule", metric_column="units")
    app = make_app("dana", page=RULES)
    assert app.selectbox(key="alert_rules_selected").value == first["id"]
    app.checkbox(key="alert_rules_confirm_delete").check()
    app.button(key="alert_rules_delete").click().run()
    assert not app.exception
    assert not app.error
    assert "Rule deleted" in _join(app.success)
    assert app.selectbox(key="alert_rules_selected").value == second["id"]
    assert list(_frame_with(app, "Method")["Name"]) == ["Second rule"]
    assert [r["id"] for r in anomaly_service.list_rules(tenant["admin"])] == [second["id"]]


def test_selecting_the_second_rule_edits_that_rule(make_app, tenant):
    _iqr_rule(tenant, "First rule")
    second = _iqr_rule(tenant, "Second rule", metric_column="units")
    app = make_app("dana", page=RULES)
    app.selectbox(key="alert_rules_selected").select(second["id"]).run()
    assert app.text_input(key=f"alert_rules_edit_{second['id']}_iqr_sum_name").value == "Second rule"
    app.button(key="alert_rules_toggle").click().run()
    states = {r["name"]: r["is_active"] for r in anomaly_service.list_rules(tenant["admin"])}
    assert states == {"First rule": True, "Second rule": False}


def test_acknowledging_while_filtered_to_new_shows_the_remaining_event(make_app, tenant):
    first = _iqr_rule(tenant, "First")
    second = _iqr_rule(tenant, "Second", metric_column="units", threshold=0.1)
    anomaly_service.run_rule(tenant["admin"], first["id"])
    anomaly_service.run_rule(tenant["admin"], second["id"])
    app = make_app("dana", page=EVENTS)
    app.selectbox(key="alert_events_status").select("new").run()
    shown = _frame_with(app, "What was flagged")
    total = len(shown)
    assert total >= 2
    event_key = "alert_events_item_all_new"
    selected = app.selectbox(key=event_key).value
    app.button(key=f"alert_events_ack_{selected}").click().run()
    assert not app.exception
    assert not app.error
    remaining = _frame_with(app, "What was flagged")
    assert len(remaining) == total - 1 and set(remaining["Status"]) == {"new"}
    assert app.selectbox(key=event_key).value != selected
    assert [
        e["status"] for e in anomaly_service.list_events(tenant["admin"]) if e["id"] == selected
    ] == ["acknowledged"]
