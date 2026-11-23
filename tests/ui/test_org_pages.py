"""Organization Admin pages: rendering, primary actions and error states."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from auth import authentication
from components.navigation import load_capabilities, page_keys_for
from components.session import CTX_KEY
from core.constants import ANALYST_DEFAULT_PERMISSIONS, Permission, UsageMetric
from core.errors import AppError, NotFoundError
from database import session_scope
from database.models import AnalysisHistory, User
from modules import org_settings, organization_admin, usage_tracking
from services import audit_service, dataset_service, email_service
from tests.conftest import DEFAULT_PASSWORD
from tests.ui.conftest import ROOT
from tests.ui.conftest import texts as _raw_texts

PAGES = {
    "dashboard": "views/org/dashboard.py",
    "users": "views/org/users.py",
    "usage": "views/org/usage.py",
    "settings": "views/org/settings.py",
    "audit": "views/org/audit.py",
}
STRONG = "Gr3at!Secret-Pass"


def texts(app: AppTest) -> str:
    """Visible text with Markdown escape backslashes removed."""
    return _raw_texts(app).replace("\\", "")


def _button(app: AppTest, label: str):
    matches = [b for b in app.button if b.label == label]
    assert matches, f"no button labelled {label!r}; have {[b.label for b in app.button]}"
    return matches[0]


def _frame(app: AppTest, column: str):
    return next(df.value for df in app.dataframe if column in df.value.columns)


@pytest.fixture
def admin(factory, org_id):
    user_id = factory.user(org_id, "org_admin", username="dana")
    return factory.ctx(user_id)


@pytest.fixture
def open_page(make_app, admin):
    def _open(page: str) -> AppTest:
        return make_app(admin.username, page=PAGES[page])

    return _open


def _create_analyst_via_form(
    app: AppTest, username: str, *, email: str | None = None, password: str = STRONG, gen: int = 0
) -> None:
    app.text_input(key=f"org_users_create_username_{gen}").input(username)
    app.text_input(key=f"org_users_create_email_{gen}").input(email or f"{username}@acme.example")
    app.text_input(key=f"org_users_create_name_{gen}").input(username.title())
    app.text_input(key=f"org_users_create_password_{gen}").input(password)
    _button(app, "Create analyst").click().run()


def _user(ctx, username: str) -> dict:
    return next(user for user in organization_admin.list_users(ctx) if user["username"] == username)


@pytest.mark.parametrize("page", list(PAGES))
def test_every_org_page_renders_for_org_admin(open_page, page):
    app = open_page(page)
    assert not app.exception
    assert app.title


@pytest.mark.parametrize("page", list(PAGES))
def test_every_org_page_renders_on_basic_plan(make_app, factory, page):
    basic_org = factory.organization("Basic Co", plan="basic")
    factory.user(basic_org, "org_admin", username="basicadmin")
    app = make_app("basicadmin", page=PAGES[page])
    assert not app.exception
    assert app.title


def test_org_pages_are_not_offered_to_analysts_and_refuse_direct_runs(factory, org_id, admin):
    analyst = factory.analyst(org_id)
    org_keys = {"org_dashboard", "org_users", "org_usage", "org_settings", "org_audit"}
    analyst_keys = {key for keys in page_keys_for(load_capabilities(analyst)).values() for key in keys}
    admin_keys = {key for keys in page_keys_for(load_capabilities(admin)).values() for key in keys}
    assert not org_keys & analyst_keys
    assert org_keys <= admin_keys
    for path in PAGES.values():
        app = AppTest.from_file(str(ROOT / path), default_timeout=60)
        app.session_state[CTX_KEY] = analyst
        app.run()
        assert not app.exception
        assert "only available to Organization Admins" in texts(app)
        assert not app.button and not app.text_input


def test_dashboard_shows_plan_usage_and_recent_operations(open_page, factory, admin, org_id, sales_frame):
    factory.user(org_id, "analyst", username="ana")
    dataset_id = factory.dataset(admin, sales_frame, "Sales 2025")
    with session_scope() as db:
        usage_tracking.record_usage(db, org_id, admin.user_id, UsageMetric.ANALYSIS_QUERY, 3)
    app = open_page("dashboard")
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Plan"] == "Premium"
    assert metrics["Users"] == "2 active"
    assert metrics["Datasets"] == "1"
    assert metrics["Average quality"].endswith("of 100")
    view = organization_admin.org_dashboard(admin)
    assert view["datasets"]["active"] == 1 and view["analyses_this_month"] == 3
    progress_texts = " ".join(str(getattr(element, "text", "")) for element in app.get("progress"))
    assert "Analyses this month" in progress_texts.replace("\\", "") and "3 of" in progress_texts
    assert "Voice analysis" in texts(app) and "Data profiling" in texts(app)
    assert dataset_id


def test_dashboard_marks_features_missing_from_basic_plan(make_app, factory):
    basic_org = factory.organization("Basic Co", plan="basic")
    factory.user(basic_org, "org_admin", username="basicadmin")
    app = make_app("basicadmin", page=PAGES["dashboard"])
    assert "Voice analysis (not in this plan)" in texts(app)
    assert "Anomaly detection is not part of your plan" in texts(app)


def test_create_analyst_through_form(open_page, admin):
    app = open_page("users")
    assert set(app.multiselect(key="org_users_create_permissions_0").value) == set(
        ANALYST_DEFAULT_PERMISSIONS
    )
    app.multiselect(key="org_users_create_permissions_0").select(Permission.CLEAN_DATA.value)
    _create_analyst_via_form(app, "Ana.Lyst")
    assert not app.exception and "ana.lyst created" in texts(app)
    created = _user(admin, "ana.lyst")
    assert created["role"] == "analyst" and created["is_active"]
    assert set(created["permissions"]) == set(ANALYST_DEFAULT_PERMISSIONS) | {Permission.CLEAN_DATA.value}
    assert "ana.lyst" in set(_frame(app, "Username")["Username"])
    login = authentication.login("ana.lyst", STRONG)
    assert login.ctx.username == "ana.lyst"
    assert app.text_input(key="org_users_create_username_1").value == ""


def test_create_analyst_validation_errors(open_page, admin, org_id):
    app = open_page("users")
    _create_analyst_via_form(app, "weakling", password="short")
    assert app.error
    _create_analyst_via_form(app, "bad.mail", email="not-an-email")
    assert "valid e-mail" in texts(app)
    _create_analyst_via_form(app, "dana")
    assert "already taken" in texts(app)
    _create_analyst_via_form(app, "ab")
    assert "Usernames must be 3 to 64" in texts(app)
    assert [user["username"] for user in organization_admin.list_users(admin)] == ["dana"]
    # the entered values are kept so the user can correct them
    assert app.text_input(key="org_users_create_username_0").value == "ab"


def test_user_quota_error_is_shown(make_app, factory):
    from modules import super_admin

    basic_org = factory.organization("Tiny Co", plan="basic")
    factory.user(basic_org, "org_admin", username="tinyadmin")
    root = factory.super_admin()
    plan = next(p for p in super_admin.list_plans(root) if p["code"] == "basic")
    super_admin.update_plan(root, plan["id"], max_users=1)
    app = make_app("tinyadmin", page=PAGES["users"])
    _create_analyst_via_form(app, "overflow")
    assert app.error and "limit" in texts(app).lower()


def test_edit_user_details(open_page, admin, org_id, factory):
    analyst_id = factory.user(org_id, "analyst", username="ana")
    app = open_page("users")
    app.selectbox(key="org_users_selected").select(analyst_id).run()
    app.text_input(key=f"org_users_edit_email_{analyst_id}").input("ana.new@acme.example")
    app.text_input(key=f"org_users_edit_name_{analyst_id}").input("Ana Analyst")
    _button(app, "Save user").click().run()
    assert not app.exception and "ana saved" in texts(app)
    updated = _user(admin, "ana")
    assert updated["email"] == "ana.new@acme.example" and updated["full_name"] == "Ana Analyst"
    app.text_input(key=f"org_users_edit_email_{analyst_id}").input("broken")
    _button(app, "Save user").click().run()
    assert "valid e-mail" in texts(app)
    assert _user(admin, "ana")["email"] == "ana.new@acme.example"


def test_deactivate_and_reactivate_user(open_page, admin, org_id, factory):
    analyst_id = factory.user(org_id, "analyst", username="ana")
    app = open_page("users")
    app.selectbox(key="org_users_selected").select(analyst_id).run()
    _button(app, "Deactivate user").click().run()
    assert not app.exception and "ana deactivated" in texts(app)
    assert _user(admin, "ana")["is_active"] is False
    with pytest.raises(AppError):
        authentication.login("ana", DEFAULT_PASSWORD)
    _button(app, "Reactivate user").click().run()
    assert _user(admin, "ana")["is_active"] is True
    assert authentication.login("ana", DEFAULT_PASSWORD).ctx.username == "ana"


def test_cannot_deactivate_yourself(open_page, admin):
    app = open_page("users")
    assert app.selectbox(key="org_users_selected").value == admin.user_id
    _button(app, "Deactivate user").click().run()
    assert "cannot deactivate your own account" in texts(app)
    assert _user(admin, "dana")["is_active"] is True


def test_reset_password_needs_confirmation_and_forces_change(open_page, admin, org_id, factory):
    analyst_id = factory.user(org_id, "analyst", username="ana")
    app = open_page("users")
    app.selectbox(key="org_users_selected").select(analyst_id).run()
    app.text_input(key=f"org_users_reset_password_{analyst_id}_0").input(STRONG)
    _button(app, "Reset password").click().run()
    assert "Tick the confirmation" in texts(app)
    assert not _user(admin, "ana")["must_change_password"]
    app.checkbox(key=f"org_users_reset_confirm_{analyst_id}_0").check().run()
    _button(app, "Reset password").click().run()
    assert not app.exception and "Password reset for ana" in texts(app)
    assert _user(admin, "ana")["must_change_password"] is True
    result = authentication.login("ana", STRONG)
    assert result.must_change_password
    with pytest.raises(AppError):
        authentication.login("ana", DEFAULT_PASSWORD)


def test_reset_password_rejects_weak_password(open_page, admin, org_id, factory):
    analyst_id = factory.user(org_id, "analyst", username="ana")
    app = open_page("users")
    app.selectbox(key="org_users_selected").select(analyst_id).run()
    app.text_input(key=f"org_users_reset_password_{analyst_id}_0").input("weak")
    app.checkbox(key=f"org_users_reset_confirm_{analyst_id}_0").check().run()
    _button(app, "Reset password").click().run()
    assert app.error
    assert not _user(admin, "ana")["must_change_password"]
    assert authentication.login("ana", DEFAULT_PASSWORD).ctx.username == "ana"


def test_edit_analyst_permissions(open_page, admin, org_id, factory):
    analyst_id = factory.user(org_id, "analyst", username="ana")
    app = open_page("users")
    app.selectbox(key="org_users_selected").select(analyst_id).run()
    box = app.multiselect(key=f"org_users_permissions_{analyst_id}")
    assert set(box.value) == set(ANALYST_DEFAULT_PERMISSIONS)
    box.select(Permission.CLEAN_DATA.value).unselect(Permission.USE_VOICE.value).run()
    _button(app, "Save permissions").click().run()
    assert not app.exception and "Permissions saved for ana" in texts(app)
    assert set(_user(admin, "ana")["permissions"]) == {
        Permission.EXPORT_DATA.value,
        Permission.SAVE_REPORTS.value,
        Permission.CLEAN_DATA.value,
    }


def test_permissions_tab_explains_admins_hold_everything(open_page):
    app = open_page("users")
    assert "Organization admins hold every permission" in texts(app)
    assert not [b for b in app.button if b.label == "Save permissions"]


def test_grant_and_revoke_dataset_access(open_page, admin, org_id, factory, sales_frame):
    analyst_id = factory.user(org_id, "analyst", username="ana")
    analyst = factory.ctx(analyst_id)
    dataset_id = factory.dataset(admin, sales_frame, "Sales 2025")
    assert dataset_service.list_datasets(analyst) == []
    app = open_page("users")
    assert not app.exception
    assert app.selectbox(key="org_users_access_dataset").value == dataset_id
    assert "No analyst has access to this dataset yet" in texts(app)
    app.selectbox(key=f"org_users_grant_user_{dataset_id}").select(analyst_id).run()
    _button(app, "Grant access").click().run()
    assert not app.exception and "ana can now open" in texts(app)
    assert [d["dataset_id"] for d in dataset_service.list_datasets(analyst)] == [dataset_id]
    assert "ana" in set(_frame(app, "Granted (UTC)")["Username"])
    _button(app, "Revoke access").click().run()
    assert not app.exception and "ana can no longer open" in texts(app)
    assert dataset_service.list_datasets(analyst) == []
    actions = [e["action"] for e in audit_service.list_audit_logs(admin, action_prefix="dataset.")]
    assert actions[:2] == ["dataset.revoke", "dataset.grant"]


def test_dataset_access_empty_states(open_page, admin, org_id, factory, sales_frame):
    app = open_page("users")
    assert "No datasets yet" in texts(app)
    factory.dataset(admin, sales_frame, "Sales 2025")
    app = open_page("users")
    assert "There are no analysts yet" in texts(app)


def test_usage_page_table_chart_and_history_filters(open_page, admin, org_id, factory, sales_frame):
    ana_id = factory.user(org_id, "analyst", username="ana")
    dataset_id = factory.dataset(admin, sales_frame, "Sales 2025")
    with session_scope() as db:
        usage_tracking.record_usage(db, org_id, ana_id, UsageMetric.ANALYSIS_QUERY, 4)
        usage_tracking.record_usage(db, org_id, ana_id, UsageMetric.EXPORT, 2)
        for user_id, question in ((ana_id, "total revenue"), (admin.user_id, "top products")):
            db.add(
                AnalysisHistory(
                    organization_id=org_id,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    question=question,
                    status="success",
                    duration_ms=12,
                )
            )
    app = open_page("usage")
    assert not app.exception
    table = _frame(app, "Month")
    assert len(table) == 6 and table.iloc[-1]["Text analyses"] == 4 and table.iloc[-1]["Exports"] == 2
    assert len(app.get("plotly_chart")) == 1
    app.selectbox(key="org_usage_months").select(12).run()
    assert len(_frame(app, "Month")) == 12
    app.selectbox(key="org_usage_metric").select(UsageMetric.EXPORT.value).run()
    assert not app.exception
    history = _frame(app, "Question")
    assert set(history["Question"]) == {"total revenue", "top products"}
    app.selectbox(key="org_usage_history_user").select(ana_id).run()
    history = _frame(app, "Question")
    assert list(history["Question"]) == ["total revenue"] and list(history["User"]) == ["ana"]
    app.selectbox(key="org_usage_history_user").select(None).run()
    app.selectbox(key="org_usage_history_dataset").select(dataset_id).run()
    assert len(_frame(app, "Question")) == 2


def test_usage_history_empty_state(open_page):
    app = open_page("usage")
    assert "No analyses match these filters yet" in texts(app)


def test_settings_form_saves_and_applies(open_page, admin, org_id):
    app = open_page("settings")
    assert not app.exception
    assert app.text_area(key="org_settings_placeholders").value.splitlines()
    app.text_area(key="org_settings_placeholders").input("N/A\n--\n  missing  \n--\n")
    radio = app.radio(key="org_settings_date_order")
    radio.set_value(radio.options[1])
    app.text_input(key="org_settings_text_placeholder").input("Unknown value")
    app.text_area(key="org_settings_recipients").input("ops@acme.example\nboss@acme.example\n")
    _button(app, "Save settings").click().run()
    assert not app.exception and "Settings saved" in texts(app)
    with session_scope() as db:
        stored = org_settings.get_org_settings(db, org_id)
    assert stored[org_settings.MISSING_PLACEHOLDERS] == ["N/A", "--", "missing"]
    assert stored[org_settings.DATE_DAYFIRST] is False
    assert stored[org_settings.TEXT_PLACEHOLDER] == "Unknown value"
    assert stored[org_settings.ALERT_DEFAULT_RECIPIENTS] == ["ops@acme.example", "boss@acme.example"]
    fresh = open_page("settings")
    assert fresh.text_area(key="org_settings_placeholders").value == "N/A\n--\nmissing"
    assert fresh.radio(key="org_settings_date_order").value == radio.options[1]


def test_settings_validation_errors(open_page, org_id):
    app = open_page("settings")
    app.text_input(key="org_settings_text_placeholder").input("")
    _button(app, "Save settings").click().run()
    assert "text placeholder must be 1 to 50" in texts(app)
    app.text_input(key="org_settings_text_placeholder").input("Not Provided")
    app.text_area(key="org_settings_recipients").input("not-an-address")
    _button(app, "Save settings").click().run()
    assert "Invalid e-mail address" in texts(app)
    app.text_area(key="org_settings_recipients").input("")
    app.text_area(key="org_settings_placeholders").input("x" * 60)
    _button(app, "Save settings").click().run()
    assert "at most 50 characters" in texts(app)
    with session_scope() as db:
        assert org_settings.get_org_settings(db, org_id)[org_settings.ALERT_DEFAULT_RECIPIENTS] == []


def test_smtp_status_without_configuration_and_test_email_is_skipped(open_page, admin):
    app = open_page("settings")
    assert "E-mail is not configured" in texts(app) or any("not configured" in w.value for w in app.warning)
    app.text_input(key="org_settings_test_to").input("someone@example.com")
    _button(app, "Send test e-mail").click().run()
    assert not app.exception and "Skipped: no message was sent" in texts(app)
    logs = audit_service.list_audit_logs(admin, action_prefix="email.test")
    assert logs and logs[0]["details"] == {"recipient": "someone@example.com", "status": "skipped"}


def test_test_email_rejects_invalid_address(open_page, admin):
    app = open_page("settings")
    app.text_input(key="org_settings_test_to").input("nobody")
    _button(app, "Send test e-mail").click().run()
    assert app.error
    assert audit_service.list_audit_logs(admin, action_prefix="email.test") == []


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"status": "sent", "error": None}, "Test e-mail sent to someone@example.com"),
        (
            {"status": "failed", "error": "The mail server refused the connection."},
            "Failed: the message could not be delivered",
        ),
    ],
)
def test_test_email_outcomes_are_distinct(open_page, monkeypatch, result, expected):
    monkeypatch.setattr(email_service, "send_email", lambda **kwargs: result)
    app = open_page("settings")
    app.text_input(key="org_settings_test_to").input("someone@example.com")
    _button(app, "Send test e-mail").click().run()
    assert not app.exception and expected in texts(app)


def test_smtp_status_shows_configuration_but_never_the_password(open_page, app_settings, monkeypatch):
    from dataclasses import replace

    from config.settings import set_settings

    set_settings(
        replace(
            app_settings,
            smtp_host="smtp.example.com",
            smtp_from="alerts@example.com",
            smtp_password="TopSecret-123",
        )
    )
    app = open_page("settings")
    assert not app.exception
    assert "smtp.example.com" in texts(app)
    shown = texts(app) + " ".join(str(m.value) for m in app.metric)
    assert "TopSecret-123" not in shown
    assert {m.label: m.value for m in app.metric}["Sender address"].replace("\\", "") == "alerts@example.com"


def test_audit_page_lists_and_filters_actions(open_page, admin, org_id, factory):
    factory.organization("Other Co")
    other_admin = factory.org_admin(factory.organization("Elsewhere Ltd"))
    organization_admin.create_analyst(
        admin, username="newbie", email="n@acme.example", full_name="N", password=STRONG
    )
    organization_admin.update_org_settings(admin, {org_settings.TEXT_PLACEHOLDER: "n/a"})
    organization_admin.create_analyst(
        other_admin, username="foreign", email="f@x.example", full_name="F", password=STRONG
    )
    app = open_page("audit")
    assert not app.exception
    frame = _frame(app, "Action")
    assert {"user.create", "org.settings.update"} <= set(frame["Action"])
    assert "foreign" not in " ".join(frame["Details"])
    app.text_input(key="org_audit_prefix").input("org.").run()
    assert set(_frame(app, "Action")["Action"]) == {"org.settings.update"}
    app.text_input(key="org_audit_prefix").input("nothing.matches").run()
    assert "No audit entries match this filter" in texts(app)


def test_cross_tenant_selection_cannot_reach_other_org(open_page, admin, factory, sales_frame):
    other_org = factory.organization("Elsewhere Ltd")
    other_admin = factory.org_admin(other_org)
    foreign_dataset = factory.dataset(other_admin, sales_frame, "Foreign")
    with session_scope() as db:
        foreign_user = db.query(User).filter(User.organization_id == other_org).first().id
    with pytest.raises(NotFoundError):
        organization_admin.list_dataset_access(admin, foreign_dataset)
    with pytest.raises(NotFoundError):
        organization_admin.set_user_active(admin, foreign_user, False)
    app = open_page("users")
    assert not app.exception
    assert "No datasets yet" in texts(app)
    assert "Foreign" not in texts(app)
