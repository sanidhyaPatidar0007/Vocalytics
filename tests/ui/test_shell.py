"""App shell: sign-in form, session expiry, sign-out, role navigation, escaping."""

from __future__ import annotations

from datetime import timedelta

from components.feedback import md_escape
from components.navigation import Capabilities, page_keys_for
from core import timeutil
from core.constants import Feature, Permission, Role
from tests.conftest import DEFAULT_PASSWORD
from tests.ui.conftest import texts


def _sign_in(app, username: str, password: str):
    app.text_input[0].input(username)
    app.text_input[1].input(password)
    app.button[0].click().run()
    return app


def test_sign_in_page_renders(make_app):
    app = make_app()
    assert not app.exception
    assert app.title[0].value == "Voice-Enabled Data Analyst"
    assert [b.label for b in app.button] == ["Sign in"]


def test_wrong_password_shows_generic_error(make_app, factory, org_id):
    factory.user(org_id, "org_admin", username="dana")
    app = _sign_in(make_app(), "dana", "Wrong-Passw0rd!")
    assert not app.exception
    assert "Invalid username or password" in texts(app)
    assert app.title[0].value == "Voice-Enabled Data Analyst"


def test_sign_in_and_sign_out(make_app, factory, org_id):
    factory.user(org_id, "org_admin", username="dana")
    app = _sign_in(make_app(), "dana", DEFAULT_PASSWORD)
    assert not app.exception
    assert app.title[0].value != "Voice-Enabled Data Analyst"
    assert [b.label for b in app.sidebar.button] == ["Sign out"]
    app.sidebar.button[0].click().run()
    assert app.title[0].value == "Voice-Enabled Data Analyst"
    assert "signed out" in texts(app)


def test_expired_session_returns_to_sign_in(make_app, factory, org_id, monkeypatch):
    factory.user(org_id, "analyst", username="ana")
    app = make_app("ana")
    assert app.title[0].value != "Voice-Enabled Data Analyst"
    later = timeutil.utcnow() + timedelta(hours=13)
    monkeypatch.setattr("core.timeutil.utcnow", lambda: later)
    app.run()
    assert app.title[0].value == "Voice-Enabled Data Analyst"
    assert "expired" in texts(app).lower()


def test_suspended_org_session_ends(make_app, factory, org_id):
    from database import session_scope
    from database.models import Organization

    factory.user(org_id, "analyst", username="ana")
    app = make_app("ana")
    with session_scope() as db:
        db.get(Organization, org_id).status = "suspended"
    app.run()
    assert app.title[0].value == "Voice-Enabled Data Analyst"
    assert "not active" in texts(app)


def _caps(role: Role, features=(), permissions=()):
    return Capabilities(
        role, frozenset(f.value for f in features), frozenset(p.value for p in permissions), "Org", "Plan"
    )


def test_navigation_by_role_plan_and_permission():
    from core.constants import BASIC_FEATURES, PREMIUM_FEATURES

    premium = [Feature(f) for f in PREMIUM_FEATURES]
    basic = [Feature(f) for f in BASIC_FEATURES]
    super_pages = page_keys_for(_caps(Role.SUPER_ADMIN))
    assert set(super_pages) == {"Platform", "Account"}
    admin = page_keys_for(_caps(Role.ORG_ADMIN, premium))
    assert "combine" in admin["Data"] and admin["Alerts"] == ["alert_rules", "alert_events"]
    assert "org_users" in admin["Organization"]
    basic_admin = page_keys_for(_caps(Role.ORG_ADMIN, basic))
    assert (
        "combine" not in basic_admin["Data"]
        and "Alerts" not in basic_admin
        and "reports" not in basic_admin["Analysis"]
    )
    reader = page_keys_for(_caps(Role.ANALYST, premium, [Permission.EXPORT_DATA]))
    assert reader["Data"] == ["datasets", "quality", "versions"]
    assert "Organization" not in reader and "Alerts" not in reader
    cleaner = page_keys_for(_caps(Role.ANALYST, premium, [Permission.CLEAN_DATA, Permission.SAVE_REPORTS]))
    assert {"missing", "duplicates", "cleaning", "combine"} <= set(cleaner["Data"])
    assert "reports" in cleaner["Analysis"]


def test_markdown_escaping_neutralises_injection():
    payload = "![x](http://example.com/p.png) **bold** :red[alert] <b>tag</b>"
    escaped = md_escape(payload)
    assert "![" not in escaped and "](" not in escaped
    assert "\\*\\*" in escaped and "\\:red\\[" in escaped


def test_must_change_password_limits_navigation_to_account():
    caps = Capabilities(Role.ORG_ADMIN, frozenset(), frozenset(), "Org", "Plan", True)
    assert page_keys_for(caps) == {"Account": ["account"]}


def test_admin_created_account_must_choose_own_password_first(make_app, admin_ctx):
    """Regression: accounts created by an admin start with a password the admin knows,
    so the user must replace it before any other page is offered."""
    from components.navigation import CAPS_KEY
    from modules import organization_admin

    organization_admin.create_analyst(
        admin_ctx, username="newbie", email="newbie@example.com", full_name="New Bie", password=DEFAULT_PASSWORD
    )
    app = _sign_in(make_app(), "newbie", DEFAULT_PASSWORD)
    assert not app.exception
    assert app.title[0].value == "Account"
    assert "set by an administrator" in texts(app)
    assert page_keys_for(app.session_state[CAPS_KEY]) == {"Account": ["account"]}

    def field(label):
        return next(t for t in app.text_input if t.label == label)

    field("Current password").input(DEFAULT_PASSWORD)
    field("New password").input("Fresh!Passw0rd-42")
    field("Repeat new password").input("Fresh!Passw0rd-42")
    next(b for b in app.button if b.label == "Change password").click().run()
    assert not app.exception
    assert "Password changed" in texts(app)
    offered = page_keys_for(app.session_state[CAPS_KEY])
    assert "Data" in offered and "Analysis" in offered


def test_table_download_button_hidden_only_for_analysts_without_export(make_app, factory, org_id):
    """Regression (live browser check): Streamlit's built-in table 'Download as CSV' is a file
    download, so analysts without the export permission must not be offered it."""
    marker = 'aria-label="Download as CSV"'
    factory.user(org_id, "analyst", username="noexport", permissions=set())
    hidden = make_app("noexport")
    assert any(marker in str(m.value) for m in hidden.markdown)
    factory.user(org_id, "analyst", username="exporter", permissions={"export_data"})
    shown = make_app("exporter")
    assert not any(marker in str(m.value) for m in shown.markdown)
    factory.user(org_id, "org_admin", username="boss")
    assert not any(marker in str(m.value) for m in make_app("boss").markdown)


def test_account_page_escapes_organization_name_and_username(make_app, factory):
    """UI6: st.metric renders values as Markdown, so names go through md_escape."""
    org_name = "![x](http://127.0.0.1:9/x.png)"
    org = factory.organization(org_name)
    factory.user(org, "org_admin", username="_bob_")
    app = make_app("_bob_", page="views/account.py")
    assert not app.exception
    values = {m.label: m.value for m in app.metric}
    assert values["Organization"] == md_escape(org_name) and "![" not in values["Organization"]
    assert values["Username"] == md_escape("_bob_")
