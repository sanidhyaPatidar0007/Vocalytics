"""Super Admin pages: rendering, primary actions and error states."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from auth import authentication
from components.navigation import load_capabilities, page_keys_for
from components.session import CTX_KEY
from core.constants import ALL_FEATURES, Feature
from core.errors import AppError
from modules import super_admin
from services import audit_service
from tests.conftest import DEFAULT_PASSWORD
from tests.ui.conftest import ROOT
from tests.ui.conftest import texts as _raw_texts

PAGES = {
    "overview": "views/platform/overview.py",
    "organizations": "views/platform/organizations.py",
    "plans": "views/platform/plans.py",
    "subscriptions": "views/platform/subscriptions.py",
    "audit": "views/platform/audit.py",
}
STRONG = "Gr3at!Secret-Pass"


def texts(app: AppTest) -> str:
    """Visible text with Markdown escape backslashes removed."""
    return _raw_texts(app).replace("\\", "")


@pytest.fixture
def root_ctx(factory):
    return factory.super_admin()


@pytest.fixture
def root_app(make_app, root_ctx):
    def _open(page: str) -> AppTest:
        return make_app(root_ctx.username, page=PAGES[page])

    return _open


def _button(app: AppTest, label: str):
    matches = [b for b in app.button if b.label == label]
    assert matches, f"no button labelled {label!r}; have {[b.label for b in app.button]}"
    return matches[0]


def _fill_create_org(app: AppTest, name: str, **extra: str) -> None:
    app.text_input(key="platform_orgs_create_name_0").input(name)
    for field, value in extra.items():
        app.text_input(key=f"platform_orgs_create_{field}_0").input(value)


@pytest.mark.parametrize("page", list(PAGES))
def test_every_platform_page_renders_for_super_admin(root_app, page):
    app = root_app(page)
    assert not app.exception
    assert app.title, "page title missing"


def test_overview_shows_metrics_and_error_references(root_app, factory, root_ctx):
    org = factory.organization("Acme Retail")
    factory.user(org, "org_admin", username="oa")
    from services.audit_service import record_error

    reference = record_error("test", RuntimeError("boom"), ctx=root_ctx)
    app = root_app("overview")
    assert not app.exception
    labels = {metric.label: metric.value for metric in app.metric}
    assert labels["Organizations"] == "1"
    assert labels["Errors, last 7 days"] == "1"
    errors = [
        df.value for df in app.dataframe if "Reference" in df.value.columns and "Message" in df.value.columns
    ]
    assert errors and reference in set(errors[0]["Reference"])


def test_platform_pages_blocked_for_org_roles(factory, org_id):
    admin = factory.org_admin(org_id)
    analyst = factory.analyst(org_id)
    for ctx in (admin, analyst):
        keys = [key for keys in page_keys_for(load_capabilities(ctx)).values() for key in keys]
        assert not any(key.startswith("platform_") for key in keys)
    for path in PAGES.values():
        app = AppTest.from_file(str(ROOT / path), default_timeout=60)
        app.session_state[CTX_KEY] = admin
        app.run()
        assert not app.exception
        assert "only available to the platform Super Admin" in texts(app)
        assert not app.button and not app.text_input


def test_super_admin_creates_organization_with_first_admin(root_app, root_ctx):
    app = root_app("organizations")
    _fill_create_org(
        app,
        "Northwind Traders",
        contact="ops@northwind.example",
        admin_username="nora",
        admin_email="nora@northwind.example",
        admin_name="Nora North",
        admin_password=STRONG,
    )
    _button(app, "Create organization").click().run()
    assert not app.exception
    assert "Northwind Traders" in texts(app) and "created" in texts(app)
    orgs = {org["name"]: org for org in super_admin.list_organizations(root_ctx)}
    created = orgs["Northwind Traders"]
    assert created["status"] == "active" and created["contact_email"] == "ops@northwind.example"
    assert created["active_users"] == 1
    login = authentication.login("nora", STRONG)
    assert login.ctx.organization_id == created["id"]
    table = app.dataframe[0].value
    assert "Northwind Traders" in set(table["Name"])
    assert app.text_input(key="platform_orgs_create_name_1").value == "", "form should be empty after success"


def test_organization_names_are_escaped_in_notices(root_app, root_ctx):
    app = root_app("organizations")
    hostile = "![x](http://example.com/p.png) **bold**"
    app.text_input(key="platform_orgs_create_name_0").input(hostile)
    _button(app, "Create organization").click().run()
    assert not app.exception
    notice = app.success[0].value
    assert "![" not in notice and "](" not in notice and r"\*\*bold" in notice
    assert hostile in {org["name"] for org in super_admin.list_organizations(root_ctx)}


def test_create_organization_validation_errors(root_app, root_ctx, org_id):
    app = root_app("organizations")
    _button(app, "Create organization").click().run()
    assert "organization name is required" in texts(app)
    _fill_create_org(app, "Acme Retail")
    _button(app, "Create organization").click().run()
    assert app.error, "duplicate name should be rejected"
    _fill_create_org(app, "Half Admin", admin_username="halfadmin")
    _button(app, "Create organization").click().run()
    assert "username, e-mail and password" in texts(app)
    _fill_create_org(
        app, "Weak Pass Co", admin_username="weakling", admin_email="w@example.com", admin_password="short"
    )
    _button(app, "Create organization").click().run()
    assert app.error
    assert [o["name"] for o in super_admin.list_organizations(root_ctx)] == ["Acme Retail"]


def test_edit_suspend_and_reactivate_organization(root_app, root_ctx, factory, org_id):
    app = root_app("organizations")
    assert not app.exception
    app.text_input(key=f"platform_orgs_edit_name_{org_id}").input("Acme Retail Group")
    app.text_input(key=f"platform_orgs_edit_contact_{org_id}").input("boss@acme.example")
    _button(app, "Save changes").click().run()
    org = super_admin.list_organizations(root_ctx)[0]
    assert org["name"] == "Acme Retail Group" and org["contact_email"] == "boss@acme.example"

    app.text_input(key=f"platform_orgs_edit_contact_{org_id}").input("not-an-email")
    _button(app, "Save changes").click().run()
    assert "valid e-mail" in texts(app)
    assert super_admin.list_organizations(root_ctx)[0]["contact_email"] == "boss@acme.example"

    # suspend: refused without the confirmation, applied with it
    app.selectbox(key=f"platform_orgs_status_choice_{org_id}").select("suspended").run()
    app.text_input(key=f"platform_orgs_status_reason_{org_id}_suspended").input("unpaid invoice")
    _button(app, "Suspend organization").click().run()
    assert "Tick the confirmation" in texts(app)
    assert super_admin.list_organizations(root_ctx)[0]["status"] == "active"
    app.checkbox(key=f"platform_orgs_status_confirm_{org_id}_suspended").check().run()
    _button(app, "Suspend organization").click().run()
    assert not app.exception
    org = super_admin.list_organizations(root_ctx)[0]
    assert org["status"] == "suspended" and org["status_reason"] == "unpaid invoice"
    assert "now suspended" in texts(app)

    _button(app, "Activate organization").click().run()
    org = super_admin.list_organizations(root_ctx)[0]
    assert org["status"] == "active"
    actions = [entry["action"] for entry in audit_service.list_audit_logs(root_ctx, organization_id=org_id)]
    assert "platform.organization.suspended" in actions and "platform.organization.active" in actions


def test_suspended_member_cannot_sign_in(root_app, root_ctx, factory, org_id):
    factory.user(org_id, "org_admin", username="oa")
    app = root_app("organizations")
    app.selectbox(key=f"platform_orgs_status_choice_{org_id}").select("deactivated").run()
    app.checkbox(key=f"platform_orgs_status_confirm_{org_id}_deactivated").check().run()
    _button(app, "Deactivate organization").click().run()
    assert super_admin.list_organizations(root_ctx)[0]["status"] == "deactivated"
    with pytest.raises(AppError):
        authentication.login("oa", DEFAULT_PASSWORD)


def test_add_organization_admin(root_app, root_ctx, org_id):
    app = root_app("organizations")
    app.text_input(key=f"platform_orgs_admin_username_{org_id}_0").input("second.admin")
    app.text_input(key=f"platform_orgs_admin_email_{org_id}_0").input("second@acme.example")
    app.text_input(key=f"platform_orgs_admin_password_{org_id}_0").input(STRONG)
    _button(app, "Add organization admin").click().run()
    assert not app.exception and "second.admin" in texts(app)
    assert authentication.login("second.admin", STRONG).ctx.organization_id == org_id
    app.text_input(key=f"platform_orgs_admin_username_{org_id}_1").input("second.admin")
    app.text_input(key=f"platform_orgs_admin_email_{org_id}_1").input("again@acme.example")
    app.text_input(key=f"platform_orgs_admin_password_{org_id}_1").input(STRONG)
    _button(app, "Add organization admin").click().run()
    assert "already taken" in texts(app)


def _plan(ctx, code: str) -> dict:
    return next(plan for plan in super_admin.list_plans(ctx) if plan["code"] == code)


def test_create_plan_through_form(root_app, root_ctx):
    app = root_app("plans")
    app.text_input(key="platform_plans_create_code_0").input("team")
    app.text_input(key="platform_plans_create_name_0").input("Team")
    app.number_input(key="platform_plans_create_price_0").set_value(19.5)
    app.number_input(key="platform_plans_create_max_users_0").set_value(12)
    app.multiselect(key="platform_plans_create_features_0").select(Feature.DATA_PROFILING.value).select(
        Feature.VOICE_ANALYSIS.value
    )
    _button(app, "Create plan").click().run()
    assert not app.exception and "Team" in texts(app)
    plan = _plan(root_ctx, "team")
    assert plan["price_cents"] == 1950 and plan["max_users"] == 12 and plan["currency"] == "USD"
    assert {key for key, on in plan["features"].items() if on} == {
        Feature.DATA_PROFILING.value,
        Feature.VOICE_ANALYSIS.value,
    }
    assert "team" in set(app.dataframe[0].value["Code"])


def test_create_plan_validation_errors(root_app, root_ctx):
    app = root_app("plans")
    app.text_input(key="platform_plans_create_code_0").input("Bad Code!")
    app.text_input(key="platform_plans_create_name_0").input("Bad")
    _button(app, "Create plan").click().run()
    assert app.error
    app.text_input(key="platform_plans_create_code_0").input("basic")
    _button(app, "Create plan").click().run()
    assert app.error
    assert {plan["code"] for plan in super_admin.list_plans(root_ctx)} == {"basic", "premium"}


def test_toggle_feature_and_save(root_app, root_ctx):
    plan = _plan(root_ctx, "basic")
    assert plan["features"][Feature.VOICE_ANALYSIS.value] is False
    app = root_app("plans")
    basic_id = plan["id"]
    selected = app.selectbox(key="platform_plans_selected")
    selected.select(basic_id).run()
    _button(app, "Save features").click().run()
    assert "No feature was changed" in texts(app)
    app.checkbox(key=f"platform_plans_feature_{basic_id}_{Feature.VOICE_ANALYSIS.value}").check().run()
    assert "1 unsaved change" in texts(app)
    _button(app, "Save features").click().run()
    assert not app.exception and "Saved 1 feature change" in texts(app)
    after = _plan(root_ctx, "basic")
    assert after["features"][Feature.VOICE_ANALYSIS.value] is True
    others = {k: v for k, v in after["features"].items() if k != Feature.VOICE_ANALYSIS.value}
    assert others == {k: v for k, v in plan["features"].items() if k != Feature.VOICE_ANALYSIS.value}
    actions = [e for e in audit_service.list_audit_logs(root_ctx, action_prefix="platform.plan.feature")]
    assert len(actions) == 1


def test_edit_plan_and_toggle_active(root_app, root_ctx):
    plan = _plan(root_ctx, "basic")
    app = root_app("plans")
    app.selectbox(key="platform_plans_selected").select(plan["id"]).run()
    app.text_input(key=f"platform_plans_edit_name_{plan['id']}").input("Basic Plus")
    app.number_input(key=f"platform_plans_edit_max_users_{plan['id']}").set_value(plan["max_users"] + 3)
    _button(app, "Save plan changes").click().run()
    assert not app.exception
    after = _plan(root_ctx, "basic")
    assert after["name"] == "Basic Plus" and after["max_users"] == plan["max_users"] + 3
    assert after["max_storage_mb"] == plan["max_storage_mb"]

    app.number_input(key=f"platform_plans_edit_max_versions_per_dataset_{plan['id']}").set_value(1)
    _button(app, "Save plan changes").click().run()
    assert "at least 2 versions" in texts(app)

    _button(app, "Deactivate plan").click().run()
    assert _plan(root_ctx, "basic")["is_active"] is False
    _button(app, "Activate plan").click().run()
    assert _plan(root_ctx, "basic")["is_active"] is True


def test_plan_without_changes_reports_nothing_to_update(root_app, root_ctx):
    app = root_app("plans")
    _button(app, "Save plan changes").click().run()
    assert "Nothing to update" in texts(app)


def test_assign_plan_and_history_shows_ended_subscription(root_app, root_ctx, org_id):
    app = root_app("subscriptions")
    assert not app.exception
    app.selectbox(key="platform_subscriptions_assign_plan_0").select("basic")
    app.text_input(key="platform_subscriptions_assign_reference_0").input("INV-100")
    app.text_area(key="platform_subscriptions_assign_notes_0").input("downgrade at customer request")
    _button(app, "Assign plan").click().run()
    assert not app.exception and "assigned to Acme Retail" in texts(app)
    history = super_admin.list_subscriptions(root_ctx, organization_id=org_id)
    assert [(row["plan_code"], row["status"]) for row in history] == [
        ("basic", "active"),
        ("premium", "ended"),
    ]
    assert history[0]["billing_reference"] == "INV-100"
    frame = next(
        df.value for df in app.dataframe if "Plan" in df.value.columns and "Assigned by" in df.value.columns
    )
    assert set(frame["Status"]) == {"active", "ended"}

    app.checkbox(key="platform_subscriptions_history_ended").uncheck().run()
    frame = next(df.value for df in app.dataframe if "Assigned by" in df.value.columns)
    assert set(frame["Status"]) == {"active"}


def test_assign_inactive_plan_is_not_offered(root_app, root_ctx, org_id):
    premium = _plan(root_ctx, "premium")
    super_admin.set_plan_active(root_ctx, premium["id"], False)
    app = root_app("subscriptions")
    assert not app.exception
    assert "premium" not in app.selectbox(key="platform_subscriptions_assign_plan_0").options


def test_audit_page_filters_by_prefix_and_organization(root_app, root_ctx, factory, org_id):
    other = factory.organization("Other Co")
    super_admin.update_organization(root_ctx, org_id, name="Acme Retail Ltd")
    super_admin.set_plan_active(root_ctx, _plan(root_ctx, "basic")["id"], False)
    app = root_app("audit")
    assert not app.exception
    frame = next(df.value for df in app.dataframe if "Action" in df.value.columns)
    assert {"platform.organization.update", "platform.plan.deactivate"} <= set(frame["Action"])
    app.text_input(key="platform_audit_prefix").input("platform.plan").run()
    frame = next(df.value for df in app.dataframe if "Action" in df.value.columns)
    assert set(frame["Action"]) == {"platform.plan.deactivate"}
    app.text_input(key="platform_audit_prefix").input("").run()
    app.selectbox(key="platform_audit_org").select(org_id).run()
    frame = next(df.value for df in app.dataframe if "Action" in df.value.columns)
    assert "platform.plan.deactivate" not in set(frame["Action"])
    assert other not in {
        entry["organization_id"] for entry in audit_service.list_audit_logs(root_ctx, organization_id=org_id)
    }


def test_audit_page_lists_error_events(root_app, root_ctx):
    from services.audit_service import record_error

    reference = record_error("pages", ValueError("bad thing"), ctx=root_ctx)
    app = root_app("audit")
    frames = [df.value for df in app.dataframe if "Reference" in df.value.columns]
    assert frames and reference in set(frames[0]["Reference"])


def test_all_features_are_toggleable_on_plans_page(root_app, root_ctx):
    app = root_app("plans")
    plan_id = app.selectbox(key="platform_plans_selected").value
    keys = {c.key for c in app.checkbox}
    assert {f"platform_plans_feature_{plan_id}_{feature}" for feature in ALL_FEATURES} <= keys
