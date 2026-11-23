"""Setup scripts: database initialization, Super Admin creation, diagnostics.

The interactive prompt is driven with mocked input()/getpass(); the real
terminal interaction is not exercised by automated tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from core.errors import ConflictError, ValidationError
from database import session_scope
from database.models import AuditLog, User
from scripts import check_environment, create_super_admin, initialize_database
from tests.conftest import DEFAULT_PASSWORD


def _inputs(*values):
    iterator = iter(values)
    return lambda _prompt="": next(iterator)


def test_initialize_database_is_idempotent(capsys):
    assert initialize_database.main([]) == 0
    assert initialize_database.main([]) == 0
    out = capsys.readouterr().out
    assert "already up to date" in out
    assert initialize_database.main(["--status"]) == 0
    latest = initialize_database.MIGRATIONS[-1].version
    assert f"Schema version: {latest} (latest available: {latest})" in capsys.readouterr().out


def test_create_super_admin_stores_only_a_hash():
    user_id = create_super_admin.create_super_admin(
        "Root.Admin", "root@example.com", DEFAULT_PASSWORD, "Root"
    )
    with session_scope() as db:
        user = db.get(User, user_id)
        assert user.username == "root.admin" and user.role == "super_admin" and user.organization_id is None
        assert DEFAULT_PASSWORD not in user.password_hash and user.password_hash.startswith("$argon2id$")
        audit = db.execute(
            select(AuditLog).where(AuditLog.action == "platform.super_admin.create")
        ).scalar_one()
        assert DEFAULT_PASSWORD not in audit.details_json


@pytest.mark.parametrize(
    ("username", "email", "password", "error"),
    [
        ("ab", "a@example.com", DEFAULT_PASSWORD, ValidationError),
        ("valid_name", "not-an-email", DEFAULT_PASSWORD, ValidationError),
        ("valid_name", "a@example.com", "short", ValidationError),
        ("valid_name", "a@example.com", "alllowercaseletters", ValidationError),
    ],
)
def test_create_super_admin_validation(username, email, password, error):
    with pytest.raises(error):
        create_super_admin.create_super_admin(username, email, password)


def test_duplicate_username_is_rejected():
    create_super_admin.create_super_admin("boss", "boss@example.com", DEFAULT_PASSWORD)
    with pytest.raises(ConflictError):
        create_super_admin.create_super_admin("BOSS", "other@example.com", DEFAULT_PASSWORD)


def test_interactive_prompt_retries_mismatched_password(capsys):
    code = create_super_admin.main(
        [],
        input_fn=_inputs("admin1", "admin1@example.com", "Admin One"),
        getpass_fn=_inputs(DEFAULT_PASSWORD, "different", DEFAULT_PASSWORD, DEFAULT_PASSWORD),
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "do not match" in output and DEFAULT_PASSWORD not in output
    assert create_super_admin.super_admin_count() == 1


def test_interactive_prompt_refuses_second_admin_without_flag(capsys):
    create_super_admin.create_super_admin("first", "first@example.com", DEFAULT_PASSWORD)
    assert create_super_admin.main([], input_fn=_inputs(), getpass_fn=_inputs()) == 1
    assert "already exist" in capsys.readouterr().out
    code = create_super_admin.main(
        ["--allow-additional"],
        input_fn=_inputs("second", "second@example.com", ""),
        getpass_fn=_inputs(DEFAULT_PASSWORD, DEFAULT_PASSWORD),
    )
    assert code == 0 and create_super_admin.super_admin_count() == 2


def test_if_missing_skips_quietly_when_an_admin_exists_and_creates_one_otherwise(capsys):
    """The one-click launchers call the script on every first-run attempt."""
    code = create_super_admin.main(
        ["--if-missing"],
        input_fn=_inputs("owner", "owner@example.com", ""),
        getpass_fn=_inputs(DEFAULT_PASSWORD, DEFAULT_PASSWORD),
    )
    assert code == 0 and create_super_admin.super_admin_count() == 1
    assert create_super_admin.main(["--if-missing"], input_fn=_inputs(), getpass_fn=_inputs()) == 0
    assert "already exists" in capsys.readouterr().out
    assert create_super_admin.super_admin_count() == 1


def test_interactive_prompt_gives_up_after_three_weak_passwords():
    code = create_super_admin.main(
        [],
        input_fn=_inputs("admin2", "admin2@example.com", ""),
        getpass_fn=_inputs("weak", "weak", "weak", "weak", "weak", "weak"),
    )
    assert code == 1 and create_super_admin.super_admin_count() == 0


def test_from_env_mode(monkeypatch):
    monkeypatch.setenv("SUPER_ADMIN_USERNAME", "envadmin")
    monkeypatch.setenv("SUPER_ADMIN_EMAIL", "env@example.com")
    monkeypatch.setenv("SUPER_ADMIN_PASSWORD", DEFAULT_PASSWORD)
    assert create_super_admin.main(["--from-env"]) == 0
    monkeypatch.delenv("SUPER_ADMIN_PASSWORD")
    assert create_super_admin.main(["--from-env", "--allow-additional"]) == 1


def test_seed_requires_a_super_admin(capsys):
    from scripts import seed_demo_data

    assert seed_demo_data.main() == 1
    assert "Create a Super Admin first" in capsys.readouterr().out


def test_seed_demo_data_creates_org_users_and_datasets(capsys):
    from auth.password_manager import validate_password_strength
    from scripts import seed_demo_data
    from services import dataset_service, versioning_service

    for _ in range(20):
        validate_password_strength(seed_demo_data.generate_password())
    create_super_admin.create_super_admin("platform", "platform@example.com", DEFAULT_PASSWORD)
    assert seed_demo_data.main() == 0
    out = capsys.readouterr().out
    assert "demo.admin" in out and "demo.analyst" in out
    with session_scope() as db:
        analyst_id = db.execute(select(User.id).where(User.username == "demo.analyst")).scalar_one()
        from auth.authorization import load_context

        analyst = load_context(db, analyst_id)
    names = {d["name"] for d in dataset_service.list_datasets(analyst)}
    assert names == {"Sales 2024", "Sales 2025 Q1", "Customers"}
    sales = next(d for d in dataset_service.list_datasets(analyst) if d["name"] == "Sales 2024")
    frame = versioning_service.load_frame(analyst, sales["dataset_id"])
    assert str(frame["revenue"].dtype) == "float64" and str(frame["quantity"].dtype) == "Int64"
    assert frame["customer_id"].str.startswith("C0").all()
    assert frame.duplicated().sum() > 0 and frame["quantity"].isna().sum() > 0
    assert seed_demo_data.main() == 0
    assert "already exists" in capsys.readouterr().out


def test_environment_checks_report_without_failures():
    checks = check_environment.run_checks()
    names = {c.name: c for c in checks}
    assert names["Python"].status == "PASS"
    assert all(c.status != "FAIL" for c in checks), [c for c in checks if c.status == "FAIL"]
    assert names["Ollama"].status in {"PASS", "WARN"}
    assert all("password" not in c.detail.lower() for c in checks)


def test_environment_checks_report_the_regex_engine():
    """Cleaning runs user patterns on the pinned third-party regex engine (SEC1)."""
    names = {c.name: c for c in check_environment.run_checks()}
    assert names["Package regex"].status == "PASS"
    assert check_environment._pins()["regex"] == "2026.9.29"
