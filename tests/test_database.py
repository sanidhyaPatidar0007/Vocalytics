"""Database layer: foreign-key enforcement on real connections and the migration backup/upgrade path."""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from config.settings import get_settings
from database import get_engine, session_scope
from database import migrations as migration_module
from database.migrations import Migration, apply_migrations, backup_database, current_version
from database.models import UsageRecord

# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------


def test_foreign_keys_pragma_is_on_for_every_connection():
    engine = get_engine()
    for _ in range(3):
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_orphan_tenant_row_is_rejected_by_the_database():
    with pytest.raises(IntegrityError, match="FOREIGN KEY"), session_scope() as db:
        db.add(UsageRecord(organization_id=9999, metric="queries", quantity=1.0))


def test_orphan_row_is_rejected_by_raw_sql_on_a_pooled_connection():
    with pytest.raises(IntegrityError, match="FOREIGN KEY"), get_engine().begin() as connection:
        connection.execute(
            text(
                "INSERT INTO usage_records (organization_id, metric, quantity, details_json, created_at) "
                "VALUES (9999, 'queries', 1.0, '{}', CURRENT_TIMESTAMP)"
            )
        )


def test_row_with_an_existing_organization_is_accepted(factory):
    org_id = factory.organization()
    with session_scope() as db:
        db.add(UsageRecord(organization_id=org_id, metric="queries", quantity=1.0))


# ---------------------------------------------------------------------------
# Migrations: backup before upgrade
# ---------------------------------------------------------------------------


def _probe_migration(version: int, *, fail: bool = False) -> Migration:
    def apply(connection) -> None:  # noqa: ANN001 - SQLAlchemy connection
        connection.execute(text("CREATE TABLE migration_probe (id INTEGER PRIMARY KEY)"))
        if fail:
            raise RuntimeError("simulated migration failure")

    return Migration(version, "probe", apply)


def _latest() -> int:
    return max(m.version for m in migration_module.MIGRATIONS)


def _backups() -> list:
    directory = get_settings().backup_dir
    return sorted(directory.glob("*.db")) if directory.exists() else []


def test_registry_is_contiguous_and_the_database_is_at_the_latest_version():
    versions = [m.version for m in migration_module.MIGRATIONS]
    assert versions == list(range(1, len(versions) + 1))
    assert current_version(get_engine()) == _latest()
    assert apply_migrations(get_engine()) == []
    assert _backups() == []  # nothing pending: no backup is written


def test_upgrading_an_existing_database_writes_a_backup_first(monkeypatch):
    before = _latest()
    monkeypatch.setattr(
        migration_module, "MIGRATIONS", (*migration_module.MIGRATIONS, _probe_migration(before + 1))
    )
    engine = get_engine()
    assert apply_migrations(engine) == [before + 1]
    assert current_version(engine) == before + 1
    assert "migration_probe" in inspect(engine).get_table_names()
    [backup] = _backups()
    assert backup.name.endswith(f"-before-v{before + 1}.db")
    # The backup is a consistent copy of the database as it was before the upgrade.
    copy = sqlite3.connect(str(backup))
    try:
        tables = {r[0] for r in copy.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "migration_probe" not in tables and "schema_migrations" in tables
        assert copy.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == before
    finally:
        copy.close()


def test_backup_can_be_skipped_explicitly(monkeypatch):
    before = _latest()
    monkeypatch.setattr(
        migration_module, "MIGRATIONS", (*migration_module.MIGRATIONS, _probe_migration(before + 1))
    )
    assert apply_migrations(get_engine(), backup=False) == [before + 1]
    assert _backups() == []


def test_failed_upgrade_keeps_the_version_and_the_backup_is_restorable(monkeypatch):
    before = _latest()
    monkeypatch.setattr(
        migration_module,
        "MIGRATIONS",
        (*migration_module.MIGRATIONS, _probe_migration(before + 1, fail=True)),
    )
    with pytest.raises(RuntimeError, match="simulated migration failure"):
        apply_migrations(get_engine())
    assert current_version(get_engine()) == before  # the failed migration is not recorded
    [backup] = _backups()
    copy = sqlite3.connect(str(backup))
    try:
        assert copy.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == before
        assert copy.execute("SELECT COUNT(*) FROM subscription_plans").fetchone()[0] > 0
    finally:
        copy.close()


def test_backup_database_skips_a_missing_or_empty_file(tmp_path):
    assert backup_database(tmp_path / "missing.db", tmp_path / "backups", "x") is None
    empty = tmp_path / "empty.db"
    empty.write_bytes(b"")
    assert backup_database(empty, tmp_path / "backups", "x") is None
    assert not (tmp_path / "backups").exists()
