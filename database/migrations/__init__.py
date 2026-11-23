"""Forward-only, versioned schema migrations.

Each migration runs inside a transaction and is recorded in
``schema_migrations``. Before an existing database is upgraded, a consistent
copy is written to ``BACKUP_DIR`` with SQLite's online backup API, so a failed
upgrade can be rolled back by restoring that file.

Adding a migration: append ``Migration(<next version>, "<name>", <function>)``
to ``MIGRATIONS``. Write the function idempotently (check before ALTER) and
never edit a migration that has already shipped.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine

from config.settings import get_settings
from core import timeutil
from database.models import Base, SchemaMigration


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[Connection], None]


def _m0001_initial_schema(connection: Connection) -> None:
    Base.metadata.create_all(connection)


def _m0002_version_file_pending_delete(connection: Connection) -> None:
    columns = {c["name"] for c in inspect(connection).get_columns("dataset_versions")}
    if "file_pending_delete" not in columns:
        connection.execute(
            text("ALTER TABLE dataset_versions ADD COLUMN file_pending_delete BOOLEAN NOT NULL DEFAULT 0")
        )
        # Files of versions purged before this column existed may have been left behind:
        # flag them so the next publish or worker run checks and deletes them.
        connection.execute(text("UPDATE dataset_versions SET file_pending_delete = 1 WHERE is_purged = 1"))


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "initial schema", _m0001_initial_schema),
    Migration(2, "track purged version files awaiting deletion", _m0002_version_file_pending_delete),
)


def current_version(engine: Engine) -> int:
    if "schema_migrations" not in inspect(engine).get_table_names():
        return 0
    with engine.connect() as connection:
        value = connection.execute(text("SELECT MAX(version) FROM schema_migrations")).scalar()
    return int(value or 0)


def backup_database(db_path: Path, backup_dir: Path, label: str) -> Path | None:
    """Write a consistent copy of the SQLite file (works with WAL)."""
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = timeutil.utcnow().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"{db_path.stem}-{stamp}-{label}.db"
    source = sqlite3.connect(str(db_path))
    destination = sqlite3.connect(str(target))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target


def apply_migrations(engine: Engine, *, backup: bool = True) -> list[int]:
    """Apply pending migrations in order. Returns the versions applied."""
    current = current_version(engine)
    pending = [m for m in MIGRATIONS if m.version > current]
    if not pending:
        return []
    if backup and current > 0:
        settings = get_settings()
        backup_database(settings.database_path, settings.backup_dir, f"before-v{pending[0].version}")
    applied: list[int] = []
    for migration in pending:
        with engine.begin() as connection:
            migration.apply(connection)
            SchemaMigration.__table__.create(connection, checkfirst=True)
            connection.execute(
                SchemaMigration.__table__.insert().values(
                    version=migration.version, name=migration.name, applied_at=timeutil.utcnow()
                )
            )
        applied.append(migration.version)
    return applied
