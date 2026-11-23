"""Schema migrations owned by the versioning kernel."""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy import create_engine, inspect, text

from core import timeutil
from database.migrations import apply_migrations, current_version
from database.models import Base, SchemaMigration


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35, 0), reason="DROP COLUMN needs SQLite 3.35")
def test_version_1_database_gains_the_pending_delete_column(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    try:
        with engine.begin() as connection:  # a database created by the version 1 schema
            Base.metadata.create_all(connection)
            connection.execute(text("ALTER TABLE dataset_versions DROP COLUMN file_pending_delete"))
            connection.execute(
                SchemaMigration.__table__.insert().values(
                    version=1, name="initial schema", applied_at=timeutil.utcnow()
                )
            )
            for number, purged in ((1, 0), (2, 1)):  # foreign keys are not enforced on this engine
                connection.execute(
                    text(
                        "INSERT INTO dataset_versions (organization_id, dataset_id, version_number, file_path, "
                        "file_sha256, size_bytes, row_count, column_count, schema_json, operation_type, note, "
                        "is_pinned, is_purged, created_at) VALUES (1, 1, :n, :path, 'x', 1, 1, 1, '[]', 'import', "
                        "'', 0, :purged, '2026-01-01')"
                    ),
                    {"n": number, "path": f"orgs/1/versions/1/v{number:04d}.parquet", "purged": purged},
                )
        assert "file_pending_delete" not in {c["name"] for c in inspect(engine).get_columns("dataset_versions")}
        assert apply_migrations(engine, backup=False) == [2]
        assert current_version(engine) >= 2
        assert "file_pending_delete" in {c["name"] for c in inspect(engine).get_columns("dataset_versions")}
        with engine.connect() as connection:  # already purged versions are re-checked for left-over files
            flags = connection.execute(
                text("SELECT version_number, file_pending_delete FROM dataset_versions ORDER BY version_number")
            ).all()
        assert [tuple(row) for row in flags] == [(1, 0), (2, 1)]
        assert apply_migrations(engine, backup=False) == []
    finally:
        engine.dispose()
