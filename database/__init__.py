"""Database package: models, engine/session handling, migrations and seeding."""

from __future__ import annotations

from config.settings import get_settings
from database.connection import get_engine, new_session, reset_engine, session_scope


def init_database(*, seed: bool = True) -> dict:
    """Create folders, apply migrations and seed default plans (idempotent)."""
    from database.migrations import apply_migrations
    from database.seed import seed_default_plans

    settings = get_settings()
    for folder in (
        settings.database_path.parent,
        settings.storage_root,
        settings.log_dir,
        settings.backup_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)
    applied = apply_migrations(get_engine())
    seeded: list[str] = []
    if seed:
        with session_scope() as db:
            seeded = seed_default_plans(db)
    return {"applied_migrations": applied, "seeded_plans": seeded}


__all__ = ["get_engine", "init_database", "new_session", "reset_engine", "session_scope"]
