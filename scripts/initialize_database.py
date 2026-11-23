"""Create or upgrade the SQLite database and seed the default plans.

Usage (PowerShell, from the project root):

    python scripts\\initialize_database.py            # create / upgrade (idempotent)
    python scripts\\initialize_database.py --status   # show the schema version only

Upgrading an existing database first writes a consistent backup to BACKUP_DIR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from core.logging_setup import configure_logging  # noqa: E402
from database import get_engine, init_database  # noqa: E402
from database.migrations import MIGRATIONS, current_version  # noqa: E402


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _has_super_admin() -> bool:
    from sqlalchemy import select

    from core.constants import Role
    from database import session_scope
    from database.models import User

    with session_scope() as db:
        return db.execute(select(User.id).where(User.role == Role.SUPER_ADMIN.value).limit(1)).first() is not None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize or upgrade the application database.")
    parser.add_argument("--status", action="store_true", help="only report the current schema version")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    latest = MIGRATIONS[-1].version
    if args.status:
        version = current_version(get_engine()) if settings.database_path.exists() else 0
        print(f"Database: {_display(settings.database_path)}")
        print(f"Schema version: {version} (latest available: {latest})")
        return 0
    result = init_database(seed=True)
    print(f"Database: {_display(settings.database_path)}")
    print(f"Applied migrations: {result['applied_migrations'] or 'none (already up to date)'}")
    print(f"Seeded plans: {result['seeded_plans'] or 'none (already present)'}")
    print(f"Storage folder: {_display(settings.storage_root)}")
    if not _has_super_admin():
        print("Next step: python scripts\\create_super_admin.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
