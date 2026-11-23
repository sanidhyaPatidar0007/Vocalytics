"""Create a Super Admin account through a secure interactive prompt.

There is no default or hardcoded password. The password is typed twice with
echo disabled (getpass), checked against the password policy and stored only
as an Argon2id hash.

Usage (PowerShell, from the project root):

    python scripts\\create_super_admin.py
    python scripts\\create_super_admin.py --allow-additional   # add another Super Admin

Automation (for example a provisioning script): set SUPER_ADMIN_USERNAME,
SUPER_ADMIN_EMAIL and SUPER_ADMIN_PASSWORD in the environment and pass
--from-env. Remove the variables afterwards.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select  # noqa: E402

from auth.password_manager import hash_password, validate_password_strength  # noqa: E402
from core.constants import Role  # noqa: E402
from core.errors import AppError, ConflictError, ValidationError  # noqa: E402
from core.logging_setup import configure_logging  # noqa: E402
from database import init_database, session_scope  # noqa: E402
from database.models import User  # noqa: E402
from services.audit_service import record_audit  # noqa: E402

_USERNAME = re.compile(r"^[a-z0-9._-]{3,64}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def super_admin_count() -> int:
    with session_scope() as db:
        return int(
            db.execute(select(func.count(User.id)).where(User.role == Role.SUPER_ADMIN.value)).scalar_one()
        )


def create_super_admin(username: str, email: str, password: str, full_name: str = "") -> int:
    """Validate and create a Super Admin. Returns the new user id."""
    name = (username or "").strip().lower()
    if not _USERNAME.match(name):
        raise ValidationError(
            "Username must be 3-64 characters: lowercase letters, digits, dot, dash or underscore."
        )
    address = (email or "").strip()
    if not _EMAIL.match(address) or len(address) > 254:
        raise ValidationError("Enter a valid e-mail address.")
    validate_password_strength(password, username=name)
    with session_scope() as db:
        if db.execute(select(User.id).where(User.username == name)).first():
            raise ConflictError("That username is already taken.")
        user = User(
            organization_id=None,
            username=name,
            email=address,
            full_name=(full_name or "").strip()[:200],
            role=Role.SUPER_ADMIN.value,
            password_hash=hash_password(password),
            is_active=True,
        )
        db.add(user)
        db.flush()
        record_audit(
            db,
            None,
            "platform.super_admin.create",
            target_type="user",
            target_id=user.id,
            details={"username": name, "method": "setup script"},
        )
        return user.id


def _prompt(label: str, *, input_fn=input) -> str:
    value = input_fn(label).strip()
    while not value:
        value = input_fn(label).strip()
    return value


def main(argv: list[str] | None = None, *, input_fn=input, getpass_fn=getpass.getpass) -> int:
    parser = argparse.ArgumentParser(description="Create a Super Admin account.")
    parser.add_argument("--allow-additional", action="store_true", help="create another Super Admin")
    parser.add_argument("--from-env", action="store_true", help="read SUPER_ADMIN_* environment variables")
    parser.add_argument(
        "--if-missing", action="store_true", help="do nothing (exit 0) when a Super Admin already exists"
    )
    args = parser.parse_args(argv)
    configure_logging()
    init_database(seed=True)
    existing = super_admin_count()
    if existing and args.if_missing:
        print("A Super Admin account already exists; nothing to do.")
        return 0
    if existing and not args.allow_additional:
        print(f"{existing} Super Admin account(s) already exist. Use --allow-additional to add another.")
        return 1
    try:
        if args.from_env:
            username = os.environ.get("SUPER_ADMIN_USERNAME", "")
            email = os.environ.get("SUPER_ADMIN_EMAIL", "")
            password = os.environ.get("SUPER_ADMIN_PASSWORD", "")
            full_name = os.environ.get("SUPER_ADMIN_FULL_NAME", "")
            if not (username and email and password):
                print("SUPER_ADMIN_USERNAME, SUPER_ADMIN_EMAIL and SUPER_ADMIN_PASSWORD must all be set.")
                return 1
        else:
            print("Create the Super Admin account. The password is not shown while you type.")
            username = _prompt("Username: ", input_fn=input_fn)
            email = _prompt("E-mail: ", input_fn=input_fn)
            full_name = input_fn("Full name (optional): ").strip()
            for _attempt in range(3):
                password = getpass_fn("Password: ")
                if getpass_fn("Repeat password: ") != password:
                    print("The passwords do not match. Try again.")
                    continue
                try:
                    validate_password_strength(password, username=username.strip().lower())
                    break
                except ValidationError as exc:
                    print(exc.user_message)
            else:
                print("No valid password entered. Nothing was created.")
                return 1
        user_id = create_super_admin(username, email, password, full_name)
    except AppError as exc:
        print(f"Not created: {exc.user_message}")
        return 1
    print(
        f"Super Admin '{username.strip().lower()}' created (user id {user_id}). "
        "Start the app with: streamlit run app.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
