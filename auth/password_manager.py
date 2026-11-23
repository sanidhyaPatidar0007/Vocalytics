"""Password hashing with Argon2id (salted, memory-hard)."""

from __future__ import annotations

import re

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from config.settings import get_settings
from core.errors import ValidationError

_COMMON_PASSWORDS = frozenset(
    {
        "password",
        "password1",
        "password123",
        "123456789",
        "1234567890",
        "qwertyuiop",
        "letmein123",
        "welcome123",
        "admin12345",
        "iloveyou123",
    }
)


def _hasher() -> PasswordHasher:
    s = get_settings()
    return PasswordHasher(
        time_cost=s.password_hash_time_cost,
        memory_cost=s.password_hash_memory_kib,
        parallelism=s.password_hash_parallelism,
    )


def hash_password(password: str) -> str:
    """Return an Argon2id hash. A random salt is generated for every call."""
    if not isinstance(password, str) or not password:
        raise ValidationError("A password is required.")
    return _hasher().hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Check a password against a stored hash. Never raises for a wrong password."""
    if not password_hash or not isinstance(password, str):
        return False
    try:
        return _hasher().verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _hasher().check_needs_rehash(password_hash)
    except InvalidHashError:
        return True


def validate_password_strength(password: str, *, username: str | None = None) -> None:
    """Enforce the password policy. Raises ValidationError describing the problem."""
    min_length = get_settings().password_min_length
    if not isinstance(password, str) or len(password) < min_length:
        raise ValidationError(f"Password must be at least {min_length} characters long.")
    if len(password) > 256:
        raise ValidationError("Password must be at most 256 characters long.")
    classes = sum(
        bool(re.search(pattern, password)) for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]")
    )
    if classes < 3:
        raise ValidationError(
            "Password must contain at least three of: lowercase letters, uppercase letters, digits, symbols."
        )
    if password.lower() in _COMMON_PASSWORDS:
        raise ValidationError("This password is too common. Choose a different one.")
    if username and len(username) >= 3 and username.lower() in password.lower():
        raise ValidationError("Password must not contain the username.")
