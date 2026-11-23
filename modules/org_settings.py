"""Per-organization settings stored in ``organizations.settings_json``.

Only keys defined here are accepted. Reading falls back to global defaults.
Writing is done by ``modules.organization_admin.update_org_settings`` (Org Admin only).
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from config.settings import get_settings
from core.errors import NotFoundError, ValidationError
from database.models import Organization

MISSING_PLACEHOLDERS = "missing_placeholders"
DATE_DAYFIRST = "date_dayfirst"
TEXT_PLACEHOLDER = "text_missing_placeholder"
ALERT_DEFAULT_RECIPIENTS = "alert_default_recipients"

ALLOWED_KEYS = (MISSING_PLACEHOLDERS, DATE_DAYFIRST, TEXT_PLACEHOLDER, ALERT_DEFAULT_RECIPIENTS)


def defaults() -> dict[str, Any]:
    settings = get_settings()
    return {
        MISSING_PLACEHOLDERS: list(settings.default_missing_placeholders),
        DATE_DAYFIRST: settings.default_date_dayfirst,
        TEXT_PLACEHOLDER: "Not Provided",
        ALERT_DEFAULT_RECIPIENTS: [],
    }


def get_org_settings(db: Session, organization_id: int) -> dict[str, Any]:
    org = db.get(Organization, organization_id)
    if org is None:
        raise NotFoundError("Organization not found.")
    stored = json.loads(org.settings_json or "{}")
    merged = defaults()
    merged.update({k: v for k, v in stored.items() if k in ALLOWED_KEYS})
    return merged


def validate_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a settings update. Unknown keys are rejected."""
    clean: dict[str, Any] = {}
    for key, value in changes.items():
        if key not in ALLOWED_KEYS:
            raise ValidationError(f"Unknown setting '{key}'.")
        if key == MISSING_PLACEHOLDERS:
            if not isinstance(value, (list, tuple)):
                raise ValidationError("Missing-value placeholders must be a list of strings.")
            tokens = []
            for item in value:
                token = str(item).strip()
                if not token:
                    continue
                if len(token) > 50:
                    raise ValidationError("Each placeholder must be at most 50 characters.")
                if token not in tokens:
                    tokens.append(token)
            if len(tokens) > 50:
                raise ValidationError("At most 50 placeholders can be configured.")
            clean[key] = tokens
        elif key == DATE_DAYFIRST:
            if not isinstance(value, bool):
                raise ValidationError("Date order must be true (day first) or false (month first).")
            clean[key] = value
        elif key == TEXT_PLACEHOLDER:
            text = str(value).strip()
            if not text or len(text) > 50:
                raise ValidationError("The text placeholder must be 1 to 50 characters.")
            clean[key] = text
        elif key == ALERT_DEFAULT_RECIPIENTS:
            if not isinstance(value, (list, tuple)):
                raise ValidationError("Alert recipients must be a list of e-mail addresses.")
            emails = [str(v).strip() for v in value if str(v).strip()]
            for email in emails:
                if "@" not in email or len(email) > 254 or any(c in email for c in "\r\n,;"):
                    raise ValidationError(f"Invalid e-mail address: {email[:60]}")
            clean[key] = emails
    return clean


def get_import_defaults(ctx) -> dict[str, Any]:  # noqa: ANN001 - AuthContext (imported lazily to avoid a cycle)
    """Upload defaults any member of the organization may read.

    Analysts need the organization's missing-value placeholders and date order
    to prefill the import form; the rest of the settings stay admin-only.
    """
    from auth.authorization import authorize
    from database.connection import session_scope

    with session_scope() as db:
        org_id = authorize(db, ctx)
        current = get_org_settings(db, org_id)
        return {key: current[key] for key in (MISSING_PLACEHOLDERS, DATE_DAYFIRST, TEXT_PLACEHOLDER)}
