"""Audit trail for security-sensitive and data-changing actions, plus error capture."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from sqlalchemy import select

from auth.authorization import AuthContext, authorize, authorize_super_admin
from core.constants import Role
from core.logging_setup import redact
from database.connection import session_scope
from database.models import AuditLog, ErrorEvent

logger = logging.getLogger(__name__)

_SENSITIVE_KEY = re.compile(r"(?i)pass|secret|token|hash|credential|smtp|path|cookie")
_MAX_STRING = 500
_MAX_ITEMS = 50


def sanitize_details(value: Any, depth: int = 0) -> Any:
    """Remove secrets and file paths, and bound the size of audit details."""
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                clean["..."] = f"{len(value) - _MAX_ITEMS} more"
                break
            name = str(key)
            clean[name] = "***" if _SENSITIVE_KEY.search(name) else sanitize_details(item, depth + 1)
        return clean
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        result = [sanitize_details(item, depth + 1) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            result.append(f"... {len(items) - _MAX_ITEMS} more")
        return result
    if isinstance(value, str):
        text = redact(value)
        return text if len(text) <= _MAX_STRING else text[:_MAX_STRING] + "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_details(str(value), depth + 1)


def record_audit(
    db,  # noqa: ANN001 - SQLAlchemy Session
    ctx: AuthContext | None,
    action: str,
    *,
    target_type: str | None = None,
    target_id: object | None = None,
    details: dict[str, Any] | None = None,
    organization_id: int | None = None,
) -> None:
    """Append an audit entry inside the caller's transaction."""
    org_id = organization_id if organization_id is not None else (ctx.organization_id if ctx else None)
    db.add(
        AuditLog(
            organization_id=org_id,
            actor_user_id=ctx.user_id if ctx else None,
            actor_role=ctx.role.value if ctx else None,
            action=action,
            target_type=target_type,
            target_id=None if target_id is None else str(target_id),
            details_json=json.dumps(sanitize_details(details or {}), default=str),
        )
    )


def list_audit_logs(
    ctx: AuthContext,
    *,
    organization_id: int | None = None,
    action_prefix: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Super Admins see every entry (optionally one org); Org Admins only their own."""
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    with session_scope() as db:
        query = select(AuditLog)
        if ctx.role == Role.SUPER_ADMIN:
            authorize_super_admin(db, ctx)
            if organization_id is not None:
                query = query.where(AuditLog.organization_id == organization_id)
        else:
            org_id = authorize(db, ctx, roles=(Role.ORG_ADMIN,))
            query = query.where(AuditLog.organization_id == org_id)
        if action_prefix:
            query = query.where(AuditLog.action.startswith(action_prefix))
        rows = db.execute(query.order_by(AuditLog.id.desc()).limit(limit).offset(offset)).scalars().all()
        return [
            {
                "id": row.id,
                "created_at": row.created_at,
                "organization_id": row.organization_id,
                "actor_user_id": row.actor_user_id,
                "actor_role": row.actor_role,
                "action": row.action,
                "target_type": row.target_type,
                "target_id": row.target_id,
                "details": json.loads(row.details_json or "{}"),
            }
            for row in rows
        ]


def record_error(component: str, exc: BaseException, *, ctx: AuthContext | None = None) -> str:
    """Log an unexpected error with a short reference the user can quote.

    Full detail goes to the server log only; the database keeps a redacted,
    truncated message for the Super Admin monitoring page. Never raises.
    """
    reference = uuid.uuid4().hex[:12]
    logger.error("Unexpected error [%s] in %s", reference, component, exc_info=exc)
    try:
        with session_scope() as db:
            db.add(
                ErrorEvent(
                    organization_id=ctx.organization_id if ctx else None,
                    user_id=ctx.user_id if ctx else None,
                    component=component[:100],
                    error_type=type(exc).__name__[:100],
                    message=redact(str(exc))[:_MAX_STRING],
                    reference=reference,
                )
            )
    except Exception:  # noqa: BLE001 - error capture must never mask the original error
        logger.exception("Could not persist error event %s", reference)
    return reference
