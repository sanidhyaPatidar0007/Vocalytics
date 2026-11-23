"""Usage metering (per organization) and monthly quota enforcement."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core import timeutil
from core.constants import UsageMetric
from core.errors import PlanLimitExceededError
from database.models import UsageRecord
from modules import subscriptions


def record_usage(
    db: Session,
    organization_id: int,
    user_id: int | None,
    metric: UsageMetric | str,
    quantity: float = 1.0,
    details: dict | None = None,
) -> None:
    key = metric.value if isinstance(metric, UsageMetric) else str(metric)
    db.add(
        UsageRecord(
            organization_id=organization_id,
            user_id=user_id,
            metric=key,
            quantity=float(quantity),
            details_json=json.dumps(details or {}, default=str),
        )
    )


def total_since(db: Session, organization_id: int, metric: UsageMetric | str, since: datetime) -> float:
    key = metric.value if isinstance(metric, UsageMetric) else str(metric)
    value = db.execute(
        select(func.coalesce(func.sum(UsageRecord.quantity), 0.0)).where(
            UsageRecord.organization_id == organization_id,
            UsageRecord.metric == key,
            UsageRecord.created_at >= since,
        )
    ).scalar_one()
    return float(value)


def monthly_total(
    db: Session, organization_id: int, metric: UsageMetric | str, now: datetime | None = None
) -> float:
    moment = now or timeutil.utcnow()
    return total_since(db, organization_id, metric, timeutil.month_start(moment))


def enforce_monthly_analysis_quota(db: Session, organization_id: int) -> None:
    limits = subscriptions.get_limits(db, organization_id)
    used = monthly_total(db, organization_id, UsageMetric.ANALYSIS_QUERY)
    if used >= limits.max_monthly_analyses:
        raise PlanLimitExceededError(
            f"Your plan allows {limits.max_monthly_analyses} analyses per month and this month's quota is used up."
        )


def usage_summary(db: Session, organization_id: int, since: datetime | None = None) -> dict[str, float]:
    query = select(UsageRecord.metric, func.sum(UsageRecord.quantity)).where(
        UsageRecord.organization_id == organization_id
    )
    if since is not None:
        query = query.where(UsageRecord.created_at >= since)
    rows = db.execute(query.group_by(UsageRecord.metric)).all()
    return {metric: float(total or 0) for metric, total in rows}
