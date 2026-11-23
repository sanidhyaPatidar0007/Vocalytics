"""Idempotent seeding of the default Basic and Premium plans.

Existing plans are never modified, so Super Admin edits survive restarts.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import get_settings
from core.constants import ALL_FEATURES, BASIC_FEATURES, PREMIUM_FEATURES
from database.models import PlanFeature, SubscriptionPlan


def default_plan_definitions() -> list[dict]:
    settings = get_settings()
    return [
        {
            "code": "basic",
            "name": "Basic",
            "description": "Uploads, profiling, standard cleaning and text-based AI analysis.",
            "price_cents": 4900,
            "currency": "USD",
            "billing_period": "monthly",
            "max_users": 5,
            "max_storage_mb": 2048,
            "max_upload_mb": settings.basic_max_upload_mb,
            "max_datasets": 25,
            "max_monthly_analyses": 500,
            "max_versions_per_dataset": 10,
            "features": BASIC_FEATURES,
        },
        {
            "code": "premium",
            "name": "Premium",
            "description": "Everything in Basic plus voice, advanced cleaning, append/merge, "
            "advanced analytics, anomaly detection, e-mail alerts and saved reports.",
            "price_cents": 19900,
            "currency": "USD",
            "billing_period": "monthly",
            "max_users": 50,
            "max_storage_mb": 51200,
            "max_upload_mb": settings.premium_max_upload_mb,
            "max_datasets": 500,
            "max_monthly_analyses": 10000,
            "max_versions_per_dataset": 50,
            "features": PREMIUM_FEATURES,
        },
    ]


def seed_default_plans(db: Session) -> list[str]:
    """Create missing default plans. Returns the codes that were created."""
    created: list[str] = []
    for definition in default_plan_definitions():
        exists = db.execute(
            select(SubscriptionPlan.id).where(SubscriptionPlan.code == definition["code"])
        ).first()
        if exists:
            continue
        features = definition.pop("features")
        plan = SubscriptionPlan(**definition)
        plan.features = [PlanFeature(feature_key=key, enabled=key in features) for key in ALL_FEATURES]
        db.add(plan)
        created.append(plan.code)
    db.flush()
    return created
