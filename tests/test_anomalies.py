"""Anomaly detection: pure detectors, rule management, runs, duplicates, cooldown, e-mail, scheduler."""

from __future__ import annotations

import json
import math
import os
import smtplib
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select, update

from config.settings import get_settings, set_settings
from core.constants import Feature
from core.errors import (
    ConflictError,
    FeatureNotAvailableError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from database import session_scope
from database.models import (
    AnomalyEvent,
    AnomalyRule,
    AuditLog,
    ErrorEvent,
    NotificationLog,
    Organization,
    PlanFeature,
    Subscription,
    SubscriptionPlan,
    UsageRecord,
)
from modules import subscriptions
from scripts import run_anomaly_checks
from services import anomaly_service as svc
from services import append_service, versioning_service

T0 = datetime(2025, 3, 1, 12, 0)
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def monthly_frame(values: list[float]) -> pd.DataFrame:
    """One row per month (the 15th) starting 2024-01."""
    dates = pd.date_range("2024-01-01", periods=len(values), freq="MS") + pd.Timedelta(days=14)
    return pd.DataFrame(
        {
            "order_date": dates,
            "revenue": pd.array(values, dtype="float64"),
            "units": pd.array(range(len(values)), dtype="Int64"),
            "region": pd.array(["North"] * len(values), dtype="string"),
        }
    )


def append_month(ctx, dataset_id: int, value: float) -> None:
    """Publish a new version with one more monthly row."""

    def transform(df: pd.DataFrame):
        row = df.iloc[[-1]].copy()
        next_month = (df["order_date"].max().to_period("M") + 1).to_timestamp() + pd.Timedelta(days=14)
        row["order_date"] = next_month
        row["revenue"] = value
        return pd.concat([df, row], ignore_index=True), {"rows_added": 1}, {}

    versioning_service.apply_frame_operation(
        ctx, dataset_id, operation_type="append", parameters={}, transform=transform
    )


def edit_frame(ctx, dataset_id: int, change) -> None:
    def transform(df: pd.DataFrame):
        out = df.copy()
        change(out)
        return out, {}, {}

    versioning_service.apply_frame_operation(
        ctx, dataset_id, operation_type="edit", parameters={}, transform=transform
    )


@pytest.fixture
def dataset(admin_ctx, factory) -> int:
    # Jan 100, Feb 100, Mar 100, Apr 60 (-40%), May 100 (+66.7%)
    return factory.dataset(admin_ctx, monthly_frame([100, 100, 100, 60, 100]), "sales")


def make_rule(ctx, dataset_id: int, **overrides):
    config = {
        "dataset_id": dataset_id,
        "name": "Revenue swings",
        "method": "period_change",
        "threshold": 30,
        "metric_column": "revenue",
        "date_column": "order_date",
        "frequency": "M",
    }
    config.update(overrides)
    return svc.create_rule(ctx, **config)


class Outbox:
    """Records what the fake SMTP server received and can be told to fail."""

    def __init__(self) -> None:
        self.messages: list[tuple[object, list[str]]] = []
        self.fail_all: BaseException | None = None
        self.fail_for: set[str] = set()


@pytest.fixture
def smtp(monkeypatch, app_settings) -> Outbox:
    outbox = Outbox()

    class FakeSMTP:
        def __init__(self, host, port, timeout=None, context=None):
            if outbox.fail_all:
                raise outbox.fail_all

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, context=None):
            pass

        def login(self, user, password):
            pass

        def send_message(self, message, from_addr=None, to_addrs=None):
            if set(to_addrs) & outbox.fail_for:
                raise smtplib.SMTPRecipientsRefused({a: (550, b"no") for a in to_addrs})
            outbox.messages.append((message, list(to_addrs)))

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    set_settings(app_settings.with_overrides(smtp_host="smtp.example.test", smtp_from="alerts@example.test"))
    return outbox


def disable_feature(org_id: int, feature: Feature) -> None:
    with session_scope() as db:
        plan_id = subscriptions.get_active_subscription(db, org_id).plan_id
        db.execute(
            update(PlanFeature)
            .where(PlanFeature.plan_id == plan_id, PlanFeature.feature_key == feature.value)
            .values(enabled=False)
        )


def events_of(org_id: int) -> list[AnomalyEvent]:
    with session_scope() as db:
        return list(
            db.execute(
                select(AnomalyEvent).where(AnomalyEvent.organization_id == org_id).order_by(AnomalyEvent.id)
            )
            .scalars()
            .all()
        )


def logs_of(org_id: int) -> list[NotificationLog]:
    with session_scope() as db:
        return list(
            db.execute(select(NotificationLog).where(NotificationLog.organization_id == org_id))
            .scalars()
            .all()
        )


def period_series(values: dict[str, float], freq: str = "M") -> pd.Series:
    return pd.Series(
        list(values.values()), index=pd.PeriodIndex(list(values.keys()), freq=freq), dtype="float64"
    )


# ---------------------------------------------------------------------------
# Pure detectors
# ---------------------------------------------------------------------------


def _iqr_base(extra: float) -> pd.Series:
    return pd.Series([float(i) for i in range(1, 11)] + [extra])


def test_iqr_upper_bound_is_exclusive_and_aggregated():
    # Q1 = 3.5, Q3 = 8.5, IQR = 5 -> upper limit 16.0
    assert svc.detect_iqr(_iqr_base(16.0), 1.5, column="x", period="version 1") == []
    findings = svc.detect_iqr(_iqr_base(16.5), 1.5, column="x", period="version 1")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.direction == "above" and finding.observed == 16.5 and finding.upper_bound == 16.0
    assert finding.details["count"] == 1 and finding.details["q1"] == 3.5 and finding.details["q3"] == 8.5
    assert finding.expected == 6.0  # the median is the expected value
    assert "Possible" in finding.title and "fraud" not in finding.title.lower()


def test_iqr_lower_bound_and_two_directions_give_one_finding_each():
    base = [float(i) for i in range(1, 11)]
    assert svc.detect_iqr(pd.Series([-5.0] + base), 1.5, column="x", period="v") == []  # exactly on the limit
    below = svc.detect_iqr(pd.Series([-7.5] + base), 1.5, column="x", period="v")
    assert [f.direction for f in below] == ["below"] and below[0].observed == -7.5
    both = svc.detect_iqr(pd.Series([-50.0, 60.0, -40.0, 70.0] + base), 1.5, column="x", period="v")
    assert sorted(f.direction for f in both) == ["above", "below"]
    assert all(f.details["count"] == 2 for f in both)


def test_iqr_threshold_scales_the_fences_and_small_samples_are_ignored():
    series = _iqr_base(16.5)
    assert svc.detect_iqr(series, 1.7, column="x", period="v") == []  # fence moves to 17.0
    assert (
        svc.detect_iqr(pd.Series([1.0, 2.0, 100.0]), 1.5, column="x", period="v") == []
    )  # fewer than 4 values
    assert (
        svc.detect_iqr(pd.Series([None, None, None, None, 5.0], dtype="float64"), 1.5, column="x", period="v")
        == []
    )


def test_iqr_ignores_missing_and_non_finite_values():
    values = pd.Series([float(i) for i in range(1, 11)] + [16.5, math.nan, math.inf, -math.inf])
    findings = svc.detect_iqr(values, 1.5, column="x", period="v")
    assert len(findings) == 1 and findings[0].details["total_values"] == 11


def test_zscore_boundary_is_strict():
    series = pd.Series([10.0] * 20 + [100.0])
    z = float((series.iloc[-1] - series.mean()) / series.std(ddof=1))
    assert svc.detect_zscore(series, z, column="x", period="v") == []
    findings = svc.detect_zscore(series, z - 0.001, column="x", period="v")
    assert len(findings) == 1 and findings[0].direction == "above" and findings[0].observed == 100.0
    assert findings[0].details["std"] == pytest.approx(series.std(ddof=1))
    assert svc.detect_zscore(series, 5.0, column="x", period="v") == []


def test_zscore_both_directions_and_degenerate_inputs():
    wide = pd.Series([0.0] * 30 + [100.0, -100.0])
    assert sorted(f.direction for f in svc.detect_zscore(wide, 3.0, column="x", period="v")) == [
        "above",
        "below",
    ]
    assert svc.detect_zscore(pd.Series([5.0] * 10), 3.0, column="x", period="v") == []  # no variation
    assert (
        svc.detect_zscore(pd.Series([1.0, 100.0]), 1.0, column="x", period="v") == []
    )  # fewer than 3 values


def test_aggregate_by_period_functions_and_labels():
    frame = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-31", "2024-03-05", "2024-03-06", None]),
            "m": [1.0, 3.0, 10.0, None, 99.0],
        }
    )
    expected = {
        "sum": [4.0, 10.0],
        "mean": [2.0, 10.0],
        "min": [1.0, 10.0],
        "max": [3.0, 10.0],
        "median": [2.0, 10.0],
        "count": [2.0, 1.0],
    }
    for aggregation, values in expected.items():
        series = svc.aggregate_by_period(
            frame, date_column="d", metric_column="m", aggregation=aggregation, frequency="M"
        )
        assert [str(p) for p in series.index] == [
            "2024-01",
            "2024-03",
        ]  # February has no data and is not invented
        assert series.tolist() == values
    daily = svc.aggregate_by_period(
        frame, date_column="d", metric_column="m", aggregation="sum", frequency="D"
    )
    assert [str(p) for p in daily.index] == ["2024-01-01", "2024-01-31", "2024-03-05"]  # 03-06 has no value
    weekly = svc.aggregate_by_period(
        frame, date_column="d", metric_column="m", aggregation="sum", frequency="W"
    )
    assert [str(p) for p in weekly.index] == [
        "2024-01-01/2024-01-07",
        "2024-01-29/2024-02-04",
        "2024-03-04/2024-03-10",
    ]


def test_aggregate_by_period_drops_all_missing_periods_and_rejects_bad_options():
    frame = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-02-01"]), "m": [None, 5.0]})
    series = svc.aggregate_by_period(
        frame, date_column="d", metric_column="m", aggregation="sum", frequency="M"
    )
    assert [str(p) for p in series.index] == ["2024-02"]  # an all-missing period is not a zero
    with pytest.raises(ValidationError):
        svc.aggregate_by_period(frame, date_column="d", metric_column="m", aggregation="mode", frequency="M")
    with pytest.raises(ValidationError):
        svc.aggregate_by_period(frame, date_column="d", metric_column="m", aggregation="sum", frequency="Y")


def test_period_change_flags_both_directions_with_neutral_wording():
    series = period_series({"2024-01": 100, "2024-02": 100, "2024-03": 60, "2024-04": 100})
    findings = svc.detect_period_change(series, 30, column="revenue", frequency="M")
    assert [(f.period, f.direction) for f in findings] == [("2024-03", "decrease"), ("2024-04", "increase")]
    assert findings[0].title == "Possible unusual drop in revenue for 2024-03: -40.0% vs previous month"
    assert findings[1].title == "Possible unusual increase in revenue for 2024-04: +66.7% vs previous month"
    assert findings[0].observed == 60 and findings[0].expected == 100
    assert findings[0].details["percent_change"] == pytest.approx(-40.0)
    assert all("fraud" not in f.title.lower() for f in findings)


def test_period_change_threshold_boundary_is_strict():
    series = period_series({"2024-01": 100, "2024-02": 110})
    assert svc.detect_period_change(series, 10, column="x", frequency="M") == []
    assert len(svc.detect_period_change(series, 9.99, column="x", frequency="M")) == 1
    falling = period_series({"2024-01": 100, "2024-02": 90})
    assert svc.detect_period_change(falling, 10, column="x", frequency="M") == []
    assert len(svc.detect_period_change(falling, 9.99, column="x", frequency="M")) == 1


def test_period_change_skips_zero_base_and_handles_short_series():
    assert (
        svc.detect_period_change(period_series({"2024-01": 0, "2024-02": 50}), 10, column="x", frequency="M")
        == []
    )
    assert svc.detect_period_change(period_series({"2024-01": 5}), 10, column="x", frequency="M") == []
    negative = period_series({"2024-01": -100, "2024-02": -50})  # relative to the absolute previous value
    assert (
        svc.detect_period_change(negative, 10, column="x", frequency="M")[0].details["percent_change"] == 50.0
    )


def test_period_change_units_follow_the_frequency():
    daily = period_series({"2024-01-01": 100, "2024-01-02": 10}, "D")
    assert svc.detect_period_change(daily, 10, column="x", frequency="D")[0].title.endswith("vs previous day")
    weekly = period_series({"2024-01-01": 100, "2024-01-08": 10}, "W")
    assert svc.detect_period_change(weekly, 10, column="x", frequency="W")[0].title.endswith(
        "vs previous week"
    )


def test_metric_drop_ignores_increases_and_exact_threshold():
    series = period_series({"2024-01": 100, "2024-02": 160, "2024-03": 96, "2024-04": 100})
    findings = svc.detect_period_change(series, 40, column="revenue", frequency="M", drops_only=True)
    assert findings == []  # 160 -> 96 is exactly -40%
    findings = svc.detect_period_change(series, 39.9, column="revenue", frequency="M", drops_only=True)
    assert [f.period for f in findings] == ["2024-03"] and findings[0].method == "metric_drop"


def test_baseline_deviation_uses_previous_six_periods():
    values = {f"2024-0{i}": v for i, v in enumerate([100, 102, 98, 101, 99, 100, 130], start=1)}
    findings = svc.detect_baseline_deviation(period_series(values), 3.0, column="revenue", frequency="M")
    assert len(findings) == 1
    finding = findings[0]
    assert finding.period == "2024-07" and finding.direction == "above" and finding.expected == 100.0
    assert finding.details["baseline_std"] == pytest.approx(math.sqrt(2))
    assert finding.details["z_score"] == pytest.approx(30 / math.sqrt(2))
    assert "21.2 standard deviations above the mean of the previous 6 months" in finding.title
    low = dict(values, **{"2024-07": 70})
    assert (
        svc.detect_baseline_deviation(period_series(low), 3.0, column="r", frequency="M")[0].direction
        == "below"
    )


def test_baseline_deviation_boundary_and_degenerate_cases():
    values = {f"2024-0{i}": v for i, v in enumerate([10, 12, 10, 12, 10, 12, 14], start=1)}
    series = period_series(values)
    baseline = series.iloc[:6]
    z = (14 - baseline.mean()) / baseline.std(ddof=1)
    assert svc.detect_baseline_deviation(series, float(z), column="r", frequency="M") == []
    assert len(svc.detect_baseline_deviation(series, float(z) - 0.01, column="r", frequency="M")) == 1
    # Only the 6 periods before the latest count, older history is ignored.
    longer = period_series({"2023-12": 1000, **values})
    assert svc.detect_baseline_deviation(longer, float(z), column="r", frequency="M") == []
    assert (
        svc.detect_baseline_deviation(
            period_series({k: v for k, v in list(values.items())[:6]}), 1.0, column="r", frequency="M"
        )
        == []
    )
    flat = period_series({f"2024-0{i}": v for i, v in enumerate([5, 5, 5, 5, 5, 5, 50], start=1)})
    assert svc.detect_baseline_deviation(flat, 1.0, column="r", frequency="M") == []  # no spread to scale by


@pytest.mark.parametrize(("values", "drops_only"), [([7.70, 8.47], False), ([1.10, 0.99], True), ([1.10, 0.99], False)])
def test_period_change_exactly_on_threshold_is_not_flagged_despite_float_error(values, drops_only):
    series = pd.Series(values, index=pd.period_range("2024-01", periods=2, freq="M"))
    assert svc.detect_period_change(series, 10.0, column="revenue", frequency="M", drops_only=drops_only) == []
    assert len(svc.detect_period_change(series, 9.99, column="revenue", frequency="M", drops_only=drops_only)) == 1


def test_missing_increase_boundary_and_matching_columns():
    current = {"a": 20.0, "b": 5.0, "new_col": 90.0}
    previous = {"a": 10.0, "b": 5.0, "gone": 1.0}
    assert (
        svc.detect_missing_increase(
            current, previous, 10.0, current_label="version 2", previous_label="version 1"
        )
        == []
    )
    findings = svc.detect_missing_increase(
        current, previous, 9.99, current_label="version 2", previous_label="version 1"
    )
    assert [f.column for f in findings] == ["a"]
    assert findings[0].observed == 20.0 and findings[0].expected == 10.0
    assert (
        findings[0].title
        == "Possible increase in missing data in a: 10.0% to 20.0% (+10.0 points) in version 2 vs version 1"
    )
    assert (
        svc.detect_missing_increase({"a": 5.0}, {"a": 50.0}, 1.0, current_label="v2", previous_label="v1")
        == []
    )  # decrease


def test_schedule_due_logic():
    now = T0
    assert svc.rule_is_due("hourly", None, now) is True
    assert svc.rule_is_due("hourly", now - timedelta(minutes=54), now) is False
    assert svc.rule_is_due("hourly", now - timedelta(minutes=55), now) is True  # five minute grace
    assert svc.rule_is_due("daily", now - timedelta(hours=23), now) is False
    assert svc.rule_is_due("daily", now - timedelta(hours=24), now) is True
    assert svc.rule_is_due("weekly", now - timedelta(days=6), now) is False
    assert svc.rule_is_due("weekly", now - timedelta(days=7), now) is True
    assert svc.rule_is_due("manual", None, now) is False


def test_fingerprint_is_stable_and_discriminating():
    base = svc.make_fingerprint(1, "iqr", "revenue", "version 1", "above")
    assert base == svc.make_fingerprint(1, "iqr", "revenue", "version 1", "above") and len(base) == 64
    for other in (
        svc.make_fingerprint(2, "iqr", "revenue", "version 1", "above"),
        svc.make_fingerprint(1, "zscore", "revenue", "version 1", "above"),
        svc.make_fingerprint(1, "iqr", "units", "version 1", "above"),
        svc.make_fingerprint(1, "iqr", "revenue", "version 2", "above"),
        svc.make_fingerprint(1, "iqr", "revenue", "version 1", "below"),
    ):
        assert other != base


# ---------------------------------------------------------------------------
# Rule management
# ---------------------------------------------------------------------------


def test_create_rule_stores_a_validated_rule(admin_ctx, dataset, org_id):
    rule = make_rule(
        admin_ctx,
        dataset,
        recipients=["Ops@Example.com", "ops@example.com", " lead@example.com "],
        severity="high",
        schedule="daily",
    )
    assert rule["id"] > 0 and rule["dataset_id"] == dataset and rule["dataset_name"] == "sales"
    assert rule["method"] == "period_change" and rule["threshold"] == 30.0 and rule["aggregation"] == "sum"
    assert rule["recipients"] == ["Ops@Example.com", "lead@example.com"]
    assert rule["cooldown_minutes"] == get_settings().alert_default_cooldown_minutes == 60
    assert rule["is_active"] is True and rule["last_checked_at"] is None and rule["event_count"] == 0
    with session_scope() as db:
        audit = db.execute(select(AuditLog).where(AuditLog.action == "anomaly.rule.create")).scalar_one()
        assert audit.organization_id == org_id and "ops@example.com" not in audit.details_json.lower()


def test_create_value_rule_clears_irrelevant_settings(admin_ctx, dataset):
    rule = svc.create_rule(
        admin_ctx,
        dataset_id=dataset,
        name="Revenue outliers",
        method="iqr",
        threshold=1.5,
        metric_column="revenue",
        date_column="order_date",
        frequency="M",
        aggregation="mean",
        cooldown_minutes=0,
    )
    assert rule["date_column"] is None and rule["frequency"] is None and rule["aggregation"] == "sum"
    assert rule["cooldown_minutes"] == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "prophet"},
        {"method": None},
        {"threshold": 0},
        {"threshold": -1},
        {"threshold": math.nan},
        {"threshold": math.inf},
        {"threshold": True},
        {"threshold": "5"},
        {"method": "metric_drop", "threshold": 100.5},
        {"method": "missing_increase", "threshold": 101},
        {"metric_column": "nope"},
        {"metric_column": None},
        {"metric_column": "region"},  # text column cannot be summed
        {"date_column": None},
        {"date_column": "revenue"},  # not a date
        {"date_column": "nope"},
        {"date_column": "order_date", "metric_column": "order_date", "aggregation": "count"},
        {"frequency": "Y"},
        {"frequency": None},
        {"aggregation": "mode"},
        {"severity": "urgent"},
        {"schedule": "minutely"},
        {"cooldown_minutes": -1},
        {"cooldown_minutes": True},
        {"cooldown_minutes": 1.5},
        {"name": "   "},
        {"name": "x" * 201},
        {"name": None},
        {"recipients": ["not-an-email"]},
        {"recipients": "a@example.com"},
        {"recipients": ["a@example.com\r\nBcc: x@example.com"]},
        {"is_active": "yes"},
    ],
)
def test_create_rule_validation_errors(admin_ctx, dataset, overrides):
    with pytest.raises(ValidationError):
        make_rule(admin_ctx, dataset, **overrides)
    assert svc.list_rules(admin_ctx) == []


def test_text_column_is_allowed_when_counting(admin_ctx, dataset):
    rule = make_rule(admin_ctx, dataset, metric_column="region", aggregation="count")
    assert rule["aggregation"] == "count"


def test_value_methods_need_numeric_columns(admin_ctx, dataset):
    for method in ("iqr", "zscore"):
        with pytest.raises(ValidationError):
            svc.create_rule(
                admin_ctx, dataset_id=dataset, name=method, method=method, threshold=2, metric_column="region"
            )
        with pytest.raises(ValidationError):
            svc.create_rule(admin_ctx, dataset_id=dataset, name=method, method=method, threshold=2)


def test_missing_increase_rule_needs_no_column_but_validates_one_when_given(admin_ctx, dataset):
    rule = svc.create_rule(
        admin_ctx, dataset_id=dataset, name="missing", method="missing_increase", threshold=10
    )
    assert rule["metric_column"] is None
    with pytest.raises(ValidationError):
        svc.create_rule(
            admin_ctx,
            dataset_id=dataset,
            name="missing2",
            method="missing_increase",
            threshold=10,
            metric_column="nope",
        )


def test_duplicate_rule_name_conflicts_case_insensitively(admin_ctx, dataset):
    make_rule(admin_ctx, dataset)
    with pytest.raises(ConflictError):
        make_rule(admin_ctx, dataset, name="revenue SWINGS")


def test_rule_names_with_non_ascii_letters_cannot_collide(admin_ctx, dataset):
    make_rule(admin_ctx, dataset, name="Über alerts")
    with pytest.raises(ConflictError):
        make_rule(admin_ctx, dataset, name="Über alerts")
    other = make_rule(admin_ctx, dataset, name="Other")
    with pytest.raises(ConflictError):
        svc.update_rule(admin_ctx, other["id"], name="Über alerts")


def test_recipients_need_the_email_alerts_feature(admin_ctx, dataset, org_id):
    disable_feature(org_id, Feature.EMAIL_ALERTS)
    with pytest.raises(ValidationError) as info:
        make_rule(admin_ctx, dataset, recipients=["ops@example.com"])
    assert "plan" in info.value.user_message
    rule = make_rule(admin_ctx, dataset)  # without recipients the rule is fine: events are still recorded
    with pytest.raises(ValidationError):
        svc.update_rule(admin_ctx, rule["id"], recipients=["ops@example.com"])
    assert svc.update_rule(admin_ctx, rule["id"], recipients=[])["recipients"] == []


def test_update_rule(admin_ctx, dataset, org_id):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"])
    updated = svc.update_rule(
        admin_ctx,
        rule["id"],
        threshold=45,
        severity="critical",
        schedule="hourly",
        cooldown_minutes=5,
        name="Renamed",
    )
    assert (
        updated["threshold"],
        updated["severity"],
        updated["schedule"],
        updated["cooldown_minutes"],
        updated["name"],
    ) == (45.0, "critical", "hourly", 5, "Renamed")
    assert updated["recipients"] == ["ops@example.com"]
    switched = svc.update_rule(admin_ctx, rule["id"], method="iqr", threshold=1.5)
    assert switched["method"] == "iqr" and switched["date_column"] is None and switched["frequency"] is None
    assert svc.update_rule(admin_ctx, rule["id"], recipients=["a@example.com", "b@example.com"])[
        "recipients"
    ] == ["a@example.com", "b@example.com"]
    with session_scope() as db:
        assert (
            db.execute(
                select(func.count(AuditLog.id)).where(AuditLog.action == "anomaly.rule.update")
            ).scalar_one()
            == 3
        )


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"dataset_id": 5},
        {"unknown": 1},
        {"organization_id": 2},
        {"threshold": -3},
        {"method": "missing_increase", "threshold": 500},
        {"metric_column": "region"},  # text column with a sum aggregation
        {"date_column": "revenue"},
        {"frequency": "Q"},
        {"name": ""},
        {"is_active": None},
        {"recipients": ["bad"]},
    ],
)
def test_update_rule_validation_errors(admin_ctx, dataset, changes):
    rule = make_rule(admin_ctx, dataset)
    with pytest.raises(ValidationError):
        svc.update_rule(admin_ctx, rule["id"], **changes)
    assert svc.list_rules(admin_ctx)[0]["threshold"] == 30.0


def test_update_rule_name_conflict(admin_ctx, dataset):
    make_rule(admin_ctx, dataset, name="one")
    two = make_rule(admin_ctx, dataset, name="two")
    with pytest.raises(ConflictError):
        svc.update_rule(admin_ctx, two["id"], name="ONE")
    assert svc.update_rule(admin_ctx, two["id"], name="TWO")["name"] == "TWO"  # renaming to itself is fine


def test_set_rule_active_and_list_rules(admin_ctx, dataset):
    rule = make_rule(admin_ctx, dataset)
    assert svc.set_rule_active(admin_ctx, rule["id"], False)["is_active"] is False
    assert svc.list_rules(admin_ctx)[0]["is_active"] is False
    assert svc.set_rule_active(admin_ctx, rule["id"], True)["is_active"] is True
    with pytest.raises(ValidationError):
        svc.set_rule_active(admin_ctx, rule["id"], "no")
    with session_scope() as db:
        actions = set(
            db.execute(select(AuditLog.action).where(AuditLog.action.like("anomaly.rule.%"))).scalars()
        )
    assert {"anomaly.rule.activate", "anomaly.rule.deactivate"} <= actions


def test_delete_rule_only_without_events(admin_ctx, dataset, org_id):
    unused = make_rule(admin_ctx, dataset, name="unused")
    assert svc.delete_rule(admin_ctx, unused["id"]) == {
        "deleted": True,
        "rule_id": unused["id"],
        "name": "unused",
    }
    assert svc.list_rules(admin_ctx) == []
    used = make_rule(admin_ctx, dataset, name="used")
    svc.run_rule(admin_ctx, used["id"], now=T0)
    with pytest.raises(ValidationError) as info:
        svc.delete_rule(admin_ctx, used["id"])
    assert "Deactivate" in info.value.user_message
    assert len(svc.list_rules(admin_ctx)) == 1


def test_list_rules_counts_events(admin_ctx, dataset):
    rule = make_rule(admin_ctx, dataset)
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert svc.list_rules(admin_ctx)[0]["event_count"] == 2


@pytest.mark.parametrize("bad_id", [0, -1, True, "1", None])
def test_invalid_rule_ids_are_validation_errors(admin_ctx, bad_id):
    with pytest.raises(ValidationError):
        svc.run_rule(admin_ctx, bad_id)
    with pytest.raises(ValidationError):
        svc.acknowledge_event(admin_ctx, bad_id)


# ---------------------------------------------------------------------------
# Running rules
# ---------------------------------------------------------------------------


def test_run_rule_creates_explainable_events(admin_ctx, dataset, org_id):
    rule = make_rule(admin_ctx, dataset, severity="high")
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert (
        result["findings"],
        result["new_events"],
        result["duplicate_suppressed"],
        result["cooldown_suppressed"],
    ) == (2, 2, 0, 0)
    assert result["notifications"] == [] and result["version_number"] == 1
    events = svc.list_events(admin_ctx)
    assert [e["title"] for e in sorted(events, key=lambda e: e["id"])] == [
        "Possible unusual drop in revenue for 2024-04: -40.0% vs previous month",
        "Possible unusual increase in revenue for 2024-05: +66.7% vs previous month",
    ]
    drop = min(events, key=lambda e: e["id"])
    assert drop["severity"] == "high" and drop["status"] == "new" and drop["detected_at"] == T0
    assert (
        drop["observed_value"] == 60.0
        and drop["expected_value"] == 100.0
        and drop["rule_name"] == "Revenue swings"
    )
    details = drop["details"]
    assert (
        details["method"] == "period_change"
        and details["threshold"] == 30.0
        and details["column"] == "revenue"
    )
    assert (
        details["period"] == "2024-04"
        and details["direction"] == "decrease"
        and details["aggregation"] == "sum"
    )
    assert (
        details["observed"] == 60.0
        and details["expected"] == 100.0
        and details["previous_period"] == "2024-03"
    )
    assert details["percent_change"] == pytest.approx(-40.0) and "not a confirmed problem" in details["note"]
    with session_scope() as db:
        stored = db.get(AnomalyRule, rule["id"])
        assert stored.last_checked_at == T0
        assert (
            db.execute(
                select(func.count(UsageRecord.id)).where(UsageRecord.metric == "anomaly_check")
            ).scalar_one()
            == 1
        )
        audit = db.execute(select(AuditLog).where(AuditLog.action == "anomaly.run")).scalar_one()
        assert json.loads(audit.details_json)["new_events"] == 2


def test_running_twice_creates_no_duplicates(admin_ctx, dataset, org_id):
    rule = make_rule(admin_ctx, dataset)
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    second = svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(hours=1))
    assert (second["findings"], second["new_events"], second["duplicate_suppressed"]) == (2, 0, 2)
    assert len(events_of(org_id)) == 2
    append_month(admin_ctx, dataset, 20)  # a new period with a new drop creates exactly one new event
    third = svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(hours=2))
    assert (third["findings"], third["new_events"], third["duplicate_suppressed"]) == (3, 1, 2)
    assert "2024-06" in third["events"][0]["title"] and third["version_number"] == 2
    with session_scope() as db:
        assert db.get(AnomalyRule, rule["id"]).last_checked_at == T0 + timedelta(hours=2)


def test_metric_drop_only_reports_drops(admin_ctx, dataset):
    rule = make_rule(admin_ctx, dataset, method="metric_drop", threshold=30)
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert result["new_events"] == 1 and "drop" in result["events"][0]["title"]


def test_baseline_deviation_rule(admin_ctx, factory):
    ds = factory.dataset(admin_ctx, monthly_frame([100, 102, 98, 101, 99, 100, 130]), "baseline")
    rule = make_rule(admin_ctx, ds, method="baseline_deviation", threshold=3)
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert result["new_events"] == 1
    assert (
        result["events"][0]["details"]["baseline_periods"] == 6 and result["events"][0]["period"] == "2024-07"
    )
    assert svc.run_rule(admin_ctx, rule["id"], now=T0)["duplicate_suppressed"] == 1


def test_iqr_and_zscore_rules_are_deduplicated_per_version(admin_ctx, factory, org_id):
    values = [float(i) for i in range(1, 11)] + [500.0]
    ds = factory.dataset(admin_ctx, monthly_frame(values), "outliers")
    for method, threshold in (("iqr", 1.5), ("zscore", 3.0)):
        rule = svc.create_rule(
            admin_ctx, dataset_id=ds, name=method, method=method, threshold=threshold, metric_column="revenue"
        )
        first = svc.run_rule(admin_ctx, rule["id"], now=T0)
        assert first["new_events"] == 1 and "version 1" in first["events"][0]["title"]
        assert svc.run_rule(admin_ctx, rule["id"], now=T0)["duplicate_suppressed"] == 1
    edit_frame(
        admin_ctx, ds, lambda df: df.__setitem__("units", df["units"] + 1)
    )  # new version, same outlier
    rule_id = svc.list_rules(admin_ctx)[0]["id"]
    again = svc.run_rule(admin_ctx, rule_id, now=T0)
    assert again["new_events"] == 1 and again["version_number"] == 2


def test_missing_increase_across_two_versions(admin_ctx, factory):
    frame = monthly_frame([100] * 10)
    ds = factory.dataset(admin_ctx, frame, "quality")
    rule = svc.create_rule(admin_ctx, dataset_id=ds, name="missing", method="missing_increase", threshold=20)
    first = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert first["findings"] == 0 and first["new_events"] == 0  # no previous version to compare with

    def blank_revenue(df):
        df.loc[:3, "revenue"] = pd.NA  # 4 of 10 rows: 0% -> 40%
        df.loc[:1, "units"] = pd.NA  # 2 of 10 rows: 0% -> 20%, not above the threshold

    edit_frame(admin_ctx, ds, blank_revenue)
    result = svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(hours=1))
    assert result["new_events"] == 1
    event = result["events"][0]
    assert (
        event["title"] == "Possible increase in missing data in revenue: 0.0% to 40.0% (+40.0 points) "
        "in version 2 vs version 1"
    )
    assert (
        event["observed"] == 40.0
        and event["expected"] == 0.0
        and event["details"]["compared_with"] == "version 1"
    )
    assert svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(hours=2))["duplicate_suppressed"] == 1
    # Restricting the rule to one column only watches that column.
    units_rule = svc.create_rule(
        admin_ctx, dataset_id=ds, name="units", method="missing_increase", threshold=10, metric_column="units"
    )
    assert (
        svc.run_rule(admin_ctx, units_rule["id"], now=T0)["events"][0]["details"]["increase_points"] == 20.0
    )
    revenue_rule = svc.create_rule(
        admin_ctx, dataset_id=ds, name="rev", method="missing_increase", threshold=50, metric_column="revenue"
    )
    assert svc.run_rule(admin_ctx, revenue_rule["id"], now=T0)["findings"] == 0


def test_missing_increase_flags_rise_in_invalid_values_kept_as_text(factory, org_id, admin_ctx):
    target = factory.dataset(admin_ctx, pd.DataFrame({"amount": [float(i) for i in range(1, 19)]}), "target")
    source = factory.dataset(admin_ctx, pd.DataFrame({"amount": ["bad", "12x"]}), "source")
    rule = svc.create_rule(
        admin_ctx, dataset_id=target, name="quality", method="missing_increase", threshold=5
    )
    assert svc.run_rule(admin_ctx, rule["id"])["findings"] == 0
    append_service.apply_append(
        admin_ctx, target, source, type_conflict_policy="cast_to_text", confirm=True
    )
    result = svc.run_rule(admin_ctx, rule["id"])
    # version 2: 2 of 20 amounts (10%) are not numbers, up from 0%; spec 17 asks for this to be flagged
    assert result["findings"] == 1
    assert [e["details"]["column"] for e in result["events"]] == ["amount"]
    event = result["events"][0]
    assert event["observed"] == 10.0 and event["expected"] == 0.0
    assert event["title"].startswith("Possible increase in missing or invalid data in amount")


def test_missing_increase_counts_all_non_numeric_values_after_a_type_change_to_text(factory, admin_ctx):
    target = factory.dataset(admin_ctx, pd.DataFrame({"amount": [float(i) for i in range(1, 11)]}), "target")
    source = factory.dataset(
        admin_ctx, pd.DataFrame({"amount": ["n/a?", "abc", "12x", "--", "bad"] * 2}), "source"
    )
    rule = svc.create_rule(admin_ctx, dataset_id=target, name="inv", method="missing_increase", threshold=5)
    assert svc.run_rule(admin_ctx, rule["id"])["findings"] == 0
    append_service.apply_append(
        admin_ctx, target, source, type_conflict_policy="cast_to_text", confirm=True
    )
    result = svc.run_rule(admin_ctx, rule["id"])
    assert result["findings"] == 1 and result["events"][0]["observed"] == 50.0


def test_missing_increase_still_counts_only_nulls_when_values_are_valid(factory, admin_ctx):
    ds = factory.dataset(admin_ctx, pd.DataFrame({"city": pd.array(["a", "b", "c", "d"], dtype="string")}), "t")
    rule = svc.create_rule(admin_ctx, dataset_id=ds, name="m", method="missing_increase", threshold=5)
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    edit_frame(admin_ctx, ds, lambda df: df.__setitem__("city", pd.array(["a", None, "c", "d"], dtype="string")))
    result = svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(hours=1))
    assert result["events"][0]["observed"] == 25.0
    assert result["events"][0]["title"].startswith("Possible increase in missing data in city")


def test_manual_run_works_for_inactive_rules(admin_ctx, dataset):
    rule = make_rule(admin_ctx, dataset, is_active=False)
    assert svc.run_rule(admin_ctx, rule["id"], now=T0)["new_events"] == 2


def test_run_rule_reports_missing_column_in_user_safe_words(admin_ctx, dataset):
    rule = make_rule(admin_ctx, dataset)
    edit_frame(admin_ctx, dataset, lambda df: None)  # keeps the schema
    versioning_service.apply_frame_operation(
        admin_ctx,
        dataset,
        operation_type="drop_columns",
        parameters={},
        transform=lambda df: (df.drop(columns=["revenue"]), {}, {}),
    )
    with pytest.raises(ValidationError) as info:
        svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert "revenue" in info.value.user_message and "Traceback" not in info.value.user_message


def test_column_names_with_line_breaks_are_neutralised(admin_ctx, factory, smtp):
    frame = monthly_frame([100, 100, 100, 60, 100]).rename(
        columns={"revenue": "rev\r\nBcc: x@evil.com<script>"}
    )
    ds = factory.dataset(admin_ctx, frame, "evil")
    rule = svc.create_rule(
        admin_ctx,
        dataset_id=ds,
        name="Rule\r\nBcc: x@evil.com",
        method="period_change",
        threshold=30,
        metric_column="rev\r\nBcc: x@evil.com<script>",
        date_column="order_date",
        frequency="M",
        recipients=["ops@example.com"],
    )
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert result["new_events"] == 2 and result["notifications"][0]["status"] == "sent"
    message, _ = smtp.messages[0]
    assert "\r" not in str(message["Subject"]) and "\n" not in str(message["Subject"])
    assert all("\n" not in e["title"] and "\r" not in e["title"] for e in result["events"])
    html = message.get_payload()[1].get_payload(decode=True).decode()
    assert "<script>" not in html and "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Notifications and cooldown
# ---------------------------------------------------------------------------


def test_missing_smtp_logs_skipped_and_keeps_events_new(admin_ctx, dataset, org_id):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"])
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert [n["status"] for n in result["notifications"]] == ["skipped"] and "SMTP_HOST" in result[
        "notifications"
    ][0]["error"]
    assert {e.status for e in events_of(org_id)} == {"new"}
    logs = logs_of(org_id)
    assert len(logs) == 2 and {log.status for log in logs} == {"skipped"} and logs[0].error_message
    assert svc.list_notifications(admin_ctx)[0]["recipient"] == "ops@example.com"


def test_successful_notification_marks_events_notified(admin_ctx, dataset, org_id, smtp):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com", "lead@example.com"])
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert [(n["recipient"], n["status"]) for n in result["notifications"]] == [
        ("ops@example.com", "sent"),
        ("lead@example.com", "sent"),
    ]
    assert {e.status for e in events_of(org_id)} == {"notified"} and {
        e["status"] for e in result["events"]
    } == {"notified"}
    assert len(smtp.messages) == 2  # one digest per recipient, not one per event
    message, recipients = smtp.messages[0]
    assert recipients == ["ops@example.com"] and "2 possible anomalies" in str(message["Subject"])
    assert "[MEDIUM]" in str(message["Subject"])
    text = message.get_payload()[0].get_payload(decode=True).decode()
    assert "Possible unusual drop in revenue for 2024-04" in text and "not confirmed problems" in text
    logs = logs_of(org_id)
    assert (
        len(logs) == 4
        and {log.status for log in logs} == {"sent"}
        and all(log.created_at == T0 for log in logs)
    )
    listed = svc.list_notifications(admin_ctx)
    assert len(listed) == 4 and listed[0]["rule_name"] == "Revenue swings" and listed[0]["channel"] == "email"
    assert len(svc.list_notifications(admin_ctx, limit=1)) == 1


def test_smtp_failure_logs_failed_without_secrets_and_events_stay_new(admin_ctx, dataset, org_id, smtp):
    smtp.fail_all = ConnectionRefusedError("refused password=hunter2")
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"])
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert [n["status"] for n in result["notifications"]] == ["failed"]
    assert "hunter2" not in json.dumps(result["notifications"])
    assert {e.status for e in events_of(org_id)} == {"new"}
    logs = logs_of(org_id)
    assert {log.status for log in logs} == {"failed"} and all(
        "hunter2" not in (log.error_message or "") for log in logs
    )


def test_partial_delivery_still_counts_as_notified(admin_ctx, dataset, org_id, smtp):
    smtp.fail_for = {"bad@example.com"}
    rule = make_rule(admin_ctx, dataset, recipients=["good@example.com", "bad@example.com"])
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert [n["status"] for n in result["notifications"]] == ["sent", "failed"]
    assert {e.status for e in events_of(org_id)} == {"notified"}
    assert sorted(log.status for log in logs_of(org_id)) == ["failed", "failed", "sent", "sent"]


def test_no_recipients_means_events_without_mail(admin_ctx, dataset, org_id, smtp):
    rule = make_rule(admin_ctx, dataset)
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert result["new_events"] == 2 and result["notifications"] == [] and smtp.messages == []
    assert logs_of(org_id) == []


def test_plan_without_email_alerts_records_events_but_sends_nothing(admin_ctx, dataset, org_id, smtp):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"])
    disable_feature(org_id, Feature.EMAIL_ALERTS)  # plan downgraded after the rule was created
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert result["new_events"] == 2 and result["notifications"] == [] and smtp.messages == []
    assert {e.status for e in events_of(org_id)} == {"new"} and logs_of(org_id) == []


def test_cooldown_suppresses_notifications_until_it_expires(admin_ctx, dataset, org_id, smtp):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"], cooldown_minutes=60)
    first = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert first["cooldown_suppressed"] == 0 and len(smtp.messages) == 1

    append_month(admin_ctx, dataset, 20)  # new finding 2024-06 (-80%)
    second = svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(minutes=30))
    assert (second["new_events"], second["cooldown_suppressed"], second["notifications"]) == (1, 1, [])
    assert len(smtp.messages) == 1
    suppressed = [e for e in events_of(org_id) if e.status == "suppressed"]
    assert len(suppressed) == 1 and suppressed[0].suppressed_reason == "cooldown"
    assert second["events"][0]["status"] == "suppressed"

    append_month(admin_ctx, dataset, 200)  # new finding 2024-07 (+900%) after the cooldown has expired
    third = svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(minutes=61))
    assert (third["new_events"], third["cooldown_suppressed"]) == (1, 0)
    assert [n["status"] for n in third["notifications"]] == ["sent"] and len(smtp.messages) == 2
    statuses = sorted(e.status for e in events_of(org_id))
    assert statuses == ["notified", "notified", "notified", "suppressed"]


def test_cooldown_boundary_and_zero_cooldown(admin_ctx, dataset, org_id, smtp):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"], cooldown_minutes=10)
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    append_month(admin_ctx, dataset, 20)
    assert (
        svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(minutes=10))["cooldown_suppressed"] == 0
    )  # exactly expired
    zero = make_rule(admin_ctx, dataset, name="zero", recipients=["ops@example.com"], cooldown_minutes=0)
    svc.run_rule(admin_ctx, zero["id"], now=T0)
    append_month(admin_ctx, dataset, 2000)
    assert svc.run_rule(admin_ctx, zero["id"], now=T0)["cooldown_suppressed"] == 0


def test_cooldown_ignores_failed_and_skipped_attempts(admin_ctx, dataset, org_id):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"], cooldown_minutes=60)
    svc.run_rule(admin_ctx, rule["id"], now=T0)  # SMTP missing: skipped
    append_month(admin_ctx, dataset, 20)
    second = svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(minutes=5))
    assert second["cooldown_suppressed"] == 0 and second["new_events"] == 1


def test_digest_lists_at_most_25_events_and_sends_one_message(admin_ctx, factory, org_id, smtp):
    values = [100.0 if i % 2 == 0 else 10.0 for i in range(40)]
    ds = factory.dataset(admin_ctx, monthly_frame(values), "flapping")
    rule = make_rule(admin_ctx, ds, recipients=["ops@example.com"])
    result = svc.run_rule(admin_ctx, rule["id"], now=T0)
    assert result["new_events"] == 39 and len(smtp.messages) == 1
    text = smtp.messages[0][0].get_payload()[0].get_payload(decode=True).decode()
    assert "25. " in text and "26. " not in text and "and 14 more" in text


# ---------------------------------------------------------------------------
# run_all_rules and the scheduler
# ---------------------------------------------------------------------------


def test_run_all_rules_runs_active_rules_and_isolates_failures(admin_ctx, dataset, factory, org_id):
    good = make_rule(admin_ctx, dataset, name="good")
    inactive = make_rule(admin_ctx, dataset, name="inactive", is_active=False)
    broken = svc.create_rule(
        admin_ctx, dataset_id=dataset, name="broken", method="iqr", threshold=1.5, metric_column="units"
    )
    versioning_service.apply_frame_operation(
        admin_ctx,
        dataset,
        operation_type="drop_columns",
        parameters={},
        transform=lambda df: (df.drop(columns=["units"]), {}, {}),
    )
    results = {r["rule_id"]: r for r in svc.run_all_rules(admin_ctx)}
    assert set(results) == {good["id"], broken["id"]} and inactive["id"] not in results
    assert results[good["id"]]["status"] == "ok" and results[good["id"]]["new_events"] == 2
    assert results[broken["id"]]["status"] == "failed" and "units" in results[broken["id"]]["error"]
    assert results[broken["id"]]["rule_name"] == "broken"


def _two_scheduled_rules(admin_ctx, dataset):
    hourly = make_rule(admin_ctx, dataset, name="hourly", schedule="hourly")
    daily = make_rule(admin_ctx, dataset, name="daily", schedule="daily")
    make_rule(admin_ctx, dataset, name="manual", schedule="manual")
    make_rule(admin_ctx, dataset, name="paused", schedule="hourly", is_active=False)
    return hourly, daily


def test_scheduler_runs_only_due_rules(admin_ctx, dataset, org_id):
    hourly, daily = _two_scheduled_rules(admin_ctx, dataset)
    first = svc.run_scheduled_checks(now=T0)
    assert (
        first["organizations_checked"],
        first["rules_due"],
        first["rules_run"],
        first["rules_failed"],
    ) == (1, 2, 2, 0)
    assert first["new_events"] == 4 and first["duplicate_suppressed"] == 0  # duplicates are tracked per rule
    with session_scope() as db:
        assert db.get(AnomalyRule, hourly["id"]).last_checked_at == T0

    soon = svc.run_scheduled_checks(now=T0 + timedelta(minutes=30))
    assert (soon["rules_due"], soon["rules_not_due"]) == (0, 2)
    hour = svc.run_scheduled_checks(now=T0 + timedelta(minutes=56))  # inside the five minute grace
    assert (hour["rules_run"], hour["rules_not_due"]) == (1, 1)
    day = svc.run_scheduled_checks(now=T0 + timedelta(hours=24, minutes=1))
    assert day["rules_run"] == 2
    with session_scope() as db:
        assert db.get(AnomalyRule, daily["id"]).last_checked_at == T0 + timedelta(hours=24, minutes=1)
        audit = db.execute(select(AuditLog).where(AuditLog.action == "anomaly.run").limit(1)).scalar_one()
        assert audit.actor_user_id is None and audit.actor_role == "system"


def test_scheduler_sends_notifications_and_counts_them(admin_ctx, dataset, smtp):
    make_rule(admin_ctx, dataset, name="hourly", schedule="hourly", recipients=["ops@example.com"])
    summary = svc.run_scheduled_checks(now=T0)
    assert summary["notifications_sent"] == 1 and summary["new_events"] == 2 and len(smtp.messages) == 1


def test_scheduler_skips_ineligible_organizations(admin_ctx, dataset, factory, org_id):
    make_rule(admin_ctx, dataset, name="mine", schedule="hourly")
    suspended_org = factory.organization("Suspended")
    suspended_admin = factory.org_admin(suspended_org)
    suspended_ds = factory.dataset(suspended_admin, monthly_frame([100, 50, 100]), "s")
    make_rule(suspended_admin, suspended_ds, schedule="hourly")
    downgraded_org = factory.organization("Downgraded")
    downgraded_admin = factory.org_admin(downgraded_org)
    downgraded_ds = factory.dataset(downgraded_admin, monthly_frame([100, 50, 100]), "d")
    make_rule(downgraded_admin, downgraded_ds, schedule="hourly")
    with session_scope() as db:
        db.get(Organization, suspended_org).status = "suspended"
        basic_id = db.execute(
            select(SubscriptionPlan.id).where(SubscriptionPlan.code == "basic")
        ).scalar_one()
        db.execute(
            update(Subscription)
            .where(Subscription.organization_id == downgraded_org)
            .values(plan_id=basic_id)
        )
    summary = svc.run_scheduled_checks(now=T0)
    assert (summary["organizations_checked"], summary["rules_run"]) == (1, 1)
    assert events_of(suspended_org) == [] and events_of(downgraded_org) == []


def test_scheduler_covers_every_eligible_organization(admin_ctx, dataset, factory, org_id):
    make_rule(admin_ctx, dataset, name="a", schedule="hourly")
    other_org = factory.organization("Other")
    other_admin = factory.org_admin(other_org)
    other_ds = factory.dataset(other_admin, monthly_frame([100, 50, 100]), "o")
    make_rule(other_admin, other_ds, name="b", schedule="daily")
    summary = svc.run_scheduled_checks(now=T0)
    assert (summary["organizations_checked"], summary["rules_run"]) == (2, 2)
    assert len(events_of(org_id)) == 2 and len(events_of(other_org)) == 2


def test_one_failing_rule_does_not_stop_the_others(admin_ctx, dataset, org_id, monkeypatch):
    first = make_rule(admin_ctx, dataset, name="first", schedule="hourly")
    second = make_rule(admin_ctx, dataset, name="second", schedule="hourly")
    third = make_rule(admin_ctx, dataset, name="third", schedule="hourly")
    real = svc.run_rule

    def flaky(ctx, rule_id, **kwargs):
        if rule_id == first["id"]:
            raise RuntimeError("boom")
        if rule_id == second["id"]:
            raise NotFoundError("gone")
        return real(ctx, rule_id, **kwargs)

    monkeypatch.setattr(svc, "run_rule", flaky)
    summary = svc.run_scheduled_checks(now=T0)
    assert (summary["rules_due"], summary["rules_run"], summary["rules_failed"]) == (3, 1, 2)
    with session_scope() as db:
        assert db.get(AnomalyRule, third["id"]).last_checked_at == T0
        assert db.get(AnomalyRule, first["id"]).last_checked_at is None
        errors = db.execute(select(ErrorEvent)).scalars().all()
        assert [e.component for e in errors] == ["anomaly.scheduled_check"] and errors[
            0
        ].organization_id == org_id


def test_scheduler_counts_a_broken_rule_as_failed(admin_ctx, dataset):
    make_rule(admin_ctx, dataset, name="ok", schedule="hourly")
    svc.create_rule(
        admin_ctx,
        dataset_id=dataset,
        name="units",
        method="iqr",
        threshold=1.5,
        metric_column="units",
        schedule="hourly",
    )
    versioning_service.apply_frame_operation(
        admin_ctx,
        dataset,
        operation_type="drop_columns",
        parameters={},
        transform=lambda df: (df.drop(columns=["units"]), {}, {}),
    )
    summary = svc.run_scheduled_checks(now=T0)
    assert (summary["rules_run"], summary["rules_failed"]) == (1, 1)


def test_scheduler_with_nothing_to_do():
    summary = svc.run_scheduled_checks(now=T0)
    assert summary["rules_due"] == 0 and summary["rules_run"] == 0 and summary["organizations_checked"] == 0


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_list_events_filters_orders_and_limits(admin_ctx, dataset):
    one = make_rule(admin_ctx, dataset, name="one")
    two = make_rule(admin_ctx, dataset, name="two", method="metric_drop")
    svc.run_rule(admin_ctx, one["id"], now=T0)
    svc.run_rule(admin_ctx, two["id"], now=T0 + timedelta(hours=1))
    everything = svc.list_events(admin_ctx)
    assert len(everything) == 3 and everything[0]["rule_name"] == "two"  # newest first
    assert len(svc.list_events(admin_ctx, rule_id=one["id"])) == 2
    assert (
        len(svc.list_events(admin_ctx, status="new")) == 3
        and svc.list_events(admin_ctx, status="notified") == []
    )
    assert len(svc.list_events(admin_ctx, limit=1)) == 1 and len(svc.list_events(admin_ctx, limit=0)) == 1
    with pytest.raises(ValidationError):
        svc.list_events(admin_ctx, status="open")


def test_acknowledge_event(admin_ctx, dataset, org_id):
    rule = make_rule(admin_ctx, dataset)
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    event_id = svc.list_events(admin_ctx)[0]["id"]
    acknowledged = svc.acknowledge_event(admin_ctx, event_id)
    assert (
        acknowledged["status"] == "acknowledged" and acknowledged["acknowledged_by_id"] == admin_ctx.user_id
    )
    assert acknowledged["acknowledged_at"] is not None
    again = svc.acknowledge_event(admin_ctx, event_id)
    assert again["acknowledged_at"] == acknowledged["acknowledged_at"]
    assert len(svc.list_events(admin_ctx, status="acknowledged")) == 1
    with session_scope() as db:
        assert (
            db.execute(
                select(func.count(AuditLog.id)).where(AuditLog.action == "anomaly.event.acknowledge")
            ).scalar_one()
            == 1
        )
    # An acknowledged event keeps suppressing the same finding.
    assert svc.run_rule(admin_ctx, rule["id"], now=T0)["new_events"] == 0


# ---------------------------------------------------------------------------
# Authorization and tenant isolation
# ---------------------------------------------------------------------------


def test_analyst_cannot_use_any_anomaly_function(admin_ctx, analyst_ctx, dataset):
    rule = make_rule(admin_ctx, dataset)
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    event_id = svc.list_events(admin_ctx)[0]["id"]
    calls = [
        lambda: svc.create_rule(
            analyst_ctx, dataset_id=dataset, name="x", method="iqr", threshold=1.5, metric_column="revenue"
        ),
        lambda: svc.update_rule(analyst_ctx, rule["id"], threshold=5),
        lambda: svc.set_rule_active(analyst_ctx, rule["id"], False),
        lambda: svc.delete_rule(analyst_ctx, rule["id"]),
        lambda: svc.list_rules(analyst_ctx),
        lambda: svc.run_rule(analyst_ctx, rule["id"]),
        lambda: svc.run_all_rules(analyst_ctx),
        lambda: svc.list_events(analyst_ctx),
        lambda: svc.acknowledge_event(analyst_ctx, event_id),
        lambda: svc.list_notifications(analyst_ctx),
    ]
    for call in calls:
        with pytest.raises(PermissionDeniedError):
            call()
    assert svc.list_rules(admin_ctx)[0]["threshold"] == 30.0


def test_super_admin_is_not_an_organization_member(factory, dataset):
    with pytest.raises(PermissionDeniedError):
        svc.list_rules(factory.super_admin())


def test_basic_plan_is_denied_everywhere(factory):
    basic_org = factory.organization(plan="basic")
    admin = factory.org_admin(basic_org)
    ds = factory.dataset(admin, monthly_frame([1, 2, 3]), "b")
    calls = [
        lambda: svc.create_rule(
            admin, dataset_id=ds, name="x", method="iqr", threshold=1.5, metric_column="revenue"
        ),
        lambda: svc.update_rule(admin, 1, threshold=5),
        lambda: svc.set_rule_active(admin, 1, True),
        lambda: svc.delete_rule(admin, 1),
        lambda: svc.list_rules(admin),
        lambda: svc.run_rule(admin, 1),
        lambda: svc.run_all_rules(admin),
        lambda: svc.list_events(admin),
        lambda: svc.acknowledge_event(admin, 1),
        lambda: svc.list_notifications(admin),
    ]
    for call in calls:
        with pytest.raises(FeatureNotAvailableError):
            call()


def test_cross_tenant_ids_are_not_found(admin_ctx, dataset, factory, org_id):
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"])
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    event_id = svc.list_events(admin_ctx)[0]["id"]

    other_org = factory.organization("Rival")
    other = factory.org_admin(other_org)
    other_ds = factory.dataset(other, monthly_frame([1, 2, 3]), "theirs")
    own_rule = make_rule(other, other_ds, name="theirs")

    with pytest.raises(NotFoundError):
        svc.create_rule(
            other, dataset_id=dataset, name="x", method="iqr", threshold=1.5, metric_column="revenue"
        )
    for call in (
        lambda: svc.update_rule(other, rule["id"], threshold=5),
        lambda: svc.set_rule_active(other, rule["id"], False),
        lambda: svc.delete_rule(other, rule["id"]),
        lambda: svc.run_rule(other, rule["id"]),
        lambda: svc.list_events(other, rule_id=rule["id"]),
        lambda: svc.acknowledge_event(other, event_id),
    ):
        with pytest.raises(NotFoundError):
            call()
    assert (
        svc.list_rules(other) == [svc.list_rules(other)[0]]
        and svc.list_rules(other)[0]["id"] == own_rule["id"]
    )
    assert svc.list_events(other) == [] and svc.list_notifications(other) == []
    assert svc.run_all_rules(other)[0]["rule_id"] == own_rule["id"]
    assert svc.list_rules(admin_ctx)[0]["threshold"] == 30.0
    assert {e.rule_id for e in events_of(other_org)} == {own_rule["id"]}


def test_archived_dataset_cannot_get_new_rules(admin_ctx, dataset):
    from database.models import Dataset

    with session_scope() as db:
        db.get(Dataset, dataset).status = "archived"
    with pytest.raises(NotFoundError):
        make_rule(admin_ctx, dataset)


# ---------------------------------------------------------------------------
# Worker script
# ---------------------------------------------------------------------------


def test_script_prints_a_summary_and_exits_zero(admin_ctx, dataset, capsys, monkeypatch):
    make_rule(admin_ctx, dataset, schedule="hourly")
    monkeypatch.setattr(run_anomaly_checks, "configure_logging", lambda settings: None)
    assert run_anomaly_checks.main([]) == 0
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0 and "1 rule(s) run" in out and "2 new event(s)" in out and "0 failed" in out


def test_script_rejects_unknown_arguments(capsys):
    with pytest.raises(SystemExit) as info:
        run_anomaly_checks.main(["--bogus"])
    assert info.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


def test_script_exits_one_when_the_worker_fails(capsys, monkeypatch):
    monkeypatch.setattr(run_anomaly_checks, "configure_logging", lambda settings: None)

    def explode(**kwargs):
        raise RuntimeError("database is locked secret-detail")

    monkeypatch.setattr(run_anomaly_checks.anomaly_service, "run_scheduled_checks", explode)
    assert run_anomaly_checks.main([]) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err and "secret-detail" not in captured.err and captured.out == ""


def test_script_runs_from_the_project_root(tmp_path):
    env = {
        **os.environ,
        "DATABASE_PATH": str(tmp_path / "worker.db"),
        "STORAGE_ROOT": str(tmp_path / "storage"),
        "LOG_DIR": str(tmp_path / "logs"),
        "BACKUP_DIR": str(tmp_path / "backups"),
        "APP_ENV": "test",
    }
    done = subprocess.run(
        [sys.executable, "scripts/run_anomaly_checks.py"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "Anomaly checks finished: 0 rule(s) run" in done.stdout
    assert (tmp_path / "worker.db").exists()


def test_concurrent_runs_cannot_duplicate_events_or_alerts(admin_ctx, dataset, org_id, smtp, monkeypatch):
    """Regression: (rule_id, fingerprint) is unique in the database, so a run that
    raced another run of the same rule rolls back instead of duplicating alerts."""
    rule = make_rule(admin_ctx, dataset, recipients=["ops@example.com"])
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    before_events, before_logs, before_mail = len(events_of(org_id)), len(logs_of(org_id)), len(smtp.messages)
    # Simulate the race: this run's duplicate check misses the rows the other run committed.
    monkeypatch.setattr(svc, "_existing_fingerprints", lambda *args, **kwargs: set())
    with pytest.raises(ConflictError, match="same moment"):
        svc.run_rule(admin_ctx, rule["id"], now=T0 + timedelta(hours=2))
    assert len(events_of(org_id)) == before_events
    assert len(logs_of(org_id)) == before_logs
    assert len(smtp.messages) == before_mail


def test_event_fingerprints_are_unique_per_rule_in_the_database(admin_ctx, dataset, org_id):
    from sqlalchemy.exc import IntegrityError

    rule = make_rule(admin_ctx, dataset)
    svc.run_rule(admin_ctx, rule["id"], now=T0)
    existing = events_of(org_id)[0]
    with pytest.raises(IntegrityError):
        with session_scope() as db:
            db.add(
                AnomalyEvent(
                    organization_id=org_id,
                    rule_id=existing.rule_id,
                    dataset_id=existing.dataset_id,
                    fingerprint=existing.fingerprint,
                    severity="low",
                    title="dup",
                    status="new",
                )
            )
