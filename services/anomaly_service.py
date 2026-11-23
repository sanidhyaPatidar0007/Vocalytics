"""Explainable anomaly detection with history, duplicate suppression, cooldowns and e-mail alerts.

Everything here is a statistical flag for human review. Titles use neutral wording
("Possible unusual drop in revenue ...") and never call a value fraud or a confirmed
error. All management functions need the Organization Admin role and the
``ANOMALY_DETECTION`` plan feature; rule recipients additionally need ``EMAIL_ALERTS``.

Methods and thresholds (a value exactly on the threshold is never flagged: every
comparison is strictly greater, so a threshold of 10 flags 10.01 but not 10)

* ``iqr``: numeric ``metric_column``. Flags values outside ``Q1 - k*IQR .. Q3 + k*IQR``
  with ``k = threshold`` (typical 1.5; quartiles by linear interpolation). Needs at
  least 4 values. One aggregated finding per direction (above and below) per dataset version.
* ``zscore``: numeric ``metric_column``. Flags values with ``|z| > threshold`` (typical 3.0)
  where ``z = (x - mean) / std`` and ``std`` is the sample standard deviation. Needs at
  least 3 values and a non-zero deviation. One aggregated finding per direction per version.
* ``period_change``: ``date_column`` + ``metric_column`` + ``aggregation`` + ``frequency``
  (D, W or M). The metric is aggregated per calendar period; a period is flagged when its
  percent change against the previous period that has data exceeds ``threshold`` percent
  in either direction. A previous value of 0 gives no defined percent change and is skipped.
* ``metric_drop``: like ``period_change`` but only drops of more than ``threshold`` percent
  (revenue or sales falling). ``threshold`` is at most 100.
* ``baseline_deviation``: an explainable predictive baseline. The latest period is compared with
  the mean of the previous 6 periods; it is flagged when it lies more than ``threshold``
  sample standard deviations away. Needs 7 periods and a non-zero spread in the baseline.
* ``missing_increase``: compares the share of missing or invalid cells per column between the
  current and the previous dataset version and flags increases of more than ``threshold``
  percentage points (``threshold`` at most 100). ``metric_column`` is optional (default: all
  columns); with no earlier version there is nothing to compare and nothing is flagged.
  A cell is missing when it is null. A non-null text value is invalid when it does not parse
  as the number or date the column holds: in a column that was numeric or date in the
  previous version and is text now (for example after an append with ``cast_to_text``) every
  value that is not a number or date is invalid; in any other text column that is mostly
  numeric or date-like (at least 80 % of values) the few values that are not are invalid.
  Unparseable values kept as text are therefore not null, but they still count here.

A latest period that is still incomplete (for example the current month) can look like a drop;
review such flags with that in mind.

Event identity and notifications
* ``fingerprint = sha256(rule id | method | column | period | direction)``. For ``iqr``, ``zscore``
  and ``missing_increase`` the "period" is the dataset version ("version 3"), so a new version
  can raise a new event while re-running on the same version is a duplicate.
* A finding whose fingerprint already exists for the rule creates no event
  (``duplicate_suppressed``).
* Cooldown: when a notification for the rule was sent less than ``cooldown_minutes`` minutes ago
  (exactly that long ago counts as expired), new events are stored as ``suppressed`` (reason
  ``cooldown``) and not e-mailed. It is evaluated once per run.
* One digest e-mail per run and recipient lists the new events (a first run over a long history
  must not send hundreds of messages). Every attempt is logged per event and recipient in
  ``notification_logs`` as sent, failed or skipped. An event becomes ``notified`` when at least
  one recipient received the digest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth.authorization import AuthContext, authorize, authorize_dataset
from config.settings import get_settings
from core import timeutil
from core.constants import Feature, LogicalType, OrgStatus, Role, UsageMetric
from core.errors import AppError, ConflictError, NotFoundError, ValidationError
from database.connection import session_scope
from database.models import (
    AnomalyEvent,
    AnomalyRule,
    Dataset,
    DatasetVersion,
    NotificationLog,
    Organization,
)
from modules import subscriptions
from modules.usage_tracking import record_usage
from services import email_service, versioning_service
from services.audit_service import record_audit, record_error
from services.quality_score import count_invalid_text, text_parse_counts

logger = logging.getLogger(__name__)

METHODS = ("iqr", "zscore", "period_change", "metric_drop", "baseline_deviation", "missing_increase")
VALUE_METHODS = ("iqr", "zscore")
PERIOD_METHODS = ("period_change", "metric_drop", "baseline_deviation")
AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")
FREQUENCIES = ("D", "W", "M")
SEVERITIES = ("low", "medium", "high", "critical")
SCHEDULES = ("manual", "hourly", "daily", "weekly")
EVENT_STATUSES = ("new", "notified", "suppressed", "acknowledged")
SCHEDULE_INTERVALS = {"hourly": timedelta(hours=1), "daily": timedelta(days=1), "weekly": timedelta(days=7)}
DUE_GRACE = timedelta(minutes=5)
BASELINE_WINDOW = 6
MIN_IQR_VALUES = 4
MIN_ZSCORE_VALUES = 3
MAX_COOLDOWN_MINUTES = 60 * 24 * 30
MAX_DIGEST_EVENTS = 25
_SQL_CHUNK = 500
_UNIT = {"D": "day", "W": "week", "M": "month"}
_NUMERIC_TYPES = (LogicalType.INTEGER.value, LogicalType.FLOAT.value)
_UPDATABLE = (
    "name",
    "method",
    "threshold",
    "metric_column",
    "aggregation",
    "date_column",
    "frequency",
    "severity",
    "recipients",
    "schedule",
    "cooldown_minutes",
    "is_active",
)
_COLUMN_SETTINGS = ("method", "metric_column", "aggregation", "date_column", "frequency")


# ---------------------------------------------------------------------------
# Pure detectors (no database, no I/O)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """One explainable flag produced by a detector."""

    method: str
    column: str
    period: str
    direction: str
    observed: float
    expected: float | None
    lower_bound: float | None
    upper_bound: float | None
    threshold: float
    title: str
    details: dict[str, Any] = field(default_factory=dict)


def _clean_text(value: object, limit: int = 80) -> str:
    """Make untrusted text (column names) safe for titles and subjects: one line, bounded."""
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", str(value)).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _finite_values(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    return numeric[np.isfinite(numeric)].reset_index(drop=True)


def _extremes(values: pd.Series, *, largest: bool, count: int = 5) -> list[float]:
    picked = values.nlargest(count) if largest else values.nsmallest(count)
    return [float(v) for v in picked]


def detect_iqr(values: pd.Series, threshold: float = 1.5, *, column: str, period: str) -> list[Finding]:
    """Flag values outside ``Q1 - k*IQR .. Q3 + k*IQR`` (``k = threshold``), per direction."""
    data = _finite_values(values)
    if len(data) < MIN_IQR_VALUES:
        return []
    q1, q3 = float(data.quantile(0.25)), float(data.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    median = float(data.median())
    name = _clean_text(column)
    findings: list[Finding] = []
    for direction, mask, largest in (("above", data > upper, True), ("below", data < lower, False)):
        flagged = data[mask]
        if flagged.empty:
            continue
        extreme = _extremes(flagged, largest=largest)
        bound = upper if largest else lower
        findings.append(
            Finding(
                method="iqr",
                column=column,
                period=period,
                direction=direction,
                observed=extreme[0],
                expected=median,
                lower_bound=lower,
                upper_bound=upper,
                threshold=float(threshold),
                title=(
                    f"Possible unusual values in {name}: {len(flagged)} value(s) {direction} the expected "
                    f"range limit of {_fmt(bound)} (most extreme {_fmt(extreme[0])}), {period}"
                ),
                details={
                    "count": int(len(flagged)),
                    "total_values": int(len(data)),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "most_extreme_values": extreme,
                },
            )
        )
    return findings


def detect_zscore(values: pd.Series, threshold: float = 3.0, *, column: str, period: str) -> list[Finding]:
    """Flag values with ``|z| > threshold`` (sample standard deviation), per direction."""
    data = _finite_values(values)
    if len(data) < MIN_ZSCORE_VALUES:
        return []
    mean, std = float(data.mean()), float(data.std(ddof=1))
    if not math.isfinite(std) or std == 0:
        return []
    z = (data - mean) / std
    lower, upper = mean - threshold * std, mean + threshold * std
    name = _clean_text(column)
    findings: list[Finding] = []
    for direction, mask, largest in (("above", z > threshold, True), ("below", z < -threshold, False)):
        flagged = data[mask]
        if flagged.empty:
            continue
        extreme = _extremes(flagged, largest=largest)
        findings.append(
            Finding(
                method="zscore",
                column=column,
                period=period,
                direction=direction,
                observed=extreme[0],
                expected=mean,
                lower_bound=lower,
                upper_bound=upper,
                threshold=float(threshold),
                title=(
                    f"Possible unusual values in {name}: {len(flagged)} value(s) more than {_fmt(threshold)} "
                    f"standard deviations {direction} the mean (most extreme {_fmt(extreme[0])}, "
                    f"z = {(extreme[0] - mean) / std:+.2f}), {period}"
                ),
                details={
                    "count": int(len(flagged)),
                    "total_values": int(len(data)),
                    "mean": mean,
                    "std": std,
                    "most_extreme_values": extreme,
                },
            )
        )
    return findings


def aggregate_by_period(
    frame: pd.DataFrame, *, date_column: str, metric_column: str, aggregation: str, frequency: str
) -> pd.Series:
    """Aggregate ``metric_column`` per calendar period (D, W or M) into a period-indexed series.

    Rows without a date are ignored and only periods that contain data appear (an empty period
    is not turned into a zero). A period whose metric values are all missing is dropped.
    """
    if aggregation not in AGGREGATIONS or frequency not in FREQUENCIES:
        raise ValidationError("Unsupported aggregation or frequency.")
    if aggregation == "count":
        values = frame[metric_column]
    else:
        values = frame[metric_column].astype("float64")
    data = pd.DataFrame({"period": frame[date_column].dt.to_period(frequency), "value": values})
    grouped = data.dropna(subset=["period"]).groupby("period", sort=True)["value"]
    series = grouped.sum(min_count=1) if aggregation == "sum" else getattr(grouped, aggregation)()
    return series.astype("float64").dropna()


def detect_period_change(
    series: pd.Series, threshold: float, *, column: str, frequency: str, drops_only: bool = False
) -> list[Finding]:
    """Flag periods whose percent change versus the previous period exceeds ``threshold``.

    ``drops_only`` keeps only decreases (the ``metric_drop`` method). A previous value of 0
    has no percent change and is skipped.
    """
    method = "metric_drop" if drops_only else "period_change"
    unit = _UNIT.get(frequency, "period")
    name = _clean_text(column)
    values = series.dropna()
    findings: list[Finding] = []
    for position in range(1, len(values)):
        previous, current = float(values.iloc[position - 1]), float(values.iloc[position])
        if previous == 0:
            continue
        percent = (current - previous) * 100.0 / abs(previous)
        # Rounded for the comparison only: 8.47 vs 7.70 is a float error away from exactly 10 percent,
        # and a value exactly on the threshold is never flagged.
        compared = round(percent, 9)
        if not (compared < -threshold if drops_only else abs(compared) > threshold):
            continue
        period, previous_period = str(values.index[position]), str(values.index[position - 1])
        decrease = percent < 0
        wording = "drop" if decrease else "increase"
        findings.append(
            Finding(
                method=method,
                column=column,
                period=period,
                direction="decrease" if decrease else "increase",
                observed=current,
                expected=previous,
                lower_bound=None,
                upper_bound=None,
                threshold=float(threshold),
                title=(
                    f"Possible unusual {wording} in {name} for {period}: {percent:+.1f}% vs previous {unit}"
                ),
                details={
                    "previous_period": previous_period,
                    "percent_change": percent,
                    "frequency": frequency,
                },
            )
        )
    return findings


def detect_baseline_deviation(
    series: pd.Series, threshold: float, *, column: str, frequency: str, window: int = BASELINE_WINDOW
) -> list[Finding]:
    """Flag the latest period when it is more than ``threshold`` deviations from the recent mean."""
    values = series.dropna()
    if len(values) < window + 1:
        return []
    baseline = values.iloc[-(window + 1) : -1].astype("float64")
    mean, std = float(baseline.mean()), float(baseline.std(ddof=1))
    if not math.isfinite(std) or std == 0:
        return []
    latest = float(values.iloc[-1])
    z = (latest - mean) / std
    if abs(z) <= threshold:
        return []
    period = str(values.index[-1])
    unit = _UNIT.get(frequency, "period")
    direction = "above" if z > 0 else "below"
    return [
        Finding(
            method="baseline_deviation",
            column=column,
            period=period,
            direction=direction,
            observed=latest,
            expected=mean,
            lower_bound=mean - threshold * std,
            upper_bound=mean + threshold * std,
            threshold=float(threshold),
            title=(
                f"Possible unusual value of {_clean_text(column)} for {period}: {abs(z):.1f} standard "
                f"deviations {direction} the mean of the previous {window} {unit}s"
            ),
            details={
                "z_score": z,
                "baseline_mean": mean,
                "baseline_std": std,
                "baseline_periods": window,
                "frequency": frequency,
            },
        )
    ]


def detect_missing_increase(
    current: Mapping[str, float],
    previous: Mapping[str, float],
    threshold: float,
    *,
    current_label: str,
    previous_label: str,
    invalid_current: Mapping[str, float] | None = None,
    invalid_previous: Mapping[str, float] | None = None,
) -> list[Finding]:
    """Flag columns whose missing (or invalid) percentage grew by more than ``threshold`` points.

    ``current`` and ``previous`` hold the combined percentage per column. ``invalid_current`` and
    ``invalid_previous`` give the invalid part of it, which only changes the wording and details.
    """
    invalid_current = invalid_current or {}
    invalid_previous = invalid_previous or {}
    findings: list[Finding] = []
    for column, now_pct in current.items():
        if column not in previous:
            continue
        before = float(previous[column])
        delta = float(now_pct) - before
        if delta <= threshold:
            continue
        invalid_now = float(invalid_current.get(column, 0.0))
        invalid_before = float(invalid_previous.get(column, 0.0))
        has_invalid = invalid_now > 0 or invalid_before > 0
        kind = "missing or invalid data" if has_invalid else "missing data"
        findings.append(
            Finding(
                method="missing_increase",
                column=column,
                period=current_label,
                direction="increase",
                observed=float(now_pct),
                expected=before,
                lower_bound=None,
                upper_bound=None,
                threshold=float(threshold),
                title=(
                    f"Possible increase in {kind} in {_clean_text(column)}: {before:.1f}% to "
                    f"{float(now_pct):.1f}% ({delta:+.1f} points) in {current_label} vs {previous_label}"
                ),
                details={
                    "previous_percent_missing": before,
                    "percent_missing": float(now_pct),
                    "increase_points": delta,
                    "compared_with": previous_label,
                    **(
                        {"percent_invalid": invalid_now, "previous_percent_invalid": invalid_before}
                        if has_invalid
                        else {}
                    ),
                },
            )
        )
    return findings


def rule_is_due(schedule: str, last_checked_at: datetime | None, now: datetime) -> bool:
    """Whether a scheduled rule should run now (manual rules never; a 5 minute grace is allowed)."""
    interval = SCHEDULE_INTERVALS.get(schedule)
    if interval is None:
        return False
    return last_checked_at is None or now - last_checked_at >= interval - DUE_GRACE


def make_fingerprint(rule_id: int, method: str, column: str, period: str, direction: str) -> str:
    """Stable identity of a finding for one rule (used for duplicate suppression)."""
    return hashlib.sha256(f"{rule_id}|{method}|{column}|{period}|{direction}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers shared by the database functions
# ---------------------------------------------------------------------------


def _authorize_admin(db: Session, ctx: AuthContext) -> int:
    return authorize(db, ctx, roles=(Role.ORG_ADMIN,), feature=Feature.ANOMALY_DETECTION)


def _check_id(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValidationError(f"Invalid {label}.")
    return value


def _get_rule(db: Session, org_id: int, rule_id: object) -> AnomalyRule:
    rule = db.get(AnomalyRule, _check_id(rule_id, "rule"))
    if rule is None or rule.organization_id != org_id:
        raise NotFoundError("Alert rule not found.")
    return rule


def _chunks(items: list[Any], size: int = _SQL_CHUNK) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and non-finite floats so the value is valid JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _finite_or_none(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _event_counts(db: Session, org_id: int, rule_ids: list[int]) -> dict[int, int]:
    if not rule_ids:
        return {}
    rows = db.execute(
        select(AnomalyEvent.rule_id, func.count(AnomalyEvent.id))
        .where(AnomalyEvent.organization_id == org_id, AnomalyEvent.rule_id.in_(rule_ids))
        .group_by(AnomalyEvent.rule_id)
    ).all()
    return {rule_id: int(count) for rule_id, count in rows}


def _rule_dict(rule: AnomalyRule, dataset_name: str, event_count: int = 0) -> dict[str, Any]:
    return {
        "id": rule.id,
        "dataset_id": rule.dataset_id,
        "dataset_name": dataset_name,
        "name": rule.name,
        "method": rule.method,
        "metric_column": rule.metric_column,
        "aggregation": rule.aggregation,
        "date_column": rule.date_column,
        "frequency": rule.frequency,
        "threshold": rule.threshold,
        "severity": rule.severity,
        "recipients": json.loads(rule.recipients_json or "[]"),
        "schedule": rule.schedule,
        "cooldown_minutes": rule.cooldown_minutes,
        "is_active": rule.is_active,
        "last_checked_at": rule.last_checked_at,
        "created_at": rule.created_at,
        "event_count": event_count,
    }


def _rule_view(db: Session, org_id: int, rule: AnomalyRule) -> dict[str, Any]:
    dataset = db.get(Dataset, rule.dataset_id)
    count = _event_counts(db, org_id, [rule.id]).get(rule.id, 0)
    return _rule_dict(rule, dataset.name if dataset else "", count)


# ---------------------------------------------------------------------------
# Rule validation and management
# ---------------------------------------------------------------------------


def _dataset_types(db: Session, dataset: Dataset) -> dict[str, str]:
    version = db.get(DatasetVersion, dataset.current_version_id) if dataset.current_version_id else None
    if version is None:
        raise NotFoundError("This dataset has no data yet.")
    return {column["name"]: column["logical_type"] for column in json.loads(version.schema_json)}


def _validate_threshold(method: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValidationError("The threshold must be a number greater than 0.")
    if method in ("metric_drop", "missing_increase") and value > 100:
        raise ValidationError("For this method the threshold is a percentage and cannot exceed 100.")
    return float(value)


def _validate_name(db: Session, org_id: int, name: object, exclude_id: int | None) -> str:
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 200:
        raise ValidationError("The rule name must be 1 to 200 characters.")
    cleaned = name.strip()
    query = select(AnomalyRule.id).where(
        AnomalyRule.organization_id == org_id,
        or_(AnomalyRule.name == cleaned, func.lower(AnomalyRule.name) == cleaned.lower()),
    )
    if exclude_id is not None:
        query = query.where(AnomalyRule.id != exclude_id)
    if db.execute(query).first() is not None:
        raise ConflictError("An alert rule with this name already exists.")
    return cleaned


def _validate_choice(value: object, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise ValidationError(f"The {label} must be one of: {', '.join(allowed)}.")
    return str(value)


def _validate_columns(fields: dict[str, Any], types: dict[str, str]) -> None:
    """Check and normalise the method-specific column settings in place."""
    method = fields["method"]
    metric, date = fields.get("metric_column"), fields.get("date_column")
    for label, column in (("metric", metric), ("date", date)):
        if column is not None and (not isinstance(column, str) or column not in types):
            raise ValidationError(
                f"The {label} column '{_clean_text(column, 60)}' does not exist in this dataset."
            )
    if method in VALUE_METHODS:
        if metric is None or types[metric] not in _NUMERIC_TYPES:
            raise ValidationError("This method needs a numeric metric column.")
        fields.update(date_column=None, frequency=None, aggregation="sum")
    elif method in PERIOD_METHODS:
        aggregation = _validate_choice(fields.get("aggregation"), AGGREGATIONS, "aggregation")
        fields["frequency"] = _validate_choice(fields.get("frequency"), FREQUENCIES, "frequency")
        if metric is None or (aggregation != "count" and types[metric] not in _NUMERIC_TYPES):
            raise ValidationError("This method needs a numeric metric column (any column when counting).")
        if date is None or types[date] != LogicalType.DATETIME.value:
            raise ValidationError("This method needs a date column.")
        if date == metric:
            raise ValidationError("The date column and the metric column must be different.")
    else:
        fields.update(date_column=None, frequency=None, aggregation="sum")


def _validate_recipients(db: Session, org_id: int, recipients: object) -> list[str]:
    cleaned = email_service.normalize_recipients([] if recipients is None else recipients)
    if cleaned and not subscriptions.org_has_feature(db, org_id, Feature.EMAIL_ALERTS):
        raise ValidationError(
            "E-mail alerts are not included in your organization's plan, so alert recipients cannot be set. "
            "Events are still recorded in the alert history."
        )
    return cleaned


def _validate_common(
    db: Session, org_id: int, fields: dict[str, Any], changed: set[str], exclude_id: int | None
) -> None:
    if "name" in changed:
        fields["name"] = _validate_name(db, org_id, fields["name"], exclude_id)
    if "method" in changed:
        _validate_choice(fields["method"], METHODS, "method")
    if "threshold" in changed or "method" in changed:
        fields["threshold"] = _validate_threshold(fields["method"], fields["threshold"])
    if "severity" in changed:
        _validate_choice(fields["severity"], SEVERITIES, "severity")
    if "schedule" in changed:
        _validate_choice(fields["schedule"], SCHEDULES, "schedule")
    if "cooldown_minutes" in changed:
        cooldown = fields["cooldown_minutes"]
        if (
            isinstance(cooldown, bool)
            or not isinstance(cooldown, int)
            or not 0 <= cooldown <= MAX_COOLDOWN_MINUTES
        ):
            raise ValidationError(
                f"The cooldown must be a whole number of minutes from 0 to {MAX_COOLDOWN_MINUTES}."
            )
    if "is_active" in changed and not isinstance(fields["is_active"], bool):
        raise ValidationError("The active flag must be true or false.")
    if "recipients" in changed:
        fields["recipients"] = _validate_recipients(db, org_id, fields["recipients"])


def create_rule(
    ctx: AuthContext,
    *,
    dataset_id: int,
    name: str,
    method: str,
    threshold: float,
    metric_column: str | None = None,
    aggregation: str = "sum",
    date_column: str | None = None,
    frequency: str | None = None,
    severity: str = "medium",
    recipients: list[str] | None = None,
    schedule: str = "manual",
    cooldown_minutes: int | None = None,
    is_active: bool = True,
) -> dict[str, Any]:
    """Create an alert rule on a dataset after validating columns and types against its schema."""
    with session_scope() as db:
        dataset = authorize_dataset(
            db, ctx, dataset_id, roles=(Role.ORG_ADMIN,), feature=Feature.ANOMALY_DETECTION
        )
        org_id = dataset.organization_id
        fields: dict[str, Any] = {
            "name": name,
            "method": method,
            "threshold": threshold,
            "metric_column": metric_column,
            "aggregation": aggregation,
            "date_column": date_column,
            "frequency": frequency,
            "severity": severity,
            "recipients": recipients,
            "schedule": schedule,
            "cooldown_minutes": get_settings().alert_default_cooldown_minutes
            if cooldown_minutes is None
            else cooldown_minutes,
            "is_active": is_active,
        }
        _validate_common(db, org_id, fields, set(fields), None)
        _validate_columns(fields, _dataset_types(db, dataset))
        rule = AnomalyRule(
            organization_id=org_id,
            dataset_id=dataset.id,
            name=fields["name"],
            method=fields["method"],
            metric_column=fields["metric_column"],
            aggregation=fields["aggregation"],
            date_column=fields["date_column"],
            frequency=fields["frequency"],
            threshold=fields["threshold"],
            severity=fields["severity"],
            recipients_json=json.dumps(fields["recipients"]),
            schedule=fields["schedule"],
            cooldown_minutes=fields["cooldown_minutes"],
            is_active=fields["is_active"],
            created_by_id=ctx.user_id,
        )
        db.add(rule)
        db.flush()
        record_audit(
            db,
            ctx,
            "anomaly.rule.create",
            target_type="anomaly_rule",
            target_id=rule.id,
            details={
                "name": rule.name,
                "method": rule.method,
                "dataset_id": dataset.id,
                "recipient_count": len(fields["recipients"]),
            },
            organization_id=org_id,
        )
        return _rule_view(db, org_id, rule)


def update_rule(ctx: AuthContext, rule_id: int, **changes: Any) -> dict[str, Any]:
    """Change settings of a rule (the dataset cannot be changed; create a new rule instead).

    Accepted keys: name, method, threshold, metric_column, aggregation, date_column, frequency,
    severity, recipients, schedule, cooldown_minutes, is_active. Column settings are re-validated
    against the dataset whenever any of them changes.
    """
    unknown = sorted(set(changes) - set(_UPDATABLE))
    if unknown:
        raise ValidationError(f"These settings cannot be changed: {', '.join(unknown)}.")
    if not changes:
        raise ValidationError("No changes were given.")
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        rule = _get_rule(db, org_id, rule_id)
        fields: dict[str, Any] = {key: getattr(rule, key) for key in _UPDATABLE if key != "recipients"}
        fields["recipients"] = json.loads(rule.recipients_json or "[]")
        fields.update(changes)
        _validate_common(db, org_id, fields, set(changes), rule.id)
        if set(changes) & set(_COLUMN_SETTINGS):
            _validate_columns(fields, _dataset_types(db, db.get(Dataset, rule.dataset_id)))
        for key in _UPDATABLE:
            if key == "recipients":
                rule.recipients_json = json.dumps(fields["recipients"])
            else:
                setattr(rule, key, fields[key])
        record_audit(
            db,
            ctx,
            "anomaly.rule.update",
            target_type="anomaly_rule",
            target_id=rule.id,
            details={"changed": sorted(changes)},
            organization_id=org_id,
        )
        db.flush()
        return _rule_view(db, org_id, rule)


def set_rule_active(ctx: AuthContext, rule_id: int, active: bool) -> dict[str, Any]:
    """Activate or deactivate a rule (deactivated rules keep their history and are not scheduled)."""
    if not isinstance(active, bool):
        raise ValidationError("The active flag must be true or false.")
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        rule = _get_rule(db, org_id, rule_id)
        rule.is_active = active
        record_audit(
            db,
            ctx,
            "anomaly.rule.activate" if active else "anomaly.rule.deactivate",
            target_type="anomaly_rule",
            target_id=rule.id,
            details={"name": rule.name},
            organization_id=org_id,
        )
        db.flush()
        return _rule_view(db, org_id, rule)


def delete_rule(ctx: AuthContext, rule_id: int) -> dict[str, Any]:
    """Delete a rule that never produced events; otherwise ask to deactivate it to keep history."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        rule = _get_rule(db, org_id, rule_id)
        has_events = db.execute(
            select(AnomalyEvent.id).where(AnomalyEvent.rule_id == rule.id).limit(1)
        ).first()
        has_logs = db.execute(
            select(NotificationLog.id).where(NotificationLog.rule_id == rule.id).limit(1)
        ).first()
        if has_events or has_logs:
            raise ValidationError(
                "This rule already has recorded events, so it cannot be deleted. "
                "Deactivate it instead to stop checks and keep the history."
            )
        name = rule.name
        db.execute(delete(AnomalyRule).where(AnomalyRule.id == rule.id))
        record_audit(
            db,
            ctx,
            "anomaly.rule.delete",
            target_type="anomaly_rule",
            target_id=rule_id,
            details={"name": name},
            organization_id=org_id,
        )
        return {"deleted": True, "rule_id": rule_id, "name": name}


def list_rules(ctx: AuthContext) -> list[dict[str, Any]]:
    """All alert rules of the organization with their event counts."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        rows = db.execute(
            select(AnomalyRule, Dataset.name)
            .join(Dataset, Dataset.id == AnomalyRule.dataset_id)
            .where(AnomalyRule.organization_id == org_id)
            .order_by(AnomalyRule.id)
        ).all()
        counts = _event_counts(db, org_id, [rule.id for rule, _ in rows])
        return [_rule_dict(rule, dataset_name, counts.get(rule.id, 0)) for rule, dataset_name in rows]


# ---------------------------------------------------------------------------
# Running rules
# ---------------------------------------------------------------------------


def _missing_invalid_percentages(
    ctx: AuthContext,
    dataset_id: int,
    ref: versioning_service.VersionRef,
    columns: list[str],
    expected_types: Mapping[str, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Exact percentage of missing-or-invalid cells and of the invalid part alone, per column.

    Counted batch by batch. A null cell is missing; a non-null text value that does not parse
    as the column's number or date is invalid (see the module docstring). ``expected_types``
    maps a column to the logical type it had in the earlier version, when that was numeric or
    a date.
    """
    text_columns = {c["name"] for c in ref.schema if c["logical_type"] == "text"}
    nulls = dict.fromkeys(columns, 0)
    parsed = {c: (0, 0, 0) for c in columns if c in text_columns}
    for batch in versioning_service.iter_frame_batches(
        ctx, dataset_id, version_id=ref.version_id, columns=columns, feature=Feature.ANOMALY_DETECTION
    ):
        for column in columns:
            nulls[column] += int(batch[column].isna().sum())
            if column in parsed:
                nonnull, numeric_like, date_like = text_parse_counts(batch[column])
                total = parsed[column]
                parsed[column] = (total[0] + nonnull, total[1] + numeric_like, total[2] + date_like)
    rows = ref.rows
    invalid = {
        column: count_invalid_text(*parsed[column], expected_types.get(column)) if column in parsed else 0
        for column in columns
    }
    combined = {c: ((nulls[c] + invalid[c]) * 100.0 / rows if rows else 0.0) for c in columns}
    invalid_only = {c: (invalid[c] * 100.0 / rows if rows else 0.0) for c in columns}
    return combined, invalid_only


def _previous_version_id(ctx: AuthContext, dataset_id: int, version_number: int) -> int | None:
    with session_scope() as db:
        return db.execute(
            select(DatasetVersion.id)
            .where(
                DatasetVersion.organization_id == ctx.organization_id,
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version_number < version_number,
                DatasetVersion.is_purged.is_(False),
            )
            .order_by(DatasetVersion.version_number.desc())
            .limit(1)
        ).scalar_one_or_none()


def _missing_findings(
    ctx: AuthContext, rule: dict[str, Any], ref: versioning_service.VersionRef
) -> list[Finding]:
    previous_id = _previous_version_id(ctx, rule["dataset_id"], ref.version_number)
    if previous_id is None:
        return []
    previous_ref = versioning_service.get_version_ref(
        ctx, rule["dataset_id"], previous_id, feature=Feature.ANOMALY_DETECTION
    )
    shared = [c["name"] for c in ref.schema if c["name"] in {p["name"] for p in previous_ref.schema}]
    if rule["metric_column"]:
        shared = [c for c in shared if c == rule["metric_column"]]
    if not shared:
        return []
    previous_types = {c["name"]: c["logical_type"] for c in previous_ref.schema}
    expected_types = {
        name: previous_types[name]
        for name in shared
        if previous_types[name] in ("integer", "float", "datetime")
    }
    current, invalid_current = _missing_invalid_percentages(
        ctx, rule["dataset_id"], ref, shared, expected_types
    )
    before, invalid_before = _missing_invalid_percentages(
        ctx, rule["dataset_id"], previous_ref, shared, {}
    )
    return detect_missing_increase(
        current,
        before,
        rule["threshold"],
        current_label=f"version {ref.version_number}",
        previous_label=f"version {previous_ref.version_number}",
        invalid_current=invalid_current,
        invalid_previous=invalid_before,
    )


def _collect_findings(
    ctx: AuthContext, rule: dict[str, Any]
) -> tuple[list[Finding], versioning_service.VersionRef]:
    """Load only the needed columns of the current version and run the rule's detector."""
    method, dataset_id = rule["method"], rule["dataset_id"]
    if method == "missing_increase":
        ref = versioning_service.get_version_ref(ctx, dataset_id, feature=Feature.ANOMALY_DETECTION)
        return _missing_findings(ctx, rule, ref), ref
    metric = rule["metric_column"]
    columns = [metric] if method in VALUE_METHODS else [rule["date_column"], metric]
    frame, ref = versioning_service.load_frame_with_ref(
        ctx, dataset_id, columns=columns, feature=Feature.ANOMALY_DETECTION
    )
    if method == "iqr":
        return detect_iqr(
            frame[metric], rule["threshold"], column=metric, period=f"version {ref.version_number}"
        ), ref
    if method == "zscore":
        return detect_zscore(
            frame[metric], rule["threshold"], column=metric, period=f"version {ref.version_number}"
        ), ref
    series = aggregate_by_period(
        frame,
        date_column=rule["date_column"],
        metric_column=metric,
        aggregation=rule["aggregation"],
        frequency=rule["frequency"],
    )
    if method == "baseline_deviation":
        return detect_baseline_deviation(
            series, rule["threshold"], column=metric, frequency=rule["frequency"]
        ), ref
    return (
        detect_period_change(
            series,
            rule["threshold"],
            column=metric,
            frequency=rule["frequency"],
            drops_only=method == "metric_drop",
        ),
        ref,
    )


def _event_details(
    rule: dict[str, Any], finding: Finding, ref: versioning_service.VersionRef
) -> dict[str, Any]:
    details = {
        "method": finding.method,
        "threshold": finding.threshold,
        "column": finding.column,
        "period": finding.period,
        "direction": finding.direction,
        "observed": finding.observed,
        "expected": finding.expected,
        "lower_bound": finding.lower_bound,
        "upper_bound": finding.upper_bound,
        "dataset_version": ref.version_number,
        "aggregation": rule["aggregation"] if rule["method"] in PERIOD_METHODS else None,
        "note": "A statistical flag for review, not a confirmed problem.",
    }
    details.update(finding.details)
    return _jsonable(details)


def _existing_fingerprints(db: Session, org_id: int, rule_id: int, fingerprints: list[str]) -> set[str]:
    found: set[str] = set()
    for chunk in _chunks(fingerprints):
        found.update(
            db.execute(
                select(AnomalyEvent.fingerprint).where(
                    AnomalyEvent.organization_id == org_id,
                    AnomalyEvent.rule_id == rule_id,
                    AnomalyEvent.fingerprint.in_(chunk),
                )
            ).scalars()
        )
    return found


def _cooldown_active(db: Session, org_id: int, rule: AnomalyRule, now: datetime) -> bool:
    if rule.cooldown_minutes <= 0:
        return False
    since = now - timedelta(minutes=rule.cooldown_minutes)
    return (
        db.execute(
            select(NotificationLog.id)
            .where(
                NotificationLog.organization_id == org_id,
                NotificationLog.rule_id == rule.id,
                NotificationLog.status == "sent",
                NotificationLog.created_at > since,
            )
            .limit(1)
        ).first()
        is not None
    )


def _store_events(
    ctx: AuthContext,
    org_id: int,
    rule: dict[str, Any],
    ref: versioning_service.VersionRef,
    findings: list[Finding],
    now: datetime,
) -> dict[str, Any]:
    """Store new events; a concurrent run that stored the same fingerprint wins.

    ``anomaly_events`` has a unique (rule_id, fingerprint) constraint, so two
    runs of one rule at the same moment can never create duplicate events. The
    losing run's transaction rolls back before any e-mail is sent.
    """
    try:
        return _insert_events(ctx, org_id, rule, ref, findings, now)
    except IntegrityError as exc:
        raise ConflictError(
            "Another check of this rule finished at the same moment. No duplicate alerts were created; "
            "run the check again to see the latest results."
        ) from exc


def _insert_events(
    ctx: AuthContext,
    org_id: int,
    rule: dict[str, Any],
    ref: versioning_service.VersionRef,
    findings: list[Finding],
    now: datetime,
) -> dict[str, Any]:
    """Insert new events (skipping duplicates), stamp the rule and record usage and audit."""
    with session_scope() as db:
        authorize(db, ctx, roles=(Role.ORG_ADMIN,), feature=Feature.ANOMALY_DETECTION)
        row = _get_rule(db, org_id, rule["id"])
        pairs = [(f, make_fingerprint(row.id, f.method, f.column, f.period, f.direction)) for f in findings]
        known = _existing_fingerprints(db, org_id, row.id, [fingerprint for _, fingerprint in pairs])
        cooldown = _cooldown_active(db, org_id, row, now)
        status, reason = ("suppressed", "cooldown") if cooldown else ("new", None)
        email_enabled = subscriptions.org_has_feature(db, org_id, Feature.EMAIL_ALERTS)
        seen: set[str] = set()
        events: list[dict[str, Any]] = []
        duplicates = 0
        for finding, fingerprint in pairs:
            if fingerprint in known or fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            details = _event_details(rule, finding, ref)
            event = AnomalyEvent(
                organization_id=org_id,
                rule_id=row.id,
                dataset_id=row.dataset_id,
                version_id=ref.version_id,
                fingerprint=fingerprint,
                severity=row.severity,
                title=finding.title[:300],
                details_json=json.dumps(details),
                observed_value=_finite_or_none(finding.observed),
                expected_value=_finite_or_none(finding.expected),
                status=status,
                suppressed_reason=reason,
                detected_at=now,
            )
            db.add(event)
            db.flush()
            events.append(
                {
                    "id": event.id,
                    "title": event.title,
                    "severity": event.severity,
                    "status": status,
                    "period": finding.period,
                    "observed": event.observed_value,
                    "expected": event.expected_value,
                    "details": details,
                }
            )
        row.last_checked_at = now
        record_usage(
            db,
            org_id,
            ctx.user_id,
            UsageMetric.ANOMALY_CHECK,
            1.0,
            {"rule_id": row.id, "findings": len(findings)},
        )
        record_audit(
            db,
            ctx,
            "anomaly.run",
            target_type="anomaly_rule",
            target_id=row.id,
            details={
                "findings": len(findings),
                "new_events": len(events),
                "duplicate_suppressed": duplicates,
                "cooldown_suppressed": len(events) if cooldown else 0,
            },
            organization_id=org_id,
        )
        recipients = json.loads(row.recipients_json or "[]")
    return {
        "events": events,
        "duplicates": duplicates,
        "cooldown": cooldown,
        "recipients": recipients if email_enabled else [],
    }


def _compose_digest(
    rule: dict[str, Any], dataset_name: str, version_number: int, events: list[dict[str, Any]], now: datetime
) -> tuple[str, str, str]:
    """Build the (subject, text, html) of the digest e-mail; untrusted text is cleaned or escaped."""
    app_name = get_settings().app_name
    rule_name = _clean_text(rule["name"], 100)
    noun = "possible anomaly" if len(events) == 1 else "possible anomalies"
    subject = f"[{rule['severity'].upper()}] {app_name}: {len(events)} {noun} for rule {rule_name}"[:300]
    shown = events[:MAX_DIGEST_EVENTS]
    header = [
        f"Rule: {rule_name} (severity {rule['severity']})",
        f"Dataset: {_clean_text(dataset_name, 100)} (version {version_number})",
        f"Checked at: {now:%Y-%m-%d %H:%M} UTC",
        "These are statistical flags for review, not confirmed problems.",
    ]
    lines = []
    for number, event in enumerate(shown, start=1):
        lines.append(f"{number}. {event['title']}")
        lines.append(
            f"   Observed: {_fmt_optional(event['observed'])}   Expected: {_fmt_optional(event['expected'])}"
        )
    if len(events) > len(shown):
        lines.append(f"... and {len(events) - len(shown)} more. Open the alert history to see all of them.")
    text = "\n".join(header + [""] + lines)
    items = "".join(
        f"<li>{escape(event['title'])}<br>Observed: {escape(_fmt_optional(event['observed']))}, "
        f"expected: {escape(_fmt_optional(event['expected']))}</li>"
        for event in shown
    )
    more = f"<p>... and {len(events) - len(shown)} more.</p>" if len(events) > len(shown) else ""
    intro = "<br>".join(escape(line) for line in header)
    html = f"<p>{intro}</p><ol>{items}</ol>{more}"
    return subject, text, html


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else _fmt(value)


def _send_digest(recipients: list[str], subject: str, text: str, html: str) -> list[dict[str, Any]]:
    """Send the digest to each recipient separately; delivery problems become ``failed`` outcomes."""
    outcomes = []
    for recipient in recipients:
        try:
            result = email_service.send_email(to=[recipient], subject=subject, body_text=text, body_html=html)
        except AppError as exc:
            result = {"status": "failed", "error": exc.user_message}
        outcomes.append({"recipient": recipient, "status": result["status"], "error": result["error"]})
    return outcomes


def _record_notifications(
    org_id: int,
    rule_id: int,
    event_ids: list[int],
    subject: str,
    outcomes: list[dict[str, Any]],
    now: datetime,
) -> None:
    with session_scope() as db:
        for event_id in event_ids:
            for outcome in outcomes:
                db.add(
                    NotificationLog(
                        organization_id=org_id,
                        event_id=event_id,
                        rule_id=rule_id,
                        channel="email",
                        recipient=outcome["recipient"],
                        subject=subject,
                        status=outcome["status"],
                        error_message=outcome["error"],
                        created_at=now,
                    )
                )
        if any(outcome["status"] == "sent" for outcome in outcomes):
            for chunk in _chunks(event_ids):
                db.execute(
                    update(AnomalyEvent)
                    .where(AnomalyEvent.organization_id == org_id, AnomalyEvent.id.in_(chunk))
                    .values(status="notified")
                )


def run_rule(ctx: AuthContext, rule_id: int, *, now: datetime | None = None) -> dict[str, Any]:
    """Run one rule now (also for inactive rules) and notify the recipients of new events.

    Returns ``{"findings", "new_events", "duplicate_suppressed", "cooldown_suppressed",
    "notifications"}`` (the first four are counts; ``notifications`` lists
    ``{"recipient", "status", "error"}`` per recipient) plus ``rule_id``, ``version_number`` and
    ``events`` (the new events). Database sessions are never held while data is loaded or e-mail
    is sent. ``now`` exists for tests and the scheduler.
    """
    moment = now or timeutil.utcnow()
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        row = _get_rule(db, org_id, rule_id)
        rule = _rule_view(db, org_id, row)
    findings, ref = _collect_findings(ctx, rule)
    stored = _store_events(ctx, org_id, rule, ref, findings, moment)
    events = stored["events"]
    notifications: list[dict[str, Any]] = []
    if events and not stored["cooldown"] and stored["recipients"]:
        subject, text, html = _compose_digest(rule, ref.dataset_name, ref.version_number, events, moment)
        notifications = _send_digest(stored["recipients"], subject, text, html)
        _record_notifications(org_id, rule["id"], [e["id"] for e in events], subject, notifications, moment)
        if any(n["status"] == "sent" for n in notifications):
            for event in events:
                event["status"] = "notified"
    return {
        "rule_id": rule["id"],
        "version_number": ref.version_number,
        "findings": len(findings),
        "new_events": len(events),
        "duplicate_suppressed": stored["duplicates"],
        "cooldown_suppressed": len(events) if stored["cooldown"] else 0,
        "notifications": notifications,
        "events": events,
    }


def run_all_rules(ctx: AuthContext) -> list[dict[str, Any]]:
    """Manual "check all now": run every active rule; one failing rule does not stop the others.

    Each item is the ``run_rule`` result with ``rule_name`` and ``status`` "ok", or
    ``{"rule_id", "rule_name", "status": "failed", "error"}`` with a user-safe message.
    """
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        rules = db.execute(
            select(AnomalyRule.id, AnomalyRule.name)
            .where(AnomalyRule.organization_id == org_id, AnomalyRule.is_active.is_(True))
            .order_by(AnomalyRule.id)
        ).all()
    results: list[dict[str, Any]] = []
    for rule_id, name in rules:
        try:
            result = run_rule(ctx, rule_id)
            results.append({**result, "rule_name": name, "status": "ok"})
        except AppError as exc:
            results.append(
                {"rule_id": rule_id, "rule_name": name, "status": "failed", "error": exc.user_message}
            )
        except Exception as exc:  # noqa: BLE001 - one broken rule must not stop the others
            reference = record_error("anomaly.run_all_rules", exc, ctx=ctx)
            results.append(
                {
                    "rule_id": rule_id,
                    "rule_name": name,
                    "status": "failed",
                    "error": f"The check failed unexpectedly (reference {reference}).",
                }
            )
    return results


def _due_rules(now: datetime) -> tuple[list[tuple[int, int]], int, int]:
    """Return ``([(org_id, rule_id)], organizations_checked, rules_not_due)`` for scheduled rules."""
    due: list[tuple[int, int]] = []
    not_due = 0
    with session_scope() as db:
        org_ids = (
            db.execute(
                select(Organization.id)
                .where(Organization.status == OrgStatus.ACTIVE.value)
                .order_by(Organization.id)
            )
            .scalars()
            .all()
        )
        checked = 0
        for org_id in org_ids:
            if not subscriptions.org_has_feature(db, org_id, Feature.ANOMALY_DETECTION):
                continue
            checked += 1
            rules = db.execute(
                select(AnomalyRule.id, AnomalyRule.schedule, AnomalyRule.last_checked_at)
                .where(
                    AnomalyRule.organization_id == org_id,
                    AnomalyRule.is_active.is_(True),
                    AnomalyRule.schedule != "manual",
                )
                .order_by(AnomalyRule.id)
            ).all()
            for rule_id, schedule, last_checked in rules:
                if rule_is_due(schedule, last_checked, now):
                    due.append((org_id, rule_id))
                else:
                    not_due += 1
    return due, checked, not_due


def run_scheduled_checks(*, now: datetime | None = None) -> dict[str, int]:
    """Worker entry point: run every due scheduled rule of every eligible organization.

    Organizations must be active and their plan must include anomaly detection. Rules run with
    ``AuthContext.system``; a failing rule is logged and counted but never stops the others.
    Returns counts: organizations_checked, rules_due, rules_run, rules_failed, rules_not_due,
    new_events, duplicate_suppressed, cooldown_suppressed and notifications_sent.
    """
    moment = now or timeutil.utcnow()
    due, checked, not_due = _due_rules(moment)
    totals = {
        "organizations_checked": checked,
        "rules_due": len(due),
        "rules_run": 0,
        "rules_failed": 0,
        "rules_not_due": not_due,
        "new_events": 0,
        "duplicate_suppressed": 0,
        "cooldown_suppressed": 0,
        "notifications_sent": 0,
    }
    for org_id, rule_id in due:
        ctx = AuthContext.system(org_id)
        try:
            result = run_rule(ctx, rule_id, now=moment)
        except AppError as exc:
            logger.warning("Scheduled check of rule %s failed: %s", rule_id, exc.user_message)
            totals["rules_failed"] += 1
        except Exception as exc:  # noqa: BLE001 - keep the worker going
            record_error("anomaly.scheduled_check", exc, ctx=ctx)
            totals["rules_failed"] += 1
        else:
            totals["rules_run"] += 1
            totals["new_events"] += result["new_events"]
            totals["duplicate_suppressed"] += result["duplicate_suppressed"]
            totals["cooldown_suppressed"] += result["cooldown_suppressed"]
            totals["notifications_sent"] += sum(1 for n in result["notifications"] if n["status"] == "sent")
    return totals


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def _event_dict(event: AnomalyEvent, rule_name: str) -> dict[str, Any]:
    return {
        "id": event.id,
        "rule_id": event.rule_id,
        "rule_name": rule_name,
        "dataset_id": event.dataset_id,
        "version_id": event.version_id,
        "severity": event.severity,
        "title": event.title,
        "details": json.loads(event.details_json or "{}"),
        "observed_value": event.observed_value,
        "expected_value": event.expected_value,
        "status": event.status,
        "suppressed_reason": event.suppressed_reason,
        "detected_at": event.detected_at,
        "acknowledged_by_id": event.acknowledged_by_id,
        "acknowledged_at": event.acknowledged_at,
    }


def list_events(
    ctx: AuthContext, *, rule_id: int | None = None, status: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    """Anomaly history, newest first, optionally for one rule and/or one status."""
    if status is not None:
        _validate_choice(status, EVENT_STATUSES, "status")
    limit = max(1, min(int(limit), 1000))
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        query = (
            select(AnomalyEvent, AnomalyRule.name)
            .join(AnomalyRule, AnomalyRule.id == AnomalyEvent.rule_id)
            .where(AnomalyEvent.organization_id == org_id)
        )
        if rule_id is not None:
            query = query.where(AnomalyEvent.rule_id == _get_rule(db, org_id, rule_id).id)
        if status is not None:
            query = query.where(AnomalyEvent.status == status)
        rows = db.execute(
            query.order_by(AnomalyEvent.detected_at.desc(), AnomalyEvent.id.desc()).limit(limit)
        ).all()
        return [_event_dict(event, name) for event, name in rows]


def acknowledge_event(ctx: AuthContext, event_id: int) -> dict[str, Any]:
    """Mark an event as reviewed (idempotent)."""
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        event = db.get(AnomalyEvent, _check_id(event_id, "event"))
        if event is None or event.organization_id != org_id:
            raise NotFoundError("Alert event not found.")
        if event.status != "acknowledged":
            event.status = "acknowledged"
            event.acknowledged_by_id = ctx.user_id
            event.acknowledged_at = timeutil.utcnow()
            record_audit(
                db,
                ctx,
                "anomaly.event.acknowledge",
                target_type="anomaly_event",
                target_id=event.id,
                organization_id=org_id,
            )
        db.flush()
        rule = db.get(AnomalyRule, event.rule_id)
        return _event_dict(event, rule.name if rule else "")


def list_notifications(ctx: AuthContext, *, limit: int = 200) -> list[dict[str, Any]]:
    """E-mail notification log of the organization, newest first."""
    limit = max(1, min(int(limit), 1000))
    with session_scope() as db:
        org_id = _authorize_admin(db, ctx)
        rows = db.execute(
            select(NotificationLog, AnomalyRule.name)
            .outerjoin(AnomalyRule, AnomalyRule.id == NotificationLog.rule_id)
            .where(NotificationLog.organization_id == org_id)
            .order_by(NotificationLog.created_at.desc(), NotificationLog.id.desc())
            .limit(limit)
        ).all()
        return [
            {
                "id": log.id,
                "event_id": log.event_id,
                "rule_id": log.rule_id,
                "rule_name": name,
                "channel": log.channel,
                "recipient": log.recipient,
                "subject": log.subject,
                "status": log.status,
                "error_message": log.error_message,
                "created_at": log.created_at,
            }
            for log, name in rows
        ]
