"""Missing-value analysis and treatment (specification section 8).

A cell is missing when it is null/NaN/NaT or, in text columns, when it is an
empty string, whitespace only, or equal (trimmed, case-insensitive) to one of
the organization's configured placeholder tokens such as "N/A". Zero, False
and categories such as "Unknown" are ordinary values unless an administrator
lists them as placeholders.

Analysis streams the current version in bounded batches. Previews load only
the planned columns; applying loads the whole version (memory-checked) and
publishes a new version, so every change can be undone by restoring.

Treatment order (identical in preview and apply):

1. Fill statistics (mean, median, mode) are computed exactly from the whole
   column as it is before any rows are dropped, or only from the rows selected
   by ``fit_filter`` (for example the training split) to prevent leakage.
2. Rows that are missing a value in any ``drop_rows`` column are removed.
3. Fills are applied to the remaining rows; forward/backward fill follows the
   remaining row order.
4. ``drop_column`` columns are removed (only with ``confirm_drop_columns``).

Row numbers are zero-based positions in the analysed version.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from auth.authorization import AuthContext
from core.constants import Feature, LogicalType, Permission
from core.errors import ValidationError
from database.connection import session_scope
from modules import org_settings
from services import versioning_service
from services.duplicate_manager import column_name_tokens, looks_like_identifier_name
from services.frame_utils import json_safe, records
from services.resource_manager import OperationGuard
from services.type_inference import BOOLEAN_FALSE, BOOLEAN_TRUE
from services.versioning_service import VersionRef

NUMERIC = "numeric"
TEXT = "text"
BOOLEAN = "boolean"
DATETIME = "datetime"

STRATEGIES: dict[str, tuple[str, ...]] = {
    NUMERIC: ("mean", "median", "constant", "ffill", "bfill", "leave", "drop_rows", "drop_column"),
    TEXT: ("mode", "constant", "placeholder", "ffill", "bfill", "leave", "drop_rows", "drop_column"),
    BOOLEAN: ("mode", "constant", "leave", "drop_rows", "drop_column"),
    DATETIME: ("date_value", "ffill", "bfill", "leave", "drop_rows", "drop_column"),
}
STATISTIC_STRATEGIES = frozenset({"mean", "median", "mode"})
VALUE_STRATEGIES = frozenset({"constant", "date_value"})
ORDER_STRATEGIES = frozenset({"ffill", "bfill"})
IDENTIFIER_ALLOWED = frozenset({"constant", "leave", "drop_rows", "drop_column"})

HIGH_MISSING_PCT = 50.0
SMALL_DATASET_ROWS = 30
IDENTIFIER_DISTINCT_RATIO = 0.9
CATEGORICAL_MAX_DISTINCT = 50
CATEGORICAL_MAX_RATIO = 0.5
MIXED_MIN_SHARE = 0.2
MAX_SAMPLE_ROWS = 500
MAX_CONSTANT_LENGTH = 500

SENSITIVE_TOKENS = frozenset(
    {
        "ssn",
        "password",
        "passwd",
        "salary",
        "dob",
        "birth",
        "birthdate",
        "birthday",
        "tax",
        "passport",
        "iban",
        "card",
    }
)
TARGET_NAMES = frozenset({"target", "label", "class", "outcome", "churn", "y", "is_fraud"})
_NUMBER_LIKE = re.compile(r"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)$")


# ---------------------------------------------------------------------------
# Shared rules
# ---------------------------------------------------------------------------


def type_group(logical_type: str) -> str:
    """Strategy group of a logical type: numeric, text, boolean or datetime."""
    if logical_type in (LogicalType.INTEGER.value, LogicalType.FLOAT.value):
        return NUMERIC
    if logical_type == LogicalType.BOOLEAN.value:
        return BOOLEAN
    if logical_type == LogicalType.DATETIME.value:
        return DATETIME
    return TEXT


def is_sensitive_name(name: str) -> bool:
    """Whole-token match on sensitive words (ssn, password, salary, dob, birth, tax, passport, iban, card)."""
    return any(token in SENSITIVE_TOKENS for token in column_name_tokens(name))


def is_target_name(name: str) -> bool:
    """True for typical machine-learning target/label column names (whole name match)."""
    return "_".join(column_name_tokens(name)) in TARGET_NAMES


def placeholder_set(tokens: list[str] | tuple[str, ...]) -> frozenset[str]:
    return frozenset(str(t).strip().casefold() for t in tokens if str(t).strip())


def missing_mask(series: pd.Series, logical_type: str, placeholders: frozenset[str]) -> pd.Series:
    """True where a cell counts as missing (see module docstring)."""
    mask = series.isna()
    if type_group(logical_type) == TEXT:
        stripped = series.astype("string").str.strip()
        mask = mask | stripped.eq("") | stripped.str.casefold().isin(placeholders)
    return mask.fillna(True).astype(bool)


def _types(ref: VersionRef) -> dict[str, str]:
    return {column["name"]: column["logical_type"] for column in ref.schema}


def _org_settings(organization_id: int) -> dict[str, Any]:
    with session_scope() as db:
        return org_settings.get_org_settings(db, organization_id)


def _distinct_hashes(values: pd.Series) -> np.ndarray:
    if values.empty:
        return np.empty(0, dtype=np.uint64)
    return np.unique(pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class _ColumnTally:
    """Streaming per-column counts (distinct values tracked as 64-bit hashes)."""

    name: str
    logical_type: str
    missing: int = 0
    non_missing: int = 0
    number_like: int = 0
    distinct: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))

    def update(self, series: pd.Series, mask: pd.Series) -> None:
        present = series[~mask]
        self.missing += int(mask.sum())
        self.non_missing += len(present)
        if type_group(self.logical_type) == TEXT:
            self.number_like += int(
                present.astype("string").str.strip().str.fullmatch(_NUMBER_LIKE).fillna(False).sum()
            )
        self.distinct = np.union1d(self.distinct, _distinct_hashes(present))


def _semantic(tally: _ColumnTally) -> str:
    group = type_group(tally.logical_type)
    distinct = len(tally.distinct)
    if group in (NUMERIC, TEXT) and tally.non_missing and looks_like_identifier_name(tally.name):
        if distinct >= IDENTIFIER_DISTINCT_RATIO * tally.non_missing:
            return "identifier"
    if group != TEXT:
        return group
    if (
        tally.non_missing
        and 0 < tally.number_like < tally.non_missing
        and tally.number_like >= MIXED_MIN_SHARE * tally.non_missing
    ):
        return "mixed"
    if distinct <= CATEGORICAL_MAX_DISTINCT or distinct <= CATEGORICAL_MAX_RATIO * tally.non_missing:
        return "categorical"
    return "free_text"


def _flags(tally: _ColumnTally, rows: int, semantic: str) -> dict[str, bool]:
    pct = 100.0 * tally.missing / rows if rows else 0.0
    return {
        "identifier": semantic == "identifier",
        "sensitive": is_sensitive_name(tally.name),
        "target_label": is_target_name(tally.name),
        "all_null": rows > 0 and tally.non_missing == 0,
        "high_missing": pct >= HIGH_MISSING_PCT,
        "single_non_null": tally.non_missing == 1,
        "small_dataset": rows < SMALL_DATASET_ROWS,
        "mixed_types": semantic == "mixed",
    }


_LEAVE_REASONS = (
    (
        "all_null",
        "Every value is missing, so there is nothing to learn a fill value from. "
        "Dropping a column is never automatic; choose drop_column with confirmation if it is not needed.",
    ),
    (
        "identifier",
        "This looks like an identifier. Identifiers are never altered silently; "
        "leave them, drop the rows, or set an explicit constant.",
    ),
    (
        "sensitive",
        "This looks like a sensitive field. It is only filled when you explicitly allow changes to sensitive columns.",
    ),
    (
        "target_label",
        "This looks like a target or label column. Filling labels invents outcomes, "
        "so it needs your explicit permission.",
    ),
    (
        "high_missing",
        "Half or more of the values are missing, so any fill would mostly be invented data. "
        "Dropping a column is never automatic; review it and choose explicitly.",
    ),
    ("single_non_null", "Only one value is present, so filling would turn the column into a constant."),
)


def _recommend(tally: _ColumnTally, flags: dict[str, bool], semantic: str) -> tuple[str, str]:
    """Recommended strategy and the reason for it (only automatic-safe fills are recommended)."""
    if tally.missing == 0:
        return "leave", "No missing values."
    for flag, reason in _LEAVE_REASONS:
        if flags[flag]:
            return "leave", reason
    group = type_group(tally.logical_type)
    small = (
        " With fewer than 30 rows the statistic is less reliable; review the preview."
        if flags["small_dataset"]
        else ""
    )
    if group == DATETIME:
        return (
            "leave",
            "Dates are never invented. Leave them, drop the rows, or enter an explicit replacement date.",
        )
    if group == NUMERIC:
        integer = (
            " Integer columns stay whole numbers (the median is rounded half to even)."
            if tally.logical_type == LogicalType.INTEGER.value
            else ""
        )
        return (
            "median",
            "The median is robust to outliers and is computed exactly over the whole column."
            + integer
            + small,
        )
    if group == BOOLEAN:
        return "mode", "The most frequent value is the safest fill for a yes/no column." + small
    if semantic == "mixed":
        return "leave", "The column mixes numbers and text. Fix or convert the values before filling them."
    if semantic == "categorical":
        return "mode", "The most frequent category is a reasonable fill for a low-cardinality column." + small
    return "leave", "Free text cannot be inferred. Leave it, or use the configured placeholder explicitly."


def _analysis_tallies(
    ctx: AuthContext, ref: VersionRef, placeholders: frozenset[str]
) -> tuple[list[_ColumnTally], int]:
    types = _types(ref)
    tallies = {name: _ColumnTally(name, logical) for name, logical in types.items()}
    rows_with_missing = 0
    for batch in versioning_service.iter_frame_batches(
        ctx, ref.dataset_id, version_id=ref.version_id, feature=Feature.MISSING_VALUES
    ):
        any_missing = np.zeros(len(batch), dtype=bool)
        for name, logical in types.items():
            mask = missing_mask(batch[name], logical, placeholders)
            tallies[name].update(batch[name], mask)
            any_missing |= mask.to_numpy(dtype=bool)
        rows_with_missing += int(any_missing.sum())
    return list(tallies.values()), rows_with_missing


def analyze_missing(ctx: AuthContext, dataset_id: int) -> dict[str, Any]:
    """Per-column missing counts, flags, recommended strategy, reason and expected impact."""
    ref = versioning_service.get_version_ref(ctx, dataset_id, feature=Feature.MISSING_VALUES)
    settings = _org_settings(ref.organization_id)
    placeholders = placeholder_set(settings[org_settings.MISSING_PLACEHOLDERS])
    tallies, rows_with_missing = _analysis_tallies(ctx, ref, placeholders)
    columns = []
    for tally in tallies:
        semantic = _semantic(tally)
        flags = _flags(tally, ref.rows, semantic)
        strategy, reason = _recommend(tally, flags, semantic)
        columns.append(
            {
                "name": tally.name,
                "logical_type": tally.logical_type,
                "semantic": semantic,
                "missing_count": tally.missing,
                "missing_pct": round(100.0 * tally.missing / ref.rows, 4) if ref.rows else 0.0,
                "distinct_values": int(len(tally.distinct)),
                "recommended_strategy": strategy,
                "reason": reason,
                "expected_impact": {
                    "cells_changed": tally.missing if strategy != "leave" else 0,
                    "rows_removed": 0,
                },
                "flags": flags,
            }
        )
    total_missing = sum(t.missing for t in tallies)
    cells = ref.rows * len(tallies)
    return {
        "rows": ref.rows,
        "columns_count": len(tallies),
        "total_missing_cells": total_missing,
        "missing_pct": round(100.0 * total_missing / cells, 4) if cells else 0.0,
        "rows_with_missing": rows_with_missing,
        "small_dataset": ref.rows < SMALL_DATASET_ROWS,
        "placeholders": list(settings[org_settings.MISSING_PLACEHOLDERS]),
        "columns": columns,
        "notes": [
            "Empty, whitespace-only and configured placeholder text values count as missing.",
            "Zero, False and categories such as 'Unknown' are never treated as missing unless configured.",
            "Dropping columns is never recommended automatically.",
        ],
    }


def recommend_plan(ctx: AuthContext, dataset_id: int) -> list[dict[str, Any]]:
    """Automatic-safe plan entries (median/mode) for columns that have missing values."""
    analysis = analyze_missing(ctx, dataset_id)
    return [
        {"column": c["name"], "strategy": c["recommended_strategy"], "value": None}
        for c in analysis["columns"]
        if c["missing_count"] > 0 and c["recommended_strategy"] != "leave"
    ]


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreatmentOptions:
    """Explicit permissions and context for a treatment (see preview_missing_treatment)."""

    row_order_meaningful: bool = False
    fit_filter: dict[str, Any] | None = None
    allow_sensitive: bool = False
    allow_target: bool = False
    confirm_drop_columns: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_order_meaningful": self.row_order_meaningful,
            "fit_filter": self.fit_filter,
            "allow_sensitive": self.allow_sensitive,
            "allow_target": self.allow_target,
            "confirm_drop_columns": self.confirm_drop_columns,
        }


@dataclass(frozen=True)
class ColumnPlan:
    column: str
    strategy: str
    logical_type: str
    value: Any = None


def _parse_integer(value: Any, column: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"The value for '{column}' must be a whole number.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*[+-]?\d+\s*", value):
        return int(value)
    raise ValidationError(f"The value for '{column}' must be a whole number.")


def _parse_float(value: Any, column: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"The value for '{column}' must be a number.")
    try:
        number = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        raise ValidationError(f"The value for '{column}' must be a number.") from None
    if not math.isfinite(number):
        raise ValidationError(f"The value for '{column}' must be a finite number.")
    return number


def _parse_boolean(value: Any, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().casefold()
    if token in BOOLEAN_TRUE:
        return True
    if token in BOOLEAN_FALSE:
        return False
    raise ValidationError(f"The value for '{column}' must be true or false.")


def _parse_text(value: Any, column: str, placeholders: frozenset[str]) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_CONSTANT_LENGTH:
        raise ValidationError(
            f"Enter a replacement text of 1 to {MAX_CONSTANT_LENGTH} characters for '{column}'."
        )
    if value.strip().casefold() in placeholders:
        raise ValidationError(
            f"'{value.strip()}' is itself treated as missing, so it cannot be used to fill '{column}'."
        )
    return value


def _parse_date(value: Any, column: str) -> pd.Timestamp:
    if not isinstance(value, (str, datetime, date, pd.Timestamp)) or (
        isinstance(value, str) and not value.strip()
    ):
        raise ValidationError(f"Enter a replacement date for '{column}'.")
    try:
        stamp = pd.Timestamp(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError, OverflowError):
        raise ValidationError(f"'{str(value)[:40]}' is not a valid date for '{column}'.") from None
    if stamp is pd.NaT or pd.isna(stamp):
        raise ValidationError(f"'{str(value)[:40]}' is not a valid date for '{column}'.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _parse_value(plan_value: Any, column: str, logical: str, placeholders: frozenset[str]) -> Any:
    if plan_value is None:
        raise ValidationError(f"Enter the replacement value for '{column}'.")
    if logical == LogicalType.INTEGER.value:
        return _parse_integer(plan_value, column)
    if logical == LogicalType.FLOAT.value:
        return _parse_float(plan_value, column)
    if logical == LogicalType.BOOLEAN.value:
        return _parse_boolean(plan_value, column)
    if logical == LogicalType.DATETIME.value:
        return _parse_date(plan_value, column)
    return _parse_text(plan_value, column, placeholders)


def _validate_fit_filter(fit_filter: Any, types: dict[str, str]) -> None:
    if fit_filter is None:
        return
    if not isinstance(fit_filter, dict) or set(fit_filter) != {"column", "values"}:
        raise ValidationError(
            "The fit filter needs a 'column' and a list of 'values', for example the training split."
        )
    if fit_filter["column"] not in types:
        raise ValidationError(f"Unknown column: {fit_filter['column']}")
    values = fit_filter["values"]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValidationError("List at least one value of the split column to fit statistics on.")


def _check_permissions(plan: ColumnPlan, options: TreatmentOptions) -> None:
    if plan.strategy == "leave":
        return
    if is_sensitive_name(plan.column) and not options.allow_sensitive:
        raise ValidationError(
            f"'{plan.column}' looks like a sensitive field. Allow changes to sensitive columns explicitly to treat it."
        )
    if is_target_name(plan.column) and not options.allow_target:
        raise ValidationError(
            f"'{plan.column}' looks like a target or label column. Filling or dropping labels changes outcomes, "
            "so allow changes to target columns explicitly."
        )
    if plan.strategy in ORDER_STRATEGIES and not options.row_order_meaningful:
        raise ValidationError(
            f"Forward or backward fill copies values from neighbouring rows, which only makes sense when row order "
            f"is meaningful (for example a time series). Confirm that row order is meaningful "
            f"to use it on '{plan.column}'."
        )
    if plan.strategy == "drop_column" and not options.confirm_drop_columns:
        raise ValidationError(
            f"Dropping the column '{plan.column}' removes it from the new version. Confirm column removal to continue."
        )


def validate_plans(
    plans: Any, types: dict[str, str], options: TreatmentOptions, settings: dict[str, Any]
) -> list[ColumnPlan]:
    """Schema-level validation of a treatment plan (data-level checks happen in ``treat_frame``)."""
    if not isinstance(plans, (list, tuple)) or not plans:
        raise ValidationError("Choose a treatment for at least one column.")
    placeholders = placeholder_set(settings[org_settings.MISSING_PLACEHOLDERS])
    _validate_fit_filter(options.fit_filter, types)
    result: list[ColumnPlan] = []
    seen: set[str] = set()
    for entry in plans:
        if not isinstance(entry, dict) or "column" not in entry or "strategy" not in entry:
            raise ValidationError("Each plan entry needs a column and a strategy.")
        column, strategy = str(entry["column"]), str(entry["strategy"])
        if column not in types:
            raise ValidationError(f"Unknown column: {column}")
        if column in seen:
            raise ValidationError(f"'{column}' appears more than once in the plan.")
        seen.add(column)
        logical = types[column]
        allowed = STRATEGIES[type_group(logical)]
        if strategy not in allowed:
            raise ValidationError(
                f"'{strategy}' cannot be used for the {logical} column '{column}'. Allowed: {', '.join(allowed)}."
            )
        value: Any = None
        if strategy in VALUE_STRATEGIES:
            value = _parse_value(entry.get("value"), column, logical, placeholders)
        elif strategy == "placeholder":
            value = _parse_text(str(settings[org_settings.TEXT_PLACEHOLDER]), column, placeholders)
        plan = ColumnPlan(column=column, strategy=strategy, logical_type=logical, value=value)
        _check_permissions(plan, options)
        result.append(plan)
    return result


# ---------------------------------------------------------------------------
# Treatment
# ---------------------------------------------------------------------------


@dataclass
class Treatment:
    frame: pd.DataFrame
    columns: list[dict[str, Any]]
    stats: dict[str, int]
    columns_removed: list[str]
    fit_rows: int
    warnings: list[str]
    sample: list[dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "columns_removed": self.columns_removed,
            "fit_rows": self.fit_rows,
            "warnings": self.warnings,
        }


def _fit_mask(frame: pd.DataFrame, fit_filter: dict[str, Any] | None) -> pd.Series:
    if fit_filter is None:
        return pd.Series(True, index=frame.index)
    wanted = [str(v) for v in fit_filter["values"]]
    mask = frame[fit_filter["column"]].astype("string").isin(wanted).fillna(False).astype(bool)
    if not mask.any():
        raise ValidationError("No rows match the fit filter, so no statistics can be computed from it.")
    return mask


def _is_identifier(series: pd.Series, column: str, mask: pd.Series) -> bool:
    present = series[~mask]
    if not len(present) or not looks_like_identifier_name(column):
        return False
    return present.nunique() >= IDENTIFIER_DISTINCT_RATIO * len(present)


def _mode(values: pd.Series, column: str, warnings: list[str]) -> Any:
    """Most frequent value; ties go to the value that appears first in the column."""
    counts = values.value_counts()
    top = int(counts.max())
    tied = set(counts[counts == top].index.tolist())
    winner = next(v for v in values.tolist() if v in tied)
    if len(tied) > 1:
        warnings.append(
            f"'{column}': {len(tied)} values are equally frequent; the first one in row order ({winner}) was used."
        )
    return winner


def _statistic(plan: ColumnPlan, values: pd.Series, warnings: list[str]) -> Any:
    if not len(values):
        raise ValidationError(
            f"'{plan.column}' has no values in the fitted rows to compute the {plan.strategy} from."
        )
    if plan.strategy == "mode":
        return _mode(values, plan.column, warnings)
    exact = float(
        values.astype("float64").mean() if plan.strategy == "mean" else values.astype("float64").median()
    )
    if plan.logical_type != LogicalType.INTEGER.value:
        return exact
    rounded = int(round(exact))  # Python rounds half to even
    if rounded != exact:
        warnings.append(
            f"'{plan.column}' is an integer column, so the {plan.strategy} {exact:g} was rounded "
            f"half to even to {rounded}."
        )
    return rounded


def _fill_value(
    plan: ColumnPlan, frame: pd.DataFrame, masks: dict[str, pd.Series], fit: pd.Series, warnings: list[str]
) -> Any:
    if plan.strategy in STATISTIC_STRATEGIES:
        column = frame[plan.column]
        return _statistic(plan, column[fit & ~masks[plan.column]], warnings)
    return plan.value


def _apply_fill(series: pd.Series, mask: pd.Series, plan: ColumnPlan, value: Any) -> pd.Series:
    dtype = series.dtype
    cleared = series.mask(mask)
    if plan.strategy == "ffill":
        return cleared.ffill().astype(dtype)
    if plan.strategy == "bfill":
        return cleared.bfill().astype(dtype)
    return cleared.mask(mask, value).astype(dtype)


def _sample(
    before: pd.DataFrame,
    after: pd.DataFrame,
    removed: pd.Series,
    changed: pd.Series,
    columns: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    positions = before.index[(removed | changed).to_numpy(dtype=bool)][:limit]
    kept_columns = [c for c in columns if c in after.columns]
    sample = []
    for position in positions.tolist():
        row_before = records(before.loc[[position], columns])[0]
        row_after = None if bool(removed.loc[position]) else records(after.loc[[position], kept_columns])[0]
        sample.append(
            {
                "row_index": int(position),
                "removed": row_after is None,
                "before": row_before,
                "after": row_after,
            }
        )
    return sample


def treat_frame(
    frame: pd.DataFrame,
    plans: list[ColumnPlan],
    options: TreatmentOptions,
    placeholders: frozenset[str],
    *,
    sample_rows: int = 20,
    check: Callable[[], None] | None = None,
) -> Treatment:
    """Apply validated plans to ``frame`` (not mutated) following the order in the module docstring.

    ``check`` (for example ``OperationGuard.check``) runs before each column is treated, so a
    timeout or a cancel request stops a large treatment between columns.
    """
    frame = frame.reset_index(drop=True)
    masks = {p.column: missing_mask(frame[p.column], p.logical_type, placeholders) for p in plans}
    for plan in plans:
        if plan.strategy not in IDENTIFIER_ALLOWED and _is_identifier(
            frame[plan.column], plan.column, masks[plan.column]
        ):
            raise ValidationError(
                f"'{plan.column}' looks like an identifier. Identifiers are never filled automatically; "
                "use an explicit constant, drop the rows, or leave it."
            )
    fit = _fit_mask(frame, options.fit_filter)
    warnings: list[str] = []
    values = {
        p.column: _fill_value(p, frame, masks, fit, warnings)
        for p in plans
        if p.strategy not in ("leave", "drop_rows", "drop_column")
    }

    removed = pd.Series(False, index=frame.index)
    for plan in plans:
        if plan.strategy == "drop_rows":
            removed |= masks[plan.column]
    result = frame[~removed].copy()
    changed = pd.Series(False, index=frame.index)
    report = []
    for plan in plans:
        if check is not None:
            check()
        kept_mask = masks[plan.column][~removed]
        cells = 0
        if plan.column in values:
            result[plan.column] = _apply_fill(result[plan.column], kept_mask, plan, values[plan.column])
            filled = kept_mask & ~missing_mask(result[plan.column], plan.logical_type, placeholders)
            cells = int(filled.sum())
            changed.loc[filled.index[filled.to_numpy(dtype=bool)]] = True
        after_missing = (
            0
            if plan.strategy == "drop_column"
            else int(missing_mask(result[plan.column], plan.logical_type, placeholders).sum())
        )
        if plan.strategy in ORDER_STRATEGIES and after_missing:
            warnings.append(
                f"'{plan.column}': {after_missing} missing value(s) at the "
                f"{'start' if plan.strategy == 'ffill' else 'end'} had no neighbour to copy from and remain missing."
            )
        report.append(
            {
                "column": plan.column,
                "strategy": plan.strategy,
                "logical_type": plan.logical_type,
                "missing_before": int(masks[plan.column].sum()),
                "missing_after": after_missing,
                "fill_value": json_safe(values.get(plan.column)),
                "cells_changed": cells,
                "fit_rows": int(fit.sum()) if plan.strategy in STATISTIC_STRATEGIES else None,
            }
        )
    dropped = [p.column for p in plans if p.strategy == "drop_column"]
    result = result.drop(columns=dropped)
    sample = _sample(frame, result, removed, changed, [p.column for p in plans], sample_rows)
    result = result.reset_index(drop=True)
    stats = {
        "cells_changed": sum(r["cells_changed"] for r in report),
        "rows_affected": int(changed.sum()),
        "rows_removed": int(removed.sum()),
    }
    return Treatment(
        frame=result,
        columns=report,
        stats=stats,
        columns_removed=dropped,
        fit_rows=int(fit.sum()),
        warnings=warnings,
        sample=sample,
    )


def _options(
    row_order_meaningful: bool,
    fit_filter: dict | None,
    allow_sensitive: bool,
    allow_target: bool,
    confirm_drop_columns: bool,
) -> TreatmentOptions:
    return TreatmentOptions(
        row_order_meaningful=row_order_meaningful is True,
        fit_filter=fit_filter,
        allow_sensitive=allow_sensitive is True,
        allow_target=allow_target is True,
        confirm_drop_columns=confirm_drop_columns is True,
    )


@dataclass
class _Prepared:
    ref: VersionRef
    plans: list[ColumnPlan]
    placeholders: frozenset[str]
    treatment: Treatment


def _prepare(
    ctx: AuthContext,
    dataset_id: int,
    plans: Any,
    options: TreatmentOptions,
    sample_rows: int,
    permission: Permission | None,
) -> _Prepared:
    """Authorize, validate the plan and run it on the planned columns of the current version."""
    ref = versioning_service.get_version_ref(
        ctx, dataset_id, permission=permission, feature=Feature.MISSING_VALUES
    )
    if (
        not isinstance(sample_rows, int)
        or isinstance(sample_rows, bool)
        or not 0 <= sample_rows <= MAX_SAMPLE_ROWS
    ):
        raise ValidationError(f"The sample size must be between 0 and {MAX_SAMPLE_ROWS} rows.")
    settings = _org_settings(ref.organization_id)
    validated = validate_plans(plans, _types(ref), options, settings)
    placeholders = placeholder_set(settings[org_settings.MISSING_PLACEHOLDERS])
    needed = [p.column for p in validated]
    if options.fit_filter is not None and options.fit_filter["column"] not in needed:
        needed.append(options.fit_filter["column"])
    frame = versioning_service.load_frame(
        ctx,
        dataset_id,
        version_id=ref.version_id,
        columns=needed,
        permission=permission,
        feature=Feature.MISSING_VALUES,
    )
    treatment = treat_frame(frame, validated, options, placeholders, sample_rows=sample_rows)
    return _Prepared(ref=ref, plans=validated, placeholders=placeholders, treatment=treatment)


def preview_missing_treatment(
    ctx: AuthContext,
    dataset_id: int,
    plans: list[dict],
    *,
    row_order_meaningful: bool = False,
    fit_filter: dict | None = None,
    allow_sensitive: bool = False,
    allow_target: bool = False,
    confirm_drop_columns: bool = False,
    sample_rows: int = 20,
) -> dict[str, Any]:
    """Read-only preview: before/after missing counts, fill values, impact, sample rows and warnings."""
    options = _options(row_order_meaningful, fit_filter, allow_sensitive, allow_target, confirm_drop_columns)
    prepared = _prepare(ctx, dataset_id, plans, options, sample_rows, None)
    treatment = prepared.treatment
    return {
        "rows_before": prepared.ref.rows,
        "rows_after": prepared.ref.rows - treatment.stats["rows_removed"],
        "columns": treatment.columns,
        **treatment.stats,
        "columns_removed": treatment.columns_removed,
        "fit_rows": treatment.fit_rows,
        "sample": treatment.sample,
        "warnings": treatment.warnings,
        "read_only": True,
    }


def apply_missing_treatment(
    ctx: AuthContext,
    dataset_id: int,
    plans: list[dict],
    *,
    row_order_meaningful: bool = False,
    fit_filter: dict | None = None,
    allow_sensitive: bool = False,
    allow_target: bool = False,
    confirm_drop_columns: bool = False,
    sample_rows: int = 20,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Validate, apply the plan to the whole current version and publish a new version.

    The operation timeout applies; pass ``guard`` to report progress or allow cancelling.
    """
    guard = guard or OperationGuard()
    options = _options(row_order_meaningful, fit_filter, allow_sensitive, allow_target, confirm_drop_columns)
    prepared = _prepare(ctx, dataset_id, plans, options, sample_rows, Permission.CLEAN_DATA)
    outcome: list[Treatment] = []

    def transform(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
        treatment = treat_frame(
            frame, prepared.plans, options, prepared.placeholders, sample_rows=sample_rows, check=guard.check
        )
        outcome.append(treatment)
        return treatment.frame, treatment.stats, treatment.summary()

    parameters = {
        "plans": [
            {"column": p.column, "strategy": p.strategy, "value": json_safe(p.value)} for p in prepared.plans
        ],
        **options.to_dict(),
    }
    published = versioning_service.apply_frame_operation(
        ctx,
        dataset_id,
        operation_type="missing_values",
        parameters=parameters,
        transform=transform,
        permission=Permission.CLEAN_DATA,
        feature=Feature.MISSING_VALUES,
        guard=guard,
    )
    treatment = outcome[-1]
    return {
        "version_number": published.version_number,
        "version_id": published.version_id,
        "rows": published.rows,
        "stats": treatment.stats,
        "columns": treatment.columns,
        "columns_removed": treatment.columns_removed,
        "fit_rows": treatment.fit_rows,
        "warnings": treatment.warnings,
        "quality_before": published.quality_before,
        "quality_after": published.quality_score,
    }


# ---------------------------------------------------------------------------
# Placeholder detection and standardization
# ---------------------------------------------------------------------------


def _placeholder_tokens(placeholders: Any, settings: dict[str, Any]) -> list[str]:
    if placeholders is None:
        return list(settings[org_settings.MISSING_PLACEHOLDERS])
    return org_settings.validate_changes({org_settings.MISSING_PLACEHOLDERS: placeholders})[
        org_settings.MISSING_PLACEHOLDERS
    ]


def _text_columns(ref: VersionRef, columns: list[str] | None) -> list[str]:
    types = _types(ref)
    text = [name for name, logical in types.items() if logical == LogicalType.TEXT.value]
    if columns is None:
        return text
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValidationError("Select at least one text column.")
    unknown = [c for c in columns if c not in types]
    if unknown:
        raise ValidationError(f"Unknown column(s): {', '.join(map(str, unknown))}")
    not_text = [c for c in columns if c not in text]
    if not_text:
        raise ValidationError(f"Placeholders only apply to text columns: {', '.join(not_text)}")
    return list(dict.fromkeys(columns))


@dataclass
class _PlaceholderTally:
    already_missing: int = 0
    empty: int = 0
    whitespace_only: int = 0
    matches: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.empty + self.whitespace_only + sum(self.matches.values())


def _placeholder_hits(
    series: pd.Series, lookup: dict[str, str]
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Masks for: real nulls, empty strings, whitespace-only strings and the canonical matched token."""
    text = series.astype("string")
    nulls = text.isna()
    empty = text.eq("").fillna(False)
    stripped = text.str.strip()
    whitespace = (stripped.eq("") & ~empty).fillna(False)
    token = stripped.str.casefold().map(lookup, na_action="ignore")
    return nulls, empty, whitespace, token


def _tally_batch(tally: _PlaceholderTally, series: pd.Series, lookup: dict[str, str]) -> pd.Series:
    nulls, empty, whitespace, token = _placeholder_hits(series, lookup)
    tally.already_missing += int(nulls.sum())
    tally.empty += int(empty.sum())
    tally.whitespace_only += int(whitespace.sum())
    for name, count in token.dropna().value_counts().items():
        tally.matches[str(name)] = tally.matches.get(str(name), 0) + int(count)
    return (empty | whitespace | token.notna()).fillna(False).astype(bool)


def _lookup(tokens: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for token in tokens:
        lookup.setdefault(token.strip().casefold(), token)
    return lookup


def _count_placeholders(
    ctx: AuthContext, ref: VersionRef, columns: list[str], tokens: list[str]
) -> dict[str, _PlaceholderTally]:
    tallies = {name: _PlaceholderTally() for name in columns}
    if not columns:
        return tallies
    lookup = _lookup(tokens)
    for batch in versioning_service.iter_frame_batches(
        ctx, ref.dataset_id, version_id=ref.version_id, columns=columns, feature=Feature.MISSING_VALUES
    ):
        for name in columns:
            _tally_batch(tallies[name], batch[name], lookup)
    return tallies


def _tally_dict(name: str, tally: _PlaceholderTally) -> dict[str, Any]:
    return {
        "column": name,
        "empty": tally.empty,
        "whitespace_only": tally.whitespace_only,
        "placeholder_matches": dict(tally.matches),
        "total": tally.total,
        "already_missing": tally.already_missing,
    }


def detect_placeholders(
    ctx: AuthContext, dataset_id: int, placeholders: list[str] | None = None
) -> dict[str, Any]:
    """Count empty, whitespace-only and placeholder text values per text column (read-only).

    Uses ``placeholders`` when given, else the organization's configured
    tokens. Matching trims and ignores case. Nothing else (zero, False,
    "Unknown") is counted unless it is listed.
    """
    ref = versioning_service.get_version_ref(ctx, dataset_id, feature=Feature.MISSING_VALUES)
    tokens = _placeholder_tokens(placeholders, _org_settings(ref.organization_id))
    columns = _text_columns(ref, None)
    tallies = _count_placeholders(ctx, ref, columns, tokens)
    return {
        "placeholders": tokens,
        "columns": [_tally_dict(name, tallies[name]) for name in columns],
        "total": sum(t.total for t in tallies.values()),
        "note": (
            "Zero, False and categories such as 'Unknown' are never counted "
            "unless they are listed as placeholders."
        ),
    }


def standardize_placeholders(
    ctx: AuthContext, dataset_id: int, placeholders: list[str], *, columns: list[str] | None = None
) -> dict[str, Any]:
    """Convert empty, whitespace-only and listed placeholder text values to real missing values.

    Publishes a new version (operation ``standardize_placeholders``). Nothing is
    published when no value matches.
    """
    ref = versioning_service.get_version_ref(
        ctx, dataset_id, permission=Permission.CLEAN_DATA, feature=Feature.MISSING_VALUES
    )
    if not isinstance(placeholders, (list, tuple)):
        raise ValidationError("Placeholders must be a list of text values.")
    tokens = _placeholder_tokens(placeholders, _org_settings(ref.organization_id))
    targets = _text_columns(ref, columns)
    counts = _count_placeholders(ctx, ref, targets, tokens)
    per_column = {name: _tally_dict(name, tally) for name, tally in counts.items() if tally.total}
    total = sum(item["total"] for item in per_column.values())
    if not total:
        return {"changed": False, "version_number": None, "cells_changed": 0, "columns": per_column}
    lookup = _lookup(tokens)

    def transform(batch: pd.DataFrame, _offset: int) -> tuple[pd.DataFrame, dict[str, int]]:
        out = batch.copy()
        touched = np.zeros(len(batch), dtype=bool)
        cells = 0
        for name in per_column:
            hits = _tally_batch(_PlaceholderTally(), batch[name], lookup)
            if hits.any():
                out[name] = batch[name].astype("string").mask(hits)
                cells += int(hits.sum())
                touched |= hits.to_numpy(dtype=bool)
        return out, {"cells_changed": cells, "rows_affected": int(touched.sum())}

    parameters = {"placeholders": tokens, "columns": targets}
    published = versioning_service.apply_streaming_operation(
        ctx,
        dataset_id,
        operation_type="standardize_placeholders",
        parameters=parameters,
        transform_batch=transform,
        permission=Permission.CLEAN_DATA,
        feature=Feature.MISSING_VALUES,
        summary={"placeholders": tokens, "columns": per_column, "cells_changed": total},
    )
    return {
        "changed": True,
        "version_number": published.version_number,
        "version_id": published.version_id,
        "cells_changed": total,
        "columns": per_column,
        "quality_before": published.quality_before,
        "quality_after": published.quality_score,
    }
