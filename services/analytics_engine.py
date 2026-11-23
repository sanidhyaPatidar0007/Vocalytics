"""Safe natural-language analytics: plan validation, pandas execution and verification.

Pipeline (spec sections 13, 14 and 16):

1. ``get_schema_summary`` builds a bounded, sanitized description of the dataset.
2. ``llm_service.generate_plan`` (or the deterministic ``rule_based_plan``) proposes a
   JSON plan. Model output is never executed: it is only data.
3. ``validate_plan`` checks the plan against a strict allowlist (operations,
   aggregations, filter operators, existing columns, type rules, bounded literals)
   and returns a normalized copy.
4. ``execute_plan`` runs the plan with plain pandas code paths chosen from fixed
   tables, never by looking names up dynamically.
5. ``verify_result`` recomputes the key numbers with an independent numpy path;
   a mismatch means the numbers are never shown.
6. ``answer_question`` / ``run_plan`` add authorization, plan features, quotas,
   history, usage metering and a deterministic business-language summary. An
   optional model narrative is kept only when every number in it matches a
   computed number.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from auth.authorization import AuthContext, authorize, authorize_dataset
from config.settings import get_settings
from core.constants import Feature, LogicalType, Permission, UsageMetric
from core.errors import (
    AIBusyError,
    AppError,
    LLMResponseError,
    LLMUnavailableError,
    PlanValidationError,
    ValidationError,
)
from database.connection import session_scope
from database.models import AnalysisHistory
from modules import subscriptions, usage_tracking
from services import llm_service, versioning_service
from services.audit_service import record_audit, record_error
from services.frame_utils import frame_schema, hash_rows, json_safe, logical_type_for_dtype, records

# ---------------------------------------------------------------------------
# Allowlists and limits
# ---------------------------------------------------------------------------

OPERATIONS = llm_service.OPERATIONS
AGGREGATIONS = llm_service.AGGREGATIONS
FILTER_OPS = llm_service.FILTER_OPS
FREQUENCIES = llm_service.FREQUENCIES
CHART_TYPES = llm_service.CHART_TYPES
PLAN_KEYS = (
    "operation",
    "metrics",
    "group_by",
    "filters",
    "time_column",
    "frequency",
    "sort",
    "limit",
    "columns",
    "chart",
    "clarification_question",
)

ADVANCED_OPERATIONS = frozenset({"correlation", "growth", "anomalies"})
TIME_OPERATIONS = frozenset({"time_series", "growth", "anomalies"})
GROUP_OPERATIONS = frozenset({"group_aggregate", "compare", "top_n"})
SINGLE_METRIC_OPERATIONS = frozenset({"top_n", "growth", "anomalies"})

NUMERIC_TYPES = frozenset({LogicalType.INTEGER.value, LogicalType.FLOAT.value})
ORDERED_TYPES = NUMERIC_TYPES | {LogicalType.DATETIME.value}
NUMERIC_ONLY_AGGS = frozenset({"sum", "mean", "median", "std"})
ORDERED_AGGS = frozenset({"min", "max"})
ORDERED_FILTER_OPS = frozenset({"gt", "gte", "lt", "lte", "between"})
LIST_FILTER_OPS = frozenset({"in", "not_in"})
NULL_FILTER_OPS = frozenset({"is_null", "not_null"})

MAX_METRICS = 5
MAX_GROUP_BY = 2
MAX_FILTERS = 5
MAX_COLUMNS = 20
MAX_LITERAL_CHARS = 200
MAX_LIST_ITEMS = 50
MAX_CLARIFICATION_CHARS = 500
MAX_QUESTION_CHARS = 1000
DEFAULT_TOP_N = 10
DEFAULT_FREQUENCY = "M"
MAX_HISTOGRAM_BINS = 50
ROBUST_Z_THRESHOLD = 3.5
IQR_FACTOR = 1.5
MIN_ANOMALY_CHANGES = 4
NARRATIVE_ROWS = 20

MAX_SUMMARY_COLUMNS = 200
SUMMARY_SAMPLE_VALUES = 5
SUMMARY_SAMPLE_CHARS = 40
SUMMARY_DISTINCT_TRACK = 1000
CATEGORICAL_DISTINCT = 50
CATEGORICAL_RATIO = 0.5
IDENTIFIER_RATIO = 0.9
_IDENTIFIER_NAME = r"(?i)(^id$|_id$|^id_|\sid$|code$|number$|^no$|_no$|uuid|guid|sku$)"

FALLBACK_NOTE = "Local AI model unavailable; answered with the built-in rule-based interpreter."
GROWTH_ZERO_WARNING = "growth undefined because the previous period was zero"
ANOMALY_NOTE = "unusual change, review"
NO_ROWS_WARNING = "No rows match the filters, so there is nothing to calculate."
NARRATIVE_DISCARDED = (
    "The AI narrative was discarded because it mentioned numbers that are not in the computed results."
)
NARRATIVE_UNAVAILABLE = "The AI narrative is unavailable; the summary is computed directly from the data."

# Result-table labels that plans may reference in sort and chart settings.
OUTPUT_LABELS = frozenset(
    {
        "period",
        "row_count",
        "growth_pct",
        "change",
        "robust_z",
        "flagged",
        "note",
        "column",
        "missing_count",
        "missing_pct",
        "count",
        "percent",
        "bin",
        "bin_start",
        "bin_end",
        "value",
        "rows",
    }
)

AGG_WORDS = {
    "sum": "total",
    "mean": "average",
    "median": "median",
    "min": "minimum",
    "max": "maximum",
    "count": "count of",
    "nunique": "distinct count of",
    "std": "standard deviation of",
}
_PERIOD_FREQ = {"D": "D", "W": "W", "M": "M", "Q": "Q", "Y": "Y"}
_ZERO_FILLED_AGGS = frozenset({"sum", "count", "nunique"})


class ResultVerificationError(AppError):
    """Independent recomputation disagreed with the executed result."""

    default_message = "The result could not be independently verified, so it is not shown."


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clean(value: Any, max_chars: int) -> str:
    """Single-line text without control characters, truncated (safe for messages)."""
    return llm_service.clean_text(value, max_chars)


def _display(name: Any) -> str:
    return _clean(name, 60)


def metric_label(metric: dict[str, str]) -> str:
    """Result column name of a metric, for example ``sum_revenue``."""
    return f"{metric['agg']}_{metric['column']}"


def metric_phrase(metric: dict[str, str] | None) -> str:
    """Business wording of a metric, for example ``total revenue``."""
    if metric is None:
        return "number of rows"
    return f"{AGG_WORDS[metric['agg']]} {metric['column']}"


def _metrics_or_count(plan: dict[str, Any]) -> list[dict[str, str] | None]:
    """Plan metrics, or a single row-count placeholder (None) when there are none."""
    return list(plan["metrics"]) or [None]


def _label_for(metric: dict[str, str] | None) -> str:
    return "row_count" if metric is None else metric_label(metric)


def _plain(value: Any) -> Any:
    """JSON-safe Python scalar (numpy float subclasses become plain floats)."""
    value = json_safe(value)
    if isinstance(value, float):
        return float(value)
    return value


def _fact(label: str, value: Any) -> dict[str, Any]:
    return {"label": label, "value": _plain(value)}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def format_number(value: Any) -> str:
    """Readable number: thousands separators, at most two decimals."""
    if value is None:
        return "not available"
    if not _is_number(value):
        return str(value)
    number = float(value)
    if math.isnan(number):
        return "not available"
    if number.is_integer():
        return f"{number:,.0f}"
    if 0 < abs(number) < 0.01:
        return f"{number:.4g}"
    return f"{number:,.2f}"


def _schema_index(schema: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """Map case-folded column name to (exact name, logical type)."""
    index: dict[str, tuple[str, str]] = {}
    for column in schema or []:
        if isinstance(column, dict) and "name" in column:
            name = str(column["name"])
            index[name.casefold()] = (name, str(column.get("logical_type", LogicalType.TEXT.value)))
    return index


# ---------------------------------------------------------------------------
# Plan validation (pure)
# ---------------------------------------------------------------------------


def _resolve_column(value: Any, index: dict[str, tuple[str, str]], where: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_LITERAL_CHARS:
        raise PlanValidationError(f"Each entry in {where} must be a column name.")
    found = index.get(value.casefold())
    if found is None:
        raise PlanValidationError(f'Unknown column "{_display(value)}" in {where}.')
    return found


def _list_field(plan: dict[str, Any], key: str, max_items: int) -> list[Any]:
    value = plan.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise PlanValidationError(f"{key} must be a list.")
    if len(value) > max_items:
        raise PlanValidationError(f"{key} may contain at most {max_items} entries.")
    return value


def _check_keys(item: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = [k for k in item if k not in allowed]
    if extra:
        raise PlanValidationError(
            f"Unsupported field(s) in {where}: {', '.join(_display(k) for k in extra)}."
        )


def _validate_metrics(plan: dict[str, Any], index: dict[str, tuple[str, str]]) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    for item in _list_field(plan, "metrics", MAX_METRICS):
        if not isinstance(item, dict):
            raise PlanValidationError("Each metric must be an object with a column and an aggregation.")
        _check_keys(item, {"column", "agg"}, "a metric")
        name, logical = _resolve_column(item.get("column"), index, "metrics")
        agg = item.get("agg")
        if not isinstance(agg, str) or agg not in AGGREGATIONS:
            raise PlanValidationError(f"Unsupported aggregation. Use one of: {', '.join(AGGREGATIONS)}.")
        if agg in NUMERIC_ONLY_AGGS and logical not in NUMERIC_TYPES:
            raise PlanValidationError(f'"{agg}" needs a numeric column, but "{_display(name)}" is {logical}.')
        if agg in ORDERED_AGGS and logical not in ORDERED_TYPES:
            raise PlanValidationError(
                f'"{agg}" needs a numeric or date column, but "{_display(name)}" is {logical}.'
            )
        metric = {"column": name, "agg": agg}
        if metric in metrics:
            raise PlanValidationError(f'The metric {agg} of "{_display(name)}" is listed twice.')
        metrics.append(metric)
    return metrics


def _validate_columns(
    plan: dict[str, Any], key: str, max_items: int, index: dict[str, tuple[str, str]]
) -> list[str]:
    names: list[str] = []
    for value in _list_field(plan, key, max_items):
        name, _logical = _resolve_column(value, index, key)
        if name in names:
            raise PlanValidationError(f'Column "{_display(name)}" is listed twice in {key}.')
        names.append(name)
    return names


def _literal(value: Any, logical: str) -> Any:
    """Validate one filter literal for a column type; return a JSON-safe normalized value."""
    if isinstance(value, (list, dict)) or value is None:
        raise PlanValidationError("Filter values must be single numbers, dates or short texts here.")
    if isinstance(value, str) and len(value) > MAX_LITERAL_CHARS:
        raise PlanValidationError(f"Filter texts may be at most {MAX_LITERAL_CHARS} characters long.")
    if logical in NUMERIC_TYPES:
        return _numeric_literal(value)
    if logical == LogicalType.DATETIME.value:
        return _date_literal(value)
    if logical == LogicalType.BOOLEAN.value:
        return _boolean_literal(value)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise PlanValidationError("Text filters need a text value.")
    return str(value)


def _numeric_literal(value: Any) -> float:
    if isinstance(value, bool):
        raise PlanValidationError("Numeric filters need a number.")
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", text):
            raise PlanValidationError("Numeric filters need a number.")
        value = float(text)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PlanValidationError("Numeric filters need a finite number.")
    return float(value)


def _date_literal(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError("Date filters need a date such as 2024-01-31.")
    try:
        stamp = pd.Timestamp(value.strip())
    except (ValueError, TypeError, OverflowError) as exc:
        raise PlanValidationError(f'"{_display(value)}" is not a date that can be understood.') from exc
    if pd.isna(stamp):
        raise PlanValidationError(f'"{_display(value)}" is not a date that can be understood.')
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.isoformat()


def _boolean_literal(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no"}:
        return False
    raise PlanValidationError("True/false filters need true or false.")


def _validate_filter(item: Any, index: dict[str, tuple[str, str]]) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise PlanValidationError("Each filter must be an object with a column, an operator and a value.")
    _check_keys(item, {"column", "op", "value"}, "a filter")
    name, logical = _resolve_column(item.get("column"), index, "filters")
    op = item.get("op")
    if not isinstance(op, str) or op not in FILTER_OPS:
        raise PlanValidationError(f"Unsupported filter operator. Use one of: {', '.join(FILTER_OPS)}.")
    value = item.get("value")
    if op in NULL_FILTER_OPS:
        if value is not None:
            raise PlanValidationError(f'The "{op}" filter does not take a value.')
        return {"column": name, "op": op, "value": None}
    if op in ORDERED_FILTER_OPS and logical not in ORDERED_TYPES:
        raise PlanValidationError(
            f'"{op}" filters need a numeric or date column, but "{_display(name)}" is {logical}.'
        )
    if op == "contains" and logical != LogicalType.TEXT.value:
        raise PlanValidationError(
            f'"contains" filters need a text column, but "{_display(name)}" is {logical}.'
        )
    if op in LIST_FILTER_OPS or op == "between":
        if not isinstance(value, list) or not value or len(value) > MAX_LIST_ITEMS:
            raise PlanValidationError(f'The "{op}" filter needs a list of 1 to {MAX_LIST_ITEMS} values.')
        values = [_literal(v, logical) for v in value]
        if op == "between":
            if len(values) != 2:
                raise PlanValidationError(
                    'The "between" filter needs exactly two values: a lower and an upper bound.'
                )
            if _ordered_key(values[0]) > _ordered_key(values[1]):
                raise PlanValidationError(
                    'The lower bound of a "between" filter must not exceed the upper bound.'
                )
        return {"column": name, "op": op, "value": values}
    literal = _literal(value, logical)
    if op == "contains" and not literal.strip():
        raise PlanValidationError('The "contains" filter needs some text to look for.')
    return {"column": name, "op": op, "value": literal}


def _ordered_key(value: Any) -> float:
    return pd.Timestamp(value).value if isinstance(value, str) else float(value)


def _reference(value: Any, allowed: set[str], where: str) -> str | None:
    """A sort/chart reference: a dataset column, a metric label or a result-table label."""
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > MAX_LITERAL_CHARS:
        raise PlanValidationError(f"{where} must name a column.")
    lookup = {a.casefold(): a for a in allowed}
    found = lookup.get(value.casefold())
    if found is None:
        raise PlanValidationError(f'Unknown column "{_display(value)}" in {where}.')
    return found


def _validate_sort(plan: dict[str, Any], allowed: set[str]) -> dict[str, Any] | None:
    sort = plan.get("sort")
    if sort is None:
        return None
    if not isinstance(sort, dict):
        raise PlanValidationError("sort must be an object with by and ascending.")
    _check_keys(sort, {"by", "ascending"}, "sort")
    ascending = sort.get("ascending", False)
    if not isinstance(ascending, bool):
        raise PlanValidationError("sort.ascending must be true or false.")
    by = _reference(sort.get("by"), allowed, "sort")
    if by is None:
        raise PlanValidationError("sort.by must name a column.")
    return {"by": by, "ascending": ascending}


def _validate_limit(plan: dict[str, Any]) -> int | None:
    limit = plan.get("limit")
    if limit is None:
        return None
    max_rows = get_settings().result_max_rows
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= max_rows:
        raise PlanValidationError(f"limit must be a whole number from 1 to {max_rows}.")
    return limit


def _validate_chart(plan: dict[str, Any], allowed: set[str]) -> dict[str, Any] | None:
    chart = plan.get("chart")
    if chart is None:
        return None
    if not isinstance(chart, dict):
        raise PlanValidationError("chart must be an object.")
    _check_keys(chart, {"type", "x", "y", "color", "stacked"}, "chart")
    chart_type = chart.get("type", "auto")
    if not isinstance(chart_type, str) or chart_type not in CHART_TYPES:
        raise PlanValidationError(f"Unsupported chart type. Use one of: {', '.join(CHART_TYPES)}.")
    stacked = chart.get("stacked", False)
    if not isinstance(stacked, bool):
        raise PlanValidationError("chart.stacked must be true or false.")
    return {
        "type": chart_type,
        "x": _reference(chart.get("x"), allowed, "chart.x"),
        "y": _reference(chart.get("y"), allowed, "chart.y"),
        "color": _reference(chart.get("color"), allowed, "chart.color"),
        "stacked": stacked,
    }


def _validate_time(plan: dict[str, Any], index: dict[str, tuple[str, str]]) -> tuple[str | None, str | None]:
    time_value, frequency = plan.get("time_column"), plan.get("frequency")
    time_column = None
    if time_value is not None:
        time_column, logical = _resolve_column(time_value, index, "time_column")
        if logical != LogicalType.DATETIME.value:
            raise PlanValidationError(
                f'"{_display(time_column)}" is not a date column, so it cannot be used over time.'
            )
    if frequency is not None:
        if not isinstance(frequency, str) or frequency not in FREQUENCIES:
            raise PlanValidationError(f"frequency must be one of: {', '.join(FREQUENCIES)}.")
        if time_column is None:
            raise PlanValidationError("frequency can only be used together with a date time_column.")
    return time_column, frequency


def _validate_clarification(plan: dict[str, Any]) -> str | None:
    value = plan.get("clarification_question")
    if value is None:
        return None
    if not isinstance(value, str):
        raise PlanValidationError("clarification_question must be text.")
    if len(value) > MAX_CLARIFICATION_CHARS:
        raise PlanValidationError(
            f"clarification_question may be at most {MAX_CLARIFICATION_CHARS} characters long."
        )
    return _clean(value, MAX_CLARIFICATION_CHARS) or None


def _numeric_columns(index: dict[str, tuple[str, str]]) -> list[str]:
    return [name for name, logical in index.values() if logical in NUMERIC_TYPES]


def _check_operation_rules(plan: dict[str, Any], index: dict[str, tuple[str, str]]) -> None:
    """Operation-specific requirements; fills operation defaults in place."""
    op = plan["operation"]
    if op == "clarify":
        if not plan["clarification_question"]:
            raise PlanValidationError("A clarification plan needs a clarification_question.")
        return
    if op in TIME_OPERATIONS:
        if plan["time_column"] is None:
            raise PlanValidationError(f"The {op} operation needs a date time_column.")
        if plan["group_by"]:
            raise PlanValidationError(f"The {op} operation does not support group_by.")
        plan["frequency"] = plan["frequency"] or DEFAULT_FREQUENCY
    elif plan["time_column"] is not None:
        raise PlanValidationError("time_column is only used by time_series, growth and anomalies.")
    if op in SINGLE_METRIC_OPERATIONS and len(plan["metrics"]) > 1:
        raise PlanValidationError(f"The {op} operation uses a single metric.")
    if op == "aggregate":
        if not plan["metrics"]:
            raise PlanValidationError("The aggregate operation needs at least one metric.")
        if plan["group_by"]:
            raise PlanValidationError("Use group_aggregate to calculate metrics per group.")
    if op in GROUP_OPERATIONS and not plan["group_by"]:
        raise PlanValidationError(f"The {op} operation needs a group_by column.")
    if op == "top_n":
        if len(plan["group_by"]) != 1:
            raise PlanValidationError("The top_n operation groups by exactly one column.")
        plan["limit"] = plan["limit"] or DEFAULT_TOP_N
    if op == "correlation":
        candidates = plan["columns"] or _numeric_columns(index)
        if any(index[c.casefold()][1] not in NUMERIC_TYPES for c in candidates):
            raise PlanValidationError("Correlation only works with numeric columns.")
        if len(candidates) < 2:
            raise PlanValidationError("Correlation needs at least two numeric columns.")
    if op == "distribution":
        if len(plan["columns"]) != 1:
            raise PlanValidationError("The distribution operation needs exactly one column in columns.")
        if index[plan["columns"][0].casefold()][1] == LogicalType.DATETIME.value:
            raise PlanValidationError("Use a time series to see how a date column is spread over time.")


def validate_plan(plan: dict, schema: list[dict]) -> dict:
    """Validate an analysis plan against the allowlist and the dataset schema.

    ``schema`` is a list of ``{"name", "logical_type"}`` entries. Returns a new,
    normalized plan with every key present (missing optional keys get defaults)
    and column names spelled exactly as in the schema. Raises
    PlanValidationError with a user-friendly reason for anything unsupported.
    Pure: no I/O, no settings other than the result row cap.
    """
    if not isinstance(plan, dict):
        raise PlanValidationError("The analysis plan must be a JSON object.")
    extra = [k for k in plan if k not in PLAN_KEYS]
    if extra:
        raise PlanValidationError(
            f"The plan contains unsupported field(s): {', '.join(_display(k) for k in extra)}."
        )
    operation = plan.get("operation")
    if not isinstance(operation, str) or operation not in OPERATIONS:
        raise PlanValidationError(f"Unsupported operation. Use one of: {', '.join(OPERATIONS)}.")
    index = _schema_index(schema)
    metrics = _validate_metrics(plan, index)
    time_column, frequency = _validate_time(plan, index)
    references = {name for name, _ in index.values()} | {metric_label(m) for m in metrics} | OUTPUT_LABELS
    normalized: dict[str, Any] = {
        "operation": operation,
        "metrics": metrics,
        "group_by": _validate_columns(plan, "group_by", MAX_GROUP_BY, index),
        "filters": [_validate_filter(item, index) for item in _list_field(plan, "filters", MAX_FILTERS)],
        "time_column": time_column,
        "frequency": frequency,
        "sort": _validate_sort(plan, references),
        "limit": _validate_limit(plan),
        "columns": _validate_columns(plan, "columns", MAX_COLUMNS, index),
        "chart": _validate_chart(plan, references),
        "clarification_question": _validate_clarification(plan),
    }
    _check_operation_rules(normalized, index)
    return normalized


def required_features(plan: dict[str, Any]) -> list[Feature]:
    """Plan features needed to run a validated plan."""
    features = [Feature.TEXT_AI_ANALYSIS]
    chart = plan.get("chart") or {}
    if plan["operation"] in ADVANCED_OPERATIONS or chart.get("type") == "heatmap":
        features.append(Feature.ADVANCED_ANALYTICS)
    return features


def columns_needed(plan: dict[str, Any], schema: list[dict[str, Any]]) -> list[str] | None:
    """Columns to load for a validated plan, in schema order; None means every column."""
    op = plan["operation"]
    if op in {"describe", "missing_values", "duplicates"} and not plan["columns"]:
        return None
    wanted = (
        {m["column"] for m in plan["metrics"]}
        | set(plan["group_by"])
        | {f["column"] for f in plan["filters"]}
    )
    wanted |= set(plan["columns"])
    if plan["time_column"]:
        wanted.add(plan["time_column"])
    names = [str(c["name"]) for c in schema]
    # Sort and chart references that name dataset columns are loaded too, so the
    # re-validation inside execute_plan accepts them; execution then decides
    # whether they appear in the result (otherwise the chart default applies).
    chart = plan["chart"] or {}
    references = [(plan["sort"] or {}).get("by"), chart.get("x"), chart.get("y"), chart.get("color")]
    wanted |= {name for name in references if name in names}
    if op == "correlation" and not plan["columns"]:
        wanted |= set(_numeric_columns(_schema_index(schema)))
    if not wanted and names:
        wanted.add(names[0])  # count_rows without filters still needs the row count
    return [name for name in names if name in wanted]


# ---------------------------------------------------------------------------
# Execution (pure pandas)
# ---------------------------------------------------------------------------


def _text_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def _pandas_literal(value: Any, logical: str) -> Any:
    if logical in NUMERIC_TYPES:
        return float(value)
    if logical == LogicalType.DATETIME.value:
        return pd.Timestamp(value)
    if logical == LogicalType.BOOLEAN.value:
        return bool(value)
    return str(value).strip().casefold()


def _filter_mask(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.Series:
    """Rows passing every filter. Missing values never match a comparison (SQL semantics)."""
    mask = pd.Series(True, index=frame.index)
    for item in filters:
        series = frame[item["column"]]
        logical = logical_type_for_dtype(series.dtype)
        op, value = item["op"], item["value"]
        if op == "is_null":
            part = series.isna()
        elif op == "not_null":
            part = series.notna()
        else:
            target = _text_key(series) if logical == LogicalType.TEXT.value else series
            whole_days = logical == LogicalType.DATETIME.value and _has_time_of_day(series)
            part = _compare(target, op, value, logical, whole_days) & series.notna()
        mask &= part.fillna(False).astype(bool)
    return mask


def _has_time_of_day(series: pd.Series) -> bool:
    stamps = series.dropna()
    return bool((stamps != stamps.dt.normalize()).any())


_WHOLE_DAY_OPS = frozenset({"eq", "ne", "lte", "gt"})


def _compare(target: pd.Series, op: str, value: Any, logical: str, whole_days: bool = False) -> pd.Series:
    """One filter comparison. With ``whole_days`` (a timestamp column with times of day), a
    midnight date bound means the whole calendar day for eq, ne, lte, gt and a between upper bound."""
    if op == "contains":
        return target.str.contains(_pandas_literal(value, logical), regex=False)
    if op in LIST_FILTER_OPS:
        values = [_pandas_literal(v, logical) for v in value]
        member = target.isin(values)
        return member if op == "in" else ~member
    if op == "between":
        low, high = (_pandas_literal(v, logical) for v in value)
        if whole_days and high == high.normalize():
            return (target >= low) & (target < high + pd.Timedelta(days=1))
        return (target >= low) & (target <= high)
    literal = _pandas_literal(value, logical)
    if whole_days and op in _WHOLE_DAY_OPS and literal == literal.normalize():
        next_day = literal + pd.Timedelta(days=1)
        day = (target >= literal) & (target < next_day)
        return {"eq": day, "ne": ~day, "lte": target < next_day, "gt": target >= next_day}[op]
    comparisons: dict[str, Callable[[], pd.Series]] = {
        "eq": lambda: target == literal,
        "ne": lambda: target != literal,
        "gt": lambda: target > literal,
        "gte": lambda: target >= literal,
        "lt": lambda: target < literal,
        "lte": lambda: target <= literal,
    }
    return comparisons[op]()


def _metric_input(series: pd.Series, agg: str) -> pd.Series:
    """Nullable integers become float64 for statistics that are not exact integer sums."""
    if agg in {"mean", "median", "std"}:
        return series.astype("float64")
    return series


def _aggregate_series(series: pd.Series, agg: str) -> Any:
    values = _metric_input(series, agg)
    results: dict[str, Callable[[], Any]] = {
        "sum": lambda: values.sum(min_count=1),  # no values: not available, never a made-up zero
        "mean": values.mean,
        "median": values.median,
        "min": values.min,
        "max": values.max,
        "count": values.count,
        "nunique": lambda: values.nunique(dropna=True),
        "std": values.std,
    }
    return json_safe(results[agg]())


_GROUP_AGGS: dict[str, Callable[[Any], pd.Series]] = {
    "sum": lambda grouped: grouped.sum(min_count=1),
    "mean": lambda grouped: grouped.mean(),
    "median": lambda grouped: grouped.median(),
    "min": lambda grouped: grouped.min(),
    "max": lambda grouped: grouped.max(),
    "count": lambda grouped: grouped.count(),
    "nunique": lambda grouped: grouped.nunique(dropna=True),
    "std": lambda grouped: grouped.std(),
}


def _group_series(frame: pd.DataFrame, keys: list[pd.Series], metric: dict[str, str] | None) -> pd.Series:
    """One metric per group (missing keys are excluded); None means the row count."""
    if metric is None:
        return frame.groupby(keys, dropna=True, sort=True, observed=True).size()
    values = _metric_input(frame[metric["column"]], metric["agg"])
    return _GROUP_AGGS[metric["agg"]](values.groupby(keys, dropna=True, sort=True, observed=True))


def _sort_table(
    table: pd.DataFrame, plan: dict[str, Any], default_by: str | None, default_ascending: bool
) -> pd.DataFrame:
    """Apply the plan sort (or the default) with a stable algorithm; missing values go last."""
    sort = plan["sort"]
    by, ascending = (
        (_plan_sort_column(table, plan), sort["ascending"]) if sort else (default_by, default_ascending)
    )
    if by is None or by not in table.columns or table.empty:
        return table
    return table.sort_values(by, ascending=ascending, kind="mergesort", na_position="last").reset_index(
        drop=True
    )


def _plan_sort_column(table: pd.DataFrame, plan: dict[str, Any]) -> str | None:
    """Result column the plan sort applies to (a metric's source column maps to its label), or None."""
    if not plan["sort"]:
        return None
    by = plan["sort"]["by"]
    if by not in table.columns:
        by = next((metric_label(m) for m in plan["metrics"] if m["column"] == by), None)
    return by if by is not None and by in table.columns else None


def _limit_periods(table: pd.DataFrame, plan: dict[str, Any]) -> pd.DataFrame:
    """Apply the plan limit to a period table: the first rows of a plan sort, otherwise the latest periods."""
    if not plan["limit"]:
        return table
    kept = table.head(plan["limit"]) if _plan_sort_column(table, plan) else table.tail(plan["limit"])
    return kept.reset_index(drop=True)


def _exec_aggregate(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    row = {metric_label(m): _aggregate_series(data[m["column"]], m["agg"]) for m in plan["metrics"]}
    facts = [_fact(metric_phrase(m).capitalize(), row[metric_label(m)]) for m in plan["metrics"]]
    facts.append(_fact("Rows analysed", len(data)))
    return pd.DataFrame([row]), facts


def _exec_group(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    group_by = plan["group_by"]
    missing_keys = int(data[group_by].isna().any(axis=1).sum())
    if missing_keys:
        warnings.append(
            f"{missing_keys} row(s) with a missing {', '.join(group_by)} value were left out of the groups."
        )
    keys = [data[g] for g in group_by]
    metrics = _metrics_or_count(plan)
    parts = [_group_series(data, keys, m).rename(_label_for(m)) for m in metrics]
    table = pd.concat(parts, axis=1).reset_index()
    groups = len(table)
    ascending = bool(plan["sort"]["ascending"]) if plan["sort"] else False
    table = _sort_table(table, plan, _label_for(metrics[0]), False)
    ranked = _ranked_by_first_metric(plan, metrics[0])
    tied = _leading_ties(table, _label_for(metrics[0])) if ranked else 1
    _warn_cut_inside_tie(table, plan, metrics, warnings)
    rank_text = "" if table.empty else _group_key_text(table, group_by, tied) + (" (tie)" if tied > 1 else "")
    if plan["limit"]:
        table = table.head(plan["limit"]).reset_index(drop=True)
    facts: list[dict] = []
    if not table.empty:
        first = table.iloc[0]
        key_text = ", ".join(str(json_safe(first[g])) for g in group_by)
        if ranked:
            rank = "Lowest" if ascending else "Highest"
            facts.append(_fact(f"{rank} {' and '.join(group_by)} by {metric_phrase(metrics[0])}", rank_text))
            if tied > 1:
                warnings.append(f"{tied} groups tie for {rank.lower()} {metric_phrase(metrics[0])}.")
        for metric in metrics:
            facts.append(
                _fact(f"{metric_phrase(metric).capitalize()} for {key_text}", first[_label_for(metric)])
            )
    facts.append(_fact("Groups", groups))
    return table, facts


MAX_TIED_NAMES = 5


def _leading_ties(table: pd.DataFrame, label: str) -> int:
    """How many rows at the top of a sorted table share the first row's value (1 when no tie)."""
    values = table[label].tolist() if label in table.columns else []
    if not values or _key_of(values[0]) is None:
        return 1
    count = 1
    while count < len(values) and _same(values[0], values[count]):
        count += 1
    return count


def _group_key_text(table: pd.DataFrame, group_by: list[str], rows: int) -> str:
    """Readable keys of the first ``rows`` groups, for example "A, B and C" or "A, B, ... and 7 more"."""
    keys = [
        ", ".join(str(json_safe(table.iloc[i][g])) for g in group_by)
        for i in range(min(rows, MAX_TIED_NAMES, len(table)))
    ]
    separator = "; " if len(group_by) > 1 else ", "
    if rows > len(keys):
        return f"{separator.join(keys)} and {rows - len(keys)} more"
    return keys[0] if len(keys) == 1 else f"{separator.join(keys[:-1])} and {keys[-1]}"


def _warn_cut_inside_tie(
    table: pd.DataFrame, plan: dict[str, Any], metrics: list[dict[str, str] | None], warnings: list[str]
) -> None:
    """Warn when the limit drops groups whose ranking value equals the last group shown."""
    limit = plan["limit"]
    if not limit or len(table) <= limit:
        return
    label = _plan_sort_column(table, plan) if plan["sort"] else _label_for(metrics[0])
    if label is None or label not in table.columns:
        return
    last, cut = table.iloc[limit - 1][label], table.iloc[limit][label]
    if _key_of(last) is not None and _same(last, cut):
        matches = [m for m in metrics if _label_for(m) == label]
        phrase = metric_phrase(matches[0]) if matches else _display(label)
        warnings.append(
            f"The limit of {limit} groups falls inside a tie: other groups have the same {phrase} "
            "as the last group shown."
        )


def _ranked_by_first_metric(plan: dict[str, Any], metric: dict[str, str] | None) -> bool:
    """True when the first table row is the top (or bottom) group of the first metric."""
    sort = plan["sort"]
    if sort is None:
        return True
    return sort["by"] == _label_for(metric) or (metric is not None and sort["by"] == metric["column"])


def _period_label(period: pd.Period, frequency: str) -> str:
    if frequency == "D":
        return period.strftime("%Y-%m-%d")
    if frequency == "W":
        return period.start_time.strftime("%Y-%m-%d")
    if frequency == "M":
        return period.strftime("%Y-%m")
    if frequency == "Q":
        return f"{period.year}-Q{period.quarter}"
    return f"{period.year}"


def _time_table(data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]) -> pd.DataFrame:
    """Metrics per calendar period, including empty periods between the first and last one."""
    frequency = plan["frequency"]
    stamps = data[plan["time_column"]]
    dropped = int(stamps.isna().sum())
    if dropped:
        warnings.append(
            f"{dropped} row(s) without a {plan['time_column']} value were left out of the time series."
        )
    periods = stamps.dt.to_period(_PERIOD_FREQ[frequency])
    metrics = _metrics_or_count(plan)
    if periods.notna().sum() == 0:
        return pd.DataFrame(columns=["period", *[_label_for(m) for m in metrics]])
    full_range = pd.period_range(periods.min(), periods.max(), freq=_PERIOD_FREQ[frequency])
    columns: dict[str, Any] = {"period": [_period_label(p, frequency) for p in full_range]}
    for metric in metrics:
        grouped = _group_series(data, [periods], metric)
        if metric is None or metric["agg"] in _ZERO_FILLED_AGGS:
            # Periods without rows count as zero; a period whose rows have no values keeps NaN.
            grouped = grouped.reindex(full_range, fill_value=0)
            if metric is None or metric["agg"] != "sum":
                grouped = grouped.astype("int64")
        else:
            grouped = grouped.reindex(full_range)
        columns[_label_for(metric)] = grouped.to_numpy()
    return pd.DataFrame(columns)


def _exec_time_series(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    table = _time_table(data, plan, warnings)
    facts = [_fact("Periods", len(table))]
    if not table.empty:
        metric = _metrics_or_count(plan)[0]
        label = _label_for(metric)
        last = table.iloc[-1]
        facts.append(
            _fact(f"{metric_phrase(metric).capitalize()} in {last['period']} (latest period)", last[label])
        )
        values = pd.to_numeric(table[label], errors="coerce")
        if values.notna().any():
            best = table.iloc[int(np.nanargmax(values.to_numpy(dtype="float64", na_value=np.nan)))]
            facts.append(_fact(f"Highest period for {metric_phrase(metric)}", best["period"]))
    table = _limit_periods(_sort_table(table, plan, None, True), plan)
    return table, facts


def growth_percentages(values: np.ndarray, labels: list[str], warnings: list[str]) -> np.ndarray:
    """Period-over-period percent change; a zero previous value gives NaN plus a warning."""
    growth = np.full(len(values), np.nan)
    zero_periods = []
    for i in range(1, len(values)):
        previous, current = values[i - 1], values[i]
        if np.isnan(previous) or np.isnan(current):
            continue
        if previous == 0:
            zero_periods.append(labels[i])
            continue
        growth[i] = (current - previous) / abs(previous) * 100.0
    if zero_periods:
        warnings.append(f"For {', '.join(zero_periods[:10])}: {GROWTH_ZERO_WARNING}.")
    return growth


def _exec_growth(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    table = _time_table(data, plan, warnings)
    metric = _metrics_or_count(plan)[0]
    label = _label_for(metric)
    values = pd.to_numeric(table[label], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    table["growth_pct"] = growth_percentages(values, list(table["period"]), warnings)
    facts = [_fact("Periods", len(table))]
    known = table[table["growth_pct"].notna()]
    if not known.empty:
        last = known.iloc[-1]
        facts.append(_fact(f"Growth of {metric_phrase(metric)} in {last['period']} (%)", last["growth_pct"]))
        facts.append(
            _fact(f"Average period growth of {metric_phrase(metric)} (%)", float(known["growth_pct"].mean()))
        )
    table = _limit_periods(_sort_table(table, plan, None, True), plan)
    return table, facts


def flag_unusual_changes(changes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust z-score (median/MAD) and 1.5*IQR fences on period-over-period changes.

    Returns (robust_z, flagged). NaN changes are never flagged. With MAD == 0 the
    z-score is undefined (NaN) and only the IQR fence applies.
    """
    valid = ~np.isnan(changes)
    z = np.full(len(changes), np.nan)
    flagged = np.zeros(len(changes), dtype=bool)
    observed = changes[valid]
    if observed.size < MIN_ANOMALY_CHANGES:
        return z, flagged
    median = np.median(observed)
    mad = np.median(np.abs(observed - median))
    if mad > 0:
        z[valid] = 0.6745 * (observed - median) / mad
    q1, q3 = np.percentile(observed, [25, 75])
    low, high = q1 - IQR_FACTOR * (q3 - q1), q3 + IQR_FACTOR * (q3 - q1)
    outside = (changes < low) | (changes > high)
    flagged = valid & ((np.abs(np.nan_to_num(z, nan=0.0)) > ROBUST_Z_THRESHOLD) | outside)
    return z, flagged


def _exec_anomalies(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    table = _time_table(data, plan, warnings)
    metric = _metrics_or_count(plan)[0]
    values = pd.to_numeric(table[_label_for(metric)], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    changes = np.concatenate([[np.nan], np.diff(values)]) if len(values) else np.array([])
    if np.count_nonzero(~np.isnan(changes)) < MIN_ANOMALY_CHANGES:
        warnings.append(
            f"At least {MIN_ANOMALY_CHANGES + 1} periods with values are needed to judge unusual changes; "
            "nothing was flagged."
        )
    z, flagged = flag_unusual_changes(changes)
    table["change"] = changes
    table["robust_z"] = z
    table["flagged"] = flagged
    table["note"] = np.where(flagged, ANOMALY_NOTE, "")
    flagged_periods = list(table.loc[table["flagged"], "period"])
    facts = [_fact("Periods", len(table)), _fact(f"Periods with an {ANOMALY_NOTE}", len(flagged_periods))]
    if flagged_periods:
        facts.append(_fact("Flagged periods", ", ".join(flagged_periods[:10])))
    table = _limit_periods(_sort_table(table, plan, None, True), plan)
    return table, facts


def _target_columns(data: pd.DataFrame, plan: dict[str, Any]) -> list[str]:
    return list(plan["columns"]) or [str(c) for c in data.columns]


def _exec_describe(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    facts = []
    for name in _target_columns(data, plan):
        series = data[name]
        logical = logical_type_for_dtype(series.dtype)
        row: dict[str, Any] = {
            "column": name,
            "type": logical,
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
        }
        if logical in NUMERIC_TYPES:
            values = series.astype("float64")
            row.update(
                mean=json_safe(values.mean()),
                std=json_safe(values.std()),
                min=json_safe(values.min()),
                p25=json_safe(values.quantile(0.25)),
                median=json_safe(values.median()),
                p75=json_safe(values.quantile(0.75)),
                max=json_safe(values.max()),
            )
            if len(facts) < 5:
                facts.append(_fact(f"Average {name}", row["mean"]))
        elif logical == LogicalType.DATETIME.value:
            row.update(min=json_safe(series.min()), max=json_safe(series.max()))
        else:
            counts = series.value_counts(dropna=True)
            row.update(
                unique=int(series.nunique(dropna=True)),
                top=json_safe(counts.index[0]) if len(counts) else None,
            )
        rows.append(row)
    facts.insert(0, _fact("Columns described", len(rows)))
    facts.insert(1, _fact("Rows analysed", len(data)))
    table = _sort_table(pd.DataFrame(rows), plan, None, True)
    return table, facts


def _exec_correlation(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    names = list(plan["columns"]) or [
        c for c in data.columns if logical_type_for_dtype(data[c].dtype) in NUMERIC_TYPES
    ]
    matrix = data[names].astype("float64").corr(method="pearson")
    table = matrix.reset_index(names="column")
    facts = [_fact("Columns compared", len(names))]
    best: tuple[float, str, str] | None = None
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            value = matrix.loc[first, second]
            if not np.isnan(value) and (best is None or abs(value) > abs(best[0])):
                best = (float(value), first, second)
    if best is None:
        warnings.append("No correlation could be calculated (columns are constant or have too few values).")
    else:
        facts.append(_fact(f"Strongest correlation ({best[1]} and {best[2]})", best[0]))
    if matrix.isna().to_numpy().any() and best is not None:
        warnings.append("Some correlations are undefined because a column is constant or has too few values.")
    return table, facts


def _exec_missing(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    names = _target_columns(data, plan)
    total = len(data)
    counts = [int(data[n].isna().sum()) for n in names]
    table = pd.DataFrame(
        {
            "column": names,
            "missing_count": counts,
            "missing_pct": [(c / total * 100.0) if total else 0.0 for c in counts],
        }
    )
    table = _sort_table(table, plan, "missing_count", False)
    if plan["limit"]:
        table = table.head(plan["limit"]).reset_index(drop=True)
    facts = [
        _fact("Columns with missing values", sum(1 for c in counts if c)),
        _fact("Total missing cells", sum(counts)),
        _fact("Rows analysed", total),
    ]
    return table, facts


def _exec_duplicates(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    subset = _target_columns(data, plan)
    involved = data.duplicated(subset=subset, keep=False)
    extra = int(data.duplicated(subset=subset, keep="first").sum())
    groups = data.loc[involved, subset].groupby(subset, dropna=False, sort=True, observed=True).size()
    table = groups.rename("count").reset_index()
    table = _sort_table(table, plan, "count", False)
    if plan["limit"]:
        table = table.head(plan["limit"]).reset_index(drop=True)
    scope = "all columns" if not plan["columns"] else ", ".join(subset)
    facts = [
        _fact(f"Duplicate rows beyond the first occurrence ({scope})", extra),
        _fact("Duplicate groups", len(groups)),
        _fact("Rows involved in duplicates", int(involved.sum())),
    ]
    return table, facts


def histogram_bins(values: np.ndarray) -> int:
    """Sturges' rule, capped at MAX_HISTOGRAM_BINS."""
    return max(1, min(MAX_HISTOGRAM_BINS, int(math.ceil(math.log2(max(values.size, 1)))) + 1))


def _exec_distribution(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    name = plan["columns"][0]
    series = data[name]
    logical = logical_type_for_dtype(series.dtype)
    non_null = series.dropna()
    facts = [_fact(f"Values of {name} counted", len(non_null))]
    if logical in NUMERIC_TYPES:
        values = non_null.astype("float64").to_numpy()
        if values.size == 0:
            return pd.DataFrame(columns=["bin", "bin_start", "bin_end", "count"]), facts
        counts, edges = np.histogram(values, bins=histogram_bins(values))
        table = pd.DataFrame(
            {
                "bin": [
                    f"{format_number(edges[i])} to {format_number(edges[i + 1])}" for i in range(len(counts))
                ],
                "bin_start": edges[:-1],
                "bin_end": edges[1:],
                "count": counts.astype("int64"),
            }
        )
        busiest = table.iloc[int(counts.argmax())]
        facts.append(_fact("Most common range", busiest["bin"]))
        return _sort_table(table, plan, None, True), facts
    counts = non_null.value_counts(dropna=True, sort=False)
    table = pd.DataFrame({"value": counts.index.astype(str), "count": counts.to_numpy(dtype="int64")})
    table["percent"] = table["count"] / max(len(non_null), 1) * 100.0
    table = table.sort_values(["count", "value"], ascending=[False, True], kind="mergesort").reset_index(
        drop=True
    )
    table = _sort_table(table, plan, None, False)
    if plan["limit"]:
        table = table.head(plan["limit"]).reset_index(drop=True)
    facts.append(_fact("Distinct values", len(counts)))
    if len(table):
        facts.append(_fact("Most common value", table.iloc[0]["value"]))
    return table, facts


def _exec_count_rows(
    data: pd.DataFrame, plan: dict[str, Any], warnings: list[str]
) -> tuple[pd.DataFrame, list[dict]]:
    label = "Rows matching the filters" if plan["filters"] else "Rows"
    return pd.DataFrame({"rows": [len(data)]}), [_fact(label, len(data))]


_EXECUTORS: dict[
    str, Callable[[pd.DataFrame, dict[str, Any], list[str]], tuple[pd.DataFrame, list[dict]]]
] = {
    "aggregate": _exec_aggregate,
    "group_aggregate": _exec_group,
    "compare": _exec_group,
    "top_n": _exec_group,
    "time_series": _exec_time_series,
    "growth": _exec_growth,
    "anomalies": _exec_anomalies,
    "describe": _exec_describe,
    "correlation": _exec_correlation,
    "missing_values": _exec_missing,
    "duplicates": _exec_duplicates,
    "distribution": _exec_distribution,
    "count_rows": _exec_count_rows,
}


# ---------------------------------------------------------------------------
# Chart specification
# ---------------------------------------------------------------------------


def _default_chart(plan: dict[str, Any], table: pd.DataFrame) -> dict[str, Any] | None:
    op = plan["operation"]
    first = _label_for(_metrics_or_count(plan)[0])
    if op in TIME_OPERATIONS:
        return {"type": "line", "x": "period", "y": first, "color": None}
    if op in GROUP_OPERATIONS:
        color = plan["group_by"][1] if len(plan["group_by"]) > 1 else None
        return {"type": "bar", "x": plan["group_by"][0], "y": first, "color": color}
    if op == "correlation":
        return {"type": "heatmap", "x": None, "y": None, "color": None}
    if op == "distribution":
        if "bin" in table.columns:
            # The table is already binned (text bin labels with counts), so it is drawn
            # as bars in bin order; a histogram would re-bin the labels.
            return {"type": "bar", "x": "bin", "y": "count", "color": None}
        return {"type": "bar", "x": "value", "y": "count", "color": None}
    if op == "missing_values":
        return {"type": "bar", "x": "column", "y": "missing_count", "color": None}
    return None


def _table_reference(name: str | None, plan: dict[str, Any], table: pd.DataFrame) -> str | None:
    """Map a plan chart reference to a result column (a metric's source column maps to its label)."""
    if name is None or name in table.columns:
        return name
    return next(
        (
            metric_label(m)
            for m in plan["metrics"]
            if m["column"] == name and metric_label(m) in table.columns
        ),
        None,
    )


def build_chart_spec(
    plan: dict[str, Any], table: pd.DataFrame | None, warnings: list[str]
) -> dict[str, Any] | None:
    """Chart spec for the result table: auto selection, manual override and safety fallbacks."""
    requested = plan["chart"] or {"type": "auto", "x": None, "y": None, "color": None, "stacked": False}
    if requested["type"] == "none" or table is None or table.empty:
        return None
    default = _default_chart(plan, table)
    if requested["type"] == "auto":
        spec = default
    elif requested["type"] == "heatmap" and plan["operation"] != "correlation":
        warnings.append("A heatmap needs a correlation matrix; the default chart is shown instead.")
        spec = default
    else:
        spec = {"type": requested["type"]}
        for axis in ("x", "y", "color"):
            mapped = _table_reference(requested[axis], plan, table)
            if requested[axis] is not None and mapped is None:
                warnings.append(
                    f'Chart column "{_display(requested[axis])}" is not in the result; the default was used.'
                )
            spec[axis] = mapped if mapped is not None else (default or {}).get(axis)
        if spec["type"] != "heatmap" and (spec["x"] is None or spec["y"] is None):
            warnings.append(
                "The requested chart needs result columns for both axes; the default chart is shown."
            )
            spec = default
    if spec is None:
        return None
    spec = {**spec, "stacked": bool(requested.get("stacked"))}
    if spec["type"] in {"pie", "donut"}:
        values = pd.to_numeric(table[spec["y"]], errors="coerce") if spec["y"] in table.columns else None
        if values is None or (values < 0).any():
            warnings.append("A pie chart cannot show negative values; a bar chart is shown instead.")
            spec["type"] = "bar"
    spec["title"] = _chart_title(plan, spec)
    return spec


def _human_name(name: object) -> str:
    """Readable name for titles: "sum_revenue" -> "sum revenue"."""
    return " ".join(str(name).replace("_", " ").split())


def _chart_title(plan: dict[str, Any], spec: dict[str, Any]) -> str:
    """Sentence-case title such as "Sum revenue by period"."""
    if spec["type"] == "heatmap":
        return "Correlation matrix"
    title = f"{_human_name(spec['y'])} by {_human_name(spec['x'])}"
    if spec.get("color"):
        title += f" and {_human_name(spec['color'])}"
    return _clean(title[:1].upper() + title[1:], 120)


# ---------------------------------------------------------------------------
# execute_plan
# ---------------------------------------------------------------------------


def _empty_result(plan: dict[str, Any], warnings: list[str], filtered: bool) -> dict[str, Any]:
    label = "Rows matching the filters" if filtered else "Rows in the dataset"
    warnings.append(NO_ROWS_WARNING if filtered else "The dataset has no rows to analyse.")
    return {
        "operation": plan["operation"],
        "table": pd.DataFrame(),
        "truncated": False,
        "total_rows": 0,
        "facts": [_fact(label, 0)],
        "chart": None,
        "warnings": warnings,
        "rows_analysed": 0,
    }


def execute_plan(frame: pd.DataFrame, plan: dict) -> dict:
    """Execute a plan with pure pandas and return the result.

    The plan is (re)validated against the frame's own schema first, so this
    function never runs anything outside the allowlist even when called
    directly. Returns ``{"table", "truncated", "total_rows", "facts", "chart",
    "warnings", "operation", "rows_analysed"}``; the table holds at most
    RESULT_MAX_ROWS rows and ``total_rows`` is its length before that cap.
    """
    plan = validate_plan(plan, frame_schema(frame))
    warnings: list[str] = []
    if plan["operation"] == "clarify":
        return {
            "operation": "clarify",
            "table": None,
            "truncated": False,
            "total_rows": 0,
            "facts": [],
            "chart": None,
            "warnings": warnings,
            "rows_analysed": 0,
        }
    data = frame.loc[_filter_mask(frame, plan["filters"])] if plan["filters"] else frame
    if data.empty and plan["operation"] != "count_rows":
        return _empty_result(plan, warnings, bool(plan["filters"]))
    table, facts = _EXECUTORS[plan["operation"]](data, plan, warnings)
    if plan["filters"] and plan["operation"] != "count_rows":
        facts.append(_fact("Rows matching the filters", len(data)))
    max_rows = get_settings().result_max_rows
    total_rows = len(table)
    if total_rows > max_rows:
        warnings.append(f"Only the first {max_rows} of {total_rows} result rows are shown.")
    table = table.head(max_rows).reset_index(drop=True)
    return {
        "operation": plan["operation"],
        "table": table,
        "truncated": total_rows > max_rows,
        "total_rows": total_rows,
        "facts": facts,
        "chart": build_chart_spec(plan, table, warnings),
        "warnings": warnings,
        "rows_analysed": len(data),
    }


# ---------------------------------------------------------------------------
# Independent verification (numpy over boolean masks)
# ---------------------------------------------------------------------------


@dataclass
class _NpColumn:
    """A column as plain numpy data: kind is numeric, datetime or object."""

    kind: str
    values: np.ndarray
    null: np.ndarray


def _np_column(series: pd.Series) -> _NpColumn:
    logical = logical_type_for_dtype(series.dtype)
    null = series.isna().to_numpy(dtype=bool)
    if logical in NUMERIC_TYPES:
        return _NpColumn("numeric", series.to_numpy(dtype="float64", na_value=np.nan), null)
    if logical == LogicalType.DATETIME.value:
        raw = series.to_numpy(dtype="datetime64[ns]")
        return _NpColumn("datetime", raw.view("int64"), null)
    values = np.array([None if n else v for v, n in zip(series.tolist(), null, strict=True)], dtype=object)
    return _NpColumn("object", values, null)


def _np_literal(value: Any, column: _NpColumn, logical: str) -> Any:
    if column.kind == "numeric":
        return float(value)
    if column.kind == "datetime":
        return np.datetime64(pd.Timestamp(value).to_datetime64(), "ns").view("int64")
    if logical == LogicalType.BOOLEAN.value:
        return bool(value)
    return str(value).strip().casefold()


def _np_text_keys(column: _NpColumn) -> np.ndarray:
    return np.array([None if v is None else str(v).strip().casefold() for v in column.values], dtype=object)


def _np_filter_mask(frame: pd.DataFrame, filters: list[dict[str, Any]]) -> np.ndarray:
    """Independent filter evaluation over numpy arrays (same semantics as ``_filter_mask``)."""
    mask = np.ones(len(frame), dtype=bool)
    for item in filters:
        column = _np_column(frame[item["column"]])
        logical = logical_type_for_dtype(frame[item["column"]].dtype)
        op, value = item["op"], item["value"]
        if op in NULL_FILTER_OPS:
            mask &= column.null if op == "is_null" else ~column.null
            continue
        target = _np_text_keys(column) if logical == LogicalType.TEXT.value else column.values
        mask &= _np_compare(target, column, op, value, logical) & ~column.null
    return mask


_DAY_NS = 86_400 * 10**9


def _np_compare(target: np.ndarray, column: _NpColumn, op: str, value: Any, logical: str) -> np.ndarray:
    if op == "contains":
        needle = _np_literal(value, column, logical)
        return np.array([v is not None and needle in v for v in target], dtype=bool)
    if op in LIST_FILTER_OPS:
        allowed = {_np_literal(v, column, logical) for v in value}
        member = np.array([v in allowed for v in target.tolist()], dtype=bool)
        return member if op == "in" else ~member
    # On a timestamp column with times of day, a midnight bound means the whole calendar day.
    whole_days = column.kind == "datetime" and bool(np.any(column.values[~column.null] % _DAY_NS != 0))
    if op == "between":
        low, high = (_np_literal(v, column, logical) for v in value)
        if whole_days and high % _DAY_NS == 0:
            end = high + _DAY_NS
            return np.array([v is not None and low <= v < end for v in target.tolist()], dtype=bool)
        return np.array([v is not None and low <= v <= high for v in target.tolist()], dtype=bool)
    literal = _np_literal(value, column, logical)
    if whole_days and literal % _DAY_NS == 0 and op in {"eq", "ne", "lte", "gt"}:
        start, end = literal, literal + _DAY_NS
        day_checks = {
            "eq": lambda v: start <= v < end,
            "ne": lambda v: not start <= v < end,
            "lte": lambda v: v < end,
            "gt": lambda v: v >= end,
        }
        day = day_checks[op]
        return np.array([v is not None and day(v) for v in target.tolist()], dtype=bool)
    results = {
        "eq": lambda v: v == literal,
        "ne": lambda v: v != literal,
        "gt": lambda v: v > literal,
        "gte": lambda v: v >= literal,
        "lt": lambda v: v < literal,
        "lte": lambda v: v <= literal,
    }
    check = results[op]
    return np.array([v is not None and bool(check(v)) for v in target.tolist()], dtype=bool)


def _np_aggregate(column: _NpColumn, rows: np.ndarray, agg: str | None) -> Any:
    """Recompute one metric for the selected row indices with numpy only."""
    if agg is None:
        return int(rows.size)
    present = rows[~column.null[rows]]
    if agg == "count":
        return int(present.size)
    values = column.values[present]
    if agg == "nunique":
        return int(len(set(values.tolist())))
    if agg == "sum":
        return float(np.sum(values)) if values.size else None
    if values.size == 0:
        return None
    if column.kind == "datetime" and agg in ORDERED_AGGS:
        return pd.Timestamp(int(values.min() if agg == "min" else values.max()))
    calculations: dict[str, Callable[[], Any]] = {
        "mean": lambda: float(np.mean(values)),
        "median": lambda: float(np.median(values)),
        "min": lambda: float(np.min(values)),
        "max": lambda: float(np.max(values)),
        "std": lambda: float(np.std(values, ddof=1)) if values.size > 1 else None,
    }
    return calculations[agg]()


def _key_of(value: Any) -> Any:
    """Hashable, type-normalized key for numpy values and table cells.

    Timestamps become integer nanoseconds, numpy scalars become Python scalars
    and missing values become None. Equal ints and floats hash alike, so keys
    from Int64 tables match float64 numpy values.
    """
    if isinstance(value, pd.Timestamp):
        return int(value.value)
    return json_safe(value)


def _np_groups(columns: list[_NpColumn], rows: np.ndarray, *, include_null: bool) -> dict[tuple, list[int]]:
    """Row indices per distinct key combination (dict pass; no pandas grouping)."""
    lists = [column.values.tolist() for column in columns]
    nulls = [column.null for column in columns]
    groups: dict[tuple, list[int]] = {}
    for i in rows.tolist():
        if not include_null and any(null[i] for null in nulls):
            continue
        key = tuple(
            None if null[i] else _key_of(values[i]) for values, null in zip(lists, nulls, strict=True)
        )
        groups.setdefault(key, []).append(i)
    return groups


def _same(expected: Any, actual: Any) -> bool:
    expected, actual = _key_of(expected), _key_of(actual)
    if expected is None or actual is None:
        return expected is None and actual is None
    if _is_number(expected) and _is_number(actual):
        if isinstance(expected, int) and isinstance(actual, int):
            return expected == actual
        return bool(np.isclose(float(expected), float(actual), rtol=1e-9, atol=1e-9))
    return expected == actual


class _Checker:
    """Counts comparisons and raises on the first mismatch."""

    def __init__(self) -> None:
        self.checks = 0

    def equal(self, expected: Any, actual: Any, what: str) -> None:
        self.checks += 1
        if not _same(expected, actual):
            raise ResultVerificationError(details={"check": what})


def _cell(table: pd.DataFrame, row: int, column: str) -> Any:
    if column not in table.columns:
        raise ResultVerificationError(details={"check": f"missing column {column}"})
    return table.iloc[row][column]


def _verify_aggregate(frame, plan, result, rows, check) -> None:
    for metric in plan["metrics"]:
        expected = _np_aggregate(_np_column(frame[metric["column"]]), rows, metric["agg"])
        check.equal(expected, _cell(result["table"], 0, metric_label(metric)), metric_label(metric))


def _verify_group(frame, plan, result, rows, check) -> None:
    table = result["table"]
    groups = _np_groups([_np_column(frame[g]) for g in plan["group_by"]], rows, include_null=False)
    metrics = _metrics_or_count(plan)
    columns = {m["column"]: _np_column(frame[m["column"]]) for m in metrics if m is not None}
    expected_total = min(len(groups), plan["limit"]) if plan["limit"] else len(groups)
    check.equal(expected_total, result["total_rows"], "group count")
    for i in range(len(table)):
        key = tuple(_key_of(table.iloc[i][g]) for g in plan["group_by"])
        members = np.array(groups.get(key, []), dtype=np.int64)
        check.equal(True, members.size > 0, "group exists")
        for metric in metrics:
            column = columns[metric["column"]] if metric else None
            expected = _np_aggregate(column, members, metric["agg"]) if metric else int(members.size)
            check.equal(expected, table.iloc[i][_label_for(metric)], f"{_label_for(metric)} for row {i}")
    if plan["sort"] is None and len(table) > 1:
        values = pd.to_numeric(table[_label_for(metrics[0])], errors="coerce").dropna().to_numpy()
        check.equal(True, bool(np.all(np.diff(values) <= 1e-9)), "descending order")


def _np_period_ordinals(column: _NpColumn, frequency: str) -> np.ndarray:
    """Consecutive integer number of the calendar period of each row (numpy datetime units)."""
    stamps = column.values.view("datetime64[ns]")
    if frequency == "D":
        return stamps.astype("datetime64[D]").astype("int64")
    if frequency == "M":
        return stamps.astype("datetime64[M]").astype("int64")
    if frequency == "Y":
        return stamps.astype("datetime64[Y]").astype("int64")
    if frequency == "Q":
        return stamps.astype("datetime64[M]").astype("int64") // 3
    days = stamps.astype("datetime64[D]").astype("int64")
    return (days + 3) // 7  # weeks start on Monday; 1970-01-01 was a Thursday


def _np_ordinal_labels(ordinals: np.ndarray, frequency: str) -> np.ndarray:
    """Calendar period label for each period number from ``_np_period_ordinals``."""
    if frequency == "D":
        labels = np.datetime_as_string(ordinals.astype("datetime64[D]"), unit="D")
    elif frequency == "M":
        labels = np.datetime_as_string(ordinals.astype("datetime64[M]"), unit="M")
    elif frequency == "Y":
        labels = np.datetime_as_string(ordinals.astype("datetime64[Y]"), unit="Y")
    elif frequency == "Q":
        labels = np.array([f"{1970 + q // 4}-Q{q % 4 + 1}" for q in ordinals.tolist()], dtype=object)
    else:
        labels = np.datetime_as_string((ordinals * 7 - 3).astype("datetime64[D]"), unit="D")
    return np.asarray(labels, dtype=object)


def _np_period_table(frame, plan, rows) -> tuple[list[str], dict[str, list[Any]]]:
    """Every calendar period from the first to the last one with data, in order, with its metrics.

    Periods without rows are included (zero for sum, count and nunique), as in the result table.
    """
    time_column = _np_column(frame[plan["time_column"]])
    ordinals = _np_period_ordinals(time_column, plan["frequency"])
    members: dict[int, list[int]] = {}
    for i in rows[~time_column.null[rows]].tolist():
        members.setdefault(int(ordinals[i]), []).append(i)
    if not members:
        return [], {}
    span = np.arange(min(members), max(members) + 1, dtype="int64")
    labels = [str(label) for label in _np_ordinal_labels(span, plan["frequency"])]
    values: dict[str, list[Any]] = {}
    for metric in _metrics_or_count(plan):
        column = _np_column(frame[metric["column"]]) if metric else None
        series = []
        for ordinal in span.tolist():
            indices = np.array(members.get(ordinal, []), dtype=np.int64)
            if metric is None or (indices.size == 0 and metric["agg"] in _ZERO_FILLED_AGGS):
                series.append(int(indices.size))
            else:
                series.append(_np_aggregate(column, indices, metric["agg"]))
        values[_label_for(metric)] = series
    return labels, values


def _np_growth(values: list[Any]) -> list[float | None]:
    """Percent change from each period's calendar predecessor (None when undefined)."""
    numbers = [np.nan if v is None else float(_key_of(v)) for v in values]
    growth: list[float | None] = [None]
    for previous, current in zip(numbers[:-1], numbers[1:], strict=True):
        undefined = np.isnan(previous) or np.isnan(current) or previous == 0
        growth.append(None if undefined else (current - previous) / abs(previous) * 100)
    return growth


def _verify_time(frame, plan, result, rows, check) -> None:
    table = result["table"]
    labels, values = _np_period_table(frame, plan, rows)
    position = {label: j for j, label in enumerate(labels)}
    first = _label_for(_metrics_or_count(plan)[0])
    expected_columns = dict(values)
    if plan["operation"] == "growth" and labels:
        expected_columns["growth_pct"] = _np_growth(values[first])
    if plan["operation"] == "anomalies" and labels:
        expected_columns.update(_np_anomaly_columns(values[first]))
    for i in range(len(table)):
        period = table.iloc[i]["period"]
        check.equal(True, period in position, f"period {period} is in the data range")
        for name, expected in expected_columns.items():
            check.equal(expected[position[period]], _cell(table, i, name), f"{name} for {period}")
    if not plan["limit"] and not result["truncated"]:
        check.equal(True, set(labels) <= set(table["period"]), "every period is shown")
    _verify_kept_periods(plan, result, labels, expected_columns, check)


def _np_anomaly_columns(values: list[Any]) -> dict[str, list[Any]]:
    """Change from the calendar predecessor, robust z-score, flag and note for every period."""
    numbers = np.array([np.nan if v is None else float(_key_of(v)) for v in values], dtype="float64")
    changes = np.concatenate([[np.nan], numbers[1:] - numbers[:-1]])
    z, flagged = flag_unusual_changes(changes)
    return {
        "change": [_key_of(c) for c in changes.tolist()],
        "robust_z": [_key_of(v) for v in z.tolist()],
        "flagged": [bool(f) for f in flagged.tolist()],
        "note": [ANOMALY_NOTE if f else "" for f in flagged.tolist()],
    }


def _np_ranked(keys: list[Any], ascending: bool) -> list[int]:
    """Positions ordered by key (stable), missing keys last: the order of the plan sort."""
    present = [j for j, key in enumerate(keys) if _key_of(key) is not None]
    missing = [j for j, key in enumerate(keys) if _key_of(key) is None]
    present.sort(key=lambda j: _key_of(keys[j]), reverse=not ascending)
    return present + missing


def _verify_kept_periods(plan, result, labels, expected_columns, check) -> None:
    """The rows kept under a limit are the first rows of the plan sort, or the latest periods."""
    table = result["table"]
    if not labels:  # no row has a date: the result must be empty
        check.equal(0, result["total_rows"], "period count")
        check.equal(0, len(table), "periods shown")
        return
    sort_column = _plan_sort_column(table, plan)
    order = list(range(len(labels)))
    keys: list[Any] = labels
    if sort_column is not None:
        keys = labels if sort_column == "period" else expected_columns[sort_column]
        order = _np_ranked(keys, plan["sort"]["ascending"])
        order = order[: plan["limit"]] if plan["limit"] else order
    elif plan["limit"]:
        order = order[-plan["limit"] :]
    check.equal(len(order), result["total_rows"], "period count")
    check.equal(min(len(order), get_settings().result_max_rows), len(table), "periods shown")
    shown = sort_column or "period"
    for i in range(len(table)):
        check.equal(keys[order[i]], _cell(table, i, shown), f"{shown} at row {i}")


def _verify_count(frame, plan, result, rows, check) -> None:
    check.equal(int(rows.size), _cell(result["table"], 0, "rows"), "row count")


def _verify_missing(frame, plan, result, rows, check) -> None:
    table = result["table"]
    for i in range(len(table)):
        column = _np_column(frame[table.iloc[i]["column"]])
        missing = int(column.null[rows].sum())
        check.equal(missing, table.iloc[i]["missing_count"], f"missing count for row {i}")
        check.equal(
            missing / rows.size * 100.0 if rows.size else 0.0, table.iloc[i]["missing_pct"], "missing percent"
        )


def _verify_duplicates(frame, plan, result, rows, check) -> None:
    subset = list(plan["columns"]) or [str(c) for c in frame.columns]
    hashes = hash_rows(frame.iloc[rows][subset]) if rows.size else np.empty(0, dtype=np.uint64)
    _unique, counts = np.unique(hashes, return_counts=True)
    extra = int(rows.size - _unique.size)
    check.equal(extra, result["facts"][0]["value"], "duplicate rows")
    check.equal(int(counts[counts > 1].sum()), result["facts"][2]["value"], "rows involved")
    groups = _np_groups([_np_column(frame[c]) for c in subset], rows, include_null=True)
    table = result["table"]
    for i in range(len(table)):
        key = tuple(_key_of(table.iloc[i][c]) for c in subset)
        check.equal(len(groups.get(key, [])), table.iloc[i]["count"], f"duplicate group {i}")


def _verify_distribution(frame, plan, result, rows, check) -> None:
    column = _np_column(frame[plan["columns"][0]])
    present = rows[~column.null[rows]]
    table = result["table"]
    if "bin_start" in table.columns:
        values = column.values[present]
        top = float(np.max(values)) if values.size else 0.0
        for i in range(len(table)):
            low, high = float(table.iloc[i]["bin_start"]), float(table.iloc[i]["bin_end"])
            last = high >= top
            inside = (values >= low) & ((values <= high) if last else (values < high))
            check.equal(int(inside.sum()), table.iloc[i]["count"], f"bin {i}")
        if not result["truncated"]:
            check.equal(int(present.size), int(table["count"].sum()), "binned values")
        return
    text = [str(v) for v in column.values[present].tolist()]
    tally: dict[str, int] = {}
    for value in text:
        tally[value] = tally.get(value, 0) + 1
    for i in range(len(table)):
        check.equal(tally.get(table.iloc[i]["value"], 0), table.iloc[i]["count"], f"value count {i}")


def _verify_describe(frame, plan, result, rows, check) -> None:
    table = result["table"]
    for i in range(len(table)):
        column = _np_column(frame[table.iloc[i]["column"]])
        check.equal(_np_aggregate(column, rows, "count"), table.iloc[i]["count"], f"count for row {i}")
        if column.kind == "numeric":
            check.equal(_np_aggregate(column, rows, "mean"), table.iloc[i]["mean"], f"mean for row {i}")


def _verify_correlation(frame, plan, result, rows, check) -> None:
    table = result["table"]
    names = [c for c in table.columns if c != "column"]
    for i, first in enumerate(names):
        a = _np_column(frame[first]).values[rows]
        for second in names[i + 1 :]:
            b = _np_column(frame[second]).values[rows]
            both = ~np.isnan(a) & ~np.isnan(b)
            expected = None
            if both.sum() > 1 and np.std(a[both]) > 0 and np.std(b[both]) > 0:
                expected = float(np.corrcoef(a[both], b[both])[0, 1])
            check.equal(
                expected, table.loc[table["column"] == first, second].iloc[0], f"correlation {first}/{second}"
            )


_VERIFIERS: dict[str, Callable[..., None]] = {
    "aggregate": _verify_aggregate,
    "group_aggregate": _verify_group,
    "compare": _verify_group,
    "top_n": _verify_group,
    "time_series": _verify_time,
    "growth": _verify_time,
    "anomalies": _verify_time,
    "count_rows": _verify_count,
    "missing_values": _verify_missing,
    "duplicates": _verify_duplicates,
    "distribution": _verify_distribution,
    "describe": _verify_describe,
    "correlation": _verify_correlation,
}


def verify_result(frame: pd.DataFrame, plan: dict, result: dict) -> dict:
    """Recompute the key numbers of ``result`` with numpy over boolean masks.

    Filters, group keys and calendar periods are evaluated independently of
    the pandas execution path. Any disagreement beyond floating-point tolerance
    raises ResultVerificationError so unverified numbers are never shown.
    Returns ``{"verified": True, "checks": <number of comparisons>}``.
    """
    plan = validate_plan(plan, frame_schema(frame))
    if plan["operation"] == "clarify":
        return {"verified": True, "checks": 0}
    check = _Checker()
    rows = np.flatnonzero(_np_filter_mask(frame, plan["filters"]))
    table = result.get("table")
    if not isinstance(table, pd.DataFrame):
        raise ResultVerificationError(details={"check": "missing table"})
    if rows.size == 0 and plan["operation"] != "count_rows":
        check.equal(0, len(table), "empty result")
        return {"verified": True, "checks": check.checks}
    _VERIFIERS[plan["operation"]](frame, plan, result, rows, check)
    return {"verified": True, "checks": check.checks}


# ---------------------------------------------------------------------------
# Deterministic summary and narrative grounding
# ---------------------------------------------------------------------------

_LEADS = {
    "aggregate": "Here are the calculated totals.",
    "group_aggregate": "Results were calculated per group.",
    "compare": "Here is the comparison across groups.",
    "top_n": "Here are the top groups.",
    "time_series": "Here is how the metric changes over time.",
    "growth": "Here is the period-over-period growth.",
    "anomalies": (
        f"Period-over-period changes were checked; flagged periods are an {ANOMALY_NOTE}, "
        "not a confirmed problem."
    ),
    "describe": "Here are descriptive statistics.",
    "correlation": "Here are the Pearson correlations between numeric columns.",
    "missing_values": "Here are the missing values per column.",
    "duplicates": "Here are the duplicate rows found.",
    "distribution": "Here is how the values are distributed.",
    "count_rows": "Here is the row count.",
}


def build_summary(plan: dict[str, Any], result: dict[str, Any]) -> str:
    """Business-language answer built only from computed facts."""
    if plan["operation"] == "clarify":
        return str(plan["clarification_question"])
    if result.get("rows_analysed") == 0 and plan["operation"] != "count_rows":
        return NO_ROWS_WARNING if plan["filters"] else "The dataset has no rows to analyse."
    lead = _LEADS[plan["operation"]]
    if plan["operation"] == "duplicates" and result.get("total_rows") == 0:
        lead = "No duplicate rows were found."
    parts = [f"{fact['label']}: {format_number(fact['value'])}." for fact in result["facts"]]
    return " ".join([lead, *parts])


# Calendar labels and ISO dates (2024-03, 2024-03-12, 2024-Q1, 2024-03-12T00:00:00) are atomic tokens.
_DATE_PATTERN = r"\d{4}-(?:Q[1-4]|\d{2}(?:-\d{2})?)(?:T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
# Any run of characters that contains a digit, for example 3,800 or 12.5% or 10k or Q3.
_DIGIT_RUN_PATTERN = r"[\w.,%]*\d[\w.,%]*"
_PLAIN_NUMBER_PATTERN = r"(\d{1,3}(,\d{3})+|\d+)(\.\d+)?%?"


def _dates_in(text: str) -> tuple[set[str], str]:
    """Date-like tokens (as text, plus their date part) and the text with them blanked out."""
    found: set[str] = set()
    for match in re.finditer(_DATE_PATTERN, text):
        found.add(match.group(0))
        found.add(match.group(0).split("T")[0])
    return found, re.sub(_DATE_PATTERN, " ", text)


def _numbers_in(text: str) -> list[tuple[float, int]] | None:
    """Numbers in free text as (value, decimals), ignoring thousands separators.

    Returns None when a digit appears in a form that cannot be checked
    exactly, such as 10k, 1.5x or Q3 (the caller then treats the text as
    ungrounded). Date-like tokens must be removed first.
    """
    found = []
    for match in re.finditer(_DIGIT_RUN_PATTERN, text):
        token = match.group(0).strip(".,")
        if not re.fullmatch(_PLAIN_NUMBER_PATTERN, token):
            return None
        number = token.rstrip("%").replace(",", "")
        decimals = len(number.split(".")[1]) if "." in number else 0
        found.append((float(number), decimals))
    return found


def _grounding_numbers(
    question: str, plan: dict[str, Any], result: dict[str, Any]
) -> tuple[list[float], set[str]]:
    """Numbers and date tokens the narrative may legitimately mention.

    Numbers are absolute values from facts, table cells, labels and the
    question, plus percent forms of fractions and the years of result dates.
    Digits inside date labels are never harvested as separate numbers.
    """
    candidates: list[Any] = [result.get("total_rows"), result.get("rows_analysed"), plan.get("limit")]
    texts = [question]
    for fact in result.get("facts", []):
        candidates.append(fact.get("value"))
        texts.extend([str(fact.get("label", "")), str(fact.get("value", ""))])
    table = result.get("table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        texts.extend(str(c) for c in table.columns)
        for row in table.itertuples(index=False):
            for value in row:
                value = json_safe(value)
                if _is_number(value):
                    candidates.append(value)
                elif isinstance(value, str):
                    texts.append(value)
    numbers = [abs(float(v)) for v in candidates if _is_number(v) and math.isfinite(float(v))]
    dates: set[str] = set()
    for text in texts:
        text_dates, remainder = _dates_in(text)
        dates |= text_dates
        numbers.extend(value for value, _ in _numbers_in(remainder) or [])
    years = {float(d[:4]) for d in dates}
    return numbers + [n * 100.0 for n in numbers if n <= 1.0] + sorted(years), dates


def narrative_is_grounded(
    narrative: str, question: str, plan: dict[str, Any], result: dict[str, Any]
) -> bool:
    """True when every number in ``narrative`` matches a computed number.

    A narrative number with ``d`` decimals matches when it equals a computed
    value rounded to ``d`` decimals (so 0 to 2 decimal roundings, thousands
    separators and percentages are accepted). Dates must match result dates
    exactly. Numbers that cannot be checked (10k, 1.5x) reject the text.
    """
    allowed, allowed_dates = _grounding_numbers(question, plan, result)
    narrative_dates, remainder = _dates_in(narrative)
    if not narrative_dates <= allowed_dates:
        return False
    numbers = _numbers_in(remainder)
    if numbers is None:
        return False
    for value, decimals in numbers:
        tolerance = 0.5 * 10 ** (-decimals) + 1e-9
        if not any(abs(value - candidate) <= tolerance for candidate in allowed):
            return False
    return True


# ---------------------------------------------------------------------------
# Schema summary
# ---------------------------------------------------------------------------


@dataclass
class _ColumnSummary:
    """Accumulates bounded per-column facts over parquet batches."""

    name: str
    logical: str
    missing: int = 0
    minimum: Any = None
    maximum: Any = None
    counts: dict[str, int] = field(default_factory=dict)
    overflow: bool = False

    def add(self, series: pd.Series) -> None:
        self.missing += int(series.isna().sum())
        present = series.dropna()
        if present.empty:
            return
        if self.logical in ORDERED_TYPES:
            low, high = present.min(), present.max()
            self.minimum = low if self.minimum is None else min(self.minimum, low)
            self.maximum = high if self.maximum is None else max(self.maximum, high)
        elif not self.overflow:
            for value, count in present.astype(str).value_counts().items():
                self.counts[value] = self.counts.get(value, 0) + int(count)
            self.overflow = len(self.counts) > SUMMARY_DISTINCT_TRACK
            if self.overflow:
                self.counts.clear()

    def semantic(self, rows: int) -> str:
        non_null = rows - self.missing
        if non_null <= 0:
            return "empty"
        if self.logical == LogicalType.BOOLEAN.value:
            return "boolean"
        if self.logical == LogicalType.DATETIME.value:
            return "datetime"
        id_like = re.search(_IDENTIFIER_NAME, self.name) is not None
        if self.logical in NUMERIC_TYPES:
            return "identifier" if id_like and self.logical == LogicalType.INTEGER.value else "numeric"
        distinct = len(self.counts)
        if id_like and (self.overflow or distinct >= IDENTIFIER_RATIO * non_null):
            return "identifier"
        if not self.overflow and (
            distinct <= CATEGORICAL_DISTINCT or distinct <= CATEGORICAL_RATIO * non_null
        ):
            return "categorical"
        return "text"

    def result(self, rows: int) -> dict[str, Any]:
        semantic = self.semantic(rows)
        entry: dict[str, Any] = {
            "name": self.name,
            "logical_type": self.logical,
            "semantic": semantic,
            "missing_pct": round(self.missing / rows * 100.0, 2) if rows else 0.0,
        }
        if self.logical in ORDERED_TYPES and self.minimum is not None:
            entry["min"], entry["max"] = json_safe(self.minimum), json_safe(self.maximum)
        if (
            semantic in {"categorical", "boolean"}
            and self.counts
            and len(self.counts) <= CATEGORICAL_DISTINCT
        ):
            ranked = sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))
            entry["sample_values"] = [
                _clean(v, SUMMARY_SAMPLE_CHARS) for v, _ in ranked[:SUMMARY_SAMPLE_VALUES]
            ]
        return entry


def get_schema_summary(ctx: AuthContext, dataset_id: int, *, version_id: int | None = None) -> dict:
    """Bounded, sanitized schema description for planning and the UI.

    Per column (at most MAX_SUMMARY_COLUMNS): name, logical_type, semantic hint,
    missing_pct, min/max for numeric and date columns, and up to five short
    sample values (40 characters, control characters removed) for
    low-cardinality text columns only. Identifiers and free text never get
    samples. Reads the data in bounded batches. Requires dataset access.
    """
    ref = versioning_service.get_version_ref(ctx, dataset_id, version_id)
    summaries = [
        _ColumnSummary(str(c["name"]), str(c["logical_type"])) for c in ref.schema[:MAX_SUMMARY_COLUMNS]
    ]
    rows = 0
    if summaries:
        batches = versioning_service.iter_frame_batches(
            ctx, dataset_id, version_id=ref.version_id, columns=[s.name for s in summaries]
        )
        for batch in batches:
            rows += len(batch)
            for summary in summaries:
                summary.add(batch[summary.name])
    return {
        "dataset_id": ref.dataset_id,
        "dataset_name": _clean(ref.dataset_name, 120),
        "version_id": ref.version_id,
        "version_number": ref.version_number,
        "rows": rows,
        "column_count": ref.columns,
        "columns": [s.result(rows) for s in summaries],
        "columns_truncated": len(ref.schema) > MAX_SUMMARY_COLUMNS,
    }


# ---------------------------------------------------------------------------
# Rule-based interpreter (deterministic fallback)
# ---------------------------------------------------------------------------

SYNONYM_GROUPS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales", "amount", "turnover"),
    "profit": ("profit", "margin", "earnings"),
    "product": ("product", "item", "sku"),
    "region": ("region", "area", "territory", "zone"),
    "category": ("category", "segment", "type"),
    "customer": ("customer", "client"),
    "date": ("order_date", "date", "day", "month"),
}
MEASURE_GROUPS = ("revenue", "profit")
DIMENSION_GROUPS = ("product", "region", "category", "customer")
_FREQUENCY_WORDS = {
    "daily": "D",
    "day": "D",
    "weekly": "W",
    "week": "W",
    "monthly": "M",
    "month": "M",
    "quarterly": "Q",
    "quarter": "Q",
    "yearly": "Y",
    "annual": "Y",
    "annually": "Y",
    "year": "Y",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _word_forms(word: str) -> list[str]:
    forms = [word]
    if word.endswith("ies") and len(word) > 4:
        forms.append(word[:-3] + "y")
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        forms.append(word[:-1])
    return forms


def _columns_of(schema: list[dict[str, Any]], kinds: frozenset[str] | set[str]) -> list[tuple[str, str]]:
    return [
        (str(c["name"]), _normalize_name(str(c["name"]))) for c in schema if c.get("logical_type") in kinds
    ]


def _match_word(word: str, columns: list[tuple[str, str]], groups: tuple[str, ...]) -> str | None:
    """Column for a question word: exact normalized name first, then its synonym group."""
    for form in _word_forms(word):
        for name, normalized in columns:
            if normalized == form:
                return name
        for group in groups:
            synonyms = SYNONYM_GROUPS[group]
            if form not in synonyms:
                continue
            for synonym in synonyms:
                for name, normalized in columns:
                    if normalized == synonym:
                        return name
            for synonym in synonyms:
                for name, normalized in columns:
                    if synonym in normalized.split("_"):
                        return name
    return None


def _find_in_question(
    words: list[str], columns: list[tuple[str, str]], groups: tuple[str, ...], exclude: set[str]
) -> str | None:
    for word in words:
        found = _match_word(word, columns, groups)
        if found is not None and found not in exclude:
            return found
    return None


def _date_column(words: list[str], schema: list[dict[str, Any]]) -> str | None:
    columns = _columns_of(schema, {LogicalType.DATETIME.value})
    if len(columns) == 1:
        return columns[0][0]
    found = _find_in_question(words, columns, ("date",), set())
    if found is not None:
        return found
    for synonym in SYNONYM_GROUPS["date"]:
        for name, normalized in columns:
            if normalized == synonym or synonym in normalized.split("_"):
                return name
    return None


def _frequency(text: str, words: list[str]) -> str:
    match = re.search(r"\b(day|week|month|quarter|year)[\s-]+over[\s-]+\1\b", text)
    if match:
        return _FREQUENCY_WORDS[match.group(1)]
    if re.search(r"\b(mom|m-o-m)\b", text):
        return "M"
    if re.search(r"\b(yoy|y-o-y)\b", text):
        return "Y"
    return next((_FREQUENCY_WORDS[w] for w in words if w in _FREQUENCY_WORDS), DEFAULT_FREQUENCY)


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def rule_based_plan(question: str, schema: list[dict]) -> dict | None:
    """Deterministic interpreter for common business questions (spec section 13).

    Matches columns by normalized names and synonym groups (sales, revenue,
    amount and turnover; profit, margin and earnings; product, item and sku;
    region, area, territory and zone; category, segment and type; customer and
    client; date, day, order_date and month). Returns a validated plan, or
    None when the question is not recognised or a needed column does not exist.
    It never invents columns.
    """
    if not isinstance(question, str) or not schema:
        return None
    text = _clean(question, MAX_QUESTION_CHARS).lower()
    words = re.findall(r"[a-z0-9]+", text)
    numeric = _columns_of(schema, NUMERIC_TYPES)
    categorical = _columns_of(schema, {LogicalType.TEXT.value, LogicalType.BOOLEAN.value})
    measure = _find_in_question(words, numeric, MEASURE_GROUPS, set())
    dimension = _find_in_question(words, categorical, DIMENSION_GROUPS, set())
    plan = _interpret(text, words, schema, measure, dimension)
    if plan is None:
        return None
    try:
        return validate_plan(plan, schema)
    except PlanValidationError:
        return None


def _interpret(
    text: str, words: list[str], schema: list[dict], measure: str | None, dimension: str | None
) -> dict | None:
    if _has(text, r"\b(missing|null|blank|empty)\b"):
        return {"operation": "missing_values"}
    if _has(text, r"\bduplicat"):
        return {"operation": "duplicates"}
    if _has(text, r"\b(correlat|relationship)"):
        return {"operation": "correlation"}
    time_plan = _interpret_time(text, words, schema, measure)
    if time_plan is not None or _has(
        text, r"\b(growth|over time|trend|trends|unusual|anomal|spike|abnormal)"
    ):
        return time_plan
    return _interpret_groups(text, words, measure, dimension)


def _interpret_time(text: str, words: list[str], schema: list[dict], measure: str | None) -> dict | None:
    growth = _has(text, r"\b(growth|grow|(day|week|month|quarter|year)[\s-]+over[\s-]+\2|mom|yoy)\b")
    unusual = _has(text, r"\b(unusual|anomal\w*|spikes?|abnormal|sudden)\b")
    trend = _has(
        text,
        r"\b(trends?|over time|daily|weekly|monthly|quarterly|yearly|per (day|week|month|quarter|year))\b",
    )
    if not (growth or unusual or trend):
        return None
    date = _date_column(words, schema)
    if date is None:
        return None
    if (growth or unusual) and measure is None:
        measure = _default_measure(schema)
    metrics = [{"column": measure, "agg": "sum"}] if measure else []
    operation = "anomalies" if unusual else "growth" if growth else "time_series"
    return {
        "operation": operation,
        "metrics": metrics,
        "time_column": date,
        "frequency": _frequency(text, words),
    }


def _default_measure(schema: list[dict]) -> str | None:
    """The main business amount when the question names none: a revenue-like, then profit-like column."""
    numeric = _columns_of(schema, NUMERIC_TYPES)
    for group in MEASURE_GROUPS:
        for synonym in SYNONYM_GROUPS[group]:
            found = _match_word(synonym, numeric, ())
            if found is not None:
                return found
    return None


def _aggregation_word(text: str) -> str:
    if _has(text, r"\b(average|avg|mean)\b"):
        return "mean"
    if _has(text, r"\bmedian\b"):
        return "median"
    return "sum"


def _interpret_groups(text: str, words: list[str], measure: str | None, dimension: str | None) -> dict | None:
    agg = _aggregation_word(text)
    metrics = [{"column": measure, "agg": agg}] if measure else []
    top = re.search(r"\b(top|bottom)\s+(\d{1,4})\b", text)
    superlative = _has(
        text,
        r"\b(highest|most|largest|biggest|best|top|maximum|lowest|least|smallest|worst|minimum|bottom)\b",
    )
    ascending = _has(text, r"\b(lowest|least|smallest|worst|minimum|bottom)\b")
    if dimension and (top or superlative):
        if not measure and not top:
            return None
        limit = min(int(top.group(2)), get_settings().result_max_rows) if top else DEFAULT_TOP_N
        sort = {"by": metric_label(metrics[0]) if metrics else "row_count", "ascending": ascending}
        return {
            "operation": "top_n",
            "metrics": metrics,
            "group_by": [dimension],
            "limit": max(limit, 1),
            "sort": sort,
        }
    if dimension and measure and _has(text, r"\b(by|per|each|across|compare|comparison)\b"):
        operation = "compare" if _has(text, r"\bcompar") else "group_aggregate"
        return {"operation": operation, "metrics": metrics, "group_by": [dimension]}
    if measure and _has(text, r"\b(total|sum|overall|average|avg|mean|median|how much)\b"):
        return {"operation": "aggregate", "metrics": metrics}
    if _has(text, r"\b(how many|number of|count)\b.*\b(rows|records|orders|entries|transactions)\b"):
        return {"operation": "count_rows"}
    if _has(text, r"\b(describe|summary|statistics|stats)\b"):
        return {"operation": "describe"}
    return None


# ---------------------------------------------------------------------------
# Orchestration: answer_question and run_plan
# ---------------------------------------------------------------------------


@dataclass
class _Planning:
    """Outcome of the planning stage."""

    plan: dict[str, Any] | None = None
    planner: str | None = None
    model: str | None = None
    status: str | None = None
    message: str | None = None
    notes: list[str] = field(default_factory=list)
    llm_calls: int = 0


@dataclass
class _Outcome:
    """Everything needed for the response and the history row."""

    status: str
    planning: _Planning
    answer_text: str = ""
    narrative: str | None = None
    result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None


def clean_question(question: Any) -> str:
    """Strip control characters and validate the length (1 to 1000 characters)."""
    if not isinstance(question, str):
        raise ValidationError("Please type a question.")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", "", question).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        raise ValidationError("Please type a question.")
    if len(text) > MAX_QUESTION_CHARS:
        raise ValidationError(f"Questions may be at most {MAX_QUESTION_CHARS} characters long.")
    return text


def _authorize_analysis(ctx: AuthContext, dataset_id: int, input_mode: str) -> None:
    """Dataset access, text-analysis feature, voice rights when needed and the monthly quota."""
    with session_scope() as db:
        authorize_dataset(db, ctx, dataset_id, feature=Feature.TEXT_AI_ANALYSIS)
        if input_mode == "voice":
            authorize(db, ctx, permission=Permission.USE_VOICE, feature=Feature.VOICE_ANALYSIS)
        usage_tracking.enforce_monthly_analysis_quota(db, ctx.organization_id)


def _require_plan_features(ctx: AuthContext, plan: dict[str, Any]) -> None:
    with session_scope() as db:
        for feature in required_features(plan):
            authorize(db, ctx, feature=feature)


def _validate_mode(input_mode: str, allowed: tuple[str, ...]) -> str:
    if input_mode not in allowed:
        raise ValidationError(f"Input mode must be one of: {', '.join(allowed)}.")
    return input_mode


def _validate_chart_override(chart_type: str | None) -> None:
    if chart_type is not None and chart_type not in CHART_TYPES:
        raise ValidationError(f"Chart type must be one of: {', '.join(CHART_TYPES)}.")


def _rules_or(
    question: str, schema: list[dict[str, Any]], planning: _Planning, note: str, failure: _Planning
) -> _Planning:
    plan = rule_based_plan(question, schema) if get_settings().ai_fallback_mode == "rules" else None
    if plan is None:
        return failure
    planning.plan, planning.planner, planning.model = plan, "rules", None
    planning.notes.append(note)
    dropped = _dropped_dimension(question, schema, plan)
    if dropped is not None:
        planning.notes.append(
            f'Results over time cannot be split by "{_display(dropped)}" yet; '
            f"the total over all {_display(dropped)} values is shown."
        )
    return planning


def _dropped_dimension(question: str, schema: list[dict[str, Any]], plan: dict[str, Any]) -> str | None:
    """A category the question names that a rule-based plan over time leaves out (no per-category series)."""
    if plan["operation"] not in TIME_OPERATIONS:
        return None
    words = re.findall(r"[a-z0-9]+", _clean(question, MAX_QUESTION_CHARS).lower())
    categorical = _columns_of(schema, {LogicalType.TEXT.value, LogicalType.BOOLEAN.value})
    return _find_in_question(words, categorical, DIMENSION_GROUPS, set())


def _plan_question(question: str, summary: dict[str, Any], schema: list[dict[str, Any]]) -> _Planning:
    """LLM plan with one corrective retry, then the labelled rule-based fallback."""
    planning = _Planning()
    feedback: str | None = None
    for _attempt in range(2):
        try:
            raw, model = llm_service.generate_plan(question, summary, feedback=feedback)
            planning.llm_calls += 1
        except AIBusyError as exc:
            planning.status, planning.message = "error", exc.user_message
            return planning
        except LLMUnavailableError as exc:
            return _unavailable(question, schema, planning, exc)
        except LLMResponseError as exc:
            planning.llm_calls += 1
            feedback = exc.user_message
            continue
        try:
            planning.plan, planning.planner, planning.model = validate_plan(raw, schema), "llm", model
            return planning
        except PlanValidationError as exc:
            feedback = exc.user_message
    rejected = _Planning(
        status="rejected",
        message=f"The question could not be turned into a safe analysis plan: {feedback}",
        llm_calls=planning.llm_calls,
    )
    note = (
        f"The AI model's plan was rejected ({feedback}); answered with the built-in rule-based interpreter."
    )
    return _rules_or(question, schema, planning, note, rejected)


def _unavailable(
    question: str, schema: list[dict[str, Any]], planning: _Planning, exc: LLMUnavailableError
) -> _Planning:
    if get_settings().ai_fallback_mode != "rules":
        planning.status, planning.message = "error", exc.user_message
        return planning
    clarification = _Planning(
        status="clarification",
        message=(
            "The local AI model is unavailable and the built-in interpreter did not recognise this question. "
            'Try a question such as "total revenue", "show monthly revenue trends" or '
            '"which columns contain missing values".'
        ),
        llm_calls=planning.llm_calls,
    )
    return _rules_or(question, schema, planning, FALLBACK_NOTE, clarification)


def _with_chart_override(
    plan: dict[str, Any], chart_type: str | None, schema: list[dict[str, Any]]
) -> dict[str, Any]:
    if chart_type is None or plan["operation"] == "clarify":
        return plan
    chart = dict(plan["chart"] or {"x": None, "y": None, "color": None, "stacked": False})
    chart["type"] = chart_type
    return validate_plan({**plan, "chart": chart}, schema)


def _narrative(question: str, plan: dict[str, Any], result: dict[str, Any], outcome: _Outcome) -> str | None:
    """Optional model narrative, kept only when every number in it is grounded."""
    table = result["table"]
    sample = records(table, NARRATIVE_ROWS) if isinstance(table, pd.DataFrame) else []
    try:
        text = llm_service.generate_explanation(question, plan, result["facts"], sample)
        outcome.planning.llm_calls += 1
    except LLMResponseError:
        outcome.planning.llm_calls += 1
        outcome.warnings.append(NARRATIVE_UNAVAILABLE)
        return None
    except (LLMUnavailableError, AIBusyError):
        outcome.warnings.append(NARRATIVE_UNAVAILABLE)
        return None
    if not narrative_is_grounded(text, question, plan, result):
        outcome.warnings.append(NARRATIVE_DISCARDED)
        return None
    return text


NO_CHART = {"type": "none", "x": None, "y": None, "color": None, "stacked": False}


def _charts_enabled(ctx: AuthContext) -> bool:
    """Whether the organization's plan includes standard charts (results carry no chart otherwise)."""
    with session_scope() as db:
        return subscriptions.org_has_feature(db, ctx.organization_id, Feature.STANDARD_CHARTS)


def _execute(
    ctx: AuthContext,
    dataset_id: int,
    version_id: int,
    question: str,
    planning: _Planning,
    schema: list[dict[str, Any]],
) -> _Outcome:
    """Load only the needed columns, execute, verify and explain a validated plan."""
    plan = planning.plan
    outcome = _Outcome(status="success", planning=planning, warnings=list(planning.notes))
    if plan["operation"] == "clarify":
        outcome.status, outcome.answer_text = "clarification", str(plan["clarification_question"])
        return outcome
    try:
        frame = versioning_service.load_frame(
            ctx,
            dataset_id,
            version_id=version_id,
            columns=columns_needed(plan, schema),
            feature=Feature.TEXT_AI_ANALYSIS,
        )
        run = plan if _charts_enabled(ctx) else {**plan, "chart": {**NO_CHART}}
        result = execute_plan(frame, run)
        outcome.verification = verify_result(frame, run, result)
    except AppError as exc:
        if isinstance(exc, ResultVerificationError):
            record_error("analytics_engine.verify", exc, ctx=ctx)
        outcome.status, outcome.error_message = "error", exc.user_message
        outcome.answer_text = exc.user_message
        return outcome
    except Exception as exc:  # noqa: BLE001 - never show a stack trace; keep a reference instead
        reference = record_error("analytics_engine.execute", exc, ctx=ctx)
        message = f"The analysis could not be completed (reference {reference})."
        outcome.status, outcome.error_message, outcome.answer_text = "error", message, message
        return outcome
    outcome.result = result
    outcome.warnings.extend(result["warnings"])
    outcome.answer_text = build_summary(plan, result)
    settings = get_settings()
    if (
        planning.planner == "llm"
        and settings.ai_narrative_enabled
        and llm_service.narrative_allowed()
        and result["rows_analysed"]
    ):
        outcome.narrative = _narrative(question, plan, result, outcome)
    return outcome


def preview_payload(outcome: _Outcome) -> dict[str, Any]:
    """The documented ``result_preview_json`` object."""
    result = outcome.result or {}
    table = result.get("table")
    columns: list[str] = []
    rows: list[list[Any]] = []
    if isinstance(table, pd.DataFrame):
        limited = table.head(get_settings().result_max_rows)
        columns = [str(c) for c in limited.columns]
        rows = [[json_safe(v) for v in row] for row in limited.itertuples(index=False)]
    return {
        "columns": columns,
        "rows": rows,
        "truncated": bool(result.get("truncated", False)),
        "total_rows": int(result.get("total_rows", 0)),
        "chart": result.get("chart"),
        "facts": result.get("facts", []),
        "warnings": outcome.warnings,
    }


def _record(
    ctx: AuthContext,
    dataset_id: int,
    version_id: int | None,
    question: str,
    input_mode: str,
    outcome: _Outcome,
    started: float,
) -> int:
    """Write the history row and usage records in their own transaction."""
    planning = outcome.planning
    chart = (outcome.result or {}).get("chart")
    with session_scope() as db:
        entry = AnalysisHistory(
            organization_id=ctx.organization_id,
            user_id=ctx.user_id,
            dataset_id=dataset_id,
            version_id=version_id,
            question=question,
            input_mode=input_mode,
            planner=planning.planner,
            plan_json=json.dumps(planning.plan, default=str) if planning.plan else None,
            status=outcome.status,
            answer_text=outcome.answer_text,
            result_preview_json=json.dumps(preview_payload(outcome), default=str),
            chart_type=chart["type"] if chart else None,
            error_message=outcome.error_message,
            duration_ms=int((time.perf_counter() - started) * 1000),
            model_name=planning.model,
        )
        db.add(entry)
        usage_tracking.record_usage(
            db, ctx.organization_id, ctx.user_id, UsageMetric.ANALYSIS_QUERY, 1, {"dataset_id": dataset_id}
        )
        if input_mode == "voice":
            usage_tracking.record_usage(
                db, ctx.organization_id, ctx.user_id, UsageMetric.VOICE_QUERY, 1, {"dataset_id": dataset_id}
            )
        if planning.llm_calls:
            usage_tracking.record_usage(
                db, ctx.organization_id, ctx.user_id, UsageMetric.LLM_CALL, planning.llm_calls
            )
        if outcome.status == "rejected":
            record_audit(
                db,
                ctx,
                "analysis.plan_rejected",
                target_type="dataset",
                target_id=dataset_id,
                details={"reason": outcome.error_message, "input_mode": input_mode},
            )
        db.flush()
        return entry.id


def _response(outcome: _Outcome, history_id: int) -> dict[str, Any]:
    result = outcome.result or {}
    return {
        "status": outcome.status,
        "answer_text": outcome.answer_text,
        "narrative": outcome.narrative,
        "facts": result.get("facts", []),
        "table": result.get("table"),
        "chart": result.get("chart"),
        "plan": outcome.planning.plan,
        "planner": outcome.planning.planner,
        "model": outcome.planning.model,
        "warnings": outcome.warnings,
        "verification": outcome.verification,
        "history_id": history_id,
        "truncated": bool(result.get("truncated", False)),
        "total_rows": int(result.get("total_rows", 0)),
    }


def _planning_failure(planning: _Planning) -> _Outcome:
    message = planning.message or "The question could not be answered."
    error = None if planning.status == "clarification" else message
    return _Outcome(
        status=planning.status or "error", planning=planning, answer_text=message, error_message=error
    )


def answer_question(
    ctx: AuthContext,
    dataset_id: int,
    question: str,
    *,
    input_mode: str = "text",
    version_id: int | None = None,
    chart_type: str | None = None,
) -> dict:
    """Answer a natural-language question about an authorized dataset.

    Raises for authorization, plan features, quota and invalid input (nothing
    is recorded then). Every planned question is recorded in the analysis
    history with status success, clarification, error or rejected, and the
    response always has the same keys (see the module docstring).
    """
    started = time.perf_counter()
    text = clean_question(question)
    mode = _validate_mode(input_mode, ("text", "voice"))
    _validate_chart_override(chart_type)
    _authorize_analysis(ctx, dataset_id, mode)
    ref = versioning_service.get_version_ref(ctx, dataset_id, version_id, feature=Feature.TEXT_AI_ANALYSIS)
    summary = get_schema_summary(ctx, dataset_id, version_id=ref.version_id)
    planning = _plan_question(text, summary, ref.schema)
    if planning.plan is None:
        outcome = _planning_failure(planning)
    else:
        planning.plan = _with_chart_override(planning.plan, chart_type, ref.schema)
        _require_plan_features(ctx, planning.plan)
        outcome = _execute(ctx, dataset_id, ref.version_id, text, planning, ref.schema)
    history_id = _record(ctx, dataset_id, ref.version_id, text, mode, outcome, started)
    return _response(outcome, history_id)


def run_plan(ctx: AuthContext, dataset_id: int, plan: dict, *, input_mode: str = "builder") -> dict:
    """Run a plan from the manual query builder (no LLM) with the same safety pipeline.

    An invalid plan is recorded and returned with status "rejected" and the
    validation reason; everything else behaves like ``answer_question``.
    """
    started = time.perf_counter()
    mode = _validate_mode(input_mode, ("builder", "text", "voice"))
    _authorize_analysis(ctx, dataset_id, mode)
    ref = versioning_service.get_version_ref(ctx, dataset_id, feature=Feature.TEXT_AI_ANALYSIS)
    operation = plan.get("operation") if isinstance(plan, dict) else None
    question = (
        f"Query builder: {_clean(operation, 40)}" if isinstance(operation, str) else "Query builder plan"
    )
    try:
        planning = _Planning(plan=validate_plan(plan, ref.schema), planner="builder")
    except PlanValidationError as exc:
        planning = _Planning(status="rejected", message=exc.user_message)
    if planning.plan is None:
        outcome = _planning_failure(planning)
    else:
        _require_plan_features(ctx, planning.plan)
        outcome = _execute(ctx, dataset_id, ref.version_id, question, planning, ref.schema)
    history_id = _record(ctx, dataset_id, ref.version_id, question, mode, outcome, started)
    return _response(outcome, history_id)
