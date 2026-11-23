"""Dataset profiling, the automated data health check and compact AI column summaries.

Statistics are EXACT and memory is bounded:

* Per-column statistics read the version file in column batches (at most
  PROFILE_BATCH_COLUMNS columns, PROFILE_BATCH_CELLS cells and PROFILE_BATCH_BYTES
  of uncompressed data, or one column when a single column is larger), each
  memory checked. A batch that does not fit the memory budget is read one column
  at a time, so batching never refuses a dataset whose single columns fit. Peak
  memory is one batch, never the whole dataset. Access is authorized once per
  profile and the file is opened once per batch, so the work grows linearly with
  width.
* Whole-row work (missing cells, exact duplicate rows, quality score) iterates
  Parquet batches through ``compute_quality_batches``; duplicate detection uses
  64-bit row hashes across all batches, so the result never depends on the
  batch size.

Findings are split into two lists that are never mixed:

* ``confirmed_issues`` are definite data problems: missing values, exact
  duplicate rows, invalid values in mostly-numeric or mostly-date text columns,
  inconsistent date formats, leading/trailing whitespace, empty columns, values
  that could not be converted (mixed currencies) and rows the import skipped
  (malformed or blank rows, from the import report).
* ``suspected_issues`` need a human decision: statistical outliers, inconsistent
  category spellings, possibly duplicated business keys, constant columns and
  very high missing percentages. A legitimate outlier or a repeated category is
  never reported as a confirmed problem.

Conversion problems are detected live on the profiled version with the same
rules the importer uses, so a column fixed by a later cleaning step is never
reported from a stale import report. Import-only facts (skipped rows) carry
``"source": "import"``; everything else carries ``"source": "data"``.

Results are cached in a bounded in-process LRU keyed by the immutable version
(organization, version id and file). Authorization always runs before the cache
is consulted, and callers receive deep copies.
"""

from __future__ import annotations

import copy
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from auth.authorization import AuthContext, authorize_dataset
from core import timeutil
from core.constants import Feature, LogicalType
from core.errors import ValidationError
from database.connection import session_scope
from database.models import Dataset
from services import resource_manager, versioning_service
from services.audit_service import record_audit
from services.frame_utils import json_safe
from services.quality_score import compute_quality_batches
from services.resource_manager import OperationGuard
from services.type_inference import (
    CATEGORICAL_MAX_DISTINCT,
    CATEGORICAL_MAX_RATIO,
    DATE_FORMATS,
    DATE_SHAPE,
    EXAMPLE_LIMIT,
    IDENTIFIER_NAME,
    ISO_FORMAT,
    MOSTLY_THRESHOLD,
    NumericPatterns,
    ParseOptions,
    numeric_patterns,
    parse_dates,
    parse_numeric_token,
)

FEATURE = Feature.DATA_PROFILING

TOP_VALUES = 10
CASE_GROUP_LIMIT = 20
AI_SAMPLE_VALUES = 5
AI_VALUE_CHARS = 40
REPEATED_TOP_MAX = 1000
CACHE_SIZE = 32
PROFILE_BATCH_COLUMNS = 100
PROFILE_BATCH_BYTES = 64 * 1024 * 1024
PROFILE_BATCH_CELLS = 2_000_000

IQR_MULTIPLIER = 1.5
Z_THRESHOLD = 3.0
IDENTIFIER_MIN_UNIQUENESS = 0.9
BUSINESS_KEY_MIN_UNIQUENESS = 0.95
HIGH_MISSING_PCT = 50.0

# Health check thresholds (quality score is 0-100).
CRITICAL_SCORE_BELOW = 60.0
ATTENTION_SCORE_BELOW = 90.0
HEALTH_TOP_ISSUES = 5

OUTLIER_WORDING = "possible outliers, review before changing; they may be legitimate"

# Columns named like a record identifier: any repeated value is suspicious.
_ID_NAME = re.compile(r"(?i)(^id$|[ _-]id$|^id[ _-]|uuid|guid)")
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

_cache: OrderedDict[tuple[int, int, str], dict[str, Any]] = OrderedDict()
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def clear_profile_cache() -> None:
    """Drop every cached profile (used by tests and after settings changes)."""
    with _cache_lock:
        _cache.clear()


def _cache_get(key: tuple[int, int, str]) -> dict[str, Any] | None:
    with _cache_lock:
        hit = _cache.get(key)
        if hit is None:
            return None
        _cache.move_to_end(key)
        return copy.deepcopy(hit)


def _cache_put(key: tuple[int, int, str], profile: dict[str, Any]) -> None:
    with _cache_lock:
        _cache[key] = copy.deepcopy(profile)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _one_line(value: Any) -> str:
    """Display text from dataset values on a single line (newlines and tabs become spaces).

    A blank line inside a value would otherwise end an HTML block in the UI
    and let the rest render as Markdown.
    """
    return " ".join(str(value).split())


def _pct(part: int | float, whole: int | float) -> float:
    """Percentage rounded to two decimals; 0.0 when ``whole`` is zero."""
    return round(100.0 * part / whole, 2) if whole else 0.0


def _top_values(series: pd.Series, limit: int = TOP_VALUES) -> list[dict[str, Any]]:
    """Most frequent non-missing values, ties broken by value text (deterministic)."""
    counts = series.dropna().value_counts()
    if counts.empty:
        return []
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return [{"value": json_safe(value), "count": int(count)} for value, count in ordered[:limit]]


def _parse_options(import_report: dict[str, Any]) -> ParseOptions:
    """Separators used at import, so mostly-numeric checks read values the same way."""
    try:
        return ParseOptions(
            decimal_separator=import_report.get("decimal_separator") or ".",
            thousands_separator=import_report.get("thousands_separator"),
        )
    except ValidationError:
        return ParseOptions()


def _derive_semantic(name: str, logical: str, non_null: int, distinct: int) -> str:
    """Semantic role from the data when the import report has none for this column."""
    if non_null == 0:
        return "empty"
    if logical == LogicalType.BOOLEAN.value:
        return "boolean"
    if logical == LogicalType.DATETIME.value:
        return "datetime"
    identifier = bool(IDENTIFIER_NAME.search(name)) and distinct >= IDENTIFIER_MIN_UNIQUENESS * non_null
    if logical in (LogicalType.INTEGER.value, LogicalType.FLOAT.value):
        return "identifier" if identifier else "numeric"
    if identifier:
        return "identifier"
    if distinct <= CATEGORICAL_MAX_DISTINCT or distinct <= CATEGORICAL_MAX_RATIO * non_null:
        return "categorical"
    return "text"


def _semantic(name: str, logical: str, non_null: int, distinct: int, import_types: dict[str, Any]) -> str:
    info = import_types.get(name)
    if isinstance(info, dict) and info.get("semantic") and info.get("logical_type") == logical:
        return str(info["semantic"])
    return _derive_semantic(name, logical, non_null, distinct)


# ---------------------------------------------------------------------------
# Per-type column statistics
# ---------------------------------------------------------------------------


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    """Exact descriptive statistics and outlier counts for an integer/float column.

    ``std`` is the sample standard deviation (ddof=1, the pandas default) and the
    z-score uses the same value. IQR outliers lie strictly outside
    [q1 - 1.5*IQR, q3 + 1.5*IQR]; z outliers have |z| strictly greater than 3.
    """
    values = series.dropna()
    numbers = values.astype("float64")
    numbers = numbers[np.isfinite(numbers.to_numpy())]
    if numbers.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "q1": None,
            "q3": None,
            "outliers_iqr": 0,
            "outliers_z": 0,
            "outlier_candidates": 0,
            "zero_count": 0,
            "negative_count": 0,
        }
    q1, q3 = float(numbers.quantile(0.25)), float(numbers.quantile(0.75))
    iqr = q3 - q1
    iqr_mask = (numbers < q1 - IQR_MULTIPLIER * iqr) | (numbers > q3 + IQR_MULTIPLIER * iqr)
    std = float(numbers.std()) if len(numbers) > 1 else None
    if std:
        z_mask = ((numbers - numbers.mean()) / std).abs() > Z_THRESHOLD
    else:
        z_mask = pd.Series(False, index=numbers.index)
    return {
        "min": json_safe(values.min()),
        "max": json_safe(values.max()),
        "mean": float(numbers.mean()),
        "median": float(numbers.median()),
        "std": std,
        "q1": q1,
        "q3": q3,
        "outliers_iqr": int(iqr_mask.sum()),
        "outliers_z": int(z_mask.sum()),
        "outlier_candidates": int((iqr_mask | z_mask).sum()),
        "zero_count": int((numbers == 0).sum()),
        "negative_count": int((numbers < 0).sum()),
    }


def _datetime_stats(series: pd.Series) -> dict[str, Any]:
    values = series.dropna()
    if values.empty:
        return {"min": None, "max": None, "distinct_days": 0}
    return {
        "min": json_safe(values.min()),
        "max": json_safe(values.max()),
        "distinct_days": int(values.dt.normalize().nunique()),
    }


def _boolean_stats(series: pd.Series) -> dict[str, int]:
    values = series.dropna()
    return {"true_count": int(values.sum()), "false_count": int((~values.astype(bool)).sum())}


def _case_variant_groups(counts: pd.Series) -> list[dict[str, Any]]:
    """Groups of values equal after strip+casefold but spelled differently."""
    keys = pd.Series(counts.index, dtype="string").str.strip().str.casefold()
    frame = pd.DataFrame({"value": list(counts.index), "count": counts.to_numpy(), "key": keys.to_numpy()})
    frame = frame[frame["key"] != ""]
    spellings = frame.groupby("key")["value"].transform("size")
    frame = frame[spellings > 1]
    groups = []
    for key, members in frame.groupby("key", sort=False):
        members = members.sort_values(["count", "value"], ascending=[False, True])
        groups.append(
            {
                "normalized": str(key),
                "total": int(members["count"].sum()),
                "variants": [
                    {"value": str(v), "count": int(c)}
                    for v, c in zip(members["value"], members["count"], strict=True)
                ],
            }
        )
    groups.sort(key=lambda g: (-g["total"], g["normalized"]))
    return groups


def _numeric_text_check(uniques: list[str], counts: np.ndarray, patterns: NumericPatterns) -> dict[str, Any]:
    """How many text values parse as numbers, with invalid examples and currencies."""
    parsed = 0
    examples: list[str] = []
    currencies: set[str] = set()
    for token, count in zip(uniques, counts.tolist(), strict=True):
        result = parse_numeric_token(token, patterns)
        if result is None:
            if len(examples) < EXAMPLE_LIMIT:
                examples.append(_one_line(token))
            continue
        parsed += int(count)
        if result.currency:
            currencies.add(result.currency)
    return {"parsed": parsed, "examples": examples, "currencies": sorted(currencies)}


def _date_text_check(uniques: pd.Series, counts: np.ndarray, non_missing: int) -> dict[str, Any] | None:
    """Mostly-date analysis (same rules as the importer).

    A column qualifies when at least MOSTLY_THRESHOLD of its values have a date
    shape. Invalid values are those without a date shape plus date-shaped values
    no known format parses. Formats are inconsistent when no single format
    (from DATE_FORMATS or ISO 8601) parses every parseable value.
    """
    shaped = uniques.str.match(DATE_SHAPE).fillna(False).to_numpy(dtype=bool)
    shaped_count = int(counts[shaped].sum())
    if non_missing == 0 or shaped_count / non_missing < MOSTLY_THRESHOLD:
        return None
    values = uniques[shaped].reset_index(drop=True)
    value_counts = counts[shaped]
    masks = {
        fmt: parse_dates(values, fmt).notna().to_numpy(dtype=bool) for fmt in (*DATE_FORMATS, ISO_FORMAT)
    }
    parsed_any = np.zeros(len(values), dtype=bool)
    for mask in masks.values():
        parsed_any |= mask
    formats_seen: dict[str, int] = {}
    claimed = np.zeros(len(values), dtype=bool)
    for fmt, mask in masks.items():
        new = mask & ~claimed
        if new.any():
            formats_seen[fmt] = int(value_counts[new].sum())
            claimed |= new
    covering = any(mask[parsed_any].all() for mask in masks.values())
    inconsistent = bool(parsed_any.any()) and not covering
    invalid = (non_missing - shaped_count) + int(value_counts[~parsed_any].sum())
    if invalid == 0 and not inconsistent:
        return None
    examples = uniques[~shaped].tolist() + values[~parsed_any].tolist()
    return {
        "count": invalid,
        "examples": [_one_line(v) for v in examples[:EXAMPLE_LIMIT]],
        "date_pct": _pct(shaped_count, non_missing),
        "formats_seen": formats_seen,
        "inconsistent_formats": inconsistent,
    }


def _text_stats(series: pd.Series, patterns: NumericPatterns) -> dict[str, Any]:
    """Whitespace, lengths, case variants and mostly-numeric / mostly-date checks.

    ``case_variant_groups`` lists at most CASE_GROUP_LIMIT groups (a sample);
    ``case_variant_group_count`` and ``case_variant_rows`` (values not written in
    the dominant spelling of their group) are exact over all groups.
    """
    values = series.dropna().astype("string")
    stripped = values.str.strip()
    lengths = values.str.len()
    counts = values.value_counts()
    groups = _case_variant_groups(counts)
    analysed = stripped[stripped != ""].value_counts()
    non_blank = int(analysed.sum())
    uniques = [str(v) for v in analysed.index]
    numeric = _numeric_text_check(uniques, analysed.to_numpy(), patterns)
    numeric_ratio = numeric["parsed"] / non_blank if non_blank else 0.0
    mostly_numeric = None
    if MOSTLY_THRESHOLD <= numeric_ratio < 1:
        mostly_numeric = {
            "count": non_blank - numeric["parsed"],
            "examples": numeric["examples"],
            "numeric_pct": _pct(numeric["parsed"], non_blank),
        }
    mixed = (
        numeric["currencies"] if non_blank and numeric_ratio == 1 and len(numeric["currencies"]) > 1 else []
    )
    return {
        "whitespace_count": int((values != stripped).sum()),
        "empty_string_count": int((values == "").sum()),
        "min_length": int(lengths.min()) if len(values) else None,
        "max_length": int(lengths.max()) if len(values) else None,
        "case_variant_groups": groups[:CASE_GROUP_LIMIT],
        "case_variant_group_count": len(groups),
        "case_variant_rows": sum(g["total"] - g["variants"][0]["count"] for g in groups),
        "mostly_numeric_invalid": mostly_numeric,
        "mostly_date_invalid": _date_text_check(
            pd.Series(uniques, dtype=object), analysed.to_numpy(), non_blank
        ),
        "mixed_currencies": mixed,
    }


def _profile_column(
    name: str,
    logical: str,
    series: pd.Series,
    rows: int,
    import_types: dict[str, Any],
    patterns: NumericPatterns,
) -> dict[str, Any]:
    """Exact profile of one fully loaded column."""
    non_null = int(series.notna().sum())
    distinct = int(series.nunique(dropna=True))
    profile: dict[str, Any] = {
        "name": name,
        "logical_type": logical,
        "semantic": _semantic(name, logical, non_null, distinct, import_types),
        "non_null": non_null,
        "missing": rows - non_null,
        "missing_pct": _pct(rows - non_null, rows),
        "distinct": distinct,
        "distinct_pct": _pct(distinct, non_null),
        "is_constant": non_null > 0 and distinct == 1,
        "is_empty": non_null == 0,
        "top_values": _top_values(series),
    }
    if logical in (LogicalType.INTEGER.value, LogicalType.FLOAT.value):
        profile["numeric"] = _numeric_stats(series)
    elif logical == LogicalType.DATETIME.value:
        profile["datetime"] = _datetime_stats(series)
    elif logical == LogicalType.BOOLEAN.value:
        profile["boolean"] = _boolean_stats(series)
    else:
        profile["text"] = _text_stats(series, patterns)
    return profile


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------


def _issue(
    kind: str, column: str | None, count: int, message: str, severity: str, source: str = "data"
) -> dict[str, Any]:
    return {
        "type": kind,
        "column": column,
        "count": int(count),
        "message": _one_line(message),
        "severity": severity,
        "source": source,
    }


def _missing_severity(pct: float) -> str:
    return "high" if pct >= HIGH_MISSING_PCT else "medium" if pct >= 10 else "low"


def _column_confirmed(col: dict[str, Any]) -> list[dict[str, Any]]:
    name = col["name"]
    if col["is_empty"]:
        if col["missing"] == 0:
            return []
        return [_issue("empty_column", name, col["missing"], f"Column '{name}' has no values.", "medium")]
    issues = []
    if col["missing"]:
        issues.append(
            _issue(
                "missing_values",
                name,
                col["missing"],
                f"Column '{name}' has {col['missing']} missing value(s) ({col['missing_pct']}%).",
                _missing_severity(col["missing_pct"]),
            )
        )
    text = col.get("text")
    if not text:
        return issues
    if text["mostly_numeric_invalid"]:
        info = text["mostly_numeric_invalid"]
        issues.append(
            _issue(
                "invalid_numeric",
                name,
                info["count"],
                f"Column '{name}' is {info['numeric_pct']}% numeric but {info['count']} value(s) are not valid "
                f"numbers (for example: {', '.join(info['examples'])}).",
                "medium",
            )
        )
    if text["mostly_date_invalid"]:
        info = text["mostly_date_invalid"]
        if info["count"]:
            issues.append(
                _issue(
                    "invalid_dates",
                    name,
                    info["count"],
                    f"Column '{name}' looks like dates but {info['count']} value(s) are not valid dates "
                    f"(for example: {', '.join(info['examples'])}).",
                    "medium",
                )
            )
        if info["inconsistent_formats"]:
            formats = ", ".join(f"{fmt} ({n})" for fmt, n in info["formats_seen"].items())
            issues.append(
                _issue(
                    "inconsistent_date_formats",
                    name,
                    sum(info["formats_seen"].values()),
                    f"Column '{name}' mixes date formats: {formats}.",
                    "medium",
                )
            )
    if text["mixed_currencies"]:
        issues.append(
            _issue(
                "conversion_failure",
                name,
                col["non_null"],
                f"Column '{name}' mixes currencies ({', '.join(text['mixed_currencies'])}), so it could not be "
                "converted to numbers and is kept as text.",
                "medium",
            )
        )
    if text["whitespace_count"]:
        pct = _pct(text["whitespace_count"], col["non_null"])
        issues.append(
            _issue(
                "whitespace",
                name,
                text["whitespace_count"],
                f"Column '{name}' has {text['whitespace_count']} value(s) with leading or trailing spaces.",
                "medium" if pct >= 10 else "low",
            )
        )
    return issues


def _business_key_count(col: dict[str, Any]) -> int:
    """Repeated values in an identifier-like column (0 when not suspicious)."""
    non_null, distinct = col["non_null"], col["distinct"]
    repeats = non_null - distinct
    if non_null == 0 or repeats == 0:
        return 0
    uniqueness = distinct / non_null
    if _ID_NAME.search(col["name"]):
        return repeats
    if col["semantic"] == "identifier" and uniqueness >= BUSINESS_KEY_MIN_UNIQUENESS:
        return repeats
    return 0


def _column_suspected(col: dict[str, Any], rows: int) -> list[dict[str, Any]]:
    name = col["name"]
    issues = []
    numeric = col.get("numeric")
    if numeric and col["semantic"] != "identifier" and numeric["outlier_candidates"]:
        issues.append(
            _issue(
                "outliers",
                name,
                numeric["outlier_candidates"],
                f"Column '{name}': {numeric['outlier_candidates']} {OUTLIER_WORDING} "
                f"({numeric['outliers_iqr']} outside 1.5 x IQR, {numeric['outliers_z']} with |z| > 3).",
                "low",
            )
        )
    text = col.get("text")
    if text and text["case_variant_group_count"]:
        example = " / ".join(v["value"] for v in text["case_variant_groups"][0]["variants"][:3])
        issues.append(
            _issue(
                "inconsistent_categories",
                name,
                text["case_variant_rows"],
                f"Column '{name}' has {text['case_variant_group_count']} value(s) spelled in different ways "
                f"(for example: {example}). "
                "Review whether they mean the same thing.",
                "low",
            )
        )
    repeats = _business_key_count(col)
    if repeats:
        issues.append(
            _issue(
                "duplicate_business_key",
                name,
                repeats,
                f"Column '{name}' looks like an identifier but {repeats} value(s) repeat. "
                "Check whether the same record appears more than once.",
                "medium",
            )
        )
    if col["is_constant"] and rows > 1:
        issues.append(
            _issue(
                "constant_column",
                name,
                col["non_null"],
                f"Column '{name}' has the same value in every non-missing row; it may not be useful for analysis.",
                "low",
            )
        )
    if not col["is_empty"] and col["missing_pct"] >= HIGH_MISSING_PCT:
        issues.append(
            _issue(
                "high_missing",
                name,
                col["missing"],
                f"Column '{name}' is {col['missing_pct']}% missing; consider whether it is reliable enough to use.",
                "medium",
            )
        )
    return issues


def _import_issues(import_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows the importer skipped. These are facts about the upload, not the stored data."""
    issues = []
    malformed = int(import_report.get("malformed_rows_skipped") or 0)
    if malformed:
        lines = ", ".join(
            str(item.get("line")) for item in (import_report.get("malformed_row_examples") or [])[:10]
        )
        detail = f" (lines {lines})" if lines else ""
        issues.append(
            _issue(
                "malformed_rows",
                None,
                malformed,
                f"{malformed} malformed row(s) were skipped at import because their field count did not match "
                f"the header{detail}.",
                "medium",
                source="import",
            )
        )
    blank = int(import_report.get("blank_rows_skipped") or 0)
    if blank:
        issues.append(
            _issue(
                "blank_rows",
                None,
                blank,
                f"{blank} completely blank row(s) were skipped at import.",
                "low",
                source="import",
            )
        )
    return issues


def _sort_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(issues, key=lambda i: (_SEVERITY_RANK.get(i["severity"], 3), -i["count"]))


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def _guarded(
    batches: Iterator[pd.DataFrame], guard: OperationGuard | None, rows: int
) -> Iterator[pd.DataFrame]:
    """Report progress per batch; close the file iterator even when the guard stops the work."""
    seen = 0
    try:
        for batch in batches:
            if guard is not None:
                guard.step(0.8 + 0.2 * (seen / rows if rows else 1.0), "Checking duplicate rows and quality")
            seen += len(batch)
            yield batch
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()


def _resolve(
    ctx: AuthContext, dataset_id: int, version_id: int | None
) -> tuple[versioning_service.VersionRef, dict[str, Any]]:
    """Authorize (fresh DB checks) and return the version plus the dataset's import report."""
    with session_scope() as db:
        ref = versioning_service.resolve_version(db, ctx, dataset_id, version_id, feature=FEATURE)
        dataset = db.get(Dataset, ref.dataset_id)
        report = json.loads(dataset.import_report_json or "{}") if dataset is not None else {}
    return ref, report if isinstance(report, dict) else {}


def _column_batches(path: Any, names: list[str]) -> list[list[str]]:
    """Group columns, in order, so each batch stays under the column, cell and byte budgets."""
    metadata = pq.read_metadata(path)
    sizes: dict[str, int] = {}
    for rg_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg_index)
        for col_index in range(row_group.num_columns):
            column = row_group.column(col_index)
            sizes[column.path_in_schema] = sizes.get(column.path_in_schema, 0) + column.total_uncompressed_size
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    rows = max(metadata.num_rows, 1)
    for name in names:
        size = sizes.get(name, 0)
        if current and (
            len(current) >= PROFILE_BATCH_COLUMNS
            or rows * (len(current) + 1) > PROFILE_BATCH_CELLS
            or used + size > PROFILE_BATCH_BYTES
        ):
            batches.append(current)
            current, used = [], 0
        current.append(name)
        used += size
    if current:
        batches.append(current)
    return batches


def profile_dataset(
    ctx: AuthContext,
    dataset_id: int,
    *,
    version_id: int | None = None,
    guard: OperationGuard | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Exact profile of a dataset version (the current one by default).

    Authorization (dataset access plus the DATA_PROFILING plan feature) runs on
    every call before the cache is consulted. ``guard`` receives progress and can
    cancel or time out the work. Returns a JSON-safe dict (see module docstring).
    """
    ref, import_report = _resolve(ctx, dataset_id, version_id)
    key = (ref.organization_id, ref.version_id, str(ref.path))
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached
    resource_manager.check_memory(ref.rows * 16, operation="Checking duplicate rows")
    patterns = numeric_patterns(_parse_options(import_report))
    import_types = import_report.get("column_types") or {}
    columns_profile = []
    total = max(len(ref.schema), 1)
    logical_types = {column["name"]: column["logical_type"] for column in ref.schema}
    index = 0
    budget = resource_manager.memory_budget_bytes()
    for batch in _column_batches(ref.path, list(logical_types)):
        needed = resource_manager.estimate_parquet_memory(ref.path, batch)
        groups = [batch]
        if len(batch) > 1 and budget is not None and needed > budget:
            groups = [[name] for name in batch]  # same memory need as profiling one column at a time
        for group in groups:
            if guard is not None:
                guard.check()
            group_need = needed if group is batch else resource_manager.estimate_parquet_memory(ref.path, group)
            resource_manager.check_memory(group_need, operation="Profiling this dataset")
            frame = versioning_service.read_parquet(ref.path, group)
            for name in group:
                if guard is not None:
                    guard.step(0.8 * index / total, f"Profiling column {index + 1} of {len(ref.schema)}")
                columns_profile.append(
                    _profile_column(name, logical_types[name], frame[name], ref.rows, import_types, patterns)
                )
                index += 1
            del frame
    batches = versioning_service.iter_frame_batches(
        ctx, dataset_id, version_id=ref.version_id, feature=FEATURE
    )
    quality = compute_quality_batches(_guarded(batches, guard, ref.rows)).to_dict()

    confirmed = [issue for col in columns_profile for issue in _column_confirmed(col)]
    if quality["duplicate_rows"]:
        count = quality["duplicate_rows"]
        pct = _pct(count, ref.rows)
        confirmed.append(
            _issue(
                "duplicate_rows",
                None,
                count,
                f"{count} row(s) ({pct}%) are exact copies of an earlier row.",
                "high" if pct >= 10 else "medium" if pct >= 1 else "low",
            )
        )
    confirmed.extend(_import_issues(import_report))
    suspected = [issue for col in columns_profile for issue in _column_suspected(col, ref.rows)]
    profile = {
        "dataset_id": ref.dataset_id,
        "dataset_name": ref.dataset_name,
        "version_id": ref.version_id,
        "version_number": ref.version_number,
        "rows": ref.rows,
        "columns": ref.columns,
        "quality": quality,
        "duplicate_rows": {
            "count": quality["duplicate_rows"],
            "pct": _pct(quality["duplicate_rows"], ref.rows),
        },
        "columns_profile": columns_profile,
        "confirmed_issues": _sort_issues(confirmed),
        "suspected_issues": _sort_issues(suspected),
        "import_report": import_report,
        "generated_at": timeutil.utcnow().isoformat(),
    }
    if guard is not None:
        guard.step(1.0, "Profile complete")
    _cache_put(key, profile)
    return copy.deepcopy(profile)


# ---------------------------------------------------------------------------
# Repeated values
# ---------------------------------------------------------------------------


def _validate_selection(columns: list[str], top: int) -> list[str]:
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValidationError("Select at least one column.")
    names = [str(c) for c in columns]
    if len(set(names)) != len(names):
        raise ValidationError("Each column can be selected only once.")
    if not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= REPEATED_TOP_MAX:
        raise ValidationError(f"The number of results must be between 1 and {REPEATED_TOP_MAX}.")
    return names


def repeated_values(
    ctx: AuthContext,
    dataset_id: int,
    columns: list[str],
    *,
    trim: bool = True,
    case_insensitive: bool = False,
    top: int = 50,
) -> pd.DataFrame:
    """Values (or value combinations) that occur more than once in the selected columns.

    ``trim`` and ``case_insensitive`` normalize text only for comparison; the
    returned values are the first original spelling seen, and the data is never
    changed. Rows with a missing value in any selected column are not counted
    (missing values are covered by the missing-value analysis). Returns the
    selected columns plus ``count``, most frequent first, at most ``top`` rows.
    """
    names = _validate_selection(columns, top)
    frame = versioning_service.load_frame(ctx, dataset_id, columns=names, feature=FEATURE)
    frame = frame.dropna(how="any").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame({**{n: frame[n] for n in names}, "count": pd.Series([], dtype="int64")})
    keys = pd.DataFrame(index=frame.index)
    for name in names:
        key = frame[name]
        if isinstance(key.dtype, pd.StringDtype):
            key = key.str.strip() if trim else key
            key = key.str.casefold() if case_insensitive else key
        keys[name] = key
    group_ids = keys.groupby(names, sort=False, dropna=False).ngroup().to_numpy()
    counts = np.bincount(group_ids)
    first_row = pd.Series(np.arange(len(frame))).groupby(group_ids).min().to_numpy()
    repeated = np.flatnonzero(counts > 1)
    order = sorted(repeated.tolist(), key=lambda g: (-counts[g], first_row[g]))[:top]
    result = frame.iloc[[int(first_row[g]) for g in order]].reset_index(drop=True)
    result["count"] = np.array([int(counts[g]) for g in order], dtype="int64")
    return result


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

_NEXT_STEPS = {
    "missing_values": "Review missing values in the Missing values workspace.",
    "empty_column": "Remove empty columns in the Cleaning workspace.",
    "duplicate_rows": "Review and remove exact duplicate rows in the Duplicates workspace.",
    "invalid_numeric": "Fix or convert the invalid numeric values in the Cleaning workspace.",
    "invalid_dates": "Standardize dates and review invalid values in the Cleaning workspace.",
    "inconsistent_date_formats": "Standardize the date format in the Cleaning workspace.",
    "conversion_failure": "Review the values that could not be converted in the Cleaning workspace.",
    "whitespace": "Trim leading and trailing spaces in the Cleaning workspace.",
    "malformed_rows": "Check the source file and upload it again if the skipped rows are needed.",
    "blank_rows": "No action is needed unless the blank rows were meant to hold data.",
    "outliers": "Review the possible outliers in the Cleaning workspace before changing anything.",
    "inconsistent_categories": "Map the different spellings to one category in the Cleaning workspace.",
    "duplicate_business_key": "Check the repeated identifiers in the Duplicates workspace.",
    "constant_column": "Consider removing the constant column in the Cleaning workspace.",
    "high_missing": "Decide whether to fill or drop this column in the Missing values workspace.",
    "no_rows": "Upload a file that contains data rows, or restore an earlier version that has rows.",
}


def _health_status(score: float, confirmed: list[dict[str, Any]], rows: int | None = None) -> str:
    """Status from the quality score and confirmed issues found in the stored data.

    empty: the version has no rows (its quality score of 100 means nothing).
    critical: score below 60 or any high-severity data issue.
    needs attention: score below 90 or any medium-severity data issue.
    healthy: otherwise. Low-severity issues, suspected issues and import-time
    facts (skipped rows) are listed but never lower the status, because they
    either need a judgement call or cannot be fixed in the stored data.
    """
    if rows == 0:
        return "empty"
    severities = {i["severity"] for i in confirmed if i["source"] == "data"}
    if score < CRITICAL_SCORE_BELOW or "high" in severities:
        return "critical"
    if score < ATTENTION_SCORE_BELOW or "medium" in severities:
        return "needs attention"
    return "healthy"


def _with_step(issue: dict[str, Any], kind: str) -> dict[str, Any]:
    return {**issue, "kind": kind, "recommended_step": _NEXT_STEPS.get(issue["type"], "Review this issue.")}


def health_check(ctx: AuthContext, dataset_id: int) -> dict[str, Any]:
    """Automated data health check of the current version, recorded in the audit log."""
    profile = profile_dataset(ctx, dataset_id)
    score = float(profile["quality"]["score"])
    confirmed = profile["confirmed_issues"]
    suspected = profile["suspected_issues"]
    status = _health_status(score, confirmed, profile["rows"])
    if status == "empty":
        no_rows = _issue(
            "no_rows",
            None,
            0,
            "This dataset has no rows, only column headers, so there is no data to check.",
            "medium",
        )
        confirmed = [no_rows, *confirmed]
    result = {
        "dataset_id": profile["dataset_id"],
        "dataset_name": profile["dataset_name"],
        "version_id": profile["version_id"],
        "version_number": profile["version_number"],
        "status": status,
        "quality_score": score,
        "confirmed_issue_count": len(confirmed),
        "suspected_issue_count": len(suspected),
        "top_issues": [_with_step(i, "confirmed") for i in confirmed[:HEALTH_TOP_ISSUES]],
        "suspected_issues": [_with_step(i, "suspected") for i in suspected[:HEALTH_TOP_ISSUES]],
        "thresholds": {
            "empty": "the dataset has no rows",
            "critical": f"quality score below {CRITICAL_SCORE_BELOW:g} or any high-severity issue in the data",
            "needs attention": (
                f"quality score below {ATTENTION_SCORE_BELOW:g} or any medium-severity issue in the data"
            ),
            "healthy": "otherwise",
        },
        "generated_at": timeutil.utcnow().isoformat(),
    }
    with session_scope() as db:
        authorize_dataset(db, ctx, dataset_id, feature=FEATURE)
        record_audit(
            db,
            ctx,
            "dataset.health_check",
            target_type="dataset",
            target_id=profile["dataset_id"],
            details={
                "version": profile["version_number"],
                "status": status,
                "quality_score": score,
                "confirmed_issues": len(confirmed),
                "suspected_issues": len(suspected),
            },
        )
    return result


# ---------------------------------------------------------------------------
# AI summary
# ---------------------------------------------------------------------------


def column_summary_for_ai(ctx: AuthContext, dataset_id: int) -> list[dict[str, Any]]:
    """Compact per-column summary for LLM prompts (at most 5 short sample values per column)."""
    profile = profile_dataset(ctx, dataset_id)
    summary = []
    for col in profile["columns_profile"]:
        item: dict[str, Any] = {
            "name": col["name"],
            "logical_type": col["logical_type"],
            "semantic": col["semantic"],
            "missing_pct": col["missing_pct"],
            "distinct": col["distinct"],
        }
        if "numeric" in col:
            item.update({k: col["numeric"][k] for k in ("min", "max", "mean")})
        elif "datetime" in col:
            item.update({k: col["datetime"][k] for k in ("min", "max")})
        elif "text" in col:
            item["top_values"] = [
                str(v["value"])[:AI_VALUE_CHARS] for v in col["top_values"][:AI_SAMPLE_VALUES]
            ]
        summary.append(item)
    return summary
