"""Data cleaning operations: read-only previews and versioned applies.

Plan features (docs/ARCHITECTURE.md permission/feature matrix):

* Standard cleaning (``Feature.STANDARD_CLEANING``): trim_whitespace,
  remove_control_chars, rename_columns, remove_empty, normalize_case,
  drop_columns, plus every read-only detection report (detect_casing_issues,
  identify_redundant_columns, outlier_report).
* Advanced cleaning (``Feature.ADVANCED_CLEANING``): map_categories,
  standardize_dates, convert_numeric, convert_types, handle_outliers (cap,
  exclude, flag), fix_invalid_values and validate_constraints.

Previews need dataset access plus the operation's feature. Applies also need
the ``clean_data`` permission. Destructive applies (remove_empty, drop_columns,
handle_outliers with action "exclude", fix_invalid_values with action
"set_missing", and any on_failure="set_missing") require ``confirm=True``.

How an operation runs:

1. One version is resolved and authorized. Every read below uses that version.
2. Options are validated against the version's schema. Unknown options are
   rejected; nothing is guessed.
3. Operations that need whole-column knowledge (the chosen date format, integer
   versus decimal typing, exact outlier bounds, which columns are fully empty)
   compute it first from a column-subset read after a memory check
   (``_read_columns``) or a bounded batch scan.
4. With that context every transform is row-local, so apply always streams
   through ``versioning_service.apply_streaming_operation`` with memory bounded
   by PARQUET_BATCH_ROWS instead of loading the whole dataset with
   ``apply_frame_operation``. Before the first batch the current version is
   checked again, so context prepared for an older version is never used.
5. A preview runs exactly the same batch function over the same version and
   never writes anything, so previewed and applied numbers agree.

Definitions used in every result:

* Row numbers are 1-based positions in the version.
* A text value is empty when it is missing, empty or whitespace only.
* ``cells_changed`` counts cells whose stored value changes (missing-aware).
  A type conversion changes every non-missing source cell, including
  whitespace-only text that becomes missing. A flag column counts its True
  cells. ``rows_affected`` counts rows with a changed cell plus removed rows.
* Statistics are exact over the whole column. IQR bounds use pandas linear
  quantile interpolation; z-score bounds use the mean and the sample standard
  deviation (ddof=1). Missing values are never outliers.
* Report-only modes (handle_outliers "retain", fix_invalid_values "report" and
  validate_constraints) never change data: their apply raises ValidationError
  and the preview shows the report.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import partial
from typing import Any

try:  # regex syntax tree for the ReDoS guard: re._parser on Python 3.11+, sre_parse on 3.10
    from re import _parser as _sre_parse
except ImportError:  # pragma: no cover - Python 3.10
    import sre_parse as _sre_parse  # type: ignore[no-redef]

import numpy as np
import pandas as pd
import regex as regex_engine

from auth.authorization import AuthContext
from config.settings import get_settings
from core.constants import Feature, LogicalType, Permission
from core.errors import ConflictError, OperationTimeoutError, ValidationError
from database.connection import session_scope
from modules import org_settings
from services import resource_manager, versioning_service
from services.frame_utils import json_safe, normalize_frame
from services.quality_score import QualityAccumulator
from services.resource_manager import OperationGuard
from services.type_inference import (
    BOOLEAN_FALSE,
    BOOLEAN_TRUE,
    DATE_FORMATS,
    ISO_FORMAT,
    ParseOptions,
    choose_date_format,
    decimal_is_lossless,
    numeric_patterns,
    parse_dates,
    parse_numeric_token,
)
from services.versioning_service import VersionRef

TEXT = LogicalType.TEXT.value
INTEGER = LogicalType.INTEGER.value
FLOAT = LogicalType.FLOAT.value
BOOLEAN = LogicalType.BOOLEAN.value
DATETIME = LogicalType.DATETIME.value
NUMERIC_TYPES = (INTEGER, FLOAT)

EXAMPLE_LIMIT = 20
RULE_EXAMPLE_LIMIT = 5
MAX_SAMPLE_ROWS = 200
MAX_COLUMN_NAME = 120
MAX_REGEX_LENGTH = 200
# User regular expressions run on the third-party ``regex`` engine with a time
# limit per value (also capped by the operation's remaining time) and with the
# GIL released, so one slow pattern cannot stall other sessions. Values longer
# than MAX_REGEX_VALUE_LENGTH are refused instead of matched.
REGEX_MATCH_SECONDS = 0.5
MAX_REGEX_VALUE_LENGTH = 1000
MAX_MAPPING_ENTRIES = 1000
MAX_MAPPED_VALUE = 500
MAX_RULES = 50
MAX_ALLOWED_VALUES = 1000
MAX_THRESHOLD = 100.0
DISTINCT_LIMIT = 50_000
CORRELATION_THRESHOLD = 0.999
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
FLOAT_EXACT_INT = 2**53

# Pragmatic e-mail syntax (WHATWG HTML "valid e-mail address"), additionally
# requiring a dot-separated domain such as example.com. Checked with fullmatch
# on the trimmed value.
EMAIL_REGEX = (
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)
_EMAIL = re.compile(EMAIL_REGEX)

# C0 and C1 control characters that are removed, and the whitespace controls
# (tab, newline, carriage return, vertical tab, form feed, next line) whose runs
# become a single space.
_CONTROL_REMOVE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x84\x86-\x9f]")
_CONTROL_SPACE = re.compile(r"[\t\n\r\x0b\x0c\x85]+")
_CONTROL_ANY = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_PERCENT = re.compile(r"^(?P<open>\(?)\s*(?P<body>[^%]*?)\s*%\s*(?P<close>\)?)$")
_CUSTOM_DATE_FORMAT = re.compile(r"^(?:%[YymdbBHMSfIpjaA]|[-/.:, T])+$")
_REPEAT_OPS = frozenset({"MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"})
_FEATURES = {"standard": Feature.STANDARD_CLEANING, "advanced": Feature.ADVANCED_CLEANING}
_FAIL = object()


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _Prepared:
    """Validated options plus whole-column context, positional over the version's rows."""

    params: dict[str, Any]
    column: str | None = None  # column whose original values are reported as failures
    replacement: pd.Series | None = None  # new values of ``column``
    changed: np.ndarray | None = None  # rows whose ``column`` cell changes
    failed: np.ndarray | None = None  # rows whose value could not be converted/validated
    marked: np.ndarray | None = None  # rows flagged for review (outliers)
    removed: np.ndarray | None = None  # rows removed
    added: dict[str, pd.Series] = field(default_factory=dict)  # new columns
    drop: list[str] = field(default_factory=list)
    rename: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)  # reported context (bounds, format)
    warnings: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)  # internal helpers, never reported

    @property
    def failure_count(self) -> int:
        return 0 if self.failed is None else int(self.failed.sum())


@dataclass
class _Effect:
    """What one batch transform did. Masks are aligned with the input batch."""

    frame: pd.DataFrame
    changed: np.ndarray
    removed: np.ndarray
    cells_changed: int
    failed: np.ndarray | None = None
    marked: np.ndarray | None = None
    counts: dict[str, int] = field(default_factory=dict)


Validator = Callable[[dict[str, Any], VersionRef], dict[str, Any]]
Preparer = Callable[[VersionRef, dict[str, Any], OperationGuard], _Prepared]
BatchFunction = Callable[[pd.DataFrame, int, _Prepared], _Effect]


@dataclass(frozen=True)
class _Spec:
    label: str
    tier: str
    params_doc: str
    validate: Validator
    batch: BatchFunction | None = None
    prepare: Preparer | None = None
    report_only: bool = False
    sample_all: bool = False
    destructive: bool = False
    confirm_when: str = ""
    no_change: Callable[[dict[str, Any]], str | None] | None = None

    @property
    def feature(self) -> Feature:
        return _FEATURES[self.tier]


class _Accumulator:
    """Sums batch effects and keeps bounded examples and sample rows."""

    def __init__(
        self, failure_column: str | None, sample_rows: int, sample_all: bool, track_quality: bool
    ) -> None:
        self.failure_column = failure_column
        self.sample_limit = sample_rows
        self.sample_all = sample_all
        self.rows_before = 0
        self.rows_after = 0
        self.rows_removed = 0
        self.rows_affected = 0
        self.cells_changed = 0
        self.failure_count = 0
        self.failure_examples: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.sample: list[dict[str, Any]] = []
        self.columns_after: list[str] | None = None
        self.quality = QualityAccumulator() if track_quality else None

    def add(self, batch: pd.DataFrame, effect: _Effect, offset: int) -> None:
        removed = effect.removed
        changed = effect.changed & ~removed
        self.rows_before += len(batch)
        self.rows_after += len(effect.frame)
        self.rows_removed += int(removed.sum())
        self.rows_affected += int((changed | removed).sum())
        self.cells_changed += int(effect.cells_changed)
        for key, value in effect.counts.items():
            self.counts[key] = self.counts.get(key, 0) + int(value)
        if self.columns_after is None:
            self.columns_after = [str(c) for c in effect.frame.columns]
        if effect.failed is not None and self.failure_column is not None:
            self._add_failures(batch, effect.failed, offset)
        if len(self.sample) < self.sample_limit:
            self._add_sample(batch, effect, changed, offset)
        if self.quality is not None:
            self.quality.update(normalize_frame(effect.frame))

    def _add_failures(self, batch: pd.DataFrame, failed: np.ndarray, offset: int) -> None:
        positions = np.flatnonzero(failed)
        self.failure_count += len(positions)
        values = batch[self.failure_column]
        for position in positions[: max(EXAMPLE_LIMIT - len(self.failure_examples), 0)]:
            self.failure_examples.append(
                {"row": offset + int(position) + 1, "value": json_safe(values.iloc[position])}
            )

    def _add_sample(self, batch: pd.DataFrame, effect: _Effect, changed: np.ndarray, offset: int) -> None:
        interesting = changed | effect.removed
        for extra in (effect.failed, effect.marked):
            if extra is not None:
                interesting = interesting | extra
        if self.sample_all:
            interesting = np.ones(len(batch), dtype=bool)
        kept_position = np.cumsum(~effect.removed) - 1
        for position in np.flatnonzero(interesting)[: self.sample_limit - len(self.sample)]:
            after = (
                None if effect.removed[position] else _row_record(effect.frame, int(kept_position[position]))
            )
            self.sample.append(
                {
                    "row": offset + int(position) + 1,
                    "before": _row_record(batch, int(position)),
                    "after": after,
                }
            )

    def failures(self) -> dict[str, Any]:
        return {"count": self.failure_count, "examples": list(self.failure_examples)}

    def stats(self) -> dict[str, int]:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_added": 0,
            "rows_removed": self.rows_removed,
            "rows_affected": self.rows_affected,
            "cells_changed": self.cells_changed,
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _row_record(frame: pd.DataFrame, position: int) -> dict[str, Any]:
    return {str(name): json_safe(frame[name].iloc[position]) for name in frame.columns}


def _display(value: object, limit: int = 60) -> str:
    """Short, control-character-free text for user messages."""
    text = _CONTROL_ANY.sub("", str(value))
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _bool_array(values: pd.Series) -> np.ndarray:
    return values.fillna(False).to_numpy(dtype=bool)


def _diff_mask(before: pd.Series, after: pd.Series) -> np.ndarray:
    """True where the value changed; two missing values count as equal."""
    both_missing = before.isna().to_numpy() & after.isna().to_numpy()
    equal = _bool_array(before.reset_index(drop=True) == after.reset_index(drop=True))
    return ~(equal | both_missing)


def _stripped(series: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Trimmed text values (object dtype) and the mask of empty values."""
    text = series.astype("string").str.strip()
    missing = _bool_array(text.isna() | (text == ""))
    return text.astype(object), missing


def _empty_mask(series: pd.Series) -> np.ndarray:
    """Missing values, plus empty or whitespace-only text values."""
    if pd.api.types.is_string_dtype(series.dtype):
        return _stripped(series)[1]
    return series.isna().to_numpy()


def _slice(series: pd.Series, offset: int, index: pd.Index) -> pd.Series:
    return series.iloc[offset : offset + len(index)].set_axis(index)


def _types(ref: VersionRef) -> dict[str, str]:
    return {column["name"]: column["logical_type"] for column in ref.schema}


def _batches(ref: VersionRef, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    return versioning_service.iter_parquet(ref.path, get_settings().parquet_batch_rows, columns)


def _read_columns(ref: VersionRef, columns: list[str], operation: str) -> pd.DataFrame:
    """Load only ``columns`` of the version after a memory check."""
    resource_manager.check_memory(
        resource_manager.estimate_parquet_memory(ref.path, columns), operation=operation
    )
    return versioning_service.read_parquet(ref.path, columns)


def _convert_present(
    values: pd.Series, missing: np.ndarray, convert: Callable[[Any], Any], dtype: str
) -> tuple[pd.Series, np.ndarray]:
    """Apply ``convert`` to each non-missing value (cached per distinct value).

    ``convert`` returns the new value or ``_FAIL``. Missing values and failures
    become missing in the result; failures are returned as a mask.
    """
    out: list[Any] = [None] * len(values)
    failed = np.zeros(len(values), dtype=bool)
    positions = np.flatnonzero(~missing)
    cache: dict[Any, Any] = {}
    for position, value in zip(positions.tolist(), values.iloc[positions].tolist(), strict=True):
        if value not in cache:
            cache[value] = convert(value)
        result = cache[value]
        if result is _FAIL:
            failed[position] = True
        else:
            out[position] = result
    return pd.Series(out, dtype=dtype), failed


# ---------------------------------------------------------------------------
# Option validation helpers
# ---------------------------------------------------------------------------


def _params_dict(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValidationError("Operation options must be a set of named values.")
    return {str(key): value for key, value in params.items()}


def _check_keys(params: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValidationError(f"Unknown option(s): {', '.join(_display(k) for k in unknown)}.")


def _bool(params: dict[str, Any], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValidationError(f"Option '{key}' must be true or false.")
    return value


def _choice(params: dict[str, Any], key: str, choices: tuple[str, ...], default: str) -> str:
    value = params.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, str) or value not in choices:
        raise ValidationError(f"Option '{key}' must be one of: {', '.join(choices)}.")
    return value


def _column(ref: VersionRef, name: Any, allowed: tuple[str, ...] | None = None, purpose: str = "") -> str:
    if not isinstance(name, str) or not name:
        raise ValidationError("Choose a column.")
    types = _types(ref)
    if name not in types:
        raise ValidationError(f"Unknown column: '{_display(name)}'.")
    if allowed is not None and types[name] not in allowed:
        wanted = "number" if set(allowed) == set(NUMERIC_TYPES) else " or ".join(allowed)
        raise ValidationError(
            f"Column '{_display(name)}' is {types[name]}; {purpose or 'this operation'} needs a {wanted} "
            "column."
        )
    return name


def _columns(
    ref: VersionRef, value: Any, *, allowed: tuple[str, ...] | None, default_text: bool, purpose: str
) -> list[str]:
    """A non-empty list of existing columns, or every text column when omitted."""
    if value is None and default_text:
        names = [name for name, logical in _types(ref).items() if logical == TEXT]
        if not names:
            raise ValidationError("This dataset has no text columns.")
        return names
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not value:
        raise ValidationError("Choose at least one column.")
    names: list[str] = []
    for item in value:
        name = _column(ref, item, allowed, purpose)
        if name not in names:
            names.append(name)
    return names


def _on_failure(params: dict[str, Any]) -> str:
    return _choice(params, "on_failure", ("abort", "set_missing"), "abort")


def _threshold(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError("The threshold must be a number.")
    if not 0 < float(value) <= MAX_THRESHOLD:
        raise ValidationError(f"The threshold must be greater than 0 and at most {MAX_THRESHOLD:g}.")
    return float(value)


def _iter_subpatterns(argument: Any) -> Iterator[Any]:
    """Every regex sub-pattern nested anywhere inside an opcode argument."""
    if isinstance(argument, _sre_parse.SubPattern):
        yield argument
    elif isinstance(argument, (list, tuple)):
        for item in argument:
            yield from _iter_subpatterns(item)


def _has_nested_quantifier(pattern: Any, inside_repeat: bool = False) -> bool:
    """True when a repeat that can match more than once contains another such repeat."""
    for op, argument in pattern:
        if str(op) in _REPEAT_OPS:
            _low, high, body = argument
            repeats = high > 1
            if inside_repeat and repeats:
                return True
            if _has_nested_quantifier(body, inside_repeat or repeats):
                return True
            continue
        for child in _iter_subpatterns(argument):
            if _has_nested_quantifier(child, inside_repeat):
                return True
    return False


def compile_safe_regex(text: Any) -> regex_engine.Pattern[str]:
    """Compile a user regular expression for the ``regex`` engine (``re``-compatible VERSION0 syntax).

    At most MAX_REGEX_LENGTH characters, in the syntax Python's ``re`` accepts.
    Nested quantifiers such as ``(a+)+`` or ``(\\w+\\s?)*`` are refused up front.
    Other slow patterns (overlapping alternations, chains of lazy wildcards)
    cannot all be detected from the syntax, and their cost grows with the
    length of the value, not of the pattern. They are contained at match time
    instead: every match goes through ``_user_fullmatch``, which applies a time
    limit and releases the GIL.
    """
    if not isinstance(text, str) or not text:
        raise ValidationError("Enter a regular expression.")
    if len(text) > MAX_REGEX_LENGTH:
        raise ValidationError(f"Regular expressions can be at most {MAX_REGEX_LENGTH} characters.")
    try:
        if _has_nested_quantifier(_sre_parse.parse(text)):
            raise ValidationError(
                "This regular expression nests repetitions (for example (a+)+), which can take extremely "
                "long "
                "to evaluate. Simplify the pattern."
            )
        return regex_engine.compile(text, flags=regex_engine.VERSION0)
    except (re.error, regex_engine.error, RecursionError, OverflowError, ValueError, TypeError) as exc:
        raise ValidationError("The regular expression is not valid.") from exc


# ---------------------------------------------------------------------------
# Generic batch application
# ---------------------------------------------------------------------------


def _finish(frame: pd.DataFrame, removed: np.ndarray, prepared: _Prepared) -> pd.DataFrame:
    """Apply row removal, column drops and renames to a transformed batch."""
    if removed.any():
        frame = frame.loc[~removed].reset_index(drop=True)
    if prepared.drop:
        frame = frame.drop(columns=prepared.drop)
    if prepared.rename:
        frame = frame.rename(columns=prepared.rename)
    return frame


def _prepared_batch(batch: pd.DataFrame, offset: int, prepared: _Prepared) -> _Effect:
    """Batch function for operations whose effect was computed by ``prepare``."""
    rows = len(batch)
    window = slice(offset, offset + rows)
    out = batch.copy()
    changed = np.zeros(rows, dtype=bool)
    if prepared.replacement is not None and prepared.column is not None:
        out[prepared.column] = _slice(prepared.replacement, offset, batch.index)
        changed = prepared.changed[window].copy() if prepared.changed is not None else changed
    for name, series in prepared.added.items():
        out[name] = _slice(series, offset, batch.index)
        changed = changed | _bool_array(_slice(series, offset, batch.index))
    removed = prepared.removed[window] if prepared.removed is not None else np.zeros(rows, dtype=bool)
    kept_changed = changed & ~removed
    return _Effect(
        frame=_finish(out, removed, prepared),
        changed=kept_changed,
        removed=removed,
        cells_changed=int(kept_changed.sum()),
        failed=None if prepared.failed is None else prepared.failed[window],
        marked=None if prepared.marked is None else prepared.marked[window],
    )


def _text_effect(
    batch: pd.DataFrame, columns: list[str], transform: Callable[[pd.Series], pd.Series]
) -> _Effect:
    out = batch.copy()
    changed = np.zeros(len(batch), dtype=bool)
    cells = 0
    for name in columns:
        new = transform(batch[name]).astype("string")
        diff = _diff_mask(batch[name], new)
        out[name] = new
        changed |= diff
        cells += int(diff.sum())
    return _Effect(frame=out, changed=changed, removed=np.zeros(len(batch), dtype=bool), cells_changed=cells)


# ---------------------------------------------------------------------------
# Standard operations
# ---------------------------------------------------------------------------


def _validate_text_columns(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"columns"})
    return {
        "columns": _columns(
            ref, params.get("columns"), allowed=(TEXT,), default_text=True, purpose="text cleaning"
        )
    }


def _trim_batch(batch: pd.DataFrame, _offset: int, prepared: _Prepared) -> _Effect:
    return _text_effect(batch, prepared.params["columns"], lambda s: s.str.strip())


def _remove_control_characters(series: pd.Series) -> pd.Series:
    """Remove C0/C1 control characters; runs of tabs or line breaks become one space."""
    text = series.astype("string")
    return text.str.replace(_CONTROL_REMOVE, "", regex=True).str.replace(_CONTROL_SPACE, " ", regex=True)


def _control_batch(batch: pd.DataFrame, _offset: int, prepared: _Prepared) -> _Effect:
    return _text_effect(batch, prepared.params["columns"], _remove_control_characters)


def _validate_case(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"columns", "mode"})
    columns = _columns(
        ref, params.get("columns"), allowed=(TEXT,), default_text=True, purpose="case normalization"
    )
    return {"columns": columns, "mode": _choice(params, "mode", ("lower", "upper", "title"), "lower")}


def _case_batch(batch: pd.DataFrame, _offset: int, prepared: _Prepared) -> _Effect:
    mode = prepared.params["mode"]
    return _text_effect(batch, prepared.params["columns"], lambda s: getattr(s.str, mode)())


def _validate_rename(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"mapping"})
    mapping = params.get("mapping")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValidationError("Provide at least one column to rename.")
    clean: dict[str, str] = {}
    for old, new in mapping.items():
        _column(ref, old)
        if not isinstance(new, str) or not new.strip():
            raise ValidationError(f"The new name for '{_display(old)}' must not be blank.")
        name = new.strip()
        if len(name) > MAX_COLUMN_NAME:
            raise ValidationError(f"Column names can be at most {MAX_COLUMN_NAME} characters.")
        if _CONTROL_ANY.search(name):
            raise ValidationError("Column names must not contain control characters.")
        if name != old:
            clean[str(old)] = name
    if not clean:
        raise ValidationError("Every new name equals the current name, so there is nothing to rename.")
    final = [clean.get(name, name) for name in _types(ref)]
    seen: dict[str, str] = {}
    for name in final:
        key = name.casefold()
        if key in seen:
            raise ValidationError(
                f"Renaming would create two columns named '{_display(name)}' (names are case-insensitive)."
            )
        seen[key] = name
    return {"mapping": clean}


def _prepare_rename(_ref: VersionRef, params: dict[str, Any], _guard: OperationGuard) -> _Prepared:
    return _Prepared(params=params, rename=dict(params["mapping"]))


def _validate_drop(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"columns"})
    columns = _columns(ref, params.get("columns"), allowed=None, default_text=False, purpose="drop")
    if len(columns) >= len(ref.schema):
        raise ValidationError("A dataset must keep at least one column.")
    return {"columns": columns}


def _prepare_drop(_ref: VersionRef, params: dict[str, Any], _guard: OperationGuard) -> _Prepared:
    return _Prepared(params=params, drop=list(params["columns"]))


def _validate_remove_empty(params: dict[str, Any], _ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"rows", "columns"})
    clean = {"rows": _bool(params, "rows", True), "columns": _bool(params, "columns", True)}
    if not (clean["rows"] or clean["columns"]):
        raise ValidationError("Choose empty rows, empty columns or both.")
    return clean


def _prepare_remove_empty(ref: VersionRef, params: dict[str, Any], guard: OperationGuard) -> _Prepared:
    prepared = _Prepared(params=params)
    if not params["columns"]:
        return prepared
    names = [column["name"] for column in ref.schema]
    filled = dict.fromkeys(names, 0)
    offset = 0
    with closing(_batches(ref)) as batches:
        for batch in batches:
            guard.step(offset / max(ref.rows, 1), "Looking for empty columns")
            for name in names:
                filled[name] += int((~_empty_mask(batch[name])).sum())
            offset += len(batch)
    empty = [name for name in names if filled[name] == 0]
    if empty and len(empty) == len(names):
        raise ValidationError(
            "Every column is empty, so removing them would leave no data. Delete or archive the dataset "
            "instead."
        )
    prepared.drop = empty
    prepared.details["empty_columns"] = empty
    return prepared


def _remove_empty_batch(batch: pd.DataFrame, _offset: int, prepared: _Prepared) -> _Effect:
    rows = len(batch)
    removed = np.zeros(rows, dtype=bool)
    if prepared.params["rows"] and rows:
        removed = np.column_stack([_empty_mask(batch[name]) for name in batch.columns]).all(axis=1)
    return _Effect(
        frame=_finish(batch.copy(), removed, prepared),
        changed=np.zeros(rows, dtype=bool),
        removed=removed,
        cells_changed=0,
    )


# ---------------------------------------------------------------------------
# Advanced operations: categories, dates, numbers, types
# ---------------------------------------------------------------------------


def _validate_map_categories(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"column", "mapping", "case_insensitive"})
    column = _column(ref, params.get("column"), (TEXT,), "category mapping")
    mapping = params.get("mapping")
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValidationError("Provide at least one mapping from an existing value to a new value.")
    if len(mapping) > MAX_MAPPING_ENTRIES:
        raise ValidationError(f"At most {MAX_MAPPING_ENTRIES} mappings can be applied at once.")
    case_insensitive = _bool(params, "case_insensitive", False)
    clean: dict[str, str] = {}
    folded: dict[str, str] = {}
    for old, new in mapping.items():
        if not isinstance(old, str) or old == "":
            raise ValidationError("Each value to replace must be non-empty text.")
        if not isinstance(new, str) or not new.strip():
            raise ValidationError(f"The replacement for '{_display(old)}' must not be blank.")
        if len(new) > MAX_MAPPED_VALUE:
            raise ValidationError(f"Replacement values can be at most {MAX_MAPPED_VALUE} characters.")
        key = old.casefold() if case_insensitive else old
        if key in folded and clean[folded[key]] != new:
            raise ValidationError(
                f"'{_display(folded[key])}' and '{_display(old)}' match the same values but map to "
                "different results."
            )
        folded.setdefault(key, old)
        clean[old] = new
    return {"column": column, "mapping": clean, "case_insensitive": case_insensitive}


def _prepare_map_categories(_ref: VersionRef, params: dict[str, Any], _guard: OperationGuard) -> _Prepared:
    lookup: dict[str, str] = {}
    for old in params["mapping"]:
        lookup.setdefault(old.casefold() if params["case_insensitive"] else old, old)
    return _Prepared(params=params, column=params["column"], state={"lookup": lookup})


def _map_categories_batch(batch: pd.DataFrame, _offset: int, prepared: _Prepared) -> _Effect:
    column, mapping = prepared.params["column"], prepared.params["mapping"]
    source = batch[column].astype(object)
    keys = (
        source.map(lambda v: v.casefold() if isinstance(v, str) else v)
        if prepared.params["case_insensitive"]
        else source
    )
    entry = keys.map(prepared.state["lookup"])
    matched = _bool_array(entry.notna())
    new = source.copy()
    new[matched] = entry[matched].map(mapping)
    new = new.astype("string")
    diff = _diff_mask(batch[column], new)
    out = batch.copy()
    out[column] = new
    counts = {str(k): int(v) for k, v in entry[diff].value_counts().items()}
    return _Effect(
        frame=out,
        changed=diff,
        removed=np.zeros(len(batch), dtype=bool),
        cells_changed=int(diff.sum()),
        counts=counts,
    )


def _validate_date_format(value: Any) -> str:
    if value is None or value == "auto":
        return "auto"
    if not isinstance(value, str) or len(value) > 40:
        raise ValidationError("Choose 'auto' or a date format such as %d/%m/%Y.")
    if value == ISO_FORMAT or value in DATE_FORMATS:
        return value
    if "%" not in value or not _CUSTOM_DATE_FORMAT.match(value):
        raise ValidationError(
            "The date format may only use %Y %y %m %d %b %B %H %M %S %f %I %p %j %a %A and - / . : , T or "
            "spaces."
        )
    return value


def _default_dayfirst(ref: VersionRef) -> bool:
    with session_scope() as db:
        return bool(org_settings.get_org_settings(db, ref.organization_id)[org_settings.DATE_DAYFIRST])


def _validate_standardize_dates(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"column", "input_format", "dayfirst", "on_failure"})
    column = _column(ref, params.get("column"), (TEXT,), "date standardization")
    dayfirst = params.get("dayfirst")
    return {
        "column": column,
        "input_format": _validate_date_format(params.get("input_format", "auto")),
        "dayfirst": _default_dayfirst(ref) if dayfirst is None else _bool(params, "dayfirst", False),
        "on_failure": _on_failure(params),
    }


def _parse_formatted(values: pd.Series, fmt: str) -> pd.Series:
    try:
        return parse_dates(values, fmt)
    except (ValueError, TypeError) as exc:
        raise ValidationError("The date format could not be used. Check it and try again.") from exc


def _pick_date_format(
    column: str, uniques: pd.Series, weights: np.ndarray, input_format: str, dayfirst: bool
) -> tuple[str | None, bool]:
    """The explicit format, or the format that reads the most values ("auto").

    Ties are resolved by ``type_inference.choose_date_format`` (ISO 8601 first,
    then unambiguous formats, then ``dayfirst``). Values the chosen format
    cannot read are reported as failures.
    """
    if input_format != "auto":
        return input_format, False
    if uniques.empty:
        return None, False
    coverage = {
        fmt: int(weights[_bool_array(parse_dates(uniques, fmt).notna())].sum())
        for fmt in (ISO_FORMAT, *DATE_FORMATS)
    }
    best = max(coverage.values())
    if best == 0:
        raise ValidationError(
            f"None of the values in '{_display(column)}' could be read as dates. Choose the input format "
            "explicitly."
        )
    return choose_date_format({fmt for fmt, count in coverage.items() if count == best}, dayfirst)


def _convert_dates(
    series: pd.Series, column: str, input_format: str, dayfirst: bool
) -> tuple[pd.Series, np.ndarray, dict[str, Any], list[str]]:
    stripped, missing = _stripped(series)
    counts = stripped[~missing].value_counts()
    uniques = pd.Series(counts.index.to_numpy(dtype=object), dtype=object)
    fmt, ambiguous = _pick_date_format(column, uniques, counts.to_numpy(), input_format, dayfirst)
    parsed: dict[Any, Any] = {}
    if fmt is not None and not uniques.empty:
        parsed = dict(zip(uniques.tolist(), _parse_formatted(uniques, fmt).tolist(), strict=True))

    def convert(token: str) -> Any:
        value = parsed.get(token)
        return _FAIL if value is None or pd.isna(value) else value

    converted, failed = _convert_present(stripped, missing, convert, "datetime64[ns]")
    warnings: list[str] = []
    if ambiguous:
        order = "day/month" if dayfirst else "month/day"
        warnings.append(
            f"Day and month order is ambiguous; values were read as {order}. Change 'dayfirst' if this is "
            "wrong."
        )
    if fmt is None:
        warnings.append("Every value is empty, so the column becomes an empty date column.")
    return converted, failed, {"date_format": fmt, "ambiguous": ambiguous}, warnings


def _conversion_prepared(
    params: dict[str, Any], series: pd.Series, converted: pd.Series, failed: np.ndarray
) -> _Prepared:
    """Conversion result: every non-missing source cell changes representation."""
    return _Prepared(
        params=params,
        column=params["column"],
        replacement=converted,
        changed=series.notna().to_numpy(),
        failed=failed,
        state={"source": series},
    )


def _prepare_standardize_dates(ref: VersionRef, params: dict[str, Any], _guard: OperationGuard) -> _Prepared:
    column = params["column"]
    series = _read_columns(ref, [column], "Date standardization")[column]
    converted, failed, details, warnings = _convert_dates(
        series, column, params["input_format"], params["dayfirst"]
    )
    prepared = _conversion_prepared(params, series, converted, failed)
    prepared.details.update(details)
    prepared.warnings.extend(warnings)
    return prepared


@dataclass(frozen=True)
class _Number:
    value: Decimal
    decimal_notation: bool  # decimal point/exponent, currency, or a percentage stored as a fraction
    currency: str | None
    leading_zero: bool
    digits: int


def _parse_number(
    token: str, patterns: Any, *, strip_currency: bool, strip_percent: bool, percent_as_fraction: bool
) -> _Number | None:
    """Parse one trimmed token, or None when it is not an acceptable number."""
    text = token
    percent = False
    match = _PERCENT.match(text)
    if match:
        if not strip_percent:
            return None
        text = f"{match['open']}{match['body']}{match['close']}"
        percent = True
    parsed = parse_numeric_token(text, patterns)
    if parsed is None or (parsed.currency is not None and not strip_currency):
        return None
    try:
        value = Decimal(parsed.normalized)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    fraction = percent and percent_as_fraction
    if fraction:
        value = value / 100
    return _Number(
        value=value,
        decimal_notation=parsed.kind == "dec" or parsed.currency is not None or fraction,
        currency=parsed.currency,
        leading_zero=parsed.leading_zero,
        digits=parsed.significant_digits,
    )


def _as_int(value: Decimal) -> Any:
    """A whole value inside the Int64 range, else _FAIL (range checked before int())."""
    if not INT64_MIN <= value <= INT64_MAX or value != value.to_integral_value():
        return _FAIL
    return int(value)


def _as_float(number: _Number) -> Any:
    """A float that keeps the value exactly as written and is finite, else _FAIL."""
    if not decimal_is_lossless(str(number.value)):
        return _FAIL
    result = float(number.value)
    return result if math.isfinite(result) else _FAIL


def _validate_convert_numeric(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(
        params,
        {
            "column",
            "decimal_separator",
            "thousands_separator",
            "strip_currency",
            "strip_percent",
            "percent_as_fraction",
            "on_failure",
        },
    )
    column = _column(ref, params.get("column"), (TEXT,), "numeric conversion")
    decimal = _choice(params, "decimal_separator", (".", ","), ".")
    thousands = params.get("thousands_separator")
    if thousands is not None and thousands not in (",", ".", " ", "'"):
        raise ValidationError("The thousands separator must be none, ',', '.', a space or an apostrophe.")
    ParseOptions(decimal_separator=decimal, thousands_separator=thousands)  # validates the combination
    return {
        "column": column,
        "decimal_separator": decimal,
        "thousands_separator": thousands,
        "strip_currency": _bool(params, "strip_currency", False),
        "strip_percent": _bool(params, "strip_percent", False),
        "percent_as_fraction": _bool(params, "percent_as_fraction", False),
        "on_failure": _on_failure(params),
    }


def _prepare_convert_numeric(ref: VersionRef, params: dict[str, Any], _guard: OperationGuard) -> _Prepared:
    """Parse the whole column once, then type it: integer unless any value uses decimal notation."""
    column = params["column"]
    series = _read_columns(ref, [column], "Numeric conversion")[column]
    patterns = numeric_patterns(
        ParseOptions(
            decimal_separator=params["decimal_separator"], thousands_separator=params["thousands_separator"]
        )
    )
    stripped, missing = _stripped(series)
    parsed = {
        token: _parse_number(
            token,
            patterns,
            strip_currency=params["strip_currency"],
            strip_percent=params["strip_percent"],
            percent_as_fraction=params["percent_as_fraction"],
        )
        for token in pd.unique(stripped[~missing].to_numpy(dtype=object))
    }
    numbers = [number for number in parsed.values() if number is not None]
    currencies = sorted({number.currency for number in numbers if number.currency})
    if len(currencies) > 1:
        raise ValidationError(
            f"The column mixes currencies ({', '.join(currencies)}). Convert each currency separately."
        )
    as_float = any(number.decimal_notation for number in numbers)

    def convert(token: str) -> Any:
        number = parsed[token]
        if number is None:
            return _FAIL
        return _as_float(number) if as_float else _as_int(number.value)

    converted, failed = _convert_present(stripped, missing, convert, "float64" if as_float else "Int64")
    prepared = _conversion_prepared(params, series, converted, failed)
    prepared.details.update(
        {"result_type": FLOAT if as_float else INTEGER, "currency": currencies[0] if currencies else None}
    )
    leading = sum(
        int(count)
        for token, count in stripped[~missing].value_counts().items()
        if parsed[token] is not None and parsed[token].leading_zero
    )
    if leading:
        prepared.warnings.append(
            f"{leading} value(s) have leading zeros, which a number cannot keep (for example 007 becomes 7)."
        )
    return prepared


_TYPE_TARGETS = (INTEGER, FLOAT, BOOLEAN, DATETIME, TEXT)
_SUPPORTED_CONVERSIONS = {
    TEXT: {INTEGER, FLOAT, BOOLEAN, DATETIME},
    INTEGER: {FLOAT, BOOLEAN, TEXT},
    FLOAT: {INTEGER, BOOLEAN, TEXT},
    BOOLEAN: {INTEGER, FLOAT, TEXT},
    DATETIME: {TEXT},
}


def _validate_convert_types(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"column", "target", "on_failure", "dayfirst"})
    column = _column(ref, params.get("column"))
    target = _choice(params, "target", _TYPE_TARGETS, "text")
    source = _types(ref)[column]
    if source == target:
        raise ValidationError(f"Column '{_display(column)}' is already {target}.")
    if target not in _SUPPORTED_CONVERSIONS[source]:
        raise ValidationError(f"Converting a {source} column to {target} is not supported.")
    return {
        "column": column,
        "source": source,
        "target": target,
        "on_failure": _on_failure(params),
        "dayfirst": _default_dayfirst(ref)
        if params.get("dayfirst") is None
        else _bool(params, "dayfirst", False),
    }


def _text_number(token: str) -> _Number | None:
    patterns = numeric_patterns(ParseOptions())
    return _parse_number(
        token, patterns, strip_currency=False, strip_percent=False, percent_as_fraction=False
    )


def _from_text(token: str, target: str) -> Any:
    if target == BOOLEAN:
        folded = token.casefold()
        if folded in BOOLEAN_TRUE:
            return True
        return False if folded in BOOLEAN_FALSE else _FAIL
    number = _text_number(token)
    if number is None:
        return _FAIL
    return _as_int(number.value) if target == INTEGER else _as_float(number)


def _from_number(value: Any, source: str, target: str) -> Any:
    if target == TEXT:
        return str(int(value)) if source == INTEGER else repr(float(value))
    if target == BOOLEAN:
        if value == 0:
            return False
        return True if value == 1 else _FAIL
    if target == FLOAT:
        return float(value) if abs(int(value)) <= FLOAT_EXACT_INT else _FAIL
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        return _FAIL
    return int(number) if INT64_MIN <= number < 2**63 else _FAIL


def _from_boolean(value: Any, target: str) -> Any:
    flag = bool(value)
    if target == TEXT:
        return "true" if flag else "false"
    return int(flag) if target == INTEGER else float(flag)


def _datetime_text_format(series: pd.Series) -> str:
    """One format for the whole column: date only, seconds, or fractional seconds."""
    present = series.dropna()
    if present.empty:
        return "%Y-%m-%d"
    if ((present.dt.microsecond != 0) | (present.dt.nanosecond != 0)).any():
        return "%Y-%m-%d %H:%M:%S.%f"
    if (present != present.dt.normalize()).any():
        return "%Y-%m-%d %H:%M:%S"
    return "%Y-%m-%d"


def _value_converter(
    series: pd.Series, source: str, target: str, details: dict[str, Any]
) -> Callable[[Any], Any]:
    """Per-value converter for a non-text source column."""
    if source == BOOLEAN:
        return partial(_from_boolean, target=target)
    if source == DATETIME:
        fmt = _datetime_text_format(series)
        details["text_format"] = fmt
        return lambda value: value.strftime(fmt)
    return partial(_from_number, source=source, target=target)


def _prepare_convert_types(ref: VersionRef, params: dict[str, Any], _guard: OperationGuard) -> _Prepared:
    column, source, target = params["column"], params["source"], params["target"]
    series = _read_columns(ref, [column], "Type conversion")[column]
    details: dict[str, Any] = {"source_type": source, "target_type": target}
    warnings: list[str] = []
    dtype = {
        INTEGER: "Int64",
        FLOAT: "float64",
        BOOLEAN: "boolean",
        DATETIME: "datetime64[ns]",
        TEXT: "string",
    }[target]
    if source == TEXT and target == DATETIME:
        converted, failed, date_details, warnings = _convert_dates(series, column, "auto", params["dayfirst"])
        details.update(date_details)
    elif source == TEXT:
        stripped, missing = _stripped(series)
        converted, failed = _convert_present(
            stripped, missing, lambda token: _from_text(token, target), dtype
        )
    else:
        convert = _value_converter(series, source, target, details)
        converted, failed = _convert_present(series.astype(object), series.isna().to_numpy(), convert, dtype)
    prepared = _conversion_prepared(params, series, converted, failed)
    prepared.details.update(details)
    prepared.warnings.extend(warnings)
    return prepared


# ---------------------------------------------------------------------------
# Advanced operations: outliers, invalid values, constraints
# ---------------------------------------------------------------------------


def _validate_outliers(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"column", "method", "threshold", "action"})
    column = _column(ref, params.get("column"), NUMERIC_TYPES, "outlier handling")
    method = _choice(params, "method", ("iqr", "zscore"), "iqr")
    action = _choice(params, "action", ("retain", "cap", "exclude", "flag"), "retain")
    clean = {
        "column": column,
        "method": method,
        "threshold": _threshold(params.get("threshold"), 1.5 if method == "iqr" else 3.0),
        "action": action,
    }
    if action == "flag":
        flag = f"{column}_outlier"
        if flag.casefold() in {name.casefold() for name in _types(ref)}:
            raise ValidationError(
                f"A column named '{_display(flag)}' already exists. Rename it before flagging outliers."
            )
    return clean


def _outlier_bounds(values: pd.Series, method: str, threshold: float) -> dict[str, Any]:
    """Exact bounds over non-missing ``values`` (float64). ``lower``/``upper`` are None when undefined."""
    present = values.dropna().astype("float64")
    bounds: dict[str, Any] = {
        "method": method,
        "threshold": threshold,
        "non_missing": int(len(present)),
        "lower": None,
        "upper": None,
    }
    if method == "iqr":
        if present.empty:
            return bounds
        q1, q3 = float(present.quantile(0.25)), float(present.quantile(0.75))
        spread = q3 - q1
        bounds.update(q1=q1, q3=q3, iqr=spread, lower=q1 - threshold * spread, upper=q3 + threshold * spread)
        return bounds
    if len(present) < 2:
        return bounds
    mean, std = float(present.mean()), float(present.std(ddof=1))
    bounds.update(mean=mean, std=std)
    if std > 0 and math.isfinite(std):
        bounds.update(lower=mean - threshold * std, upper=mean + threshold * std)
    return bounds


def _prepare_outliers(ref: VersionRef, params: dict[str, Any], _guard: OperationGuard) -> _Prepared:
    column, action = params["column"], params["action"]
    series = _read_columns(ref, [column], "Outlier detection")[column]
    numbers = series.astype("float64")
    bounds = _outlier_bounds(numbers, params["method"], params["threshold"])
    prepared = _Prepared(params=params, column=column)
    lower, upper = bounds["lower"], bounds["upper"]
    if lower is None:
        mask = np.zeros(len(series), dtype=bool)
        prepared.warnings.append(
            "Outlier bounds are undefined for this column (too few values or no variation), so no outliers "
            "were found."
        )
    else:
        mask = _bool_array((numbers < lower) | (numbers > upper))
    if bounds.get("iqr") == 0 and mask.any():
        prepared.warnings.append(
            "The interquartile range is zero, so every value different from the typical value counts as an "
            "outlier."
        )
    prepared.marked = mask
    prepared.details.update(bounds=bounds, outliers=int(mask.sum()))
    if action == "cap":
        lo, hi = lower, upper
        if _types(ref)[column] == INTEGER and lower is not None:
            lo, hi = math.ceil(lower), math.floor(upper)
            if lo > hi:
                raise ValidationError(
                    "The outlier bounds contain no whole number, so this integer column cannot be capped. "
                    "Use a larger threshold."
                )
            prepared.details["integer_bounds"] = {"lower": lo, "upper": hi}
            prepared.warnings.append(
                "Bounds were rounded inward to whole numbers so the column stays integer."
            )
        capped = series.clip(lower=lo, upper=hi) if lower is not None else series
        prepared.replacement = series.mask(pd.Series(mask, index=series.index), capped)
        prepared.changed = mask & _diff_mask(series, prepared.replacement)
    elif action == "exclude":
        prepared.removed = mask
    elif action == "flag":
        flags = pd.Series(mask, dtype="boolean")
        flags[series.isna().to_numpy()] = pd.NA
        prepared.added[f"{column}_outlier"] = flags
    return prepared


def _validate_invalid_values(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"column", "pattern", "regex", "action"})
    column = _column(ref, params.get("column"), (TEXT,), "pattern validation")
    pattern = _choice(params, "pattern", ("email", "regex"), "email")
    regex = params.get("regex")
    if pattern == "regex":
        compile_safe_regex(regex)
    elif regex is not None:
        raise ValidationError("A custom regular expression is only used with pattern 'regex'.")
    return {
        "column": column,
        "pattern": pattern,
        "regex": regex,
        "action": _choice(params, "action", ("report", "set_missing"), "report"),
    }


def _prepare_invalid_values(_ref: VersionRef, params: dict[str, Any], guard: OperationGuard) -> _Prepared:
    regex = _EMAIL if params["pattern"] == "email" else compile_safe_regex(params["regex"])
    prepared = _Prepared(params=params, column=params["column"], state={"regex": regex, "guard": guard})
    prepared.details["pattern"] = EMAIL_REGEX if params["pattern"] == "email" else params["regex"]
    return prepared


def _guard_remaining(guard: OperationGuard) -> float | None:
    """Seconds left before ``guard`` times out, or None without a time limit."""
    return guard.remaining()


def _user_fullmatch(
    pattern: regex_engine.Pattern[str], token: str, guard: OperationGuard, column: str
) -> bool:
    """Whether ``token`` fully matches a user pattern, within the time limits.

    The match runs with ``concurrent=True`` (the GIL is released, so other
    sessions keep running) and a timeout of REGEX_MATCH_SECONDS, shortened to
    the operation's remaining time. The first timeout aborts the whole
    operation: ValidationError when the pattern is too slow for one value,
    OperationTimeoutError when the operation as a whole ran out of time.
    """
    guard.check()
    if len(token) > MAX_REGEX_VALUE_LENGTH:
        raise ValidationError(
            f"Column '{_display(column)}' has values longer than {MAX_REGEX_VALUE_LENGTH:,} characters. "
            f"Custom regular expressions can only check values up to {MAX_REGEX_VALUE_LENGTH:,} "
            "characters; use the e-mail pattern or shorten the values first."
        )
    remaining = _guard_remaining(guard)
    if remaining is not None and remaining <= 0:
        raise OperationTimeoutError()
    limited_by_guard = remaining is not None and remaining < REGEX_MATCH_SECONDS
    budget = remaining if limited_by_guard else REGEX_MATCH_SECONDS
    try:
        return pattern.fullmatch(token, timeout=budget, concurrent=True) is not None
    except TimeoutError as exc:
        if limited_by_guard:
            raise OperationTimeoutError() from exc
        raise ValidationError(
            f"The regular expression took too long to check a value in column '{_display(column)}' "
            f"(more than {REGEX_MATCH_SECONDS:g} seconds), so the operation was stopped. Simplify the "
            "pattern, for example by removing repeated '.*' or overlapping alternatives, and try again."
        ) from exc


def _invalid_mask(
    series: pd.Series,
    regex: re.Pattern[str] | regex_engine.Pattern[str],
    guard: OperationGuard | None = None,
) -> np.ndarray:
    """Non-empty values whose trimmed text does not fully match ``regex``.

    Built-in patterns (``guard`` None) match directly. User patterns pass the
    operation's guard and go through ``_user_fullmatch``.
    """
    stripped, missing = _stripped(series)
    tokens = pd.unique(stripped[~missing].to_numpy(dtype=object))
    if guard is None:
        verdicts = {token: regex.fullmatch(token) is None for token in tokens}
    else:
        column = str(series.name)
        verdicts = {token: not _user_fullmatch(regex, token, guard, column) for token in tokens}
    invalid = np.zeros(len(series), dtype=bool)
    positions = np.flatnonzero(~missing)
    invalid[positions] = [verdicts[token] for token in stripped.iloc[positions].tolist()]
    return invalid


def _invalid_values_batch(batch: pd.DataFrame, _offset: int, prepared: _Prepared) -> _Effect:
    column = prepared.params["column"]
    user_pattern = prepared.params["pattern"] == "regex"
    guard = prepared.state["guard"] if user_pattern else None
    invalid = _invalid_mask(batch[column], prepared.state["regex"], guard)
    out = batch.copy()
    changed = np.zeros(len(batch), dtype=bool)
    if prepared.params["action"] == "set_missing":
        out[column] = batch[column].mask(pd.Series(invalid, index=batch.index))
        changed = invalid
    return _Effect(
        frame=out,
        changed=changed,
        removed=np.zeros(len(batch), dtype=bool),
        cells_changed=int(changed.sum()),
        failed=invalid,
        counts={"invalid": int(invalid.sum())},
    )


_RULE_KEYS = {"column", "not_null", "unique", "min", "max", "allowed_values", "regex"}


def _rule_value(value: Any, logical: str, column: str, option: str) -> Any:
    """Validate a bound or allowed value against the column type."""
    if logical in NUMERIC_TYPES:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValidationError(f"'{option}' for '{_display(column)}' must be a number.")
        return value
    if logical == DATETIME:
        try:
            stamp = pd.Timestamp(value) if isinstance(value, str) else None
        except (ValueError, TypeError):
            stamp = None
        if stamp is None or pd.isna(stamp):
            raise ValidationError(f"'{option}' for '{_display(column)}' must be a date such as 2024-01-31.")
        return stamp.isoformat()
    if logical == BOOLEAN:
        if not isinstance(value, bool):
            raise ValidationError(f"'{option}' for '{_display(column)}' must be true or false.")
        return value
    if not isinstance(value, str):
        raise ValidationError(f"'{option}' for '{_display(column)}' must be text.")
    return value


def _validate_rule(rule: Any, ref: VersionRef) -> dict[str, Any]:
    if not isinstance(rule, Mapping):
        raise ValidationError("Each constraint must be a set of named options.")
    rule = {str(key): value for key, value in rule.items()}
    _check_keys(rule, _RULE_KEYS)
    column = _column(ref, rule.get("column"))
    logical = _types(ref)[column]
    clean: dict[str, Any] = {
        "column": column,
        "not_null": _bool(rule, "not_null", False),
        "unique": _bool(rule, "unique", False),
    }
    for option in ("min", "max"):
        if rule.get(option) is not None:
            if logical not in (*NUMERIC_TYPES, DATETIME):
                raise ValidationError(f"'{option}' only applies to number or date columns.")
            clean[option] = _rule_value(rule[option], logical, column, option)
    allowed = rule.get("allowed_values")
    if allowed is not None:
        if isinstance(allowed, str) or not isinstance(allowed, (list, tuple)) or not allowed:
            raise ValidationError("'allowed_values' must be a non-empty list.")
        if len(allowed) > MAX_ALLOWED_VALUES:
            raise ValidationError(f"At most {MAX_ALLOWED_VALUES} allowed values can be listed.")
        clean["allowed_values"] = [_rule_value(v, logical, column, "allowed_values") for v in allowed]
    if rule.get("regex") is not None:
        if logical != TEXT:
            raise ValidationError("'regex' only applies to text columns.")
        compile_safe_regex(rule["regex"])
        clean["regex"] = rule["regex"]
    has_value_check = any(option in clean for option in ("min", "max", "allowed_values", "regex"))
    if not (clean["not_null"] or clean["unique"] or has_value_check):
        raise ValidationError(f"The constraint for '{_display(column)}' has no checks.")
    return clean


def _validate_constraints(params: dict[str, Any], ref: VersionRef) -> dict[str, Any]:
    _check_keys(params, {"rules"})
    rules = params.get("rules")
    if isinstance(rules, (str, Mapping)) or not isinstance(rules, (list, tuple)) or not rules:
        raise ValidationError("Provide at least one constraint.")
    if len(rules) > MAX_RULES:
        raise ValidationError(f"At most {MAX_RULES} constraints can be checked at once.")
    return {"rules": [_validate_rule(rule, ref) for rule in rules]}


def _rule_violations(
    series: pd.Series, rule: dict[str, Any], logical: str, guard: OperationGuard
) -> list[tuple[str, np.ndarray]]:
    """(check name, violation mask) for each check in ``rule``."""
    empty = _empty_mask(series)
    present = ~empty
    comparable = series.astype("datetime64[ns]") if logical == DATETIME else series
    bound = (lambda v: pd.Timestamp(v)) if logical == DATETIME else (lambda v: v)
    checks: list[tuple[str, np.ndarray]] = []
    if rule["not_null"]:
        checks.append(("not_null", empty))
    if rule["unique"]:
        key = _stripped(series)[0] if logical == TEXT else series
        checks.append(("unique", present & key.duplicated(keep=False).to_numpy()))
    if "min" in rule:
        checks.append(("min", present & _bool_array(comparable < bound(rule["min"]))))
    if "max" in rule:
        checks.append(("max", present & _bool_array(comparable > bound(rule["max"]))))
    if "allowed_values" in rule:
        allowed = [bound(v) for v in rule["allowed_values"]]
        checks.append(("allowed_values", present & ~_bool_array(comparable.isin(allowed))))
    if "regex" in rule:
        checks.append(("regex", _invalid_mask(series, compile_safe_regex(rule["regex"]), guard)))
    return checks


def _constraint_report(
    ref: VersionRef, rules: list[dict[str, Any]], guard: OperationGuard
) -> tuple[list[dict[str, Any]], np.ndarray]:
    columns = list(dict.fromkeys(rule["column"] for rule in rules))
    frame = _read_columns(ref, columns, "Constraint validation")
    types = _types(ref)
    any_violation = np.zeros(len(frame), dtype=bool)
    report: list[dict[str, Any]] = []
    for rule in rules:
        guard.check()
        series = frame[rule["column"]]
        results = []
        for check, mask in _rule_violations(series, rule, types[rule["column"]], guard):
            any_violation |= mask
            positions = np.flatnonzero(mask)[:RULE_EXAMPLE_LIMIT]
            examples = [{"row": int(p) + 1, "value": json_safe(series.iloc[p])} for p in positions]
            results.append({"check": check, "count": int(mask.sum()), "examples": examples})
        violations = sum(item["count"] for item in results)
        report.append(
            {
                "column": rule["column"],
                "rule": _rule_summary(rule),
                "checks": results,
                "violations": violations,
                "passed": violations == 0,
            }
        )
    return report, any_violation


def _rule_summary(rule: dict[str, Any]) -> dict[str, Any]:
    """The rule without disabled checks, for the report."""
    return {key: value for key, value in rule.items() if value is not False and value is not None}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _retain_reason(params: dict[str, Any]) -> str | None:
    if params.get("action") == "retain":
        return (
            "Action 'retain' only reports outliers and never changes data. Use the preview or the outlier "
            "report."
        )
    return None


def _report_reason(params: dict[str, Any]) -> str | None:
    if params.get("action") == "report":
        return (
            "Action 'report' only lists invalid values and never changes data. Use the preview to see them."
        )
    return None


_SPECS: dict[str, _Spec] = {
    "trim_whitespace": _Spec(
        label="Trim leading and trailing whitespace",
        tier="standard",
        params_doc="columns: list of text columns (default: every text column)",
        validate=_validate_text_columns,
        batch=_trim_batch,
    ),
    "remove_control_chars": _Spec(
        label="Remove control characters",
        tier="standard",
        params_doc=(
            "columns: list of text columns (default: every text column). Tabs and line breaks become single "
            "spaces."
        ),
        validate=_validate_text_columns,
        batch=_control_batch,
    ),
    "rename_columns": _Spec(
        label="Rename columns",
        tier="standard",
        params_doc=(
            "mapping: {old name: new name}; new names non-blank, unique ignoring case, at most 120 characters"
        ),
        validate=_validate_rename,
        prepare=_prepare_rename,
        batch=_prepared_batch,
        sample_all=True,
    ),
    "remove_empty": _Spec(
        label="Remove fully empty rows and columns",
        tier="standard",
        params_doc=(
            "rows: bool (default true), columns: bool (default true); empty means missing, empty or "
            "whitespace-only"
        ),
        validate=_validate_remove_empty,
        prepare=_prepare_remove_empty,
        batch=_remove_empty_batch,
        destructive=True,
        confirm_when="always",
    ),
    "normalize_case": _Spec(
        label="Normalize letter case",
        tier="standard",
        params_doc="columns: list of text columns (default: every text column), mode: lower | upper | title",
        validate=_validate_case,
        batch=_case_batch,
    ),
    "drop_columns": _Spec(
        label="Drop columns",
        tier="standard",
        params_doc="columns: list of columns to remove (at least one column must remain)",
        validate=_validate_drop,
        prepare=_prepare_drop,
        batch=_prepared_batch,
        sample_all=True,
        destructive=True,
        confirm_when="always",
    ),
    "map_categories": _Spec(
        label="Map category labels",
        tier="advanced",
        params_doc=(
            "column: text column, mapping: {old value: new value}, case_insensitive: bool (default false); "
            "unmapped values are untouched"
        ),
        validate=_validate_map_categories,
        prepare=_prepare_map_categories,
        batch=_map_categories_batch,
    ),
    "standardize_dates": _Spec(
        label="Standardize dates",
        tier="advanced",
        params_doc=(
            "column: text column, input_format: 'auto' or a format such as %d/%m/%Y, dayfirst: bool "
            "(default: organization setting), on_failure: abort | set_missing"
        ),
        validate=_validate_standardize_dates,
        prepare=_prepare_standardize_dates,
        batch=_prepared_batch,
        confirm_when="on_failure='set_missing'",
    ),
    "convert_numeric": _Spec(
        label="Convert text to numbers",
        tier="advanced",
        params_doc=(
            "column: text column, decimal_separator: '.' | ',', thousands_separator: none | ',' | '.' | ' ' "
            '| "\'", '
            "strip_currency: bool, strip_percent: bool, percent_as_fraction: bool, on_failure: abort | "
            "set_missing"
        ),
        validate=_validate_convert_numeric,
        prepare=_prepare_convert_numeric,
        batch=_prepared_batch,
        confirm_when="on_failure='set_missing'",
    ),
    "convert_types": _Spec(
        label="Convert column type",
        tier="advanced",
        params_doc=(
            "column, target: integer | float | boolean | datetime | text, on_failure: abort | set_missing, "
            "dayfirst: bool for text to datetime (default: organization setting)"
        ),
        validate=_validate_convert_types,
        prepare=_prepare_convert_types,
        batch=_prepared_batch,
        confirm_when="on_failure='set_missing'",
    ),
    "handle_outliers": _Spec(
        label="Handle outliers",
        tier="advanced",
        params_doc=(
            "column: number column, method: iqr | zscore, threshold: k (default 1.5) or z (default 3.0), "
            "action: retain | cap | exclude | flag"
        ),
        validate=_validate_outliers,
        prepare=_prepare_outliers,
        batch=_prepared_batch,
        confirm_when="action='exclude'",
        no_change=_retain_reason,
    ),
    "fix_invalid_values": _Spec(
        label="Validate e-mail addresses and patterns",
        tier="advanced",
        params_doc=(
            "column: text column, pattern: email | regex, regex: pattern (at most 200 characters, full "
            "match), action: report | set_missing"
        ),
        validate=_validate_invalid_values,
        prepare=_prepare_invalid_values,
        batch=_invalid_values_batch,
        confirm_when="action='set_missing'",
        no_change=_report_reason,
    ),
    "validate_constraints": _Spec(
        label="Validate column constraints",
        tier="advanced",
        params_doc="rules: [{column, not_null, unique, min, max, allowed_values, regex}] (read-only report)",
        validate=_validate_constraints,
        report_only=True,
    ),
}

OPERATIONS: dict[str, dict[str, Any]] = {
    name: {
        "label": spec.label,
        "feature": spec.tier,
        "destructive": spec.destructive,
        "confirm_when": spec.confirm_when,
        "read_only": spec.report_only,
        "params": spec.params_doc,
    }
    for name, spec in _SPECS.items()
}


def list_operations() -> list[dict[str, Any]]:
    """Every cleaning operation with its label, plan tier and options (for the UI)."""
    return [{"name": name, **meta} for name, meta in OPERATIONS.items()]


def requires_confirmation(operation: str, params: Mapping[str, Any] | None) -> bool:
    """Whether applying ``operation`` with ``params`` needs ``confirm=True``."""
    spec = _spec(operation)
    options = _params_dict(params)
    return (
        spec.destructive
        or (operation == "handle_outliers" and options.get("action") == "exclude")
        or (operation == "fix_invalid_values" and options.get("action") == "set_missing")
        or options.get("on_failure") == "set_missing"
    )


def _spec(operation: Any) -> _Spec:
    if not isinstance(operation, str) or operation not in _SPECS:
        raise ValidationError("Unknown cleaning operation.")
    return _SPECS[operation]


# ---------------------------------------------------------------------------
# Running operations
# ---------------------------------------------------------------------------


def _prepare(spec: _Spec, ref: VersionRef, params: dict[str, Any], guard: OperationGuard) -> _Prepared:
    if spec.prepare is not None:
        return spec.prepare(ref, params, guard)
    column = params.get("column")
    return _Prepared(params=params, column=column if isinstance(column, str) else None)


def _simulate(
    ref: VersionRef, spec: _Spec, prepared: _Prepared, *, sample_rows: int, guard: OperationGuard
) -> _Accumulator:
    """Run the batch function over the version without writing anything."""
    accumulator = _Accumulator(prepared.column, sample_rows, spec.sample_all, track_quality=True)
    offset = 0
    with closing(_batches(ref)) as batches:
        for batch in batches:
            guard.step(offset / max(ref.rows, 1), f"Previewing rows {offset + 1:,}-{offset + len(batch):,}")
            assert spec.batch is not None
            accumulator.add(batch, spec.batch(batch, offset, prepared), offset)
            offset += len(batch)
    return accumulator


def _column_changes(ref: VersionRef, after: list[str], rename: dict[str, str]) -> dict[str, Any]:
    before = [column["name"] for column in ref.schema]
    mapped = [rename.get(name, name) for name in before]
    return {
        "columns_added": [name for name in after if name not in mapped],
        "columns_removed": [name for name in before if rename.get(name, name) not in after],
        "columns_renamed": dict(rename),
    }


def _failure_message(column: str | None, failures: dict[str, Any]) -> str:
    examples = ", ".join(
        f"row {item['row']}: '{_display(item['value'], 30)}'" for item in failures["examples"][:5]
    )
    return (
        f"{failures['count']} value(s) in column '{_display(column)}' could not be converted, so nothing "
        "was changed. "
        f"Examples: {examples}. Fix these values first or choose on_failure 'set_missing'."
    )


def _sample_limit(sample_rows: Any) -> int:
    if (
        isinstance(sample_rows, bool)
        or not isinstance(sample_rows, int)
        or not 0 <= sample_rows <= MAX_SAMPLE_ROWS
    ):
        raise ValidationError(f"The sample size must be between 0 and {MAX_SAMPLE_ROWS} rows.")
    return sample_rows


def _constraints_preview(
    ref: VersionRef, operation: str, params: dict[str, Any], guard: OperationGuard
) -> dict[str, Any]:
    report, any_violation = _constraint_report(ref, params["rules"], guard)
    examples = [
        {"row": example["row"], "value": example["value"], "column": rule["column"], "check": check["check"]}
        for rule in report
        for check in rule["checks"]
        for example in check["examples"]
    ]
    total = sum(rule["violations"] for rule in report)
    return {
        "operation": operation,
        "label": _SPECS[operation].label,
        "dataset_id": ref.dataset_id,
        "version_number": ref.version_number,
        "params": params,
        "rows_before": ref.rows,
        "rows_after": ref.rows,
        "rows_added": 0,
        "rows_removed": 0,
        "rows_affected": 0,
        "cells_changed": 0,
        "columns_before": ref.columns,
        "columns_after": ref.columns,
        "columns_added": [],
        "columns_removed": [],
        "columns_renamed": {},
        "failures": {"count": total, "examples": examples[:EXAMPLE_LIMIT]},
        "constraints": report,
        "rows_with_violations": int(any_violation.sum()),
        "passed": total == 0,
        "counts": {},
        "details": {},
        "sample": [],
        "warnings": [],
        "requires_confirmation": False,
        "can_apply": False,
        "quality_before": ref.quality_score,
        "quality_after": ref.quality_score,
    }


def preview_cleaning(
    ctx: AuthContext,
    dataset_id: int,
    operation: str,
    params: Mapping[str, Any] | None,
    *,
    sample_rows: int = 20,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Compute the exact effect of a cleaning operation on the current version without saving.

    Requires dataset access and the operation's plan feature. Returns rows
    before/after, cells_changed, rows_affected, column changes, failures
    (count and examples with 1-based row numbers), up to ``sample_rows``
    before/after rows, warnings, the quality score before and after, and
    whether the operation can be applied as configured.
    """
    spec = _spec(operation)
    ref = versioning_service.get_version_ref(ctx, dataset_id, feature=spec.feature)
    limit = _sample_limit(sample_rows)
    clean = spec.validate(_params_dict(params), ref)
    guard = guard or OperationGuard()
    if spec.report_only:
        return _constraints_preview(ref, operation, clean, guard)
    prepared = _prepare(spec, ref, clean, guard)
    accumulator = _simulate(ref, spec, prepared, sample_rows=limit, guard=guard)
    failures = accumulator.failures()
    warnings = list(prepared.warnings)
    blocked = spec.no_change(clean) if spec.no_change else None
    if blocked:
        warnings.append(blocked)
    aborts = clean.get("on_failure") == "abort" and failures["count"] > 0
    if aborts:
        warnings.append(_failure_message(prepared.column, failures))
    quality = accumulator.quality.finalize() if accumulator.quality is not None else None
    return {
        "operation": operation,
        "label": spec.label,
        "dataset_id": ref.dataset_id,
        "version_number": ref.version_number,
        "params": clean,
        **accumulator.stats(),
        "columns_before": ref.columns,
        "columns_after": len(accumulator.columns_after or []),
        **_column_changes(ref, accumulator.columns_after or [], prepared.rename),
        "failures": failures,
        "counts": dict(accumulator.counts),
        "details": _json_details(prepared.details),
        "sample": accumulator.sample,
        "warnings": warnings,
        "requires_confirmation": requires_confirmation(operation, clean),
        "can_apply": not blocked and not aborts,
        "quality_before": ref.quality_score,
        "quality_after": quality.score if quality is not None else None,
    }


def _json_details(details: dict[str, Any]) -> dict[str, Any]:
    """Operation context (bounds, chosen format) converted to plain JSON values."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return json_safe(value)

    return convert(details)


def _ensure_current(ctx: AuthContext, dataset_id: int, ref: VersionRef, feature: Feature) -> None:
    current = versioning_service.get_version_ref(
        ctx, dataset_id, permission=Permission.CLEAN_DATA, feature=feature
    )
    if current.version_id != ref.version_id:
        raise ConflictError(
            "The dataset changed while this operation was being prepared. Review the latest version and try "
            "again."
        )


def apply_cleaning(
    ctx: AuthContext,
    dataset_id: int,
    operation: str,
    params: Mapping[str, Any] | None,
    *,
    confirm: bool = False,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Apply a cleaning operation and publish the result as a new version.

    Requires the ``clean_data`` permission and the operation's plan feature.
    Destructive configurations need ``confirm=True``. With on_failure "abort"
    any unconvertible value refuses the whole operation (recorded in the
    history) and nothing is published. The data streams batch by batch; the
    source version is never modified and stays restorable. Returns the new
    version number, statistics, failures, column changes and the quality
    score before and after.
    """
    spec = _spec(operation)
    ref = versioning_service.get_version_ref(
        ctx, dataset_id, permission=Permission.CLEAN_DATA, feature=spec.feature
    )
    if spec.report_only:
        raise ValidationError(
            "Constraint validation is a read-only report and changes nothing. Use the preview to see the "
            "results."
        )
    clean = spec.validate(_params_dict(params), ref)
    blocked = spec.no_change(clean) if spec.no_change else None
    if blocked:
        raise ValidationError(blocked)
    if requires_confirmation(operation, clean) and confirm is not True:
        raise ValidationError(
            "This operation removes or blanks data. Review the preview and confirm to continue."
        )
    guard = guard or OperationGuard()
    prepared = _prepare(spec, ref, clean, guard)
    if clean.get("on_failure") == "abort" and prepared.failure_count:
        failures = _prepared_failures(prepared)
        error = ValidationError(_failure_message(prepared.column, failures), details={"failures": failures})
        versioning_service.record_failed_operation(ctx, dataset_id, operation, clean, error)
        raise error

    accumulator = _Accumulator(prepared.column, 0, False, track_quality=False)
    summary: dict[str, Any] = {}  # filled while streaming; serialized when the version is published

    def transform_batch(batch: pd.DataFrame, offset: int) -> tuple[pd.DataFrame, dict[str, int]]:
        if offset == 0:
            _ensure_current(ctx, dataset_id, ref, spec.feature)
        assert spec.batch is not None
        effect = spec.batch(batch, offset, prepared)
        accumulator.add(batch, effect, offset)
        summary.update(
            failures=accumulator.failures(),
            counts=dict(accumulator.counts),
            details=_json_details(prepared.details),
        )
        kept_changed = effect.changed & ~effect.removed
        return effect.frame, {
            "rows_removed": int(effect.removed.sum()),
            "rows_affected": int((kept_changed | effect.removed).sum()),
            "cells_changed": int(effect.cells_changed),
        }

    result = versioning_service.apply_streaming_operation(
        ctx,
        dataset_id,
        operation_type=operation,
        parameters=clean,
        transform_batch=transform_batch,
        feature=spec.feature,
        guard=guard,
        summary=summary,
    )
    return {
        "dataset_id": result.dataset_id,
        "version_id": result.version_id,
        "version_number": result.version_number,
        "operation": operation,
        "stats": accumulator.stats(),
        "rows_before": ref.rows,
        "rows_after": result.rows,
        "columns_before": ref.columns,
        "columns_after": result.columns,
        **_column_changes(ref, accumulator.columns_after or [], prepared.rename),
        "failures": accumulator.failures(),
        "counts": dict(accumulator.counts),
        "details": _json_details(prepared.details),
        "warnings": list(prepared.warnings),
        "quality_before": result.quality_before,
        "quality_after": result.quality_score,
    }


def _prepared_failures(prepared: _Prepared) -> dict[str, Any]:
    """Failure count and examples from the source column kept by a conversion's prepare step."""
    assert prepared.failed is not None
    positions = np.flatnonzero(prepared.failed)
    values = prepared.state["source"]
    return {
        "count": len(positions),
        "examples": [
            {"row": int(p) + 1, "value": json_safe(values.iloc[p])} for p in positions[:EXAMPLE_LIMIT]
        ],
    }


# ---------------------------------------------------------------------------
# Read-only detection reports (dataset access + STANDARD_CLEANING)
# ---------------------------------------------------------------------------


def detect_casing_issues(
    ctx: AuthContext, dataset_id: int, *, guard: OperationGuard | None = None
) -> list[dict[str, Any]]:
    """Groups of text values that differ only in letter case, per text column (exact counts).

    Each group lists its variants with counts and the most common variant as a
    suggestion for review. A column with more than DISTINCT_LIMIT distinct
    case-folded values is reported as skipped instead of being approximated.
    """
    ref = versioning_service.get_version_ref(ctx, dataset_id, feature=Feature.STANDARD_CLEANING)
    guard = guard or OperationGuard()
    columns = [name for name, logical in _types(ref).items() if logical == TEXT]
    if not columns:
        return []
    tables: dict[str, dict[str, Counter[str]]] = {name: {} for name in columns}
    skipped: set[str] = set()
    offset = 0
    with closing(_batches(ref, columns)) as batches:
        for batch in batches:
            guard.step(offset / max(ref.rows, 1), "Checking letter case")
            for name in columns:
                if name in skipped:
                    continue
                table = tables[name]
                for value, count in batch[name].dropna().value_counts().items():
                    table.setdefault(str(value).casefold(), Counter())[str(value)] += int(count)
                if len(table) > DISTINCT_LIMIT:
                    skipped.add(name)
                    table.clear()
            offset += len(batch)
    findings: list[dict[str, Any]] = []
    for name in columns:
        if name in skipped:
            findings.append(
                {
                    "column": name,
                    "kind": "skipped",
                    "reason": f"More than {DISTINCT_LIMIT:,} distinct values; not checked.",
                }
            )
            continue
        groups = [variants for variants in tables[name].values() if len(variants) > 1]
        groups.sort(key=lambda variants: (-sum(variants.values()), min(variants)))
        for variants in groups:
            ordered = sorted(variants.items(), key=lambda item: (-item[1], item[0]))
            findings.append(
                {
                    "column": name,
                    "kind": "casing_variants",
                    "variants": [{"value": value, "count": count} for value, count in ordered],
                    "total": sum(variants.values()),
                    "suggested": ordered[0][0],
                    "status": "for review",
                }
            )
    return findings


def _equal_with_missing(left: pd.Series, right: pd.Series) -> bool:
    left_missing, right_missing = left.isna().to_numpy(), right.isna().to_numpy()
    if (left_missing != right_missing).any():
        return False
    return bool(
        (left[~left_missing].to_numpy(dtype=object) == right[~right_missing].to_numpy(dtype=object)).all()
    )


def _redundancy_scan(
    ref: VersionRef, guard: OperationGuard
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[str, Any], set[str], Counter[str]]:
    """Stream the version once: identical pairs, case-insensitive pairs and constant columns."""
    types = _types(ref)
    names = list(types)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if types[a] == types[b]]
    identical = set(pairs)
    folded = {(a, b) for a, b in pairs if types[a] == TEXT}
    first: dict[str, Any] = {}
    varied: set[str] = set()
    present: Counter[str] = Counter()
    offset = 0
    with closing(_batches(ref)) as batches:
        for batch in batches:
            guard.step(offset / max(ref.rows, 1), "Comparing columns")
            for a, b in list(identical):
                if not _equal_with_missing(batch[a], batch[b]):
                    identical.discard((a, b))
            for a, b in list(folded):
                if not _equal_with_missing(batch[a].str.casefold(), batch[b].str.casefold()):
                    folded.discard((a, b))
            for name in names:
                if name in varied:
                    continue
                values = pd.unique(batch[name].dropna().to_numpy(dtype=object))
                present[name] += int(batch[name].notna().sum())
                if len(values) > 1 or (len(values) == 1 and name in first and first[name] != values[0]):
                    varied.add(name)
                elif len(values) == 1:
                    first.setdefault(name, values[0])
            offset += len(batch)
    return identical, folded - identical, first, varied, present


def identify_redundant_columns(
    ctx: AuthContext, dataset_id: int, *, guard: OperationGuard | None = None
) -> list[dict[str, Any]]:
    """Columns that may be redundant, each for review (nothing is changed).

    Reports identical columns (same values and missing positions), text
    columns identical after case folding, empty and constant columns, and
    number column pairs whose Pearson correlation over complete pairs (at least
    3) satisfies |r| >= 0.999.
    """
    ref = versioning_service.get_version_ref(ctx, dataset_id, feature=Feature.STANDARD_CLEANING)
    guard = guard or OperationGuard()
    if ref.rows == 0:
        return []
    identical, folded, first, varied, present = _redundancy_scan(ref, guard)
    findings: list[dict[str, Any]] = []
    for a, b in sorted(identical):
        findings.append(
            {
                "kind": "identical",
                "columns": [a, b],
                "detail": "Both columns contain exactly the same values.",
                "status": "for review",
            }
        )
    for a, b in sorted(folded):
        findings.append(
            {
                "kind": "identical_ignoring_case",
                "columns": [a, b],
                "detail": "The columns differ only in letter case.",
                "status": "for review",
            }
        )
    for name in _types(ref):
        if present[name] == 0:
            findings.append(
                {
                    "kind": "empty",
                    "columns": [name],
                    "detail": "Every value is missing.",
                    "status": "for review",
                }
            )
        elif name not in varied:
            findings.append(
                {
                    "kind": "constant",
                    "columns": [name],
                    "value": json_safe(first[name]),
                    "missing": ref.rows - present[name],
                    "detail": "Every non-missing value is the same.",
                    "status": "for review",
                }
            )
    numeric = [name for name, logical in _types(ref).items() if logical in NUMERIC_TYPES]
    if len(numeric) >= 2:
        frame = _read_columns(ref, numeric, "Correlation check").astype("float64")
        matrix = frame.corr(min_periods=3)
        for i, a in enumerate(numeric):
            for b in numeric[i + 1 :]:
                value = matrix.loc[a, b]
                if pd.notna(value) and abs(value) >= CORRELATION_THRESHOLD and (a, b) not in identical:
                    findings.append(
                        {
                            "kind": "high_correlation",
                            "columns": [a, b],
                            "correlation": round(float(value), 6),
                            "detail": "Almost perfectly correlated; one column may derive from the other.",
                            "status": "for review",
                        }
                    )
    return findings


def outlier_report(
    ctx: AuthContext,
    dataset_id: int,
    column: str,
    *,
    method: str = "iqr",
    threshold: float | None = None,
    sample_rows: int = 20,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Exact outlier bounds, count and sample rows for one number column. Never changes data."""
    ref = versioning_service.get_version_ref(ctx, dataset_id, feature=Feature.STANDARD_CLEANING)
    limit = _sample_limit(sample_rows)
    guard = guard or OperationGuard()
    spec = _SPECS["handle_outliers"]
    params = spec.validate(
        {"column": column, "method": method, "threshold": threshold, "action": "retain"}, ref
    )
    prepared = _prepare(spec, ref, params, guard)
    accumulator = _simulate(ref, spec, prepared, sample_rows=limit, guard=guard)
    bounds = prepared.details["bounds"]
    count = prepared.details["outliers"]
    return {
        "dataset_id": ref.dataset_id,
        "version_number": ref.version_number,
        "column": column,
        "method": params["method"],
        "threshold": params["threshold"],
        "bounds": _json_details(bounds),
        "count": count,
        "non_missing": bounds["non_missing"],
        "percent": round(100 * count / bounds["non_missing"], 2) if bounds["non_missing"] else 0.0,
        "sample": [{"row": item["row"], "values": item["before"]} for item in accumulator.sample],
        "warnings": list(prepared.warnings),
    }
