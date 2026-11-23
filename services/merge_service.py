"""Join two datasets on one or more key columns (inner, left, right or outer).

Semantics:

* Keys are matched exactly. Integer and float keys are compatible (compared
  as integers when both are integer, otherwise as floats); text, datetime and
  boolean keys only match keys of the same type. Text vs numeric keys are
  rejected with a suggestion to convert the type first.
* A row whose key has a missing value in any key column never matches
  anything (unlike pandas' default, where missing keys match each other). Such
  rows are kept as unmatched rows by the join types that keep unmatched rows
  of their side (left: left rows; right: right rows; outer: both).
* Before anything runs, the exact output row count is computed from the key
  value counts without performing the merge: inner = sum over keys of
  L(k) * R(k); left = inner + unmatched left rows (null keys included);
  right is symmetric; outer = inner + unmatched left + unmatched right.
* When a key pair has the same name on both sides, the key appears once in the
  output (coalesced); otherwise both key columns are kept. Other columns
  present on both sides (compared case-insensitively) receive the suffixes.
* Unmatched counts per side are always reported, including the rows an inner,
  left or right join excludes, so nothing is discarded silently.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from auth.authorization import AuthContext, authorize
from config.settings import get_settings
from core.constants import PANDAS_DTYPE_FOR, Feature, LogicalType, Permission
from core.errors import AppError, ResourceLimitError, ValidationError
from database.connection import session_scope
from services import resource_manager, versioning_service
from services.append_service import (
    create_derived_dataset,
    dataset_summary,
    default_dataset_name,
    ensure_unchanged,
    publish_result,
    record_operation_failure,
    require_confirmation,
    resolve_pair,
    resource_report,
    schema_types,
    validate_choice,
    validate_sample_rows,
    version_file_bytes,
)
from services.frame_utils import hash_rows, normalize_series, records
from services.resource_manager import MB, OperationGuard
from services.versioning_service import VersionRef, iter_parquet, read_parquet

JOIN_TYPES = ("inner", "left", "right", "outer")
OUTPUT_MODES = ("new_dataset", "new_version")
GROWTH_WARNING_FACTOR = 1.5
MAX_SUFFIX_LENGTH = 30
# Preview: rows read per side are capped at this multiple of the requested sample size.
PREVIEW_ROW_BUDGET_FACTOR = 5

_NUMERIC = frozenset({LogicalType.INTEGER.value, LogicalType.FLOAT.value})
# Internal column names start with a NUL character, which never occurs in stored column names.
_KEY_PREFIX = "\x00merge_key_"
_RIGHT_KEY_PREFIX = "\x00merge_right_"


# ---------------------------------------------------------------------------
# Validation and output layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyPair:
    left: str
    right: str
    left_type: str
    right_type: str
    match_type: str

    @property
    def shared(self) -> bool:
        """Same name on both sides: the key appears once in the output."""
        return self.left == self.right

    def to_dict(self) -> dict[str, str]:
        return {
            "left_column": self.left,
            "right_column": self.right,
            "left_type": self.left_type,
            "right_type": self.right_type,
            "match_type": self.match_type,
        }


@dataclass(frozen=True)
class MergeLayout:
    """Output column names and dtypes, decided from the two schemas only."""

    left_names: dict[str, str]  # left column -> output name
    right_names: dict[str, str]  # right column (shared keys excluded) -> output name
    dtypes: dict[str, str]  # output name -> storage dtype
    suffixed: list[dict[str, str]] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return list(self.left_names.values()) + list(self.right_names.values())


def _key_list(value: Sequence[str] | str, side: str) -> list[str]:
    keys = [value] if isinstance(value, str) else list(value or [])
    if not keys or not all(isinstance(k, str) and k for k in keys):
        raise ValidationError(f"Choose at least one {side} key column.")
    if len(set(keys)) != len(keys):
        raise ValidationError(f"The {side} key columns must not repeat.")
    return keys


def _match_type(left: str, right: str, left_type: str, right_type: str) -> str:
    if left_type == right_type:
        return left_type
    if {left_type, right_type} <= _NUMERIC:
        return LogicalType.FLOAT.value
    raise ValidationError(
        f"Key '{left}' is {left_type} but key '{right}' is {right_type}, so they can never match. "
        "Convert one of them to the same type first (for example with the type conversion cleaning step)."
    )


def validate_keys(
    left_types: dict[str, str],
    right_types: dict[str, str],
    left_on: Sequence[str],
    right_on: Sequence[str],
    how: str,
) -> list[KeyPair]:
    """Check the join type, key counts, key existence and key type compatibility."""
    validate_choice(how, JOIN_TYPES, "The join type")
    left_keys, right_keys = _key_list(left_on, "left"), _key_list(right_on, "right")
    if len(left_keys) != len(right_keys):
        raise ValidationError("Choose the same number of key columns on both sides.")
    missing = [f"left '{k}'" for k in left_keys if k not in left_types]
    missing += [f"right '{k}'" for k in right_keys if k not in right_types]
    if missing:
        raise ValidationError(f"Key column(s) not found: {', '.join(missing)}.")
    return [
        KeyPair(lk, rk, left_types[lk], right_types[rk], _match_type(lk, rk, left_types[lk], right_types[rk]))
        for lk, rk in zip(left_keys, right_keys, strict=True)
    ]


def _validate_suffixes(suffixes: tuple[str, str]) -> tuple[str, str]:
    if not isinstance(suffixes, (tuple, list)) or len(suffixes) != 2:
        raise ValidationError("Provide two suffixes, one for each side.")
    left, right = suffixes
    if not all(isinstance(s, str) and s.strip() and len(s) <= MAX_SUFFIX_LENGTH for s in (left, right)):
        raise ValidationError(f"Suffixes must be non-empty and at most {MAX_SUFFIX_LENGTH} characters.")
    if left.casefold() == right.casefold():
        raise ValidationError("The two suffixes must be different.")
    return left, right


def build_layout(
    left_types: dict[str, str], right_types: dict[str, str], keys: list[KeyPair], suffixes: tuple[str, str]
) -> MergeLayout:
    """Name every output column; overlapping non-coalesced names get the suffixes."""
    left_suffix, right_suffix = _validate_suffixes(suffixes)
    shared = {pair.right for pair in keys if pair.shared}
    right_columns = [name for name in right_types if name not in shared]
    overlap = {n.casefold() for n in left_types} & {n.casefold() for n in right_columns}
    left_names = {n: f"{n}{left_suffix}" if n.casefold() in overlap else n for n in left_types}
    right_names = {n: f"{n}{right_suffix}" if n.casefold() in overlap else n for n in right_columns}
    match_types = {pair.left: pair.match_type for pair in keys if pair.shared}
    dtypes = {out: PANDAS_DTYPE_FOR[match_types.get(src, left_types[src])] for src, out in left_names.items()}
    dtypes.update({out: PANDAS_DTYPE_FOR[right_types[src]] for src, out in right_names.items()})
    names = list(left_names.values()) + list(right_names.values())
    if len({n.casefold() for n in names}) != len(names):
        raise ValidationError("The suffixes would create duplicate column names. Choose different suffixes.")
    suffixed = [
        {"left_column": n, "right_column": r, "left_output": left_names[n], "right_output": right_names[r]}
        for n in left_types
        for r in right_columns
        if n.casefold() == r.casefold()
    ]
    return MergeLayout(left_names=left_names, right_names=right_names, dtypes=dtypes, suffixed=suffixed)


# ---------------------------------------------------------------------------
# Canonical keys and exact counting
# ---------------------------------------------------------------------------


def _key_names(count: int) -> list[str]:
    return [f"{_KEY_PREFIX}{i}__" for i in range(count)]


def canonical_keys(frame: pd.DataFrame, columns: list[str], keys: list[KeyPair]) -> pd.DataFrame:
    """Key columns cast to their common match type, under internal names shared by both sides."""
    data = {}
    for name, column, pair in zip(_key_names(len(keys)), columns, keys, strict=True):
        series = frame[column]
        if pair.match_type == LogicalType.FLOAT.value:
            series = series.astype("float64")
        data[name] = normalize_series(series).reset_index(drop=True)
    return pd.DataFrame(data, index=pd.RangeIndex(len(frame)))


def null_key_mask(keys: pd.DataFrame) -> np.ndarray:
    """True where any key component is missing (such rows never match)."""
    return keys.isna().any(axis=1).to_numpy(dtype=bool)


def key_counts(keys: pd.DataFrame) -> pd.DataFrame:
    """Occurrences of each distinct non-null key value (column ``n``)."""
    valid = keys[~null_key_mask(keys)]
    counts = valid.groupby(list(valid.columns), sort=False).size().rename("n").reset_index()
    for name in keys.columns:
        counts[name] = counts[name].astype(keys[name].dtype)
    return counts


@dataclass(frozen=True)
class KeyStatistics:
    """Exact join statistics derived from key value counts only."""

    left_rows: int
    right_rows: int
    left_null_keys: int
    right_null_keys: int
    left_duplicate_keys: int
    right_duplicate_keys: int
    left_duplicate_key_rows: int
    right_duplicate_key_rows: int
    matched_keys: int
    many_to_many_keys: int
    left_matched_rows: int
    right_matched_rows: int
    inner_rows: int

    @property
    def left_unmatched(self) -> int:
        return self.left_rows - self.left_matched_rows

    @property
    def right_unmatched(self) -> int:
        return self.right_rows - self.right_matched_rows

    def output_rows(self, how: str) -> int:
        extra_left = self.left_unmatched if how in ("left", "outer") else 0
        extra_right = self.right_unmatched if how in ("right", "outer") else 0
        return self.inner_rows + extra_left + extra_right

    @property
    def relationship(self) -> str:
        left = "many" if self.left_duplicate_keys else "one"
        right = "many" if self.right_duplicate_keys else "one"
        return f"{left}-to-{right}"


def _as_int(values: pd.Series) -> int:
    """Exact integer sum using Python integers (no overflow)."""
    return int(values.astype(object).sum()) if len(values) else 0


def key_statistics(left_keys: pd.DataFrame, right_keys: pd.DataFrame) -> tuple[KeyStatistics, pd.DataFrame]:
    """Compute exact statistics and the per-key count table (merged counts of matched keys)."""
    left_counts, right_counts = key_counts(left_keys), key_counts(right_keys)
    matched = left_counts.merge(right_counts, on=list(left_keys.columns), how="inner", suffixes=("_l", "_r"))
    stats = KeyStatistics(
        left_rows=len(left_keys),
        right_rows=len(right_keys),
        left_null_keys=int(null_key_mask(left_keys).sum()),
        right_null_keys=int(null_key_mask(right_keys).sum()),
        left_duplicate_keys=int((left_counts["n"] > 1).sum()),
        right_duplicate_keys=int((right_counts["n"] > 1).sum()),
        left_duplicate_key_rows=_as_int(left_counts.loc[left_counts["n"] > 1, "n"]),
        right_duplicate_key_rows=_as_int(right_counts.loc[right_counts["n"] > 1, "n"]),
        matched_keys=len(matched),
        many_to_many_keys=int(((matched["n_l"] > 1) & (matched["n_r"] > 1)).sum()),
        left_matched_rows=_as_int(matched["n_l"]),
        right_matched_rows=_as_int(matched["n_r"]),
        inner_rows=_as_int(matched["n_l"].astype(object) * matched["n_r"].astype(object)),
    )
    return stats, _count_table(left_counts, right_counts, list(left_keys.columns))


def _count_table(left_counts: pd.DataFrame, right_counts: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Outer join of the two count tables (n_l / n_r are 0 for a side without the key)."""
    table = left_counts.merge(right_counts, on=names, how="outer", suffixes=("_l", "_r"))
    table["n_l"] = table["n_l"].fillna(0).astype("int64")
    table["n_r"] = table["n_r"].fillna(0).astype("int64")
    for name in names:
        table[name] = table[name].astype(left_counts[name].dtype)
    return table


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class MergeAnalysis:
    left: VersionRef
    right: VersionRef
    how: str
    keys: list[KeyPair]
    layout: MergeLayout
    stats: KeyStatistics
    counts: pd.DataFrame
    resource_check: dict[str, Any]
    estimated_memory_bytes: int
    estimated_output_bytes: int
    combined_input_bytes: int

    @property
    def output_rows(self) -> int:
        return self.stats.output_rows(self.how)

    @property
    def growth_factor(self) -> float:
        return self.output_rows / max(self.stats.left_rows, self.stats.right_rows, 1)

    def unmatched(self) -> dict[str, Any]:
        """Unmatched rows per side and whether the join keeps or excludes them."""
        keeps_left = self.how in ("left", "outer")
        keeps_right = self.how in ("right", "outer")
        return {
            "left_unmatched_rows": self.stats.left_unmatched,
            "right_unmatched_rows": self.stats.right_unmatched,
            "left_null_key_rows": self.stats.left_null_keys,
            "right_null_key_rows": self.stats.right_null_keys,
            "left_unmatched_kept": self.stats.left_unmatched if keeps_left else 0,
            "right_unmatched_kept": self.stats.right_unmatched if keeps_right else 0,
            "left_rows_excluded": 0 if keeps_left else self.stats.left_unmatched,
            "right_rows_excluded": 0 if keeps_right else self.stats.right_unmatched,
        }

    def warnings(self) -> list[str]:
        stats, items = self.stats, []
        if stats.relationship == "many-to-many":
            items.append(
                "Both sides have duplicate keys (many-to-many): every matching pair of rows is combined, "
                "which multiplies rows. Confirm explicitly to continue."
            )
        if self.growth_factor > GROWTH_WARNING_FACTOR:
            items.append(
                f"The result will have {self.output_rows:,} rows, {self.growth_factor:.1f} times the larger input."
            )
        if stats.left_null_keys or stats.right_null_keys:
            items.append(
                f"Rows with a missing key never match (left: {stats.left_null_keys:,}, "
                f"right: {stats.right_null_keys:,})."
            )
        unmatched = self.unmatched()
        if unmatched["left_rows_excluded"] or unmatched["right_rows_excluded"]:
            items.append(
                f"This {self.how} join leaves out {unmatched['left_rows_excluded']:,} unmatched left row(s) and "
                f"{unmatched['right_rows_excluded']:,} unmatched right row(s)."
            )
        for item in self.layout.suffixed:
            items.append(
                f"Column '{item['left_column']}' exists on both sides and becomes "
                f"'{item['left_output']}' and '{item['right_output']}'."
            )
        if not self.resource_check["ok"]:
            items.append(self.resource_check["message"])
        return items

    def to_dict(self) -> dict[str, Any]:
        stats = self.stats
        by_join = {how: stats.output_rows(how) for how in JOIN_TYPES}
        return {
            "left": dataset_summary(self.left),
            "right": dataset_summary(self.right),
            "how": self.how,
            "left_on": [k.left for k in self.keys],
            "right_on": [k.right for k in self.keys],
            "key_types": [k.to_dict() for k in self.keys],
            "left_rows": stats.left_rows,
            "right_rows": stats.right_rows,
            "left_null_key_rows": stats.left_null_keys,
            "right_null_key_rows": stats.right_null_keys,
            "left_duplicate_keys": stats.left_duplicate_keys,
            "right_duplicate_keys": stats.right_duplicate_keys,
            "left_duplicate_key_rows": stats.left_duplicate_key_rows,
            "right_duplicate_key_rows": stats.right_duplicate_key_rows,
            "relationship": stats.relationship,
            "many_to_many": stats.relationship == "many-to-many",
            "many_to_many_keys": stats.many_to_many_keys,
            "matched_keys": stats.matched_keys,
            "left_matched_rows": stats.left_matched_rows,
            "right_matched_rows": stats.right_matched_rows,
            "left_unmatched_rows": stats.left_unmatched,
            "right_unmatched_rows": stats.right_unmatched,
            "unmatched": self.unmatched(),
            "output_rows": self.output_rows,
            "output_rows_by_join": by_join,
            "row_count_is_exact": True,
            "growth_factor": round(self.growth_factor, 4),
            "overlapping_columns": list(self.layout.suffixed),
            "output_columns": self.layout.columns,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "estimated_output_bytes": self.estimated_output_bytes,
            "combined_input_bytes": self.combined_input_bytes,
            "input_target_bytes": get_settings().append_merge_input_target_mb * MB,
            "resource_check": self.resource_check,
            "warnings": self.warnings(),
        }


def _load_keys(ref: VersionRef, columns: list[str], keys: list[KeyPair]) -> pd.DataFrame:
    """Load only the key columns (memory checked) and return their canonical form."""
    resource_manager.check_memory(
        resource_manager.estimate_parquet_memory(ref.path, columns) * 2, operation="Inspecting the merge keys"
    )
    return canonical_keys(read_parquet(ref.path, columns), columns, keys)


def _merge_resources(
    left: VersionRef, right: VersionRef, output_rows: int, growth: float
) -> tuple[int, int, int, Any]:
    """Memory/disk estimate for the in-memory merge and a check raising ResourceLimitError."""
    settings = get_settings()
    inputs = resource_manager.estimate_parquet_memory(left.path) + resource_manager.estimate_parquet_memory(
        right.path
    )
    output_bytes = int(inputs * max(growth, 1.0))
    memory = inputs + output_bytes
    combined = version_file_bytes(left) + version_file_bytes(right)
    target = settings.append_merge_input_target_mb * MB

    def check() -> None:
        if output_rows > settings.merge_max_output_rows:
            raise ResourceLimitError(
                f"The merge would produce {output_rows:,} rows, above the limit of "
                f"{settings.merge_max_output_rows:,} rows. Filter the data or use keys without duplicates."
            )
        if combined > target:
            raise ResourceLimitError(
                f"The two datasets together use {resource_manager.human_bytes(combined)}, above the "
                f"{resource_manager.human_bytes(target)} limit for merges. Reduce the data first."
            )
        resource_manager.check_memory(memory, operation="Merging these datasets")
        resource_manager.check_disk(int(combined * max(growth, 1.0)) * 2, operation="Merging these datasets")

    return memory, output_bytes, combined, check


def _analyze(
    left: VersionRef,
    right: VersionRef,
    left_on: Sequence[str],
    right_on: Sequence[str],
    how: str,
    suffixes: tuple[str, str],
) -> MergeAnalysis:
    left_types, right_types = schema_types(left), schema_types(right)
    keys = validate_keys(left_types, right_types, left_on, right_on, how)
    layout = build_layout(left_types, right_types, keys, suffixes)
    limit = get_settings().max_columns
    if len(layout.columns) > limit:
        raise ValidationError(
            f"The merged dataset would have {len(layout.columns):,} columns, above the limit of {limit:,} "
            "columns. Remove unused columns from one of the datasets first."
        )
    left_keys = _load_keys(left, [k.left for k in keys], keys)
    right_keys = _load_keys(right, [k.right for k in keys], keys)
    stats, counts = key_statistics(left_keys, right_keys)
    output_rows = stats.output_rows(how)
    growth = output_rows / max(stats.left_rows, stats.right_rows, 1)
    memory, output_bytes, combined, check = _merge_resources(left, right, output_rows, growth)
    return MergeAnalysis(
        left=left,
        right=right,
        how=how,
        keys=keys,
        layout=layout,
        stats=stats,
        counts=counts,
        resource_check=resource_report(check),
        estimated_memory_bytes=memory,
        estimated_output_bytes=output_bytes,
        combined_input_bytes=combined,
    )


# ---------------------------------------------------------------------------
# Performing the merge
# ---------------------------------------------------------------------------


def _typed(frame: pd.DataFrame, columns: list[str], dtypes: dict[str, str]) -> pd.DataFrame:
    """Reindex to ``columns`` with exact dtypes; missing columns become typed nulls."""
    data = {}
    for name in columns:
        dtype = dtypes[name]
        if name in frame.columns:
            data[name] = frame[name].reset_index(drop=True).astype(dtype)
        else:
            data[name] = pd.Series([None] * len(frame), dtype=dtype)
    return pd.DataFrame(data, index=pd.RangeIndex(len(frame)))


def perform_merge(
    left: pd.DataFrame, right: pd.DataFrame, keys: list[KeyPair], layout: MergeLayout, how: str
) -> pd.DataFrame:
    """Merge with null-key semantics: null keys never match; unmatched rows kept per join type."""
    key_names = _key_names(len(keys))
    right_key_names = {pair.right: f"{_RIGHT_KEY_PREFIX}{pair.right}__" for pair in keys if pair.shared}
    left_frame = left.rename(columns=layout.left_names).reset_index(drop=True)
    right_frame = right.rename(columns={**layout.right_names, **right_key_names})[
        list(layout.right_names.values()) + list(right_key_names.values())
    ].reset_index(drop=True)
    left_keys = canonical_keys(left, [k.left for k in keys], keys)
    right_keys = canonical_keys(right, [k.right for k in keys], keys)
    left_frame = pd.concat([left_frame, left_keys], axis=1)
    right_frame = pd.concat([right_frame, right_keys], axis=1)
    left_null, right_null = null_key_mask(left_keys), null_key_mask(right_keys)

    matched = pd.merge(left_frame[~left_null], right_frame[~right_null], on=key_names, how=how, sort=False)
    parts = [matched]
    if how in ("left", "outer"):
        parts.append(left_frame[left_null])
    if how in ("right", "outer"):
        parts.append(right_frame[right_null])
    dtypes = dict(layout.dtypes)
    helper_columns = list(right_key_names.values())
    for pair in keys:
        if pair.shared:
            dtypes[right_key_names[pair.right]] = PANDAS_DTYPE_FOR[pair.match_type]
    all_columns = layout.columns + helper_columns
    typed_parts = [_typed(part, all_columns, dtypes) for part in parts if len(part)]
    result = (
        pd.concat(typed_parts, ignore_index=True) if typed_parts else _typed(matched, all_columns, dtypes)
    )
    for pair in keys:
        if pair.shared:
            output = layout.left_names[pair.left]
            result[output] = result[output].where(result[output].notna(), result[right_key_names[pair.right]])
    return result[layout.columns]


# ---------------------------------------------------------------------------
# Preview sampling (a bounded subset of keys, never the full data)
# ---------------------------------------------------------------------------


def _choose_sample_keys(analysis: MergeAnalysis, sample_rows: int) -> pd.DataFrame:
    """Pick whole keys whose rows fit the preview budget so the sampled merge rows are exact."""
    budget = sample_rows * PREVIEW_ROW_BUDGET_FACTOR
    table = analysis.counts
    matched = table[(table["n_l"] > 0) & (table["n_r"] > 0)]
    left_only = (
        table[(table["n_l"] > 0) & (table["n_r"] == 0)]
        if analysis.how in ("left", "outer")
        else table.iloc[0:0]
    )
    right_only = (
        table[(table["n_l"] == 0) & (table["n_r"] > 0)]
        if analysis.how in ("right", "outer")
        else table.iloc[0:0]
    )
    chosen, used, produced = [], 0, 0
    for group in (matched, left_only, right_only):
        # Keys too large for the budget are skipped; at most sample_rows candidates per group are examined.
        fitting = group[(group["n_l"] + group["n_r"]) <= budget].head(sample_rows)
        for index, n_left, n_right in zip(fitting.index, fitting["n_l"], fitting["n_r"], strict=True):
            rows_needed = int(n_left) + int(n_right)
            if used + rows_needed > budget or produced >= sample_rows:
                break
            chosen.append(index)
            used += rows_needed
            produced += max(int(n_left), 1) * max(int(n_right), 1)
    return table.loc[chosen]


def _collect_rows(
    ref: VersionRef, key_columns: list[str], keys: list[KeyPair], wanted: np.ndarray, null_rows: int
) -> pd.DataFrame:
    """Stream one dataset and keep rows whose key hash is wanted, plus up to ``null_rows`` null-key rows.

    Memory stays bounded by one batch plus the kept rows, whose number the
    caller limits by choosing whole keys within the preview budget.
    """
    kept: list[pd.DataFrame] = []
    template: pd.DataFrame | None = None
    nulls = 0
    batches = iter_parquet(ref.path, get_settings().parquet_batch_rows)
    try:
        for batch in batches:
            batch = batch.reset_index(drop=True)
            if template is None:
                template = batch.iloc[0:0]
            canonical = canonical_keys(batch, key_columns, keys)
            null_mask = null_key_mask(canonical)
            selected = np.isin(hash_rows(canonical), wanted) & ~null_mask
            take_nulls = np.flatnonzero(null_mask)[: max(null_rows - nulls, 0)]
            selected[take_nulls] = True
            nulls += len(take_nulls)
            if selected.any():
                kept.append(batch[selected])
    finally:
        batches.close()
    if not kept:
        assert template is not None  # iter_parquet always yields at least one (possibly empty) batch
        return template
    return pd.concat(kept, ignore_index=True)


def _sample_merge(analysis: MergeAnalysis, sample_rows: int) -> pd.DataFrame:
    chosen = _choose_sample_keys(analysis, sample_rows)
    names = _key_names(len(analysis.keys))
    wanted = hash_rows(chosen[names]) if len(chosen) else np.empty(0, dtype=np.uint64)
    left_nulls = sample_rows if analysis.how in ("left", "outer") else 0
    right_nulls = sample_rows if analysis.how in ("right", "outer") else 0
    left = _collect_rows(analysis.left, [k.left for k in analysis.keys], analysis.keys, wanted, left_nulls)
    right = _collect_rows(
        analysis.right, [k.right for k in analysis.keys], analysis.keys, wanted, right_nulls
    )
    return perform_merge(left, right, analysis.keys, analysis.layout, analysis.how).head(sample_rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _unmatched_row(analysis: MergeAnalysis, side: str) -> dict[str, Any]:
    """One side's unmatched rows and how many of them the chosen join keeps or excludes."""
    unmatched = analysis.unmatched()
    return {
        "side": side,
        "unmatched_rows": unmatched[f"{side}_unmatched_rows"],
        "null_key_rows": unmatched[f"{side}_null_key_rows"],
        "kept_in_output": unmatched[f"{side}_unmatched_kept"],
        "excluded_from_output": unmatched[f"{side}_rows_excluded"],
    }


def analyze_merge(
    ctx: AuthContext,
    left_id: int,
    right_id: int,
    *,
    left_on: list[str],
    right_on: list[str],
    how: str,
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> dict[str, Any]:
    """Validate the keys and compute the exact output size and risks without merging.

    Loads only the key columns. Raises ValidationError for an invalid join
    type, unequal key counts, unknown keys or incompatible key types. The
    merge's resource check (memory, disk, row ceiling, combined-input limit)
    is reported as ``{"ok", "message"}`` rather than raised; only when even
    the key columns cannot be loaded safely is ResourceLimitError raised.
    """
    left, right = resolve_pair(ctx, left_id, right_id, permission=None)
    return _analyze(left, right, left_on, right_on, how, suffixes).to_dict()


def preview_merge(
    ctx: AuthContext,
    left_id: int,
    right_id: int,
    *,
    left_on: list[str],
    right_on: list[str],
    how: str,
    suffixes: tuple[str, str] = ("_left", "_right"),
    sample_rows: int = 20,
) -> dict[str, Any]:
    """The analysis plus output columns and sample rows from merging a small subset of keys.

    Sample keys are chosen whole (all their rows on both sides) so every
    sample row is a real output row; at most a bounded number of rows per
    side is kept while the datasets are streamed. Nothing is written.
    """
    left, right = resolve_pair(ctx, left_id, right_id, permission=None)
    validate_sample_rows(sample_rows)
    analysis = _analyze(left, right, left_on, right_on, how, suffixes)
    sample = _sample_merge(analysis, sample_rows)
    result = analysis.to_dict()
    result.update(
        {
            "columns": [{"name": n, "dtype": analysis.layout.dtypes[n]} for n in analysis.layout.columns],
            "sample_rows": records(sample, sample_rows),
            "sample_is_subset": True,
            "unmatched_row_counts": [_unmatched_row(analysis, "left"), _unmatched_row(analysis, "right")],
        }
    )
    return result


def apply_merge(
    ctx: AuthContext,
    left_id: int,
    right_id: int,
    *,
    left_on: list[str],
    right_on: list[str],
    how: str,
    suffixes: tuple[str, str] = ("_left", "_right"),
    output: str = "new_dataset",
    new_dataset_name: str | None = None,
    confirm: bool = False,
    confirm_many_to_many: bool = False,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Merge the datasets and publish a new dataset (default) or a new version of the left dataset.

    Requires ``confirm=True``, plus ``confirm_many_to_many=True`` for a
    many-to-many relationship. Rejects with ResourceLimitError when memory,
    disk, the combined-input limit or MERGE_MAX_OUTPUT_ROWS would be exceeded.
    The produced row count must equal the exact analysis estimate, otherwise
    nothing is saved. Failures are recorded in the left dataset's history.
    """
    left, right = resolve_pair(ctx, left_id, right_id, permission=Permission.CLEAN_DATA)
    validate_choice(output, OUTPUT_MODES, "The output")
    require_confirmation(confirm, "merge these datasets")
    started = time.monotonic()
    guard = guard or OperationGuard()
    analysis = _analyze(left, right, left_on, right_on, how, suffixes)
    if analysis.stats.relationship == "many-to-many" and confirm_many_to_many is not True:
        raise ValidationError(
            "Both datasets have duplicate keys (many-to-many), which multiplies rows "
            f"({analysis.output_rows:,} output rows). Confirm the many-to-many merge to continue."
        )
    parameters: dict[str, Any] = {
        "right_dataset_id": right.dataset_id,
        "right_version": right.version_number,
        "left_version": left.version_number,
        "how": how,
        "left_on": [k.left for k in analysis.keys],
        "right_on": [k.right for k in analysis.keys],
        "suffixes": list(suffixes),
        "output": output,
    }
    try:
        if not analysis.resource_check["ok"]:
            raise ResourceLimitError(analysis.resource_check["message"])
        guard.step(0.1, "Loading the left dataset")
        left_frame = read_parquet(left.path)
        guard.step(0.3, "Loading the right dataset")
        right_frame = read_parquet(right.path)
        guard.step(0.5, "Merging")
        merged = perform_merge(left_frame, right_frame, analysis.keys, analysis.layout, how)
        del left_frame, right_frame
        if len(merged) != analysis.output_rows:
            raise AppError(
                "The merge produced an unexpected number of rows, so nothing was saved. Please report this problem."
            )
        guard.step(0.8, "Saving the result")
        summary = {
            "left": dataset_summary(left),
            "right": dataset_summary(right),
            "relationship": analysis.stats.relationship,
            "output_rows": len(merged),
            "suffixed_columns": analysis.layout.suffixed,
            **analysis.unmatched(),
        }
        published = _publish_merge(
            ctx, analysis, merged, output, new_dataset_name, parameters, summary, started, guard=guard
        )
    except BaseException as exc:
        record_operation_failure(ctx, left.dataset_id, "merge", parameters, exc)
        raise
    return {
        **published,
        "relationship": analysis.stats.relationship,
        "summary": summary,
        **analysis.unmatched(),
    }


def _publish_merge(
    ctx: AuthContext,
    analysis: MergeAnalysis,
    merged: pd.DataFrame,
    output: str,
    new_dataset_name: str | None,
    parameters: dict[str, Any],
    summary: dict[str, Any],
    started: float,
    *,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    with session_scope() as db:
        if output == "new_version":
            dataset = ensure_unchanged(db, ctx, analysis.left)
            result = versioning_service.publish_version(
                db,
                ctx,
                dataset,
                operation_type="merge",
                parameters=parameters,
                frame=merged,
                summary=summary,
                note=f"Merged with '{analysis.right.dataset_name}' ({analysis.how} join)",
                started=started,
                guard=guard,
            )
            return publish_result(result, output)
        authorize(db, ctx, permission=Permission.CLEAN_DATA, feature=Feature.APPEND_MERGE)
        _, result = create_derived_dataset(
            db,
            ctx,
            name=new_dataset_name or default_dataset_name("merge", analysis.left, analysis.right),
            original_filename=default_dataset_name("merge", analysis.left, analysis.right),
            operation_type="merge",
            parameters=parameters,
            summary=summary,
            frame=merged,
        )
        return publish_result(result, output)
