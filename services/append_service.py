"""Append the rows of one dataset (the source) to another (the target).

Columns are matched by exact name. Columns that exist on one side only are
handled by an explicit policy and always reported:

* ``union``: keep every column; rows from the side that lacks a column get nulls.
* ``intersection``: keep only the common columns; dropped columns are listed.
* ``strict``: the column sets must be identical, otherwise ValidationError.

Names that differ only by letter case are reported as suggestions and never
matched automatically. Type conflicts are resolved as follows: integer with
float always becomes float; any other mismatch is rejected under the
``error`` policy and becomes text under ``cast_to_text``.

Processing is bounded: both versions are streamed batch by batch (at most
PARQUET_BATCH_ROWS rows in memory) into a Parquet file in the organization's
temporary directory, which is then published as a new version of the target
or as a new derived dataset. Possible duplicates after appending are counted
exactly with 64-bit row hashes over both datasets, aligned to the output
schema so equal rows hash equally on both sides.

This module also provides helpers shared with ``services.merge_service``
(authorized pair resolution, failure recording, derived dataset creation).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from sqlalchemy.orm import Session

from auth.authorization import AuthContext, authorize, authorize_dataset
from config.settings import get_settings
from core import storage
from core.constants import PANDAS_DTYPE_FOR, Feature, LogicalType, Permission
from core.errors import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ResourceLimitError,
    ValidationError,
)
from database.connection import session_scope
from services import resource_manager, versioning_service
from services.frame_utils import hash_rows, normalize_series, records
from services.resource_manager import MB, OperationGuard
from services.type_inference import arrow_type_for
from services.versioning_service import ParquetBatchWriter, PublishResult, VersionRef, iter_parquet

UNMATCHED_POLICIES = ("union", "intersection", "strict")
TYPE_CONFLICT_POLICIES = ("error", "cast_to_text")
OUTPUT_MODES = ("new_version", "new_dataset")
# numpy keeps one uint64 per row plus temporary copies made by sorting and np.isin.
HASH_BYTES_PER_ROW = 8 * 4
MAX_NAME_LENGTH = 200

_NUMERIC = frozenset({LogicalType.INTEGER.value, LogicalType.FLOAT.value})
_NOT_RECORDED = (AuthenticationError, PermissionDeniedError, NotFoundError)


# ---------------------------------------------------------------------------
# Helpers shared with merge_service
# ---------------------------------------------------------------------------


def resolve_pair(
    ctx: AuthContext, first_id: int, second_id: int, *, permission: Permission | None
) -> tuple[VersionRef, VersionRef]:
    """Authorize the plan feature first, then resolve the current version of both datasets.

    Raises FeatureNotAvailableError before any dataset is looked up, and
    NotFoundError when either dataset is not visible to ``ctx`` (including
    datasets of another organization).
    """
    with session_scope() as db:
        authorize(db, ctx, permission=permission, feature=Feature.APPEND_MERGE)
        first = versioning_service.resolve_version(
            db, ctx, first_id, permission=permission, feature=Feature.APPEND_MERGE
        )
        second = versioning_service.resolve_version(
            db, ctx, second_id, permission=permission, feature=Feature.APPEND_MERGE
        )
    return first, second


def version_file_bytes(ref: VersionRef) -> int:
    """Size of a version's Parquet file on disk."""
    return ref.path.stat().st_size


def dataset_summary(ref: VersionRef) -> dict[str, Any]:
    """Plain description of one input dataset for results and reports."""
    return {
        "dataset_id": ref.dataset_id,
        "name": ref.dataset_name,
        "version_id": ref.version_id,
        "version_number": ref.version_number,
        "rows": ref.rows,
        "columns": ref.columns,
    }


def schema_types(ref: VersionRef) -> dict[str, str]:
    """Column name to logical type, in column order."""
    return {column["name"]: column["logical_type"] for column in ref.schema}


def resource_report(check: Any) -> dict[str, Any]:
    """Run a resource check callable and report ``{"ok", "message"}`` instead of raising."""
    try:
        check()
    except ResourceLimitError as exc:
        return {"ok": False, "message": exc.user_message}
    return {"ok": True, "message": "Enough memory and disk space are available for this operation."}


def validate_choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise ValidationError(f"{label} must be one of: {', '.join(allowed)}.")
    return value


def validate_sample_rows(sample_rows: int) -> int:
    limit = get_settings().preview_max_rows
    if not isinstance(sample_rows, int) or isinstance(sample_rows, bool) or not 1 <= sample_rows <= limit:
        raise ValidationError(f"The number of sample rows must be between 1 and {limit}.")
    return sample_rows


def require_confirmation(confirm: bool, action: str) -> None:
    if confirm is not True:
        raise ValidationError(f"Review the preview and confirm to {action}.")


def default_dataset_name(verb: str, first: VersionRef, second: VersionRef) -> str:
    """A readable default name such as "merge of Sales and Customers" (at most 200 characters)."""
    return f"{verb} of {first.dataset_name} and {second.dataset_name}"[:MAX_NAME_LENGTH]


def record_operation_failure(
    ctx: AuthContext, dataset_id: int, operation_type: str, parameters: dict[str, Any], exc: BaseException
) -> None:
    """Record a failed or cancelled append/merge in the dataset's history (never raises).

    Authorization failures are not recorded, matching the versioning service.
    """
    versioning_service.record_failure(ctx, dataset_id, operation_type, parameters, exc)


def ensure_unchanged(db: Session, ctx: AuthContext, ref: VersionRef) -> Any:
    """Re-authorize the output dataset and fail if its current version changed since it was read."""
    dataset = authorize_dataset(
        db, ctx, ref.dataset_id, permission=Permission.CLEAN_DATA, feature=Feature.APPEND_MERGE
    )
    if dataset.current_version_id != ref.version_id:
        raise ConflictError(
            "The dataset changed while this operation was running. Review the latest version and try again."
        )
    return dataset


def create_derived_dataset(
    db: Session,
    ctx: AuthContext,
    *,
    name: str,
    original_filename: str,
    operation_type: str,
    parameters: dict[str, Any],
    summary: dict[str, Any],
    frame: pd.DataFrame | None = None,
    parquet_source: Any = None,
) -> tuple[int, PublishResult]:
    """Create a new derived dataset whose first version is an append/merge result.

    The kernel records the first version under ``operation_type`` with the real
    parameters and summary, counts it as a cleaning operation and audits it.
    """
    dataset, result = versioning_service.create_dataset_with_initial_version(
        db,
        ctx,
        name=name,
        source_type="derived",
        original_filename=original_filename,
        frame=frame,
        parquet_source=parquet_source,
        description=f"Created by {operation_type}.",
        import_options={"operation": operation_type, **parameters},
        import_report=summary,
        operation_type=operation_type,
        operation_parameters=parameters,
        operation_summary=summary,
    )
    return dataset.id, result


def publish_result(result: PublishResult, output: str) -> dict[str, Any]:
    return {
        "output": output,
        "dataset_id": result.dataset_id,
        "version_id": result.version_id,
        "version_number": result.version_number,
        "rows": result.rows,
        "columns": result.columns,
        "quality_score": result.quality_score,
    }


# ---------------------------------------------------------------------------
# Schema comparison and the output plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputColumn:
    name: str
    logical_type: str
    in_target: bool
    in_source: bool


@dataclass(frozen=True)
class AppendPlan:
    """How each input column maps to the output (fixed for every batch)."""

    columns: tuple[OutputColumn, ...]
    dropped_target_columns: tuple[str, ...]
    dropped_source_columns: tuple[str, ...]
    label_column: str | None

    def arrow_schema(self) -> pa.Schema:
        fields = [pa.field(c.name, arrow_type_for(c.logical_type)) for c in self.columns]
        if self.label_column:
            fields.append(pa.field(self.label_column, arrow_type_for(LogicalType.TEXT.value)))
        return pa.schema(fields)

    @property
    def width(self) -> int:
        return len(self.columns) + (1 if self.label_column else 0)


def resolve_type(target_type: str, source_type: str, policy: str) -> str | None:
    """Output type for a matched column, or None when the policy rejects the combination."""
    if target_type == source_type:
        return target_type
    if {target_type, source_type} <= _NUMERIC:
        return LogicalType.FLOAT.value
    if policy == "cast_to_text":
        return LogicalType.TEXT.value
    return None


def compare_schemas(target: dict[str, str], source: dict[str, str]) -> dict[str, Any]:
    """Exact-name matching, one-sided columns, case-only differences and type conflicts."""
    matching = [name for name in target if name in source]
    only_target = [name for name in target if name not in source]
    only_source = [name for name in source if name not in target]
    source_by_fold = {name.casefold(): name for name in only_source}
    case_only = [
        {
            "target_column": name,
            "source_column": source_by_fold[name.casefold()],
            "suggestion": f"Rename '{source_by_fold[name.casefold()]}' or '{name}' so the names match exactly; "
            "columns are never matched automatically when only the letter case differs.",
        }
        for name in only_target
        if name.casefold() in source_by_fold
    ]
    conflicts = []
    for name in matching:
        if target[name] == source[name]:
            continue
        conflicts.append(
            {
                "column": name,
                "target_type": target[name],
                "source_type": source[name],
                "resulting_type": {
                    policy: resolve_type(target[name], source[name], policy)
                    for policy in TYPE_CONFLICT_POLICIES
                },
                "requires_cast_to_text": resolve_type(target[name], source[name], "error") is None,
            }
        )
    return {
        "matching_columns": matching,
        "only_in_target": only_target,
        "only_in_source": only_source,
        "case_only_differences": case_only,
        "type_conflicts": conflicts,
        "schemas_identical": not only_target and not only_source and not conflicts,
    }


def _validate_label_column(label: str | None, names: list[str]) -> str | None:
    if label is None:
        return None
    clean = label.strip() if isinstance(label, str) else ""
    if not clean or len(clean) > MAX_NAME_LENGTH:
        raise ValidationError("The source label column needs a name of 1 to 200 characters.")
    if clean.casefold() in {name.casefold() for name in names}:
        raise ValidationError(f"The source label column '{clean}' already exists. Choose a different name.")
    return clean


def build_plan(
    target: dict[str, str],
    source: dict[str, str],
    *,
    unmatched_policy: str,
    type_conflict_policy: str,
    label_column: str | None = None,
) -> AppendPlan:
    """Decide the output columns and types, or raise ValidationError explaining what blocks the append."""
    validate_choice(unmatched_policy, UNMATCHED_POLICIES, "The unmatched-column policy")
    validate_choice(type_conflict_policy, TYPE_CONFLICT_POLICIES, "The type-conflict policy")
    comparison = compare_schemas(target, source)
    only_target, only_source = comparison["only_in_target"], comparison["only_in_source"]
    if unmatched_policy == "strict" and (only_target or only_source):
        raise ValidationError(
            "The column sets differ (only in target: "
            f"{', '.join(only_target) or 'none'}; only in source: {', '.join(only_source) or 'none'}). "
            "Choose the union or intersection policy, or align the columns first."
        )
    if unmatched_policy == "union" and comparison["case_only_differences"]:
        pairs = ", ".join(
            f"'{c['target_column']}'/'{c['source_column']}'" for c in comparison["case_only_differences"]
        )
        raise ValidationError(
            f"Column names differ only by letter case ({pairs}). Rename them to match exactly before appending."
        )
    blocked = [c for c in comparison["type_conflicts"] if c["resulting_type"][type_conflict_policy] is None]
    if blocked:
        details = ", ".join(f"'{c['column']}' ({c['target_type']} vs {c['source_type']})" for c in blocked)
        raise ValidationError(
            f"Incompatible column types: {details}. Convert the types first or choose the cast-to-text policy."
        )
    columns = [
        OutputColumn(name, resolve_type(target[name], source[name], type_conflict_policy) or "", True, True)
        for name in comparison["matching_columns"]
    ]
    if unmatched_policy == "union":
        columns += [OutputColumn(name, target[name], True, False) for name in only_target]
        columns += [OutputColumn(name, source[name], False, True) for name in only_source]
    if not columns:
        raise ValidationError("The datasets have no columns in common, so the intersection would be empty.")
    dropped = unmatched_policy == "intersection"
    plan = AppendPlan(
        columns=tuple(columns),
        dropped_target_columns=tuple(only_target) if dropped else (),
        dropped_source_columns=tuple(only_source) if dropped else (),
        label_column=_validate_label_column(label_column, [c.name for c in columns]),
    )
    limit = get_settings().max_columns
    if plan.width > limit:
        raise ValidationError(
            f"The appended result would have {plan.width:,} columns; datasets can have at most {limit:,} columns. "
            "Use the intersection policy or remove columns first."
        )
    return plan


# ---------------------------------------------------------------------------
# Batch alignment and streaming
# ---------------------------------------------------------------------------


def _datetime_text(series: pd.Series) -> pd.Series:
    """Element-wise text for datetimes: date only at midnight, else date and time."""
    full = series.dt.strftime("%Y-%m-%d %H:%M:%S").astype("string")
    date_only = series.dt.strftime("%Y-%m-%d").astype("string")
    midnight = (series == series.dt.normalize()).fillna(False)
    return full.where(~midnight, date_only)


def cast_series(series: pd.Series, logical_type: str) -> pd.Series:
    """Cast one input column to its output type (same type, integer to float, or text)."""
    if logical_type == LogicalType.TEXT.value and pd.api.types.is_datetime64_any_dtype(series.dtype):
        return _datetime_text(series)
    if logical_type == LogicalType.TEXT.value:
        return normalize_series(series).astype("string")
    if logical_type == LogicalType.FLOAT.value:
        return series.astype("float64")
    return normalize_series(series)


def null_series(length: int, logical_type: str, index: pd.Index | None = None) -> pd.Series:
    """An all-missing column of the storage dtype for ``logical_type``."""
    return pd.Series([None] * length, dtype=PANDAS_DTYPE_FOR[logical_type], index=index)


def align_batch(batch: pd.DataFrame, plan: AppendPlan, *, from_target: bool) -> pd.DataFrame:
    """Reshape one input batch to the output columns and dtypes (label column excluded)."""
    data = {}
    for column in plan.columns:
        present = column.in_target if from_target else column.in_source
        if present:
            data[column.name] = cast_series(batch[column.name], column.logical_type).reset_index(drop=True)
        else:
            data[column.name] = null_series(len(batch), column.logical_type)
    return pd.DataFrame(data, index=pd.RangeIndex(len(batch)))


def _input_columns(plan: AppendPlan, *, from_target: bool) -> list[str]:
    return [c.name for c in plan.columns if (c.in_target if from_target else c.in_source)]


def _iter_aligned(
    ref: VersionRef, plan: AppendPlan, *, from_target: bool, batch_rows: int
) -> Iterator[pd.DataFrame]:
    """Aligned batches of one input; the Parquet file is closed when iteration stops."""
    batches = iter_parquet(ref.path, batch_rows, _input_columns(plan, from_target=from_target))
    try:
        for batch in batches:
            yield align_batch(batch, plan, from_target=from_target)
    finally:
        batches.close()


@dataclass
class DuplicateCounter:
    """Exact duplicate counting with 64-bit row hashes (memory: 8 bytes per row)."""

    target_hashes: list[np.ndarray]
    source_hashes: list[np.ndarray]

    @classmethod
    def empty(cls) -> DuplicateCounter:
        return cls([], [])

    def result(self) -> dict[str, Any]:
        target = (
            np.unique(np.concatenate(self.target_hashes)) if self.target_hashes else np.empty(0, np.uint64)
        )
        source = np.concatenate(self.source_hashes) if self.source_hashes else np.empty(0, np.uint64)
        in_target = np.isin(source, target)
        new_rows = source[~in_target]
        matching_target = int(in_target.sum())
        repeated_new = int(len(new_rows) - len(np.unique(new_rows)))
        return {
            "source_rows_matching_target": matching_target,
            "duplicates_within_source": int(len(source) - len(np.unique(source))),
            "possible_duplicates": matching_target + repeated_new,
            "method": "Exact comparison of whole rows using 64-bit row hashes over both datasets.",
        }


def stream_append(
    target_ref: VersionRef,
    source_ref: VersionRef,
    plan: AppendPlan,
    *,
    guard: OperationGuard,
    writer: ParquetBatchWriter | None = None,
) -> dict[str, Any]:
    """Stream target then source rows, optionally writing them, and count duplicates exactly.

    Only one batch per input is held in memory at a time; hashes are computed
    without the source label column so they reflect the data rows only.
    """
    batch_rows = get_settings().parquet_batch_rows
    counter = DuplicateCounter.empty()
    total = max(target_ref.rows + source_ref.rows, 1)
    done = 0
    for ref, from_target in ((target_ref, True), (source_ref, False)):
        hashes = counter.target_hashes if from_target else counter.source_hashes
        for aligned in _iter_aligned(ref, plan, from_target=from_target, batch_rows=batch_rows):
            guard.step(done / total, f"Appending rows {done + 1:,}-{done + len(aligned):,}")
            hashes.append(hash_rows(aligned))
            if writer is not None:
                if plan.label_column:
                    aligned[plan.label_column] = pd.array([ref.dataset_name] * len(aligned), dtype="string")
                writer.write(aligned)
            done += len(aligned)
    return counter.result()


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppendResources:
    combined_input_bytes: int
    input_target_bytes: int
    memory_bytes: int
    disk_bytes: int

    @property
    def within_input_target(self) -> bool:
        return self.combined_input_bytes <= self.input_target_bytes

    def check(self) -> None:
        """Raise ResourceLimitError when memory or disk is insufficient."""
        resource_manager.check_memory(self.memory_bytes, operation="Appending these datasets")
        resource_manager.check_disk(self.disk_bytes, operation="Appending these datasets")


def estimate_resources(target_ref: VersionRef, source_ref: VersionRef, width: int) -> AppendResources:
    """Bounded-memory estimate: a few batches plus one hash per row; disk for the output file."""
    settings = get_settings()
    combined = version_file_bytes(target_ref) + version_file_bytes(source_ref)
    batch_memory = resource_manager.estimate_cells_memory(settings.parquet_batch_rows, max(width, 1)) * 3
    hash_memory = (target_ref.rows + source_ref.rows) * HASH_BYTES_PER_ROW
    return AppendResources(
        combined_input_bytes=combined,
        input_target_bytes=settings.append_merge_input_target_mb * MB,
        memory_bytes=batch_memory + hash_memory,
        disk_bytes=combined * 2,
    )


def _size_warning(resources: AppendResources) -> list[str]:
    if resources.within_input_target:
        return []
    return [
        f"The combined input ({resource_manager.human_bytes(resources.combined_input_bytes)}) is above the "
        f"{resource_manager.human_bytes(resources.input_target_bytes)} processing target, so it will be processed "
        "in batches and may take longer."
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_append(ctx: AuthContext, target_id: int, source_id: int) -> dict[str, Any]:
    """Read-only schema and size analysis of appending ``source_id`` to ``target_id``.

    Reads only metadata (no data rows). Reports exact-name matches, one-sided
    columns, case-only differences (suggestions, never matched), type
    conflicts with the resulting type under each policy, row counts,
    estimated output size and the resource check as ``{"ok", "message"}``.
    """
    target_ref, source_ref = resolve_pair(ctx, target_id, source_id, permission=None)
    target, source = schema_types(target_ref), schema_types(source_ref)
    comparison = compare_schemas(target, source)
    width = len(target) + len(comparison["only_in_source"])
    resources = estimate_resources(target_ref, source_ref, width)
    warnings = _size_warning(resources)
    if comparison["case_only_differences"]:
        warnings.append("Some column names differ only by letter case; rename them before appending.")
    if any(c["requires_cast_to_text"] for c in comparison["type_conflicts"]):
        warnings.append("Some matched columns have incompatible types; they can only be appended as text.")
    return {
        "target": dataset_summary(target_ref),
        "source": dataset_summary(source_ref),
        **comparison,
        "target_types": target,
        "source_types": source,
        "estimated_output_rows": target_ref.rows + source_ref.rows,
        "estimated_output_bytes": resources.combined_input_bytes,
        "combined_input_bytes": resources.combined_input_bytes,
        "input_target_bytes": resources.input_target_bytes,
        "within_input_target": resources.within_input_target,
        "resource_check": resource_report(resources.check),
        "warnings": warnings,
    }


def _plan_warnings(plan: AppendPlan, target: dict[str, str], source: dict[str, str]) -> list[str]:
    warnings = []
    if plan.dropped_target_columns or plan.dropped_source_columns:
        warnings.append(
            "Columns not present in both datasets will be dropped: "
            f"from the target {', '.join(plan.dropped_target_columns) or 'none'}; "
            f"from the source {', '.join(plan.dropped_source_columns) or 'none'}."
        )
    for column in plan.columns:
        if not column.in_source:
            warnings.append(
                f"Column '{column.name}' exists only in the target; appended rows get empty values."
            )
        elif not column.in_target:
            warnings.append(
                f"Column '{column.name}' exists only in the source; existing rows get empty values."
            )
        elif target[column.name] != source[column.name]:
            warnings.append(
                f"Column '{column.name}' combines {target[column.name]} and {source[column.name]} values "
                f"and becomes {column.logical_type}."
            )
    return warnings


def _sample(ref: VersionRef, plan: AppendPlan, *, from_target: bool, rows: int) -> list[dict[str, Any]]:
    batches = _iter_aligned(ref, plan, from_target=from_target, batch_rows=rows)
    try:
        first = next(batches)
    finally:
        batches.close()
    return records(first, rows)


def preview_append(
    ctx: AuthContext,
    target_id: int,
    source_id: int,
    *,
    unmatched_policy: str = "union",
    type_conflict_policy: str = "error",
    sample_rows: int = 20,
    source_label_column: str | None = None,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Read-only preview: output columns and types, samples from both sides, exact duplicate counts.

    Raises ValidationError when the policies reject the schemas (strict with
    differing columns, an incompatible type under ``error``, case-only names
    under ``union``). When memory or disk is insufficient, the duplicate
    count is skipped and reported as unavailable instead of being estimated.
    """
    target_ref, source_ref = resolve_pair(ctx, target_id, source_id, permission=None)
    validate_sample_rows(sample_rows)
    target, source = schema_types(target_ref), schema_types(source_ref)
    plan = build_plan(
        target,
        source,
        unmatched_policy=unmatched_policy,
        type_conflict_policy=type_conflict_policy,
        label_column=source_label_column,
    )
    resources = estimate_resources(target_ref, source_ref, plan.width)
    check = resource_report(resources.check)
    warnings = _size_warning(resources) + _plan_warnings(plan, target, source)
    duplicates: dict[str, Any] | None = None
    if check["ok"]:
        duplicates = stream_append(target_ref, source_ref, plan, guard=guard or OperationGuard())
        if duplicates["possible_duplicates"]:
            warnings.append(
                f"{duplicates['possible_duplicates']:,} appended row(s) would duplicate an existing row "
                "(exact whole-row comparison)."
            )
    else:
        warnings.append(
            "Duplicate detection was skipped because resources are insufficient: " + check["message"]
        )
    columns = [
        {
            "name": c.name,
            "logical_type": c.logical_type,
            "from_target": c.in_target,
            "from_source": c.in_source,
        }
        for c in plan.columns
    ]
    if plan.label_column:
        columns.append(
            {
                "name": plan.label_column,
                "logical_type": LogicalType.TEXT.value,
                "from_target": True,
                "from_source": True,
            }
        )
    return {
        "target": dataset_summary(target_ref),
        "source": dataset_summary(source_ref),
        "unmatched_policy": unmatched_policy,
        "type_conflict_policy": type_conflict_policy,
        "columns": columns,
        "dropped_columns": {
            "target": list(plan.dropped_target_columns),
            "source": list(plan.dropped_source_columns),
        },
        "target_rows": target_ref.rows,
        "source_rows": source_ref.rows,
        "output_rows": target_ref.rows + source_ref.rows,
        "sample_target_rows": _sample(target_ref, plan, from_target=True, rows=sample_rows),
        "sample_source_rows": _sample(source_ref, plan, from_target=False, rows=sample_rows),
        "duplicates": duplicates,
        "possible_duplicates": duplicates["possible_duplicates"] if duplicates else None,
        "resource_check": check,
        "warnings": warnings,
    }


def apply_append(
    ctx: AuthContext,
    target_id: int,
    source_id: int,
    *,
    unmatched_policy: str = "union",
    type_conflict_policy: str = "error",
    output: str = "new_version",
    new_dataset_name: str | None = None,
    source_label_column: str | None = None,
    confirm: bool = False,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Append the source rows to the target and publish the result.

    ``output="new_version"`` publishes a new version of the target (failing
    with ConflictError if the target changed meanwhile); ``"new_dataset"``
    creates a derived dataset. Both inputs are streamed batch by batch through
    a Parquet writer in the organization's temporary directory, which is
    removed on success, failure and cancellation. Failures are recorded in the
    target's history and publish nothing.
    """
    target_ref, source_ref = resolve_pair(ctx, target_id, source_id, permission=Permission.CLEAN_DATA)
    validate_choice(output, OUTPUT_MODES, "The output")
    require_confirmation(confirm, "append these datasets")
    started = time.monotonic()
    guard = guard or OperationGuard()
    target, source = schema_types(target_ref), schema_types(source_ref)
    plan = build_plan(
        target,
        source,
        unmatched_policy=unmatched_policy,
        type_conflict_policy=type_conflict_policy,
        label_column=source_label_column,
    )
    parameters: dict[str, Any] = {
        "source_dataset_id": source_ref.dataset_id,
        "source_version": source_ref.version_number,
        "unmatched_policy": unmatched_policy,
        "type_conflict_policy": type_conflict_policy,
        "source_label_column": plan.label_column,
        "output": output,
    }
    try:
        estimate_resources(target_ref, source_ref, plan.width).check()
        with storage.temporary_directory(target_ref.organization_id, prefix="append-") as tmp:
            result_path = tmp / "append.parquet"
            with ParquetBatchWriter(result_path, plan.arrow_schema()) as writer:
                duplicates = stream_append(target_ref, source_ref, plan, guard=guard, writer=writer)
            guard.step(1.0, "Publishing the result")
            summary = {
                "target": dataset_summary(target_ref),
                "source": dataset_summary(source_ref),
                "rows_added": source_ref.rows,
                "dropped_columns": {
                    "target": list(plan.dropped_target_columns),
                    "source": list(plan.dropped_source_columns),
                },
                "null_filled_columns": [c.name for c in plan.columns if not (c.in_target and c.in_source)],
                **duplicates,
            }
            published = _publish_append(
                ctx,
                target_ref,
                source_ref,
                output,
                new_dataset_name,
                parameters,
                summary,
                result_path,
                started,
                guard=guard,
            )
    except BaseException as exc:
        record_operation_failure(ctx, target_ref.dataset_id, "append", parameters, exc)
        raise
    return {
        **published,
        "rows_added": source_ref.rows,
        "possible_duplicates": duplicates["possible_duplicates"],
        "duplicates": duplicates,
        "dropped_columns": summary["dropped_columns"],
    }


def _publish_append(
    ctx: AuthContext,
    target_ref: VersionRef,
    source_ref: VersionRef,
    output: str,
    new_dataset_name: str | None,
    parameters: dict[str, Any],
    summary: dict[str, Any],
    result_path: Any,
    started: float,
    *,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    with session_scope() as db:
        if output == "new_version":
            dataset = ensure_unchanged(db, ctx, target_ref)
            result = versioning_service.publish_version(
                db,
                ctx,
                dataset,
                operation_type="append",
                parameters=parameters,
                parquet_source=result_path,
                stats={"rows_added": source_ref.rows, "rows_removed": 0},
                summary=summary,
                note=f"Appended {source_ref.rows:,} row(s) from '{source_ref.dataset_name}'",
                started=started,
                guard=guard,
            )
            return publish_result(result, output)
        authorize(db, ctx, permission=Permission.CLEAN_DATA, feature=Feature.APPEND_MERGE)
        _, result = create_derived_dataset(
            db,
            ctx,
            name=new_dataset_name or default_dataset_name("append", target_ref, source_ref),
            original_filename=default_dataset_name("append", target_ref, source_ref),
            operation_type="append",
            parameters=parameters,
            summary=summary,
            parquet_source=result_path,
        )
        return publish_result(result, output)
