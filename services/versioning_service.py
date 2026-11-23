"""Immutable dataset versions stored as Parquet files.

Publishing protocol (atomic and recoverable):

1. The new version is written, scored and hashed in a staging folder (the
   organization's temporary area) before any database write, so the SQLite
   write lock is not held during this heavy work.
2. A short write transaction (``BEGIN IMMEDIATE``) re-checks that the dataset
   did not change, takes the next version number, applies retention and the
   storage quota, moves the staged file to its unique final name with
   ``os.replace`` and writes the version row, the operation record and the
   dataset's current-version pointer in the caller's transaction.
3. If that transaction rolls back, the new file is deleted. Only after it
   commits are the files of versions removed by the retention policy deleted;
   a file that cannot be deleted yet (still open on Windows) stays flagged,
   counts toward the storage quota and is retried by later publishes and by
   the scheduled worker.

A crash between the move and the commit can only leave an unreferenced file,
never a partially published version. The original upload is stored separately, made
read-only, and is never modified or deleted by any operation. Every version is
a complete Parquet file, so restoring one is a real data restore, not just a
pointer change: ``restore_version`` copies it forward as a new version.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import event, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth.authorization import AuthContext, authorize_dataset
from config.settings import get_settings
from core import storage, timeutil
from core.constants import (
    PANDAS_DTYPE_FOR,
    SUPPORTED_UPLOAD_EXTENSIONS,
    DatasetStatus,
    Feature,
    LogicalType,
    Permission,
    Role,
    UsageMetric,
)
from core.errors import (
    AppError,
    AuthenticationError,
    ConflictError,
    NotFoundError,
    OperationCancelledError,
    OperationTimeoutError,
    PermissionDeniedError,
    ValidationError,
)
from database.connection import begin_write, session_scope
from database.models import CleaningOperation, Dataset, DatasetAccess, DatasetVersion, User
from modules import subscriptions, usage_tracking
from services import resource_manager
from services.audit_service import record_audit
from services.frame_utils import arrow_schema_for_frame, frame_schema, normalize_frame
from services.quality_score import QualityResult, compute_quality, compute_quality_batches
from services.resource_manager import OperationGuard

logger = logging.getLogger(__name__)

PARQUET_COMPRESSION = "zstd"
_HOOKS_KEY = "vdas_file_hooks"


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------


def _types_mapper(arrow_type: pa.DataType) -> Any:
    if pa.types.is_integer(arrow_type):
        return pd.Int64Dtype()
    if pa.types.is_boolean(arrow_type):
        return pd.BooleanDtype()
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return pd.StringDtype()
    return None


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns, memory_map=False)
    return table.to_pandas(types_mapper=_types_mapper)


def iter_parquet(path: Path, batch_rows: int, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    """Yield the file in batches; the file handle is closed when iteration ends."""
    with open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        if parquet.metadata.num_rows == 0:
            yield (
                parquet.schema_arrow.empty_table()
                .select(columns or parquet.schema_arrow.names)
                .to_pandas(types_mapper=_types_mapper)
            )
            return
        for batch in parquet.iter_batches(batch_size=max(1, batch_rows), columns=columns):
            yield batch.to_pandas(types_mapper=_types_mapper)


def logical_type_for_arrow(arrow_type: pa.DataType) -> str:
    if pa.types.is_integer(arrow_type):
        return LogicalType.INTEGER.value
    if pa.types.is_floating(arrow_type):
        return LogicalType.FLOAT.value
    if pa.types.is_boolean(arrow_type):
        return LogicalType.BOOLEAN.value
    if pa.types.is_timestamp(arrow_type) or pa.types.is_date(arrow_type):
        return LogicalType.DATETIME.value
    return LogicalType.TEXT.value


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    result = []
    for field in schema:
        logical = logical_type_for_arrow(field.type)
        result.append({"name": field.name, "logical_type": logical, "dtype": PANDAS_DTYPE_FOR[logical]})
    return result


class ParquetBatchWriter:
    """Write DataFrame batches with a fixed schema to an atomically published file."""

    def __init__(self, dest: Path, schema: pa.Schema) -> None:
        self.dest = dest
        self.schema = schema
        self.rows = 0
        self._atomic = storage.atomic_output(dest)
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> ParquetBatchWriter:
        tmp = self._atomic.__enter__()
        self._writer = pq.ParquetWriter(tmp, self.schema, compression=PARQUET_COMPRESSION)
        return self

    def write(self, df: pd.DataFrame) -> None:
        assert self._writer is not None
        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        self._writer.write_table(table)
        self.rows += len(df)

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        close_error: BaseException | None = None
        try:
            if self._writer is not None:
                self._writer.close()
        except Exception as error:  # noqa: BLE001 - re-raised below
            close_error = error
        if exc_type is None and close_error is not None:
            self._atomic.__exit__(type(close_error), close_error, close_error.__traceback__)
            raise close_error
        return bool(self._atomic.__exit__(exc_type, exc, tb))


def write_frame(df: pd.DataFrame, dest: Path) -> None:
    with ParquetBatchWriter(dest, arrow_schema_for_frame(df)) as writer:
        writer.write(df)


# ---------------------------------------------------------------------------
# Transaction-bound file cleanup
# ---------------------------------------------------------------------------


class _FileTransactionHooks:
    """Delete new files if the transaction rolls back; purge old files after commit."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.new_files: list[Path] = []
        self.purge_files: list[Path] = []
        self.done = False
        event.listen(db, "after_commit", self._on_commit)
        event.listen(db, "after_rollback", self._on_rollback)

    def _finish(self) -> None:
        self.done = True
        if self.db.info.get(_HOOKS_KEY) is self:
            self.db.info.pop(_HOOKS_KEY, None)

    def _on_commit(self, _session: Session) -> None:
        if self.done:
            return
        for path in self.purge_files:
            if not storage.remove_file_quietly(path):
                logger.warning("Could not remove purged version file %s", path.name)
        self._finish()

    def _on_rollback(self, _session: Session) -> None:
        if self.done:
            return
        for path in self.new_files:
            storage.remove_file_quietly(path)
        self._finish()


def file_hooks(db: Session) -> _FileTransactionHooks:
    hooks = db.info.get(_HOOKS_KEY)
    if hooks is None or hooks.done:
        hooks = _FileTransactionHooks(db)
        db.info[_HOOKS_KEY] = hooks
    return hooks


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


@dataclass
class PublishResult:
    dataset_id: int
    version_id: int
    version_number: int
    operation_id: int
    rows: int
    columns: int
    quality_score: float | None
    quality_before: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _next_version_number(db: Session, dataset_id: int) -> int:
    current = db.execute(
        select(func.max(DatasetVersion.version_number)).where(DatasetVersion.dataset_id == dataset_id)
    ).scalar_one()
    return int(current or 0) + 1


def _place_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, dest)
    except OSError:
        storage.atomic_copy(source, dest)
        storage.remove_file_quietly(source)


def _retention_candidates(
    db: Session, dataset: Dataset, max_versions: int, adding: int
) -> list[DatasetVersion]:
    live = (
        db.execute(
            select(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset.id, DatasetVersion.is_purged.is_(False))
            .order_by(DatasetVersion.version_number)
        )
        .scalars()
        .all()
    )
    excess = len(live) + adding - max_versions
    if excess <= 0:
        return []
    # Only pinned versions are protected: the new version is not in ``live`` yet, and the
    # outgoing current version is superseded by it, so at the minimum limit of 2 (the pinned
    # original plus the new version) it is purged too. Oldest versions go first.
    removable = [v for v in live if not v.is_pinned]
    return removable[:excess]


CHANGED_MESSAGE = (
    "The dataset changed while this operation was running. Review the latest version and try again."
)


@dataclass
class _PreparedVersion:
    """A complete, scored version file waiting in a staging folder (no database rows yet)."""

    path: Path
    rows: int
    columns: int
    schema: list[dict[str, str]]
    quality: QualityResult
    size: int
    sha256: str


def _checked_batches(batches: Iterator[pd.DataFrame], guard: OperationGuard | None) -> Iterator[pd.DataFrame]:
    for batch in batches:
        if guard is not None:
            guard.check()  # once per batch: cancellation and the operation timeout apply to scoring
        yield batch


def _prepare_version(
    staging: Path,
    frame: pd.DataFrame | None,
    parquet_source: Path | None,
    guard: OperationGuard | None = None,
) -> _PreparedVersion:
    """The heavy half of a publish: write, score and hash the file. Runs before any database write."""
    staged = staging / "version.parquet"
    if frame is not None:
        normalized = normalize_frame(frame)
        if guard is not None:
            guard.check()
        quality = compute_quality(normalized)
        if guard is not None:
            guard.check()
        write_frame(normalized, staged)
        rows, columns, schema = len(normalized), normalized.shape[1], frame_schema(normalized)
    else:
        assert parquet_source is not None
        _place_file(parquet_source, staged)
        metadata = pq.read_metadata(staged)
        rows, columns = metadata.num_rows, metadata.num_columns
        schema = parquet_schema(staged)
        # Close the reader deterministically: an open handle would block the cleanup on Windows.
        with contextlib.closing(iter_parquet(staged, get_settings().parquet_batch_rows)) as batches:
            quality = compute_quality_batches(_checked_batches(batches, guard))
    return _PreparedVersion(
        path=staged,
        rows=rows,
        columns=columns,
        schema=schema,
        quality=quality,
        size=staged.stat().st_size,
        sha256=storage.sha256_file(staged),
    )


def _pending_purged_files(db: Session, org_id: int) -> list[Path]:
    """Files of the organization's purged versions that may still exist (to retry after commit).

    A purged file whose deletion failed (on Windows, a reader still had it
    open) keeps ``file_pending_delete``; rows whose file is gone are cleared
    here. Only recently purged versions carry the flag, so this stays small.
    """
    rows = (
        db.execute(
            select(DatasetVersion).where(
                DatasetVersion.organization_id == org_id,
                DatasetVersion.is_purged.is_(True),
                DatasetVersion.file_pending_delete.is_(True),
            )
        )
        .scalars()
        .all()
    )
    retry: list[Path] = []
    for row in rows:
        try:
            path = storage.resolve_relative(row.file_path)
        except PermissionDeniedError:
            logger.warning("Purged version %s has an invalid stored path; not deleting it", row.id)
            row.file_pending_delete = False
            continue
        if path.exists():
            retry.append(path)
        else:
            row.file_pending_delete = False
    return retry


def reclaim_purged_files() -> dict[str, int]:
    """Delete purged version files left behind earlier (for the scheduled worker; never raises on locks).

    Returns how many pending rows were confirmed deleted and how many files are
    still locked (they are retried on the next run or the next publish).
    """
    with session_scope() as db:
        pending = db.execute(
            select(DatasetVersion.id, DatasetVersion.file_path).where(
                DatasetVersion.is_purged.is_(True), DatasetVersion.file_pending_delete.is_(True)
            )
        ).all()
    reclaimed: list[int] = []
    still_locked = 0
    for version_id, relative in pending:
        try:
            path = storage.resolve_relative(relative)
        except PermissionDeniedError:
            logger.warning("Purged version %s has an invalid stored path; not deleting it", version_id)
            reclaimed.append(version_id)
            continue
        if storage.remove_file_quietly(path):
            reclaimed.append(version_id)
        else:
            still_locked += 1
    if reclaimed:
        with session_scope() as db:
            db.execute(
                update(DatasetVersion)
                .where(DatasetVersion.id.in_(reclaimed))
                .values(file_pending_delete=False)
            )
    return {"reclaimed": len(reclaimed), "still_locked": still_locked}


def _is_version_number_clash(error: IntegrityError) -> bool:
    text = str(error.orig)
    return "uq_dataset_versions_number" in text or "dataset_versions.version_number" in text


def _record_version(
    db: Session,
    ctx: AuthContext,
    dataset: Dataset,
    prepared: _PreparedVersion,
    *,
    expected_current: int | None,
    operation_type: str,
    parameters: dict[str, Any] | None,
    stats: dict[str, int] | None,
    summary: dict[str, Any] | None,
    note: str,
    started: float | None,
    pinned: bool,
) -> PublishResult:
    """The short half of a publish: under the write lock, check, place the file and write the rows.

    The lock is taken first, so the "did the dataset change?" check, the
    version number, retention and the storage quota all see the latest
    committed state, and no other writer can slip in before the commit.
    """
    org_id = dataset.organization_id
    begin_write(db)
    current = db.execute(select(Dataset.current_version_id).where(Dataset.id == dataset.id)).scalar_one()
    if current != expected_current:
        raise ConflictError(CHANGED_MESSAGE)
    previous = db.get(DatasetVersion, current) if current else None
    number = _next_version_number(db, dataset.id)
    limits = subscriptions.get_limits(db, org_id)
    purge = _retention_candidates(db, dataset, limits.max_versions_per_dataset, adding=1)
    subscriptions.enforce_storage_quota(db, org_id, prepared.size, freed_bytes=sum(v.size_bytes for v in purge))
    retry_purge = _pending_purged_files(db, org_id)
    rows, columns, quality = prepared.rows, prepared.columns, prepared.quality
    dest = storage.new_version_path(org_id, dataset.id, number)
    hooks = file_hooks(db)
    hooks.new_files.append(dest)
    try:
        _place_file(prepared.path, dest)
        version = DatasetVersion(
            organization_id=org_id,
            dataset_id=dataset.id,
            version_number=number,
            parent_version_id=previous.id if previous else None,
            file_path=storage.to_relative(dest),
            file_sha256=prepared.sha256,
            size_bytes=prepared.size,
            row_count=rows,
            column_count=columns,
            schema_json=json.dumps(prepared.schema),
            quality_score=quality.score,
            quality_json=json.dumps(quality.to_dict()),
            operation_type=operation_type,
            note=note[:2000],
            is_pinned=pinned,
            created_by_id=ctx.user_id,
        )
        db.add(version)
        try:
            db.flush()
        except IntegrityError as exc:  # backstop only: the write lock should make this impossible
            if _is_version_number_clash(exc):
                raise ConflictError(CHANGED_MESSAGE) from exc
            raise

        stats = stats or {}
        rows_before = previous.row_count if previous else 0
        operation = CleaningOperation(
            organization_id=org_id,
            dataset_id=dataset.id,
            input_version_id=previous.id if previous else None,
            output_version_id=version.id,
            user_id=ctx.user_id,
            operation_type=operation_type,
            parameters_json=json.dumps(parameters or {}, default=str),
            status="applied",
            rows_before=previous.row_count if previous else None,
            rows_after=rows,
            columns_before=previous.column_count if previous else None,
            columns_after=columns,
            rows_added=int(stats.get("rows_added", max(rows - rows_before, 0))),
            rows_removed=int(stats.get("rows_removed", max(rows_before - rows, 0))),
            rows_affected=int(stats.get("rows_affected", 0)),
            cells_changed=int(stats.get("cells_changed", 0)),
            quality_before=previous.quality_score if previous else None,
            quality_after=quality.score,
            summary_json=json.dumps(summary or {}, default=str),
            duration_ms=int((time.monotonic() - started) * 1000) if started is not None else None,
        )
        db.add(operation)
        dataset.current_version_id = version.id
        dataset.updated_at = timeutil.utcnow()
        now = timeutil.utcnow()
        for old in purge:
            old.is_purged = True
            old.purged_at = now
            old.file_pending_delete = True  # cleared once the file is confirmed gone
            hooks.purge_files.append(storage.resolve_relative(old.file_path))
        hooks.purge_files.extend(retry_purge)
        db.flush()
        record_audit(
            db,
            ctx,
            f"dataset.{operation_type}",
            target_type="dataset",
            target_id=dataset.id,
            details={
                "version": number,
                "rows": rows,
                "columns": columns,
                "rows_removed": operation.rows_removed,
                "cells_changed": operation.cells_changed,
                "purged_versions": [v.version_number for v in purge],
            },
        )
        if operation_type != "import":
            usage_tracking.record_usage(
                db, org_id, ctx.user_id, UsageMetric.CLEANING_OPERATION, details={"operation": operation_type}
            )
    except BaseException:
        storage.remove_file_quietly(dest)
        raise
    return PublishResult(
        dataset_id=dataset.id,
        version_id=version.id,
        version_number=number,
        operation_id=operation.id,
        rows=rows,
        columns=columns,
        quality_score=quality.score,
        quality_before=previous.quality_score if previous else None,
    )


def publish_version(
    db: Session,
    ctx: AuthContext,
    dataset: Dataset,
    *,
    operation_type: str,
    parameters: dict[str, Any] | None = None,
    frame: pd.DataFrame | None = None,
    parquet_source: Path | None = None,
    stats: dict[str, int] | None = None,
    summary: dict[str, Any] | None = None,
    note: str = "",
    started: float | None = None,
    pinned: bool = False,
    guard: OperationGuard | None = None,
) -> PublishResult:
    """Write a new immutable version and make it current (inside ``db``'s transaction).

    The caller must already have authorized ``ctx`` for ``dataset``. Exactly one
    of ``frame`` (in-memory result) or ``parquet_source`` (a complete Parquet
    file produced batch by batch) must be given.

    The file is written, scored and hashed first, without the database write
    lock. Then, under the lock, the dataset's current version is compared with
    the one ``dataset`` had when this call started: if another operation
    published in between, ConflictError is raised and nothing is published.
    A ``guard`` is checked while the new file is scored (once per batch).
    """
    if (frame is None) == (parquet_source is None):
        raise ValueError("Provide exactly one of frame or parquet_source.")
    if ctx.organization_id != dataset.organization_id:
        raise PermissionDeniedError()
    expected_current = dataset.current_version_id
    with storage.temporary_directory(dataset.organization_id, prefix="publish-") as staging:
        prepared = _prepare_version(staging, frame, parquet_source, guard)
        return _record_version(
            db,
            ctx,
            dataset,
            prepared,
            expected_current=expected_current,
            operation_type=operation_type,
            parameters=parameters,
            stats=stats,
            summary=summary,
            note=note,
            started=started,
            pinned=pinned,
        )


def validate_dataset_name(name: str) -> str:
    clean = " ".join((name or "").split())
    if not clean:
        raise ValidationError("A dataset name is required.")
    if len(clean) > 200:
        raise ValidationError("Dataset names can be at most 200 characters.")
    return clean


def create_dataset_with_initial_version(
    db: Session,
    ctx: AuthContext,
    *,
    name: str,
    source_type: str,
    original_filename: str,
    frame: pd.DataFrame | None = None,
    parquet_source: Path | None = None,
    original_source: Path | None = None,
    description: str = "",
    sheet_name: str | None = None,
    import_options: dict[str, Any] | None = None,
    import_report: dict[str, Any] | None = None,
    operation_type: str = "import",
    operation_parameters: dict[str, Any] | None = None,
    operation_summary: dict[str, Any] | None = None,
) -> tuple[Dataset, PublishResult]:
    """Register a dataset, keep its original file (read-only) and publish version 1.

    The caller must have authorized the upload (permission + plan feature).
    Datasets produced by an operation (append, merge) pass that operation as
    ``operation_type`` so their first version is recorded as what it is.

    The heavy work (copying and hashing the original, writing and scoring the
    version file) runs first in a staging folder, without any database write,
    so the write lock is not held while it runs. Then a short write
    transaction re-checks the name and the plan limits, inserts the rows and
    moves the staged files into place. If anything fails, every placed file is
    deleted (immediately, and again by the rollback hook).
    """
    if ctx.organization_id is None or ctx.role == Role.SUPER_ADMIN:
        raise PermissionDeniedError()
    org_id = ctx.organization_id
    clean_name = validate_dataset_name(name)
    if source_type not in {"csv", "xlsx", "xls", "derived"}:
        raise ValidationError("Unsupported dataset source.")
    if (frame is None) == (parquet_source is None):
        raise ValueError("Provide exactly one of frame or parquet_source.")
    extension = f".{source_type}"
    if original_source is not None and extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValidationError("Unsupported file type.")
    # Cheap early checks, so a doomed upload does not do the heavy work. They are
    # repeated under the write lock below, because the state can change meanwhile.
    _check_new_dataset_allowed(db, org_id, clean_name)
    original_size = original_source.stat().st_size if original_source is not None else 0
    subscriptions.enforce_storage_quota(db, org_id, original_size)

    with storage.temporary_directory(org_id, prefix="create-") as staging:
        staged_original: Path | None = None
        original_sha256: str | None = None
        if original_source is not None:
            staged_original = staging / f"original{extension}"
            storage.atomic_copy(original_source, staged_original)
            original_sha256 = storage.sha256_file(staged_original)
        prepared = _prepare_version(staging, frame, parquet_source)

        begin_write(db)
        _check_new_dataset_allowed(db, org_id, clean_name)
        return _record_new_dataset(
            db,
            ctx,
            prepared,
            staged_original=staged_original,
            original_sha256=original_sha256,
            original_size=original_size,
            clean_name=clean_name,
            source_type=source_type,
            original_filename=original_filename,
            description=description,
            sheet_name=sheet_name,
            import_options=import_options,
            import_report=import_report,
            operation_type=operation_type,
            operation_parameters=operation_parameters,
            operation_summary=operation_summary,
        )


def _check_new_dataset_allowed(db: Session, org_id: int, clean_name: str) -> None:
    duplicate = db.execute(
        select(Dataset.id).where(
            Dataset.organization_id == org_id, func.lower(Dataset.name) == clean_name.lower()
        )
    ).first()
    if duplicate:
        raise ConflictError("A dataset with this name already exists in your organization.")
    subscriptions.enforce_dataset_quota(db, org_id)


def _record_new_dataset(
    db: Session,
    ctx: AuthContext,
    prepared: _PreparedVersion,
    *,
    staged_original: Path | None,
    original_sha256: str | None,
    original_size: int,
    clean_name: str,
    source_type: str,
    original_filename: str,
    description: str,
    sheet_name: str | None,
    import_options: dict[str, Any] | None,
    import_report: dict[str, Any] | None,
    operation_type: str,
    operation_parameters: dict[str, Any] | None,
    operation_summary: dict[str, Any] | None,
) -> tuple[Dataset, PublishResult]:
    """The short write phase of a dataset creation (the caller holds the write lock)."""
    org_id = ctx.organization_id
    assert org_id is not None
    dataset = Dataset(
        organization_id=org_id,
        name=clean_name,
        description=(description or "")[:2000],
        source_type=source_type,
        original_filename=storage.sanitize_display_filename(original_filename),
        sheet_name=sheet_name,
        import_options_json=json.dumps(import_options or {}, default=str),
        import_report_json=json.dumps(import_report or {}, default=str),
        status=DatasetStatus.ACTIVE.value,
        created_by_id=ctx.user_id,
        original_size_bytes=original_size,
    )
    db.add(dataset)
    try:
        db.flush()
    except IntegrityError as exc:
        raise ConflictError("A dataset with this name already exists in your organization.") from exc

    hooks = file_hooks(db)
    original_dest: Path | None = None
    try:
        if staged_original is not None:
            original_dest = storage.new_original_path(org_id, dataset.id, f".{source_type}")
            hooks.new_files.append(original_dest)
            _place_file(staged_original, original_dest)
            storage.make_read_only(original_dest)
            dataset.original_file_path = storage.to_relative(original_dest)
            dataset.original_sha256 = original_sha256

        # Re-checks the storage quota (now including the original) under the same lock.
        result = _record_version(
            db,
            ctx,
            dataset,
            prepared,
            expected_current=None,
            operation_type=operation_type,
            parameters=operation_parameters if operation_parameters is not None else (import_options or {}),
            stats=None,
            summary=(
                operation_summary
                if operation_summary is not None
                else {"source_type": source_type, "sheet_name": sheet_name}
            ),
            note="",
            started=None,
            pinned=True,
        )
        if ctx.role == Role.ANALYST and ctx.user_id is not None:
            db.add(
                DatasetAccess(
                    organization_id=org_id,
                    dataset_id=dataset.id,
                    user_id=ctx.user_id,
                    granted_by_id=ctx.user_id,
                )
            )
        if staged_original is not None:
            usage_tracking.record_usage(
                db,
                org_id,
                ctx.user_id,
                UsageMetric.UPLOAD_BYTES,
                quantity=original_size,
                details={"dataset_id": dataset.id},
            )
        record_audit(
            db,
            ctx,
            "dataset.create",
            target_type="dataset",
            target_id=dataset.id,
            details={
                "name": clean_name,
                "source_type": source_type,
                "rows": result.rows,
                "columns": result.columns,
            },
        )
        db.flush()
    except BaseException:
        if original_dest is not None:
            storage.remove_file_quietly(original_dest)
        raise
    return dataset, result


# ---------------------------------------------------------------------------
# Authorized reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionRef:
    """A resolved, authorized version. Holds no ORM objects."""

    dataset_id: int
    dataset_name: str
    organization_id: int
    version_id: int
    version_number: int
    path: Path
    rows: int
    columns: int
    schema: list[dict[str, str]]
    quality_score: float | None
    is_current: bool


def _get_version(db: Session, dataset: Dataset, version_id: int | None = None) -> DatasetVersion:
    target = version_id or dataset.current_version_id
    if target is None:
        raise NotFoundError("This dataset has no data yet.")
    version = db.get(DatasetVersion, target)
    if version is None or version.dataset_id != dataset.id:
        raise NotFoundError("Version not found.")
    if version.is_purged:
        raise NotFoundError("This version was removed by the retention policy and can no longer be loaded.")
    return version


def resolve_version(
    db: Session,
    ctx: AuthContext,
    dataset_id: int,
    version_id: int | None = None,
    *,
    permission: Permission | None = None,
    feature: Feature | None = None,
) -> VersionRef:
    dataset = authorize_dataset(db, ctx, dataset_id, permission=permission, feature=feature)
    version = _get_version(db, dataset, version_id)
    return VersionRef(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        organization_id=dataset.organization_id,
        version_id=version.id,
        version_number=version.version_number,
        path=storage.resolve_relative(version.file_path),
        rows=version.row_count,
        columns=version.column_count,
        schema=json.loads(version.schema_json),
        quality_score=version.quality_score,
        is_current=version.id == dataset.current_version_id,
    )


def get_version_ref(
    ctx: AuthContext,
    dataset_id: int,
    version_id: int | None = None,
    *,
    permission: Permission | None = None,
    feature: Feature | None = None,
) -> VersionRef:
    with session_scope() as db:
        return resolve_version(db, ctx, dataset_id, version_id, permission=permission, feature=feature)


def _check_columns(ref: VersionRef, columns: list[str] | None) -> None:
    if columns is None:
        return
    known = {c["name"] for c in ref.schema}
    unknown = [c for c in columns if c not in known]
    if unknown:
        raise ValidationError(f"Unknown column(s): {', '.join(unknown)}")


def load_frame_with_ref(
    ctx: AuthContext,
    dataset_id: int,
    *,
    version_id: int | None = None,
    columns: list[str] | None = None,
    permission: Permission | None = None,
    feature: Feature | None = None,
) -> tuple[pd.DataFrame, VersionRef]:
    ref = get_version_ref(ctx, dataset_id, version_id, permission=permission, feature=feature)
    _check_columns(ref, columns)
    resource_manager.check_memory(
        resource_manager.estimate_parquet_memory(ref.path, columns), operation="Loading this dataset"
    )
    return read_parquet(ref.path, columns), ref


def load_frame(
    ctx: AuthContext,
    dataset_id: int,
    *,
    version_id: int | None = None,
    columns: list[str] | None = None,
    permission: Permission | None = None,
    feature: Feature | None = None,
) -> pd.DataFrame:
    frame, _ = load_frame_with_ref(
        ctx, dataset_id, version_id=version_id, columns=columns, permission=permission, feature=feature
    )
    return frame


def load_preview(
    ctx: AuthContext, dataset_id: int, *, version_id: int | None = None, max_rows: int | None = None
) -> pd.DataFrame:
    limit = get_settings().preview_max_rows
    rows = min(max_rows or limit, limit)
    ref = get_version_ref(ctx, dataset_id, version_id)
    batches = iter_parquet(ref.path, rows)
    try:
        first = next(batches)
    finally:
        batches.close()
    return first.head(rows)


def iter_frame_batches(
    ctx: AuthContext,
    dataset_id: int,
    *,
    version_id: int | None = None,
    columns: list[str] | None = None,
    batch_rows: int | None = None,
    permission: Permission | None = None,
    feature: Feature | None = None,
) -> Iterator[pd.DataFrame]:
    ref = get_version_ref(ctx, dataset_id, version_id, permission=permission, feature=feature)
    _check_columns(ref, columns)
    yield from iter_parquet(ref.path, batch_rows or get_settings().parquet_batch_rows, columns)


def list_versions(ctx: AuthContext, dataset_id: int) -> list[dict[str, Any]]:
    with session_scope() as db:
        dataset = authorize_dataset(db, ctx, dataset_id)
        rows = db.execute(
            select(DatasetVersion, User.username)
            .outerjoin(User, User.id == DatasetVersion.created_by_id)
            .where(DatasetVersion.dataset_id == dataset.id)
            .order_by(DatasetVersion.version_number.desc())
        ).all()
        return [
            {
                "version_id": v.id,
                "version_number": v.version_number,
                "created_at": v.created_at,
                "created_by": username or "system",
                "operation_type": v.operation_type,
                "rows": v.row_count,
                "columns": v.column_count,
                "size_bytes": v.size_bytes,
                "quality_score": v.quality_score,
                "is_current": v.id == dataset.current_version_id,
                "is_original_import": v.is_pinned,
                "is_purged": v.is_purged,
                "note": v.note,
            }
            for v, username in rows
        ]


def list_operations(ctx: AuthContext, dataset_id: int, *, limit: int = 200) -> list[dict[str, Any]]:
    """Cleaning/operation history of a dataset (applied, failed and cancelled)."""
    with session_scope() as db:
        dataset = authorize_dataset(db, ctx, dataset_id)
        rows = db.execute(
            select(CleaningOperation, User.username)
            .outerjoin(User, User.id == CleaningOperation.user_id)
            .where(CleaningOperation.dataset_id == dataset.id)
            .order_by(CleaningOperation.id.desc())
            .limit(max(1, min(limit, 1000)))
        ).all()
        versions = {
            v.id: v.version_number
            for v in db.execute(
                select(DatasetVersion).where(DatasetVersion.dataset_id == dataset.id)
            ).scalars()
        }
        return [
            {
                "operation_id": op.id,
                "created_at": op.created_at,
                "user": username or "system",
                "operation_type": op.operation_type,
                "status": op.status,
                "parameters": json.loads(op.parameters_json or "{}"),
                "input_version": versions.get(op.input_version_id),
                "output_version": versions.get(op.output_version_id),
                "rows_before": op.rows_before,
                "rows_after": op.rows_after,
                "columns_before": op.columns_before,
                "columns_after": op.columns_after,
                "rows_added": op.rows_added,
                "rows_removed": op.rows_removed,
                "rows_affected": op.rows_affected,
                "cells_changed": op.cells_changed,
                "quality_before": op.quality_before,
                "quality_after": op.quality_after,
                "summary": json.loads(op.summary_json or "{}"),
                "error_message": op.error_message,
                "duration_ms": op.duration_ms,
            }
            for op, username in rows
        ]


def verify_original(ctx: AuthContext, dataset_id: int) -> dict[str, Any]:
    """Check that the stored original upload still matches its recorded checksum."""
    with session_scope() as db:
        dataset = authorize_dataset(db, ctx, dataset_id, include_archived=True)
        if not dataset.original_file_path:
            return {"has_original": False, "intact": None}
        path = storage.resolve_relative(dataset.original_file_path)
        intact = path.exists() and storage.sha256_file(path) == dataset.original_sha256
        return {"has_original": True, "intact": intact, "size_bytes": dataset.original_size_bytes}


# ---------------------------------------------------------------------------
# Restore and generic operation runners
# ---------------------------------------------------------------------------


_NOT_RECORDED = (AuthenticationError, PermissionDeniedError, NotFoundError)


def record_failed_operation(
    ctx: AuthContext,
    dataset_id: int,
    operation_type: str,
    parameters: dict[str, Any] | None,
    error: BaseException,
    *,
    status: str = "failed",
) -> None:
    """Persist a failed or cancelled operation in its own transaction. Never raises."""
    message = error.user_message if isinstance(error, AppError) else "Unexpected error; see the server log."
    try:
        with session_scope() as db:
            dataset = db.get(Dataset, dataset_id)
            if dataset is None or dataset.organization_id != ctx.organization_id:
                return
            db.add(
                CleaningOperation(
                    organization_id=dataset.organization_id,
                    dataset_id=dataset.id,
                    input_version_id=dataset.current_version_id,
                    user_id=ctx.user_id,
                    operation_type=operation_type,
                    parameters_json=json.dumps(parameters or {}, default=str),
                    status=status,
                    error_message=message[:500],
                )
            )
            record_audit(
                db,
                ctx,
                f"dataset.{operation_type}.{status}",
                target_type="dataset",
                target_id=dataset.id,
                details={"error": message[:200]},
            )
    except Exception:  # noqa: BLE001 - must not hide the original failure
        logger.exception("Could not record failed operation on dataset %s", dataset_id)


def record_failure(
    ctx: AuthContext, dataset_id: int, operation_type: str, parameters: Any, exc: BaseException
) -> None:
    """Record why an operation did not publish (for custom flows built on publish_version).

    Authorization and not-found errors are not recorded (they say nothing about
    the dataset and may concern another tenant's id); cancellations and
    timeouts are recorded as "cancelled", everything else as "failed".
    """
    if isinstance(exc, _NOT_RECORDED):
        return
    status = "cancelled" if isinstance(exc, (OperationCancelledError, OperationTimeoutError)) else "failed"
    record_failed_operation(ctx, dataset_id, operation_type, parameters, exc, status=status)


_record_failure = record_failure


FrameTransform = Callable[[pd.DataFrame], tuple[pd.DataFrame, dict[str, int], dict[str, Any]]]


def apply_frame_operation(
    ctx: AuthContext,
    dataset_id: int,
    *,
    operation_type: str,
    parameters: dict[str, Any],
    transform: FrameTransform,
    permission: Permission | None = Permission.CLEAN_DATA,
    feature: Feature | None = None,
    note: str = "",
    guard: OperationGuard | None = None,
) -> PublishResult:
    """Load the current version, apply ``transform`` in memory, publish a new version.

    ``transform(frame) -> (new_frame, stats, summary)`` must not mutate its input
    in place. ``stats`` may contain rows_added, rows_removed, rows_affected,
    cells_changed. The source version is never modified. If the dataset changed
    since it was loaded, a ConflictError is raised instead of overwriting. Any
    failure is recorded in the operation history and no version is published.

    ``guard`` (default: one with OPERATION_TIMEOUT_SECONDS) reports progress and
    is checked before loading, after loading and after the transform, so a
    cancellation or timeout stops the operation before it publishes. The
    transform itself is not interrupted; pass the guard into it for that.
    """
    started = time.monotonic()
    guard = guard or OperationGuard()
    try:
        guard.step(0.0, "Loading the dataset")
        frame, ref = load_frame_with_ref(ctx, dataset_id, permission=permission, feature=feature)
        guard.step(0.2, "Applying the operation")
        new_frame, stats, summary = transform(frame)
        guard.step(0.8, "Publishing the new version")
        with session_scope() as db:
            dataset = authorize_dataset(db, ctx, dataset_id, permission=permission, feature=feature)
            if dataset.current_version_id != ref.version_id:
                raise ConflictError(CHANGED_MESSAGE)
            return publish_version(
                db,
                ctx,
                dataset,
                operation_type=operation_type,
                parameters=parameters,
                frame=new_frame,
                stats=stats,
                summary=summary,
                note=note,
                started=started,
                guard=guard,
            )
    except BaseException as exc:
        _record_failure(ctx, dataset_id, operation_type, parameters, exc)
        raise


BatchTransform = Callable[[pd.DataFrame, int], tuple[pd.DataFrame, dict[str, int]]]


def apply_streaming_operation(
    ctx: AuthContext,
    dataset_id: int,
    *,
    operation_type: str,
    parameters: dict[str, Any],
    transform_batch: BatchTransform,
    permission: Permission | None = Permission.CLEAN_DATA,
    feature: Feature | None = None,
    guard: OperationGuard | None = None,
    summary: dict[str, Any] | None = None,
    note: str = "",
) -> PublishResult:
    """Stream the current version through ``transform_batch`` into a new version.

    ``transform_batch(batch, row_offset) -> (new_batch, stats_delta)`` is called
    for each batch in order; ``row_offset`` is the index of the batch's first
    row in the whole dataset. Stats deltas are summed. Memory stays bounded by
    PARQUET_BATCH_ROWS. Temporary files are removed on success, failure and
    cancellation.
    """
    started = time.monotonic()
    guard = guard or OperationGuard()
    try:
        ref = get_version_ref(ctx, dataset_id, permission=permission, feature=feature)
        resource_manager.check_disk(
            ref.path.stat().st_size * 2, operation=operation_type.replace("_", " ").capitalize()
        )
        totals: dict[str, int] = {}
        with storage.temporary_directory(ref.organization_id) as tmp:
            output = tmp / "result.parquet"
            writer: ParquetBatchWriter | None = None
            offset = 0
            try:
                for batch in iter_parquet(ref.path, get_settings().parquet_batch_rows):
                    guard.step(
                        offset / max(ref.rows, 1), f"Processing rows {offset + 1:,}-{offset + len(batch):,}"
                    )
                    new_batch, delta = transform_batch(batch, offset)
                    new_batch = normalize_frame(new_batch)
                    if writer is None:
                        writer = ParquetBatchWriter(output, arrow_schema_for_frame(new_batch)).__enter__()
                    writer.write(new_batch)
                    for key, value in delta.items():
                        totals[key] = totals.get(key, 0) + int(value)
                    offset += len(batch)
            except BaseException as exc:
                if writer is not None:
                    writer.__exit__(type(exc), exc, exc.__traceback__)
                raise
            if writer is not None:
                writer.__exit__(None, None, None)
            guard.step(1.0, "Publishing the new version")
            with session_scope() as db:
                dataset = authorize_dataset(db, ctx, dataset_id, permission=permission, feature=feature)
                if dataset.current_version_id != ref.version_id:
                    raise ConflictError(CHANGED_MESSAGE)
                return publish_version(
                    db,
                    ctx,
                    dataset,
                    operation_type=operation_type,
                    parameters=parameters,
                    parquet_source=output,
                    stats=totals,
                    summary=summary,
                    note=note,
                    started=started,
                    guard=guard,
                )
    except BaseException as exc:
        _record_failure(ctx, dataset_id, operation_type, parameters, exc)
        raise


def restore_version(ctx: AuthContext, dataset_id: int, version_id: int) -> PublishResult:
    """Make an earlier version current again by copying it forward as a new version."""
    started = time.monotonic()
    parameters: dict[str, Any] = {"version_id": version_id}
    try:
        with session_scope() as db:
            dataset = authorize_dataset(db, ctx, dataset_id, permission=Permission.CLEAN_DATA)
            version = _get_version(db, dataset, version_id)
            if version.id == dataset.current_version_id:
                raise ConflictError("This version is already the current version.")
            source = storage.resolve_relative(version.file_path)
            if not source.exists() or storage.sha256_file(source) != version.file_sha256:
                raise ConflictError(
                    "The stored data for this version failed its integrity check and cannot be restored."
                )
            parameters = {"restored_from_version": version.version_number}
            with storage.temporary_directory(dataset.organization_id) as tmp:
                copy = tmp / "restore.parquet"
                storage.atomic_copy(source, copy)
                return publish_version(
                    db,
                    ctx,
                    dataset,
                    operation_type="restore",
                    parameters=parameters,
                    parquet_source=copy,
                    summary=parameters,
                    note=f"Restored from version {version.version_number}",
                    started=started,
                )
    except BaseException as exc:
        _record_failure(ctx, dataset_id, "restore", parameters, exc)
        raise
