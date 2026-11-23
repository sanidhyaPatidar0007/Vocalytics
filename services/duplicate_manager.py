"""Duplicate detection and unique-identifier management (specification section 9).

Exact duplicate rows are found with a two-pass, cross-chunk algorithm that never
holds the whole dataset in memory:

* Pass 1 hashes every row of every chunk (64-bit ``hash_rows`` on the selected
  columns) and records the hashes in global row order. When the number of rows
  exceeds ``spill_threshold_rows`` the hash array is written to a file and read
  back through a NumPy memmap, then candidate hashes are found partition by
  partition so memory stays bounded by the threshold.
* Pass 2 reads the data again and keeps only the candidate rows (rows whose
  hash occurs more than once). Candidates are grouped by their actual values,
  so two different rows whose hashes collide are never reported as duplicates.

Chunks must use the same dtypes for every chunk (Parquet batches of one version
do; CSV chunks should be read with ``dtype=str``), otherwise equal values could
hash differently. Missing values compare equal to each other, as in pandas.

Row numbers in every result are zero-based positions in the analysed version.
Stored values are never modified by detection functions; changes are published
as new versions through ``versioning_service`` so they can be undone by restore.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from auth.authorization import AuthContext
from core import storage
from core.constants import Feature, LogicalType, Permission
from core.errors import ConflictError, ValidationError
from services import resource_manager, versioning_service
from services.frame_utils import hash_rows, json_safe, records
from services.resource_manager import OperationGuard
from services.versioning_service import PublishResult, VersionRef

KEEP_OPTIONS = ("first", "last", "none")
KEY_POLICIES = ("keep_first", "keep_last", "keep_most_complete", "drop_all")
NORMALIZATION_RULES = ("email", "phone", "identifier", "none")
ID_MODES = ("uuid4", "sequential")
ID_TARGETS = ("fill_missing_and_duplicates", "new_column")

DEFAULT_SPILL_THRESHOLD_ROWS = 5_000_000
MAX_PREVIEW_GROUPS = 1_000
SAMPLE_ROWS = 20
EXAMPLE_ROWS_PER_VALUE = 5
ORIGINAL_EXAMPLES_PER_VALUE = 3
SUMMARY_INDEX_CAP = 10_000
MAPPING_CAP = 50_000
IDENTIFIER_UNIQUENESS_RATIO = 0.95

REPEATED_VALUES_NOTE = (
    "Repeated values are not the same as duplicate records. Names, cities, categories and even e-mail "
    "addresses can legitimately repeat; only rows that are identical in every compared column are "
    "duplicate records. No stored value was changed."
)

GMAIL_NOTE = (
    "E-mail comparison trims spaces and ignores letter case. For gmail.com addresses, dots and any "
    "'+tag' in the part before '@' are also ignored, because Gmail delivers those variants to the same inbox."
)

_IDENTIFIER_TOKENS = frozenset({"id", "ids", "uuid", "guid", "code", "number", "num", "no", "sku", "key"})
_MISSING = ("<missing>",)  # hashable, self-equal marker for missing values inside value tuples
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


# ---------------------------------------------------------------------------
# Column name helpers (shared with services.missing_values)
# ---------------------------------------------------------------------------


def column_name_tokens(name: str) -> list[str]:
    """Split a column name into lowercase word tokens (snake_case, kebab, spaces, camelCase)."""
    spaced = _CAMEL.sub(" ", str(name))
    return [token for token in re.split(r"[^0-9a-zA-Z]+", spaced.lower()) if token]


def looks_like_identifier_name(name: str) -> bool:
    """True when a whole token of the name is an identifier word such as id, code or uuid."""
    return any(token in _IDENTIFIER_TOKENS for token in column_name_tokens(name))


# ---------------------------------------------------------------------------
# Cross-chunk exact duplicate algorithm
# ---------------------------------------------------------------------------


@dataclass
class DuplicateScan:
    """Internal result of the two-pass scan: verified groups of identical rows."""

    groups: list[list[int]]
    rows_scanned: int
    spilled: bool
    candidate_rows: int

    @property
    def collision_rows(self) -> int:
        """Candidate rows whose hash matched another row but whose values did not."""
        return self.candidate_rows - sum(len(group) for group in self.groups)


def _validate_keep(keep: str) -> str:
    if keep not in KEEP_OPTIONS:
        raise ValidationError("Choose which rows to keep: 'first', 'last' or 'none'.")
    return keep


def _repeated_mask(hashes: np.ndarray) -> np.ndarray:
    """Boolean mask of entries whose hash value occurs more than once."""
    if len(hashes) == 0:
        return np.zeros(0, dtype=bool)
    _, inverse, counts = np.unique(hashes, return_inverse=True, return_counts=True)
    return counts[inverse] > 1


class _HashStore:
    """Row hashes in global order, in memory or spilled to a disk-backed memmap."""

    def __init__(self, threshold: int, spill_dir: Callable[[], Path]) -> None:
        self.threshold = max(1, int(threshold))
        self._spill_dir = spill_dir
        self._parts: list[np.ndarray] = []
        self._handle: Any = None
        self.path: Path | None = None
        self.count = 0

    @property
    def spilled(self) -> bool:
        return self.path is not None

    def add(self, hashes: np.ndarray) -> None:
        hashes = np.ascontiguousarray(hashes, dtype="<u8")
        if self._handle is None and self.count + len(hashes) > self.threshold:
            self._spill()
        if self._handle is not None:
            self._handle.write(hashes.tobytes())
        else:
            self._parts.append(hashes)
        self.count += len(hashes)

    def _spill(self) -> None:
        directory = self._spill_dir()
        directory.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="dup-hashes-", suffix=".u64", dir=directory)
        self.path = Path(name)
        self._handle = os.fdopen(fd, "wb")
        for part in self._parts:
            self._handle.write(part.tobytes())
        self._parts = []

    def candidates(self) -> np.ndarray:
        """Sorted global row indices whose hash occurs more than once."""
        if not self.spilled:
            hashes = np.concatenate(self._parts) if self._parts else np.empty(0, dtype="<u8")
            return np.nonzero(_repeated_mask(hashes))[0].astype(np.int64)
        self._handle.flush()
        self._handle.close()
        self._handle = None
        return self._spilled_candidates()

    def _spilled_candidates(self) -> np.ndarray:
        """Find repeated hashes partition by partition so memory stays near ``threshold`` entries."""
        if self.count == 0:
            return np.empty(0, dtype=np.int64)
        mapped = np.memmap(self.path, dtype="<u8", mode="r", shape=(self.count,))
        try:
            partitions = max(1, math.ceil(self.count / self.threshold))
            found: list[np.ndarray] = []
            for partition in range(partitions):
                hashes: list[np.ndarray] = []
                positions: list[np.ndarray] = []
                for start in range(0, self.count, self.threshold):
                    block = np.array(mapped[start : start + self.threshold])
                    selected = np.nonzero(block % np.uint64(partitions) == np.uint64(partition))[0]
                    hashes.append(block[selected])
                    positions.append(selected.astype(np.int64) + start)
                part_hashes = np.concatenate(hashes)
                part_positions = np.concatenate(positions)
                found.append(part_positions[_repeated_mask(part_hashes)])
            return np.sort(np.concatenate(found))
        finally:
            mapped.flush()
            del mapped

    def close(self) -> None:
        """Close and delete the spill file (the memmap is already released)."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self.path is not None:
            storage.remove_file_quietly(self.path)


def _canonical(value: Any) -> Any:
    """Hashable comparison form of a cell: every kind of missing value becomes one marker."""
    if value is None or value is pd.NA or value is pd.NaT:
        return _MISSING
    if isinstance(value, float) and math.isnan(value):
        return _MISSING
    if isinstance(value, np.generic):
        return _canonical(value.item())
    return value


@contextlib.contextmanager
def _iterate(iterable: Iterable[pd.DataFrame]) -> Iterator[Iterator[pd.DataFrame]]:
    """Iterate a chunk source and close it afterwards when it supports closing (files, generators)."""
    iterator = iter(iterable)
    try:
        yield iterator
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def _select(chunk: pd.DataFrame, subset: list[str] | None) -> pd.DataFrame:
    if subset is None:
        return chunk
    missing = [name for name in subset if name not in chunk.columns]
    if missing:
        raise ValidationError(f"Unknown column(s): {', '.join(str(m) for m in missing)}")
    return chunk[subset]


def scan_duplicates(
    chunk_source: Callable[[], Iterable[pd.DataFrame]],
    *,
    subset: list[str] | None = None,
    spill_dir: Path | None = None,
    spill_threshold_rows: int = DEFAULT_SPILL_THRESHOLD_ROWS,
    guard: OperationGuard | None = None,
) -> DuplicateScan:
    """Run both passes and return verified groups of identical rows (see module docstring)."""
    with contextlib.ExitStack() as stack:
        scratch: list[Path] = []

        def spill_location() -> Path:
            if spill_dir is not None:
                return Path(spill_dir)
            if not scratch:
                scratch.append(stack.enter_context(storage.temporary_directory(None, prefix="dup-")))
            return scratch[0]

        store = _HashStore(spill_threshold_rows, spill_location)
        stack.callback(store.close)
        for chunk in chunk_source():
            if guard is not None:
                guard.check()
            store.add(hash_rows(_select(chunk, subset)))
        candidates = store.candidates()

        groups: dict[tuple[Any, ...], list[int]] = {}
        offset = 0
        if len(candidates):
            with _iterate(chunk_source()) as chunks:
                for chunk in chunks:
                    if guard is not None:
                        guard.check()
                    size = len(chunk)
                    low, high = np.searchsorted(candidates, [offset, offset + size])
                    if high > low:
                        local = candidates[low:high] - offset
                        part = _select(chunk, subset).iloc[local]
                        for position, row in zip(
                            candidates[low:high].tolist(),
                            part.itertuples(index=False, name=None),
                            strict=True,
                        ):
                            groups.setdefault(tuple(_canonical(v) for v in row), []).append(int(position))
                    offset += size
                    if offset > int(candidates[-1]):
                        break
        verified = sorted((members for members in groups.values() if len(members) > 1), key=lambda g: g[0])
        return DuplicateScan(
            groups=verified,
            rows_scanned=store.count,
            spilled=store.spilled,
            candidate_rows=int(len(candidates)),
        )


def rows_to_remove(groups: list[list[int]], keep: str) -> np.ndarray:
    """Sorted global indices removed under ``keep`` (first, last or none)."""
    _validate_keep(keep)
    removed: list[int] = []
    for group in groups:
        if keep == "first":
            removed.extend(group[1:])
        elif keep == "last":
            removed.extend(group[:-1])
        else:
            removed.extend(group)
    return np.array(sorted(removed), dtype=np.int64)


def _method_text(scan: DuplicateScan) -> str:
    storage_text = (
        "hashes were spilled to a disk-backed file and compared partition by partition"
        if scan.spilled
        else "hashes were compared in memory"
    )
    return (
        "Exact cross-chunk detection: every row was hashed with a 64-bit hash, "
        f"{storage_text}, and rows with matching hashes were verified by comparing their actual values "
        f"({scan.collision_rows} hash collision(s) rejected)."
    )


def find_duplicates_in_chunks(
    chunk_source: Callable[[], Iterable[pd.DataFrame]],
    *,
    subset: list[str] | None = None,
    keep: str = "first",
    spill_dir: Path | None = None,
    spill_threshold_rows: int = DEFAULT_SPILL_THRESHOLD_ROWS,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Exact duplicate rows across all chunks produced by ``chunk_source``.

    ``chunk_source`` is called twice and must return a fresh iterator of
    DataFrames each time, in the same order with the same dtypes. Returns the
    rows removed under ``keep`` ("first", "last" or "none" to remove every
    member of a duplicate group), the verified groups of global row indices,
    counts and a description of the method used.
    """
    _validate_keep(keep)
    if not isinstance(spill_threshold_rows, int) or spill_threshold_rows < 1:
        raise ValidationError("The spill threshold must be a positive number of rows.")
    scan = scan_duplicates(
        chunk_source,
        subset=subset,
        spill_dir=spill_dir,
        spill_threshold_rows=spill_threshold_rows,
        guard=guard,
    )
    removed = rows_to_remove(scan.groups, keep)
    return {
        "duplicate_row_indices": removed.tolist(),
        "groups": scan.groups,
        "duplicate_rows": int(len(removed)),
        "group_count": len(scan.groups),
        "rows_scanned": scan.rows_scanned,
        "spilled_to_disk": scan.spilled,
        "hash_collisions_rejected": scan.collision_rows,
        "method": _method_text(scan),
    }


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _ref(
    ctx: AuthContext, dataset_id: int, *, feature: Feature, permission: Permission | None = None
) -> VersionRef:
    """Authorize first (tenant, permission, plan feature) and pin the current version."""
    return versioning_service.get_version_ref(ctx, dataset_id, permission=permission, feature=feature)


def _types(ref: VersionRef) -> dict[str, str]:
    return {column["name"]: column["logical_type"] for column in ref.schema}


def _require_columns(ref: VersionRef, columns: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if columns is None and allow_empty:
        return []
    if not isinstance(columns, (list, tuple)) or (not columns and not allow_empty):
        raise ValidationError(f"Select at least one {label}.")
    names = [str(c) for c in columns]
    if len(set(names)) != len(names):
        raise ValidationError(f"Each {label} can be selected only once.")
    known = _types(ref)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValidationError(f"Unknown column(s): {', '.join(unknown)}")
    return names


def _source(
    ctx: AuthContext, ref: VersionRef, feature: Feature, columns: list[str] | None = None
) -> Callable[[], Iterator[pd.DataFrame]]:
    """Fresh batch iterator over the pinned version (re-authorized on every pass)."""

    def batches() -> Iterator[pd.DataFrame]:
        return versioning_service.iter_frame_batches(
            ctx, ref.dataset_id, version_id=ref.version_id, columns=columns, feature=feature
        )

    return batches


def _guarded(
    source: Callable[[], Iterator[pd.DataFrame]], guard: OperationGuard, total: int
) -> Callable[[], Iterator[pd.DataFrame]]:
    """Wrap a source so each batch reports progress and honours cancellation."""

    def batches() -> Iterator[pd.DataFrame]:
        seen = 0
        with contextlib.closing(source()) as inner:
            for batch in inner:
                guard.step(seen / max(total, 1), f"Scanning rows {seen + 1:,}-{seen + len(batch):,}")
                seen += len(batch)
                yield batch

    return batches


def _fetch_rows(source: Callable[[], Iterator[pd.DataFrame]], indices: Iterable[int]) -> pd.DataFrame:
    """Rows at the given global positions, indexed by position (one bounded pass)."""
    wanted = np.array(sorted(set(int(i) for i in indices)), dtype=np.int64)
    parts: list[pd.DataFrame] = []
    offset = 0
    template: pd.DataFrame | None = None
    with contextlib.closing(source()) as batches:
        for batch in batches:
            template = batch.iloc[0:0] if template is None else template
            if not len(wanted):
                break
            low, high = np.searchsorted(wanted, [offset, offset + len(batch)])
            if high > low:
                part = batch.iloc[wanted[low:high] - offset].copy()
                part.index = wanted[low:high]
                parts.append(part)
            offset += len(batch)
            if offset > int(wanted[-1]):
                break
    if parts:
        return pd.concat(parts)
    return template if template is not None else pd.DataFrame()


def _row_record(frame: pd.DataFrame, position: int) -> dict[str, Any]:
    return records(frame.loc[[position]])[0]


def _ensure_unchanged(ctx: AuthContext, ref: VersionRef, *, permission: Permission, feature: Feature) -> None:
    current = versioning_service.get_version_ref(ctx, ref.dataset_id, permission=permission, feature=feature)
    if current.version_id != ref.version_id:
        raise ConflictError(
            "The dataset changed while this operation was being prepared. Review the latest version and try again."
        )


def _capped(values: list[Any], cap: int) -> tuple[list[Any], bool]:
    return values[:cap], len(values) > cap


def _publish_removal(
    ctx: AuthContext,
    ref: VersionRef,
    removal: np.ndarray,
    *,
    operation_type: str,
    parameters: dict[str, Any],
    summary: dict[str, Any],
    feature: Feature,
    guard: OperationGuard | None = None,
) -> PublishResult:
    """Stream the pinned version into a new one without the rows in ``removal`` (order preserved)."""
    checked: list[bool] = []

    def transform(batch: pd.DataFrame, offset: int) -> tuple[pd.DataFrame, dict[str, int]]:
        if not checked:
            _ensure_unchanged(ctx, ref, permission=Permission.CLEAN_DATA, feature=feature)
            checked.append(True)
        low, high = np.searchsorted(removal, [offset, offset + len(batch)])
        if high == low:
            return batch, {"rows_removed": 0}
        keep = np.ones(len(batch), dtype=bool)
        keep[removal[low:high] - offset] = False
        return batch.iloc[keep].reset_index(drop=True), {"rows_removed": int(high - low)}

    return versioning_service.apply_streaming_operation(
        ctx,
        ref.dataset_id,
        operation_type=operation_type,
        parameters=parameters,
        transform_batch=transform,
        permission=Permission.CLEAN_DATA,
        feature=feature,
        guard=guard,
        summary=summary,
    )


def _removal_result(ref: VersionRef, removal: np.ndarray, published: PublishResult | None) -> dict[str, Any]:
    indices, truncated = _capped(removal.tolist(), SUMMARY_INDEX_CAP)
    return {
        "changed": published is not None,
        "version_number": published.version_number if published else None,
        "version_id": published.version_id if published else None,
        "rows_before": ref.rows,
        "rows_after": ref.rows - int(len(removal)),
        "rows_removed": int(len(removal)),
        "removed_row_indices": indices,
        "removed_row_indices_truncated": truncated,
        "quality_before": published.quality_before if published else ref.quality_score,
        "quality_after": published.quality_score if published else ref.quality_score,
    }


# ---------------------------------------------------------------------------
# A. Exact duplicate rows
# ---------------------------------------------------------------------------


def _validate_max_groups(max_groups: int) -> int:
    if (
        not isinstance(max_groups, int)
        or isinstance(max_groups, bool)
        or not 1 <= max_groups <= MAX_PREVIEW_GROUPS
    ):
        raise ValidationError(f"The number of groups to show must be between 1 and {MAX_PREVIEW_GROUPS}.")
    return max_groups


def find_exact_duplicates(
    ctx: AuthContext, dataset_id: int, *, max_groups: int = 50, guard: OperationGuard | None = None
) -> dict[str, Any]:
    """Count exact duplicate rows of the current version and preview up to ``max_groups`` groups.

    ``duplicate_rows`` is the number of extra copies (rows a keep-first removal
    would delete); ``rows_in_groups`` counts every member of a duplicate group.
    """
    ref = _ref(ctx, dataset_id, feature=Feature.DUPLICATE_DETECTION)
    _validate_max_groups(max_groups)
    source = _source(ctx, ref, Feature.DUPLICATE_DETECTION)
    scan_source = _guarded(source, guard, ref.rows) if guard is not None else source
    scan = scan_duplicates(scan_source, guard=guard)
    removable = rows_to_remove(scan.groups, "first")
    shown = scan.groups[:max_groups]
    rows = _fetch_rows(source, [group[0] for group in shown])
    return {
        "rows": ref.rows,
        "duplicate_rows": int(len(removable)),
        "pct": round(100.0 * len(removable) / ref.rows, 4) if ref.rows else 0.0,
        "group_count": len(scan.groups),
        "rows_in_groups": sum(len(group) for group in scan.groups),
        "groups": [
            {"rows": group, "size": len(group), "values": _row_record(rows, group[0])} for group in shown
        ],
        "groups_truncated": len(scan.groups) > len(shown),
        "method": _method_text(scan),
    }


def _removal_plan(
    ctx: AuthContext, ref: VersionRef, keep: str, subset: list[str] | None
) -> tuple[DuplicateScan, np.ndarray]:
    _validate_keep(keep)
    columns = _require_columns(ref, subset, "column to compare", allow_empty=True) or None
    scan = scan_duplicates(_source(ctx, ref, Feature.DUPLICATE_DETECTION), subset=columns)
    return scan, rows_to_remove(scan.groups, keep)


def preview_remove_duplicates(
    ctx: AuthContext, dataset_id: int, *, keep: str = "first", subset: list[str] | None = None
) -> dict[str, Any]:
    """Read-only preview of a duplicate removal: counts and a sample of the rows that would go."""
    ref = _ref(ctx, dataset_id, feature=Feature.DUPLICATE_DETECTION)
    scan, removal = _removal_plan(ctx, ref, keep, subset)
    sample_positions = removal[:SAMPLE_ROWS].tolist()
    rows = _fetch_rows(_source(ctx, ref, Feature.DUPLICATE_DETECTION), sample_positions)
    return {
        "keep": keep,
        "subset": subset,
        "rows_before": ref.rows,
        "rows_after": ref.rows - int(len(removal)),
        "rows_removed": int(len(removal)),
        "group_count": len(scan.groups),
        "sample_removed": [{"row_index": p, "values": _row_record(rows, p)} for p in sample_positions],
        "method": _method_text(scan),
    }


def remove_duplicates(
    ctx: AuthContext,
    dataset_id: int,
    *,
    keep: str = "first",
    subset: list[str] | None = None,
    confirm: bool = False,
    guard: OperationGuard | None = None,
) -> dict[str, Any]:
    """Remove exact duplicate rows (order preserved) and publish a new version.

    Requires ``confirm=True``. When no duplicates exist nothing is published.
    """
    ref = _ref(ctx, dataset_id, feature=Feature.DUPLICATE_DETECTION, permission=Permission.CLEAN_DATA)
    _validate_keep(keep)
    _require_columns(ref, subset, "column to compare", allow_empty=True)
    if confirm is not True:
        raise ValidationError(
            "Removing duplicate rows deletes data. Review the preview and confirm to continue."
        )
    scan, removal = _removal_plan(ctx, ref, keep, subset)
    if not len(removal):
        return _removal_result(ref, removal, None)
    indices, truncated = _capped(removal.tolist(), SUMMARY_INDEX_CAP)
    parameters = {"keep": keep, "subset": subset}
    summary = {
        "keep": keep,
        "subset": subset,
        "groups": len(scan.groups),
        "rows_removed": int(len(removal)),
        "removed_row_indices": indices,
        "removed_row_indices_truncated": truncated,
        "method": _method_text(scan),
    }
    published = _publish_removal(
        ctx,
        ref,
        removal,
        operation_type="remove_duplicates",
        parameters=parameters,
        summary=summary,
        feature=Feature.DUPLICATE_DETECTION,
        guard=guard,
    )
    return _removal_result(ref, removal, published)


# ---------------------------------------------------------------------------
# B. Repeated values in selected columns
# ---------------------------------------------------------------------------


def _blank_to_na(values: pd.Series) -> pd.Series:
    return values.mask(values.str.len() == 0)


def normalize_values(
    values: pd.Series, rule: str = "none", *, trim: bool = True, case_insensitive: bool = True
) -> pd.Series:
    """Comparison keys for ``values`` (never written back). Missing or blank values become NA.

    Rules: "email" trims and lowercases (gmail.com also ignores dots and +tags
    in the local part); "phone" keeps digits only plus a leading "+";
    "identifier" trims, uppercases and removes spaces and dashes; "none" applies
    only the ``trim`` and ``case_insensitive`` options.
    """
    text = values.astype("string")
    if rule == "email":
        text = text.str.strip().str.lower()
        parts = text.str.extract(r"^(?P<local>.*)@(?P<domain>[^@]*)$")
        gmail = parts["domain"].eq("gmail.com").fillna(False).astype(bool)
        if gmail.any():
            local = (
                parts.loc[gmail, "local"]
                .str.replace(r"\+.*$", "", regex=True)
                .str.replace(".", "", regex=False)
            )
            text = text.copy()
            text[gmail] = local + "@gmail.com"
        return _blank_to_na(text)
    if rule == "phone":
        stripped = text.str.strip()
        digits = stripped.str.replace(r"\D", "", regex=True)
        digits = digits.mask(digits.str.len() == 0)
        plus = stripped.str.startswith("+").fillna(False).astype(bool)
        return digits.mask(plus & digits.notna(), "+" + digits)
    if rule == "identifier":
        return _blank_to_na(text.str.strip().str.upper().str.replace(r"[\s\-]", "", regex=True))
    if trim:
        text = text.str.strip()
    if case_insensitive:
        text = text.str.casefold()
    return _blank_to_na(text)


@dataclass
class _ValueCounts:
    counts: Counter
    non_missing: int
    missing: int


def _count_values(
    source: Callable[[], Iterator[pd.DataFrame]], column: str, rule: str, trim: bool, ci: bool
) -> _ValueCounts:
    counts: Counter = Counter()
    non_missing = missing = 0
    for batch in source():
        keys = normalize_values(batch[column], rule, trim=trim, case_insensitive=ci)
        present = keys.dropna()
        missing += len(keys) - len(present)
        non_missing += len(present)
        counts.update(present.value_counts().to_dict())
    return _ValueCounts(counts=counts, non_missing=non_missing, missing=missing)


def _value_examples(
    source: Callable[[], Iterator[pd.DataFrame]],
    column: str,
    keys: list[str],
    rule: str,
    trim: bool,
    ci: bool,
) -> dict[str, dict[str, list[Any]]]:
    """Example row numbers and distinct original spellings for the selected keys (second pass)."""
    examples: dict[str, dict[str, list[Any]]] = {key: {"rows": [], "originals": []} for key in keys}
    if not keys:
        return examples
    wanted = set(keys)
    offset = 0
    for batch in source():
        normalized = normalize_values(batch[column], rule, trim=trim, case_insensitive=ci)
        hits = normalized.isin(wanted).fillna(False).to_numpy(dtype=bool)
        for position in np.nonzero(hits)[0].tolist():
            key = normalized.iloc[position]
            entry = examples[key]
            if len(entry["rows"]) < EXAMPLE_ROWS_PER_VALUE:
                entry["rows"].append(offset + position)
            original = json_safe(batch[column].iloc[position])
            if len(entry["originals"]) < ORIGINAL_EXAMPLES_PER_VALUE and original not in entry["originals"]:
                entry["originals"].append(original)
        offset += len(batch)
    return examples


def _repeated_values(
    source: Callable[[], Iterator[pd.DataFrame]], column: str, rule: str, trim: bool, ci: bool, top: int
) -> dict[str, Any]:
    tally = _count_values(source, column, rule, trim, ci)
    repeated = sorted(((k, c) for k, c in tally.counts.items() if c > 1), key=lambda kc: (-kc[1], str(kc[0])))
    shown = repeated[:top]
    examples = _value_examples(source, column, [k for k, _ in shown], rule, trim, ci)
    return {
        "column": column,
        "rule": rule,
        "non_missing": tally.non_missing,
        "missing": tally.missing,
        "distinct_values": len(tally.counts),
        "repeated_value_count": len(repeated),
        "rows_with_repeated_values": int(sum(c for _, c in repeated)),
        "values": [
            {
                "value": json_safe(key),
                "count": int(count),
                "example_rows": examples[key]["rows"],
                "original_values": examples[key]["originals"],
            }
            for key, count in shown
        ],
        "truncated": len(repeated) > len(shown),
    }


def find_value_duplicates(
    ctx: AuthContext,
    dataset_id: int,
    columns: list[str],
    *,
    trim: bool = True,
    case_insensitive: bool = True,
    normalization: dict[str, str] | None = None,
    top: int = 100,
) -> dict[str, Any]:
    """Repeated values per selected column after comparison-only normalization (read-only)."""
    ref = _ref(ctx, dataset_id, feature=Feature.DUPLICATE_DETECTION)
    names = _require_columns(ref, columns, "column")
    rules = dict(normalization or {})
    for column, rule in rules.items():
        if column not in names:
            raise ValidationError(f"A normalization rule was given for '{column}', which is not selected.")
        if rule not in NORMALIZATION_RULES:
            raise ValidationError(
                f"Unknown normalization rule '{rule}'. Use email, phone, identifier or none."
            )
    if not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= 10_000:
        raise ValidationError("The number of values to show must be between 1 and 10000.")
    resource_manager.check_memory(
        resource_manager.estimate_cells_memory(ref.rows, len(names)), operation="Counting repeated values"
    )
    results = []
    for name in names:
        source = _source(ctx, ref, Feature.DUPLICATE_DETECTION, [name])
        results.append(_repeated_values(source, name, rules.get(name, "none"), trim, case_insensitive, top))
    notes = [REPEATED_VALUES_NOTE]
    if "email" in rules.values():
        notes.append(GMAIL_NOTE)
    return {"rows": ref.rows, "columns": results, "note": " ".join(notes), "stored_values_modified": False}


# ---------------------------------------------------------------------------
# C. Unique identifier validation and generation
# ---------------------------------------------------------------------------


def _identifier_explanation(name: str, by_name: bool, ratio: float, is_unique: bool, missing: int) -> str:
    reasons = []
    if by_name:
        reasons.append(f"the name '{name}' looks like an identifier")
    if ratio >= IDENTIFIER_UNIQUENESS_RATIO:
        reasons.append(f"{ratio:.1%} of its non-missing values are distinct")
    if not reasons:
        return (
            f"'{name}' does not look like an identifier ({ratio:.1%} distinct values), so repeated values "
            "are probably legitimate and should not be treated as errors."
        )
    status = (
        "Every non-missing value is unique."
        if is_unique
        else "Some values repeat, which an identifier should not do."
    )
    if missing:
        status += f" {missing} row(s) have no value."
    return f"This column should probably be unique because {' and '.join(reasons)}. {status}"


def validate_identifier(
    ctx: AuthContext, dataset_id: int, column: str, *, trim: bool = True, case_insensitive: bool = False
) -> dict[str, Any]:
    """Check whether ``column`` is unique and whether it looks like it should be (read-only)."""
    ref = _ref(ctx, dataset_id, feature=Feature.DUPLICATE_DETECTION)
    (name,) = _require_columns(ref, [column], "column")
    resource_manager.check_memory(
        resource_manager.estimate_cells_memory(ref.rows, 1), operation="Checking identifier uniqueness"
    )
    source = _source(ctx, ref, Feature.DUPLICATE_DETECTION, [name])
    details = _repeated_values(source, name, "none", trim, case_insensitive, top=10)
    ratio = details["distinct_values"] / details["non_missing"] if details["non_missing"] else 0.0
    by_name = looks_like_identifier_name(name)
    is_unique = details["repeated_value_count"] == 0
    should_be_unique = by_name or (details["non_missing"] >= 2 and ratio >= IDENTIFIER_UNIQUENESS_RATIO)
    return {
        "column": name,
        "is_unique": is_unique,
        "valid_identifier": is_unique and details["missing"] == 0,
        "missing_count": details["missing"],
        "non_missing": details["non_missing"],
        "distinct_values": details["distinct_values"],
        "uniqueness_ratio": round(ratio, 6),
        "duplicate_value_count": details["repeated_value_count"],
        "rows_involved": details["rows_with_repeated_values"],
        "examples": details["values"],
        "looks_like_identifier": by_name,
        "should_be_unique": should_be_unique,
        "explanation": _identifier_explanation(name, by_name, ratio, is_unique, details["missing"]),
        "note": REPEATED_VALUES_NOTE,
    }


@dataclass
class _IdPlan:
    positions: np.ndarray
    new_values: list[Any]
    old_values: list[Any]
    missing_ids: int
    repeated_ids: int
    output_logical: str


def _validate_id_options(
    ref: VersionRef, column: str, mode: str, prefix: str, target: str, new_column: str | None, width: int
) -> str:
    """Validate id generation options and return the column's logical type."""
    (name,) = _require_columns(ref, [column], "identifier column")
    logical = _types(ref)[name]
    if mode not in ID_MODES:
        raise ValidationError("Choose an identifier mode: 'uuid4' or 'sequential'.")
    if target not in ID_TARGETS:
        raise ValidationError("Choose a target: 'fill_missing_and_duplicates' or 'new_column'.")
    if not isinstance(prefix, str) or len(prefix) > 50 or any(ord(ch) < 32 for ch in prefix):
        raise ValidationError("The prefix must be at most 50 printable characters.")
    if not isinstance(width, int) or isinstance(width, bool) or not 1 <= width <= 20:
        raise ValidationError("The number width must be between 1 and 20 digits.")
    if target == "new_column":
        if not isinstance(new_column, str) or not new_column.strip() or len(new_column) > 128:
            raise ValidationError("Enter a name of 1 to 128 characters for the new identifier column.")
        if new_column.strip().casefold() in {c.casefold() for c in _types(ref)}:
            raise ValidationError("A column with this name already exists.")
        return LogicalType.TEXT.value
    if logical == LogicalType.INTEGER.value:
        if mode != "sequential" or prefix:
            raise ValidationError(
                "A whole-number identifier column can only receive sequential numbers without a prefix."
            )
        return logical
    if logical != LogicalType.TEXT.value:
        raise ValidationError("Identifiers can only be generated in a text or whole-number column.")
    return logical


def _id_candidates(mode: str, prefix: str, width: int, logical: str) -> Iterator[Any]:
    if mode == "uuid4":
        while True:
            yield f"{prefix}{uuid.uuid4()}"
    number = 1
    while True:
        yield number if logical == LogicalType.INTEGER.value else f"{prefix}{number:0{width}d}"
        number += 1


def _plan_ids(
    values: pd.Series, *, mode: str, prefix: str, width: int, target: str, output_logical: str
) -> _IdPlan:
    """Choose the rows that receive ids and generate collision-free ids for them."""
    if output_logical == LogicalType.INTEGER.value:
        keys = values.astype("Int64")
        missing = keys.isna()
        taken: set[Any] = {int(v) for v in keys.dropna().tolist()}
    else:
        keys = values.astype("string").str.strip()
        missing = (keys.isna() | (keys.str.len() == 0)).fillna(True).astype(bool)
        taken = {str(v).casefold() for v in keys[~missing].tolist()}
    repeated = keys.duplicated(keep="first") & ~missing
    if target == "new_column":
        needed = np.ones(len(values), dtype=bool)
    else:
        needed = (missing | repeated).to_numpy(dtype=bool)
    positions = np.nonzero(needed)[0].astype(np.int64)
    generated: list[Any] = []
    candidates = _id_candidates(mode, prefix, width, output_logical)
    for _ in range(len(positions)):
        for candidate in candidates:
            marker = candidate if output_logical == LogicalType.INTEGER.value else candidate.casefold()
            if marker not in taken:
                taken.add(marker)
                generated.append(candidate)
                break
    return _IdPlan(
        positions=positions,
        new_values=generated,
        old_values=[json_safe(values.iloc[p]) for p in positions.tolist()],
        missing_ids=int(missing.sum()),
        repeated_ids=int(repeated.sum()),
        output_logical=output_logical,
    )


def _mapping(plan: _IdPlan, limit: int) -> list[dict[str, Any]]:
    return [
        {"row_index": int(p), "old_value": old, "new_value": json_safe(new)}
        for p, old, new in zip(
            plan.positions[:limit].tolist(), plan.old_values[:limit], plan.new_values[:limit], strict=True
        )
    ]


def _prepare_ids(
    ctx: AuthContext,
    ref: VersionRef,
    column: str,
    mode: str,
    prefix: str,
    target: str,
    new_column: str | None,
    width: int,
) -> _IdPlan:
    output_logical = _validate_id_options(ref, column, mode, prefix, target, new_column, width)
    frame = versioning_service.load_frame(
        ctx, ref.dataset_id, version_id=ref.version_id, columns=[column], feature=Feature.ADVANCED_CLEANING
    )
    return _plan_ids(
        frame[column], mode=mode, prefix=prefix, width=width, target=target, output_logical=output_logical
    )


def preview_generate_ids(
    ctx: AuthContext,
    dataset_id: int,
    column: str,
    *,
    mode: str = "uuid4",
    prefix: str = "",
    target: str = "fill_missing_and_duplicates",
    new_column: str | None = None,
    width: int = 4,
) -> dict[str, Any]:
    """Read-only preview of identifier generation: which rows change and sample new ids.

    UUIDs are random, so the values applied later differ from the preview;
    sequential ids are deterministic.
    """
    ref = _ref(ctx, dataset_id, feature=Feature.ADVANCED_CLEANING)
    plan = _prepare_ids(ctx, ref, column, mode, prefix, target, new_column, width)
    return {
        "column": column,
        "target": target,
        "mode": mode,
        "new_column": new_column if target == "new_column" else None,
        "rows": ref.rows,
        "ids_to_generate": int(len(plan.positions)),
        "missing_ids": plan.missing_ids,
        "repeated_ids": plan.repeated_ids,
        "mapping_sample": _mapping(plan, SAMPLE_ROWS),
        "note": (
            "Existing unique identifiers are never rewritten. Generated identifiers never repeat an existing value."
            + (
                " UUIDs are random, so the applied values will differ from this preview."
                if mode == "uuid4"
                else ""
            )
        ),
    }


def generate_ids(
    ctx: AuthContext,
    dataset_id: int,
    column: str,
    *,
    mode: str = "uuid4",
    prefix: str = "",
    target: str = "fill_missing_and_duplicates",
    new_column: str | None = None,
    width: int = 4,
    confirm: bool = False,
) -> dict[str, Any]:
    """Generate identifiers and publish a new version; the old to new mapping is kept in history.

    Requires ``confirm=True`` because it rewrites identifier values.
    """
    ref = _ref(ctx, dataset_id, feature=Feature.ADVANCED_CLEANING, permission=Permission.CLEAN_DATA)
    _validate_id_options(ref, column, mode, prefix, target, new_column, width)
    if confirm is not True:
        raise ValidationError(
            "Generating identifiers changes identifier values. Review the preview and confirm to continue."
        )
    plan = _prepare_ids(ctx, ref, column, mode, prefix, target, new_column, width)
    mapping = _mapping(plan, MAPPING_CAP)
    truncated = len(plan.positions) > MAPPING_CAP
    if not len(plan.positions):
        return {
            "changed": False,
            "version_number": None,
            "ids_generated": 0,
            "mapping": [],
            "mapping_truncated": False,
        }
    out_name = new_column.strip() if target == "new_column" and new_column else column
    dtype = "Int64" if plan.output_logical == LogicalType.INTEGER.value else "string"
    new_values = pd.array(plan.new_values, dtype=dtype)
    checked: list[bool] = []

    def transform(batch: pd.DataFrame, offset: int) -> tuple[pd.DataFrame, dict[str, int]]:
        if not checked:
            _ensure_unchanged(ctx, ref, permission=Permission.CLEAN_DATA, feature=Feature.ADVANCED_CLEANING)
            checked.append(True)
        out = batch.copy()
        base = (
            out[column].astype(dtype)
            if target != "new_column"
            else pd.Series(pd.NA, index=out.index, dtype=dtype)
        )
        low, high = np.searchsorted(plan.positions, [offset, offset + len(batch)])
        values = base.to_numpy(dtype=object)
        values[plan.positions[low:high] - offset] = list(new_values[low:high])
        out[out_name] = pd.array(values, dtype=dtype)
        return out, {"cells_changed": int(high - low), "rows_affected": int(high - low)}

    parameters = {
        "column": column,
        "mode": mode,
        "prefix": prefix,
        "target": target,
        "new_column": new_column,
        "width": width,
    }
    summary = {
        **parameters,
        "ids_generated": int(len(plan.positions)),
        "mapping": mapping,
        "mapping_truncated": truncated,
    }
    published = versioning_service.apply_streaming_operation(
        ctx,
        ref.dataset_id,
        operation_type="generate_ids",
        parameters=parameters,
        transform_batch=transform,
        permission=Permission.CLEAN_DATA,
        feature=Feature.ADVANCED_CLEANING,
        summary=summary,
    )
    return {
        "changed": True,
        "version_number": published.version_number,
        "version_id": published.version_id,
        "ids_generated": int(len(plan.positions)),
        "missing_ids": plan.missing_ids,
        "repeated_ids": plan.repeated_ids,
        "mapping": mapping,
        "mapping_truncated": truncated,
        "quality_before": published.quality_before,
        "quality_after": published.quality_score,
    }


# ---------------------------------------------------------------------------
# D. Composite-key duplicates
# ---------------------------------------------------------------------------


@dataclass
class _KeyGroups:
    groups: list[list[int]]
    rows: pd.DataFrame
    excluded_groups: int
    excluded_rows: int


def _key_groups(ctx: AuthContext, ref: VersionRef, key_columns: list[str], *, full_rows: bool) -> _KeyGroups:
    """Verified groups of rows sharing the key, without groups whose key has a missing part."""
    feature = Feature.DUPLICATE_DETECTION
    scan = scan_duplicates(_source(ctx, ref, feature, key_columns), subset=key_columns)
    members = [p for group in scan.groups for p in group]
    width = ref.columns if full_rows else len(key_columns)
    resource_manager.check_memory(
        resource_manager.estimate_cells_memory(len(members), width),
        operation="Comparing rows that share a key",
    )
    rows = _fetch_rows(_source(ctx, ref, feature, None if full_rows else key_columns), members)
    kept: list[list[int]] = []
    excluded_rows = excluded_groups = 0
    for group in scan.groups:
        if rows.loc[group[0], key_columns].isna().any():
            excluded_groups += 1
            excluded_rows += len(group)
        else:
            kept.append(group)
    return _KeyGroups(groups=kept, rows=rows, excluded_groups=excluded_groups, excluded_rows=excluded_rows)


def _count_missing_keys(source: Callable[[], Iterator[pd.DataFrame]], key_columns: list[str]) -> int:
    return int(sum(int(batch[key_columns].isna().any(axis=1).sum()) for batch in source()))


def _differing_columns(group_rows: pd.DataFrame, key_columns: list[str]) -> list[str]:
    return [
        str(name)
        for name in group_rows.columns
        if name not in key_columns and group_rows[name].nunique(dropna=False) > 1
    ]


def find_composite_key_conflicts(
    ctx: AuthContext, dataset_id: int, key_columns: list[str], *, max_groups: int = 50
) -> dict[str, Any]:
    """Groups of rows sharing the composite key, split into identical and conflicting groups.

    Rows with a missing key part are not treated as sharing a key; they are
    counted separately.
    """
    ref = _ref(ctx, dataset_id, feature=Feature.DUPLICATE_DETECTION)
    keys = _require_columns(ref, key_columns, "key column")
    _validate_max_groups(max_groups)
    found = _key_groups(ctx, ref, keys, full_rows=True)
    details = []
    identical = 0
    for group in found.groups:
        subset = found.rows.loc[group]
        differing = _differing_columns(subset, keys)
        identical += not differing
        if len(details) < max_groups:
            details.append(
                {
                    "key": records(subset[keys].head(1))[0],
                    "rows": group,
                    "size": len(group),
                    "identical": not differing,
                    "differing_columns": differing,
                    "records": records(subset),
                }
            )
    return {
        "key_columns": keys,
        "rows": ref.rows,
        "group_count": len(found.groups),
        "identical_groups": identical,
        "conflicting_groups": len(found.groups) - identical,
        "rows_in_groups": sum(len(g) for g in found.groups),
        "rows_with_missing_key": _count_missing_keys(
            _source(ctx, ref, Feature.DUPLICATE_DETECTION, keys), keys
        ),
        "groups": details,
        "groups_truncated": len(found.groups) > len(details),
        "note": (
            "Choose a resolution policy (keep_first, keep_last, keep_most_complete or drop_all) "
            "to resolve these groups."
        ),
    }


def _row_missing_counts(rows: pd.DataFrame) -> pd.Series:
    missing = rows.isna()
    for name in rows.columns:
        if isinstance(rows[name].dtype, pd.StringDtype):
            missing[name] = missing[name] | rows[name].str.strip().eq("").fillna(False)
    return missing.sum(axis=1)


def _policy_removal(found: _KeyGroups, policy: str) -> np.ndarray:
    removed: list[int] = []
    counts = _row_missing_counts(found.rows) if policy == "keep_most_complete" else None
    for group in found.groups:
        if policy == "keep_first":
            removed.extend(group[1:])
        elif policy == "keep_last":
            removed.extend(group[:-1])
        elif policy == "drop_all":
            removed.extend(group)
        else:
            assert counts is not None
            best = min(group, key=lambda p: (int(counts.loc[p]), p))
            removed.extend(p for p in group if p != best)
    return np.array(sorted(removed), dtype=np.int64)


def resolve_composite_key_duplicates(
    ctx: AuthContext,
    dataset_id: int,
    key_columns: list[str],
    *,
    policy: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    """Resolve rows sharing a composite key with an explicit policy and publish a new version.

    Policies: keep_first, keep_last, keep_most_complete (fewest missing values,
    ties keep the earliest row) and drop_all. Requires ``confirm=True``.
    """
    ref = _ref(ctx, dataset_id, feature=Feature.DUPLICATE_DETECTION, permission=Permission.CLEAN_DATA)
    keys = _require_columns(ref, key_columns, "key column")
    if policy not in KEY_POLICIES:
        raise ValidationError(
            "Choose a resolution policy: keep_first, keep_last, keep_most_complete or drop_all."
        )
    if confirm is not True:
        raise ValidationError(
            "Resolving key duplicates deletes rows. Review the conflicts and confirm to continue."
        )
    found = _key_groups(ctx, ref, keys, full_rows=policy == "keep_most_complete")
    removal = _policy_removal(found, policy)
    result = _removal_result(ref, removal, None)
    result.update({"policy": policy, "key_columns": keys, "groups_resolved": len(found.groups)})
    if not len(removal):
        return result
    indices, truncated = _capped(removal.tolist(), SUMMARY_INDEX_CAP)
    parameters = {"key_columns": keys, "policy": policy}
    summary = {
        **parameters,
        "groups_resolved": len(found.groups),
        "rows_removed": int(len(removal)),
        "removed_row_indices": indices,
        "removed_row_indices_truncated": truncated,
        "groups_with_missing_key_skipped": found.excluded_groups,
    }
    published = _publish_removal(
        ctx,
        ref,
        removal,
        operation_type="resolve_key_duplicates",
        parameters=parameters,
        summary=summary,
        feature=Feature.DUPLICATE_DETECTION,
    )
    result.update(_removal_result(ref, removal, published))
    return result
