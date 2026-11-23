"""Resource-aware processing: memory/disk checks, estimates, cancellation, timeouts.

The estimates below are deliberately conservative engineering heuristics, not
measurements. ``check_memory`` compares an estimate with a fraction
(MEMORY_SAFETY_FRACTION) of the memory that is available *right now*, because
Windows, Streamlit, Excel parsing and Ollama all share the same RAM.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil
import pyarrow.parquet as pq

from config.settings import get_settings
from core.errors import OperationCancelledError, OperationTimeoutError, ResourceLimitError

MB = 1024 * 1024

# Bytes of pandas memory per byte of input, per format (conservative).
CSV_MEMORY_FACTOR = 6.0
EXCEL_MEMORY_FACTOR = 3.0  # applied to the *uncompressed* workbook size
PARQUET_MEMORY_FACTOR = 3.0  # applied to the uncompressed Parquet column size
BYTES_PER_CELL_FALLBACK = 64


def _psutil_available() -> int | None:
    try:
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - platform without support
        return None


def _psutil_total() -> int | None:
    try:
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001
        return None


def _disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return int(shutil.disk_usage(probe).free)


# Providers are module attributes so tests can simulate low memory or disk.
memory_available_provider: Callable[[], int | None] = _psutil_available
memory_total_provider: Callable[[], int | None] = _psutil_total
disk_free_provider: Callable[[Path], int] = _disk_free


def available_memory_bytes() -> int | None:
    return memory_available_provider()


def total_memory_bytes() -> int | None:
    return memory_total_provider()


def free_disk_bytes(path: Path | None = None) -> int:
    return disk_free_provider(path or get_settings().storage_root)


def human_bytes(value: float | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def memory_budget_bytes() -> int | None:
    available = available_memory_bytes()
    if available is None:
        return None
    return int(available * get_settings().memory_safety_fraction)


def check_memory(required_bytes: int, *, operation: str) -> None:
    """Raise ResourceLimitError if ``required_bytes`` exceeds the safe budget."""
    budget = memory_budget_bytes()
    if budget is None:
        return  # cannot measure on this platform; size limits still apply
    if required_bytes > budget:
        raise ResourceLimitError(
            f"{operation} needs about {human_bytes(required_bytes)} of memory, but only "
            f"{human_bytes(budget)} can be used safely right now. Close other applications, "
            "use a smaller file, or process a subset of the data.",
            details={"required_bytes": int(required_bytes), "budget_bytes": int(budget)},
        )


def check_disk(required_bytes: int, *, operation: str, path: Path | None = None) -> None:
    """Raise ResourceLimitError if writing ``required_bytes`` would breach the free-space reserve."""
    free = free_disk_bytes(path)
    reserve = get_settings().min_free_disk_mb * MB
    if free - required_bytes < reserve:
        raise ResourceLimitError(
            f"{operation} needs about {human_bytes(required_bytes)} of disk space, but only "
            f"{human_bytes(max(free - reserve, 0))} is available above the safety reserve.",
            details={"required_bytes": int(required_bytes), "free_bytes": int(free)},
        )


def estimate_csv_memory(size_bytes: int) -> int:
    return int(size_bytes * CSV_MEMORY_FACTOR)


def estimate_excel_memory(uncompressed_bytes: int) -> int:
    return int(uncompressed_bytes * EXCEL_MEMORY_FACTOR)


def estimate_cells_memory(rows: int, columns: int) -> int:
    return int(max(rows, 0) * max(columns, 0) * BYTES_PER_CELL_FALLBACK)


def estimate_parquet_memory(path: Path, columns: list[str] | None = None) -> int:
    """Estimate the in-memory size of a Parquet file (optionally a column subset)."""
    metadata = pq.read_metadata(path)  # opens and closes the file (no handle left on Windows)
    names = None if columns is None else set(columns)
    uncompressed = 0
    for rg_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg_index)
        for col_index in range(row_group.num_columns):
            column = row_group.column(col_index)
            if names is None or column.path_in_schema in names:
                uncompressed += column.total_uncompressed_size
    width = metadata.num_columns if names is None else len(names)
    return max(
        int(uncompressed * PARQUET_MEMORY_FACTOR), estimate_cells_memory(metadata.num_rows, width) // 4
    )


class CancellationToken:
    """Thread-safe flag a UI control can set to stop a long-running operation."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


ProgressCallback = Callable[[float, str], None]


@dataclass
class OperationGuard:
    """Progress reporting plus cooperative cancellation and timeout checks.

    Long-running loops call ``guard.step(fraction, message)`` between chunks;
    the guard raises OperationCancelledError / OperationTimeoutError so the
    caller's cleanup (temporary files, transaction rollback) runs.
    """

    progress: ProgressCallback | None = None
    cancel_token: CancellationToken | None = None
    timeout_seconds: float | None = None
    _started: float = 0.0

    def __post_init__(self) -> None:
        self._started = time.monotonic()
        if self.timeout_seconds is None:
            self.timeout_seconds = float(get_settings().operation_timeout_seconds)

    def check(self) -> None:
        if self.cancel_token is not None and self.cancel_token.cancelled:
            raise OperationCancelledError()
        if self.timeout_seconds and time.monotonic() - self._started > self.timeout_seconds:
            raise OperationTimeoutError()

    def remaining(self) -> float | None:
        """Seconds left before this guard times out, or None without a time limit."""
        if not self.timeout_seconds:
            return None
        return float(self.timeout_seconds) - (time.monotonic() - self._started)

    def step(self, fraction: float, message: str = "") -> None:
        self.check()
        if self.progress is not None:
            self.progress(max(0.0, min(1.0, float(fraction))), message)


def make_guard(
    progress: ProgressCallback | None = None,
    cancel_token: CancellationToken | None = None,
    timeout_seconds: float | None = None,
) -> OperationGuard:
    return OperationGuard(progress=progress, cancel_token=cancel_token, timeout_seconds=timeout_seconds)
