"""Cross-cutting resource management: memory, disk, size targets, timeouts, cancellation, limits.

Nothing here creates genuinely large data: the limits are lowered through
``set_settings(app_settings.with_overrides(...))`` and memory/disk readings are
faked through the resource manager's provider hooks.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
from sqlalchemy import func, select

from config.settings import set_settings
from core.errors import (
    OperationCancelledError,
    OperationTimeoutError,
    PlanLimitExceededError,
    ResourceLimitError,
    ValidationError,
)
from database import session_scope
from database.models import Dataset
from modules import subscriptions
from services import append_service, dataset_service, merge_service, resource_manager, versioning_service
from services.resource_manager import CancellationToken, OperationGuard

MB = 1024 * 1024


def _csv_bytes(target_bytes: int) -> bytes:
    """A valid CSV of roughly ``target_bytes`` bytes."""
    lines = ["id,value,label"]
    size = len(lines[0]) + 1
    i = 0
    while size < target_bytes:
        line = f"{i},{i * 1.5},label{i}"
        lines.append(line)
        size += len(line) + 1
        i += 1
    return ("\n".join(lines) + "\n").encode("utf-8")


def _files(root) -> list:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def _dataset_count() -> int:
    with session_scope() as db:
        return db.execute(select(func.count(Dataset.id))).scalar_one()


def _low_memory(monkeypatch, available: int) -> None:
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: available)


def _low_disk(monkeypatch, free: int) -> None:
    monkeypatch.setattr(resource_manager, "disk_free_provider", lambda path: free)


def _drop_nothing(batch: pd.DataFrame, _offset: int):
    return batch, {}


@pytest.fixture
def small_pair(admin_ctx, factory, sales_frame) -> tuple[int, int]:
    return factory.dataset(admin_ctx, sales_frame, "first"), factory.dataset(admin_ctx, sales_frame, "second")


def test_large_csv_above_processing_target_is_rejected_on_low_memory(admin_ctx, app_settings, monkeypatch):
    set_settings(app_settings.with_overrides(csv_processing_target_mb=1))
    data = _csv_bytes(int(1.5 * MB))
    assert 1.4 * MB < len(data) < 1.7 * MB
    _low_memory(monkeypatch, 200 * MB)  # budget 100 MB, below the chunked-import estimate
    with pytest.raises(ResourceLimitError) as excinfo:
        dataset_service.import_dataset(admin_ctx, file_name="big.csv", data=data, dataset_name="big")
    assert "memory" in excinfo.value.user_message
    assert _files(app_settings.storage_root) == []
    assert _dataset_count() == 0


def test_csv_below_processing_target_is_not_memory_checked(admin_ctx, app_settings, monkeypatch):
    set_settings(app_settings.with_overrides(csv_processing_target_mb=1))
    _low_memory(monkeypatch, 200 * MB)
    result = dataset_service.import_dataset(
        admin_ctx, file_name="small.csv", data=_csv_bytes(64 * 1024), dataset_name="small"
    )
    assert result["rows"] > 1000 and result["columns"] == 3


def test_large_csv_is_accepted_when_memory_allows(admin_ctx, app_settings):
    set_settings(app_settings.with_overrides(csv_processing_target_mb=1))
    result = dataset_service.import_dataset(
        admin_ctx, file_name="big.csv", data=_csv_bytes(int(1.5 * MB)), dataset_name="big"
    )
    assert result["rows"] > 50_000


def test_low_disk_rejects_imports(admin_ctx, app_settings, monkeypatch):
    _low_disk(monkeypatch, 10 * MB)  # below the 500 MB free-space reserve
    with pytest.raises(ResourceLimitError, match="disk space"):
        dataset_service.import_dataset(admin_ctx, file_name="a.csv", data=b"a,b\n1,2\n", dataset_name="a")
    assert _files(app_settings.storage_root) == []
    assert _dataset_count() == 0


def test_low_disk_rejects_streaming_operations(admin_ctx, small_pair, app_settings, monkeypatch):
    first, second = small_pair
    before = _files(app_settings.storage_root)
    _low_disk(monkeypatch, 10 * MB)
    with pytest.raises(ResourceLimitError, match="disk space"):
        versioning_service.apply_streaming_operation(
            admin_ctx, first, operation_type="noop", parameters={}, transform_batch=_drop_nothing
        )
    with pytest.raises(ResourceLimitError, match="disk space"):
        append_service.apply_append(admin_ctx, first, second, confirm=True)
    with pytest.raises(ResourceLimitError, match="disk space"):
        merge_service.apply_merge(
            admin_ctx, first, second, left_on=["order_id"], right_on=["order_id"], how="inner", confirm=True
        )
    assert append_service.analyze_append(admin_ctx, first, second)["resource_check"]["ok"] is False
    assert sorted(_files(app_settings.storage_root)) == sorted(before)
    assert len(versioning_service.list_versions(admin_ctx, first)) == 1


def test_load_frame_rejects_when_memory_is_insufficient(admin_ctx, factory, sales_frame, monkeypatch):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    _low_memory(monkeypatch, 1024)
    with pytest.raises(ResourceLimitError, match="memory"):
        versioning_service.load_frame(admin_ctx, dataset_id)
    _low_memory(monkeypatch, 8 * 1024**3)
    assert len(versioning_service.load_frame(admin_ctx, dataset_id)) == len(sales_frame)


def _slow_guard(timeout: float) -> OperationGuard:
    """A guard whose progress callback outlasts the timeout, so the next step times out."""

    def slow(_fraction: float, _message: str) -> None:
        time.sleep(timeout * 3)

    return OperationGuard(progress=slow, timeout_seconds=timeout)


def test_timeout_stops_streaming_operation_without_publishing(admin_ctx, small_pair, app_settings):
    first, second = small_pair
    set_settings(app_settings.with_overrides(parquet_batch_rows=2))
    with pytest.raises(OperationTimeoutError):
        versioning_service.apply_streaming_operation(
            admin_ctx,
            first,
            operation_type="noop",
            parameters={},
            transform_batch=_drop_nothing,
            guard=_slow_guard(0.05),
        )
    with pytest.raises(OperationTimeoutError):
        append_service.apply_append(admin_ctx, first, second, confirm=True, guard=_slow_guard(0.05))
    assert len(versioning_service.list_versions(admin_ctx, first)) == 1
    statuses = [op["status"] for op in versioning_service.list_operations(admin_ctx, first)]
    assert statuses == ["cancelled", "cancelled", "applied"]
    assert _files(app_settings.storage_root / "orgs" / str(admin_ctx.organization_id) / "tmp") == []
    assert list((app_settings.storage_root / "orgs" / str(admin_ctx.organization_id) / "tmp").iterdir()) == []


def test_cancellation_token_stops_streaming_operation(admin_ctx, small_pair, app_settings):
    first, second = small_pair
    set_settings(app_settings.with_overrides(parquet_batch_rows=2))
    token = CancellationToken()
    progress: list[float] = []

    def cancel_on_first_step(fraction: float, _message: str) -> None:
        progress.append(fraction)
        token.cancel()

    with pytest.raises(OperationCancelledError):
        versioning_service.apply_streaming_operation(
            admin_ctx,
            first,
            operation_type="noop",
            parameters={},
            transform_batch=_drop_nothing,
            guard=OperationGuard(progress=cancel_on_first_step, cancel_token=token),
        )
    assert progress == [0.0]
    with pytest.raises(OperationCancelledError):
        merge_service.apply_merge(
            admin_ctx,
            first,
            second,
            left_on=["order_id"],
            right_on=["order_id"],
            how="inner",
            confirm=True,
            guard=OperationGuard(cancel_token=token),
        )
    assert len(versioning_service.list_versions(admin_ctx, first)) == 1
    assert _files(app_settings.storage_root / "orgs" / str(admin_ctx.organization_id) / "tmp") == []


def test_resource_messages_are_human_readable_without_paths(admin_ctx, small_pair, app_settings, monkeypatch):
    first, second = small_pair
    messages = []
    _low_memory(monkeypatch, 1024)
    for call in (
        lambda: resource_manager.check_memory(10 * MB, operation="Loading this dataset"),
        lambda: versioning_service.load_frame(admin_ctx, first),
        lambda: append_service.apply_append(admin_ctx, first, second, confirm=True),
        lambda: merge_service.apply_merge(
            admin_ctx, first, second, left_on=["order_id"], right_on=["order_id"], how="inner", confirm=True
        ),
    ):
        with pytest.raises(ResourceLimitError) as excinfo:
            call()
        messages.append(excinfo.value.user_message)
    _low_memory(monkeypatch, 8 * 1024**3)
    _low_disk(monkeypatch, MB)
    with pytest.raises(ResourceLimitError) as excinfo:
        resource_manager.check_disk(5 * MB, operation="Saving the result", path=app_settings.storage_root)
    messages.append(excinfo.value.user_message)
    root = str(app_settings.storage_root)
    for message in messages:
        assert message and message[0].isupper() and message.endswith(".")
        assert root not in message and "\\" not in message and "/" not in message
        assert ".parquet" not in message and "Traceback" not in message
        assert any(unit in message for unit in (" KB", " MB", " GB", " B "))


def test_plan_upload_limit_is_capped_by_absolute_ceiling(factory, app_settings):
    basic = factory.organization(plan="basic")
    premium = factory.organization(plan="premium")
    with session_scope() as db:
        assert subscriptions.get_limits(db, basic).max_upload_mb == app_settings.basic_max_upload_mb == 50
        assert (
            subscriptions.get_limits(db, premium).max_upload_mb == app_settings.premium_max_upload_mb == 500
        )
    set_settings(app_settings.with_overrides(absolute_upload_ceiling_mb=100))
    with session_scope() as db:
        assert subscriptions.get_limits(db, basic).max_upload_mb == 50  # plan is smaller
        assert subscriptions.get_limits(db, premium).max_upload_mb == 100  # ceiling is smaller
    set_settings(app_settings.with_overrides(absolute_upload_ceiling_mb=1))
    ctx = factory.org_admin(premium)
    with pytest.raises(PlanLimitExceededError, match="up to 1 MB"):
        dataset_service.import_dataset(
            ctx, file_name="a.csv", data=_csv_bytes(int(1.5 * MB)), dataset_name="a"
        )
    result = dataset_service.import_dataset(
        ctx, file_name="b.csv", data=_csv_bytes(512 * 1024), dataset_name="b"
    )
    assert result["rows"] > 0


# --- Regression SEC3: a column-count limit on import and merge, and interruptible typing ---


def _wide_csv(n: int, rows: int = 1) -> bytes:
    header = ",".join(f"c{i}" for i in range(n))
    body = "".join(",".join(str(i + r) for i in range(n)) + "\n" for r in range(rows))
    return f"{header}\n{body}".encode()


def test_too_many_columns_rejected_at_inspect_and_import(app_settings, admin_ctx):
    set_settings(app_settings.with_overrides(max_columns=50))
    data = _wide_csv(51)
    with pytest.raises(ValidationError, match="51 columns.*limit of 50"):
        dataset_service.inspect_upload(admin_ctx, file_name="wide.csv", data=data)
    with pytest.raises(ValidationError, match="51 columns.*limit of 50"):
        dataset_service.import_dataset(admin_ctx, file_name="wide.csv", data=data)
    assert _dataset_count() == 0
    # Exactly at the limit is still accepted.
    assert dataset_service.import_dataset(admin_ctx, file_name="ok.csv", data=_wide_csv(50))["dataset_id"]


def test_merge_wider_than_the_column_limit_is_refused(admin_ctx, small_pair, app_settings, sales_frame):
    first, second = small_pair
    width = len(sales_frame.columns)
    set_settings(app_settings.with_overrides(max_columns=width))
    with pytest.raises(ValidationError, match=f"{2 * width - 1} columns.*limit of {width}"):
        merge_service.analyze_merge(
            admin_ctx, first, second, left_on=["order_id"], right_on=["order_id"], how="inner"
        )
    with pytest.raises(ValidationError, match="limit"):
        merge_service.apply_merge(
            admin_ctx, first, second, left_on=["order_id"], right_on=["order_id"], how="inner", confirm=True
        )
    assert _dataset_count() == 2


def test_cancelling_during_type_detection_stops_before_every_column_is_typed(admin_ctx, monkeypatch):
    from services import type_inference

    calls = {"update": 0}
    original = type_inference.ColumnTypeAccumulator.update

    def counting(self, raw):
        calls["update"] += 1
        return original(self, raw)

    monkeypatch.setattr(type_inference.ColumnTypeAccumulator, "update", counting)
    token = CancellationToken()

    def progress(_fraction: float, message: str) -> None:
        if message == "Detecting column types":
            token.cancel()

    with pytest.raises(OperationCancelledError):
        dataset_service.import_dataset(
            admin_ctx, file_name="w.csv", data=_wide_csv(40), progress=progress, cancel_token=token
        )
    assert calls["update"] < 40
    assert _dataset_count() == 0
