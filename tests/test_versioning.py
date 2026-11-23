"""Dataset versioning: recoverable versions, atomic publishing, retention, integrity."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sqlalchemy import select

from core import storage
from core.errors import (
    ConflictError,
    NotFoundError,
    OperationCancelledError,
    PermissionDeniedError,
    StorageQuotaExceededError,
)
from database import session_scope
from database.models import CleaningOperation, Dataset, DatasetVersion, SubscriptionPlan
from services import versioning_service
from services.resource_manager import CancellationToken, OperationGuard


def _files_under(root) -> set[str]:
    return {p.name for p in root.rglob("*.parquet")}


def _drop_first(df: pd.DataFrame):
    return df.iloc[1:].reset_index(drop=True), {"rows_removed": 1}, {"note": "dropped"}


def test_publish_and_restore_round_trip(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame, "sales")
    v1 = versioning_service.load_frame(admin_ctx, dataset_id)
    result = versioning_service.apply_frame_operation(
        admin_ctx, dataset_id, operation_type="drop_row", parameters={"row": 0}, transform=_drop_first
    )
    assert result.version_number == 2 and result.rows == len(sales_frame) - 1
    restored = versioning_service.restore_version(
        admin_ctx, dataset_id, versioning_service.list_versions(admin_ctx, dataset_id)[-1]["version_id"]
    )
    assert restored.version_number == 3
    pd.testing.assert_frame_equal(versioning_service.load_frame(admin_ctx, dataset_id), v1)
    history = versioning_service.list_operations(admin_ctx, dataset_id)
    assert [h["operation_type"] for h in history] == ["restore", "drop_row", "import"]
    assert history[0]["parameters"] == {"restored_from_version": 1}
    assert history[1]["rows_removed"] == 1 and history[1]["quality_before"] is not None


def test_old_versions_remain_loadable(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    first_id = versioning_service.list_versions(admin_ctx, dataset_id)[0]["version_id"]
    versioning_service.apply_frame_operation(
        admin_ctx, dataset_id, operation_type="drop", parameters={}, transform=_drop_first
    )
    old = versioning_service.load_frame(admin_ctx, dataset_id, version_id=first_id)
    assert len(old) == len(sales_frame)


def test_failed_transform_publishes_nothing_and_is_recorded(admin_ctx, factory, sales_frame, app_settings):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    before = _files_under(app_settings.storage_root)

    def broken(_df):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="broken", parameters={"x": 1}, transform=broken
        )
    assert _files_under(app_settings.storage_root) == before
    versions = versioning_service.list_versions(admin_ctx, dataset_id)
    assert len(versions) == 1 and versions[0]["is_current"]
    failed = versioning_service.list_operations(admin_ctx, dataset_id)[0]
    assert failed["status"] == "failed" and failed["operation_type"] == "broken"


def test_database_failure_after_file_write_rolls_back_the_file(
    admin_ctx, factory, sales_frame, app_settings, monkeypatch
):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    before = _files_under(app_settings.storage_root)

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(versioning_service, "record_audit", fail_audit)
    with pytest.raises(RuntimeError):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="drop", parameters={}, transform=_drop_first
        )
    assert _files_under(app_settings.storage_root) == before
    with session_scope() as db:
        assert (
            db.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id))
            .scalars()
            .all()
            .__len__()
            == 1
        )


def test_rollback_after_publish_deletes_the_new_file(admin_ctx, factory, sales_frame, app_settings):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    before = _files_under(app_settings.storage_root)
    with pytest.raises(RuntimeError):
        with session_scope() as db:
            dataset = db.get(Dataset, dataset_id)
            versioning_service.publish_version(
                db, admin_ctx, dataset, operation_type="x", frame=sales_frame.head(2)
            )
            assert len(_files_under(app_settings.storage_root)) == len(before) + 1
            raise RuntimeError("caller failed after publishing")
    assert _files_under(app_settings.storage_root) == before


def test_retention_purges_files_only_after_commit(factory, sales_frame, app_settings):
    org = factory.organization(plan="basic")
    with session_scope() as db:
        db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.code == "basic")
        ).scalar_one().max_versions_per_dataset = 3
    ctx = factory.org_admin(org)
    dataset_id = factory.dataset(ctx, sales_frame)
    for _ in range(4):
        versioning_service.apply_frame_operation(
            ctx, dataset_id, operation_type="drop", parameters={}, transform=_drop_first
        )
    versions = versioning_service.list_versions(ctx, dataset_id)
    live = [v for v in versions if not v["is_purged"]]
    assert len(live) == 3
    assert any(v["is_original_import"] and not v["is_purged"] for v in versions)  # v1 is pinned
    with session_scope() as db:
        purged = db.execute(select(DatasetVersion).where(DatasetVersion.is_purged.is_(True))).scalars().all()
        paths = [storage.resolve_relative(v.file_path) for v in purged]
    assert paths and not any(p.exists() for p in paths)
    with pytest.raises(NotFoundError, match="retention"):
        versioning_service.restore_version(
            ctx, dataset_id, [v for v in versions if v["is_purged"]][0]["version_id"]
        )


def test_storage_quota_blocks_new_versions(admin_ctx, factory, sales_frame, org_id):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    with session_scope() as db:
        db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.code == "premium")
        ).scalar_one().max_storage_mb = 1
    import os

    big = pd.DataFrame({"blob": [os.urandom(600).hex() for _ in range(3000)]})  # ~3.6 MB, incompressible
    with pytest.raises(StorageQuotaExceededError):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="grow", parameters={}, transform=lambda df: (big, {}, {})
        )


def test_tampered_version_file_cannot_be_restored(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    versioning_service.apply_frame_operation(
        admin_ctx, dataset_id, operation_type="drop", parameters={}, transform=_drop_first
    )
    first = versioning_service.list_versions(admin_ctx, dataset_id)[-1]
    with session_scope() as db:
        path = storage.resolve_relative(db.get(DatasetVersion, first["version_id"]).file_path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ConflictError, match="integrity"):
        versioning_service.restore_version(admin_ctx, dataset_id, first["version_id"])


def test_concurrent_change_is_detected(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)

    def sneaky(df):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="other", parameters={}, transform=_drop_first
        )
        return df, {}, {}

    with pytest.raises(ConflictError):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="slow", parameters={}, transform=sneaky
        )


def test_streaming_operation_matches_in_memory_result(admin_ctx, factory, app_settings):
    frame = pd.DataFrame(
        {
            "n": pd.array(range(10), dtype="Int64"),
            "label": pd.array([f" v{i} " for i in range(10)], dtype="string"),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    from config.settings import set_settings

    set_settings(app_settings.with_overrides(parquet_batch_rows=3))

    def strip(batch, offset):
        out = batch.copy()
        out["label"] = out["label"].str.strip()
        return out, {"cells_changed": int((batch["label"] != out["label"]).sum())}

    result = versioning_service.apply_streaming_operation(
        admin_ctx, dataset_id, operation_type="trim", parameters={}, transform_batch=strip
    )
    loaded = versioning_service.load_frame(admin_ctx, dataset_id)
    assert loaded["label"].tolist() == [f"v{i}" for i in range(10)]
    assert versioning_service.list_operations(admin_ctx, dataset_id)[0]["cells_changed"] == 10
    assert result.rows == 10


def test_cancelled_streaming_operation_leaves_no_files(admin_ctx, factory, app_settings):
    frame = pd.DataFrame({"n": pd.array(range(10), dtype="Int64")})
    dataset_id = factory.dataset(admin_ctx, frame)
    from config.settings import set_settings

    set_settings(app_settings.with_overrides(parquet_batch_rows=2))
    token = CancellationToken()
    before = _files_under(app_settings.storage_root)

    def cancel_midway(batch, offset):
        if offset >= 4:
            token.cancel()
        return batch, {}

    with pytest.raises(OperationCancelledError):
        versioning_service.apply_streaming_operation(
            admin_ctx,
            dataset_id,
            operation_type="slow",
            parameters={},
            transform_batch=cancel_midway,
            guard=OperationGuard(cancel_token=token),
        )
    assert _files_under(app_settings.storage_root) == before
    assert not any((app_settings.storage_root / "orgs").rglob("*.partial"))
    assert versioning_service.list_operations(admin_ctx, dataset_id)[0]["status"] == "cancelled"


def test_analyst_cannot_restore_without_permission(factory, org_id, admin_ctx, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    versioning_service.apply_frame_operation(
        admin_ctx, dataset_id, operation_type="drop", parameters={}, transform=_drop_first
    )
    analyst_id = factory.user(org_id, "analyst")
    factory.grant(dataset_id, analyst_id)
    analyst = factory.ctx(analyst_id)
    first = versioning_service.list_versions(analyst, dataset_id)[-1]["version_id"]
    with pytest.raises(PermissionDeniedError):
        versioning_service.restore_version(analyst, dataset_id, first)


def test_version_schema_records_logical_types(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    with session_scope() as db:
        version = db.get(DatasetVersion, db.get(Dataset, dataset_id).current_version_id)
        schema = {c["name"]: c["logical_type"] for c in json.loads(version.schema_json)}
        operations = db.execute(select(CleaningOperation)).scalars().all()
    assert (
        schema["revenue"] == "float"
        and schema["order_id"] == "integer"
        and schema["order_date"] == "datetime"
    )
    assert [op.operation_type for op in operations] == ["import"]


def test_derived_dataset_records_its_real_operation_once(admin_ctx, sales_frame):
    """Regression: a dataset created by an operation (e.g. merge) is recorded as that
    operation, counted once as a cleaning operation and never as a 0-byte upload."""
    from database.models import AuditLog, UsageRecord

    with session_scope() as db:
        dataset, result = versioning_service.create_dataset_with_initial_version(
            db,
            admin_ctx,
            name="merged",
            source_type="derived",
            original_filename="merge of a and b",
            frame=sales_frame,
            operation_type="merge",
            operation_parameters={"how": "inner"},
            operation_summary={"unmatched_left": 2},
        )
        dataset_id = dataset.id
    history = versioning_service.list_operations(admin_ctx, dataset_id)
    assert [(h["operation_type"], h["parameters"], h["summary"]) for h in history] == [
        ("merge", {"how": "inner"}, {"unmatched_left": 2})
    ]
    assert versioning_service.list_versions(admin_ctx, dataset_id)[0]["operation_type"] == "merge"
    with session_scope() as db:
        metrics = [u.metric for u in db.execute(select(UsageRecord)).scalars()]
        actions = [a.action for a in db.execute(select(AuditLog)).scalars()]
    assert metrics.count("cleaning_operation") == 1 and "upload_bytes" not in metrics
    assert actions.count("dataset.merge") == 1 and "dataset.import" not in actions


def test_failed_scoring_of_new_version_deletes_its_file(admin_ctx, factory, app_settings, monkeypatch):
    """Regression (SR4): the batch reader is closed before cleanup, so Windows can delete the file."""
    from config.settings import set_settings
    from services import quality_score

    frame = pd.DataFrame({"n": pd.array(range(30), dtype="Int64")})
    dataset_id = factory.dataset(admin_ctx, frame)
    v1 = versioning_service.list_versions(admin_ctx, dataset_id)[0]["version_id"]
    versioning_service.apply_frame_operation(
        admin_ctx, dataset_id, operation_type="drop", parameters={},
        transform=lambda df: (df.iloc[1:].reset_index(drop=True), {}, {}),
    )
    set_settings(app_settings.with_overrides(parquet_batch_rows=5))
    before = {p.relative_to(app_settings.storage_root) for p in app_settings.storage_root.rglob("*") if p.is_file()}
    original = quality_score.QualityAccumulator.update
    calls = {"n": 0}

    def fail_on_second_batch(self, batch):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("scoring failed")
        return original(self, batch)

    monkeypatch.setattr(quality_score.QualityAccumulator, "update", fail_on_second_batch)
    with pytest.raises(RuntimeError, match="scoring failed"):
        versioning_service.restore_version(admin_ctx, dataset_id, v1)
    after = {p.relative_to(app_settings.storage_root) for p in app_settings.storage_root.rglob("*") if p.is_file()}
    assert after == before
    with session_scope() as db:
        rows = db.execute(select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id)).scalars().all()
        assert len(rows) == 2


def _limit_versions(factory, sales_frame, limit):
    org = factory.organization(plan="basic")
    with session_scope() as db:
        db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.code == "basic")
        ).scalar_one().max_versions_per_dataset = limit
    ctx = factory.org_admin(org)
    dataset_id = factory.dataset(ctx, sales_frame)

    def op():
        return versioning_service.apply_frame_operation(
            ctx, dataset_id, operation_type="drop", parameters={}, transform=_drop_first
        )

    return org, ctx, dataset_id, op


def _purge_blocked_for(path, monkeypatch):
    """Simulate a reader holding ``path`` open on Windows: deleting it fails."""
    real_remove = storage.remove_file_quietly
    monkeypatch.setattr(
        storage, "remove_file_quietly",
        lambda target, *a, **k: False if target == path else real_remove(target, *a, **k),
    )
    return real_remove


def test_purged_file_locked_at_commit_is_deleted_later(factory, sales_frame, monkeypatch):
    """Regression (SR3): a purged file that could not be deleted is retried and counted meanwhile."""
    from modules import subscriptions

    org, _ctx, _dataset_id, op = _limit_versions(factory, sales_frame, 2)
    v2 = op().version_id
    with session_scope() as db:
        v2_row = db.get(DatasetVersion, v2)
        v2_path, v2_size = storage.resolve_relative(v2_row.file_path), v2_row.size_bytes
    real_remove = _purge_blocked_for(v2_path, monkeypatch)
    op()
    op()
    with session_scope() as db:
        assert db.get(DatasetVersion, v2).is_purged
        live_bytes = sum(
            v.size_bytes
            for v in db.execute(select(DatasetVersion).where(DatasetVersion.organization_id == org)).scalars()
            if not v.is_purged
        )
        assert subscriptions.org_storage_bytes(db, org) == live_bytes + v2_size  # still on disk: counted
    assert v2_path.exists()
    monkeypatch.setattr(storage, "remove_file_quietly", real_remove)  # the reader finished
    op()
    assert not v2_path.exists()
    with session_scope() as db:
        assert subscriptions.org_storage_bytes(db, org) == sum(
            v.size_bytes
            for v in db.execute(select(DatasetVersion).where(DatasetVersion.organization_id == org)).scalars()
            if not v.is_purged
        )


def test_worker_reclaims_purged_files_left_behind(factory, sales_frame, monkeypatch):
    _org, _ctx, _dataset_id, op = _limit_versions(factory, sales_frame, 2)
    v2 = op().version_id
    with session_scope() as db:
        v2_path = storage.resolve_relative(db.get(DatasetVersion, v2).file_path)
    real_remove = _purge_blocked_for(v2_path, monkeypatch)
    op()
    op()
    assert v2_path.exists()
    monkeypatch.setattr(storage, "remove_file_quietly", real_remove)
    result = versioning_service.reclaim_purged_files()
    assert result["still_locked"] == 0 and result["reclaimed"] >= 1
    assert not v2_path.exists()
    assert versioning_service.reclaim_purged_files() == {"reclaimed": 0, "still_locked": 0}


def test_scheduled_worker_reclaims_purged_files_without_changing_its_output(
    factory, sales_frame, monkeypatch, capsys
):
    from scripts import run_anomaly_checks

    _org, _ctx, _dataset_id, op = _limit_versions(factory, sales_frame, 2)
    v2 = op().version_id
    with session_scope() as db:
        v2_path = storage.resolve_relative(db.get(DatasetVersion, v2).file_path)
    real_remove = _purge_blocked_for(v2_path, monkeypatch)
    op()
    op()
    monkeypatch.setattr(storage, "remove_file_quietly", real_remove)
    monkeypatch.setattr(run_anomaly_checks, "configure_logging", lambda settings: None)
    assert run_anomaly_checks.main([]) == 0
    assert not v2_path.exists()
    assert capsys.readouterr().out.strip().count("\n") == 0

    def broken():
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(versioning_service, "reclaim_purged_files", broken)
    assert run_anomaly_checks.main([]) == 0  # storage upkeep never fails the worker


def test_retention_never_keeps_more_live_versions_than_the_plan_allows(factory, sales_frame):
    """Regression (SR7): at the minimum limit of 2 the outgoing current version is purged too."""
    _org, ctx, dataset_id, op = _limit_versions(factory, sales_frame, 2)
    for _ in range(3):
        op()
        live = [v for v in versioning_service.list_versions(ctx, dataset_id) if not v["is_purged"]]
        assert len(live) <= 2, [v["version_number"] for v in live]
    assert [v["version_number"] for v in live] == [4, 1]  # the current version and the pinned original
    assert len(versioning_service.load_frame(ctx, dataset_id)) == len(sales_frame) - 3


def test_in_memory_operation_respects_the_operation_timeout(admin_ctx, factory, sales_frame, app_settings):
    """Regression (SR5): apply_frame_operation checks the timeout after the transform."""
    import time as _time

    from config.settings import set_settings
    from core.errors import OperationTimeoutError

    dataset_id = factory.dataset(admin_ctx, sales_frame)
    set_settings(app_settings.with_overrides(operation_timeout_seconds=1))

    def slow(df):
        _time.sleep(1.3)
        return _drop_first(df)

    with pytest.raises(OperationTimeoutError):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="slow", parameters={}, transform=slow
        )
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1
    assert versioning_service.list_operations(admin_ctx, dataset_id)[0]["status"] == "cancelled"


def test_in_memory_operation_can_be_cancelled_with_a_guard(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    token = CancellationToken()
    progress: list[float] = []

    def cancel_during(df):
        token.cancel()
        return _drop_first(df)

    with pytest.raises(OperationCancelledError):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="drop", parameters={}, transform=cancel_during,
            guard=OperationGuard(progress=lambda f, _m: progress.append(f), cancel_token=token),
        )
    assert progress and len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1


def test_cancel_during_quality_scoring_of_a_streamed_version_publishes_nothing(
    admin_ctx, factory, app_settings, monkeypatch
):
    """The quality pass over a new version checks the guard once per batch."""
    from config.settings import set_settings
    from services import quality_score

    frame = pd.DataFrame({"n": pd.array(range(40), dtype="Int64")})
    dataset_id = factory.dataset(admin_ctx, frame)
    set_settings(app_settings.with_overrides(parquet_batch_rows=10))
    before = {p.name for p in app_settings.storage_root.rglob("*") if p.is_file()}
    token = CancellationToken()
    original = quality_score.QualityAccumulator.update

    def cancel_while_scoring(self, batch):
        token.cancel()  # the user presses Cancel while the new version is being scored
        return original(self, batch)

    monkeypatch.setattr(quality_score.QualityAccumulator, "update", cancel_while_scoring)
    with pytest.raises(OperationCancelledError):
        versioning_service.apply_streaming_operation(
            admin_ctx, dataset_id, operation_type="noop", parameters={},
            transform_batch=lambda batch, _offset: (batch, {}),
            guard=OperationGuard(cancel_token=token),
        )
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1
    assert {p.name for p in app_settings.storage_root.rglob("*") if p.is_file()} == before
