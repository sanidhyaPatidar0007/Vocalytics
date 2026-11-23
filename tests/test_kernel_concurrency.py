"""Versioning kernel under concurrency: atomic publishes and short write transactions."""

from __future__ import annotations

import sqlite3
import threading
import time

import pandas as pd
import pytest
from sqlalchemy import select

from config.settings import get_settings
from core import storage
from core.errors import ConflictError, PlanLimitExceededError, StorageQuotaExceededError
from database import session_scope
from database.models import Dataset, DatasetVersion, SubscriptionPlan
from services import versioning_service
from services.audit_service import record_audit


def _drop_first(df: pd.DataFrame):
    return df.iloc[1:].reset_index(drop=True), {"rows_removed": 1}, {}


def test_concurrent_operations_on_one_dataset_conflict_cleanly(factory, org_id, admin_ctx, sales_frame, monkeypatch):
    """Two overlapping operations: one publishes, the other gets ConflictError (never a raw DB error)."""
    other_ctx = factory.org_admin(org_id)
    dataset_id = factory.dataset(admin_ctx, sales_frame, "sales")
    barrier = threading.Barrier(2)
    real_publish = versioning_service.publish_version

    def publish_after_both_checked(*args, **kwargs):
        try:  # line both operations up after the "dataset changed?" check
            barrier.wait(timeout=2)
        except threading.BrokenBarrierError:
            pass
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(versioning_service, "publish_version", publish_after_both_checked)
    outcomes: dict[str, object] = {}

    def run(name, ctx):
        try:
            outcomes[name] = versioning_service.apply_frame_operation(
                ctx, dataset_id, operation_type="drop_row", parameters={"by": name},
                transform=lambda df: (df.iloc[1:].reset_index(drop=True), {"rows_removed": 1}, {}),
            )
        except Exception as exc:  # noqa: BLE001
            outcomes[name] = exc

    threads = [threading.Thread(target=run, args=(n, c)) for n, c in (("a", admin_ctx), ("b", other_ctx))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    published = [o for o in outcomes.values() if isinstance(o, versioning_service.PublishResult)]
    errors = [o for o in outcomes.values() if isinstance(o, BaseException)]
    assert len(published) == 1 and published[0].version_number == 2
    assert len(errors) == 1 and isinstance(errors[0], ConflictError), errors
    failed = [o for o in versioning_service.list_operations(admin_ctx, dataset_id) if o["status"] == "failed"]
    assert failed and "changed while this operation was running" in failed[0]["error_message"]
    versions = versioning_service.list_versions(admin_ctx, dataset_id)
    assert [v["version_number"] for v in versions if v["is_current"]] == [2]


def test_change_committed_after_the_callers_check_is_never_overwritten(admin_ctx, factory, sales_frame, monkeypatch):
    """Regression (SR2 lost update): another operation commits after the caller checked the
    current version but before the publish; the stale result must not become current."""
    dataset_id = factory.dataset(admin_ctx, sales_frame, "sales")
    real_publish = versioning_service.publish_version
    state = {"interleaved": False}

    def publish_after_someone_else(*args, **kwargs):
        if not state["interleaved"]:
            state["interleaved"] = True
            versioning_service.apply_frame_operation(  # commits v2 in its own transaction
                admin_ctx, dataset_id, operation_type="other", parameters={}, transform=_drop_first
            )
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(versioning_service, "publish_version", publish_after_someone_else)
    with pytest.raises(ConflictError, match="changed while this operation was running"):
        versioning_service.apply_frame_operation(
            admin_ctx, dataset_id, operation_type="stale", parameters={}, transform=lambda df: (df, {}, {})
        )
    versions = versioning_service.list_versions(admin_ctx, dataset_id)
    assert [(v["version_number"], v["operation_type"], v["is_current"]) for v in versions] == [
        (2, "other", True),
        (1, "import", False),
    ]


# ---------------------------------------------------------------------------
# SR1: creating a dataset keeps the write transaction short
# ---------------------------------------------------------------------------


def _write_lock_free() -> bool:
    probe = sqlite3.connect(str(get_settings().database_path), timeout=0)
    try:
        probe.execute("BEGIN IMMEDIATE")
        probe.execute("ROLLBACK")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        probe.close()


def _all_files(root) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_new_dataset_heavy_file_work_runs_outside_the_write_transaction(
    admin_ctx, sales_frame, tmp_path, monkeypatch
):
    seen = {}
    real_quality, real_sha = versioning_service.compute_quality, storage.sha256_file

    def quality(frame):
        seen["quality"] = _write_lock_free()
        return real_quality(frame)

    def sha(path):
        seen.setdefault("sha256", _write_lock_free())
        return real_sha(path)

    monkeypatch.setattr(versioning_service, "compute_quality", quality)
    monkeypatch.setattr(storage, "sha256_file", sha)
    original = tmp_path / "upload.csv"
    sales_frame.to_csv(original, index=False)
    with session_scope() as db:
        dataset, _ = versioning_service.create_dataset_with_initial_version(
            db, admin_ctx, name="Lock probe", source_type="csv",
            original_filename="upload.csv", frame=sales_frame, original_source=original,
        )
        dataset_id = dataset.id
    assert seen == {"quality": True, "sha256": True}
    assert versioning_service.verify_original(admin_ctx, dataset_id)["intact"] is True
    assert len(versioning_service.load_frame(admin_ctx, dataset_id)) == len(sales_frame)


class _Parked:
    """Hold one dataset creator inside its heavy phase (compute_quality) until released."""

    def __init__(self, monkeypatch) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.creator: threading.Thread | None = None
        real = versioning_service.compute_quality

        def parked_quality(frame):
            if threading.current_thread() is self.creator:
                self.entered.set()
                self.release.wait(timeout=30)
            return real(frame)

        monkeypatch.setattr(versioning_service, "compute_quality", parked_quality)

    def start(self, target) -> dict[str, object]:
        outcome: dict[str, object] = {}

        def run():
            try:
                outcome["result"] = target()
            except BaseException as exc:  # noqa: BLE001
                outcome["error"] = exc

        self.creator = threading.Thread(target=run)
        self.creator.start()
        assert self.entered.wait(timeout=30), "the creator never reached its heavy phase"
        return outcome

    def finish(self) -> None:
        self.release.set()
        assert self.creator is not None
        self.creator.join(timeout=60)


def _create(ctx, frame, name):
    def target():
        with session_scope() as db:
            dataset, _ = versioning_service.create_dataset_with_initial_version(
                db, ctx, name=name, source_type="derived", original_filename=name, frame=frame
            )
            return dataset.id

    return target


def test_other_tenants_write_while_a_dataset_is_being_created(factory, admin_ctx, sales_frame, monkeypatch):
    other_ctx = factory.org_admin(factory.organization("Other tenant"))
    parked = _Parked(monkeypatch)
    outcome = parked.start(_create(admin_ctx, sales_frame, "Big upload"))
    try:
        started = time.monotonic()
        with session_scope() as db:  # for example another tenant signing in
            record_audit(db, other_ctx, "test.other_tenant_write")
        elapsed = time.monotonic() - started
        still_creating = parked.creator.is_alive()
    finally:
        parked.finish()
    assert elapsed < 5, f"the other tenant's write waited {elapsed:.1f}s for the dataset creation"
    assert still_creating
    assert "error" not in outcome, outcome
    assert versioning_service.list_versions(admin_ctx, outcome["result"])[0]["version_number"] == 1


@pytest.mark.parametrize("taken", ["name", "dataset_quota"])
def test_limits_taken_during_the_heavy_phase_are_rechecked(
    factory, admin_ctx, sales_frame, app_settings, monkeypatch, taken
):
    existing = factory.dataset(admin_ctx, sales_frame, "existing")
    if taken == "dataset_quota":
        with session_scope() as db:
            db.execute(
                select(SubscriptionPlan).where(SubscriptionPlan.code == "premium")
            ).scalar_one().max_datasets = 2
    before = _all_files(app_settings.storage_root)
    parked = _Parked(monkeypatch)
    outcome = parked.start(_create(admin_ctx, sales_frame, "Report"))
    try:  # the name or the last dataset slot is taken while the creator is busy
        other = factory.dataset(admin_ctx, sales_frame.head(2), "Report" if taken == "name" else "Other")
    finally:
        parked.finish()
    expected = ConflictError if taken == "name" else PlanLimitExceededError
    assert isinstance(outcome.get("error"), expected), outcome
    with session_scope() as db:
        names = sorted(d.name for d in db.execute(select(Dataset)).scalars())
        versions = db.execute(select(DatasetVersion)).scalars().all()
    assert names == sorted(["existing", "Report" if taken == "name" else "Other"])
    assert {v.dataset_id for v in versions} == {existing, other}
    leftover = _all_files(app_settings.storage_root) - before
    assert all(f"/{other}/" in path for path in leftover), leftover


def test_storage_taken_during_the_heavy_phase_removes_the_placed_original(
    factory, admin_ctx, sales_frame, app_settings, tmp_path, monkeypatch
):
    """The storage quota is re-checked under the write lock, after the original is placed;
    the refusal must leave no new rows and neither the original nor the version file."""
    existing = factory.dataset(admin_ctx, sales_frame, "existing")
    with session_scope() as db:
        db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.code == "premium")
        ).scalar_one().max_storage_mb = 1
    original = tmp_path / "upload.csv"
    sales_frame.to_csv(original, index=False)
    before = _all_files(app_settings.storage_root)
    parked = _Parked(monkeypatch)

    def create_upload():
        with session_scope() as db:
            versioning_service.create_dataset_with_initial_version(
                db, admin_ctx, name="Upload", source_type="csv",
                original_filename="upload.csv", frame=sales_frame, original_source=original,
            )

    outcome = parked.start(create_upload)
    try:  # another upload of the organization takes the remaining storage meanwhile
        with session_scope() as db:
            db.get(Dataset, existing).original_size_bytes = 1024 * 1024
    finally:
        parked.finish()
    assert isinstance(outcome.get("error"), StorageQuotaExceededError), outcome
    assert _all_files(app_settings.storage_root) == before
    with session_scope() as db:
        assert [d.id for d in db.execute(select(Dataset)).scalars()] == [existing]
        assert {v.dataset_id for v in db.execute(select(DatasetVersion)).scalars()} == {existing}


@pytest.mark.parametrize("failure", ["version_audit", "create_audit", "caller_after_return"])
def test_failure_after_files_are_placed_leaves_no_rows_and_no_files(
    admin_ctx, sales_frame, app_settings, tmp_path, monkeypatch, failure
):
    original = tmp_path / "upload.csv"
    sales_frame.to_csv(original, index=False)
    before = _all_files(app_settings.storage_root)
    real_audit = versioning_service.record_audit
    placed: set[str] = set()

    def failing_audit(db, ctx, action, **kwargs):
        placed.update(_all_files(app_settings.storage_root) - before)
        if (failure, action) in {("version_audit", "dataset.import"), ("create_audit", "dataset.create")}:
            raise RuntimeError("injected failure")
        return real_audit(db, ctx, action, **kwargs)

    monkeypatch.setattr(versioning_service, "record_audit", failing_audit)
    with pytest.raises(RuntimeError, match="injected failure"):
        with session_scope() as db:
            versioning_service.create_dataset_with_initial_version(
                db, admin_ctx, name="Doomed", source_type="csv",
                original_filename="upload.csv", frame=sales_frame, original_source=original,
            )
            if failure == "caller_after_return":
                raise RuntimeError("injected failure")
    assert any("/originals/" in p for p in placed) and any("/versions/" in p for p in placed), placed
    assert _all_files(app_settings.storage_root) == before
    with session_scope() as db:
        assert db.execute(select(Dataset)).scalars().all() == []
        assert db.execute(select(DatasetVersion)).scalars().all() == []
