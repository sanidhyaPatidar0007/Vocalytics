"""Duplicate detection, removal, composite keys, repeated values and identifier generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import set_settings
from core.constants import Permission
from core.errors import (
    FeatureNotAvailableError,
    NotFoundError,
    OperationCancelledError,
    PermissionDeniedError,
    ValidationError,
)
from services import duplicate_manager as dm
from services import versioning_service
from services.resource_manager import CancellationToken, OperationGuard


def _rows_frame() -> pd.DataFrame:
    """Rows 0/2/5 are identical, rows 1/4 are identical, row 3 is unique."""
    return pd.DataFrame(
        {
            "customer_id": pd.array([1, 2, 1, 3, 2, 1], dtype="Int64"),
            "name": pd.array(["Ann", "Bob", "Ann", "Cy", "Bob", "Ann"], dtype="string"),
            "amount": [10.0, 20.0, 10.0, 30.0, 20.0, 10.0],
        }
    )


def _chunks(frame: pd.DataFrame, size: int):
    return lambda: (frame.iloc[i : i + size] for i in range(0, len(frame), size))


# ---------------------------------------------------------------------------
# Cross-chunk algorithm (no database)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keep", "removed"),
    [("first", [2, 4, 5]), ("last", [0, 1, 2]), ("none", [0, 1, 2, 4, 5])],
)
def test_chunk_algorithm_keep_options(keep, removed):
    result = dm.find_duplicates_in_chunks(_chunks(_rows_frame(), 2), keep=keep)
    assert result["duplicate_row_indices"] == removed
    assert result["groups"] == [[0, 2, 5], [1, 4]]
    assert result["group_count"] == 2 and result["rows_scanned"] == 6
    assert result["duplicate_rows"] == len(removed)
    assert not result["spilled_to_disk"]


def test_duplicates_across_csv_chunks_are_found(tmp_path):
    path = tmp_path / "orders.csv"
    path.write_text(
        "order,customer,total\n"
        "A1,ann,10.50\n"
        "B2,bob,20\n"
        "C3,cy,30\n"
        "A1,ann,10.50\n"  # duplicate of row 0, in the second chunk
        "D4,dee,40\n"
        "B2,bob,20\n",  # duplicate of row 1, in the third chunk
        encoding="utf-8",
    )

    def source():
        return pd.read_csv(path, chunksize=2, dtype=str)

    # Each chunk on its own has no duplicates: a per-chunk check would miss them.
    assert all(not chunk.duplicated().any() for chunk in source())
    result = dm.find_duplicates_in_chunks(source)
    assert result["groups"] == [[0, 3], [1, 5]]
    assert result["duplicate_row_indices"] == [3, 5]
    subset = dm.find_duplicates_in_chunks(source, subset=["customer"], keep="none")
    assert subset["duplicate_row_indices"] == [0, 1, 3, 5]


def test_hash_collisions_are_verified_by_value(monkeypatch):
    monkeypatch.setattr(dm, "hash_rows", lambda df: np.zeros(len(df), dtype=np.uint64))
    distinct = pd.DataFrame({"a": ["x", "y", "z", "w"]})
    result = dm.find_duplicates_in_chunks(_chunks(distinct, 2))
    assert result["groups"] == [] and result["duplicate_rows"] == 0
    assert result["hash_collisions_rejected"] == 4
    with_dup = pd.DataFrame({"a": ["x", "y", "x", "w"]})
    result = dm.find_duplicates_in_chunks(_chunks(with_dup, 2))
    assert result["groups"] == [[0, 2]] and result["hash_collisions_rejected"] == 2


def test_spill_to_disk_gives_identical_results(tmp_path):
    spill = tmp_path / "spill"
    spill.mkdir()
    frame = _rows_frame()
    in_memory = dm.find_duplicates_in_chunks(_chunks(frame, 2), keep="last")
    on_disk = dm.find_duplicates_in_chunks(
        _chunks(frame, 2), keep="last", spill_dir=spill, spill_threshold_rows=2
    )
    assert on_disk["spilled_to_disk"] and "disk-backed" in on_disk["method"]
    for key in ("duplicate_row_indices", "groups", "duplicate_rows", "group_count", "rows_scanned"):
        assert on_disk[key] == in_memory[key]
    assert list(spill.iterdir()) == []  # spill file cleaned up


def test_spill_without_directory_uses_managed_scratch_space(app_settings):
    result = dm.find_duplicates_in_chunks(_chunks(_rows_frame(), 1), spill_threshold_rows=1)
    assert result["spilled_to_disk"] and result["groups"] == [[0, 2, 5], [1, 4]]
    assert not any((app_settings.storage_root / "tmp").rglob("*.u64"))


def test_missing_values_compare_equal_and_empty_source():
    frame = pd.DataFrame({"a": pd.array([None, "x", None], dtype="string"), "b": [np.nan, 1.0, np.nan]})
    assert dm.find_duplicates_in_chunks(_chunks(frame, 1))["groups"] == [[0, 2]]
    empty = dm.find_duplicates_in_chunks(lambda: iter([pd.DataFrame({"a": []})]))
    assert empty["groups"] == [] and empty["rows_scanned"] == 0


def test_plain_list_chunk_source_is_supported():
    frame = _rows_frame()
    result = dm.find_duplicates_in_chunks(lambda: [frame.iloc[:3], frame.iloc[3:]])
    assert result["groups"] == [[0, 2, 5], [1, 4]]


def test_chunk_algorithm_validation():
    with pytest.raises(ValidationError):
        dm.find_duplicates_in_chunks(_chunks(_rows_frame(), 2), keep="middle")
    with pytest.raises(ValidationError):
        dm.find_duplicates_in_chunks(_chunks(_rows_frame(), 2), subset=["nope"])
    with pytest.raises(ValidationError):
        dm.find_duplicates_in_chunks(_chunks(_rows_frame(), 2), spill_threshold_rows=0)


# ---------------------------------------------------------------------------
# Exact duplicates on datasets
# ---------------------------------------------------------------------------


@pytest.fixture
def small_batches(app_settings):
    set_settings(app_settings.with_overrides(parquet_batch_rows=2))


def test_find_exact_duplicates_across_batches(admin_ctx, factory, small_batches):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    result = dm.find_exact_duplicates(admin_ctx, dataset_id, max_groups=1)
    assert result["duplicate_rows"] == 3 and result["group_count"] == 2 and result["rows_in_groups"] == 5
    assert result["pct"] == 50.0
    assert result["groups"] == [
        {"rows": [0, 2, 5], "size": 3, "values": {"customer_id": 1, "name": "Ann", "amount": 10.0}}
    ]
    assert result["groups_truncated"] and "verified" in result["method"]
    with pytest.raises(ValidationError):
        dm.find_exact_duplicates(admin_ctx, dataset_id, max_groups=0)


def test_find_exact_duplicates_honours_cancellation(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelledError):
        dm.find_exact_duplicates(admin_ctx, dataset_id, guard=OperationGuard(cancel_token=token))


def test_preview_does_not_change_data(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    preview = dm.preview_remove_duplicates(admin_ctx, dataset_id, keep="first")
    assert preview["rows_before"] == 6 and preview["rows_after"] == 3 and preview["rows_removed"] == 3
    assert [s["row_index"] for s in preview["sample_removed"]] == [2, 4, 5]
    assert preview["sample_removed"][1]["values"]["name"] == "Bob"
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1


@pytest.mark.parametrize(
    ("keep", "names"),
    [("first", ["Ann", "Bob", "Cy"]), ("last", ["Cy", "Bob", "Ann"]), ("none", ["Cy"])],
)
def test_remove_duplicates_keep_options_preserve_order(admin_ctx, factory, small_batches, keep, names):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    original = versioning_service.list_versions(admin_ctx, dataset_id)[0]["version_id"]
    result = dm.remove_duplicates(admin_ctx, dataset_id, keep=keep, confirm=True)
    assert result["changed"] and result["version_number"] == 2
    frame = versioning_service.load_frame(admin_ctx, dataset_id)
    assert frame["name"].tolist() == names
    assert str(frame["customer_id"].dtype) == "Int64"
    assert len(versioning_service.load_frame(admin_ctx, dataset_id, version_id=original)) == 6
    operation = versioning_service.list_operations(admin_ctx, dataset_id)[0]
    assert operation["operation_type"] == "remove_duplicates" and operation["rows_removed"] == 6 - len(names)
    assert operation["summary"]["removed_row_indices"] == result["removed_row_indices"]


def test_remove_duplicates_on_subset_and_confirmation(admin_ctx, factory):
    frame = _rows_frame()
    frame.loc[5, "amount"] = 99.0
    dataset_id = factory.dataset(admin_ctx, frame)
    with pytest.raises(ValidationError, match="confirm"):
        dm.remove_duplicates(admin_ctx, dataset_id)
    assert dm.preview_remove_duplicates(admin_ctx, dataset_id)["rows_removed"] == 2
    result = dm.remove_duplicates(admin_ctx, dataset_id, subset=["customer_id"], confirm=True)
    assert result["rows_removed"] == 3 and result["rows_after"] == 3
    with pytest.raises(ValidationError):
        dm.remove_duplicates(admin_ctx, dataset_id, subset=["missing"], confirm=True)


def test_remove_duplicates_without_duplicates_publishes_nothing(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    result = dm.remove_duplicates(admin_ctx, dataset_id, confirm=True)
    assert not result["changed"] and result["rows_removed"] == 0
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1


# ---------------------------------------------------------------------------
# Repeated values and identifier validation
# ---------------------------------------------------------------------------


def _contacts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "city": pd.array(["Pune", " pune", "PUNE ", "Delhi", None], dtype="string"),
            "email": pd.array(
                [
                    "John.Smith+promo@gmail.com",
                    "johnsmith@gmail.com",
                    " JOHN.SMITH@example.com",
                    "john.smith@example.com",
                    "a@b.com",
                ],
                dtype="string",
            ),
            "phone": pd.array(
                ["+91 98765-43210", "+91 (98765) 43210", "98765 43210", "", "12"], dtype="string"
            ),
            "code": pd.array(["ab-12", "AB 12", "ab12", "cd-1", None], dtype="string"),
        }
    )


def test_value_duplicates_with_normalization_keep_stored_values(admin_ctx, factory):
    frame = _contacts()
    dataset_id = factory.dataset(admin_ctx, frame)
    result = dm.find_value_duplicates(
        admin_ctx,
        dataset_id,
        ["city", "email", "phone", "code"],
        normalization={"email": "email", "phone": "phone", "code": "identifier"},
    )
    by_column = {c["column"]: c for c in result["columns"]}
    city = by_column["city"]
    assert city["values"][0] == {
        "value": "pune",
        "count": 3,
        "example_rows": [0, 1, 2],
        "original_values": ["Pune", " pune", "PUNE "],
    }
    assert city["missing"] == 1 and city["rows_with_repeated_values"] == 3
    emails = {v["value"]: v["count"] for v in by_column["email"]["values"]}
    assert emails == {"johnsmith@gmail.com": 2, "john.smith@example.com": 2}
    phones = {v["value"]: v["example_rows"] for v in by_column["phone"]["values"]}
    assert phones == {"+919876543210": [0, 1]}
    assert by_column["phone"]["missing"] == 1  # the blank phone is missing, not a value
    assert by_column["code"]["values"][0]["value"] == "AB12" and by_column["code"]["values"][0]["count"] == 3
    assert "not the same as duplicate records" in result["note"] and "gmail.com" in result["note"]
    stored = versioning_service.load_frame(admin_ctx, dataset_id)
    pd.testing.assert_frame_equal(stored, frame)
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1


def test_value_duplicates_trim_and_case_options(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _contacts())
    strict = dm.find_value_duplicates(admin_ctx, dataset_id, ["city"], trim=False, case_insensitive=False)
    assert strict["columns"][0]["repeated_value_count"] == 0
    trimmed = dm.find_value_duplicates(admin_ctx, dataset_id, ["city"], trim=True, case_insensitive=False)
    assert trimmed["columns"][0]["repeated_value_count"] == 0
    folded = dm.find_value_duplicates(
        admin_ctx, dataset_id, ["city"], trim=True, case_insensitive=True, top=1
    )
    assert folded["columns"][0]["repeated_value_count"] == 1


@pytest.mark.parametrize(
    ("columns", "kwargs"),
    [
        ([], {}),
        (["nope"], {}),
        (["city"], {"normalization": {"city": "soundex"}}),
        (["city"], {"normalization": {"email": "email"}}),
        (["city"], {"top": 0}),
    ],
)
def test_value_duplicates_validation(admin_ctx, factory, columns, kwargs):
    dataset_id = factory.dataset(admin_ctx, _contacts())
    with pytest.raises(ValidationError):
        dm.find_value_duplicates(admin_ctx, dataset_id, columns, **kwargs)


def test_validate_identifier(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "invoice_id": pd.array(["INV-1", "INV-2", "inv-2", "INV-3", None, "INV-1"], dtype="string"),
            "region": pd.array(["N", "S", "N", "S", "N", "S"], dtype="string"),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    result = dm.validate_identifier(admin_ctx, dataset_id, "invoice_id")
    assert not result["is_unique"] and not result["valid_identifier"]
    assert (
        result["missing_count"] == 1 and result["duplicate_value_count"] == 1 and result["rows_involved"] == 2
    )
    assert result["examples"][0]["example_rows"] == [0, 5]
    assert result["should_be_unique"] and result["looks_like_identifier"]
    folded = dm.validate_identifier(admin_ctx, dataset_id, "invoice_id", case_insensitive=True)
    assert folded["duplicate_value_count"] == 2 and folded["rows_involved"] == 4
    region = dm.validate_identifier(admin_ctx, dataset_id, "region")
    assert not region["should_be_unique"] and "legitimate" in region["explanation"]
    with pytest.raises(ValidationError):
        dm.validate_identifier(admin_ctx, dataset_id, "nope")


# ---------------------------------------------------------------------------
# Composite keys
# ---------------------------------------------------------------------------


def _key_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": pd.array([1, 1, 2, 2, 3, None, None], dtype="Int64"),
            "day": pd.to_datetime(["2024-01-01"] * 7),
            "amount": [10.0, 10.0, 5.0, 7.0, 1.0, 2.0, 2.0],
            "note": pd.array(["a", "a", None, "full", "x", "m", "m"], dtype="string"),
        }
    )


def test_composite_key_conflicts_identical_vs_conflicting(admin_ctx, factory, small_batches):
    dataset_id = factory.dataset(admin_ctx, _key_frame())
    result = dm.find_composite_key_conflicts(admin_ctx, dataset_id, ["customer_id", "day"])
    assert (
        result["group_count"] == 2 and result["identical_groups"] == 1 and result["conflicting_groups"] == 1
    )
    assert result["rows_in_groups"] == 4 and result["rows_with_missing_key"] == 2
    first, second = result["groups"]
    assert first["rows"] == [0, 1] and first["identical"] and first["differing_columns"] == []
    assert (
        second["rows"] == [2, 3]
        and not second["identical"]
        and second["differing_columns"] == ["amount", "note"]
    )
    assert second["key"]["customer_id"] == 2 and len(second["records"]) == 2
    assert dm.find_composite_key_conflicts(admin_ctx, dataset_id, ["customer_id"], max_groups=1)[
        "groups_truncated"
    ]
    with pytest.raises(ValidationError):
        dm.find_composite_key_conflicts(admin_ctx, dataset_id, [])


@pytest.mark.parametrize(
    ("policy", "remaining"),
    [
        ("keep_first", [0, 2, 4, 5, 6]),
        ("keep_last", [1, 3, 4, 5, 6]),
        ("keep_most_complete", [0, 3, 4, 5, 6]),
        ("drop_all", [4, 5, 6]),
    ],
)
def test_resolution_policies(admin_ctx, factory, policy, remaining):
    frame = _key_frame()
    frame["row"] = pd.array(range(7), dtype="Int64")
    dataset_id = factory.dataset(admin_ctx, frame)
    result = dm.resolve_composite_key_duplicates(
        admin_ctx, dataset_id, ["customer_id", "day"], policy=policy, confirm=True
    )
    assert result["changed"] and result["groups_resolved"] == 2
    assert versioning_service.load_frame(admin_ctx, dataset_id)["row"].tolist() == remaining
    operation = versioning_service.list_operations(admin_ctx, dataset_id)[0]
    assert (
        operation["operation_type"] == "resolve_key_duplicates"
        and operation["parameters"]["policy"] == policy
    )


@pytest.mark.parametrize("policy", [None, "", "keep_random"])
def test_resolution_requires_a_policy(admin_ctx, factory, policy):
    dataset_id = factory.dataset(admin_ctx, _key_frame())
    with pytest.raises(ValidationError, match="policy"):
        dm.resolve_composite_key_duplicates(
            admin_ctx, dataset_id, ["customer_id"], policy=policy, confirm=True
        )
    with pytest.raises(ValidationError, match="confirm"):
        dm.resolve_composite_key_duplicates(admin_ctx, dataset_id, ["customer_id"], policy="keep_first")
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1


# ---------------------------------------------------------------------------
# Identifier generation
# ---------------------------------------------------------------------------


def _ids_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_ref": pd.array(["ID-0002", None, "C-9", "C-9", " ", "ID-0004"], dtype="string"),
            "value": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
        }
    )


def test_sequential_ids_avoid_collisions_and_keep_mapping(admin_ctx, factory, small_batches):
    dataset_id = factory.dataset(admin_ctx, _ids_frame())
    preview = dm.preview_generate_ids(admin_ctx, dataset_id, "customer_ref", mode="sequential", prefix="ID-")
    assert preview["ids_to_generate"] == 3 and preview["missing_ids"] == 2 and preview["repeated_ids"] == 1
    assert [m["new_value"] for m in preview["mapping_sample"]] == ["ID-0001", "ID-0003", "ID-0005"]
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1
    with pytest.raises(ValidationError, match="confirm"):
        dm.generate_ids(admin_ctx, dataset_id, "customer_ref", mode="sequential", prefix="ID-")
    result = dm.generate_ids(
        admin_ctx, dataset_id, "customer_ref", mode="sequential", prefix="ID-", confirm=True
    )
    assert result["mapping"] == [
        {"row_index": 1, "old_value": None, "new_value": "ID-0001"},
        {"row_index": 3, "old_value": "C-9", "new_value": "ID-0003"},
        {"row_index": 4, "old_value": " ", "new_value": "ID-0005"},
    ]
    ids = versioning_service.load_frame(admin_ctx, dataset_id)["customer_ref"].tolist()
    assert ids == ["ID-0002", "ID-0001", "C-9", "ID-0003", "ID-0005", "ID-0004"]
    assert len(set(ids)) == len(ids)
    operation = versioning_service.list_operations(admin_ctx, dataset_id)[0]
    assert (
        operation["operation_type"] == "generate_ids" and operation["summary"]["mapping"] == result["mapping"]
    )


def test_uuid_ids_in_new_column_leave_original_untouched(admin_ctx, factory):
    frame = _ids_frame()
    dataset_id = factory.dataset(admin_ctx, frame)
    result = dm.generate_ids(
        admin_ctx, dataset_id, "customer_ref", target="new_column", new_column="row_uuid", confirm=True
    )
    loaded = versioning_service.load_frame(admin_ctx, dataset_id)
    pd.testing.assert_series_equal(loaded["customer_ref"], frame["customer_ref"])
    assert loaded["row_uuid"].notna().all() and loaded["row_uuid"].nunique() == 6
    assert all(len(value) == 36 for value in loaded["row_uuid"])
    assert result["ids_generated"] == 6 and len(result["mapping"]) == 6


def test_integer_identifier_column_gets_unused_numbers(admin_ctx, factory):
    frame = pd.DataFrame({"order_id": pd.array([1, 3, None, 3, 2], dtype="Int64")})
    dataset_id = factory.dataset(admin_ctx, frame)
    dm.generate_ids(admin_ctx, dataset_id, "order_id", mode="sequential", confirm=True)
    loaded = versioning_service.load_frame(admin_ctx, dataset_id)["order_id"]
    assert str(loaded.dtype) == "Int64" and loaded.tolist() == [1, 3, 4, 5, 2]
    with pytest.raises(ValidationError):
        dm.preview_generate_ids(admin_ctx, dataset_id, "order_id", mode="uuid4")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "random"},
        {"target": "everything"},
        {"target": "new_column"},
        {"target": "new_column", "new_column": "VALUE"},
        {"prefix": "x" * 51},
        {"width": 0},
    ],
)
def test_id_generation_validation(admin_ctx, factory, kwargs):
    dataset_id = factory.dataset(admin_ctx, _ids_frame())
    with pytest.raises(ValidationError):
        dm.preview_generate_ids(admin_ctx, dataset_id, "customer_ref", **kwargs)


def test_ids_cannot_be_generated_in_float_column(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    with pytest.raises(ValidationError):
        dm.preview_generate_ids(admin_ctx, dataset_id, "revenue", mode="sequential")


# ---------------------------------------------------------------------------
# Permissions, plans and tenant isolation
# ---------------------------------------------------------------------------

READ_CALLS = [
    lambda ctx, d: dm.find_exact_duplicates(ctx, d),
    lambda ctx, d: dm.preview_remove_duplicates(ctx, d),
    lambda ctx, d: dm.find_value_duplicates(ctx, d, ["name"]),
    lambda ctx, d: dm.validate_identifier(ctx, d, "customer_id"),
    lambda ctx, d: dm.find_composite_key_conflicts(ctx, d, ["customer_id"]),
]
WRITE_CALLS = [
    lambda ctx, d: dm.remove_duplicates(ctx, d, confirm=True),
    lambda ctx, d: dm.resolve_composite_key_duplicates(
        ctx, d, ["customer_id"], policy="keep_first", confirm=True
    ),
    lambda ctx, d: dm.generate_ids(ctx, d, "name", mode="sequential", confirm=True),
]
ID_CALLS = [
    lambda ctx, d: dm.preview_generate_ids(ctx, d, "name"),
    lambda ctx, d: dm.generate_ids(ctx, d, "name", confirm=True),
]


@pytest.mark.parametrize("call", READ_CALLS + WRITE_CALLS + ID_CALLS)
def test_other_tenant_gets_not_found(factory, admin_ctx, call):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    outsider = factory.org_admin(factory.organization())
    with pytest.raises(NotFoundError):
        call(outsider, dataset_id)


@pytest.mark.parametrize("call", READ_CALLS)
def test_granted_analyst_can_detect(factory, org_id, admin_ctx, call):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    analyst_id = factory.user(org_id, "analyst")
    factory.grant(dataset_id, analyst_id)
    assert isinstance(call(factory.ctx(analyst_id), dataset_id), dict)


@pytest.mark.parametrize("call", READ_CALLS)
def test_ungranted_analyst_gets_not_found(factory, admin_ctx, analyst_ctx, call):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    with pytest.raises(NotFoundError):
        call(analyst_ctx, dataset_id)


@pytest.mark.parametrize("call", WRITE_CALLS)
def test_analyst_without_clean_permission_is_denied(factory, org_id, admin_ctx, call):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    analyst_id = factory.user(org_id, "analyst")
    factory.grant(dataset_id, analyst_id)
    with pytest.raises(PermissionDeniedError):
        call(factory.ctx(analyst_id), dataset_id)
    assert len(versioning_service.list_versions(admin_ctx, dataset_id)) == 1


def test_analyst_with_clean_permission_can_remove(factory, org_id, admin_ctx):
    dataset_id = factory.dataset(admin_ctx, _rows_frame())
    analyst_id = factory.user(org_id, "analyst", permissions={Permission.CLEAN_DATA.value})
    factory.grant(dataset_id, analyst_id)
    assert dm.remove_duplicates(factory.ctx(analyst_id), dataset_id, confirm=True)["rows_removed"] == 3


@pytest.mark.parametrize("call", ID_CALLS)
def test_basic_plan_cannot_generate_ids(factory, call):
    basic = factory.org_admin(factory.organization(plan="basic"))
    dataset_id = factory.dataset(basic, _rows_frame())
    with pytest.raises(FeatureNotAvailableError):
        call(basic, dataset_id)


def test_basic_plan_can_detect_and_remove_duplicates(factory):
    basic = factory.org_admin(factory.organization(plan="basic"))
    dataset_id = factory.dataset(basic, _rows_frame())
    assert dm.find_exact_duplicates(basic, dataset_id)["duplicate_rows"] == 3
    assert dm.remove_duplicates(basic, dataset_id, confirm=True)["rows_removed"] == 3


def test_negative_zero_rows_are_exact_duplicates(factory, admin_ctx):
    from services import analytics_engine, profiling_service

    frame = pd.DataFrame(
        {
            "a": pd.array([1, 2, 1, 3, 1, 2], dtype="Int64"),
            "b": pd.array(["x", "y", "x", "z", "x", "y"], dtype="string"),
            "amt": [0.0, 1.0, -0.0, 2.0, 0.0, 1.0],
        }
    )
    assert int(frame.duplicated().sum()) == 3
    ds = factory.dataset(admin_ctx, frame, "negzero")

    found = dm.find_exact_duplicates(admin_ctx, ds)
    assert found["duplicate_rows"] == 3
    assert [g["rows"] for g in found["groups"]] == [[0, 2, 4], [1, 5]]

    quality = profiling_service.profile_dataset(admin_ctx, ds, use_cache=False)["quality"]
    assert quality["duplicate_rows"] == 3

    stored = versioning_service.load_frame(admin_ctx, ds)
    plan = {"operation": "duplicates"}
    result = analytics_engine.execute_plan(stored, plan)
    assert result["facts"][0]["value"] == 3
    assert analytics_engine.verify_result(stored, plan, result)["verified"] is True
