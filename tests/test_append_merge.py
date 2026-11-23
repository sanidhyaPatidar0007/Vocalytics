"""Append and merge services: schema policies, exact counts, join semantics, safety and authorization."""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import set_settings
from core.errors import (
    AppError,
    ConflictError,
    FeatureNotAvailableError,
    NotFoundError,
    OperationCancelledError,
    PermissionDeniedError,
    ResourceLimitError,
    ValidationError,
)
from services import append_service, merge_service, resource_manager, versioning_service
from services.frame_utils import records
from services.resource_manager import CancellationToken, OperationGuard


def _ints(*values):
    return pd.array(list(values), dtype="Int64")


def _texts(*values):
    return pd.array(list(values), dtype="string")


def _tmp_entries(settings, org_id: int) -> list:
    folder = settings.storage_root / "orgs" / str(org_id) / "tmp"
    return list(folder.iterdir()) if folder.exists() else []


@pytest.fixture
def target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": _ints(1, 2, 3),
            "name": _texts("a", "b", "c"),
            "amount": [1.5, 2.0, 3.0],
        }
    )


@pytest.fixture
def source_frame() -> pd.DataFrame:
    # amount is integer here: integer + float must become float.
    return pd.DataFrame(
        {
            "id": _ints(3, 4, 4, 1),
            "name": _texts("c", "d", "d", "a"),
            "amount": _ints(3, 4, 4, None),
        }
    )


@pytest.fixture
def pair(admin_ctx, factory, target_frame, source_frame) -> tuple[int, int]:
    return factory.dataset(admin_ctx, target_frame, "target"), factory.dataset(
        admin_ctx, source_frame, "source"
    )


# ---------------------------------------------------------------------------
# Append: schema analysis and policies
# ---------------------------------------------------------------------------


def test_analyze_append_reports_columns_case_and_type_conflicts(admin_ctx, factory):
    target = factory.dataset(
        admin_ctx,
        pd.DataFrame(
            {"id": _ints(1, 2), "Region": _texts("N", "S"), "amount": [1.0, 2.5], "code": _ints(7, 8)}
        ),
        "t",
    )
    source = factory.dataset(
        admin_ctx,
        pd.DataFrame(
            {"id": _ints(3), "region": _texts("E"), "amount": _ints(4), "code": _texts("x"), "extra": [0.5]}
        ),
        "s",
    )
    result = append_service.analyze_append(admin_ctx, target, source)
    assert result["matching_columns"] == ["id", "amount", "code"]
    assert result["only_in_target"] == ["Region"]
    assert result["only_in_source"] == ["region", "extra"]
    assert [(c["target_column"], c["source_column"]) for c in result["case_only_differences"]] == [
        ("Region", "region")
    ]
    conflicts = {c["column"]: c for c in result["type_conflicts"]}
    assert conflicts["amount"]["resulting_type"] == {"error": "float", "cast_to_text": "float"}
    assert conflicts["amount"]["requires_cast_to_text"] is False
    assert conflicts["code"]["resulting_type"] == {"error": None, "cast_to_text": "text"}
    assert result["estimated_output_rows"] == 3
    assert result["estimated_output_bytes"] == result["combined_input_bytes"] > 0
    assert result["within_input_target"] is True
    assert result["resource_check"]["ok"] is True
    assert not result["schemas_identical"]


def test_union_keeps_all_columns_and_fills_nulls(admin_ctx, factory):
    target = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(1, 2), "only_t": _texts("x", "y")}), "t")
    source = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(3), "only_s": [9.5]}), "s")
    result = append_service.apply_append(admin_ctx, target, source, confirm=True)
    assert result["rows"] == 3 and result["rows_added"] == 1 and result["version_number"] == 2
    frame = versioning_service.load_frame(admin_ctx, target)
    assert list(frame.columns) == ["id", "only_t", "only_s"]
    assert frame["id"].tolist() == [1, 2, 3]
    assert frame["only_t"].isna().tolist() == [False, False, True]
    assert frame["only_s"].isna().tolist() == [True, True, False]
    assert str(frame["only_s"].dtype) == "float64" and str(frame["only_t"].dtype) == "string"
    history = versioning_service.list_operations(admin_ctx, target)[0]
    assert history["operation_type"] == "append" and history["rows_added"] == 1
    assert history["summary"]["null_filled_columns"] == ["only_t", "only_s"]


def test_intersection_lists_dropped_columns_explicitly(admin_ctx, factory):
    target = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(1), "only_t": _texts("x")}), "t")
    source = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(2), "only_s": [1.0]}), "s")
    preview = append_service.preview_append(admin_ctx, target, source, unmatched_policy="intersection")
    assert preview["dropped_columns"] == {"target": ["only_t"], "source": ["only_s"]}
    assert [c["name"] for c in preview["columns"]] == ["id"]
    assert any("dropped" in w for w in preview["warnings"])
    result = append_service.apply_append(
        admin_ctx, target, source, unmatched_policy="intersection", confirm=True
    )
    assert result["dropped_columns"] == {"target": ["only_t"], "source": ["only_s"]}
    assert list(versioning_service.load_frame(admin_ctx, target).columns) == ["id"]


def test_strict_policy_requires_identical_columns(admin_ctx, factory, pair):
    target = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(1), "x": _texts("a")}), "t")
    source = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(2)}), "s")
    with pytest.raises(ValidationError, match="column sets differ"):
        append_service.preview_append(admin_ctx, target, source, unmatched_policy="strict")
    with pytest.raises(ValidationError):
        append_service.apply_append(admin_ctx, target, source, unmatched_policy="strict", confirm=True)
    target_id, source_id = pair
    preview = append_service.preview_append(admin_ctx, target_id, source_id, unmatched_policy="strict")
    assert preview["output_rows"] == 7


def test_case_only_differences_are_reported_never_matched(admin_ctx, factory):
    target = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(1), "Region": _texts("N")}), "t")
    source = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(2), "region": _texts("S")}), "s")
    with pytest.raises(ValidationError, match="letter case"):
        append_service.preview_append(admin_ctx, target, source, unmatched_policy="union")
    preview = append_service.preview_append(admin_ctx, target, source, unmatched_policy="intersection")
    assert preview["dropped_columns"] == {"target": ["Region"], "source": ["region"]}


def test_type_conflict_error_versus_cast_to_text(admin_ctx, factory):
    target = factory.dataset(
        admin_ctx,
        pd.DataFrame({"code": _ints(5, 6), "when": pd.to_datetime(["2024-01-05 00:00", "2024-01-06 10:30"])}),
        "t",
    )
    source = factory.dataset(admin_ctx, pd.DataFrame({"code": _texts("A7"), "when": _texts("soon")}), "s")
    with pytest.raises(ValidationError, match="Incompatible column types"):
        append_service.preview_append(admin_ctx, target, source)
    with pytest.raises(ValidationError):
        append_service.apply_append(admin_ctx, target, source, confirm=True)
    append_service.apply_append(admin_ctx, target, source, type_conflict_policy="cast_to_text", confirm=True)
    frame = versioning_service.load_frame(admin_ctx, target)
    assert frame["code"].tolist() == ["5", "6", "A7"]
    assert frame["when"].tolist() == ["2024-01-05", "2024-01-06 10:30:00", "soon"]


def test_integer_plus_float_becomes_float_and_counts_are_exact(admin_ctx, pair):
    target_id, source_id = pair
    preview = append_service.preview_append(admin_ctx, target_id, source_id)
    types = {c["name"]: c["logical_type"] for c in preview["columns"]}
    assert types == {"id": "integer", "name": "text", "amount": "float"}
    assert preview["output_rows"] == 7 and preview["target_rows"] == 3 and preview["source_rows"] == 4
    assert len(preview["sample_target_rows"]) == 3 and len(preview["sample_source_rows"]) == 4
    assert preview["sample_source_rows"][0] == {"id": 3, "name": "c", "amount": 3.0}
    result = append_service.apply_append(admin_ctx, target_id, source_id, confirm=True)
    frame = versioning_service.load_frame(admin_ctx, target_id)
    assert len(frame) == result["rows"] == 7
    assert str(frame["amount"].dtype) == "float64"
    assert frame["amount"].tolist()[:6] == [1.5, 2.0, 3.0, 3.0, 4.0, 4.0] and pd.isna(frame["amount"].iloc[6])


def test_possible_duplicates_after_append_are_counted(admin_ctx, pair):
    target_id, source_id = pair
    preview = append_service.preview_append(admin_ctx, target_id, source_id)
    # Source row (3, c, 3) equals target row (3, c, 3.0) after alignment; (4, d, 4) repeats
    # within the source; (1, a, missing) differs from target (1, a, 1.5).
    assert preview["duplicates"]["source_rows_matching_target"] == 1
    assert preview["duplicates"]["duplicates_within_source"] == 1
    assert preview["possible_duplicates"] == 2
    result = append_service.apply_append(admin_ctx, target_id, source_id, confirm=True)
    assert result["possible_duplicates"] == 2 and result["duplicates"] == preview["duplicates"]


def test_source_label_column_records_origin(admin_ctx, pair):
    target_id, source_id = pair
    with pytest.raises(ValidationError, match="already exists"):
        append_service.apply_append(admin_ctx, target_id, source_id, source_label_column="NAME", confirm=True)
    append_service.apply_append(admin_ctx, target_id, source_id, source_label_column="origin", confirm=True)
    frame = versioning_service.load_frame(admin_ctx, target_id)
    assert frame["origin"].tolist() == ["target"] * 3 + ["source"] * 4


def test_bounded_streaming_gives_identical_output(
    admin_ctx, factory, target_frame, source_frame, app_settings
):
    a_target = factory.dataset(admin_ctx, target_frame, "t1")
    a_source = factory.dataset(admin_ctx, source_frame, "s1")
    b_target = factory.dataset(admin_ctx, target_frame, "t2")
    b_source = factory.dataset(admin_ctx, source_frame, "s2")
    default = append_service.apply_append(admin_ctx, a_target, a_source, confirm=True)
    set_settings(app_settings.with_overrides(parquet_batch_rows=2))
    small = append_service.apply_append(admin_ctx, b_target, b_source, confirm=True)
    pd.testing.assert_frame_equal(
        versioning_service.load_frame(admin_ctx, a_target), versioning_service.load_frame(admin_ctx, b_target)
    )
    assert default["duplicates"] == small["duplicates"]


def test_append_as_new_dataset_keeps_target_unchanged(admin_ctx, pair, target_frame):
    target_id, source_id = pair
    result = append_service.apply_append(
        admin_ctx, target_id, source_id, output="new_dataset", new_dataset_name="combined", confirm=True
    )
    assert result["output"] == "new_dataset" and result["dataset_id"] not in pair
    assert len(versioning_service.load_frame(admin_ctx, result["dataset_id"])) == 7
    pd.testing.assert_frame_equal(versioning_service.load_frame(admin_ctx, target_id), target_frame)
    history = versioning_service.list_operations(admin_ctx, result["dataset_id"])
    assert history[0]["operation_type"] == "append" and history[0]["summary"]["rows_added"] == 4
    default_name = append_service.apply_append(
        admin_ctx, target_id, source_id, output="new_dataset", confirm=True
    )
    assert len(versioning_service.list_versions(admin_ctx, default_name["dataset_id"])) == 1


def test_append_requires_confirmation_and_valid_options(admin_ctx, pair):
    target_id, source_id = pair
    with pytest.raises(ValidationError, match="confirm"):
        append_service.apply_append(admin_ctx, target_id, source_id)
    with pytest.raises(ValidationError):
        append_service.apply_append(admin_ctx, target_id, source_id, output="file", confirm=True)
    with pytest.raises(ValidationError):
        append_service.preview_append(admin_ctx, target_id, source_id, unmatched_policy="everything")
    with pytest.raises(ValidationError):
        append_service.preview_append(admin_ctx, target_id, source_id, sample_rows=0)


def test_append_preview_and_analysis_never_write_a_version(admin_ctx, pair):
    target_id, source_id = pair
    append_service.analyze_append(admin_ctx, target_id, source_id)
    append_service.preview_append(admin_ctx, target_id, source_id)
    assert len(versioning_service.list_versions(admin_ctx, target_id)) == 1
    assert len(versioning_service.list_versions(admin_ctx, source_id)) == 1


def test_append_conflict_when_target_changes_during_operation(admin_ctx, pair, app_settings):
    target_id, source_id = pair
    changed = []

    def change_target(_fraction, _message):
        if not changed:
            changed.append(True)
            versioning_service.apply_frame_operation(
                admin_ctx,
                target_id,
                operation_type="drop",
                parameters={},
                transform=lambda df: (df.iloc[1:], {"rows_removed": 1}, {}),
            )

    with pytest.raises(ConflictError):
        append_service.apply_append(
            admin_ctx, target_id, source_id, confirm=True, guard=OperationGuard(progress=change_target)
        )
    assert [v["operation_type"] for v in versioning_service.list_versions(admin_ctx, target_id)] == [
        "drop",
        "import",
    ]
    assert _tmp_entries(app_settings, admin_ctx.organization_id) == []


def test_append_cancellation_publishes_nothing_and_cleans_up(admin_ctx, pair, app_settings):
    target_id, source_id = pair
    set_settings(app_settings.with_overrides(parquet_batch_rows=1))
    token = CancellationToken()

    def cancel_after_first(fraction, _message):
        token.cancel()

    guard = OperationGuard(progress=cancel_after_first, cancel_token=token)
    with pytest.raises(OperationCancelledError):
        append_service.apply_append(admin_ctx, target_id, source_id, confirm=True, guard=guard)
    assert len(versioning_service.list_versions(admin_ctx, target_id)) == 1
    assert versioning_service.list_operations(admin_ctx, target_id)[0]["status"] == "cancelled"
    assert _tmp_entries(app_settings, admin_ctx.organization_id) == []
    append_service.apply_append(admin_ctx, target_id, source_id, confirm=True)
    assert _tmp_entries(app_settings, admin_ctx.organization_id) == []


def test_append_resource_rejection(admin_ctx, pair, monkeypatch):
    target_id, source_id = pair
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: 1024)
    analysis = append_service.analyze_append(admin_ctx, target_id, source_id)
    assert analysis["resource_check"]["ok"] is False and "memory" in analysis["resource_check"]["message"]
    preview = append_service.preview_append(admin_ctx, target_id, source_id)
    assert preview["duplicates"] is None and any("skipped" in w for w in preview["warnings"])
    with pytest.raises(ResourceLimitError):
        append_service.apply_append(admin_ctx, target_id, source_id, confirm=True)
    assert len(versioning_service.list_versions(admin_ctx, target_id)) == 1


def test_append_input_target_warning(admin_ctx, pair, app_settings, monkeypatch):
    target_id, source_id = pair
    monkeypatch.setattr(append_service, "MB", 1)
    set_settings(app_settings.with_overrides(append_merge_input_target_mb=1))
    analysis = append_service.analyze_append(admin_ctx, target_id, source_id)
    assert analysis["within_input_target"] is False
    assert any("processing target" in w for w in analysis["warnings"])
    assert analysis["resource_check"]["ok"] is True


# ---------------------------------------------------------------------------
# Append: authorization
# ---------------------------------------------------------------------------


def test_append_analyst_needs_clean_data_permission(factory, org_id, admin_ctx, pair):
    target_id, source_id = pair
    analyst = factory.analyst(org_id)
    for dataset_id in pair:
        factory.grant(dataset_id, analyst.user_id)
    assert append_service.analyze_append(analyst, target_id, source_id)["estimated_output_rows"] == 7
    assert append_service.preview_append(analyst, target_id, source_id)["output_rows"] == 7
    with pytest.raises(PermissionDeniedError):
        append_service.apply_append(analyst, target_id, source_id, confirm=True)
    with pytest.raises(PermissionDeniedError):  # permission is checked before the confirmation
        append_service.apply_append(analyst, target_id, source_id)
    cleaner = factory.analyst(org_id, permissions={"clean_data"})
    with pytest.raises(NotFoundError):
        append_service.apply_append(cleaner, target_id, source_id, confirm=True)
    for dataset_id in pair:
        factory.grant(dataset_id, cleaner.user_id)
    assert append_service.apply_append(cleaner, target_id, source_id, confirm=True)["rows"] == 7


def test_append_requires_premium_plan(factory, target_frame, source_frame):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    target, source = factory.dataset(ctx, target_frame, "t"), factory.dataset(ctx, source_frame, "s")
    with pytest.raises(FeatureNotAvailableError):
        append_service.analyze_append(ctx, target, source)
    with pytest.raises(FeatureNotAvailableError):
        append_service.preview_append(ctx, target, source)
    with pytest.raises(FeatureNotAvailableError):
        append_service.apply_append(ctx, target, source, confirm=True)
    with pytest.raises(FeatureNotAvailableError):  # the plan is checked before the confirmation
        append_service.apply_append(ctx, target, source)
    with pytest.raises(FeatureNotAvailableError):
        append_service.preview_append(ctx, target, source, sample_rows=0)


def test_append_cross_tenant_dataset_is_not_found(factory, admin_ctx, pair, source_frame):
    target_id, _ = pair
    other_ctx = factory.org_admin(factory.organization())
    foreign = factory.dataset(other_ctx, source_frame, "foreign")
    for call in (
        lambda: append_service.analyze_append(admin_ctx, target_id, foreign),
        lambda: append_service.preview_append(admin_ctx, target_id, foreign),
        lambda: append_service.apply_append(admin_ctx, target_id, foreign, confirm=True),
        lambda: append_service.apply_append(admin_ctx, foreign, target_id, confirm=True),
    ):
        with pytest.raises(NotFoundError):
            call()
    assert len(versioning_service.list_versions(other_ctx, foreign)) == 1


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

EXPECTED_ROWS = {"inner": 3, "left": 5, "right": 5, "outer": 7}


@pytest.fixture
def orders_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": _ints(10, 11, 12, 13, 14),
            "cust_id": _ints(1, 1, 2, 4, None),
            "amount": [5.0, 6.5, 7.0, 8.0, 9.0],
            "note": _texts("o1", "o2", "o3", "o4", "o5"),
        }
    )


@pytest.fixture
def customers_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cust_id": _ints(1, 2, 3, None),
            "name": _texts("Ann", "Bob", "Cy", "Nobody"),
            "note": _texts("c1", "c2", "c3", "c4"),
        }
    )


@pytest.fixture
def merge_pair(admin_ctx, factory, orders_frame, customers_frame) -> tuple[int, int]:
    return factory.dataset(admin_ctx, orders_frame, "orders"), factory.dataset(
        admin_ctx, customers_frame, "customers"
    )


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(frame.columns), na_position="last").reset_index(drop=True)


def _merge_kwargs(how: str, **extra):
    return {"left_on": ["cust_id"], "right_on": ["cust_id"], "how": how, **extra}


def _row_keys(rows: list[dict]) -> set[tuple]:
    return {tuple(sorted(row.items())) for row in rows}


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_join_types_match_reference_and_exact_estimate(
    admin_ctx, merge_pair, orders_frame, customers_frame, how
):
    left_id, right_id = merge_pair
    analysis = merge_service.analyze_merge(admin_ctx, left_id, right_id, **_merge_kwargs(how))
    assert analysis["output_rows"] == EXPECTED_ROWS[how]
    assert analysis["output_rows_by_join"] == EXPECTED_ROWS
    result = merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs(how), confirm=True)
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert len(frame) == result["rows"] == analysis["output_rows"]
    assert list(frame.columns) == ["order_id", "cust_id", "amount", "note_left", "name", "note_right"]
    reference = pd.merge(
        orders_frame.dropna(subset=["cust_id"]),
        customers_frame.dropna(subset=["cust_id"]),
        on="cust_id",
        how=how,
        suffixes=("_left", "_right"),
    )
    produced = frame[frame["cust_id"].notna()]
    pd.testing.assert_frame_equal(_sorted(produced), _sorted(reference), check_dtype=False)
    expected_nulls = {"inner": 0, "left": 1, "right": 1, "outer": 2}[how]
    assert int(frame["cust_id"].isna().sum()) == expected_nulls


def test_unmatched_counts_are_reported_never_silent(admin_ctx, merge_pair):
    left_id, right_id = merge_pair
    analysis = merge_service.analyze_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"))
    assert analysis["left_unmatched_rows"] == 2 and analysis["right_unmatched_rows"] == 2
    assert analysis["left_null_key_rows"] == 1 and analysis["right_null_key_rows"] == 1
    assert (
        analysis["unmatched"]["left_rows_excluded"] == 2 and analysis["unmatched"]["right_rows_excluded"] == 2
    )
    assert any("leaves out 2 unmatched left row(s)" in w for w in analysis["warnings"])
    result = merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("left"), confirm=True)
    assert result["left_unmatched_kept"] == 2 and result["right_rows_excluded"] == 2
    history = versioning_service.list_operations(admin_ctx, result["dataset_id"])[0]
    assert history["operation_type"] == "merge"
    assert history["summary"]["right_rows_excluded"] == 2 and history["summary"]["left_unmatched_kept"] == 2


def test_null_keys_never_match_each_other(admin_ctx, factory):
    left = factory.dataset(admin_ctx, pd.DataFrame({"k": _ints(None, 1), "a": _texts("x", "y")}), "l")
    right = factory.dataset(admin_ctx, pd.DataFrame({"k": _ints(None, 1), "b": _texts("p", "q")}), "r")
    inner = merge_service.apply_merge(
        admin_ctx, left, right, left_on=["k"], right_on=["k"], how="inner", confirm=True
    )
    assert versioning_service.load_frame(admin_ctx, inner["dataset_id"])["a"].tolist() == ["y"]
    outer = merge_service.apply_merge(
        admin_ctx,
        left,
        right,
        left_on=["k"],
        right_on=["k"],
        how="outer",
        confirm=True,
        new_dataset_name="outer",
    )
    frame = versioning_service.load_frame(admin_ctx, outer["dataset_id"])
    assert len(frame) == 3  # pandas' default would pair the two missing keys and give 2 rows
    null_rows = frame[frame["k"].isna()]
    assert sorted(null_rows["a"].fillna("-").tolist()) == ["-", "x"]
    assert sorted(null_rows["b"].fillna("-").tolist()) == ["-", "p"]


def test_composite_keys_and_partial_nulls(admin_ctx, factory):
    left = factory.dataset(
        admin_ctx,
        pd.DataFrame(
            {
                "region": _texts("N", "N", "S", "S"),
                "year": _ints(2023, 2024, 2023, None),
                "sales": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        "l",
    )
    right = factory.dataset(
        admin_ctx,
        pd.DataFrame(
            {"region": _texts("N", "S", "S"), "yr": _ints(2024, 2023, 2023), "target": [10.0, 20.0, 21.0]}
        ),
        "r",
    )
    kwargs = {"left_on": ["region", "year"], "right_on": ["region", "yr"], "how": "left"}
    analysis = merge_service.analyze_merge(admin_ctx, left, right, **kwargs)
    assert analysis["matched_keys"] == 2 and analysis["left_null_key_rows"] == 1
    assert analysis["relationship"] == "one-to-many" and analysis["output_rows"] == 5
    result = merge_service.apply_merge(admin_ctx, left, right, **kwargs, confirm=True)
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert list(frame.columns) == [
        "region",
        "year",
        "sales",
        "yr",
        "target",
    ]  # "yr" differs from "year": both kept
    assert len(frame) == 5
    assert frame.loc[frame["sales"] == 3.0, "target"].sort_values().tolist() == [20.0, 21.0]
    assert frame.loc[frame["sales"] == 4.0, "target"].isna().all()


def test_duplicate_keys_and_relationships(admin_ctx, merge_pair):
    left_id, right_id = merge_pair
    analysis = merge_service.analyze_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"))
    assert analysis["relationship"] == "many-to-one"
    assert analysis["left_duplicate_keys"] == 1 and analysis["left_duplicate_key_rows"] == 2
    assert analysis["right_duplicate_keys"] == 0
    reverse = merge_service.analyze_merge(admin_ctx, right_id, left_id, **_merge_kwargs("inner"))
    assert reverse["relationship"] == "one-to-many" and reverse["output_rows"] == 3


def test_many_to_many_is_detected_warned_and_needs_confirmation(admin_ctx, factory):
    left = factory.dataset(admin_ctx, pd.DataFrame({"k": _ints(1, 1, 1, 2), "a": [1.0, 2.0, 3.0, 4.0]}), "l")
    right = factory.dataset(admin_ctx, pd.DataFrame({"k": _ints(1, 1, 1, 3), "b": [5.0, 6.0, 7.0, 8.0]}), "r")
    kwargs = {"left_on": ["k"], "right_on": ["k"], "how": "inner"}
    analysis = merge_service.analyze_merge(admin_ctx, left, right, **kwargs)
    assert analysis["relationship"] == "many-to-many" and analysis["many_to_many_keys"] == 1
    assert analysis["output_rows"] == 9 and analysis["growth_factor"] == 2.25
    assert any("many-to-many" in w for w in analysis["warnings"])
    assert any("2.2 times" in w for w in analysis["warnings"])
    with pytest.raises(ValidationError, match="many-to-many"):
        merge_service.apply_merge(admin_ctx, left, right, **kwargs, confirm=True)
    result = merge_service.apply_merge(
        admin_ctx, left, right, **kwargs, confirm=True, confirm_many_to_many=True
    )
    assert result["rows"] == 9 and result["relationship"] == "many-to-many"


def test_integer_and_float_keys_are_compatible_text_is_not(admin_ctx, factory):
    left = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(1, 2, 3), "a": _texts("x", "y", "z")}), "l")
    floats = factory.dataset(
        admin_ctx, pd.DataFrame({"id": [1.0, 2.5, 3.0], "b": _texts("p", "q", "r")}), "f"
    )
    texts = factory.dataset(admin_ctx, pd.DataFrame({"id": _texts("1", "2"), "c": [1.0, 2.0]}), "t")
    kwargs = {"left_on": ["id"], "right_on": ["id"], "how": "inner"}
    analysis = merge_service.analyze_merge(admin_ctx, left, floats, **kwargs)
    assert analysis["key_types"][0]["match_type"] == "float" and analysis["output_rows"] == 2
    result = merge_service.apply_merge(admin_ctx, left, floats, **kwargs, confirm=True)
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert sorted(frame["id"].tolist()) == [1.0, 3.0] and str(frame["id"].dtype) == "float64"
    with pytest.raises(ValidationError, match="Convert"):
        merge_service.analyze_merge(admin_ctx, left, texts, **kwargs)
    with pytest.raises(ValidationError):
        merge_service.apply_merge(admin_ctx, left, texts, **kwargs, confirm=True)


def test_merge_input_validation(admin_ctx, merge_pair):
    left_id, right_id = merge_pair
    cases = [
        {"left_on": ["cust_id"], "right_on": ["cust_id"], "how": "cross"},
        {"left_on": ["cust_id", "order_id"], "right_on": ["cust_id"], "how": "inner"},
        {"left_on": ["missing"], "right_on": ["cust_id"], "how": "inner"},
        {"left_on": [], "right_on": [], "how": "inner"},
        {"left_on": ["cust_id"], "right_on": ["cust_id"], "how": "inner", "suffixes": ("_x", "_X")},
        {"left_on": ["cust_id"], "right_on": ["cust_id"], "how": "inner", "suffixes": ("", "_r")},
    ]
    for kwargs in cases:
        with pytest.raises(ValidationError):
            merge_service.analyze_merge(admin_ctx, left_id, right_id, **kwargs)
    with pytest.raises(ValidationError, match="confirm"):
        merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"))
    with pytest.raises(ValidationError):
        merge_service.apply_merge(
            admin_ctx, left_id, right_id, **_merge_kwargs("inner"), output="disk", confirm=True
        )
    with pytest.raises(ValidationError):
        merge_service.preview_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"), sample_rows=10**6)


def test_custom_suffixes_and_suffix_collisions(admin_ctx, factory, merge_pair):
    left_id, right_id = merge_pair
    result = merge_service.apply_merge(
        admin_ctx, left_id, right_id, **_merge_kwargs("inner"), suffixes=("_o", "_c"), confirm=True
    )
    columns = list(versioning_service.load_frame(admin_ctx, result["dataset_id"]).columns)
    assert columns == ["order_id", "cust_id", "amount", "note_o", "name", "note_c"]
    left = factory.dataset(admin_ctx, pd.DataFrame({"k": _ints(1), "x": [1.0], "x_left": [2.0]}), "l")
    right = factory.dataset(admin_ctx, pd.DataFrame({"k": _ints(1), "X": [3.0]}), "r")
    with pytest.raises(ValidationError, match="suffixes"):
        merge_service.analyze_merge(admin_ctx, left, right, left_on=["k"], right_on=["k"], how="inner")
    analysis = merge_service.analyze_merge(
        admin_ctx, left, right, left_on=["k"], right_on=["k"], how="inner", suffixes=("_a", "_b")
    )
    assert analysis["output_columns"] == ["k", "x_a", "x_left", "X_b"]
    assert analysis["overlapping_columns"][0]["left_output"] == "x_a"


def test_merge_output_as_new_version_of_left(admin_ctx, merge_pair):
    left_id, right_id = merge_pair
    result = merge_service.apply_merge(
        admin_ctx, left_id, right_id, **_merge_kwargs("left"), output="new_version", confirm=True
    )
    assert result["dataset_id"] == left_id and result["version_number"] == 2 and result["rows"] == 5
    history = versioning_service.list_operations(admin_ctx, left_id)[0]
    assert history["operation_type"] == "merge" and history["parameters"]["how"] == "left"
    assert len(versioning_service.list_versions(admin_ctx, right_id)) == 1


def test_merge_new_dataset_name_conflict(admin_ctx, merge_pair):
    left_id, right_id = merge_pair
    first = merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"), confirm=True)
    assert first["output"] == "new_dataset" and first["dataset_id"] not in merge_pair
    with pytest.raises(ConflictError):
        merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"), confirm=True)
    with pytest.raises(ConflictError):
        merge_service.apply_merge(
            admin_ctx, left_id, right_id, **_merge_kwargs("inner"), new_dataset_name="orders", confirm=True
        )


def test_preview_merge_samples_real_rows_without_writing(admin_ctx, merge_pair, app_settings):
    left_id, right_id = merge_pair
    set_settings(app_settings.with_overrides(parquet_batch_rows=2))
    preview = merge_service.preview_merge(
        admin_ctx, left_id, right_id, **_merge_kwargs("outer"), sample_rows=4
    )
    assert preview["output_rows"] == 7 and preview["sample_is_subset"] is True
    assert [c["name"] for c in preview["columns"]] == preview["output_columns"]
    assert 0 < len(preview["sample_rows"]) <= 4
    assert {r["side"]: r["excluded_from_output"] for r in preview["unmatched_row_counts"]} == {
        "left": 0,
        "right": 0,
    }
    assert len(versioning_service.list_versions(admin_ctx, left_id)) == 1
    assert len(versioning_service.list_versions(admin_ctx, right_id)) == 1
    full = merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("outer"), confirm=True)
    real = _row_keys(records(versioning_service.load_frame(admin_ctx, full["dataset_id"])))
    assert _row_keys(preview["sample_rows"]) <= real


def test_merge_resource_rejections(admin_ctx, merge_pair, app_settings, monkeypatch):
    left_id, right_id = merge_pair
    set_settings(app_settings.with_overrides(merge_max_output_rows=4))
    analysis = merge_service.analyze_merge(admin_ctx, left_id, right_id, **_merge_kwargs("outer"))
    assert analysis["resource_check"]["ok"] is False and "rows" in analysis["resource_check"]["message"]
    with pytest.raises(ResourceLimitError, match="7 rows"):
        merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("outer"), confirm=True)
    assert (
        merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"), confirm=True)[
            "rows"
        ]
        == 3
    )
    set_settings(app_settings)
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: 4096)
    with pytest.raises(ResourceLimitError, match="memory"):
        merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"), confirm=True)
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: 8 * 1024**3)
    monkeypatch.setattr(merge_service, "MB", 1)
    with pytest.raises(ResourceLimitError, match="limit for merges"):
        merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"), confirm=True)
    failed = versioning_service.list_operations(admin_ctx, left_id)[0]
    assert failed["status"] == "failed" and failed["operation_type"] == "merge"
    assert len(versioning_service.list_versions(admin_ctx, left_id)) == 1


def test_merge_memory_check_is_reported_then_enforced(admin_ctx, merge_pair, monkeypatch):
    left_id, right_id = merge_pair
    needed = merge_service.analyze_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"))[
        "estimated_memory_bytes"
    ]
    # Enough memory to inspect the key columns, but the safe budget (half) is below the merge estimate.
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: needed)
    analysis = merge_service.analyze_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"))
    assert analysis["resource_check"]["ok"] is False
    assert analysis["resource_check"]["message"].startswith("Merging these datasets needs about")
    with pytest.raises(ResourceLimitError, match="Merging these datasets"):
        merge_service.apply_merge(admin_ctx, left_id, right_id, **_merge_kwargs("inner"), confirm=True)
    failed = versioning_service.list_operations(admin_ctx, left_id)[0]
    assert failed["status"] == "failed" and failed["operation_type"] == "merge"


def test_merge_row_count_mismatch_saves_nothing(admin_ctx, merge_pair, monkeypatch, app_settings):
    left_id, right_id = merge_pair
    real = merge_service.perform_merge
    monkeypatch.setattr(merge_service, "perform_merge", lambda *a, **k: real(*a, **k).iloc[1:])
    with pytest.raises(AppError, match="unexpected number of rows"):
        merge_service.apply_merge(
            admin_ctx, left_id, right_id, **_merge_kwargs("inner"), output="new_version", confirm=True
        )
    assert len(versioning_service.list_versions(admin_ctx, left_id)) == 1
    assert _tmp_entries(app_settings, admin_ctx.organization_id) == []


def test_merge_cancellation_and_conflict(admin_ctx, merge_pair, app_settings):
    left_id, right_id = merge_pair
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelledError):
        merge_service.apply_merge(
            admin_ctx,
            left_id,
            right_id,
            **_merge_kwargs("inner"),
            confirm=True,
            guard=OperationGuard(cancel_token=token),
        )
    assert versioning_service.list_operations(admin_ctx, left_id)[0]["status"] == "cancelled"
    changed = []

    def change_left(_fraction, _message):
        if not changed:
            changed.append(True)
            versioning_service.apply_frame_operation(
                admin_ctx,
                left_id,
                operation_type="drop",
                parameters={},
                transform=lambda df: (df.iloc[1:], {}, {}),
            )

    with pytest.raises(ConflictError):
        merge_service.apply_merge(
            admin_ctx,
            left_id,
            right_id,
            **_merge_kwargs("inner"),
            output="new_version",
            confirm=True,
            guard=OperationGuard(progress=change_left),
        )
    assert [v["operation_type"] for v in versioning_service.list_versions(admin_ctx, left_id)] == [
        "drop",
        "import",
    ]
    assert _tmp_entries(app_settings, admin_ctx.organization_id) == []


def test_merge_analyst_needs_clean_data_permission(factory, org_id, merge_pair):
    left_id, right_id = merge_pair
    analyst = factory.analyst(org_id)
    for dataset_id in merge_pair:
        factory.grant(dataset_id, analyst.user_id)
    assert (
        merge_service.analyze_merge(analyst, left_id, right_id, **_merge_kwargs("inner"))["output_rows"] == 3
    )
    assert (
        merge_service.preview_merge(analyst, left_id, right_id, **_merge_kwargs("inner"))["output_rows"] == 3
    )
    with pytest.raises(PermissionDeniedError):
        merge_service.apply_merge(analyst, left_id, right_id, **_merge_kwargs("inner"), confirm=True)
    cleaner = factory.analyst(org_id, permissions={"clean_data"})
    factory.grant(left_id, cleaner.user_id)
    with pytest.raises(NotFoundError):
        merge_service.apply_merge(cleaner, left_id, right_id, **_merge_kwargs("inner"), confirm=True)
    factory.grant(right_id, cleaner.user_id)
    assert (
        merge_service.apply_merge(cleaner, left_id, right_id, **_merge_kwargs("inner"), confirm=True)["rows"]
        == 3
    )


def test_merge_requires_premium_plan(factory, orders_frame, customers_frame):
    ctx = factory.org_admin(factory.organization(plan="basic"))
    left, right = factory.dataset(ctx, orders_frame, "o"), factory.dataset(ctx, customers_frame, "c")
    for call in (
        lambda: merge_service.analyze_merge(ctx, left, right, **_merge_kwargs("inner")),
        lambda: merge_service.preview_merge(ctx, left, right, **_merge_kwargs("inner")),
        lambda: merge_service.preview_merge(ctx, left, right, **_merge_kwargs("inner"), sample_rows=0),
        lambda: merge_service.apply_merge(ctx, left, right, **_merge_kwargs("inner"), confirm=True),
        lambda: merge_service.apply_merge(
            ctx, left, right, **_merge_kwargs("inner")
        ),  # plan checked before confirm
    ):
        with pytest.raises(FeatureNotAvailableError):
            call()


def test_merge_cross_tenant_dataset_is_not_found(factory, admin_ctx, merge_pair, customers_frame):
    left_id, _ = merge_pair
    other_ctx = factory.org_admin(factory.organization())
    foreign = factory.dataset(other_ctx, customers_frame, "foreign")
    for call in (
        lambda: merge_service.analyze_merge(admin_ctx, left_id, foreign, **_merge_kwargs("inner")),
        lambda: merge_service.preview_merge(admin_ctx, left_id, foreign, **_merge_kwargs("inner")),
        lambda: merge_service.apply_merge(
            admin_ctx, left_id, foreign, **_merge_kwargs("inner"), confirm=True
        ),
        lambda: merge_service.apply_merge(
            admin_ctx, foreign, left_id, **_merge_kwargs("inner"), confirm=True
        ),
    ):
        with pytest.raises(NotFoundError):
            call()


def test_append_wider_than_the_column_limit_is_refused_before_any_work(admin_ctx, factory, app_settings):
    """SEC3: an append whose result would exceed MAX_COLUMNS is refused up front (label column included)."""
    target = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(1, 2), "only_t": _texts("x", "y")}), "t")
    source = factory.dataset(admin_ctx, pd.DataFrame({"id": _ints(3), "only_s": [9.5]}), "s")
    set_settings(app_settings.with_overrides(max_columns=2))
    for call in (
        lambda: append_service.preview_append(admin_ctx, target, source),
        lambda: append_service.apply_append(admin_ctx, target, source, confirm=True),
    ):
        with pytest.raises(ValidationError, match="would have 3 columns; datasets can have at most 2 columns"):
            call()
    set_settings(app_settings.with_overrides(max_columns=1))
    with pytest.raises(ValidationError, match="would have 2 columns"):  # id plus the label column
        append_service.apply_append(
            admin_ctx, target, source, unmatched_policy="intersection", source_label_column="origin",
            confirm=True,
        )
    assert len(versioning_service.list_versions(admin_ctx, target)) == 1
    result = append_service.apply_append(
        admin_ctx, target, source, unmatched_policy="intersection", confirm=True
    )
    assert result["columns"] == 1 and result["version_number"] == 2


def test_merge_cancelled_while_saving_publishes_no_version(admin_ctx, merge_pair):
    """Regression (SR5 follow-up): merge passes its guard to publish_version, so a cancel during
    the quality pass of the new version publishes nothing."""
    left_id, right_id = merge_pair
    token = CancellationToken()

    def progress(_fraction: float, message: str) -> None:
        if message == "Saving the result":
            token.cancel()

    with pytest.raises(OperationCancelledError):
        merge_service.apply_merge(
            admin_ctx,
            left_id,
            right_id,
            **_merge_kwargs("left"),
            output="new_version",
            confirm=True,
            guard=OperationGuard(progress=progress, cancel_token=token),
        )
    assert len(versioning_service.list_versions(admin_ctx, left_id)) == 1
