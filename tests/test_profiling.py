"""Profiling service: exact statistics, confirmed vs suspected issues, health check, AI summary."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from config.settings import set_settings
from core.errors import FeatureNotAvailableError, NotFoundError, OperationCancelledError, ValidationError
from database import session_scope
from database.models import AuditLog, PlanFeature, Subscription
from services import dataset_service, profiling_service, versioning_service
from services.quality_score import compute_quality
from services.resource_manager import CancellationToken, OperationGuard


@pytest.fixture(autouse=True)
def _fresh_cache():
    profiling_service.clear_profile_cache()
    yield
    profiling_service.clear_profile_cache()


def _upload(ctx, text: str, name: str) -> int:
    result = dataset_service.import_dataset(
        ctx, file_name=f"{name}.csv", data=text.encode("utf-8"), dataset_name=name
    )
    return result["dataset_id"]


def _col(profile: dict, name: str) -> dict:
    return next(c for c in profile["columns_profile"] if c["name"] == name)


def _types(issues: list[dict], column: str | None = None) -> set[str]:
    return {i["type"] for i in issues if column is None or i["column"] == column}


def _strip_time(profile: dict) -> dict:
    return {k: v for k, v in profile.items() if k != "generated_at"}


def _disable_profiling(org_id: int) -> None:
    with session_scope() as db:
        plan_id = db.execute(
            select(Subscription.plan_id).where(Subscription.organization_id == org_id)
        ).scalar_one()
        db.execute(
            select(PlanFeature).where(
                PlanFeature.plan_id == plan_id, PlanFeature.feature_key == "data_profiling"
            )
        ).scalar_one().enabled = False


def _strings(values: list) -> pd.array:
    return pd.array(values, dtype="string")


# ---------------------------------------------------------------------------
# Exact statistics
# ---------------------------------------------------------------------------


def test_numeric_and_text_stats_match_pandas(admin_ctx, factory):
    amounts = pd.array([5, 1, None, 9, 3, 3, 12, None, 7], dtype="Int64")
    frame = pd.DataFrame(
        {
            "amount": amounts,
            "price": [1.5, 2.25, np.nan, 4.0, -1.0, 0.0, 2.25, 3.0, 8.5],
            "city": _strings(["Pune", "Delhi", "Pune", None, "Goa", "Pune", "Delhi", "Goa", "Agra"]),
            "when": pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 12:00",
                    "2024-01-02 00:00",
                    None,
                    "2024-02-01 00:00",
                    "2024-02-01 00:00",
                    "2024-03-05 00:00",
                    "2024-03-05 08:00",
                    "2024-03-06 00:00",
                ]
            ),
            "flag": pd.array([True, False, True, None, True, True, False, True, True], dtype="boolean"),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame, "stats")
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)

    assert profile["rows"] == 9 and profile["columns"] == 5 and profile["version_number"] == 1
    assert profile["dataset_name"] == "stats" and profile["generated_at"]
    amount = _col(profile, "amount")
    expected = pd.Series(amounts).dropna().astype("float64")
    assert amount["logical_type"] == "integer" and amount["semantic"] == "numeric"
    assert amount["non_null"] == 7 and amount["missing"] == 2 and amount["missing_pct"] == round(200 / 9, 2)
    assert amount["distinct"] == expected.nunique() == 6
    stats = amount["numeric"]
    assert stats["min"] == 1 and stats["max"] == 12
    assert stats["mean"] == pytest.approx(expected.mean())
    assert stats["median"] == pytest.approx(expected.median())
    assert stats["q1"] == pytest.approx(expected.quantile(0.25))
    assert stats["q3"] == pytest.approx(expected.quantile(0.75))
    assert stats["std"] == pytest.approx(expected.std())

    price = _col(profile, "price")["numeric"]
    assert price["zero_count"] == 1 and price["negative_count"] == 1
    assert price["mean"] == pytest.approx(frame["price"].mean())

    city = _col(profile, "city")
    assert city["semantic"] == "categorical" and city["distinct"] == 4 and city["missing"] == 1
    assert city["top_values"][0] == {"value": "Pune", "count": 3}
    assert city["distinct_pct"] == round(100 * 4 / 8, 2)
    assert city["text"]["min_length"] == 3 and city["text"]["max_length"] == 5

    when = _col(profile, "when")
    assert when["datetime"]["distinct_days"] == 5
    assert when["datetime"]["min"].startswith("2024-01-01T10:00") and when["datetime"]["max"].startswith(
        "2024-03-06"
    )
    assert _col(profile, "flag")["boolean"] == {"true_count": 6, "false_count": 2}
    json.dumps(profile)  # plain, JSON-safe data


def test_profile_is_consistent_with_quality_score(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "name": _strings([" Ann", "Bob", "Bob", None, "Cy "]),
            "score": pd.array([1, 2, 2, None, 4], dtype="Int64"),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    quality = profile["quality"]
    assert quality == compute_quality(frame).to_dict()
    assert sum(c["missing"] for c in profile["columns_profile"]) == quality["missing_cells"] == 2
    assert _col(profile, "name")["text"]["whitespace_count"] == quality["whitespace_cells"] == 2
    assert profile["duplicate_rows"] == {"count": quality["duplicate_rows"], "pct": 20.0}


def test_iqr_outliers_respect_fence_boundaries(admin_ctx, factory):
    values = [-6, -5] + [10] * 10 + [20] * 10 + [35, 36]
    series = pd.Series(values, dtype="float64")
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    assert (q1, q3) == (10.0, 20.0)  # fences are exactly -5 and 35
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"v": pd.array(values, dtype="Int64")}))
    stats = _col(profiling_service.profile_dataset(admin_ctx, dataset_id), "v")["numeric"]
    # -5 and 35 sit on the fences and are not outliers; -6 and 36 are just beyond them.
    assert stats["outliers_iqr"] == 2


@pytest.mark.parametrize(("zeros", "expected"), [(9, 0), (10, 1)])
def test_z_score_outliers_at_the_threshold(admin_ctx, factory, zeros, expected):
    # n zeros plus one 100 gives that value z = n / sqrt(n + 1) with the sample std (ddof=1):
    # 9 zeros -> 2.85 (not an outlier), 10 zeros -> 3.02 (outlier).
    values = [0] * zeros + [100]
    series = pd.Series(values, dtype="float64")
    by_hand = int((((series - series.mean()) / series.std(ddof=1)).abs() > 3).sum())
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"v": pd.array(values, dtype="Int64")}))
    stats = _col(profiling_service.profile_dataset(admin_ctx, dataset_id), "v")["numeric"]
    assert stats["outliers_z"] == by_hand == expected


def test_z_score_outliers_use_sample_std(admin_ctx, factory):
    values = [10] * 30 + [11] * 30 + [1000]
    series = pd.Series(values, dtype="float64")
    expected = int((((series - series.mean()) / series.std(ddof=1)).abs() > 3).sum())
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"v": pd.array(values, dtype="Int64")}))
    stats = _col(profiling_service.profile_dataset(admin_ctx, dataset_id), "v")["numeric"]
    assert stats["outliers_z"] == expected == 1
    assert stats["outlier_candidates"] == 1


def test_constant_and_empty_columns(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "const": _strings(["A", "A", "A", None]),
            "empty": _strings([None, None, None, None]),
            "num": pd.array([7, 7, 7, 7], dtype="Int64"),
            "n": pd.array([1, 2, 3, 4], dtype="Int64"),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert _col(profile, "const")["is_constant"] and not _col(profile, "const")["is_empty"]
    empty = _col(profile, "empty")
    assert empty["is_empty"] and not empty["is_constant"] and empty["semantic"] == "empty"
    num = _col(profile, "num")
    assert num["is_constant"] and num["numeric"]["outliers_z"] == 0 and num["numeric"]["std"] == 0
    assert _types(profile["confirmed_issues"], "empty") == {"empty_column"}
    assert "constant_column" in _types(profile["suspected_issues"], "const")
    assert "constant_column" in _types(profile["suspected_issues"], "num")
    assert "constant_column" not in _types(profile["confirmed_issues"])


def test_whitespace_and_case_variants(admin_ctx, factory):
    frame = pd.DataFrame({"region": _strings(["North", "north", "NORTH ", "North", "South", "South"])})
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    text = _col(profile, "region")["text"]
    assert text["whitespace_count"] == 1
    groups = text["case_variant_groups"]
    assert len(groups) == 1 and groups[0]["normalized"] == "north" and groups[0]["total"] == 4
    assert groups[0]["variants"][0] == {"value": "North", "count": 2}
    assert {v["value"] for v in groups[0]["variants"]} == {"North", "north", "NORTH "}
    assert "whitespace" in _types(profile["confirmed_issues"], "region")
    suspected = [i for i in profile["suspected_issues"] if i["type"] == "inconsistent_categories"]
    assert suspected and suspected[0]["count"] == 2  # values not in the dominant spelling
    assert "inconsistent_categories" not in _types(profile["confirmed_issues"])


def test_case_variant_counts_are_exact_beyond_the_listed_sample(admin_ctx, factory):
    names = [f"Item{i}" for i in range(25)]
    frame = pd.DataFrame({"item": _strings(names + [n.upper() for n in names])})
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    text = _col(profile, "item")["text"]
    assert len(text["case_variant_groups"]) == profiling_service.CASE_GROUP_LIMIT == 20
    assert text["case_variant_group_count"] == 25 and text["case_variant_rows"] == 25
    issue = next(i for i in profile["suspected_issues"] if i["type"] == "inconsistent_categories")
    assert issue["count"] == 25 and "25 value(s)" in issue["message"]


def test_mostly_numeric_text_reports_invalid_values(admin_ctx):
    dataset_id = _upload(admin_ctx, "price\n10\n12\n13\nabc\n15\n", "prices")
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    price = _col(profile, "price")
    assert price["logical_type"] == "text"
    assert price["semantic"] == profile["import_report"]["column_types"]["price"]["semantic"]
    assert price["text"]["mostly_numeric_invalid"] == {"count": 1, "examples": ["abc"], "numeric_pct": 80.0}
    issue = next(i for i in profile["confirmed_issues"] if i["type"] == "invalid_numeric")
    assert issue["column"] == "price" and issue["count"] == 1 and "abc" in issue["message"]


def test_inconsistent_and_invalid_dates_are_reported(admin_ctx):
    csv = "day\n2024-01-05\n2024-02-10\n05/03/2024\n2024-04-01\nnot a date\n"
    dataset_id = _upload(admin_ctx, csv, "dates")
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    info = _col(profile, "day")["text"]["mostly_date_invalid"]
    assert info["count"] == 1 and info["examples"] == ["not a date"] and info["inconsistent_formats"]
    assert info["formats_seen"] == {"%Y-%m-%d": 3, "%d/%m/%Y": 1}
    assert {"invalid_dates", "inconsistent_date_formats"} <= _types(profile["confirmed_issues"], "day")


def test_single_ambiguous_format_is_not_inconsistent(admin_ctx, factory):
    # Every date parses as %m/%d/%Y even though first-match counting sees %d/%m/%Y for one of them.
    dates = ["01/02/2024", "02/13/2024", "03/14/2024", "04/15/2024", "05/16/2024", "x"]
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"d": _strings(dates)}))
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    info = _col(profile, "d")["text"]["mostly_date_invalid"]
    assert info is not None and info["count"] == 1 and info["examples"] == ["x"]
    assert info["inconsistent_formats"] is False
    assert _types(profile["confirmed_issues"], "d") == {"invalid_dates"}


def test_mixed_currencies_are_a_conversion_failure(admin_ctx):
    dataset_id = _upload(admin_ctx, "amount\n$10\n€12\n$5\n", "money")
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert _col(profile, "amount")["text"]["mixed_currencies"] == ["EUR", "USD"]
    assert "conversion_failure" in _types(profile["confirmed_issues"], "amount")


def test_import_report_rows_are_confirmed_import_issues(admin_ctx):
    dataset_id = _upload(admin_ctx, "a,b\n1,2\n3,4,5\n,\n6,7\n", "rough")
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    by_type = {i["type"]: i for i in profile["confirmed_issues"]}
    assert by_type["malformed_rows"]["count"] == 1 and by_type["malformed_rows"]["source"] == "import"
    assert by_type["blank_rows"]["count"] == 1
    assert profile["import_report"]["malformed_rows_skipped"] == 1
    # Skipped rows are facts about the upload: they do not lower the health status.
    assert profiling_service.health_check(admin_ctx, dataset_id)["status"] == "healthy"


def test_header_only_dataset_profiles_cleanly(admin_ctx):
    dataset_id = _upload(admin_ctx, "a,b\n", "headers")
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert profile["rows"] == 0 and profile["duplicate_rows"] == {"count": 0, "pct": 0.0}
    assert all(c["missing_pct"] == 0.0 for c in profile["columns_profile"])
    assert profile["confirmed_issues"] == []


# ---------------------------------------------------------------------------
# Confirmed vs suspected
# ---------------------------------------------------------------------------


def test_outliers_and_repeated_categories_are_only_suspected(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "line": pd.array(range(1, 21), dtype="Int64"),
            "region": _strings(["North"] * 12 + ["South"] * 8),
            "revenue": [100.0] * 10 + [110.0] * 9 + [5000.0],
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert profile["confirmed_issues"] == []
    outlier = next(i for i in profile["suspected_issues"] if i["type"] == "outliers")
    assert outlier["column"] == "revenue" and outlier["count"] == 1
    assert "possible outliers, review before changing; they may be legitimate" in outlier["message"]
    assert not _types(profile["suspected_issues"], "region")


def test_exact_duplicate_rows(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "a": pd.array([1, 2, 1, 3, 1, 2], dtype="Int64"),
            "b": _strings(["x", "y", "x", "z", "x", None]),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert profile["duplicate_rows"] == {"count": 2, "pct": round(200 / 6, 2)}
    issue = next(i for i in profile["confirmed_issues"] if i["type"] == "duplicate_rows")
    assert issue["count"] == 2 and issue["column"] is None and issue["severity"] == "high"


def test_duplicated_business_keys_are_suspected(admin_ctx, factory):
    skus = [f"S{i:03d}" for i in range(99)] + ["S000"]  # 99 percent unique
    frame = pd.DataFrame(
        {
            "sku": _strings(skus),
            "customer_id": pd.array([1, 2, 3, 3] * 25, dtype="Int64"),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert _col(profile, "sku")["semantic"] == "identifier"
    keys = {
        i["column"]: i["count"] for i in profile["suspected_issues"] if i["type"] == "duplicate_business_key"
    }
    assert keys == {"sku": 1, "customer_id": 97}
    assert "duplicate_business_key" not in _types(profile["confirmed_issues"])


def test_high_missing_is_confirmed_missing_and_suspected_unreliable(admin_ctx, factory):
    frame = pd.DataFrame(
        {"note": _strings(["a", None, None, None]), "k": pd.array([1, 2, 3, 4], dtype="Int64")}
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id)
    missing = next(i for i in profile["confirmed_issues"] if i["type"] == "missing_values")
    assert missing["count"] == 3 and missing["severity"] == "high"
    assert "high_missing" in _types(profile["suspected_issues"], "note")


# ---------------------------------------------------------------------------
# Batching, caching, versions, cancellation
# ---------------------------------------------------------------------------


def test_results_are_exact_across_batches(admin_ctx, factory, app_settings):
    frame = pd.DataFrame(
        {
            "a": pd.array([1, 2, 3, 1, 2, 3, 1], dtype="Int64"),
            "b": _strings([" x", "y", "z", " x", "y", "q", " x"]),
            "c": [0.5, 1.5, np.nan, 0.5, 1.5, 9.0, 0.5],
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    unbatched = profiling_service.profile_dataset(admin_ctx, dataset_id, use_cache=False)
    set_settings(app_settings.with_overrides(parquet_batch_rows=2))
    batched = profiling_service.profile_dataset(admin_ctx, dataset_id, use_cache=False)
    assert _strip_time(batched) == _strip_time(unbatched)
    assert batched["duplicate_rows"]["count"] == 3  # duplicates sit in different batches


def test_cache_returns_copies_and_follows_new_versions(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    first = profiling_service.profile_dataset(admin_ctx, dataset_id)
    first["columns_profile"].clear()
    again = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert len(again["columns_profile"]) == 6 and again["generated_at"] == first["generated_at"]

    def drop_two(df):
        return df.iloc[2:].reset_index(drop=True), {"rows_removed": 2}, {}

    versioning_service.apply_frame_operation(
        admin_ctx, dataset_id, operation_type="drop", parameters={}, transform=drop_two
    )
    latest = profiling_service.profile_dataset(admin_ctx, dataset_id)
    assert latest["version_number"] == 2 and latest["rows"] == 4
    old = profiling_service.profile_dataset(admin_ctx, dataset_id, version_id=first["version_id"])
    assert old["version_number"] == 1 and old["rows"] == 6


def test_cache_never_skips_authorization(admin_ctx, factory, org_id, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    profiling_service.profile_dataset(admin_ctx, dataset_id)
    _disable_profiling(org_id)
    with pytest.raises(FeatureNotAvailableError):
        profiling_service.profile_dataset(admin_ctx, dataset_id)


def test_version_of_another_dataset_is_not_found(admin_ctx, factory, sales_frame):
    first = factory.dataset(admin_ctx, sales_frame, "one")
    second = factory.dataset(admin_ctx, sales_frame, "two")
    other_version = profiling_service.profile_dataset(admin_ctx, second)["version_id"]
    with pytest.raises(NotFoundError):
        profiling_service.profile_dataset(admin_ctx, first, version_id=other_version)


def test_guard_reports_progress_and_cancels(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    steps: list[float] = []
    profiling_service.profile_dataset(
        admin_ctx, dataset_id, guard=OperationGuard(progress=lambda f, _m: steps.append(f)), use_cache=False
    )
    assert steps[0] == 0.0 and steps[-1] == 1.0 and steps == sorted(steps)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelledError):
        profiling_service.profile_dataset(
            admin_ctx, dataset_id, guard=OperationGuard(cancel_token=token), use_cache=False
        )


# ---------------------------------------------------------------------------
# Repeated values
# ---------------------------------------------------------------------------


def _repeat_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": _strings([" North", "north", "North", "South", "South", "East", None, None]),
            "product": _strings(["A", "A", "A", "B", "B", "C", "A", "A"]),
        }
    )


def test_repeated_values_exact_match(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _repeat_frame())
    result = profiling_service.repeated_values(admin_ctx, dataset_id, ["region"], trim=False)
    assert result.to_dict("records") == [{"region": "South", "count": 2}]


def test_repeated_values_trim_and_case_options(admin_ctx, factory):
    frame = _repeat_frame()
    dataset_id = factory.dataset(admin_ctx, frame)
    trimmed = profiling_service.repeated_values(admin_ctx, dataset_id, ["region"])
    assert trimmed.to_dict("records") == [{"region": " North", "count": 2}, {"region": "South", "count": 2}]
    folded = profiling_service.repeated_values(
        admin_ctx, dataset_id, ["region", "product"], case_insensitive=True
    )
    assert folded.to_dict("records") == [
        {"region": " North", "product": "A", "count": 3},
        {"region": "South", "product": "B", "count": 2},
    ]
    assert list(folded.columns) == ["region", "product", "count"]
    limited = profiling_service.repeated_values(admin_ctx, dataset_id, ["product"], top=1)
    assert limited.to_dict("records") == [{"product": "A", "count": 5}]
    # Normalization is for comparison only: the stored data is unchanged.
    pd.testing.assert_series_equal(
        versioning_service.load_frame(admin_ctx, dataset_id)["region"], frame["region"], check_names=False
    )


def test_repeated_values_with_no_repeats_is_empty(admin_ctx, factory):
    frame = pd.DataFrame({"k": pd.array([1, 2, 3], dtype="Int64")})
    dataset_id = factory.dataset(admin_ctx, frame)
    result = profiling_service.repeated_values(admin_ctx, dataset_id, ["k"])
    assert result.empty and list(result.columns) == ["k", "count"]


@pytest.mark.parametrize(
    ("columns", "top"),
    [
        ([], 10),
        (["missing_column"], 10),
        (["region", "region"], 10),
        (["region"], 0),
        (["region"], 5000),
        ("region", 10),
    ],
)
def test_repeated_values_validation(admin_ctx, factory, columns, top):
    dataset_id = factory.dataset(admin_ctx, _repeat_frame())
    with pytest.raises(ValidationError):
        profiling_service.repeated_values(admin_ctx, dataset_id, columns, top=top)


# ---------------------------------------------------------------------------
# Health check and AI summary
# ---------------------------------------------------------------------------


def test_health_check_healthy_and_audited(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    result = profiling_service.health_check(admin_ctx, dataset_id)
    assert result["status"] == "healthy" and result["quality_score"] == 100.0
    assert result["top_issues"] == [] and set(result["thresholds"]) == {
        "empty",
        "critical",
        "needs attention",
        "healthy",
    }
    with session_scope() as db:
        entry = db.execute(select(AuditLog).where(AuditLog.action == "dataset.health_check")).scalar_one()
        assert entry.target_id == str(dataset_id) and entry.organization_id == admin_ctx.organization_id


def test_health_check_needs_attention(admin_ctx, factory):
    frame = pd.DataFrame(
        {"a": pd.array([1, 2, 3, 4, 5, None, 7, 8, 9, 10], dtype="Int64"), "b": _strings(list("abcdefghij"))}
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    result = profiling_service.health_check(admin_ctx, dataset_id)
    assert result["status"] == "needs attention"
    issue = result["top_issues"][0]
    assert issue["type"] == "missing_values" and issue["kind"] == "confirmed"
    assert issue["recommended_step"] == "Review missing values in the Missing values workspace."


def test_health_check_critical(admin_ctx, factory):
    frame = pd.DataFrame({"a": pd.array([1, 1, 1, 2], dtype="Int64"), "b": _strings(["x", "x", "x", None])})
    dataset_id = factory.dataset(admin_ctx, frame)
    result = profiling_service.health_check(admin_ctx, dataset_id)
    assert result["status"] == "critical"
    assert result["top_issues"][0]["type"] == "duplicate_rows"
    assert all(i["recommended_step"] for i in result["top_issues"] + result["suspected_issues"])


def _issue(severity: str, source: str = "data") -> dict:
    return {"severity": severity, "source": source}


@pytest.mark.parametrize(
    ("score", "issues", "status"),
    [
        (59.9, [], "critical"),
        (60.0, [], "needs attention"),
        (89.9, [], "needs attention"),
        (90.0, [], "healthy"),
        (100.0, [_issue("low")], "healthy"),
        (100.0, [_issue("medium")], "needs attention"),
        (100.0, [_issue("high")], "critical"),
        (100.0, [_issue("high", "import")], "healthy"),
    ],
)
def test_health_status_thresholds(score, issues, status):
    assert profiling_service._health_status(score, issues) == status


def test_column_summary_for_ai_is_compact(admin_ctx, factory):
    long_text = "x" * 60
    frame = pd.DataFrame(
        {
            "label": _strings([long_text, "a", "b", "c", "d", "e", "f", "a"]),
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "day": pd.to_datetime([f"2024-01-0{d}" for d in range(1, 9)]),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    summary = {s["name"]: s for s in profiling_service.column_summary_for_ai(admin_ctx, dataset_id)}
    label = summary["label"]
    assert len(label["top_values"]) == 5 and label["top_values"][0] == "a"
    assert all(len(v) <= 40 for v in label["top_values"])
    assert label["distinct"] == 7 and label["missing_pct"] == 0.0
    assert (
        summary["amount"]["min"] == 1.0
        and summary["amount"]["max"] == 8.0
        and summary["amount"]["mean"] == 4.5
    )
    assert "top_values" not in summary["amount"] and summary["day"]["min"].startswith("2024-01-01")


def test_ai_summary_truncates_long_values(admin_ctx, factory):
    frame = pd.DataFrame({"label": _strings(["y" * 60, "y" * 60, "z"])})
    dataset_id = factory.dataset(admin_ctx, frame)
    (summary,) = profiling_service.column_summary_for_ai(admin_ctx, dataset_id)
    assert summary["top_values"] == ["y" * 40, "z"]


# ---------------------------------------------------------------------------
# Authorization: access grants, tenants and plans
# ---------------------------------------------------------------------------

_CALLS = {
    "profile": lambda ctx, ds: profiling_service.profile_dataset(ctx, ds),
    "repeated": lambda ctx, ds: profiling_service.repeated_values(ctx, ds, ["region"]),
    "health": lambda ctx, ds: profiling_service.health_check(ctx, ds),
    "ai_summary": lambda ctx, ds: profiling_service.column_summary_for_ai(ctx, ds),
}


@pytest.mark.parametrize("call", list(_CALLS))
def test_analyst_needs_a_dataset_grant(admin_ctx, factory, org_id, sales_frame, call):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    analyst = factory.analyst(org_id, permissions=set())
    with pytest.raises(NotFoundError):
        _CALLS[call](analyst, dataset_id)
    factory.grant(dataset_id, analyst.user_id)
    _CALLS[call](analyst, dataset_id)  # any user with dataset access may profile


@pytest.mark.parametrize("call", list(_CALLS))
def test_other_tenant_gets_not_found(admin_ctx, factory, sales_frame, call):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    _CALLS["profile"](admin_ctx, dataset_id)  # warm the cache: it must not leak across tenants
    outsider = factory.org_admin(factory.organization("Other Co"))
    with pytest.raises(NotFoundError):
        _CALLS[call](outsider, dataset_id)


@pytest.mark.parametrize("call", list(_CALLS))
def test_plan_without_profiling_is_denied(factory, sales_frame, call):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    dataset_id = factory.dataset(ctx, sales_frame)
    _disable_profiling(org)
    with pytest.raises(FeatureNotAvailableError):
        _CALLS[call](ctx, dataset_id)


# --- Regression SEC3: profiling reads the file in column batches, not once per column ---


def test_profile_reads_columns_in_batches_not_once_per_column(admin_ctx, factory, monkeypatch):
    frame = pd.DataFrame({f"c{i}": pd.array([i, i + 1, None], dtype="Int64") for i in range(30)})
    dataset_id = factory.dataset(admin_ctx, frame, "wide")
    reads = {"files": 0, "authorizations": 0}
    original_read, original_load = versioning_service.read_parquet, versioning_service.load_frame
    original_ref = versioning_service.get_version_ref

    def counting_read(*args, **kwargs):
        reads["files"] += 1
        return original_read(*args, **kwargs)

    def counting_load(*args, **kwargs):
        reads["files"] += 1
        return original_load(*args, **kwargs)

    def counting_ref(*args, **kwargs):
        reads["authorizations"] += 1
        return original_ref(*args, **kwargs)

    monkeypatch.setattr(versioning_service, "read_parquet", counting_read)
    monkeypatch.setattr(versioning_service, "load_frame", counting_load)
    monkeypatch.setattr(versioning_service, "get_version_ref", counting_ref)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id, use_cache=False)
    assert [c["name"] for c in profile["columns_profile"]] == list(frame.columns)
    assert _col(profile, "c7")["missing"] == 1 and _col(profile, "c7")["numeric"]["max"] == 8
    assert reads["files"] <= 2
    assert reads["authorizations"] <= 3


# --- Regression SEC4: dataset text in issue messages and examples stays on one line ---


def test_issue_messages_and_examples_are_single_line(admin_ctx, factory):
    payload = "oops\n\n![p](https://attacker.example/beacon.png)"
    values = [str(i * 10) for i in range(19)] + [payload]
    dates = ["2024-01-0" + str(i) for i in range(1, 10)] + ["2024-13-45\r\n\r\n[x](https://attacker.example)"]
    frame = pd.DataFrame(
        {
            "amount": _strings(values),
            "day": _strings(dates + [None] * 10),
            "label": _strings(["North\n\nRegion", "north\n\nregion"] * 10),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame, "newline values")
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id, use_cache=False)
    issues = profile["confirmed_issues"] + profile["suspected_issues"]
    assert {"invalid_numeric", "invalid_dates", "inconsistent_categories"} <= {i["type"] for i in issues}
    for issue in issues:
        assert "\n" not in issue["message"] and "\r" not in issue["message"], issue
    assert any("oops ![p](https://attacker.example/beacon.png)" in i["message"] for i in issues)
    numeric = _col(profile, "amount")["text"]["mostly_numeric_invalid"]
    assert numeric["examples"] == ["oops ![p](https://attacker.example/beacon.png)"]
    date_examples = _col(profile, "day")["text"]["mostly_date_invalid"]["examples"]
    assert all("\n" not in e and "\r" not in e for e in date_examples)


def test_header_only_dataset_health_check_is_empty_with_a_plain_issue(admin_ctx):
    dataset_id = _upload(admin_ctx, "a,b\n", "headers only")
    health = profiling_service.health_check(admin_ctx, dataset_id)
    assert health["status"] == "empty"
    assert health["confirmed_issue_count"] == 1
    issue = health["top_issues"][0]
    assert issue["type"] == "no_rows" and issue["message"].startswith("This dataset has no rows")
    assert issue["recommended_step"] != "Review this issue."
    assert "empty" in health["thresholds"]


def test_column_batches_never_refuse_a_dataset_whose_single_columns_fit(admin_ctx, factory, monkeypatch):
    """Batching must not need more memory than the old one-column-at-a-time profile."""
    from services import resource_manager

    frame = pd.DataFrame({f"c{i}": pd.array(range(2000), dtype="Int64") for i in range(30)})
    dataset_id = factory.dataset(admin_ctx, frame, "tall")
    path = versioning_service.get_version_ref(admin_ctx, dataset_id).path
    one = max(resource_manager.estimate_parquet_memory(path, [name]) for name in frame.columns)
    every = resource_manager.estimate_parquet_memory(path, list(frame.columns))
    assert one * 3 < every
    fraction = resource_manager.get_settings().memory_safety_fraction
    budget = one * 2
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: int(budget / fraction) + 1)
    monkeypatch.setattr(resource_manager, "check_disk", lambda *a, **k: None)
    profile = profiling_service.profile_dataset(admin_ctx, dataset_id, use_cache=False)
    assert len(profile["columns_profile"]) == 30
