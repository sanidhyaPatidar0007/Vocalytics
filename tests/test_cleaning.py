"""Cleaning service: exact previews, versioned applies, plan/permission/tenant guards."""

from __future__ import annotations

import importlib
import importlib.util
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import set_settings
from core.errors import (
    ConflictError,
    FeatureNotAvailableError,
    NotFoundError,
    OperationTimeoutError,
    PermissionDeniedError,
    ResourceLimitError,
    ValidationError,
)
from services import cleaning_service as cs
from services import resource_manager
from services import versioning_service as vs


def _text(values: list) -> pd.arrays.StringArray:
    return pd.array(values, dtype="string")


def _vals(series: pd.Series) -> list:
    """Plain Python values with every missing value as None."""
    return [None if pd.isna(v) else v for v in series.tolist()]


def _version_count(ctx, dataset_id: int) -> int:
    return len(vs.list_versions(ctx, dataset_id))


def _apply(ctx, dataset_id, operation, params, **kwargs):
    return cs.apply_cleaning(ctx, dataset_id, operation, params, **kwargs)


@pytest.fixture
def numbers_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": _text(["a", "b", "c", "d", "e"]),
            "amount": [10.0, 12.0, 11.0, 13.0, 100.0],
            "qty": pd.array([10, 12, 11, 13, 100], dtype="Int64"),
        }
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_operation_registry_documents_the_standard_advanced_split():
    standard = {
        "trim_whitespace",
        "remove_control_chars",
        "rename_columns",
        "remove_empty",
        "normalize_case",
        "drop_columns",
    }
    advanced = {
        "map_categories",
        "standardize_dates",
        "convert_numeric",
        "convert_types",
        "handle_outliers",
        "fix_invalid_values",
        "validate_constraints",
    }
    assert {n for n, m in cs.OPERATIONS.items() if m["feature"] == "standard"} == standard
    assert {n for n, m in cs.OPERATIONS.items() if m["feature"] == "advanced"} == advanced
    listed = cs.list_operations()
    assert [item["name"] for item in listed] == list(cs.OPERATIONS)
    assert all(item["label"] and item["params"] for item in listed)
    assert cs.OPERATIONS["drop_columns"]["destructive"] and cs.OPERATIONS["validate_constraints"]["read_only"]


def test_requires_confirmation_rules():
    assert cs.requires_confirmation("remove_empty", {})
    assert cs.requires_confirmation("drop_columns", {"columns": ["x"]})
    assert cs.requires_confirmation("handle_outliers", {"action": "exclude"})
    assert not cs.requires_confirmation("handle_outliers", {"action": "cap"})
    assert cs.requires_confirmation("fix_invalid_values", {"action": "set_missing"})
    assert cs.requires_confirmation("convert_types", {"on_failure": "set_missing"})
    assert not cs.requires_confirmation("convert_types", {"on_failure": "abort"})
    with pytest.raises(ValidationError):
        cs.requires_confirmation("explode", {})


def test_unknown_operation_and_options_are_rejected(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    with pytest.raises(ValidationError, match="Unknown cleaning operation"):
        cs.preview_cleaning(admin_ctx, dataset_id, "explode", {})
    with pytest.raises(ValidationError, match="Unknown option"):
        cs.preview_cleaning(admin_ctx, dataset_id, "trim_whitespace", {"colums": ["region"]})
    with pytest.raises(ValidationError, match="Unknown column"):
        cs.preview_cleaning(admin_ctx, dataset_id, "trim_whitespace", {"columns": ["nope"]})
    with pytest.raises(ValidationError, match="text"):
        cs.preview_cleaning(admin_ctx, dataset_id, "trim_whitespace", {"columns": ["revenue"]})
    with pytest.raises(ValidationError, match="sample size"):
        cs.preview_cleaning(admin_ctx, dataset_id, "trim_whitespace", {}, sample_rows=500)


# ---------------------------------------------------------------------------
# Standard operations
# ---------------------------------------------------------------------------


@pytest.fixture
def whitespace_dataset(admin_ctx, factory) -> int:
    frame = pd.DataFrame(
        {
            "id": pd.array([1, 2, 3, 4], dtype="Int64"),
            "name": _text([" Alice ", "Bob", "   ", None]),
            "city": _text(["Paris ", "Rome", "Oslo", "Lima"]),
        }
    )
    return factory.dataset(admin_ctx, frame, "people")


def test_trim_whitespace_preview_is_exact_and_read_only(admin_ctx, whitespace_dataset):
    preview = cs.preview_cleaning(admin_ctx, whitespace_dataset, "trim_whitespace", {})
    assert preview["params"]["columns"] == ["name", "city"]
    assert preview["cells_changed"] == 3 and preview["rows_affected"] == 2
    assert preview["rows_before"] == preview["rows_after"] == 4
    assert preview["failures"] == {"count": 0, "examples": []}
    assert [s["row"] for s in preview["sample"]] == [1, 3]
    assert (
        preview["sample"][0]["before"]["name"] == " Alice "
        and preview["sample"][0]["after"]["name"] == "Alice"
    )
    assert preview["quality_after"] > preview["quality_before"]
    assert preview["can_apply"] and not preview["requires_confirmation"]
    only_name = cs.preview_cleaning(
        admin_ctx, whitespace_dataset, "trim_whitespace", {"columns": ["name"]}, sample_rows=1
    )
    assert only_name["cells_changed"] == 2 and len(only_name["sample"]) == 1
    assert _version_count(admin_ctx, whitespace_dataset) == 1


def test_trim_whitespace_apply_publishes_a_new_version(admin_ctx, whitespace_dataset):
    original = vs.load_frame(admin_ctx, whitespace_dataset)
    result = _apply(admin_ctx, whitespace_dataset, "trim_whitespace", {})
    assert result["version_number"] == 2
    assert result["stats"]["cells_changed"] == 3 and result["stats"]["rows_affected"] == 2
    assert result["quality_after"] > result["quality_before"]
    frame = vs.load_frame(admin_ctx, whitespace_dataset)
    assert _vals(frame["name"]) == ["Alice", "Bob", "", None]
    assert _vals(frame["city"]) == ["Paris", "Rome", "Oslo", "Lima"]
    assert str(frame["id"].dtype) == "Int64" and str(frame["name"].dtype) == "string"
    history = vs.list_operations(admin_ctx, whitespace_dataset)[0]
    assert history["operation_type"] == "trim_whitespace" and history["cells_changed"] == 3
    first = vs.list_versions(admin_ctx, whitespace_dataset)[-1]["version_id"]
    pd.testing.assert_frame_equal(vs.load_frame(admin_ctx, whitespace_dataset, version_id=first), original)
    vs.restore_version(admin_ctx, whitespace_dataset, first)
    pd.testing.assert_frame_equal(vs.load_frame(admin_ctx, whitespace_dataset), original)


def test_remove_control_characters(admin_ctx, factory):
    frame = pd.DataFrame(
        {"note": _text(["a\tb", "line1\r\nline2", "bell\x07x", "c1\x85d", "ok", "\x00z", None])}
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "remove_control_chars", {"columns": ["note"]})
    assert preview["cells_changed"] == 5
    _apply(admin_ctx, dataset_id, "remove_control_chars", {"columns": ["note"]})
    assert _vals(vs.load_frame(admin_ctx, dataset_id)["note"]) == [
        "a b",
        "line1 line2",
        "bellx",
        "c1 d",
        "ok",
        "z",
        None,
    ]


def test_normalize_case(admin_ctx, factory):
    frame = pd.DataFrame({"city": _text(["new york", "PARIS", "Rome", None])})
    dataset_id = factory.dataset(admin_ctx, frame)
    preview = cs.preview_cleaning(
        admin_ctx, dataset_id, "normalize_case", {"columns": ["city"], "mode": "title"}
    )
    assert preview["cells_changed"] == 2
    _apply(admin_ctx, dataset_id, "normalize_case", {"columns": ["city"], "mode": "upper"})
    assert _vals(vs.load_frame(admin_ctx, dataset_id)["city"]) == ["NEW YORK", "PARIS", "ROME", None]
    with pytest.raises(ValidationError, match="mode"):
        cs.preview_cleaning(admin_ctx, dataset_id, "normalize_case", {"mode": "sentence"})


def test_rename_columns_and_validation(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    preview = cs.preview_cleaning(
        admin_ctx, dataset_id, "rename_columns", {"mapping": {"region": "Region", "product": " item "}}
    )
    assert preview["columns_renamed"] == {"region": "Region", "product": "item"}
    assert preview["columns_added"] == [] and preview["columns_removed"] == []
    assert preview["cells_changed"] == 0 and len(preview["sample"]) == 6
    for mapping, message in [
        ({"region": "  "}, "blank"),
        ({"region": "x" * 121}, "120"),
        ({"region": "Product"}, "case-insensitive"),
        ({"nope": "x"}, "Unknown column"),
        ({"region": "region"}, "nothing to rename"),
        ({}, "at least one"),
    ]:
        with pytest.raises(ValidationError, match=message):
            cs.preview_cleaning(admin_ctx, dataset_id, "rename_columns", {"mapping": mapping})
    _apply(admin_ctx, dataset_id, "rename_columns", {"mapping": {"region": "product", "product": "region"}})
    frame = vs.load_frame(admin_ctx, dataset_id)
    assert frame["product"].tolist() == sales_frame["region"].tolist()
    assert list(frame.columns) == ["order_id", "product", "region", "revenue", "quantity", "order_date"]


@pytest.fixture
def empty_dataset(admin_ctx, factory) -> int:
    frame = pd.DataFrame(
        {
            "a": pd.array([1, None, 3], dtype="Int64"),
            "b": _text(["x", "  ", None]),
            "c": _text([None, None, " "]),
        }
    )
    return factory.dataset(admin_ctx, frame)


def test_remove_empty_requires_confirm_and_removes_rows_and_columns(admin_ctx, empty_dataset):
    preview = cs.preview_cleaning(admin_ctx, empty_dataset, "remove_empty", {})
    assert preview["rows_removed"] == 1 and preview["rows_after"] == 2
    assert preview["columns_removed"] == ["c"] and preview["columns_after"] == 2
    assert preview["requires_confirmation"]
    assert preview["sample"] == [{"row": 2, "before": {"a": None, "b": "  ", "c": None}, "after": None}]
    with pytest.raises(ValidationError, match="confirm"):
        _apply(admin_ctx, empty_dataset, "remove_empty", {})
    assert _version_count(admin_ctx, empty_dataset) == 1
    result = _apply(admin_ctx, empty_dataset, "remove_empty", {}, confirm=True)
    assert result["stats"]["rows_removed"] == 1 and result["columns_removed"] == ["c"]
    frame = vs.load_frame(admin_ctx, empty_dataset)
    assert list(frame.columns) == ["a", "b"]
    assert _vals(frame["a"]) == [1, 3] and _vals(frame["b"]) == ["x", None]


def test_remove_empty_options(admin_ctx, empty_dataset, factory):
    only_rows = cs.preview_cleaning(admin_ctx, empty_dataset, "remove_empty", {"columns": False})
    assert only_rows["rows_removed"] == 1 and only_rows["columns_removed"] == []
    only_columns = cs.preview_cleaning(admin_ctx, empty_dataset, "remove_empty", {"rows": False})
    assert only_columns["rows_removed"] == 0 and only_columns["columns_removed"] == ["c"]
    with pytest.raises(ValidationError, match="Choose"):
        cs.preview_cleaning(admin_ctx, empty_dataset, "remove_empty", {"rows": False, "columns": False})
    all_empty = factory.dataset(admin_ctx, pd.DataFrame({"x": _text([None, " "]), "y": _text(["", None])}))
    with pytest.raises(ValidationError, match="Every column is empty"):
        cs.preview_cleaning(admin_ctx, all_empty, "remove_empty", {"rows": False})


def test_drop_columns(admin_ctx, factory, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    with pytest.raises(ValidationError, match="confirm"):
        _apply(admin_ctx, dataset_id, "drop_columns", {"columns": ["product"]})
    with pytest.raises(ValidationError, match="at least one column"):
        cs.preview_cleaning(admin_ctx, dataset_id, "drop_columns", {"columns": list(sales_frame.columns)})
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "drop_columns", {"columns": ["product", "quantity"]})
    assert preview["columns_removed"] == ["product", "quantity"] and preview["columns_after"] == 4
    result = _apply(admin_ctx, dataset_id, "drop_columns", {"columns": ["product", "quantity"]}, confirm=True)
    assert result["columns_after"] == 4
    assert list(vs.load_frame(admin_ctx, dataset_id).columns) == [
        "order_id",
        "region",
        "revenue",
        "order_date",
    ]


# ---------------------------------------------------------------------------
# Categories and dates
# ---------------------------------------------------------------------------


@pytest.fixture
def region_dataset(admin_ctx, factory) -> int:
    frame = pd.DataFrame(
        {
            "region": _text(["North", "north", "NORTH", "South", "East", None]),
            "n": pd.array(range(6), dtype="Int64"),
        }
    )
    return factory.dataset(admin_ctx, frame)


def test_map_categories_counts_per_mapping_entry(admin_ctx, region_dataset):
    params = {
        "column": "region",
        "mapping": {"north": "North", "S": "South", "East": "Eastern"},
        "case_insensitive": True,
    }
    preview = cs.preview_cleaning(admin_ctx, region_dataset, "map_categories", params)
    assert preview["counts"] == {"north": 2, "East": 1}
    assert preview["cells_changed"] == 3
    sensitive = cs.preview_cleaning(
        admin_ctx, region_dataset, "map_categories", {"column": "region", "mapping": {"north": "North"}}
    )
    assert sensitive["counts"] == {"north": 1} and sensitive["cells_changed"] == 1
    _apply(admin_ctx, region_dataset, "map_categories", params)
    assert _vals(vs.load_frame(admin_ctx, region_dataset)["region"]) == [
        "North",
        "North",
        "North",
        "South",
        "Eastern",
        None,
    ]
    assert vs.list_operations(admin_ctx, region_dataset)[0]["summary"]["counts"] == {"north": 2, "East": 1}


def test_map_categories_validation(admin_ctx, region_dataset):
    for params, message in [
        (
            {"column": "region", "mapping": {"north": "N", "NORTH": "X"}, "case_insensitive": True},
            "different results",
        ),
        ({"column": "region", "mapping": {"north": "  "}}, "blank"),
        ({"column": "region", "mapping": {}}, "at least one"),
        ({"column": "n", "mapping": {"1": "2"}}, "text"),
    ]:
        with pytest.raises(ValidationError, match=message):
            cs.preview_cleaning(admin_ctx, region_dataset, "map_categories", params)


@pytest.fixture
def dates_dataset(admin_ctx, factory) -> int:
    frame = pd.DataFrame(
        {"when": _text(["2024-01-05", "2024-02-17", " 2024-03-02 ", "", None, "not a date"])}
    )
    return factory.dataset(admin_ctx, frame)


def test_standardize_dates_reports_failures_and_abort_refuses(admin_ctx, dates_dataset):
    preview = cs.preview_cleaning(admin_ctx, dates_dataset, "standardize_dates", {"column": "when"})
    assert preview["details"]["date_format"] == "ISO8601"
    assert preview["failures"] == {"count": 1, "examples": [{"row": 6, "value": "not a date"}]}
    assert preview["cells_changed"] == 5 and not preview["can_apply"]
    assert any("could not be converted" in w for w in preview["warnings"])
    with pytest.raises(ValidationError, match="row 6") as error:
        _apply(admin_ctx, dates_dataset, "standardize_dates", {"column": "when"})
    assert error.value.details["failures"]["count"] == 1
    assert _version_count(admin_ctx, dates_dataset) == 1
    failed = vs.list_operations(admin_ctx, dates_dataset)[0]
    assert failed["status"] == "failed" and failed["operation_type"] == "standardize_dates"


def test_standardize_dates_set_missing_requires_confirm(admin_ctx, dates_dataset):
    params = {"column": "when", "on_failure": "set_missing"}
    with pytest.raises(ValidationError, match="confirm"):
        _apply(admin_ctx, dates_dataset, "standardize_dates", params)
    result = _apply(admin_ctx, dates_dataset, "standardize_dates", params, confirm=True)
    assert result["failures"]["count"] == 1 and result["stats"]["cells_changed"] == 5
    frame = vs.load_frame(admin_ctx, dates_dataset)
    assert str(frame["when"].dtype) == "datetime64[ns]"
    assert _vals(frame["when"]) == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-02-17"),
        pd.Timestamp("2024-03-02"),
        None,
        None,
        None,
    ]


def test_standardize_dates_formats_and_day_order(admin_ctx, factory):
    unambiguous = factory.dataset(admin_ctx, pd.DataFrame({"d": _text(["05/01/2024", "17/02/2024"])}))
    preview = cs.preview_cleaning(admin_ctx, unambiguous, "standardize_dates", {"column": "d"})
    assert preview["details"] == {"date_format": "%d/%m/%Y", "ambiguous": False}
    ambiguous = factory.dataset(admin_ctx, pd.DataFrame({"d": _text(["01/02/2024", "03/04/2024"])}))
    day_first = cs.preview_cleaning(
        admin_ctx, ambiguous, "standardize_dates", {"column": "d", "dayfirst": True}
    )
    assert day_first["details"]["ambiguous"] and day_first["sample"][0]["after"]["d"] == "2024-02-01T00:00:00"
    assert any("ambiguous" in w for w in day_first["warnings"])
    month_first = cs.preview_cleaning(
        admin_ctx, ambiguous, "standardize_dates", {"column": "d", "dayfirst": False}
    )
    assert month_first["sample"][0]["after"]["d"] == "2024-01-02T00:00:00"
    explicit = cs.preview_cleaning(
        admin_ctx, ambiguous, "standardize_dates", {"column": "d", "input_format": "%d/%m/%Y"}
    )
    assert explicit["sample"][1]["after"]["d"] == "2024-04-03T00:00:00"
    with pytest.raises(ValidationError, match="date format"):
        cs.preview_cleaning(admin_ctx, ambiguous, "standardize_dates", {"column": "d", "input_format": "%Q"})
    words = factory.dataset(admin_ctx, pd.DataFrame({"d": _text(["soon", "later"])}))
    with pytest.raises(ValidationError, match="could be read as dates"):
        cs.preview_cleaning(admin_ctx, words, "standardize_dates", {"column": "d"})


# ---------------------------------------------------------------------------
# Numeric and type conversion
# ---------------------------------------------------------------------------


def _converted(admin_ctx, factory, values, **params):
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"v": _text(values)}))
    result = _apply(admin_ctx, dataset_id, "convert_numeric", {"column": "v", **params}, confirm=True)
    return vs.load_frame(admin_ctx, dataset_id)["v"], result


def test_convert_numeric_integers_and_separators(admin_ctx, factory):
    series, result = _converted(admin_ctx, factory, ["1,234", "2,000", "7", None])
    assert str(series.dtype) == "Int64" and _vals(series) == [1234, 2000, 7, None]
    assert result["details"]["result_type"] == "integer" and result["stats"]["cells_changed"] == 3
    series, _ = _converted(
        admin_ctx, factory, ["1.234,5", "2,25", "10"], decimal_separator=",", thousands_separator="."
    )
    assert str(series.dtype) == "float64" and _vals(series) == [1234.5, 2.25, 10.0]
    series, _ = _converted(admin_ctx, factory, ["1'234'567", "12"], thousands_separator="'")
    assert _vals(series) == [1234567, 12]


def test_convert_numeric_currency_and_percent(admin_ctx, factory):
    series, result = _converted(admin_ctx, factory, ["$1,200", "$15.50", "USD 3"], strip_currency=True)
    assert str(series.dtype) == "float64" and _vals(series) == [1200.0, 15.5, 3.0]
    assert result["details"]["currency"] == "USD"
    series, _ = _converted(
        admin_ctx, factory, ["12.5%", "50%", "(5%)"], strip_percent=True, percent_as_fraction=True
    )
    assert _vals(series) == [0.125, 0.5, -0.05]
    series, _ = _converted(admin_ctx, factory, ["12.5%", "50%"], strip_percent=True)
    assert _vals(series) == [12.5, 50.0]
    series, _ = _converted(admin_ctx, factory, ["50%", "5%"], strip_percent=True)
    assert str(series.dtype) == "Int64" and _vals(series) == [50, 5]


def test_convert_numeric_failures(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"v": _text(["10", "ten", "20", "$5", "7%"])}))
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "convert_numeric", {"column": "v"})
    assert preview["failures"]["count"] == 3
    assert [e["row"] for e in preview["failures"]["examples"]] == [2, 4, 5]
    with pytest.raises(ValidationError, match="3 value"):
        _apply(admin_ctx, dataset_id, "convert_numeric", {"column": "v"})
    result = _apply(
        admin_ctx, dataset_id, "convert_numeric", {"column": "v", "on_failure": "set_missing"}, confirm=True
    )
    assert result["failures"]["count"] == 3
    assert _vals(vs.load_frame(admin_ctx, dataset_id)["v"]) == [10, None, 20, None, None]
    recorded = vs.list_operations(admin_ctx, dataset_id)[0]["summary"]["failures"]
    assert recorded["count"] == 3 and [e["row"] for e in recorded["examples"]] == [2, 4, 5]
    mixed = factory.dataset(admin_ctx, pd.DataFrame({"v": _text(["$5", "€5"])}))
    with pytest.raises(ValidationError, match="mixes currencies"):
        cs.preview_cleaning(admin_ctx, mixed, "convert_numeric", {"column": "v", "strip_currency": True})
    with pytest.raises(ValidationError, match="differ"):
        cs.preview_cleaning(admin_ctx, mixed, "convert_numeric", {"column": "v", "thousands_separator": "."})
    with pytest.raises(ValidationError, match="thousands"):
        cs.preview_cleaning(admin_ctx, mixed, "convert_numeric", {"column": "v", "thousands_separator": "_"})


def test_convert_numeric_warns_about_leading_zeros(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"v": _text(["007", "12"])}))
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "convert_numeric", {"column": "v"})
    assert any("leading zeros" in w for w in preview["warnings"])


def test_convert_types_float_to_integer_only_when_whole(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"x": [1.0, 2.5, 3.0, np.nan]}))
    preview = cs.preview_cleaning(
        admin_ctx, dataset_id, "convert_types", {"column": "x", "target": "integer"}
    )
    assert preview["failures"] == {"count": 1, "examples": [{"row": 2, "value": 2.5}]}
    with pytest.raises(ValidationError, match="could not be converted"):
        _apply(admin_ctx, dataset_id, "convert_types", {"column": "x", "target": "integer"})
    _apply(
        admin_ctx,
        dataset_id,
        "convert_types",
        {"column": "x", "target": "integer", "on_failure": "set_missing"},
        confirm=True,
    )
    frame = vs.load_frame(admin_ctx, dataset_id)
    assert str(frame["x"].dtype) == "Int64" and _vals(frame["x"]) == [1, None, 3, None]
    whole = factory.dataset(admin_ctx, pd.DataFrame({"x": [1.0, -2.0]}))
    _apply(admin_ctx, whole, "convert_types", {"column": "x", "target": "integer"})
    assert _vals(vs.load_frame(admin_ctx, whole)["x"]) == [1, -2]


def test_convert_types_matrix(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "i": pd.array([12, 0, None], dtype="Int64"),
            "flag": _text(["yes", "No", "maybe"]),
            "num": _text(["12", "12.0", "x"]),
            "b": pd.array([True, False, None], dtype="boolean"),
            "when": pd.Series(
                [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-02-01 10:30:00"), pd.NaT],
                dtype="datetime64[ns]",
            ),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)

    def converted(column: str, target: str, **extra):
        preview = cs.preview_cleaning(
            admin_ctx,
            dataset_id,
            "convert_types",
            {"column": column, "target": target, **extra},
            sample_rows=3,
        )
        return [s["after"][column] for s in preview["sample"]], preview["failures"]["count"]

    assert converted("i", "float") == ([12.0, 0.0], 0)
    assert converted("i", "text") == (["12", "0"], 0)
    assert converted("i", "boolean") == ([None, False], 1)
    assert converted("flag", "boolean") == ([True, False, None], 1)
    assert converted("num", "integer") == ([12, 12, None], 1)
    assert converted("num", "float") == ([12.0, 12.0, None], 1)
    assert converted("b", "integer") == ([1, 0], 0)
    assert converted("b", "text") == (["true", "false"], 0)
    assert converted("when", "text") == (["2024-01-05 00:00:00", "2024-02-01 10:30:00"], 0)
    for params, message in [
        ({"column": "when", "target": "integer"}, "not supported"),
        ({"column": "i", "target": "integer"}, "already"),
        ({"column": "i", "target": "money"}, "target"),
    ]:
        with pytest.raises(ValidationError, match=message):
            cs.preview_cleaning(admin_ctx, dataset_id, "convert_types", params)
    _apply(admin_ctx, dataset_id, "convert_types", {"column": "b", "target": "integer"})
    assert str(vs.load_frame(admin_ctx, dataset_id)["b"].dtype) == "Int64"


def test_convert_types_text_to_datetime(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"d": _text(["31/12/2023", "01/01/2024"])}))
    _apply(admin_ctx, dataset_id, "convert_types", {"column": "d", "target": "datetime"})
    assert _vals(vs.load_frame(admin_ctx, dataset_id)["d"]) == [
        pd.Timestamp("2023-12-31"),
        pd.Timestamp("2024-01-01"),
    ]


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


def test_outlier_retain_reports_and_never_changes_data(admin_ctx, factory, numbers_frame):
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "handle_outliers", {"column": "amount"})
    bounds = preview["details"]["bounds"]
    assert (bounds["q1"], bounds["q3"], bounds["iqr"], bounds["lower"], bounds["upper"]) == (
        11.0,
        13.0,
        2.0,
        8.0,
        16.0,
    )
    assert preview["details"]["outliers"] == 1 and preview["cells_changed"] == 0 and not preview["can_apply"]
    assert [s["row"] for s in preview["sample"]] == [5] and preview["sample"][0]["after"]["amount"] == 100.0
    with pytest.raises(ValidationError, match="retain"):
        _apply(admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "action": "retain"})
    assert _version_count(admin_ctx, dataset_id) == 1
    pd.testing.assert_frame_equal(vs.load_frame(admin_ctx, dataset_id), numbers_frame)


def test_outlier_cap_bounds_are_exact(admin_ctx, factory, numbers_frame):
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    result = _apply(admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "action": "cap"})
    assert result["stats"]["cells_changed"] == 1
    assert vs.load_frame(admin_ctx, dataset_id)["amount"].tolist() == [10.0, 12.0, 11.0, 13.0, 16.0]
    result = _apply(
        admin_ctx, dataset_id, "handle_outliers", {"column": "qty", "action": "cap", "threshold": 1.25}
    )
    assert result["details"]["integer_bounds"] == {"lower": 9, "upper": 15}
    qty = vs.load_frame(admin_ctx, dataset_id)["qty"]
    assert str(qty.dtype) == "Int64" and qty.tolist() == [10, 12, 11, 13, 15]


def test_outlier_exclude_needs_confirm(admin_ctx, factory, numbers_frame):
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    params = {"column": "amount", "action": "exclude"}
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "handle_outliers", params)
    assert preview["rows_removed"] == 1 and preview["requires_confirmation"]
    with pytest.raises(ValidationError, match="confirm"):
        _apply(admin_ctx, dataset_id, "handle_outliers", params)
    result = _apply(admin_ctx, dataset_id, "handle_outliers", params, confirm=True)
    assert result["rows_after"] == 4 and result["stats"]["rows_removed"] == 1
    assert vs.load_frame(admin_ctx, dataset_id)["label"].tolist() == ["a", "b", "c", "d"]


def test_outlier_flag_adds_review_column(admin_ctx, factory, numbers_frame):
    numbers_frame.loc[1, "amount"] = np.nan
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    result = _apply(admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "action": "flag"})
    assert result["columns_added"] == ["amount_outlier"] and result["stats"]["cells_changed"] == 1
    frame = vs.load_frame(admin_ctx, dataset_id)
    assert str(frame["amount_outlier"].dtype) == "boolean"
    assert _vals(frame["amount_outlier"]) == [False, None, False, False, True]
    assert frame["amount"].tolist()[4] == 100.0
    with pytest.raises(ValidationError, match="already exists"):
        cs.preview_cleaning(admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "action": "flag"})


def test_outlier_zscore_and_validation(admin_ctx, factory, numbers_frame):
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    values = numbers_frame["amount"].tolist()
    mean, std = statistics.mean(values), statistics.stdev(values)
    preview = cs.preview_cleaning(
        admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "method": "zscore", "threshold": 1.5}
    )
    bounds = preview["details"]["bounds"]
    assert bounds["mean"] == pytest.approx(mean, rel=1e-12) and bounds["std"] == pytest.approx(std, rel=1e-12)
    assert bounds["upper"] == pytest.approx(mean + 1.5 * std, rel=1e-12)
    assert preview["details"]["outliers"] == 1
    default = cs.preview_cleaning(
        admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "method": "zscore"}
    )
    assert default["params"]["threshold"] == 3.0 and default["details"]["outliers"] == 0
    for params, message in [
        ({"column": "label"}, "number"),
        ({"column": "amount", "threshold": 0}, "greater than 0"),
        ({"column": "amount", "threshold": "big"}, "number"),
        ({"column": "amount", "method": "mad"}, "method"),
    ]:
        with pytest.raises(ValidationError, match=message):
            cs.preview_cleaning(admin_ctx, dataset_id, "handle_outliers", params)


def test_outlier_report_is_read_only(admin_ctx, factory, numbers_frame):
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    report = cs.outlier_report(admin_ctx, dataset_id, "qty")
    assert report["bounds"]["lower"] == 8.0 and report["bounds"]["upper"] == 16.0
    assert report["count"] == 1 and report["non_missing"] == 5 and report["percent"] == 20.0
    assert report["sample"] == [{"row": 5, "values": {"label": "e", "amount": 100.0, "qty": 100}}]
    constant = factory.dataset(admin_ctx, pd.DataFrame({"x": [5.0, 5.0, 5.0]}))
    flat = cs.outlier_report(admin_ctx, constant, "x", method="zscore")
    assert flat["count"] == 0 and flat["warnings"]
    assert _version_count(admin_ctx, dataset_id) == 1


# ---------------------------------------------------------------------------
# Invalid values and constraints
# ---------------------------------------------------------------------------


def test_fix_invalid_emails(admin_ctx, factory):
    frame = pd.DataFrame({"email": _text(["a@example.com", "bad@", " c@d.org ", None, "x@y", "   "])})
    dataset_id = factory.dataset(admin_ctx, frame)
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "fix_invalid_values", {"column": "email"})
    assert preview["failures"] == {
        "count": 2,
        "examples": [{"row": 2, "value": "bad@"}, {"row": 5, "value": "x@y"}],
    }
    assert preview["cells_changed"] == 0 and not preview["can_apply"]
    with pytest.raises(ValidationError, match="report"):
        _apply(admin_ctx, dataset_id, "fix_invalid_values", {"column": "email"})
    params = {"column": "email", "action": "set_missing"}
    with pytest.raises(ValidationError, match="confirm"):
        _apply(admin_ctx, dataset_id, "fix_invalid_values", params)
    result = _apply(admin_ctx, dataset_id, "fix_invalid_values", params, confirm=True)
    assert result["stats"]["cells_changed"] == 2 and result["counts"] == {"invalid": 2}
    assert _vals(vs.load_frame(admin_ctx, dataset_id)["email"]) == [
        "a@example.com",
        None,
        " c@d.org ",
        None,
        None,
        "   ",
    ]


def test_fix_invalid_values_custom_regex_and_safety(admin_ctx, factory):
    dataset_id = factory.dataset(
        admin_ctx, pd.DataFrame({"code": _text(["ABC-123", "abc-123", "XYZ-999", "ABC-1234"])})
    )
    preview = cs.preview_cleaning(
        admin_ctx,
        dataset_id,
        "fix_invalid_values",
        {"column": "code", "pattern": "regex", "regex": r"[A-Z]{3}-\d{3}"},
    )
    assert preview["failures"]["count"] == 2 and [e["row"] for e in preview["failures"]["examples"]] == [2, 4]
    for regex, message in [
        ("(a+)+$", "nests repetitions"),
        (r"(\w+\s?)*", "nests repetitions"),
        (r"(?:x|(?:ab)*)+", "nests repetitions"),
        ("a" * 201, "200"),
        ("(", "not valid"),
        (None, "regular expression"),
    ]:
        with pytest.raises(ValidationError, match=message):
            cs.preview_cleaning(
                admin_ctx,
                dataset_id,
                "fix_invalid_values",
                {"column": "code", "pattern": "regex", "regex": regex},
            )
    with pytest.raises(ValidationError, match="only used"):
        cs.preview_cleaning(admin_ctx, dataset_id, "fix_invalid_values", {"column": "code", "regex": "x"})
    assert cs.compile_safe_regex(r"^\d{3}-\d+$").fullmatch("123-45")


def test_validate_constraints_report(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "id": pd.array([1, 2, 2, 4], dtype="Int64"),
            "amount": [5.0, -1.0, 50.0, np.nan],
            "status": _text(["open", "closed", "weird", "open"]),
            "code": _text(["AB1", "XX", "AB2", "AB3"]),
            "when": pd.to_datetime(["2024-01-01", "2023-06-01", "2024-02-01", "2024-03-01"]),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    rules = [
        {"column": "id", "unique": True},
        {"column": "amount", "not_null": True, "min": 0, "max": 40},
        {"column": "status", "allowed_values": ["open", "closed"]},
        {"column": "code", "regex": r"AB\d"},
        {"column": "when", "min": "2024-01-01"},
    ]
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "validate_constraints", {"rules": rules})
    by_column = {item["column"]: item for item in preview["constraints"]}
    assert by_column["id"]["checks"] == [
        {"check": "unique", "count": 2, "examples": [{"row": 2, "value": 2}, {"row": 3, "value": 2}]}
    ]
    assert [(c["check"], c["count"]) for c in by_column["amount"]["checks"]] == [
        ("not_null", 1),
        ("min", 1),
        ("max", 1),
    ]
    assert by_column["status"]["violations"] == 1 and by_column["code"]["checks"][0]["examples"] == [
        {"row": 2, "value": "XX"}
    ]
    assert by_column["when"]["violations"] == 1
    assert by_column["amount"]["rule"] == {"column": "amount", "not_null": True, "min": 0, "max": 40}
    assert (
        preview["failures"]["count"] == 8 and not preview["passed"] and preview["rows_with_violations"] == 3
    )
    assert preview["cells_changed"] == 0 and not preview["can_apply"]
    with pytest.raises(ValidationError, match="read-only"):
        _apply(admin_ctx, dataset_id, "validate_constraints", {"rules": rules})
    assert _version_count(admin_ctx, dataset_id) == 1
    for bad, message in [
        ([{"column": "status", "min": 1}], "number or date"),
        ([{"column": "status"}], "no checks"),
        ([{"column": "amount", "allowed_values": ["x"]}], "number"),
        ([{"column": "code", "regex": "(a*)*"}], "nests"),
        ([], "at least one"),
    ]:
        with pytest.raises(ValidationError, match=message):
            cs.preview_cleaning(admin_ctx, dataset_id, "validate_constraints", {"rules": bad})
    clean = cs.preview_cleaning(
        admin_ctx, dataset_id, "validate_constraints", {"rules": [{"column": "code", "not_null": True}]}
    )
    assert clean["passed"] and clean["failures"]["count"] == 0


# ---------------------------------------------------------------------------
# Detection reports
# ---------------------------------------------------------------------------


def test_detect_casing_issues(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "region": _text(["North", "North", "north", "NORTH", "South", None]),
            "code": _text(["a", "b", "c", "d", "e", "f"]),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    findings = cs.detect_casing_issues(admin_ctx, dataset_id)
    assert findings == [
        {
            "column": "region",
            "kind": "casing_variants",
            "variants": [
                {"value": "North", "count": 2},
                {"value": "NORTH", "count": 1},
                {"value": "north", "count": 1},
            ],
            "total": 4,
            "suggested": "North",
            "status": "for review",
        }
    ]


def test_identify_redundant_columns(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "a": pd.array([1, 2, 3, 4], dtype="Int64"),
            "b": pd.array([1, 2, 3, 4], dtype="Int64"),
            "c": [2.0, 4.0, 6.0, 8.001],
            "d": _text(["x", "Y", "z", "w"]),
            "e": _text(["X", "y", "Z", "W"]),
            "f": _text(["k", "k", None, "k"]),
            "g": _text([None, None, None, None]),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    findings = cs.identify_redundant_columns(admin_ctx, dataset_id)
    kinds = {(item["kind"], tuple(item["columns"])) for item in findings}
    assert kinds == {
        ("identical", ("a", "b")),
        ("identical_ignoring_case", ("d", "e")),
        ("constant", ("f",)),
        ("empty", ("g",)),
        ("high_correlation", ("a", "c")),
        ("high_correlation", ("b", "c")),
    }
    constant = next(item for item in findings if item["kind"] == "constant")
    assert constant["value"] == "k" and constant["missing"] == 1
    assert all(item["status"] == "for review" for item in findings)
    assert _version_count(admin_ctx, dataset_id) == 1


# ---------------------------------------------------------------------------
# Streaming, versions, previews
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": _text([" a ", "b", None, "  c", "d ", "e", "f"]),
            "amount": [10.0, 12.0, np.nan, 11.0, 13.0, 100.0, 12.5],
            "qty": pd.array([1, 2, None, 3, 4, 5, 60], dtype="Int64"),
            "when": _text(
                ["2024-01-05", "2024-02-01", None, "bad", "2024-03-01", "2024-04-01", "2024-05-01"]
            ),
            "num": _text(["1", "2.5", None, "x", "4", "5", "6"]),
            "blank": _text([None] * 7),
        }
    )


STREAMING_CASES = [
    ("trim_whitespace", {}, False),
    ("map_categories", {"column": "name", "mapping": {"b": "B"}}, False),
    ("handle_outliers", {"column": "amount", "action": "cap"}, False),
    ("handle_outliers", {"column": "qty", "action": "cap"}, False),
    ("handle_outliers", {"column": "amount", "action": "exclude"}, True),
    ("handle_outliers", {"column": "amount", "action": "flag"}, False),
    ("convert_numeric", {"column": "num", "on_failure": "set_missing"}, True),
    ("standardize_dates", {"column": "when", "on_failure": "set_missing"}, True),
    ("remove_empty", {}, True),
    ("drop_columns", {"columns": ["blank"]}, True),
]


@pytest.mark.parametrize(("operation", "params", "confirm"), STREAMING_CASES)
def test_streaming_apply_equals_in_memory_result(
    admin_ctx, factory, app_settings, mixed_frame, operation, params, confirm
):
    whole = factory.dataset(admin_ctx, mixed_frame, "whole")
    batched = factory.dataset(admin_ctx, mixed_frame, "batched")
    preview = cs.preview_cleaning(admin_ctx, whole, operation, params)
    single = _apply(admin_ctx, whole, operation, params, confirm=confirm)
    set_settings(app_settings.with_overrides(parquet_batch_rows=2))
    streamed = _apply(admin_ctx, batched, operation, params, confirm=confirm)
    pd.testing.assert_frame_equal(vs.load_frame(admin_ctx, whole), vs.load_frame(admin_ctx, batched))
    for key in ("stats", "failures", "counts", "columns_added", "columns_removed", "quality_after"):
        assert single[key] == streamed[key], key
    assert single["stats"]["cells_changed"] == preview["cells_changed"]
    assert single["stats"]["rows_removed"] == preview["rows_removed"]
    assert single["failures"] == preview["failures"]
    batched_preview = cs.preview_cleaning(
        admin_ctx, factory.dataset(admin_ctx, mixed_frame, "again"), operation, params
    )
    assert (
        batched_preview["sample"] == preview["sample"]
        and batched_preview["quality_after"] == preview["quality_after"]
    )


@pytest.mark.parametrize(
    ("operation", "params"),
    [
        ("trim_whitespace", {}),
        ("remove_control_chars", {}),
        ("normalize_case", {"mode": "upper"}),
        ("rename_columns", {"mapping": {"name": "full_name"}}),
        ("remove_empty", {}),
        ("drop_columns", {"columns": ["blank"]}),
        ("map_categories", {"column": "name", "mapping": {"b": "B"}}),
        ("standardize_dates", {"column": "when"}),
        ("convert_numeric", {"column": "num"}),
        ("convert_types", {"column": "qty", "target": "float"}),
        ("handle_outliers", {"column": "amount", "action": "exclude"}),
        ("fix_invalid_values", {"column": "name", "action": "set_missing"}),
        ("validate_constraints", {"rules": [{"column": "qty", "unique": True}]}),
    ],
)
def test_previews_never_change_data(admin_ctx, factory, mixed_frame, operation, params):
    dataset_id = factory.dataset(admin_ctx, mixed_frame)
    cs.preview_cleaning(admin_ctx, dataset_id, operation, params)
    assert _version_count(admin_ctx, dataset_id) == 1
    assert [op["operation_type"] for op in vs.list_operations(admin_ctx, dataset_id)] == ["import"]
    pd.testing.assert_frame_equal(
        vs.load_frame(admin_ctx, dataset_id),
        vs.load_frame(
            admin_ctx, dataset_id, version_id=vs.list_versions(admin_ctx, dataset_id)[0]["version_id"]
        ),
    )


def test_successive_applies_keep_earlier_versions_restorable(admin_ctx, factory, mixed_frame):
    dataset_id = factory.dataset(admin_ctx, mixed_frame)
    original = vs.load_frame(admin_ctx, dataset_id)
    _apply(admin_ctx, dataset_id, "trim_whitespace", {})
    trimmed = vs.load_frame(admin_ctx, dataset_id)
    third = _apply(
        admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "action": "exclude"}, confirm=True
    )
    assert third["version_number"] == 3
    versions = {v["version_number"]: v["version_id"] for v in vs.list_versions(admin_ctx, dataset_id)}
    pd.testing.assert_frame_equal(vs.load_frame(admin_ctx, dataset_id, version_id=versions[2]), trimmed)
    vs.restore_version(admin_ctx, dataset_id, versions[1])
    pd.testing.assert_frame_equal(vs.load_frame(admin_ctx, dataset_id), original)
    assert [op["operation_type"] for op in vs.list_operations(admin_ctx, dataset_id)] == [
        "restore",
        "handle_outliers",
        "trim_whitespace",
        "import",
    ]


def test_dataset_changed_after_prepare_raises_conflict(admin_ctx, factory, numbers_frame, monkeypatch):
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    original_prepare = cs._prepare

    def prepare_then_change(spec, ref, params, guard):
        prepared = original_prepare(spec, ref, params, guard)
        vs.apply_frame_operation(
            admin_ctx,
            dataset_id,
            operation_type="other",
            parameters={},
            transform=lambda df: (df.iloc[1:], {}, {}),
        )
        return prepared

    monkeypatch.setattr(cs, "_prepare", prepare_then_change)
    with pytest.raises(ConflictError):
        _apply(admin_ctx, dataset_id, "handle_outliers", {"column": "amount", "action": "cap"})
    assert _version_count(admin_ctx, dataset_id) == 2


def test_overflowing_numbers_are_failures_not_infinities(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"v": _text(["1e400", "2.5", "1e999999999"])}))
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "convert_numeric", {"column": "v"})
    assert preview["failures"]["count"] == 2 and [e["row"] for e in preview["failures"]["examples"]] == [1, 3]
    as_int = cs.preview_cleaning(admin_ctx, dataset_id, "convert_types", {"column": "v", "target": "integer"})
    assert as_int["failures"]["count"] == 3
    big = factory.dataset(
        admin_ctx, pd.DataFrame({"v": _text(["9223372036854775807", "9223372036854775808"])})
    )
    preview = cs.preview_cleaning(admin_ctx, big, "convert_numeric", {"column": "v"})
    assert preview["failures"]["examples"] == [{"row": 2, "value": "9223372036854775808"}]


def test_zero_row_dataset_preview_and_apply(admin_ctx, factory):
    frame = pd.DataFrame({"name": _text([]), "qty": pd.array([], dtype="Int64")})
    dataset_id = factory.dataset(admin_ctx, frame)
    for operation, params in [
        ("trim_whitespace", {}),
        ("convert_types", {"column": "qty", "target": "float"}),
    ]:
        preview = cs.preview_cleaning(admin_ctx, dataset_id, operation, params)
        assert preview["rows_before"] == preview["rows_after"] == 0 and preview["cells_changed"] == 0
        result = _apply(admin_ctx, dataset_id, operation, params)
        assert result["rows_after"] == 0
    loaded = vs.load_frame(admin_ctx, dataset_id)
    assert len(loaded) == 0 and str(loaded["qty"].dtype) == "float64"


def test_convert_types_text_to_datetime_uses_org_day_order(admin_ctx, factory, org_id):
    import json

    from database import session_scope
    from database.models import Organization

    with session_scope() as db:
        db.get(Organization, org_id).settings_json = json.dumps({"date_dayfirst": True})
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"d": _text(["01/02/2024", "03/04/2024"])}))
    preview = cs.preview_cleaning(
        admin_ctx, dataset_id, "convert_types", {"column": "d", "target": "datetime"}
    )
    assert (
        preview["params"]["dayfirst"] is True and preview["sample"][0]["after"]["d"] == "2024-02-01T00:00:00"
    )


def test_low_memory_blocks_column_preparation(admin_ctx, factory, numbers_frame, monkeypatch):
    dataset_id = factory.dataset(admin_ctx, numbers_frame)
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: 10)
    with pytest.raises(ResourceLimitError):
        cs.preview_cleaning(admin_ctx, dataset_id, "handle_outliers", {"column": "amount"})


# ---------------------------------------------------------------------------
# Plans, permissions, tenants
# ---------------------------------------------------------------------------


ADVANCED_CALLS = [
    ("map_categories", {"column": "region", "mapping": {"North": "N"}}),
    ("standardize_dates", {"column": "region"}),
    ("convert_numeric", {"column": "region"}),
    ("convert_types", {"column": "quantity", "target": "float"}),
    ("handle_outliers", {"column": "revenue", "action": "cap"}),
    ("fix_invalid_values", {"column": "region"}),
    ("validate_constraints", {"rules": [{"column": "region", "not_null": True}]}),
]


def test_basic_plan_gets_standard_but_not_advanced_cleaning(factory, sales_frame):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    dataset_id = factory.dataset(ctx, sales_frame)
    for operation, params in ADVANCED_CALLS:
        with pytest.raises(FeatureNotAvailableError):
            cs.preview_cleaning(ctx, dataset_id, operation, params)
        with pytest.raises(FeatureNotAvailableError):
            _apply(ctx, dataset_id, operation, params, confirm=True)
    assert cs.preview_cleaning(ctx, dataset_id, "normalize_case", {"mode": "upper"})["cells_changed"] == 6
    assert _apply(ctx, dataset_id, "normalize_case", {"mode": "upper"})["version_number"] == 2
    assert cs.detect_casing_issues(ctx, dataset_id) == []
    assert isinstance(cs.identify_redundant_columns(ctx, dataset_id), list)
    assert cs.outlier_report(ctx, dataset_id, "revenue")["count"] == 0


def test_analyst_without_clean_data_can_preview_but_not_apply(factory, org_id, admin_ctx, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    analyst_id = factory.user(org_id, "analyst")
    factory.grant(dataset_id, analyst_id)
    analyst = factory.ctx(analyst_id)
    assert cs.preview_cleaning(analyst, dataset_id, "trim_whitespace", {})["cells_changed"] == 0
    with pytest.raises(PermissionDeniedError) as error:
        _apply(analyst, dataset_id, "trim_whitespace", {})
    assert error.type is PermissionDeniedError
    with pytest.raises(PermissionDeniedError) as error:
        _apply(analyst, dataset_id, "handle_outliers", {"column": "revenue", "action": "cap"})
    assert error.type is PermissionDeniedError
    cleaner_id = factory.user(org_id, "analyst", permissions={"clean_data"})
    factory.grant(dataset_id, cleaner_id)
    assert (
        _apply(factory.ctx(cleaner_id), dataset_id, "normalize_case", {"mode": "lower"})["version_number"]
        == 2
    )


def test_other_tenants_and_ungranted_analysts_get_not_found(factory, org_id, admin_ctx, sales_frame):
    dataset_id = factory.dataset(admin_ctx, sales_frame)
    outsider = factory.org_admin(factory.organization())
    ungranted = factory.analyst(org_id, permissions={"clean_data"})
    for ctx in (outsider, ungranted):
        with pytest.raises(NotFoundError):
            cs.preview_cleaning(ctx, dataset_id, "trim_whitespace", {})
        with pytest.raises(NotFoundError):
            _apply(ctx, dataset_id, "trim_whitespace", {})
        with pytest.raises(NotFoundError):
            cs.detect_casing_issues(ctx, dataset_id)
        with pytest.raises(NotFoundError):
            cs.identify_redundant_columns(ctx, dataset_id)
        with pytest.raises(NotFoundError):
            cs.outlier_report(ctx, dataset_id, "revenue")
    assert _version_count(admin_ctx, dataset_id) == 1


CLEANING_PATH = Path(__file__).resolve().parents[1] / "services" / "cleaning_service.py"


def test_cleaning_service_imports_on_python_310_layout(monkeypatch):
    """Python 3.10 has no re._parser; the regex parser is sre_parse. The ReDoS guard must still work."""
    parser_name = "re._parser" if sys.version_info >= (3, 11) else "sre_parse"
    parser = sys.modules.get(parser_name) or importlib.import_module(parser_name)
    monkeypatch.setitem(sys.modules, "sre_parse", parser)
    if sys.version_info >= (3, 11):  # on 3.10 this layout is the real one
        monkeypatch.setitem(sys.modules, "re._parser", None)
        monkeypatch.delattr(re, "_parser", raising=False)
    name = "_cleaning_service_py310_layout"
    spec = importlib.util.spec_from_file_location(name, CLEANING_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)  # ImportError on the current code
    with pytest.raises(ValidationError):
        module.compile_safe_regex("(a+)+")
    assert module.compile_safe_regex(r"\d+").fullmatch("123")


# SEC1: user regular expressions run with a time limit and release the GIL.
# This pattern has no nested repeat, so the syntax check accepts it, but it
# backtracks combinatorially on a run of commas without a final "P".
SLOW_REGEX = ".*?," * 11 + "P"
SLOW_VALUE = "1," * 30


@pytest.mark.parametrize("regex", ["(a|a)*b", "(.|.)*!"])
def test_user_regex_cannot_backtrack_exponentially(factory, org_id, admin_ctx, regex):
    # An analyst with only default permissions (no clean_data) can preview.
    analyst = factory.analyst(org_id)
    value = "a" * 25 if regex.startswith("(a") else "North Region Office 12345"
    dataset_id = factory.dataset(admin_ctx, pd.DataFrame({"code": [value]}))
    factory.grant(dataset_id, analyst.user_id)
    started = time.perf_counter()
    try:
        cs.preview_cleaning(
            analyst,
            dataset_id,
            "fix_invalid_values",
            {"column": "code", "pattern": "regex", "regex": regex, "action": "report"},
            guard=resource_manager.OperationGuard(timeout_seconds=1),
        )
    except ValidationError:
        return  # rejected up front: fine
    elapsed = time.perf_counter() - started
    assert elapsed < 1.5, f"user regex {regex!r} ran {elapsed:.1f}s on a 25-char value"


def _slow_dataset(factory, admin_ctx, values: list[str]) -> int:
    return factory.dataset(admin_ctx, pd.DataFrame({"code": _text(values)}))


def test_slow_user_regex_in_pattern_validation_is_stopped(factory, admin_ctx):
    dataset_id = _slow_dataset(factory, admin_ctx, ["AB1", SLOW_VALUE, "AB2"])
    started = time.perf_counter()
    with pytest.raises((ValidationError, OperationTimeoutError)) as error:
        cs.preview_cleaning(
            admin_ctx,
            dataset_id,
            "fix_invalid_values",
            {"column": "code", "pattern": "regex", "regex": SLOW_REGEX},
            guard=resource_manager.OperationGuard(timeout_seconds=1),
        )
    assert time.perf_counter() - started < 1.5
    assert "too long" in error.value.user_message


def test_slow_user_regex_in_constraint_rule_is_stopped(factory, admin_ctx):
    dataset_id = _slow_dataset(factory, admin_ctx, ["AB1", SLOW_VALUE])
    started = time.perf_counter()
    with pytest.raises((ValidationError, OperationTimeoutError)) as error:
        cs.preview_cleaning(
            admin_ctx,
            dataset_id,
            "validate_constraints",
            {"rules": [{"column": "code", "regex": SLOW_REGEX}]},
            guard=resource_manager.OperationGuard(timeout_seconds=1),
        )
    assert time.perf_counter() - started < 1.5
    assert "too long" in error.value.user_message


def test_slow_user_regex_aborts_apply_without_a_new_version(factory, admin_ctx):
    dataset_id = _slow_dataset(factory, admin_ctx, ["AB1", SLOW_VALUE, "AB2"])
    started = time.perf_counter()
    with pytest.raises((ValidationError, OperationTimeoutError)):
        _apply(
            admin_ctx,
            dataset_id,
            "fix_invalid_values",
            {"column": "code", "pattern": "regex", "regex": SLOW_REGEX, "action": "set_missing"},
            confirm=True,
            guard=resource_manager.OperationGuard(timeout_seconds=1),
        )
    assert time.perf_counter() - started < 1.5
    assert _version_count(admin_ctx, dataset_id) == 1


def test_many_slow_values_hit_the_operation_timeout(factory, admin_ctx):
    # Each value alone stays under the per-value budget; together they exceed the guard.
    values = ["1," * 21 + "x" * i for i in range(30)]
    dataset_id = _slow_dataset(factory, admin_ctx, values)
    started = time.perf_counter()
    with pytest.raises(OperationTimeoutError):
        cs.preview_cleaning(
            admin_ctx,
            dataset_id,
            "fix_invalid_values",
            {"column": "code", "pattern": "regex", "regex": SLOW_REGEX},
            guard=resource_manager.OperationGuard(timeout_seconds=0.5),
        )
    assert time.perf_counter() - started < 1.5


def test_user_regex_match_lets_other_threads_run(factory, admin_ctx, monkeypatch):
    """A long match must release the GIL so other sessions in the process keep working."""
    monkeypatch.setattr(cs, "REGEX_MATCH_SECONDS", 1.5)
    dataset_id = _slow_dataset(factory, admin_ctx, [SLOW_VALUE])
    ticks = 0
    stop = threading.Event()

    def heartbeat() -> None:
        nonlocal ticks
        while not stop.wait(0.1):
            ticks += 1

    thread = threading.Thread(target=heartbeat, daemon=True)
    started = time.perf_counter()
    thread.start()
    try:
        with pytest.raises((ValidationError, OperationTimeoutError)):
            cs.preview_cleaning(
                admin_ctx,
                dataset_id,
                "fix_invalid_values",
                {"column": "code", "pattern": "regex", "regex": SLOW_REGEX},
                guard=resource_manager.OperationGuard(timeout_seconds=10),
            )
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        thread.join()
    assert elapsed < 3.0
    expected = elapsed / 0.1
    assert ticks >= expected * 0.5, f"heartbeat ticked {ticks} times in {elapsed:.1f}s"


def test_user_regex_refuses_overlong_values(factory, admin_ctx):
    dataset_id = _slow_dataset(factory, admin_ctx, ["AB1", "A" * (cs.MAX_REGEX_VALUE_LENGTH + 1)])
    with pytest.raises(ValidationError, match="longer than"):
        cs.preview_cleaning(
            admin_ctx,
            dataset_id,
            "fix_invalid_values",
            {"column": "code", "pattern": "regex", "regex": "A+"},
        )
    # The built-in e-mail check is linear and keeps working on long values.
    preview = cs.preview_cleaning(admin_ctx, dataset_id, "fix_invalid_values", {"column": "code"})
    assert preview["failures"]["count"] == 2
