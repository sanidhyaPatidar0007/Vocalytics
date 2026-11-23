"""Unit tests for the column typing contract (services/type_inference.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.errors import ValidationError
from services.type_inference import (
    ParseOptions,
    arrow_schema,
    convert_frame,
    infer_and_convert,
    infer_types,
    normalize_headers,
    numeric_patterns,
    parse_numeric_token,
)

OPTIONS = ParseOptions(missing_tokens=("NA", "N/A", "null", "None", "NaN", "nan", "#N/A"))


def _raw(**columns: list) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in columns.items()})


def _types(frame: pd.DataFrame, options: ParseOptions = OPTIONS) -> dict[str, str]:
    return {name: inf.logical_type for name, inf in infer_types([frame], options).items()}


def _chunks(frame: pd.DataFrame, size: int) -> list[pd.DataFrame]:
    return [frame.iloc[i : i + size] for i in range(0, len(frame), size)]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["1200.00", "980.00"], "float"),
        (["1", "2", "3"], "integer"),
        (["1", None, "", "  ", "4"], "integer"),
        (["0.5", "N/A", "2"], "float"),
        (["1e3", "2"], "float"),
        (["(1,200.00)", "5.5"], "float"),
        (["true", "False", "YES", "n"], "boolean"),
        (["2024-01-05", "2024-12-31"], "datetime"),
        (["2024-01-05", "2024-01-05 10:30:00"], "datetime"),
        (["00123", "456"], "text"),
        (["Alpha", "Beta"], "text"),
        ([None, "", "N/A"], "text"),
    ],
)
def test_logical_types(values, expected):
    assert _types(_raw(col=values))["col"] == expected


def test_decimal_column_is_never_downcast_even_when_whole():
    typed, inference = infer_and_convert(_raw(revenue=["1200.00", "980.00", "1500.00"]), OPTIONS)
    assert inference["revenue"].logical_type == "float"
    assert str(typed["revenue"].dtype) == "float64"
    assert typed["revenue"].tolist() == [1200.0, 980.0, 1500.0]


def test_inference_is_identical_for_every_chunk_size():
    frame = _raw(
        a=["1", "2", "3", "4.5", "6"],  # the decimal only appears in the 4th row
        b=["x", "y", "007", "z", "w"],
        c=["2024-01-01", "2024-02-01", "", "2024-03-01", "2024-04-01"],
        d=["1", "", "", "", ""],  # chunks of size 1 or 2 contain only missing values
    )
    reference = {k: v.to_dict() for k, v in infer_types([frame], OPTIONS).items()}
    for size in (1, 2, 3, 5):
        chunked = {k: v.to_dict() for k, v in infer_types(_chunks(frame, size), OPTIONS).items()}
        assert chunked == reference, f"chunk size {size} changed the inference"
    assert reference["a"]["logical_type"] == "float"
    assert reference["d"]["logical_type"] == "integer"


def test_all_missing_chunk_converts_against_the_fixed_schema():
    frame = _raw(qty=["", "", "5", "7"], when=["", "", "2024-01-01", "2024-01-02"])
    inferences = infer_types(_chunks(frame, 2), OPTIONS)
    schema = arrow_schema(["qty", "when"], inferences)
    import pyarrow as pa

    first = convert_frame(frame.iloc[:2], inferences, OPTIONS)  # entirely missing
    assert str(first["qty"].dtype) == "Int64" and first["qty"].isna().all()
    table = pa.Table.from_pandas(first, schema=schema, preserve_index=False)
    assert table.schema.field("qty").type == pa.int64()
    assert table.schema.field("when").type == pa.timestamp("ns")


def test_missing_breakdown_and_legitimate_values():
    inference = infer_types([_raw(status=["", "  ", "N/A", "NA", None, "Unknown", "0"])], OPTIONS)["status"]
    assert inference.missing == 5
    assert inference.missing_breakdown == {"<empty>": 1, "<whitespace>": 1, "N/A": 1, "NA": 1, "<null>": 1}
    assert inference.non_missing == 2  # "Unknown" and "0" are real values


def test_mostly_numeric_column_reports_failed_conversions():
    inference = infer_types([_raw(price=["10", "12", "13", "n/a?", "15"])], OPTIONS)["price"]
    assert inference.logical_type == "text"
    assert inference.invalid_numeric_count == 1
    assert inference.invalid_numeric_examples == ["n/a?"]


def test_inconsistent_date_formats_are_reported_not_coerced():
    inference = infer_types([_raw(d=["2024-01-05", "05/01/2024", "2024-03-01"])], OPTIONS)["d"]
    assert inference.logical_type == "text"
    assert len(inference.date_formats_seen) == 2
    assert any("Inconsistent date formats" in reason for reason in inference.reasons)


def test_invalid_dates_are_reported():
    inference = infer_types(
        [_raw(d=["2024-01-05", "2024-02-30", "2024-03-01", "2024-03-02", "2024-03-03"])], OPTIONS
    )["d"]
    assert inference.logical_type == "text"
    assert inference.invalid_date_count == 1
    assert "2024-02-30" in inference.invalid_date_examples


@pytest.mark.parametrize(("dayfirst", "expected_month"), [(False, 1), (True, 5)])  # "01/05/2024"
def test_ambiguous_dates_follow_the_dayfirst_option(dayfirst, expected_month):
    options = ParseOptions(missing_tokens=(), dayfirst=dayfirst)
    typed, inference = infer_and_convert(_raw(d=["01/05/2024", "02/06/2024"]), options)
    assert inference["d"].ambiguous_dates is True
    assert typed["d"].iloc[0].month == expected_month


def test_unambiguous_day_first_dates():
    typed, inference = infer_and_convert(_raw(d=["25/12/2024", "05/01/2024"]), ParseOptions())
    assert inference["d"].date_format == "%d/%m/%Y"
    assert typed["d"].iloc[0] == pd.Timestamp("2024-12-25")


def test_identifiers_and_precision_are_preserved():
    typed, inference = infer_and_convert(
        _raw(card=["4111111111111111", "4012888888881881"], precise=["0.12345678901234567", "1.5"]),
        OPTIONS,
    )
    assert inference["card"].semantic == "identifier"
    assert typed["card"].tolist() == ["4111111111111111", "4012888888881881"]
    assert inference["precise"].logical_type == "text"


def test_text_values_keep_their_original_whitespace():
    typed, _ = infer_and_convert(_raw(name=[" Alice ", "Bob"]), OPTIONS)
    assert typed["name"].tolist() == [" Alice ", "Bob"]


def test_currency_symbols_are_canonicalised():
    _, inference = infer_and_convert(_raw(amount=["Rs. 1,200", "INR 500", "₹300"]), OPTIONS)
    assert inference["amount"].logical_type == "float"
    assert inference["amount"].currency == "INR"
    _, mixed = infer_and_convert(_raw(amount=["$5", "€6"]), OPTIONS)
    assert mixed["amount"].logical_type == "text"


def test_european_number_format():
    options = ParseOptions(decimal_separator=",")
    typed, inference = infer_and_convert(_raw(v=["1.234,50", "2,5", "1 000"]), options)
    assert inference["v"].logical_type == "float"
    assert typed["v"].tolist() == [1234.5, 2.5, 1000.0]


def test_grouping_must_be_well_formed():
    patterns = numeric_patterns(ParseOptions())
    assert parse_numeric_token("1,234,567", patterns).normalized == "1234567"
    assert parse_numeric_token("1,23", patterns) is None
    assert parse_numeric_token("(5)", patterns).normalized == "-5"
    assert parse_numeric_token("(-5)", patterns) is None


def test_invalid_parse_options_are_rejected():
    with pytest.raises(ValidationError):
        ParseOptions(decimal_separator=";")
    with pytest.raises(ValidationError):
        ParseOptions(decimal_separator=",", thousands_separator=",")


def test_normalize_headers():
    names, changes = normalize_headers(["  Name ", "name", None, "a\tb", "x" * 200])
    assert names == ["Name", "name_2", "column_3", "a b", "x" * 120]
    assert [c["position"] for c in changes] == [1, 2, 3, 4, 5]


# --- Regression: decimal precision is decided by a float round trip, not a digit count ---


@pytest.mark.parametrize(
    "values, expected",
    [
        (["0.30000000000000004", "0.7000000000000001", "1.5"], "float"),  # float repr output (Excel, 0.1+0.2)
        (["1234.5600000000000", "2.50"], "float"),  # trailing zeros are not significant
        (["0.12345678901234567890", "1.5"], "text"),  # more digits than a double holds
        (["1e400", "1.5"], "text"),  # overflow must not crash or become inf
    ],
)
def test_decimal_precision_uses_round_trip(values, expected):
    from services.type_inference import decimal_is_lossless

    assert decimal_is_lossless("0.30000000000000004") and not decimal_is_lossless("1e-400")
    assert _types(_raw(x=values))["x"] == expected


def test_float_repr_decimals_are_typed_float_but_lossy_ones_stay_text():
    raw = pd.DataFrame(
        {
            "total": pd.Series(["59.97", "0.30000000000000004", "10.5"], dtype=object),
            "big": pd.Series(["1000000000000000.0", "2.0", None], dtype=object),
            "precise": pd.Series(["0.12345678901234567", "1.5", "2"], dtype=object),
        }
    )
    typed, inference = infer_and_convert(raw, OPTIONS)
    assert inference["total"].logical_type == "float"
    assert typed["total"].tolist() == [59.97, 0.1 + 0.2, 10.5]
    assert inference["big"].logical_type == "float"
    assert inference["precise"].logical_type == "text"
    assert inference["precise"].precision_loss_examples == ["0.12345678901234567"]


def test_float_conversion_is_bit_exact():
    """Every decimal typed float must equal Python's float() of the written value.

    pandas' fast string parser is off by one unit in the last place for many
    17-digit values, which would silently change numbers the type check
    promised to store exactly.
    """
    tokens = [
        "29849114.341412332",
        "0.007285605268117946",
        "0.005514662733306819",
        "2.7496936790603807e-07",
        "0.0004227846732701278",
        "0.30000000000000004",
    ]
    typed, inference = infer_and_convert(_raw(x=tokens), OPTIONS)
    assert inference["x"].logical_type == "float"
    assert [v.hex() for v in typed["x"].tolist()] == [float(t).hex() for t in tokens]


def test_iso_dates_with_and_without_times_are_not_inconsistent():
    """One format (ISO 8601) reads every parseable value, even when the column is text for another reason."""
    values = ["2024-01-05", "2024-01-06 10:30:00", "2024-01-07", "2024-01-08", "not a date"]
    inference = infer_types([_raw(d=values)], OPTIONS)["d"]
    assert inference.logical_type == "text"
    assert inference.invalid_date_count == 1
    assert not any("Inconsistent date formats" in reason for reason in inference.reasons)
    chunks = [_raw(d=values[:2]), _raw(d=values[2:])]
    assert infer_types(chunks, OPTIONS)["d"].reasons == inference.reasons


@pytest.mark.parametrize("logical_type, token", [("float", "1e400"), ("float", "-1e400"), ("integer", "1e400")])
def test_convert_series_rejects_out_of_range_values_with_the_column_name(logical_type, token):
    """Backstop: a value inference would never type numeric still fails with a clear error, never inf."""
    from services.type_inference import ColumnInference, convert_series

    inference = ColumnInference(name="amt", logical_type=logical_type, semantic="numeric", non_missing=2, missing=0)
    with pytest.raises(ValidationError, match="'amt'"):
        convert_series(pd.Series([token, "5"], dtype=object), inference, OPTIONS)
