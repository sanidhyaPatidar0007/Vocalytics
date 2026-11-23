"""Shared DataFrame helpers: storage dtypes, row hashing, JSON-safe output, export safety."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from core.constants import LogicalType
from core.errors import ValidationError
from services.type_inference import arrow_type_for

# Spreadsheet applications execute cells starting with these characters (CSV injection).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def logical_type_for_dtype(dtype: Any) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return LogicalType.BOOLEAN.value
    if pd.api.types.is_integer_dtype(dtype):
        return LogicalType.INTEGER.value
    if pd.api.types.is_float_dtype(dtype):
        return LogicalType.FLOAT.value
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return LogicalType.DATETIME.value
    return LogicalType.TEXT.value


def normalize_series(series: pd.Series) -> pd.Series:
    """Cast to one of the five storage dtypes (Int64, float64, boolean, datetime64[ns], string)."""
    logical = logical_type_for_dtype(series.dtype)
    if logical == LogicalType.INTEGER.value:
        return series.astype("Int64")
    if logical == LogicalType.FLOAT.value:
        return series.astype("float64")
    if logical == LogicalType.BOOLEAN.value:
        return series.astype("boolean")
    if logical == LogicalType.DATETIME.value:
        if getattr(series.dt, "tz", None) is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
        return series.astype("datetime64[ns]")
    if isinstance(series.dtype, pd.StringDtype):
        return series
    return series.astype("string")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate column names and normalize every column to a storage dtype."""
    names = [str(c) for c in df.columns]
    if len(set(n.casefold() for n in names)) != len(names):
        raise ValidationError("Column names must be unique (case-insensitive).")
    if any(not n.strip() for n in names):
        raise ValidationError("Column names must not be blank.")
    data = {name: normalize_series(df[col]) for name, col in zip(names, df.columns, strict=True)}
    return pd.DataFrame(data, index=df.index).reset_index(drop=True)


def frame_schema(df: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"name": str(name), "logical_type": logical_type_for_dtype(dtype), "dtype": str(dtype)}
        for name, dtype in df.dtypes.items()
    ]


def arrow_schema_for_frame(df: pd.DataFrame) -> pa.Schema:
    return pa.schema(
        [
            pa.field(str(name), arrow_type_for(logical_type_for_dtype(dtype)))
            for name, dtype in df.dtypes.items()
        ]
    )


def hash_rows(df: pd.DataFrame) -> np.ndarray:
    """64-bit hash per row (equal rows, including missing values, hash equally).

    Signed zero is canonicalized: 0.0 and -0.0 compare equal in pandas, so they hash equally.
    """
    if len(df) == 0:
        return np.empty(0, dtype=np.uint64)
    float_positions = [i for i, dtype in enumerate(df.dtypes) if pd.api.types.is_float_dtype(dtype)]
    if float_positions:
        df = df.copy()
        for position in float_positions:
            # IEEE: -0.0 + 0.0 is +0.0; NaN stays NaN.
            df.isetitem(position, df.iloc[:, position] + 0.0)
    return pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain JSON-compatible Python values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (bool, int, str)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def records(df: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    subset = df if max_rows is None else df.head(max_rows)
    return [{str(k): json_safe(v) for k, v in row.items()} for row in subset.to_dict(orient="records")]


def escape_spreadsheet_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix text that a spreadsheet would execute as a formula with an apostrophe.

    Uploaded data is untrusted; this prevents CSV/Excel formula injection in
    exported files. Text cells and column names (headers come from uploads
    too) are escaped; numbers, dates and booleans are never touched.
    """
    out = df.copy()
    for name in out.columns:
        if logical_type_for_dtype(out[name].dtype) != LogicalType.TEXT.value:
            continue
        series = out[name].astype("string")
        mask = series.str.startswith(_FORMULA_PREFIXES, na=False)
        if mask.any():
            out[name] = series.where(~mask, "'" + series)
    taken = {str(c) for c in out.columns}
    renamed: dict[Any, str] = {}
    for name in out.columns:
        text = str(name)
        if text.startswith(_FORMULA_PREFIXES):
            escaped = "'" + text
            while escaped in taken:
                escaped = "'" + escaped
            taken.add(escaped)
            renamed[name] = escaped
    return out.rename(columns=renamed) if renamed else out
