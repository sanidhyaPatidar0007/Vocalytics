"""Exact, text-first column type inference: the dataset typing contract.

Uploaded files are first read as raw text, then every column is typed by the
rules below. The rules look at EVERY non-missing value of the column across all
chunks of the file (no sampling), so the result never depends on chunk size.

1. Missing values are: empty strings, whitespace-only strings and the configured
   placeholder tokens (case-insensitive, compared after trimming). Nothing else
   is missing: 0, False or a category named "Unknown" are ordinary values.
2. integer: every value is a whole number written WITHOUT a decimal separator or
   exponent (optional sign, correctly grouped thousands separators). Stored as
   nullable Int64, so missing values never turn an integer column into floats.
3. float: every value is numeric and at least one value is written with a
   decimal separator or exponent, or carries a currency symbol/code. "1200.00"
   is decimal notation, so a revenue column like 1200.00 / 980.00 stays float:
   decimal columns are never downcast to integers because their values happen
   to be whole numbers.
4. Identifiers keep their digits: values with leading zeros ("00123") and integers
   longer than 15 digits stay text. A decimal becomes float only when a 64-bit
   float stores it without changing its value as written (``decimal_is_lossless``:
   0.30000000000000004 and 1234.5600 qualify; 0.12345678901234567890, 1e400 and
   1e-400 do not), so converting never loses zeros or precision.
5. boolean: every value is one of true/false/yes/no/y/n/t/f (case-insensitive).
6. datetime: every value parses with ONE consistent format (or as ISO 8601).
   Mixed formats keep the column as text and are reported. When day/month order
   is ambiguous the ``dayfirst`` option decides and the ambiguity is reported.
7. Anything else is text. Mostly-numeric (>= 80 %) or mostly-date text columns
   are reported with counts and examples of the values that could not be
   converted. Values are never silently coerced to missing.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from core.constants import PANDAS_DTYPE_FOR, LogicalType
from core.errors import ValidationError

MAX_SAFE_DIGITS = 15
MOSTLY_THRESHOLD = 0.8
DISTINCT_TRACK_CAP = 10_000
EXAMPLE_LIMIT = 5
CATEGORICAL_MAX_DISTINCT = 50
CATEGORICAL_MAX_RATIO = 0.5
PRECISION_TEXT_REASON = (
    "Decimals with more digits than a 64-bit float can store exactly "
    "(or out of its range) are kept as text to preserve every digit."
)
INCONSISTENT_DATES_REASON = "Inconsistent date formats: "

BOOLEAN_TRUE = frozenset({"true", "yes", "y", "t"})
BOOLEAN_FALSE = frozenset({"false", "no", "n", "f"})
BOOLEAN_TOKENS = BOOLEAN_TRUE | BOOLEAN_FALSE

_SYMBOLS = "$€£₹¥"
_CODES = r"USD|EUR|GBP|INR|JPY|RS\.?"
_NUM_BODY = r"[+-]?[\d.,\s  ']+(?:[eE][+-]?\d+)?"
_CURRENCY_PREFIX = re.compile(
    rf"^(?P<sign>[+-]?)\s*(?P<cur>[{_SYMBOLS}]|(?:{_CODES})(?=[\s\d.,+-]))\s*(?P<rest>{_NUM_BODY})$",
    re.IGNORECASE,
)
_CURRENCY_SUFFIX = re.compile(
    rf"^(?P<rest>{_NUM_BODY}?)\s*(?P<cur>[{_SYMBOLS}]|(?:{_CODES}))$", re.IGNORECASE
)
# Symbols and codes that denote the same currency compare equal.
_CURRENCY_CANONICAL = {
    "$": "USD",
    "USD": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    "₹": "INR",
    "INR": "INR",
    "RS": "INR",
    "¥": "JPY",
    "JPY": "JPY",
}

_IDENTIFIER_NAME = re.compile(
    r"(?i)(^id$|_id$|^id_|[ _-]id$|code|number|^no$|_no$|zip|postal|pin ?code|phone|mobile|account|sku|uuid|guid)"
)

ISO_FORMAT = "ISO8601"
DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y.%m.%d",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d-%m-%Y",
    "%d-%m-%Y %H:%M:%S",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m-%d-%Y",
    "%m-%d-%Y %H:%M:%S",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d-%B-%Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%d-%b-%y",
    "%d/%m/%y",
    "%m/%d/%y",
)
IDENTIFIER_NAME = _IDENTIFIER_NAME  # public alias: same identifier-name rule for every module
_DAYFIRST = frozenset(f for f in DATE_FORMATS if f.startswith("%d"))
_MONTHFIRST = frozenset(f for f in DATE_FORMATS if f.startswith("%m"))
_DATE_SHAPE = re.compile(
    r"^(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\d{1,2}[- ][A-Za-z]{3,9}[- ,]+\d{2,4}"
    r"|[A-Za-z]{3,9} \d{1,2},? \d{4})$"
)
DATE_SHAPE = _DATE_SHAPE  # public alias: "looks like a date" pre-filter shared with profiling


# ---------------------------------------------------------------------------
# Options and numeric token parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParseOptions:
    """How raw text is interpreted. ``thousands_separator=None`` accepts correctly
    grouped separators automatically (',' when the decimal separator is '.',
    '.' or spaces when it is ',')."""

    missing_tokens: tuple[str, ...] = ()
    decimal_separator: str = "."
    thousands_separator: str | None = None
    dayfirst: bool = False

    def __post_init__(self) -> None:
        if self.decimal_separator not in {".", ","}:
            raise ValidationError("Decimal separator must be '.' or ','.")
        if self.thousands_separator is not None and (
            self.thousands_separator not in {",", ".", " ", "'", "_"}
            or self.thousands_separator == self.decimal_separator
        ):
            raise ValidationError("Thousands separator must differ from the decimal separator.")

    @property
    def missing_casefold(self) -> frozenset[str]:
        return frozenset(token.strip().casefold() for token in self.missing_tokens if token and token.strip())


@dataclass(frozen=True)
class NumericPatterns:
    integer: re.Pattern[str]
    decimal: re.Pattern[str]
    group_chars: str
    decimal_char: str


def numeric_patterns(options: ParseOptions) -> NumericPatterns:
    d = re.escape(options.decimal_separator)
    if options.thousands_separator is not None:
        groups = options.thousands_separator
    else:
        groups = "," if options.decimal_separator == "." else ".   "
    g = "[" + "".join(re.escape(c) for c in groups) + "]"
    int_core = rf"(?:\d+|\d{{1,3}}(?:{g}\d{{3}})+)"
    integer = re.compile(rf"^[+-]?{int_core}$")
    decimal = re.compile(
        rf"^[+-]?(?:{int_core}{d}\d*|{d}\d+|{int_core}(?:{d}\d*)?[eE][+-]?\d+|{d}\d+[eE][+-]?\d+)$"
    )
    return NumericPatterns(
        integer=integer, decimal=decimal, group_chars=groups, decimal_char=options.decimal_separator
    )


@dataclass(frozen=True)
class NumericToken:
    kind: str  # "int" or "dec"
    normalized: str  # plain digits, '.' decimal point, sign applied: "-1234.50"
    currency: str | None
    leading_zero: bool
    significant_digits: int


def decimal_is_lossless(normalized: str) -> bool:
    """True when float64 stores this decimal without changing its value as written.

    ``normalized`` is a plain number string such as ``NumericToken.normalized``
    ("-1234.50", "1.5e3") or ``str(Decimal)``. The test is a round trip: the
    shortest repr of the parsed float must equal the written value numerically,
    so trailing zeros never count and 16-17 digit float output (Excel, 0.1+0.2)
    qualifies, while values with more digits than a double holds, overflow
    (1e400) and underflow (1e-400) do not.
    """
    try:
        written = Decimal(normalized)
        value = float(written)
    except (InvalidOperation, ValueError, OverflowError):
        return False
    return math.isfinite(value) and Decimal(repr(value)) == written


def parse_numeric_token(token: str, patterns: NumericPatterns) -> NumericToken | None:
    """Parse one trimmed token. Returns None when it is not a number."""
    text = token.strip()
    if not text:
        return None
    negative = False
    if len(text) >= 2 and text[0] == "(" and text[-1] == ")":
        text = text[1:-1].strip()
        negative = True
    currency = None
    match = _CURRENCY_PREFIX.match(text)
    if match:
        currency = match.group("cur").upper()
        text = (match.group("sign") or "") + match.group("rest").strip()
    else:
        match = _CURRENCY_SUFFIX.match(text)
        if match and match.group("rest").strip():
            currency = match.group("cur").upper()
            text = match.group("rest").strip()
    if patterns.integer.match(text):
        kind = "int"
    elif patterns.decimal.match(text):
        kind = "dec"
    else:
        return None
    sign = ""
    if text[0] in "+-":
        sign = "-" if text[0] == "-" else ""
        text = text[1:]
    if negative:
        if sign == "-":
            return None
        sign = "-"
    core = text
    for char in patterns.group_chars:
        core = core.replace(char, "")
    if patterns.decimal_char != ".":
        core = core.replace(patterns.decimal_char, ".")
    mantissa = core.lower().split("e")[0]
    int_part = mantissa.split(".")[0]
    leading_zero = len(int_part) > 1 and int_part.startswith("0")
    digits = len(mantissa.replace(".", "").lstrip("0")) or 1
    if currency is not None:
        key = currency.strip().rstrip(".").upper()
        currency = _CURRENCY_CANONICAL.get(key, key)
    return NumericToken(
        kind=kind,
        normalized=sign + core,
        currency=currency,
        leading_zero=leading_zero,
        significant_digits=digits,
    )


# ---------------------------------------------------------------------------
# Header normalization
# ---------------------------------------------------------------------------


def normalize_headers(raw: Iterable[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Clean header names: no blanks, no control characters, no duplicates.

    Duplicate detection is case-insensitive ("Revenue" and "revenue" clash).
    Every change is returned so the import report can show it.
    """
    names: list[str] = []
    changes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, header in enumerate(raw, start=1):
        original = "" if header is None or (isinstance(header, float) and np.isnan(header)) else str(header)
        clean = re.sub(r"\s+", " ", original)  # tabs/newlines become spaces first
        clean = re.sub(r"[\x00-\x1f\x7f]", "", clean)
        clean = re.sub(r" +", " ", clean).strip()[:120]
        reason = None
        if not clean:
            clean = f"column_{position}"
            reason = "blank header"
        base = clean
        suffix = 2
        while clean.casefold() in seen:
            clean = f"{base}_{suffix}"
            suffix += 1
            reason = "duplicate header"
        if reason is None and clean != original:
            reason = "whitespace or control characters removed"
        if reason is not None:
            changes.append({"position": position, "original": original, "new": clean, "reason": reason})
        seen.add(clean.casefold())
        names.append(clean)
    return names, changes


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_dates(values: pd.Series, fmt: str) -> pd.Series:
    """Parse strings with one explicit format; unparseable values become NaT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt == ISO_FORMAT:
            parsed = pd.to_datetime(values, format=ISO_FORMAT, errors="coerce", utc=True)
            return parsed.dt.tz_localize(None)
        return pd.to_datetime(values, format=fmt, errors="coerce")


def _parses_all(values: pd.Series, fmt: str) -> bool:
    if values.empty:
        return True
    head = values.iloc[:200]
    if parse_dates(head, fmt).isna().any():
        return False
    return not parse_dates(values, fmt).isna().any()


def choose_date_format(candidates: set[str], dayfirst: bool) -> tuple[str | None, bool]:
    """Pick one format among those that parse every value. Returns (format, ambiguous)."""
    if not candidates:
        return None, False
    if ISO_FORMAT in candidates:
        return ISO_FORMAT, False
    ordered = [f for f in DATE_FORMATS if f in candidates]
    day = [f for f in ordered if f in _DAYFIRST]
    month = [f for f in ordered if f in _MONTHFIRST]
    other = [f for f in ordered if f not in _DAYFIRST and f not in _MONTHFIRST]
    if other:
        return other[0], False
    if day and month:
        return (day[0] if dayfirst else month[0]), True
    return ordered[0], False


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------


@dataclass
class ColumnInference:
    name: str
    logical_type: str
    semantic: str
    non_missing: int
    missing: int
    missing_breakdown: dict[str, int] = field(default_factory=dict)
    date_format: str | None = None
    ambiguous_dates: bool = False
    reasons: list[str] = field(default_factory=list)
    invalid_numeric_count: int = 0
    invalid_numeric_examples: list[str] = field(default_factory=list)
    invalid_date_count: int = 0
    invalid_date_examples: list[str] = field(default_factory=list)
    date_formats_seen: dict[str, int] = field(default_factory=dict)
    precision_loss_examples: list[str] = field(default_factory=list)
    distinct_count: int | None = None
    currency: str | None = None

    @property
    def pandas_dtype(self) -> str:
        return PANDAS_DTYPE_FOR[self.logical_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "logical_type": self.logical_type,
            "pandas_dtype": self.pandas_dtype,
            "semantic": self.semantic,
            "non_missing": self.non_missing,
            "missing": self.missing,
            "missing_breakdown": dict(self.missing_breakdown),
            "date_format": self.date_format,
            "ambiguous_dates": self.ambiguous_dates,
            "reasons": list(self.reasons),
            "invalid_numeric_count": self.invalid_numeric_count,
            "invalid_numeric_examples": list(self.invalid_numeric_examples),
            "invalid_date_count": self.invalid_date_count,
            "invalid_date_examples": list(self.invalid_date_examples),
            "date_formats_seen": dict(self.date_formats_seen),
            "precision_loss_examples": list(self.precision_loss_examples),
            "distinct_count": self.distinct_count,
            "currency": self.currency,
        }


class ColumnTypeAccumulator:
    """Accumulates evidence about one column across chunks (exact, bounded memory)."""

    def __init__(self, name: str, options: ParseOptions, patterns: NumericPatterns) -> None:
        self.name = name
        self.options = options
        self.patterns = patterns
        self._missing_cf = options.missing_casefold
        self.non_missing = 0
        self.missing = 0
        self.missing_breakdown: Counter[str] = Counter()
        self.all_bool = True
        self.numeric_count = 0
        self.non_numeric_examples: list[str] = []
        self.any_decimal = False
        self.any_leading_zero = False
        self.any_long_integer = False
        self.any_precision_loss = False
        self.precision_loss_examples: list[str] = []
        self.currencies: set[str] = set()
        self.date_shape_count = 0
        self.non_date_examples: list[str] = []
        self.date_candidates: set[str] | None = None
        self.date_formats_seen: Counter[str] = Counter()
        # Formats that read every parseable date-shaped value (None until one is seen).
        self.covering_formats: set[str] | None = None
        self.unparseable_date_count = 0
        self.unparseable_date_examples: list[str] = []
        self.distinct: set[str] = set()
        self.distinct_overflow = False

    @staticmethod
    def _add_examples(target: list[str], items: Iterable[str]) -> None:
        for item in items:
            if len(target) >= EXAMPLE_LIMIT:
                return
            if item not in target:
                target.append(item)

    def update(self, raw: pd.Series) -> None:
        is_null = raw.isna()
        null_count = int(is_null.sum())
        if null_count:
            self.missing_breakdown["<null>"] += null_count
        text = raw[~is_null].astype(str)
        if text.empty:
            self.missing += null_count
            return
        stripped = text.str.strip()
        empty = text == ""
        whitespace = (stripped == "") & ~empty
        placeholder = (stripped != "") & stripped.str.casefold().isin(self._missing_cf)
        if empty.any():
            self.missing_breakdown["<empty>"] += int(empty.sum())
        if whitespace.any():
            self.missing_breakdown["<whitespace>"] += int(whitespace.sum())
        if placeholder.any():
            for token, count in stripped[placeholder].value_counts().items():
                self.missing_breakdown[str(token)] += int(count)
        missing_mask = empty | whitespace | placeholder
        self.missing += null_count + int(missing_mask.sum())
        values = stripped[~missing_mask]
        if values.empty:
            return
        self.non_missing += len(values)
        counts = values.value_counts()
        uniques = pd.Series(counts.index.astype(str), dtype=object)
        multiplicity = counts.to_numpy()

        if not self.distinct_overflow:
            self.distinct.update(uniques.tolist())
            if len(self.distinct) > DISTINCT_TRACK_CAP:
                self.distinct_overflow = True
                self.distinct.clear()

        if self.all_bool:
            self.all_bool = bool(uniques.str.casefold().isin(BOOLEAN_TOKENS).all())

        non_numeric: list[str] = []
        for token, count in zip(uniques.tolist(), multiplicity.tolist(), strict=True):
            parsed = parse_numeric_token(token, self.patterns)
            if parsed is None:
                non_numeric.append(token)
                continue
            self.numeric_count += int(count)
            if parsed.kind == "dec":
                self.any_decimal = True
                if not decimal_is_lossless(parsed.normalized):
                    self.any_precision_loss = True
                    self._add_examples(self.precision_loss_examples, [token])
            elif parsed.significant_digits > MAX_SAFE_DIGITS:
                self.any_long_integer = True
            if parsed.leading_zero:
                self.any_leading_zero = True
            if parsed.currency:
                self.currencies.add(parsed.currency)
        self._add_examples(self.non_numeric_examples, non_numeric)

        shaped_mask = uniques.str.match(_DATE_SHAPE).fillna(False).to_numpy(dtype=bool)
        self.date_shape_count += int(multiplicity[shaped_mask].sum())
        self._add_examples(self.non_date_examples, uniques[~shaped_mask].tolist())
        if not shaped_mask.all():
            self.date_candidates = set()
        if shaped_mask.any():
            shaped = uniques[shaped_mask].reset_index(drop=True)
            shaped_counts = multiplicity[shaped_mask]
            if self.date_candidates is None:
                self.date_candidates = {ISO_FORMAT, *DATE_FORMATS}
            if self.date_candidates:
                self.date_candidates = {fmt for fmt in self.date_candidates if _parses_all(shaped, fmt)}
            remaining = shaped
            remaining_counts = shaped_counts
            for fmt in DATE_FORMATS:
                if remaining.empty:
                    break
                ok = parse_dates(remaining, fmt).notna().to_numpy(dtype=bool)
                if ok.any():
                    self.date_formats_seen[fmt] += int(remaining_counts[ok].sum())
                    remaining = remaining[~ok].reset_index(drop=True)
                    remaining_counts = remaining_counts[~ok]
            if not remaining.empty:
                iso_ok = parse_dates(remaining, ISO_FORMAT).notna().to_numpy(dtype=bool)
                if iso_ok.any():
                    self.date_formats_seen[ISO_FORMAT] += int(remaining_counts[iso_ok].sum())
                    remaining = remaining[~iso_ok]
                    remaining_counts = remaining_counts[~iso_ok]
                self.unparseable_date_count += int(remaining_counts.sum())
                self._add_examples(self.unparseable_date_examples, remaining.tolist())
            parseable = shaped[~shaped.isin(set(remaining.tolist()))].reset_index(drop=True)
            if not parseable.empty:
                if self.covering_formats is None:
                    self.covering_formats = {ISO_FORMAT, *DATE_FORMATS}
                if self.covering_formats:
                    self.covering_formats = {
                        fmt for fmt in self.covering_formats if _parses_all(parseable, fmt)
                    }
        elif self.date_candidates is None:
            self.date_candidates = set()

    def finalize(self) -> ColumnInference:
        reasons: list[str] = []
        currency = None
        date_format = None
        ambiguous = False
        invalid_numeric = 0
        invalid_numeric_examples: list[str] = []
        invalid_dates = 0
        invalid_date_examples: list[str] = []
        distinct = None if self.distinct_overflow else len(self.distinct)

        if self.non_missing == 0:
            logical, semantic = LogicalType.TEXT, "empty"
            reasons.append("Every value is missing.")
        elif self.all_bool:
            logical, semantic = LogicalType.BOOLEAN, "boolean"
        elif self.numeric_count == self.non_missing:
            if len(self.currencies) > 1:
                logical, semantic = LogicalType.TEXT, "text"
                reasons.append(f"Mixed currencies ({', '.join(sorted(self.currencies))}) kept as text.")
            elif self.any_leading_zero:
                logical, semantic = LogicalType.TEXT, "identifier"
                reasons.append("Values with leading zeros are kept as text so no digits are lost.")
            elif self.any_long_integer:
                logical, semantic = LogicalType.TEXT, "identifier"
                reasons.append(
                    f"Numbers longer than {MAX_SAFE_DIGITS} digits are kept as text to preserve every digit."
                )
            elif self.any_precision_loss:
                logical, semantic = LogicalType.TEXT, "text"
                reasons.append(PRECISION_TEXT_REASON)
            elif self.any_decimal or self.currencies:
                logical = LogicalType.FLOAT
                semantic = "currency" if self.currencies else "numeric"
                currency = next(iter(self.currencies)) if self.currencies else None
                if self.currencies:
                    reasons.append(f"Currency amounts ({currency}) stored as decimal numbers.")
                else:
                    reasons.append(
                        "Decimal notation found, stored as decimal numbers (never downcast to integer)."
                    )
            else:
                logical = LogicalType.INTEGER
                semantic = "identifier" if _IDENTIFIER_NAME.search(self.name) else "numeric"
                if self.missing:
                    reasons.append("Whole numbers with missing values stored as nullable integers.")
        elif self.date_candidates:
            date_format, ambiguous = choose_date_format(self.date_candidates, self.options.dayfirst)
            logical, semantic = LogicalType.DATETIME, "datetime"
            if ambiguous:
                order = "day/month" if self.options.dayfirst else "month/day"
                reasons.append(f"Day and month order is ambiguous; interpreted as {order} (date option).")
        else:
            logical = LogicalType.TEXT
            semantic = self._text_semantic(distinct)
            numeric_ratio = self.numeric_count / self.non_missing
            date_ratio = self.date_shape_count / self.non_missing
            if MOSTLY_THRESHOLD <= numeric_ratio < 1:
                invalid_numeric = self.non_missing - self.numeric_count
                invalid_numeric_examples = list(self.non_numeric_examples)
                reasons.append(
                    f"{numeric_ratio:.0%} of values are numeric; {invalid_numeric} value(s) could not be "
                    "converted, so the column is kept as text."
                )
            if date_ratio >= MOSTLY_THRESHOLD:
                invalid_dates = (self.non_missing - self.date_shape_count) + self.unparseable_date_count
                invalid_date_examples = list(self.non_date_examples) + list(self.unparseable_date_examples)
                invalid_date_examples = invalid_date_examples[:EXAMPLE_LIMIT]
                if len(self.date_formats_seen) > 1 and not self.covering_formats:
                    reasons.append(
                        INCONSISTENT_DATES_REASON
                        + ", ".join(f"{fmt} ({count})" for fmt, count in self.date_formats_seen.most_common())
                    )
                if invalid_dates:
                    reasons.append(f"{invalid_dates} value(s) could not be read as dates.")
        return ColumnInference(
            name=self.name,
            logical_type=logical.value,
            semantic=semantic,
            non_missing=self.non_missing,
            missing=self.missing,
            missing_breakdown=dict(self.missing_breakdown),
            date_format=date_format,
            ambiguous_dates=ambiguous,
            reasons=reasons,
            invalid_numeric_count=invalid_numeric,
            invalid_numeric_examples=invalid_numeric_examples,
            invalid_date_count=invalid_dates,
            invalid_date_examples=invalid_date_examples,
            date_formats_seen=dict(self.date_formats_seen),
            precision_loss_examples=list(self.precision_loss_examples) if self.any_precision_loss else [],
            distinct_count=distinct,
            currency=currency,
        )

    def _text_semantic(self, distinct: int | None) -> str:
        if _IDENTIFIER_NAME.search(self.name) and distinct is not None and distinct >= 0.9 * self.non_missing:
            return "identifier"
        if distinct is not None and (
            distinct <= CATEGORICAL_MAX_DISTINCT or distinct <= CATEGORICAL_MAX_RATIO * self.non_missing
        ):
            return "categorical"
        return "text"


def infer_types(
    chunks: Iterable[pd.DataFrame], options: ParseOptions, *, check: Callable[[], None] | None = None
) -> dict[str, ColumnInference]:
    """Pass 1: infer the type of every column from raw text chunks (object dtype).

    ``check`` (for example ``OperationGuard.check``) runs before each column so
    a cancellation or timeout can interrupt a wide file.
    """
    patterns = numeric_patterns(options)
    accumulators: dict[str, ColumnTypeAccumulator] = {}
    columns: list[str] | None = None
    for chunk in chunks:
        if columns is None:
            columns = [str(c) for c in chunk.columns]
            accumulators = {name: ColumnTypeAccumulator(name, options, patterns) for name in columns}
        for name in columns:
            if check is not None:
                check()
            accumulators[name].update(chunk[name])
    if columns is None:
        return {}
    return {name: accumulators[name].finalize() for name in columns}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def missing_mask(raw: pd.Series, options: ParseOptions) -> pd.Series:
    """True where a raw text value counts as missing under ``options``."""
    is_null = raw.isna()
    stripped = raw.where(~is_null, "").astype(str).str.strip()
    return is_null | (stripped == "") | stripped.str.casefold().isin(options.missing_casefold)


def convert_series(raw: pd.Series, inference: ColumnInference, options: ParseOptions) -> pd.Series:
    """Pass 2: convert raw text to the inferred dtype. Raises if a value does not fit."""
    missing = missing_mask(raw, options)
    logical = inference.logical_type
    if logical == LogicalType.TEXT.value:
        return raw.where(~missing, None).astype("string")
    values = raw[~missing].astype(str).str.strip()
    if logical in (LogicalType.INTEGER.value, LogicalType.FLOAT.value):
        patterns = numeric_patterns(options)
        cache: dict[str, str] = {}
        for token in pd.unique(values):
            parsed = parse_numeric_token(token, patterns)
            if parsed is None:
                raise ValidationError(f"Column '{inference.name}' contains a non-numeric value.")
            cache[token] = parsed.normalized
        normalized = values.map(cache)
        out_of_range = ValidationError(
            f"Column '{inference.name}' contains a number outside the supported range."
        )
        if logical == LogicalType.INTEGER.value:
            out = pd.Series(pd.NA, index=raw.index, dtype="Int64")
            if len(normalized):
                try:
                    out.loc[normalized.index] = pd.array([int(v) for v in normalized], dtype="Int64")
                except (ValueError, OverflowError, TypeError) as exc:
                    raise out_of_range from exc
            return out
        out_f = pd.Series(np.nan, index=raw.index, dtype="float64")
        if len(normalized):
            # Python's float() rounds correctly; pandas' fast parser can be one
            # unit in the last place off for 17-digit values such as 0.1 + 0.2.
            exact: dict[str, float] = {}
            for text in pd.unique(normalized):
                try:
                    value = float(text)
                except (ValueError, OverflowError) as exc:
                    raise out_of_range from exc
                if not math.isfinite(value):
                    raise out_of_range
                exact[text] = value
            out_f.loc[normalized.index] = normalized.map(exact).astype("float64")
        return out_f
    if logical == LogicalType.BOOLEAN.value:
        out_b = pd.Series(pd.NA, index=raw.index, dtype="boolean")
        if len(values):
            out_b.loc[values.index] = values.str.casefold().isin(BOOLEAN_TRUE).to_numpy()
        return out_b
    if logical == LogicalType.DATETIME.value:
        out_d = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
        if len(values):
            parsed_dates = parse_dates(values, inference.date_format or ISO_FORMAT)
            if parsed_dates.isna().any():
                raise ValidationError(f"Column '{inference.name}' contains a value that is not a valid date.")
            out_d.loc[values.index] = parsed_dates.astype("datetime64[ns]").to_numpy()
        return out_d
    raise ValidationError(f"Unknown column type '{logical}'.")


def convert_frame(
    raw: pd.DataFrame,
    inferences: dict[str, ColumnInference],
    options: ParseOptions,
    *,
    check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Pass 2 for one chunk; ``check`` runs before each column (see ``infer_types``)."""
    converted: dict[str, pd.Series] = {}
    for name in raw.columns:
        if check is not None:
            check()
        converted[name] = convert_series(raw[name], inferences[name], options)
    return pd.DataFrame(converted, index=raw.index)


_ARROW_TYPES = {
    LogicalType.INTEGER.value: pa.int64(),
    LogicalType.FLOAT.value: pa.float64(),
    LogicalType.BOOLEAN.value: pa.bool_(),
    LogicalType.DATETIME.value: pa.timestamp("ns"),
    LogicalType.TEXT.value: pa.string(),
}


def arrow_type_for(logical_type: str) -> pa.DataType:
    return _ARROW_TYPES[logical_type]


def arrow_schema(columns: list[str], inferences: dict[str, ColumnInference]) -> pa.Schema:
    """Fixed schema so an all-missing chunk can never change a column's type."""
    return pa.schema([pa.field(name, arrow_type_for(inferences[name].logical_type)) for name in columns])


def infer_and_convert(
    raw: pd.DataFrame, options: ParseOptions
) -> tuple[pd.DataFrame, dict[str, ColumnInference]]:
    """Convenience for in-memory frames of raw text (single chunk)."""
    inferences = infer_types([raw], options)
    return convert_frame(raw, inferences, options), inferences
