"""Dataset quality score (0-100), shared by versioning, profiling and cleaning reports.

    score = 100 * (0.40 * completeness + 0.20 * uniqueness + 0.25 * validity + 0.15 * consistency)

    completeness = 1 - missing cells / total cells
    uniqueness   = 1 - exact duplicate rows / rows (exact counts from 64-bit row hashes)
    validity     = 1 - invalid cells / total cells; a cell is invalid when it breaks a
                   text column that is otherwise numeric or date-like (>= 80 % of values)
    consistency  = 1 - text cells with leading/trailing whitespace / total cells

A dataset with no rows or no columns has no cells to be wrong and scores 100.
Only confirmed problems lower the score. Suspected issues (outliers, repeated
categories, possible duplicate business keys) are reported by profiling but
never penalized, because they are often legitimate.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from services.frame_utils import hash_rows, logical_type_for_dtype
from services.type_inference import MOSTLY_THRESHOLD

WEIGHTS = {"completeness": 0.40, "uniqueness": 0.20, "validity": 0.25, "consistency": 0.15}

_NUMERIC_LIKE = re.compile(
    r"^\(?[+-]?\s*[$€£₹¥]?\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][+-]?\d+)?\s*[$€£₹¥]?\)?$"
)
_DATE_LIKE = re.compile(
    r"^(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"|\d{1,2}[- ][A-Za-z]{3,9}[- ,]+\d{2,4}|[A-Za-z]{3,9} \d{1,2},? \d{4})$"
)


def text_parse_counts(values: pd.Series) -> tuple[int, int, int]:
    """Non-null count, numeric-looking count and date-looking count of a text column batch."""
    present = values.dropna().astype(str)
    if present.empty:
        return 0, 0, 0
    stripped = present.str.strip()
    return len(present), int(stripped.str.match(_NUMERIC_LIKE).sum()), int(stripped.str.match(_DATE_LIKE).sum())


def count_invalid_text(nonnull: int, numeric_like: int, date_like: int, expected: str | None = None) -> int:
    """Invalid cells of one text column: values that break the numeric or date shape it should have.

    ``expected`` is the logical type the column is known to have had ("integer", "float" or
    "datetime"); then every value that does not parse as such is invalid. Without it a column
    counts as numeric or date-like only when at least ``MOSTLY_THRESHOLD`` of its values look
    like one, and the rest are invalid.
    """
    if not nonnull:
        return 0
    if expected in ("integer", "float"):
        return nonnull - numeric_like
    if expected == "datetime":
        return nonnull - date_like
    for like in (numeric_like, date_like):
        if MOSTLY_THRESHOLD <= like / nonnull < 1:
            return nonnull - like
    return 0


@dataclass
class QualityResult:
    score: float
    completeness: float
    uniqueness: float
    validity: float
    consistency: float
    rows: int
    columns: int
    total_cells: int
    missing_cells: int
    duplicate_rows: int
    invalid_cells: int
    whitespace_cells: int

    def to_dict(self) -> dict:
        return asdict(self)


class QualityAccumulator:
    """Computes the quality score over batches with bounded memory (8 bytes per row)."""

    def __init__(self) -> None:
        self.rows = 0
        self.columns: list[str] | None = None
        self.missing = 0
        self.whitespace = 0
        self._hashes: list[np.ndarray] = []
        self._text_nonnull: dict[str, int] = {}
        self._numeric_like: dict[str, int] = {}
        self._date_like: dict[str, int] = {}

    def update(self, batch: pd.DataFrame) -> None:
        if self.columns is None:
            self.columns = [str(c) for c in batch.columns]
        self.rows += len(batch)
        if len(batch) == 0:
            return
        self.missing += int(batch.isna().sum().sum())
        self._hashes.append(hash_rows(batch))
        for name in batch.columns:
            if logical_type_for_dtype(batch[name].dtype) != "text":
                continue
            values = batch[name].dropna().astype(str)
            if values.empty:
                continue
            key = str(name)
            stripped = values.str.strip()
            self.whitespace += int((values != stripped).sum())
            self._text_nonnull[key] = self._text_nonnull.get(key, 0) + len(values)
            self._numeric_like[key] = self._numeric_like.get(key, 0) + int(
                stripped.str.match(_NUMERIC_LIKE).sum()
            )
            self._date_like[key] = self._date_like.get(key, 0) + int(stripped.str.match(_DATE_LIKE).sum())

    def finalize(self) -> QualityResult:
        columns = len(self.columns or [])
        total = self.rows * columns
        hashes = np.concatenate(self._hashes) if self._hashes else np.empty(0, dtype=np.uint64)
        duplicates = int(len(hashes) - len(np.unique(hashes))) if len(hashes) else 0
        invalid = 0
        for key, nonnull in self._text_nonnull.items():
            invalid += count_invalid_text(nonnull, self._numeric_like.get(key, 0), self._date_like.get(key, 0))
        if total == 0:
            # No cells means nothing is missing, duplicated, invalid or inconsistent: score the
            # dataset 100 instead of presenting a header-only file as maximally broken.
            return QualityResult(100.0, 1.0, 1.0, 1.0, 1.0, self.rows, columns, 0, 0, 0, 0, 0)
        completeness = 1 - self.missing / total
        uniqueness = 1 - duplicates / self.rows if self.rows else 1.0
        validity = 1 - invalid / total
        consistency = 1 - self.whitespace / total
        score = 100 * (
            WEIGHTS["completeness"] * completeness
            + WEIGHTS["uniqueness"] * uniqueness
            + WEIGHTS["validity"] * validity
            + WEIGHTS["consistency"] * consistency
        )
        return QualityResult(
            score=round(score, 1),
            completeness=round(completeness, 4),
            uniqueness=round(uniqueness, 4),
            validity=round(validity, 4),
            consistency=round(consistency, 4),
            rows=self.rows,
            columns=columns,
            total_cells=total,
            missing_cells=self.missing,
            duplicate_rows=duplicates,
            invalid_cells=invalid,
            whitespace_cells=self.whitespace,
        )


def compute_quality(df: pd.DataFrame) -> QualityResult:
    accumulator = QualityAccumulator()
    accumulator.update(df)
    return accumulator.finalize()


def compute_quality_batches(batches: Iterable[pd.DataFrame]) -> QualityResult:
    accumulator = QualityAccumulator()
    for batch in batches:
        accumulator.update(batch)
    return accumulator.finalize()
