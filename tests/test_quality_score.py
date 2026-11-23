"""Quality score contract: 100 * (0.40 completeness + 0.20 uniqueness + 0.25 validity + 0.15 consistency)."""

from __future__ import annotations

import pandas as pd
import pytest

from services import dataset_service, profiling_service
from services.quality_score import (
    WEIGHTS,
    compute_quality,
    compute_quality_batches,
    count_invalid_text,
    text_parse_counts,
)


def test_weights_follow_the_documented_formula():
    assert WEIGHTS == {"completeness": 0.40, "uniqueness": 0.20, "validity": 0.25, "consistency": 0.15}
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_clean_frame_scores_100_and_each_problem_lowers_its_own_part():
    clean = pd.DataFrame({"a": pd.array([1, 2, 3, 4], dtype="Int64"), "b": pd.array(list("wxyz"), dtype="string")})
    assert compute_quality(clean).score == 100.0
    missing = clean.copy()
    missing.loc[0, "a"] = pd.NA
    result = compute_quality(missing)
    assert result.missing_cells == 1 and result.completeness == 0.875
    assert result.score == round(100 * (0.40 * 0.875 + 0.20 + 0.25 + 0.15), 1)
    twin = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)
    result = compute_quality(twin)
    assert result.duplicate_rows == 1 and result.uniqueness == 0.8
    padded = clean.copy()
    padded.loc[0, "b"] = " w"
    result = compute_quality(padded)
    assert result.whitespace_cells == 1 and result.consistency == 0.875


def test_batches_give_the_same_result_as_one_frame():
    frame = pd.DataFrame({"a": pd.array([1, 2, 1, 2, 3], dtype="Int64"), "x": [0.5, 1.0, 0.5, 1.0, -0.0]})
    batches = [frame.iloc[:2], frame.iloc[2:4], frame.iloc[4:]]
    assert compute_quality_batches(batches) == compute_quality(frame)
    assert compute_quality(frame).duplicate_rows == 2


@pytest.mark.parametrize("frame", [pd.DataFrame({"a": pd.array([], dtype="Int64")}), pd.DataFrame()])
def test_empty_frame_is_not_scored_as_maximally_broken(frame):
    result = compute_quality(frame)
    assert result.score == 100.0
    assert (result.completeness, result.uniqueness, result.validity, result.consistency) == (1.0, 1.0, 1.0, 1.0)
    assert result.total_cells == 0 and result.missing_cells == 0 and result.duplicate_rows == 0


def test_header_only_upload_is_not_critical(admin_ctx):
    result = dataset_service.import_dataset(admin_ctx, file_name="h.csv", data=b"a,b,c\n", dataset_name="h")
    health = profiling_service.health_check(admin_ctx, result["dataset_id"])
    assert health["quality_score"] == 100.0
    assert health["status"] == "empty"


def test_invalid_text_counting_rules():
    assert text_parse_counts(pd.Series(["1", " 2.5 ", "x", None], dtype="string")) == (3, 2, 0)
    # Mostly numeric (4 of 5): the odd one out is invalid; half numeric is not a numeric column.
    assert count_invalid_text(5, 4, 0) == 1
    assert count_invalid_text(5, 2, 0) == 0
    assert count_invalid_text(5, 5, 0) == 0  # fully numeric text has nothing invalid
    # A column known to have been numeric or a date: everything that does not parse is invalid.
    assert count_invalid_text(5, 2, 0, expected="float") == 3
    assert count_invalid_text(4, 0, 1, expected="datetime") == 3
    assert count_invalid_text(0, 0, 0, expected="float") == 0
