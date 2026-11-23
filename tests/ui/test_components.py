"""Shared UI components: result display rounding and dataset picker labels."""

from __future__ import annotations

import math

import pandas as pd

from components.chart_components import display_frame


def test_display_frame_hides_float_noise_without_touching_other_columns():
    """Regression (live browser check): a computed total of 353358.62999999995 was shown raw."""
    table = pd.DataFrame(
        {
            "sum_revenue": [353358.62999999995, float("nan"), 1e-9, 1234567890123.45, 0.1 + 0.2],
            "count": pd.array([1, 2, None, 4, 5], dtype="Int64"),
            "region": pd.array(["North", None, "East", "West", "South"], dtype="string"),
        }
    )
    shown = display_frame(table)
    assert shown["sum_revenue"].iloc[0] == 353358.63
    assert math.isnan(shown["sum_revenue"].iloc[1]) and shown["sum_revenue"].iloc[2] == 1e-9
    assert shown["sum_revenue"].iloc[3] == 1234567890123.45  # large totals keep every digit (JPY, INR)
    assert shown["sum_revenue"].iloc[4] == 0.3
    pd.testing.assert_series_equal(shown["count"], table["count"])
    pd.testing.assert_series_equal(shown["region"], table["region"])
    assert table["sum_revenue"].iloc[0] == 353358.62999999995  # the input is not modified
    assert display_frame(None) is None


def test_dataset_picker_label_keeps_one_decimal(make_app, factory, org_id):
    """Regression (live browser check): quality 99.6 was labelled 'quality 100'."""
    factory.user(org_id, "org_admin", username="boss")
    from auth import authentication

    ctx = authentication.login("boss", __import__("tests.conftest", fromlist=["DEFAULT_PASSWORD"]).DEFAULT_PASSWORD).ctx
    frame = pd.DataFrame({"a": pd.array(["x", "x", "y"], dtype="string")})  # one duplicate row -> score below 100
    factory.dataset(ctx, frame, "dupes")
    app = make_app("boss", page="views/data/quality.py")
    labels = [str(option) for option in app.selectbox[0].options]
    assert any("quality 9" in label and "." in label.split("quality ")[1] for label in labels), labels
