"""Chart selection, validation, figures and standalone HTML reports."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from core.errors import ValidationError
from services import chart_service
from services.chart_service import (
    CHART_TYPES,
    available_chart_types,
    build_figure,
    figure_to_html,
    report_html,
    suggest_chart,
    validate_chart_spec,
)

SCRIPT = "<script>alert(1)</script>"


@pytest.fixture
def by_region() -> pd.DataFrame:
    return pd.DataFrame({"region": ["North", "South", "East"], "total_revenue": [3800.0, 1740.5, 300.25]})


@pytest.fixture
def monthly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"]),
            "revenue": [300.0, 100.0, 200.0],
        }
    )


@pytest.fixture
def region_product() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "North", "South", "South"],
            "product": ["A", "B", "A", "B"],
            "revenue": [10.0, 20.0, 30.0, 40.0],
        }
    )


@pytest.fixture
def numeric_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = rng.normal(size=40)
    return pd.DataFrame({"a": base, "b": base * 2 + rng.normal(size=40) * 0.1, "c": rng.normal(size=40)})


def correlation_table(frame: pd.DataFrame, *, label_column: bool) -> pd.DataFrame:
    matrix = frame.corr()
    return matrix.reset_index().rename(columns={"index": "column"}) if label_column else matrix


# ---------------------------------------------------------------------------
# suggest_chart and available_chart_types
# ---------------------------------------------------------------------------


def test_chart_types_constant_lists_every_supported_type():
    assert CHART_TYPES == (
        "bar", "grouped_bar", "stacked_bar", "line", "area", "pie", "donut",
        "histogram", "scatter", "box", "heatmap", "correlation",
    )  # fmt: skip


def test_category_and_numeric_suggests_bar(by_region):
    spec = suggest_chart(by_region)
    assert spec == {
        "type": "bar", "x": "region", "y": "total_revenue", "color": None,
        "title": "Total revenue by Region", "stacked": False,
    }  # fmt: skip


def test_datetime_axis_suggests_line(monthly):
    spec = suggest_chart(monthly)
    assert (spec["type"], spec["x"], spec["y"]) == ("line", "order_date", "revenue")


def test_two_categories_suggest_grouped_bar(region_product):
    spec = suggest_chart(region_product)
    assert (spec["type"], spec["x"], spec["color"], spec["y"]) == (
        "grouped_bar",
        "region",
        "product",
        "revenue",
    )


def test_square_matrix_suggests_correlation_in_both_layouts(numeric_frame):
    for label_column in (True, False):
        spec = suggest_chart(correlation_table(numeric_frame, label_column=label_column))
        assert spec is not None and spec["type"] == "correlation"


def test_single_numeric_column_suggests_histogram(numeric_frame):
    spec = suggest_chart(numeric_frame[["a"]])
    assert (spec["type"], spec["x"]) == ("histogram", "a")


def test_two_numeric_columns_suggest_scatter(numeric_frame):
    spec = suggest_chart(numeric_frame[["a", "c"]])
    assert (spec["type"], spec["x"], spec["y"]) == ("scatter", "a", "c")


def test_identifier_columns_are_not_chosen_as_values():
    table = pd.DataFrame({"order_id": [1, 2, 3], "region": ["N", "S", "E"], "revenue": [5.0, 6.0, 7.0]})
    spec = suggest_chart(table)
    assert (spec["type"], spec["x"], spec["y"]) == ("bar", "region", "revenue")


def test_year_column_gives_a_line():
    table = pd.DataFrame({"year": [2022, 2023, 2024], "revenue": [1.0, 2.0, 3.0]})
    spec = suggest_chart(table)
    assert (spec["type"], spec["x"]) == ("line", "year")


def test_trend_operation_on_period_labels_gives_line():
    table = pd.DataFrame({"period": ["2024-01", "2024-02", "2024-03"], "revenue": [1.0, 2.0, 3.0]})
    assert suggest_chart(table, operation="monthly_trend")["type"] == "line"
    assert suggest_chart(table)["type"] == "bar"


def test_pie_only_for_share_questions_with_few_non_negative_slices(by_region):
    assert suggest_chart(by_region, operation="share_of_total")["type"] == "pie"
    negative = by_region.assign(total_revenue=[10.0, -5.0, 3.0])
    assert suggest_chart(negative, operation="share_of_total")["type"] == "bar"
    many = pd.DataFrame({"k": [f"c{i}" for i in range(13)], "v": [1.0] * 13})
    assert suggest_chart(many, operation="share_of_total")["type"] == "bar"


def test_correlation_operation_on_raw_numeric_columns(numeric_frame):
    spec = suggest_chart(numeric_frame, operation="correlation")
    assert spec["type"] == "correlation"


def test_valid_hint_wins_and_is_normalized(by_region):
    spec = suggest_chart(
        by_region, hint={"type": "donut", "x": "region", "y": "total_revenue", "title": "Mix"}
    )
    assert (spec["type"], spec["title"], spec["color"], spec["stacked"]) == ("donut", "Mix", None, False)


def test_hint_without_columns_gets_defaults(by_region):
    spec = suggest_chart(by_region, hint={"type": "pie"})
    assert (spec["x"], spec["y"]) == ("region", "total_revenue")


def test_invalid_hint_falls_back_to_the_automatic_choice(by_region):
    for hint in (
        {"type": "sparkline"},
        {"type": "bar", "x": "missing", "y": "total_revenue"},
        {"type": "scatter"},
    ):
        assert suggest_chart(by_region, hint=hint)["type"] == "bar"


def test_nothing_sensible_returns_none():
    assert suggest_chart(pd.DataFrame()) is None
    assert (
        suggest_chart(
            pd.DataFrame({"region": pd.Series([], dtype="string"), "v": pd.Series([], dtype="float64")})
        )
        is None
    )
    assert suggest_chart(pd.DataFrame({"total": [1234.5]})) is None
    assert suggest_chart(pd.DataFrame({"name": ["a", "b"], "note": ["x", "y"]})) is None


def test_available_chart_types_follow_dtypes(by_region, monthly, region_product, numeric_frame):
    assert available_chart_types(by_region) == ["bar", "line", "area", "pie", "donut", "histogram", "box"]
    assert available_chart_types(monthly) == ["bar", "line", "area", "pie", "donut", "histogram", "box"]
    assert available_chart_types(region_product) == [
        "bar", "grouped_bar", "stacked_bar", "line", "area", "pie", "donut", "histogram", "box", "heatmap",
    ]  # fmt: skip
    numeric = available_chart_types(numeric_frame)
    assert numeric == ["line", "area", "histogram", "scatter", "box", "correlation"]
    text_only = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    assert available_chart_types(text_only) == []
    assert available_chart_types(pd.DataFrame()) == []


def test_available_types_exclude_pie_for_negative_or_large_tables(by_region):
    assert "pie" not in available_chart_types(by_region.assign(total_revenue=[1.0, -2.0, 3.0]))
    big = pd.DataFrame({"k": [f"c{i}" for i in range(21)], "v": np.arange(21, dtype="float64") + 1})
    assert "pie" not in available_chart_types(big) and "bar" in available_chart_types(big)


def test_available_types_include_heatmap_and_correlation_for_a_matrix(numeric_frame):
    types = available_chart_types(correlation_table(numeric_frame, label_column=True))
    assert "heatmap" in types and "correlation" in types


# ---------------------------------------------------------------------------
# validate_chart_spec
# ---------------------------------------------------------------------------


def test_validate_fills_defaults_and_forces_stacking_flags(region_product):
    stacked = validate_chart_spec(region_product, {"type": "stacked_bar"})
    assert (stacked["x"], stacked["color"], stacked["y"], stacked["stacked"]) == (
        "region",
        "product",
        "revenue",
        True,
    )
    grouped = validate_chart_spec(region_product, {"type": "grouped_bar", "stacked": True})
    assert grouped["stacked"] is False
    assert validate_chart_spec(region_product, {"type": "bar", "stacked": True})["stacked"] is True


def test_validate_trims_and_bounds_titles(by_region):
    assert validate_chart_spec(by_region, {"type": "bar", "title": "  Sales  "})["title"] == "Sales"
    assert len(validate_chart_spec(by_region, {"type": "bar", "title": "t" * 500})["title"]) == 200


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ("bar", "not valid"),
        ({"type": "sparkline"}, "Unknown chart type"),
        ({}, "Unknown chart type"),
        ({"type": "bar", "x": "nope", "y": "total_revenue"}, "not in the result table"),
        ({"type": "bar", "x": "region", "y": "region"}, "must contain numbers"),
        ({"type": "bar", "x": "total_revenue", "y": "total_revenue"}, "different columns"),
        ({"type": "bar", "x": 5, "y": "total_revenue"}, "not in the result table"),
        ({"type": "scatter"}, "two numeric columns"),
        ({"type": "histogram", "x": "region"}, "must contain numbers"),
        ({"type": "box", "y": "region"}, "must contain numbers"),
        ({"type": "grouped_bar"}, "second category"),
        ({"type": "heatmap"}, "heatmap needs"),
        ({"type": "correlation"}, "at least two numeric"),
    ],
)
def test_validate_rejects_bad_specs(by_region, spec, fragment):
    with pytest.raises(ValidationError) as caught:
        validate_chart_spec(by_region, spec)
    assert fragment in caught.value.user_message


def test_validate_rejects_empty_tables_and_duplicate_columns():
    with pytest.raises(ValidationError):
        validate_chart_spec(pd.DataFrame(), {"type": "bar"})
    duplicated = pd.DataFrame([[1.0, 2.0]], columns=["v", "v"])
    with pytest.raises(ValidationError, match="duplicate column"):
        validate_chart_spec(duplicated, {"type": "histogram"})


def test_pie_rejects_negative_values_zero_totals_and_too_many_slices(by_region):
    with pytest.raises(ValidationError, match="negative"):
        validate_chart_spec(by_region.assign(total_revenue=[5.0, -1.0, 2.0]), {"type": "pie"})
    with pytest.raises(ValidationError, match="above zero"):
        validate_chart_spec(by_region.assign(total_revenue=[0.0, 0.0, 0.0]), {"type": "donut"})
    twenty = pd.DataFrame({"k": [f"c{i}" for i in range(20)], "v": [1.0] * 20})
    assert validate_chart_spec(twenty, {"type": "pie"})["type"] == "pie"
    twenty_one = pd.DataFrame({"k": [f"c{i}" for i in range(21)], "v": [1.0] * 21})
    with pytest.raises(ValidationError, match="at most 20 slices"):
        validate_chart_spec(twenty_one, {"type": "pie"})


def test_heatmap_rejects_duplicate_category_pairs(region_product):
    duplicated = pd.concat([region_product, region_product.iloc[:1]], ignore_index=True)
    with pytest.raises(ValidationError, match="one value for each pair"):
        validate_chart_spec(duplicated, {"type": "heatmap"})


def test_validation_message_does_not_echo_long_column_names(by_region):
    with pytest.raises(ValidationError) as caught:
        validate_chart_spec(by_region, {"type": "bar", "x": "c" * 300, "y": "total_revenue"})
    assert len(caught.value.user_message) < 120


# ---------------------------------------------------------------------------
# build_figure
# ---------------------------------------------------------------------------


def test_bar_figure_has_titles_labels_and_a_zero_based_axis(by_region):
    fig = build_figure(by_region, {"type": "bar", "x": "region", "y": "total_revenue"})
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "bar"
    assert fig.layout.title.text == "Total revenue by Region"
    assert fig.layout.xaxis.title.text == "Region"
    assert fig.layout.yaxis.title.text == "Total revenue"
    assert fig.layout.yaxis.rangemode == "tozero"
    assert list(fig.data[0].x) == ["North", "South", "East"]
    assert list(fig.data[0].y) == [3800.0, 1740.5, 300.25]
    assert "%{y" in fig.data[0].hovertemplate and "Region" in fig.data[0].hovertemplate
    assert fig.layout.showlegend is False


def test_all_figures_share_one_template(by_region, monthly, numeric_frame):
    figures = [
        build_figure(by_region, {"type": "bar"}),
        build_figure(monthly, {"type": "line"}),
        build_figure(numeric_frame, {"type": "histogram"}),
    ]
    assert {f.layout.template.layout.colorway is not None for f in figures} == {True}
    assert {tuple(f.layout.colorway) for f in figures} == {tuple(chart_service._COLORWAY)}


def test_grouped_and_stacked_bars_make_one_trace_per_series_with_a_legend(region_product):
    grouped = build_figure(region_product, {"type": "grouped_bar"})
    assert [t.name for t in grouped.data] == ["A", "B"]
    assert grouped.layout.barmode == "group" and grouped.layout.showlegend is True
    assert grouped.layout.legend.title.text == "Product"
    stacked = build_figure(region_product, {"type": "stacked_bar"})
    assert stacked.layout.barmode == "stack" and stacked.layout.yaxis.rangemode == "tozero"


def test_line_chart_is_sorted_by_x(monthly):
    fig = build_figure(monthly, {"type": "line", "x": "order_date", "y": "revenue"})
    assert fig.data[0].type == "scatter" and fig.data[0].mode == "lines+markers"
    assert list(fig.data[0].x) == ["2024-01-01", "2024-02-01", "2024-03-01"]
    assert list(fig.data[0].y) == [100.0, 200.0, 300.0]
    assert fig.layout.xaxis.type == "date"


def test_area_chart_fills_to_zero_and_starts_at_zero(monthly):
    fig = build_figure(monthly, {"type": "area"})
    assert fig.data[0].fill == "tozeroy" and fig.layout.yaxis.rangemode == "tozero"


def test_multi_series_line_and_stacked_area():
    table = pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02"]),
            "channel": ["web", "web", "shop", "shop"],
            "sales": [2.0, 1.0, 5.0, 6.0],
        }
    )
    line = build_figure(table, {"type": "line", "x": "day", "y": "sales", "color": "channel"})
    assert [t.name for t in line.data] == ["web", "shop"]
    assert list(line.data[0].y) == [1.0, 2.0]
    area = build_figure(
        table, {"type": "area", "x": "day", "y": "sales", "color": "channel", "stacked": True}
    )
    assert {t.stackgroup for t in area.data} == {"one"}


def test_pie_and_donut(by_region):
    pie = build_figure(by_region, {"type": "pie"})
    assert pie.data[0].type == "pie" and not pie.data[0].hole
    assert list(pie.data[0].labels) == ["North", "South", "East"]
    donut = build_figure(by_region, {"type": "donut"})
    assert donut.data[0].hole == pytest.approx(0.45)


def test_pie_rejects_negatives_when_building(by_region):
    with pytest.raises(ValidationError, match="negative"):
        build_figure(by_region.assign(total_revenue=[1.0, -1.0, 2.0]), {"type": "pie"})


def test_histogram_scatter_and_box(numeric_frame):
    hist = build_figure(numeric_frame, {"type": "histogram", "x": "a"})
    assert hist.data[0].type == "histogram"
    assert hist.layout.yaxis.title.text == "Count" and hist.layout.xaxis.title.text == "A"
    scatter = build_figure(numeric_frame, {"type": "scatter", "x": "a", "y": "b"})
    assert scatter.data[0].type == "scatter" and scatter.data[0].mode == "markers"
    assert (scatter.layout.xaxis.title.text, scatter.layout.yaxis.title.text) == ("A", "B")
    box = build_figure(numeric_frame, {"type": "box", "y": "c"})
    assert box.data[0].type == "box" and box.layout.yaxis.title.text == "C"


def test_scatter_colour_by_number_and_by_category():
    table = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0], "size": [5.0, 6.0, 7.0], "kind": ["a", "b", "a"]}
    )
    numeric = build_figure(table, {"type": "scatter", "x": "x", "y": "y", "color": "size"})
    assert numeric.data[0].marker.showscale is True and len(numeric.data) == 1
    categorical = build_figure(table, {"type": "scatter", "x": "x", "y": "y", "color": "kind"})
    assert [t.name for t in categorical.data] == ["a", "b"]


def test_box_grouped_by_category_and_series():
    table = pd.DataFrame({"g": ["a", "a", "b", "b"], "s": ["u", "v", "u", "v"], "v": [1.0, 2.0, 3.0, 4.0]})
    grouped = build_figure(table, {"type": "box", "x": "g", "y": "v"})
    assert list(grouped.data[0].x) == ["a", "a", "b", "b"]
    series = build_figure(table, {"type": "box", "x": "g", "y": "v", "color": "s"})
    assert [t.name for t in series.data] == ["u", "v"] and series.layout.boxmode == "group"


def test_correlation_matrix_is_a_annotated_heatmap_from_both_layouts(numeric_frame):
    for label_column in (True, False):
        fig = build_figure(
            correlation_table(numeric_frame, label_column=label_column), {"type": "correlation"}
        )
        trace = fig.data[0]
        assert trace.type == "heatmap" and trace.texttemplate == "%{z:.2f}"
        assert (trace.zmin, trace.zmax) == (-1, 1)
        assert list(trace.x) == ["a", "b", "c"] and list(trace.y) == ["a", "b", "c"]
        assert np.allclose(np.diag(np.array(trace.z, dtype=float)), 1.0)
        assert fig.layout.title.text == "Correlation matrix"


def test_correlation_is_computed_from_raw_numeric_columns(numeric_frame):
    fig = build_figure(numeric_frame, {"type": "correlation"})
    expected = numeric_frame.corr().to_numpy()
    assert np.allclose(np.array(fig.data[0].z, dtype=float), expected)


def test_heatmap_pivots_two_categories_and_shows_values(region_product):
    fig = build_figure(region_product, {"type": "heatmap"})
    trace = fig.data[0]
    assert trace.type == "heatmap" and trace.texttemplate
    assert list(trace.x) == ["North", "South"] and list(trace.y) == ["A", "B"]
    assert np.array(trace.z, dtype=float).tolist() == [[10.0, 30.0], [20.0, 40.0]]
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("Region", "Product")


def test_heatmap_of_a_square_matrix_without_columns(numeric_frame):
    table = correlation_table(numeric_frame, label_column=True)
    fig = build_figure(table, {"type": "heatmap"})
    assert fig.data[0].type == "heatmap" and list(fig.data[0].x) == ["a", "b", "c"]


def test_large_heatmaps_skip_cell_annotations():
    names = [f"c{i}" for i in range(25)]
    table = pd.DataFrame(np.eye(25), columns=names).corr()
    fig = build_figure(table, {"type": "correlation"})
    assert fig.data[0].texttemplate is None


def test_empty_and_all_null_tables_return_none(by_region):
    assert build_figure(pd.DataFrame(), {"type": "bar"}) is None
    assert build_figure(by_region.iloc[0:0], {"type": "bar"}) is None
    all_null = by_region.assign(total_revenue=np.nan)
    for chart_type in ("bar", "line", "pie", "histogram", "box"):
        assert build_figure(all_null, {"type": chart_type}) is None, chart_type
    constant = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
    assert build_figure(constant, {"type": "correlation"}) is None


def test_missing_values_are_dropped_and_noted_in_the_subtitle(by_region):
    table = by_region.assign(total_revenue=[3800.0, np.nan, 300.25])
    fig = build_figure(table, {"type": "bar"})
    assert list(fig.data[0].x) == ["North", "East"]
    assert "1 row(s) with missing values were left out." == fig.layout.title.subtitle.text
    clean = build_figure(by_region, {"type": "bar"})
    assert clean.layout.title.subtitle.text is None


def test_nullable_pandas_dtypes_are_supported():
    table = pd.DataFrame(
        {
            "region": pd.array(["N", "S", None], dtype="string"),
            "qty": pd.array([3, 2, 5], dtype="Int64"),
        }
    )
    fig = build_figure(table, {"type": "bar"})
    assert list(fig.data[0].x) == ["N", "S"] and list(fig.data[0].y) == [3, 2]


def test_untrusted_labels_are_escaped_inside_figures():
    table = pd.DataFrame({"<b>name</b>": [SCRIPT, "ok"], "v%{x}": [1.0, 2.0]})
    fig = build_figure(table, {"type": "bar", "title": SCRIPT})
    assert SCRIPT not in fig.to_json()
    assert "&lt;script&gt;" in fig.layout.title.text
    assert "V% {x}: %{y" in fig.data[0].hovertemplate
    assert list(fig.data[0].x)[0].startswith("&lt;script")


def test_invalid_spec_raises_from_build_figure(by_region):
    with pytest.raises(ValidationError):
        build_figure(by_region, {"type": "bar", "x": "region", "y": "region"})


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------


def test_figure_to_html_is_standalone_and_escapes_the_title(by_region):
    fig = build_figure(by_region, {"type": "bar"})
    page = figure_to_html(fig, title=f"Sales {SCRIPT}")
    assert page.startswith("<!doctype html>")
    assert SCRIPT not in page and "Sales &lt;script&gt;" in page
    assert "plotly-graph-div" in page and "<script src" not in page and 'src="http' not in page
    assert len(page) > 1_000_000  # Plotly.js is embedded, so the page works offline


def make_report(**overrides) -> str:
    table = pd.DataFrame(
        {"region": [SCRIPT, "South"], "revenue": [1234.5, 99.0], "when": pd.to_datetime(["2024-01-01", None])}
    )
    values = {
        "title": f"Revenue {SCRIPT}",
        "question": f"What is revenue? {SCRIPT}",
        "answer_text": f"Revenue is 1,333.5.\n\nSecond paragraph {SCRIPT}",
        "table": table,
        "chart_spec": {"type": "bar", "x": "region", "y": "revenue"},
        "facts": [{"label": f"Fact {SCRIPT}", "value": SCRIPT}],
        "generated_at": datetime(2024, 5, 6, 7, 8),
        "dataset_name": f"Sales {SCRIPT}",
    }
    values.update(overrides)
    return report_html(**values)


def test_report_escapes_every_untrusted_text():
    page = make_report()
    assert SCRIPT not in page
    assert page.count("&lt;script&gt;alert(1)&lt;/script&gt;") >= 7
    assert "<script>alert" not in page and "<script >alert" not in page


def test_report_has_all_sections_embeds_one_chart_and_works_offline():
    page = make_report()
    for heading in (
        "<h2>Question</h2>",
        "<h2>Answer</h2>",
        "<h2>Key facts</h2>",
        "<h2>Chart</h2>",
        "<h2>Result table</h2>",
    ):
        assert heading in page
    assert page.count("plotly-graph-div") == 1
    assert "Plotly" in page and 'src="http' not in page
    assert "Generated 2024-05-06 07:08 UTC" in page
    assert "<th>region</th>" in page and '<td class="num">1234.5</td>' in page
    assert page.count("<p>") == 3  # question plus two answer paragraphs


def test_report_without_chart_facts_or_answer_is_still_valid_html():
    page = make_report(chart_spec=None, facts=[], answer_text="")
    assert "<h2>Chart</h2>" not in page and "<h2>Key facts</h2>" not in page
    assert "No written answer." in page and "plotly-graph-div" not in page
    assert len(page) < 20_000


def test_report_explains_why_a_chart_is_missing():
    page = make_report(chart_spec={"type": "scatter", "x": "region", "y": "revenue"})
    assert "must contain numbers" in page and "plotly-graph-div" not in page
    all_null = make_report(table=pd.DataFrame({"region": ["a"], "revenue": [np.nan]}))
    assert "could not be drawn" in all_null


def test_report_with_empty_table():
    page = make_report(table=pd.DataFrame(), chart_spec={"type": "bar"})
    assert "The result table is empty." in page
