"""Charts and standalone HTML reports for result tables.

Everything here is a pure function over a result table that the caller has
already authorized: there is no database or file access. Chart choices follow
the data (dtypes and shape) so a chart never claims more than the table
contains: bars and areas start at zero, lines are sorted by x, pie and donut
charts only show a few non-negative slices, rows with missing values are left
out with a visible note, and there are no 3D or dual-axis charts.

Untrusted text (column names, cell values, questions) is HTML-escaped before it
reaches Plotly, which renders a small HTML subset in labels, and before it is
placed into report pages.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.errors import ValidationError
from services.frame_utils import json_safe

CHART_TYPES: tuple[str, ...] = (
    "bar",
    "grouped_bar",
    "stacked_bar",
    "line",
    "area",
    "pie",
    "donut",
    "histogram",
    "scatter",
    "box",
    "heatmap",
    "correlation",
)

MAX_PIE_SLICES = 20
SUGGESTED_PIE_SLICES = 12
_MAX_ANNOTATED_CELLS = 400
_MAX_MARKER_POINTS = 60
_MAX_TITLE_CHARS = 200
_COLORWAY = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]
_IDENTIFIER_NAME = re.compile(
    r"(?:^|[_\s-])(?:id|ids|index|code|year|month|quarter|week|day|rank|number|no)$", re.I
)
_TIME_NAME = re.compile(r"(?:^|[_\s-])(?:year|month|quarter|week|day|date)$", re.I)
_SHARE_OPERATION = re.compile(r"share|proportion|composition|percent|breakdown", re.I)
_TREND_OPERATION = re.compile(r"trend|growth|time|series|month|monthly|daily|weekly|yearly", re.I)


# ---------------------------------------------------------------------------
# Table shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """Column roles of a result table, derived from dtypes and names only."""

    datetimes: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    numerics: list[str] = field(default_factory=list)
    measures: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    @property
    def dims(self) -> list[str]:
        """Columns that can sit on a category or time axis."""
        return self.datetimes + self.categories + self.labels


def _is_plain_numeric(dtype: Any) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def _shape(table: pd.DataFrame) -> _Shape:
    datetimes: list[str] = []
    categories: list[str] = []
    numerics: list[str] = []
    for name, dtype in table.dtypes.items():
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetimes.append(name)
        elif _is_plain_numeric(dtype):
            numerics.append(name)
        else:
            categories.append(name)
    measures = [n for n in numerics if not _IDENTIFIER_NAME.search(str(n))]
    if not measures:
        measures = list(numerics)
    labels = [n for n in numerics if n not in measures]
    return _Shape(datetimes, categories, numerics, measures, labels)


def _has_rows(table: pd.DataFrame | None) -> bool:
    return table is not None and len(table) > 0 and table.shape[1] > 0


def _square_matrix(table: pd.DataFrame, shape: _Shape) -> pd.DataFrame | None:
    """Return the table as a labelled square matrix (such as a correlation matrix), else None."""
    count = len(shape.numerics)
    if count < 2 or len(table) != count:
        return None
    names = [str(n) for n in shape.numerics]
    if table.shape[1] == count:
        index = [str(v) for v in table.index]
        if index != names:
            return None
        values = table[shape.numerics].astype("float64")
        values.index, values.columns = names, names
        return values
    if table.shape[1] == count + 1 and len(shape.categories) == 1:
        labels = table[shape.categories[0]].astype("string")
        if labels.isna().any() or labels.duplicated().any() or set(labels) != set(names):
            return None
        values = table[shape.numerics].astype("float64")
        values.index, values.columns = list(labels), names
        return values.reindex(names)
    return None


def _non_negative(series: pd.Series) -> bool:
    return bool((series.dropna() >= 0).all())


# ---------------------------------------------------------------------------
# Available types and suggestions
# ---------------------------------------------------------------------------


def available_chart_types(table: pd.DataFrame) -> list[str]:
    """Chart types that are valid for the table's columns and rows (for a manual selector)."""
    if not _has_rows(table):
        return []
    shape = _shape(table)
    dims, measures = len(shape.dims), len(shape.measures)
    matrix = _square_matrix(table, shape) is not None
    pie_ok = (
        measures >= 1
        and dims >= 1
        and len(table) <= MAX_PIE_SLICES
        and any(_non_negative(table[m]) and table[m].sum() > 0 for m in shape.measures)
    )
    rules = {
        "bar": measures >= 1 and dims >= 1,
        "grouped_bar": measures >= 1 and dims >= 2,
        "stacked_bar": measures >= 1 and dims >= 2,
        "line": measures >= 1 and (dims >= 1 or len(shape.numerics) >= 2),
        "area": measures >= 1 and (dims >= 1 or len(shape.numerics) >= 2),
        "pie": pie_ok,
        "donut": pie_ok,
        "histogram": len(shape.numerics) >= 1,
        "scatter": len(shape.numerics) >= 2,
        "box": len(shape.numerics) >= 1,
        "heatmap": matrix or (measures >= 1 and dims >= 2),
        "correlation": matrix or measures >= 2,
    }
    return [chart_type for chart_type in CHART_TYPES if rules[chart_type]]


def suggest_chart(
    table: pd.DataFrame, *, operation: str | None = None, hint: dict | None = None
) -> dict | None:
    """Choose a chart spec for a result table, or None when no chart would be meaningful.

    A valid ``hint`` (the chart spec of the analysis plan) wins. Otherwise the
    choice follows the operation and the column types: a time axis gives a
    line, one category gives bars, two categories give grouped bars, a square
    numeric matrix gives a correlation heatmap and a single column of raw
    numbers gives a histogram. Pie and donut charts are only suggested for
    share-like questions with at most 12 non-negative slices.
    """
    if not _has_rows(table):
        return None
    if hint:
        try:
            return validate_chart_spec(table, hint)
        except ValidationError:
            pass
    candidate = _automatic_spec(table, (operation or "").strip())
    if candidate is None:
        return None
    try:
        return validate_chart_spec(table, candidate)
    except ValidationError:
        return None


def _automatic_spec(table: pd.DataFrame, operation: str) -> dict | None:
    shape = _shape(table)
    if len(table) < 2 or not shape.measures:
        return None
    if _square_matrix(table, shape) is not None or (
        "correl" in operation.lower() and len(shape.measures) >= 2
    ):
        return {"type": "correlation"}
    y = shape.measures[0]
    # Identifier-like numbers (ids, years) only serve as an axis when no real category or date exists.
    dims = [d for d in shape.datetimes + shape.categories if d != y] or [d for d in shape.labels if d != y]
    if shape.datetimes:
        x = shape.datetimes[0]
        others = [d for d in dims if d != x]
        return {"type": "line", "x": x, "y": y, "color": (shape.categories or others or [None])[0]}
    if len(dims) >= 2:
        return {"type": "grouped_bar", "x": dims[0], "y": y, "color": dims[1]}
    if dims:
        return _single_dimension_spec(table, dims[0], y, operation)
    if len(shape.numerics) >= 2:
        return {
            "type": "scatter",
            "x": shape.measures[0],
            "y": shape.measures[1] if len(shape.measures) > 1 else shape.numerics[1],
        }
    return {"type": "histogram", "x": y}


def _single_dimension_spec(table: pd.DataFrame, x: str, y: str, operation: str) -> dict:
    if (
        _SHARE_OPERATION.search(operation)
        and len(table) <= SUGGESTED_PIE_SLICES
        and _non_negative(table[y])
        and table[y].sum() > 0
    ):
        return {"type": "pie", "x": x, "y": y}
    if _TIME_NAME.search(str(x)) or _TREND_OPERATION.search(operation):
        return {"type": "line", "x": x, "y": y}
    return {"type": "bar", "x": x, "y": y}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def humanize(name: object) -> str:
    """Readable label for a column name: underscores become spaces, first letter capitalised."""
    text = re.sub(r"[_\s]+", " ", str(name)).strip()
    return text[:1].upper() + text[1:]


def _short(name: object) -> str:
    text = str(name)
    return text if len(text) <= 60 else text[:57] + "..."


def _column(table: pd.DataFrame, spec: dict, key: str) -> str | None:
    value = spec.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value not in table.columns:
        raise ValidationError(f"The column '{_short(value)}' is not in the result table.")
    return value


def _require_numeric(table: pd.DataFrame, column: str, role: str) -> None:
    if not _is_plain_numeric(table[column].dtype):
        raise ValidationError(f"The {role} column '{_short(column)}' must contain numbers for this chart.")


def _first(options: list[str], message: str) -> str:
    if not options:
        raise ValidationError(message)
    return options[0]


def _auto_title(chart_type: str, x: str | None, y: str | None, color: str | None) -> str:
    hx, hy = humanize(x) if x else "", humanize(y) if y else ""
    if chart_type == "histogram":
        return f"Distribution of {hx}"
    if chart_type == "scatter":
        return f"{hy} vs {hx}"
    if chart_type == "box":
        return f"{hy} by {hx}" if x else f"Distribution of {hy}"
    if chart_type == "correlation":
        return "Correlation matrix"
    if chart_type == "heatmap":
        return f"{hy} by {hx} and {humanize(color)}" if x else "Heatmap"
    if chart_type in ("pie", "donut"):
        return f"{hy} share by {hx}"
    if chart_type in ("line", "area"):
        return f"{hy} over {hx}"
    suffix = f" and {humanize(color)}" if color else ""
    return f"{hy} by {hx}{suffix}"


def validate_chart_spec(table: pd.DataFrame, spec: dict) -> dict:
    """Check a chart spec against the table and return it with defaults filled in.

    The result always has the keys type, x, y, color, title and stacked.
    Missing columns are chosen from the data. Raises ``ValidationError`` for an
    unknown type, a missing column, a non-numeric value column, negative
    values in a pie or donut chart, more than 20 pie slices and similar
    problems.
    """
    if not isinstance(spec, dict):
        raise ValidationError("The chart settings are not valid.")
    chart_type = spec.get("type")
    if chart_type not in CHART_TYPES:
        raise ValidationError(f"Unknown chart type. Choose one of: {', '.join(CHART_TYPES)}.")
    if not _has_rows(table):
        raise ValidationError("There is no data to chart.")
    if table.columns.duplicated().any():
        raise ValidationError("The result table has duplicate column names and cannot be charted.")
    x, y, color = (_column(table, spec, key) for key in ("x", "y", "color"))
    shape = _shape(table)
    x, y, color = _PLANNERS[chart_type](table, shape, x, y, color)
    title = spec.get("title")
    title = title.strip()[:_MAX_TITLE_CHARS] if isinstance(title, str) and title.strip() else ""
    return {
        "type": chart_type,
        "x": x,
        "y": y,
        "color": color,
        "title": title or _auto_title(chart_type, x, y, color),
        "stacked": chart_type == "stacked_bar"
        or (chart_type in ("bar", "area") and bool(spec.get("stacked"))),
    }


def _plan_categorical(chart_type: str):
    def plan(table, shape, x, y, color):
        x = x or _first(
            shape.dims or (shape.measures if chart_type in ("line", "area") else []),
            "This chart needs a column for the horizontal axis.",
        )
        y = y or _first(
            [m for m in shape.measures if m != x], "This chart needs a numeric column for the values."
        )
        _require_numeric(table, y, "value")
        if x == y:
            raise ValidationError("The horizontal axis and the values must be different columns.")
        if chart_type in ("grouped_bar", "stacked_bar"):
            color = color or _first(
                [d for d in shape.dims if d != x], "Grouped and stacked bars need a second category column."
            )
        if color is not None and color in (x, y):
            raise ValidationError("The series column must differ from the axis columns.")
        if chart_type in ("pie", "donut"):
            _check_pie(table, y)
            color = None
        return x, y, color

    return plan


def _check_pie(table: pd.DataFrame, y: str) -> None:
    values = table[y].dropna()
    if (values < 0).any():
        raise ValidationError("Pie and donut charts cannot show negative values. Use a bar chart instead.")
    if len(values) > MAX_PIE_SLICES:
        raise ValidationError(
            f"Pie and donut charts support at most {MAX_PIE_SLICES} slices. Use a bar chart instead."
        )
    if len(values) and values.sum() <= 0:
        raise ValidationError("Pie and donut charts need at least one value above zero.")


def _plan_histogram(table, shape, x, y, color):
    x = x or y or _first(shape.measures, "A histogram needs a numeric column.")
    _require_numeric(table, x, "value")
    return x, None, None


def _plan_scatter(table, shape, x, y, color):
    pool = shape.measures if len(shape.measures) >= 2 else shape.numerics
    x = x or _first([c for c in pool if c != y], "A scatter plot needs two numeric columns.")
    y = y or _first([c for c in pool if c != x], "A scatter plot needs two numeric columns.")
    _require_numeric(table, x, "horizontal")
    _require_numeric(table, y, "vertical")
    if x == y:
        raise ValidationError("A scatter plot needs two different columns.")
    if color in (x, y):
        raise ValidationError("The series column must differ from the axis columns.")
    return x, y, color


def _plan_box(table, shape, x, y, color):
    y = y or _first([m for m in shape.measures if m != x], "A box plot needs a numeric column.")
    _require_numeric(table, y, "value")
    if y in (x, color):
        raise ValidationError("The grouping columns must differ from the value column.")
    return x, y, color


def _plan_heatmap(table, shape, x, y, color):
    if x is None and y is None and color is None and _square_matrix(table, shape) is not None:
        return None, None, None
    message = (
        "A heatmap needs two category columns and one numeric column, "
        "or a square matrix such as a correlation table."
    )
    x = x or _first(shape.dims, message)
    color = color or _first([d for d in shape.dims if d != x], message)
    y = y or _first([m for m in shape.measures if m not in (x, color)], message)
    _require_numeric(table, y, "value")
    if len({x, y, color}) < 3:
        raise ValidationError("A heatmap needs three different columns.")
    pairs = table[[x, color]].dropna()
    if pairs.duplicated().any():
        raise ValidationError(
            "A heatmap needs one value for each pair of categories. Aggregate the data first."
        )
    return x, y, color


def _plan_correlation(table, shape, x, y, color):
    if _square_matrix(table, shape) is None and len(shape.measures) < 2:
        raise ValidationError("A correlation chart needs at least two numeric columns.")
    return None, None, None


_PLANNERS = {
    **{
        t: _plan_categorical(t) for t in ("bar", "grouped_bar", "stacked_bar", "line", "area", "pie", "donut")
    },
    "histogram": _plan_histogram,
    "scatter": _plan_scatter,
    "box": _plan_box,
    "heatmap": _plan_heatmap,
    "correlation": _plan_correlation,
}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _text(value: object) -> str:
    """Escape untrusted text for Plotly, which renders a subset of HTML in labels."""
    return html.escape(str(value), quote=False)


def _label(name: object) -> str:
    return _text(humanize(name))


def _template_text(value: object) -> str:
    """Escape text placed inside a hover template (also neutralises ``%{...}`` references)."""
    return _text(value).replace("%{", "% {")


def _plot_values(series: pd.Series) -> Any:
    """Values in a form Plotly serialises exactly: ISO date strings, int64/float64 arrays, escaped text."""
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        midnight = bool((series.dt.normalize() == series).all())
        return series.dt.strftime("%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M:%S").tolist()
    if _is_plain_numeric(series.dtype):
        integer = pd.api.types.is_integer_dtype(series.dtype)
        return series.to_numpy(dtype="int64" if integer else "float64")
    return [_text(v) for v in series]


def _axis_type(series: pd.Series, *, linear_ok: bool) -> str:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "date"
    return "linear" if linear_ok and _is_plain_numeric(series.dtype) else "category"


def _clean(
    table: pd.DataFrame, columns: list[str | None], *, required: str | None = None
) -> tuple[pd.DataFrame, int] | None:
    """Drop rows with missing values in the used columns. None when nothing is left to draw."""
    used = [c for c in dict.fromkeys(columns) if c is not None]
    if required is not None and table[required].notna().sum() == 0:
        return None
    frame = table[used].dropna()
    return (frame, len(table) - len(frame)) if len(frame) else None


def _groups(frame: pd.DataFrame, color: str | None):
    """Yield (series name, rows) for each value of the colour column, or one unnamed group."""
    if color is None:
        yield None, frame
        return
    for key in frame[color].drop_duplicates():
        yield _text(key), frame[frame[color] == key]


def _note(dropped: int) -> str:
    return f"{dropped} row(s) with missing values were left out." if dropped else ""


def _finish(
    fig: go.Figure,
    spec: dict,
    *,
    dropped: int = 0,
    x_title: str | None = None,
    y_title: str | None = None,
    legend_title: str | None = None,
    show_legend: bool = False,
    zero_based: bool = False,
) -> go.Figure:
    title: dict[str, Any] = {"text": _text(spec["title"]), "x": 0.02, "xanchor": "left"}
    if dropped:
        title["subtitle"] = {"text": _note(dropped)}
    fig.update_layout(
        title=title,
        template="plotly_white",
        colorway=_COLORWAY,
        showlegend=show_legend,
        legend={"title": {"text": legend_title}} if legend_title else {},
        margin={"l": 60, "r": 30, "t": 90 if dropped else 70, "b": 60},
        height=480,
        hovermode="closest",
    )
    if x_title is not None:
        fig.update_xaxes(title_text=x_title, automargin=True)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title, automargin=True)
    if zero_based:
        fig.update_yaxes(rangemode="tozero")
    return fig


def _bar_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    x, y, color = spec["x"], spec["y"], spec["color"]
    cleaned = _clean(table, [x, y, color], required=y)
    if cleaned is None:
        return None
    frame, dropped = cleaned
    hover = f"{_template_text(humanize(x))}: %{{x}}<br>{_template_text(humanize(y))}: %{{y:,}}"
    fig = go.Figure()
    for name, part in _groups(frame, color):
        series_hover = hover + (
            f"<br>{_template_text(humanize(color))}: {_template_text(name)}" if name else ""
        )
        fig.add_trace(
            go.Bar(
                x=_plot_values(part[x]),
                y=_plot_values(part[y]),
                name=name or _text(humanize(y)),
                hovertemplate=series_hover + "<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack" if spec["stacked"] else "group")
    fig.update_xaxes(type=_axis_type(frame[x], linear_ok=False))
    return _finish(
        fig,
        spec,
        dropped=dropped,
        x_title=_label(x),
        y_title=_label(y),
        legend_title=_label(color) if color else None,
        show_legend=color is not None,
        zero_based=True,
    )


def _line_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    x, y, color = spec["x"], spec["y"], spec["color"]
    area = spec["type"] == "area"
    cleaned = _clean(table, [x, y, color], required=y)
    if cleaned is None:
        return None
    frame, dropped = cleaned
    frame = frame.sort_values(x, kind="stable")
    hover = f"{_template_text(humanize(x))}: %{{x}}<br>{_template_text(humanize(y))}: %{{y:,}}"
    fig = go.Figure()
    for name, part in _groups(frame, color):
        options: dict[str, Any] = {
            "mode": "lines" if area or len(part) > _MAX_MARKER_POINTS else "lines+markers"
        }
        if area:
            options.update({"stackgroup": "one"} if spec["stacked"] and color else {"fill": "tozeroy"})
        fig.add_trace(
            go.Scatter(
                x=_plot_values(part[x]),
                y=_plot_values(part[y]),
                name=name or _text(humanize(y)),
                hovertemplate=hover + "<extra></extra>",
                **options,
            )
        )
    fig.update_xaxes(type=_axis_type(frame[x], linear_ok=True))
    return _finish(
        fig,
        spec,
        dropped=dropped,
        x_title=_label(x),
        y_title=_label(y),
        legend_title=_label(color) if color else None,
        show_legend=color is not None,
        zero_based=area,
    )


def _pie_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    x, y = spec["x"], spec["y"]
    cleaned = _clean(table, [x, y], required=y)
    if cleaned is None:
        return None
    frame, dropped = cleaned
    trace = go.Pie(
        labels=[_text(v) for v in frame[x]],
        values=_plot_values(frame[y]),
        hole=0.45 if spec["type"] == "donut" else 0,
        sort=False,
        textinfo="label+percent",
        hovertemplate=(
            f"%{{label}}<br>{_template_text(humanize(y))}: %{{value:,}}<br>%{{percent}}<extra></extra>"
        ),
    )
    return _finish(go.Figure(trace), spec, dropped=dropped, show_legend=True, legend_title=_label(x))


def _histogram_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    x = spec["x"]
    cleaned = _clean(table, [x], required=x)
    if cleaned is None:
        return None
    frame, dropped = cleaned
    trace = go.Histogram(
        x=_plot_values(frame[x]),
        name=_text(humanize(x)),
        hovertemplate=f"{_template_text(humanize(x))}: %{{x}}<br>Count: %{{y}}<extra></extra>",
    )
    fig = go.Figure(trace)
    fig.update_layout(bargap=0.05)
    return _finish(fig, spec, dropped=dropped, x_title=_label(x), y_title="Count", zero_based=True)


def _scatter_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    x, y, color = spec["x"], spec["y"], spec["color"]
    cleaned = _clean(table, [x, y, color])
    if cleaned is None:
        return None
    frame, dropped = cleaned
    hover = f"{_template_text(humanize(x))}: %{{x:,}}<br>{_template_text(humanize(y))}: %{{y:,}}"
    fig = go.Figure()
    continuous = color is not None and _is_plain_numeric(frame[color].dtype)
    if continuous:
        marker = {
            "color": _plot_values(frame[color]),
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": {"text": _label(color)}},
        }
        fig.add_trace(
            go.Scatter(
                x=_plot_values(frame[x]),
                y=_plot_values(frame[y]),
                mode="markers",
                marker=marker,
                hovertemplate=hover + "<extra></extra>",
            )
        )
    else:
        for name, part in _groups(frame, color):
            fig.add_trace(
                go.Scatter(
                    x=_plot_values(part[x]),
                    y=_plot_values(part[y]),
                    mode="markers",
                    name=name or _text(humanize(y)),
                    hovertemplate=hover + "<extra></extra>",
                )
            )
    return _finish(
        fig,
        spec,
        dropped=dropped,
        x_title=_label(x),
        y_title=_label(y),
        legend_title=_label(color) if color and not continuous else None,
        show_legend=color is not None and not continuous,
    )


def _box_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    x, y, color = spec["x"], spec["y"], spec["color"]
    cleaned = _clean(table, [x, y, color], required=y)
    if cleaned is None:
        return None
    frame, dropped = cleaned
    fig = go.Figure()
    for name, part in _groups(frame, color):
        fig.add_trace(
            go.Box(
                y=_plot_values(part[y]),
                x=_plot_values(part[x]) if x else None,
                name=name or _text(humanize(y)),
                boxpoints="outliers",
                hovertemplate=f"{_template_text(humanize(y))}: %{{y:,}}<extra></extra>",
            )
        )
    if color:
        fig.update_layout(boxmode="group")
    return _finish(
        fig,
        spec,
        dropped=dropped,
        x_title=_label(x) if x else None,
        y_title=_label(y),
        legend_title=_label(color) if color else None,
        show_legend=color is not None,
    )


def _heatmap_trace(
    matrix: pd.DataFrame, *, value_name: str, correlation: bool, x_name: str, y_name: str
) -> go.Heatmap:
    values = matrix.to_numpy(dtype="float64")
    options: dict[str, Any] = {}
    if correlation:
        options.update({"zmin": -1, "zmax": 1, "zmid": 0, "colorscale": "RdBu"})
    elif np.nanmin(values) < 0 < np.nanmax(values):
        options.update({"zmid": 0, "colorscale": "RdBu"})
    else:
        options["colorscale"] = "Blues"
    if values.size <= _MAX_ANNOTATED_CELLS:
        options["texttemplate"] = "%{z:.2f}" if correlation else "%{z:,.4g}"
    return go.Heatmap(
        z=values,
        x=[_text(c) for c in matrix.columns],
        y=[_text(i) for i in matrix.index],
        hoverongaps=False,
        colorbar={"title": {"text": value_name}},
        hovertemplate=f"{x_name}: %{{x}}<br>{y_name}: %{{y}}<br>{value_name}: %{{z:,}}<extra></extra>",
        **options,
    )


def _matrix_figure(matrix: pd.DataFrame, spec: dict, *, correlation: bool) -> go.Figure | None:
    if not matrix.notna().to_numpy().any():
        return None
    fig = go.Figure(
        _heatmap_trace(
            matrix,
            value_name="Correlation" if correlation else "Value",
            correlation=correlation,
            x_name="Column",
            y_name="Row",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, spec)


def _heatmap_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    x, y, row = spec["x"], spec["y"], spec["color"]
    if x is None:
        return _matrix_figure(_square_matrix(table, _shape(table)), spec, correlation=False)
    cleaned = _clean(table, [x, row, y], required=y)
    if cleaned is None:
        return None
    frame, dropped = cleaned
    matrix = frame.pivot(index=row, columns=x, values=y).astype("float64")
    matrix = matrix.reindex(index=frame[row].drop_duplicates(), columns=frame[x].drop_duplicates())
    matrix.index, matrix.columns = [str(i) for i in matrix.index], [str(c) for c in matrix.columns]
    trace = _heatmap_trace(
        matrix, value_name=_label(y), correlation=False, x_name=_label(x), y_name=_label(row)
    )
    fig = go.Figure(trace)
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, spec, dropped=dropped, x_title=_label(x), y_title=_label(row))


def _correlation_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    shape = _shape(table)
    matrix = _square_matrix(table, shape)
    if matrix is None:
        matrix = table[shape.measures].astype("float64").corr()
    return _matrix_figure(matrix, spec, correlation=True)


_BUILDERS = {
    "bar": _bar_figure,
    "grouped_bar": _bar_figure,
    "stacked_bar": _bar_figure,
    "line": _line_figure,
    "area": _line_figure,
    "pie": _pie_figure,
    "donut": _pie_figure,
    "histogram": _histogram_figure,
    "scatter": _scatter_figure,
    "box": _box_figure,
    "heatmap": _heatmap_figure,
    "correlation": _correlation_figure,
}


def build_figure(table: pd.DataFrame, spec: dict) -> go.Figure | None:
    """Build a Plotly figure for a result table, or None when there is nothing to draw.

    None is returned for an empty table and when the value column holds no
    data at all; the caller shows a message. An invalid spec raises
    ``ValidationError``. Rows with missing values are left out and the
    subtitle says how many.
    """
    if not _has_rows(table):
        return None
    checked = validate_chart_spec(table, spec)
    return _BUILDERS[checked["type"]](table, checked)


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

_PAGE_STYLE = """
:root { color-scheme: light; }
body { font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: #1f2933; background: #ffffff;
       margin: 0 auto; padding: 24px 16px 48px; max-width: 960px; line-height: 1.5; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.1rem; margin: 28px 0 8px; border-bottom: 1px solid #d9dee3; padding-bottom: 4px; }
.meta { color: #52606d; font-size: 0.9rem; margin: 0 0 16px; }
.note { color: #52606d; font-style: italic; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid #d9dee3; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f0f3f6; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
@media print { body { max-width: none; } }
"""


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{_PAGE_STYLE}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
    )


def _figure_div(fig: go.Figure) -> str:
    """The chart as an embeddable block with Plotly.js included, so the page works offline."""
    return fig.to_html(
        full_html=False, include_plotlyjs=True, config={"displaylogo": False, "responsive": True}
    )


def figure_to_html(fig: go.Figure, *, title: str) -> str:
    """A standalone HTML page for one figure. Plotly.js is embedded, so it works offline."""
    heading = f"<h1>{html.escape(title)}</h1>"
    return _page(title, f"{heading}\n{_figure_div(fig)}")


def _cell(value: object) -> str:
    safe = json_safe(value)
    return "" if safe is None else str(safe)


def _table_html(table: pd.DataFrame) -> str:
    numeric = [_is_plain_numeric(dtype) for dtype in table.dtypes]
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in table.columns)
    rows = []
    for values in table.itertuples(index=False, name=None):
        cells = "".join(
            f'<td class="num">{html.escape(_cell(v))}</td>' if is_num else f"<td>{html.escape(_cell(v))}</td>"
            for v, is_num in zip(values, numeric, strict=True)
        )
        rows.append(f"<tr>{cells}</tr>")
    body = "".join(rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _paragraphs(text: str) -> str:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    return "".join(f"<p>{html.escape(b.strip()).replace(chr(10), '<br>')}</p>" for b in blocks)


def _facts_html(facts: list) -> str:
    rows = []
    for fact in facts:
        label, value = (
            (fact.get("label", ""), fact.get("value", "")) if isinstance(fact, dict) else ("", fact)
        )
        rows.append(f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>")
    return f"<table><tbody>{''.join(rows)}</tbody></table>"


_NOTE_NO_ANSWER = '<p class="note">No written answer.</p>'
_NOTE_EMPTY_TABLE = '<p class="note">The result table is empty.</p>'
_NOTE_NO_CHART = (
    '<h2>Chart</h2><p class="note">The chart could not be drawn because the result has no values to show.</p>'
)


def _report_chart(table: pd.DataFrame, chart_spec: dict | None) -> str:
    if not chart_spec:
        return ""
    try:
        fig = build_figure(table, chart_spec)
    except ValidationError as exc:
        return f'<h2>Chart</h2><p class="note">{html.escape(exc.user_message)}</p>'
    if fig is None:
        return _NOTE_NO_CHART
    return f"<h2>Chart</h2>\n{_figure_div(fig)}"


def report_html(
    *,
    title: str,
    question: str,
    answer_text: str,
    table: pd.DataFrame,
    chart_spec: dict | None,
    facts: list,
    generated_at: datetime,
    dataset_name: str,
) -> str:
    """A self-contained HTML report: question, answer, key facts, chart and result table.

    All text is escaped (datasets and questions are untrusted) and the chart,
    when there is one, is embedded once together with Plotly.js.
    """
    stamp = generated_at.strftime("%Y-%m-%d %H:%M")
    meta = f'<p class="meta">Dataset: {html.escape(dataset_name)} | Generated {stamp} UTC</p>'
    answer = _paragraphs(answer_text) or _NOTE_NO_ANSWER
    result = _table_html(table) if _has_rows(table) else _NOTE_EMPTY_TABLE
    parts = [
        f"<h1>{html.escape(title)}</h1>",
        meta,
        f"<h2>Question</h2>{_paragraphs(question)}",
        f"<h2>Answer</h2>{answer}",
        f"<h2>Key facts</h2>{_facts_html(facts)}" if facts else "",
        _report_chart(table, chart_spec),
        f"<h2>Result table</h2>{result}",
    ]
    return _page(title, "\n".join(p for p in parts if p))
