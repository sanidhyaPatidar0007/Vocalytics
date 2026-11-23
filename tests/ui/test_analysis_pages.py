"""AppTest coverage of the analysis pages (home, ask, history, reports)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sqlalchemy import func, select

from config.settings import set_settings
from core.constants import Feature
from database import session_scope
from database.models import AuditLog, SubscriptionPlan
from modules import analyst, super_admin
from services import analytics_engine, llm_service
from tests.ui.conftest import texts

HOME = "views/analysis/home.py"
ASK = "views/analysis/ask.py"
HISTORY = "views/analysis/history.py"
REPORTS = "views/analysis/reports.py"


@pytest.fixture
def tenant(factory, org_id, sales_frame):
    admin = factory.org_admin(org_id)
    analyst_id = factory.user(org_id, "analyst", username="ana")
    dataset_id = factory.dataset(admin, sales_frame, "Sales")
    factory.grant(dataset_id, analyst_id)
    factory.user(org_id, "org_admin", username="dana")
    return {"org_id": org_id, "admin": admin, "analyst": factory.ctx(analyst_id), "dataset_id": dataset_id}


def _metrics(app) -> dict[str, str]:
    return {m.label: m.value for m in app.metric}


def _join(elements) -> str:
    """Visible text of elements without the Markdown escapes the pages add on purpose."""
    return " ".join(str(e.value) for e in elements).replace("\\", "")


def _captions(app) -> str:
    return _join(app.caption)


def _expanders(app) -> list[str]:
    return [e.label for e in app.expander]


def _chart_types(app) -> list[str]:
    """Plotly trace types of every chart currently on the page."""
    found = []
    for element in app.get("plotly_chart"):
        figure = json.loads(element.proto.spec)
        found.extend(trace["type"] for trace in figure["data"])
    return found


def _qb(tenant, name: str) -> str:
    return f"ask_qb_{tenant['dataset_id']}_{name}"


def _run_group_by(app, tenant, operation: str = "group_aggregate") -> None:
    app.multiselect(key=_qb(tenant, "metrics")).select("revenue").run()
    app.multiselect(key=_qb(tenant, "group_by")).select("region").run()
    app.selectbox(key=_qb(tenant, "operation")).select(operation).run()
    app.button(key=_qb(tenant, "run")).click().run()


def _history(tenant, ctx=None) -> list[dict]:
    return analyst.list_my_history(ctx or tenant["analyst"])


def _count_audit(action: str) -> int:
    with session_scope() as db:
        return db.execute(select(func.count(AuditLog.id)).where(AuditLog.action == action)).scalar_one()


def _click(app, label: str) -> None:
    [button] = [b for b in app.button if b.label == label]
    button.click().run()


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------


def test_home_renders_for_analyst(make_app, tenant):
    app = make_app("ana", page=HOME)
    assert not app.exception
    assert app.title[0].value == "Home"
    assert "Sales" in str(app.dataframe[0].value)
    assert "not asked anything" in texts(app) + _join(app.info)
    assert _metrics(app)["Datasets available to you"] == "1"


def test_home_lists_recent_questions(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    app = make_app("ana", page=HOME)
    assert not app.exception
    assert list(app.dataframe[1].value["Question"]) == ["What is the total revenue?"]


def test_home_without_datasets_guides_the_user(make_app, factory, org_id):
    factory.user(org_id, "analyst", username="lonely")
    app = make_app("lonely", page=HOME)
    assert not app.exception
    assert "No datasets yet" in _join(app.info)


# ---------------------------------------------------------------------------
# Ask the analyst
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("username", ["ana", "dana"])
def test_ask_renders_for_each_role(make_app, tenant, username):
    app = make_app(username, page=ASK)
    assert not app.exception
    assert app.title[0].value == "Ask the analyst"
    assert "Local AI is unavailable" in _join(app.info)
    assert "built-in rule-based interpreter" in _join(app.info)
    assert {"Columns in this dataset", "Query builder"} <= set(_expanders(app))
    assert "No questions yet" in _join(app.info)


def test_ask_without_datasets_shows_guidance(make_app, factory, org_id):
    factory.user(org_id, "analyst", username="lonely")
    app = make_app("lonely", page=ASK)
    assert not app.exception
    assert "No datasets yet" in _join(app.info)
    assert not app.chat_input


def test_chat_question_shows_exact_total_with_rule_based_label(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    assert not app.exception
    assert _metrics(app)["Total revenue"] == "5,840.75"
    assert "Planner: Rule-based interpreter" in _captions(app)
    assert "verified independently" in _captions(app)
    rows = _history(tenant)
    assert [(r["question"], r["status"], r["planner"], r["input_mode"]) for r in rows] == [
        ("What is the total revenue?", "success", "rules", "text")
    ]


def test_example_question_button_runs_the_same_pipeline(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.button(key="ask_example_0").click().run()
    assert not app.exception
    assert _metrics(app)["Total revenue"] == "5,840.75"
    assert _history(tenant)[0]["question"] == "What is the total revenue?"


def test_unrecognised_question_asks_for_clarification(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("xyzzy plugh").run()
    assert not app.exception
    assert "needs more detail" in _join(app.markdown)
    assert "did not recognise this question" in _join(app.info)
    assert _history(tenant)[0]["status"] == "clarification"
    assert not app.get("plotly_chart")


def test_markdown_in_a_question_is_not_rendered(make_app, tenant):
    app = make_app("ana", page=ASK)
    question = "![x](http://example.com/p.png) total revenue"
    app.chat_input[0].set_value(question).run()
    assert not app.exception
    assert question in _join(app.text)
    assert not any(question in str(m.value) for m in app.markdown)


def test_query_builder_group_by_shows_table_and_chart(make_app, tenant):
    app = make_app("ana", page=ASK)
    _run_group_by(app, tenant)
    assert not app.exception
    table = app.dataframe[-1].value
    assert dict(zip(table["region"], table["sum_revenue"], strict=True)) == {
        "North": 3800.0,
        "South": 1740.5,
        "East": 300.25,
    }
    assert _chart_types(app) == ["bar"]
    assert "Planner: Query builder" in _captions(app)
    assert _history(tenant)[0]["question"] == "Query builder: group_aggregate"


def test_query_builder_filter(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.multiselect(key=_qb(tenant, "metrics")).select("revenue").run()
    app.multiselect(key=_qb(tenant, "group_by")).select("region").run()
    app.selectbox(key=_qb(tenant, "operation")).select("group_aggregate").run()
    app.number_input(key=_qb(tenant, "filter_count")).set_value(1).run()
    app.selectbox(key=_qb(tenant, "filter_col_0")).select("revenue").run()
    app.selectbox(key=_qb(tenant, "filter_op_0")).select("gt").run()
    app.text_input(key=_qb(tenant, "filter_value_0")).input("1000").run()
    app.button(key=_qb(tenant, "run")).click().run()
    assert not app.exception
    table = app.dataframe[-1].value
    assert dict(zip(table["region"], table["sum_revenue"], strict=True)) == {"North": 3800.0}


def test_query_builder_time_series(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.multiselect(key=_qb(tenant, "metrics")).select("revenue").run()
    app.selectbox(key=_qb(tenant, "operation")).select("time_series").run()
    app.selectbox(key=_qb(tenant, "time_column")).select("order_date").run()
    app.button(key=_qb(tenant, "run")).click().run()
    assert not app.exception
    table = app.dataframe[-1].value
    assert list(table["sum_revenue"]) == [2180.5, 1800.25, 1860.0]
    assert _chart_types(app) == ["scatter"]


def test_query_builder_invalid_plan_is_rejected_with_a_reason(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.selectbox(key=_qb(tenant, "operation")).select("top_n").run()
    app.button(key=_qb(tenant, "run")).click().run()
    assert not app.exception
    assert "needs a group_by column" in _join(app.error)
    assert _history(tenant)[0]["status"] == "rejected"


def test_chart_type_override_changes_the_chart(make_app, tenant):
    app = make_app("ana", page=ASK)
    _run_group_by(app, tenant)
    assert _chart_types(app) == ["bar"]
    app.selectbox(key="ask_1_type").select("pie").run()
    assert not app.exception
    assert _chart_types(app) == ["pie"]
    app.selectbox(key="ask_1_type").select("none").run()
    assert _chart_types(app) == []
    assert "nothing to chart" in _join(app.info)


def test_basic_plan_hides_advanced_operations_and_denies_growth(make_app, factory):
    org = factory.organization("Basic Co", plan="basic")
    admin = factory.org_admin(org)
    user_id = factory.user(org, "analyst", username="basic_ana")
    frame = pd.DataFrame({"revenue": [1.0, 2.0], "order_date": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    dataset_id = factory.dataset(admin, frame, "Tiny")
    factory.grant(dataset_id, user_id)
    app = make_app("basic_ana", page=ASK)
    assert not app.exception
    operations = app.selectbox(key=f"ask_qb_{dataset_id}_operation").options
    assert "Correlation" not in operations and "Growth" not in operations and "Aggregate" in operations
    assert "Ask by voice" not in _expanders(app)
    app.chat_input[0].set_value("What is the month-over-month revenue growth?").run()
    assert not app.exception
    assert len(app.error) == 1
    assert not _metrics(app)


def test_voice_panel_hidden_without_use_voice_permission(make_app, factory, org_id, tenant):
    limited = factory.user(org_id, "analyst", username="quiet", permissions={"export_data"})
    factory.grant(tenant["dataset_id"], limited)
    app = make_app("quiet", page=ASK)
    assert not app.exception
    assert "Ask by voice" not in _expanders(app)
    app = make_app("ana", page=ASK)
    assert "Ask by voice" in _expanders(app)


def test_voice_panel_shows_engine_status_and_requires_a_recording(make_app, tenant):
    app = make_app("ana", page=ASK)
    assert "To enable offline recognition" in _join(app.info)
    assert not any("Google" in str(w.value) for w in app.warning)
    app.button(key="ask_voice_transcribe").click().run()
    assert not app.exception
    assert "Record your question first" in _join(app.warning)


def test_voice_online_warning_only_when_enabled(make_app, tenant, app_settings):
    set_settings(app_settings.with_overrides(voice_online_recognition_enabled=True))
    app = make_app("ana", page=ASK)
    assert any("Google" in str(w.value) for w in app.warning)


def test_voice_edited_text_is_what_gets_asked(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.text_area(key="ask_voice_text").input("What is the total revenew?").run()
    app.text_area(key="ask_voice_text").input("What is the total revenue?").run()
    app.button(key="ask_voice_submit").click().run()
    assert not app.exception
    assert _metrics(app)["Total revenue"] == "5,840.75"
    row = _history(tenant)[0]
    assert (row["question"], row["input_mode"]) == ("What is the total revenue?", "voice")
    assert "Asked by voice" in _captions(app)


def test_voice_submit_without_text_asks_nothing(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.button(key="ask_voice_submit").click().run()
    assert not app.exception
    assert "no text to ask" in _join(app.warning)
    assert _history(tenant) == []


def test_save_answer_as_report(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    app.text_input(key="ask_1_save_title").input("Revenue total")
    _click(app, "Save as report")
    assert not app.exception
    assert "Report saved" in _join(app.success)
    [report] = analyst.list_reports(tenant["analyst"])
    assert (report["title"], report["question"]) == ("Revenue total", "What is the total revenue?")


def test_save_as_report_with_blank_title_shows_error(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    app.text_input(key="ask_1_save_title").input("   ")
    _click(app, "Save as report")
    assert "Enter a title" in _join(app.error)
    assert analyst.list_reports(tenant["analyst"]) == []


def test_save_hidden_without_permission(make_app, factory, org_id, tenant):
    reader = factory.user(org_id, "analyst", username="reader", permissions={"export_data"})
    factory.grant(tenant["dataset_id"], reader)
    app = make_app("reader", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    assert not [b for b in app.button if b.label == "Save as report"]
    assert "permission to save reports" in _join(app.info)


def test_export_is_prepared_only_on_request_then_downloadable(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    assert not app.get("download_button")
    assert _count_audit("analysis.export") == 0
    app.button(key="ask_1_prepare_csv").click().run()
    assert not app.exception
    [download] = app.get("download_button")
    assert "analysis" in download.proto.label
    assert _count_audit("analysis.export") == 1
    app.button(key="ask_1_prepare_xlsx").click().run()
    assert len(app.get("download_button")) == 2
    app.run()
    assert len(app.get("download_button")) == 2
    assert _count_audit("analysis.export") == 2


def test_export_hidden_without_export_permission(make_app, factory, org_id, tenant):
    quiet = factory.user(org_id, "analyst", username="noexport", permissions={"save_reports"})
    factory.grant(tenant["dataset_id"], quiet)
    app = make_app("noexport", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    assert not [b for b in app.button if b.label.startswith("Prepare")]
    assert "permission to export data" in _join(app.info)


def _fake_llm(monkeypatch, narrative: str | None = None) -> None:
    plan = {"operation": "aggregate", "metrics": [{"column": "revenue", "agg": "sum"}]}
    monkeypatch.setattr(llm_service, "generate_plan", lambda *a, **k: (plan, "test-model:1b"))
    if narrative is None:
        return
    monkeypatch.setattr(llm_service, "generate_explanation", lambda *a, **k: narrative)


def test_llm_planned_answer_shows_the_model_name_and_a_labelled_narrative(make_app, tenant, monkeypatch):
    _fake_llm(monkeypatch, "Revenue adds up to 5,840.75 in total.")
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("How much did we sell?").run()
    assert not app.exception
    assert "Planner: Local AI model test-model:1b" in _captions(app)
    assert "AI-written explanation" in _join(app.markdown)
    assert "Revenue adds up to 5,840.75 in total." in _join(app.markdown)
    assert _metrics(app)["Total revenue"] == "5,840.75"
    assert _history(tenant)[0]["planner"] == "llm"


def test_ungrounded_narrative_is_not_shown(make_app, tenant, monkeypatch):
    _fake_llm(monkeypatch, "Revenue adds up to 9,999 in total.")
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("How much did we sell?").run()
    assert not app.exception
    assert "AI-written explanation" not in _join(app.markdown)
    assert "narrative was discarded" in _join(app.warning)


def test_narrative_text_is_escaped(make_app, tenant, monkeypatch):
    payload = "Total is 5,840.75 ![x](http://example.com/p.png)"
    _fake_llm(monkeypatch, payload)
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("How much did we sell?").run()
    assert not app.exception
    assert "AI-written explanation" in _join(app.markdown)
    assert any("example" in str(m.value) and r"\!" in str(m.value) for m in app.markdown)
    assert not any("![x](http" in str(m.value) for m in app.markdown)


def test_ai_status_shows_the_running_model(make_app, tenant, monkeypatch):
    status = {
        "available": True,
        "models": ["llama3:8b"],
        "configured_model": "llama3:8b",
        "model_present": True,
        "fallback_model": None,
        "fallback_present": False,
        "error": None,
    }
    monkeypatch.setattr(llm_service, "check_status", lambda: status)
    app = make_app("ana", page=ASK)
    assert "Local AI is running with model llama3:8b" in _join(app.success)


def test_ai_status_with_rules_fallback_off_explains_the_consequence(make_app, tenant, app_settings):
    set_settings(app_settings.with_overrides(ai_fallback_mode="off"))
    app = make_app("ana", page=ASK)
    assert "Questions cannot be answered until the AI is running" in _join(app.info)
    app.chat_input[0].set_value("What is the total revenue?").run()
    assert not app.exception
    assert len(app.error) == 1
    assert not _metrics(app)


def test_truncated_result_shows_a_note(make_app, tenant, app_settings):
    set_settings(app_settings.with_overrides(result_max_rows=2))
    app = make_app("ana", page=ASK)
    _run_group_by(app, tenant)
    assert not app.exception
    assert "Showing the first 2 of 3 result rows" in _captions(app)
    assert len(app.dataframe[-1].value) == 2


def test_correlation_result_draws_a_heatmap(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.selectbox(key=_qb(tenant, "operation")).select("correlation").run()
    app.button(key=_qb(tenant, "run")).click().run()
    assert not app.exception
    assert _chart_types(app) == ["heatmap"]


def test_clear_conversation_keeps_history(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    app.button(key="ask_clear").click().run()
    assert not app.exception
    assert "No questions yet" in _join(app.info)
    assert len(_history(tenant)) == 1


# ---------------------------------------------------------------------------
# Analysis history
# ---------------------------------------------------------------------------


def _answer(ctx, dataset_id: int, question: str = "What is the total revenue?") -> dict:
    return analytics_engine.answer_question(ctx, dataset_id, question)


def _group_plan() -> dict:
    return {"operation": "group_aggregate", "metrics": [{"column": "revenue", "agg": "sum"}], "group_by": ["region"]}


@pytest.mark.parametrize("username", ["ana", "dana"])
def test_history_renders_for_each_role_when_empty(make_app, tenant, username):
    app = make_app(username, page=HISTORY)
    assert not app.exception
    assert app.title[0].value == "Analysis history"
    assert "No analyses yet" in _join(app.info)


def test_history_shows_the_selected_item(make_app, tenant):
    answer = _answer(tenant["analyst"], tenant["dataset_id"])
    app = make_app("ana", page=HISTORY)
    assert not app.exception
    assert list(app.dataframe[0].value["Question"]) == ["What is the total revenue?"]
    assert _metrics(app)["Total revenue"] == "5,840.75"
    assert "Status: success" in _captions(app)
    assert "How this was calculated" in _expanders(app)
    assert app.selectbox(key="history_item_0").value == answer["history_id"]


def test_history_filters_by_dataset_and_selects_items(make_app, factory, tenant):
    other = factory.dataset(tenant["admin"], pd.DataFrame({"revenue": [1.0, 2.0, 3.0]}), "Other")
    factory.grant(other, tenant["analyst"].user_id)
    _answer(tenant["analyst"], tenant["dataset_id"])
    _answer(tenant["analyst"], other, "What is the total revenue?")
    app = make_app("ana", page=HISTORY)
    assert len(app.dataframe[0].value) == 2
    app.selectbox(key="history_dataset").select(other).run()
    assert not app.exception
    assert list(app.dataframe[0].value["Dataset"]) == ["Other"]
    assert _metrics(app)["Total revenue"] == "6"


def test_history_shows_rejected_analysis_as_an_error(make_app, tenant):
    analytics_engine.run_plan(tenant["analyst"], tenant["dataset_id"], {"operation": "top_n"})
    app = make_app("ana", page=HISTORY)
    assert not app.exception
    assert "needs a group_by column" in _join(app.error)
    assert not _metrics(app)


def test_history_is_private_to_the_user(make_app, tenant):
    _answer(tenant["analyst"], tenant["dataset_id"])
    app = make_app("dana", page=HISTORY)
    assert "No analyses yet" in _join(app.info)


def test_history_chart_override(make_app, tenant):
    answer = analytics_engine.run_plan(tenant["analyst"], tenant["dataset_id"], _group_plan())
    app = make_app("ana", page=HISTORY)
    assert _chart_types(app) == ["bar"]
    app.selectbox(key=f"history_{answer['history_id']}_type").select("pie").run()
    assert not app.exception
    assert _chart_types(app) == ["pie"]


def test_history_export_prepare_then_download(make_app, tenant):
    answer = analytics_engine.run_plan(tenant["analyst"], tenant["dataset_id"], _group_plan())
    key = f"history_{answer['history_id']}"
    app = make_app("ana", page=HISTORY)
    assert not app.get("download_button")
    assert _count_audit("analysis.export") == 0
    app.button(key=f"{key}_prepare_csv").click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 1
    assert _count_audit("analysis.export") == 1


def test_history_save_as_report(make_app, tenant):
    answer = _answer(tenant["analyst"], tenant["dataset_id"])
    app = make_app("ana", page=HISTORY)
    app.text_input(key=f"history_{answer['history_id']}_save_title").input("From history")
    _click(app, "Save as report")
    assert not app.exception
    assert "Report saved" in _join(app.success)
    assert [r["title"] for r in analyst.list_reports(tenant["analyst"])] == ["From history"]


# ---------------------------------------------------------------------------
# Saved reports
# ---------------------------------------------------------------------------


def _save(ctx, tenant, title: str = "Revenue total", notes: str = "first note") -> dict:
    answer = _answer(ctx, tenant["dataset_id"])
    return analyst.save_report(ctx, history_id=answer["history_id"], title=title, notes=notes)


def test_reports_empty_state(make_app, tenant):
    app = make_app("ana", page=REPORTS)
    assert not app.exception
    assert app.title[0].value == "Saved reports"
    assert "No saved reports yet" in _join(app.info)


@pytest.mark.parametrize("username", ["ana", "dana"])
def test_reports_render_a_saved_report_for_each_role(make_app, tenant, username):
    _save(tenant["admin"] if username == "dana" else tenant["analyst"], tenant)
    app = make_app(username, page=REPORTS)
    assert not app.exception
    assert list(app.dataframe[0].value["Title"]) == ["Revenue total"]
    assert _metrics(app)["Total revenue"] == "5,840.75"
    assert "first note" in _join(app.text)


def test_asked_and_saved_report_appears_on_the_reports_page(make_app, tenant):
    app = make_app("ana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue?").run()
    app.text_input(key="ask_1_save_title").input("Saved from ask")
    _click(app, "Save as report")
    app.switch_page(REPORTS).run()
    assert not app.exception
    assert list(app.dataframe[0].value["Title"]) == ["Saved from ask"]
    assert _metrics(app)["Total revenue"] == "5,840.75"


def test_report_edit_title_and_notes(make_app, tenant):
    report = _save(tenant["analyst"], tenant)
    key = f"reports_{report['report_id']}"
    app = make_app("ana", page=REPORTS)
    app.text_input(key=f"{key}_title").input("Renamed report")
    app.text_area(key=f"{key}_notes").input("second note")
    _click(app, "Save changes")
    assert not app.exception
    assert "Report updated" in _join(app.success)
    stored = analyst.get_report(tenant["analyst"], report["report_id"])
    assert (stored["title"], stored["notes"]) == ("Renamed report", "second note")
    assert list(app.dataframe[0].value["Title"]) == ["Renamed report"]


def test_report_edit_with_blank_title_shows_error(make_app, tenant):
    report = _save(tenant["analyst"], tenant)
    app = make_app("ana", page=REPORTS)
    app.text_input(key=f"reports_{report['report_id']}_title").input("  ")
    _click(app, "Save changes")
    assert "Enter a title" in _join(app.error)
    assert analyst.get_report(tenant["analyst"], report["report_id"])["title"] == "Revenue total"


def test_report_delete_requires_confirmation(make_app, tenant):
    report = _save(tenant["analyst"], tenant)
    key = f"reports_{report['report_id']}"
    app = make_app("ana", page=REPORTS)
    app.button(key=f"{key}_delete").click().run()
    assert "Tick the confirmation box" in _join(app.error)
    assert len(analyst.list_reports(tenant["analyst"])) == 1
    app.checkbox(key=f"{key}_confirm_delete").check()
    app.button(key=f"{key}_delete").click().run()
    assert not app.exception
    assert "Report deleted" in _join(app.success)
    assert analyst.list_reports(tenant["analyst"]) == []
    assert "No saved reports yet" in _join(app.info)


def test_report_exports_prepare_then_download(make_app, tenant):
    report = _save(tenant["analyst"], tenant)
    key = f"reports_{report['report_id']}"
    app = make_app("ana", page=REPORTS)
    assert not app.get("download_button")
    assert _count_audit("report.export") == 0
    for file_format in ("html", "csv", "xlsx", "json"):
        app.button(key=f"{key}_prepare_{file_format}").click().run()
        assert not app.exception
    assert len(app.get("download_button")) == 4
    assert _count_audit("report.export") == 4
    app.run()
    assert len(app.get("download_button")) == 4
    assert _count_audit("report.export") == 4


def test_report_of_another_user_is_read_only_for_an_analyst(make_app, tenant):
    report = _save(tenant["admin"], tenant, title="Admin report")
    app = make_app("ana", page=REPORTS)
    assert not app.exception
    assert list(app.dataframe[0].value["Title"]) == ["Admin report"]
    assert "Only the person who saved this report" in _join(app.info)
    assert not [b for b in app.button if b.label in {"Save changes", "Delete report"}]
    assert app.button(key=f"reports_{report['report_id']}_prepare_csv")


def test_admin_can_edit_an_analysts_report(make_app, tenant):
    report = _save(tenant["analyst"], tenant)
    app = make_app("dana", page=REPORTS)
    key = f"reports_{report['report_id']}"
    app.text_input(key=f"{key}_title").input("Admin renamed")
    _click(app, "Save changes")
    assert analyst.get_report(tenant["admin"], report["report_id"])["title"] == "Admin renamed"


def test_deleting_one_of_two_reports_shows_the_other(make_app, tenant):
    first = _save(tenant["analyst"], tenant, title="First report")
    second = _save(tenant["analyst"], tenant, title="Second report")
    app = make_app("ana", page=REPORTS)
    assert not app.exception
    app.selectbox(key="reports_item_0").select(first["report_id"]).run()
    assert app.subheader[0].value == "First report"
    app.checkbox(key=f"reports_{first['report_id']}_confirm_delete").check()
    app.button(key=f"reports_{first['report_id']}_delete").click().run()
    assert not app.exception
    assert not app.error, _join(app.error)
    assert list(app.dataframe[0].value["Title"]) == ["Second report"]
    assert app.subheader[0].value == "Second report"
    assert [r["report_id"] for r in analyst.list_reports(tenant["analyst"])] == [second["report_id"]]


def test_prepared_report_downloads_are_dropped_after_an_edit(make_app, tenant):
    report = _save(tenant["analyst"], tenant)
    key = f"reports_{report['report_id']}"
    app = make_app("ana", page=REPORTS)
    app.button(key=f"{key}_prepare_csv").click().run()
    assert len(app.get("download_button")) == 1
    app.text_input(key=f"{key}_title").input("Renamed")
    _click(app, "Save changes")
    assert not app.exception
    assert not app.get("download_button")


# ---------------------------------------------------------------------------
# Untrusted text and every operation
# ---------------------------------------------------------------------------

PAYLOAD = "![x](http://example.com/p.png) **b**"


def test_report_title_with_markdown_is_shown_literally(make_app, tenant):
    _save(tenant["analyst"], tenant, title=PAYLOAD)
    app = make_app("ana", page=REPORTS)
    assert not app.exception
    assert not any(PAYLOAD in str(e.value) for e in [*app.subheader, *app.markdown, *app.caption])
    assert PAYLOAD in _join(app.subheader)
    assert list(app.dataframe[0].value["Title"]) == [PAYLOAD]


def test_text_facts_from_the_data_are_not_rendered_as_markdown(make_app, factory, tenant):
    frame = pd.DataFrame({"product": [PAYLOAD, PAYLOAD, "Plain"], "revenue": [5.0, 7.0, 1.0]})
    dataset_id = factory.dataset(tenant["admin"], frame, "Risky")
    factory.grant(dataset_id, tenant["analyst"].user_id)
    plan = {
        "operation": "top_n",
        "metrics": [{"column": "revenue", "agg": "sum"}],
        "group_by": ["product"],
        "limit": 1,
    }
    answer = analytics_engine.run_plan(tenant["analyst"], dataset_id, plan)
    assert answer["status"] == "success"
    app = make_app("ana", page=HISTORY)
    assert not app.exception
    facts = next(d.value for d in app.dataframe if "Fact" in d.value.columns)
    assert PAYLOAD in " ".join(facts["Value"]) + " ".join(facts["Fact"])
    assert not any(PAYLOAD in str(m.value) for m in app.metric)
    assert not any(PAYLOAD in str(e.value) for e in [*app.markdown, *app.caption, *app.info])


OPERATION_PLANS = {
    "aggregate": {"operation": "aggregate", "metrics": [{"column": "revenue", "agg": "mean"}]},
    "group_aggregate": {
        "operation": "group_aggregate",
        "metrics": [{"column": "revenue", "agg": "sum"}],
        "group_by": ["region", "product"],
    },
    "compare": {
        "operation": "compare",
        "metrics": [{"column": "revenue", "agg": "sum"}, {"column": "quantity", "agg": "sum"}],
        "group_by": ["region"],
    },
    "top_n": {"operation": "top_n", "metrics": [{"column": "revenue", "agg": "sum"}], "group_by": ["product"]},
    "time_series": {
        "operation": "time_series",
        "metrics": [{"column": "revenue", "agg": "sum"}],
        "time_column": "order_date",
        "frequency": "M",
    },
    "growth": {
        "operation": "growth",
        "metrics": [{"column": "revenue", "agg": "sum"}],
        "time_column": "order_date",
        "frequency": "M",
    },
    "anomalies": {
        "operation": "anomalies",
        "metrics": [{"column": "revenue", "agg": "sum"}],
        "time_column": "order_date",
        "frequency": "D",
    },
    "describe": {"operation": "describe"},
    "correlation": {"operation": "correlation"},
    "missing_values": {"operation": "missing_values"},
    "duplicates": {"operation": "duplicates"},
    "distribution": {"operation": "distribution", "columns": ["revenue"]},
    "count_rows": {"operation": "count_rows"},
}


@pytest.mark.parametrize("operation", list(OPERATION_PLANS))
def test_every_operation_renders_from_history_without_errors(make_app, tenant, operation):
    answer = analytics_engine.run_plan(tenant["analyst"], tenant["dataset_id"], OPERATION_PLANS[operation])
    assert answer["status"] == "success", answer["answer_text"]
    app = make_app("ana", page=HISTORY)
    assert not app.exception
    assert not app.error, _join(app.error)
    assert not app.warning or operation == "anomalies" or "chart" not in _join(app.warning).lower()
    if answer["chart"] is not None:
        assert _chart_types(app), f"{operation} returned a chart spec but nothing was drawn"
    assert list(app.dataframe[0].value["Status"]) == ["success"]


@pytest.mark.parametrize("index", range(10))
def test_every_example_question_runs_without_errors(make_app, tenant, index):
    app = make_app("ana", page=ASK)
    app.button(key=f"ask_example_{index}").click().run()
    assert not app.exception
    assert not app.error, _join(app.error)
    assert _history(tenant)[0]["status"] in {"success", "clarification"}


@pytest.mark.parametrize(
    "operation", ["top_n", "describe", "missing_values", "duplicates", "distribution", "anomalies"]
)
def test_ask_page_renders_operations_from_the_builder(make_app, tenant, operation):
    app = make_app("ana", page=ASK)
    if operation in {"top_n"}:
        app.multiselect(key=_qb(tenant, "metrics")).select("revenue").run()
        app.multiselect(key=_qb(tenant, "group_by")).select("product").run()
    if operation == "distribution":
        app.multiselect(key=_qb(tenant, "columns")).select("revenue").run()
    if operation == "anomalies":
        app.multiselect(key=_qb(tenant, "metrics")).select("revenue").run()
        app.selectbox(key=_qb(tenant, "time_column")).select("order_date").run()
        app.selectbox(key=_qb(tenant, "frequency")).select("D").run()
    app.selectbox(key=_qb(tenant, "operation")).select(operation).run()
    app.button(key=_qb(tenant, "run")).click().run()
    assert not app.exception
    assert not app.error, _join(app.error)
    assert _history(tenant)[0]["status"] == "success"


# ---------------------------------------------------------------------------
# Plan feature gates on result charts (UI3/ST3, ST4)
# ---------------------------------------------------------------------------


def _set_plan_feature(factory, plan_code: str, feature: Feature, enabled: bool) -> None:
    with session_scope() as db:
        plan_id = db.execute(select(SubscriptionPlan.id).where(SubscriptionPlan.code == plan_code)).scalar_one()
    super_admin.set_plan_feature(factory.super_admin(), plan_id, feature.value, enabled)


def _charts_not_in_plan(app) -> bool:
    return any("chart" in str(i.value).lower() and "plan" in str(i.value).lower() for i in app.info)


def test_chart_tab_respects_standard_charts_plan_feature(make_app, factory, tenant):
    _set_plan_feature(factory, "premium", Feature.STANDARD_CHARTS, False)
    app = make_app("dana", page=ASK)
    app.chat_input[0].set_value("What is the total revenue by region?").run()
    assert not app.exception
    assert app.dataframe  # the result table is still shown
    assert _chart_types(app) == []
    assert _charts_not_in_plan(app)
    assert not [s for s in app.selectbox if s.key and s.key.endswith("_type")]
    assert "Chart type" not in [s.label for s in app.selectbox]  # the query builder hides it too


def test_stored_charts_are_hidden_on_history_and_reports_without_standard_charts(make_app, factory, tenant):
    answer = analytics_engine.run_plan(tenant["analyst"], tenant["dataset_id"], _group_plan())
    assert answer["chart"] is not None  # stored while the plan still had charts
    analyst.save_report(tenant["analyst"], history_id=answer["history_id"], title="Charted")
    _set_plan_feature(factory, "premium", Feature.STANDARD_CHARTS, False)
    for page in (HISTORY, REPORTS):
        app = make_app("ana", page=page)
        assert not app.exception, page
        assert app.dataframe, page
        assert _chart_types(app) == [], page
        assert not [s for s in app.selectbox if s.key and s.key.endswith("_type")], page
        assert _charts_not_in_plan(app), page


def test_basic_plan_result_chart_override_hides_advanced_charts(make_app, factory, sales_frame):
    """ST4: heatmap and correlation re-renders need the advanced analytics feature."""
    org = factory.organization("Basic Co", plan="basic")
    admin = factory.org_admin(org)
    user_id = factory.user(org, "analyst", username="basic_ana")
    dataset_id = factory.dataset(admin, sales_frame, "Sales")
    factory.grant(dataset_id, user_id)
    app = make_app("basic_ana", page=ASK)

    def key(name: str) -> str:
        return f"ask_qb_{dataset_id}_{name}"

    app.multiselect(key=key("metrics")).select("revenue").select("quantity").run()
    app.multiselect(key=key("group_by")).select("region").select("product").run()
    app.selectbox(key=key("operation")).select("group_aggregate").run()
    app.button(key=key("run")).click().run()
    assert not app.exception
    override = app.selectbox(key="ask_1_type")
    assert "Heatmap" not in override.options and "Correlation" not in override.options
    assert "Grouped bar" in override.options  # standard charts are still offered
    assert "heatmap" not in _chart_types(app)


def test_stored_heatmap_is_not_drawn_without_advanced_analytics(make_app, factory, tenant):
    answer = analytics_engine.run_plan(tenant["analyst"], tenant["dataset_id"], {"operation": "correlation"})
    assert answer["chart"]["type"] in ("heatmap", "correlation")  # stored while the plan had the feature
    _set_plan_feature(factory, "premium", Feature.ADVANCED_ANALYTICS, False)
    app = make_app("ana", page=HISTORY)
    assert not app.exception
    assert "heatmap" not in _chart_types(app)
    override = [s for s in app.selectbox if s.key == f"history_{answer['history_id']}_type"]
    for selectbox in override:
        assert "Heatmap" not in selectbox.options and "Correlation" not in selectbox.options


def test_missing_voice_explains_why(make_app, factory, org_id, tenant):
    """UI7: a missing plan feature or permission shows feature_unavailable, not a silent gap."""
    limited = factory.user(org_id, "analyst", username="quiet", permissions={"export_data"})
    factory.grant(tenant["dataset_id"], limited)
    app = make_app("quiet", page=ASK)
    assert not app.exception
    assert "Ask by voice" not in _expanders(app)
    assert "use voice permission" in _join(app.info)
    _set_plan_feature(factory, "premium", Feature.VOICE_ANALYSIS, False)
    app = make_app("ana", page=ASK)
    assert "Ask by voice" not in _expanders(app)
    assert "Voice questions are not included in your plan" in _join(app.info)
