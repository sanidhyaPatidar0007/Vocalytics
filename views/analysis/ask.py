"""Ask the analyst: text and voice questions, the query builder and the conversation."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from components.chart_components import render_result, result_actions
from components.dataset_picker import dataset_selector
from components.feedback import feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import require_ctx
from config.settings import get_settings
from core.constants import Feature, LogicalType, Permission
from services import analytics_engine, llm_service, voice_service

MAX_CONVERSATION = 15
STATUS_KEY = "ask_ai_status"
STATUS_MAX_AGE_SECONDS = 30
NONE_CHOICE = "(not used)"
EXAMPLES = (
    "What is the total revenue?",
    "Which product has the highest sales?",
    "Show the monthly revenue trend",
    "Which region has the highest profit?",
    "What is the average sales by category?",
    "Show the top 10 products by profit",
    "What is the month-over-month revenue growth?",
    "Are there unusual changes in revenue?",
    "Which columns have missing values?",
    "Are there duplicate customer records?",
)
FREQUENCY_LABELS = {"D": "Day", "W": "Week", "M": "Month", "Q": "Quarter", "Y": "Year"}
NUMERIC_TYPES = {LogicalType.INTEGER.value, LogicalType.FLOAT.value}
NUMERIC_ONLY_AGGS = {"sum", "mean", "median", "std"}
ORDERED_AGGS = {"min", "max"}
VALUE_FREE_OPS = {"is_null", "not_null"}
LIST_OPS = {"in", "not_in", "between"}
PLANNER_LABELS = {"rules": "Rule-based interpreter", "builder": "Query builder"}


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _planner_label(response: dict[str, Any]) -> str:
    planner = response.get("planner")
    if planner == "llm":
        model = str(response.get("model") or "")
        kind = "Gemini model" if model.startswith("gemini") else "Local AI model"
        return f"{kind} {model}".strip()
    return PLANNER_LABELS.get(planner, "No plan was made")


# ---------------------------------------------------------------------------
# AI and schema status
# ---------------------------------------------------------------------------


def _ai_status() -> dict[str, Any]:
    cached = st.session_state.get(STATUS_KEY)
    if cached is None or time.monotonic() - cached[0] > STATUS_MAX_AGE_SECONDS:
        cached = (time.monotonic(), llm_service.check_status())
        st.session_state[STATUS_KEY] = cached
    return cached[1]


def _show_ai_status() -> None:
    left, right = st.columns([5, 1])
    if right.button("Check AI status", key="ask_status_refresh"):
        st.session_state.pop(STATUS_KEY, None)
    status = _ai_status()
    rules = get_settings().ai_fallback_mode == "rules"
    fallback_note = (
        "The built-in rule-based interpreter answers common questions meanwhile "
        "(AI_FALLBACK_MODE is rules)."
        if rules
        else "Questions cannot be answered until the AI is running (AI_FALLBACK_MODE is off)."
    )
    gemini = status.get("provider") == "gemini"
    with left:
        model = status["configured_model"] if status["model_present"] else status["fallback_model"]
        if gemini and status["available"]:
            st.success(
                f"Gemini is set up with model {md_escape(model)}. Only your question and the column names "
                "and types are sent to Google; data values and results stay on this computer.",
                icon=":material/check_circle:",
            )
        elif gemini:
            st.info(f"{md_escape(status['error'] or '')} {fallback_note}", icon=":material/cloud_off:")
        elif status["available"] and (status["model_present"] or status["fallback_present"]):
            st.success(f"Local AI is running with model {md_escape(model)}.", icon=":material/check_circle:")
        elif status["available"]:
            st.warning(f"{md_escape(status['error'])} {fallback_note}", icon=":material/warning:")
        else:
            st.info(
                f"Local AI is unavailable. {md_escape(status['error'] or '')} {fallback_note}",
                icon=":material/cloud_off:",
            )


def _schema_columns(ctx, dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Schema summary of the selected dataset version, kept for this session only."""
    cached = st.session_state.get("ask_schema")
    marker = (dataset["dataset_id"], dataset.get("current_version_id"))
    if cached is not None and cached[0] == marker:
        return cached[1]
    summary = run_action(
        lambda: analytics_engine.get_schema_summary(ctx, dataset["dataset_id"]), component="ask"
    )
    columns = summary["columns"] if summary else []
    if summary:
        st.session_state["ask_schema"] = (marker, columns)
    return columns


def _show_schema(columns: list[dict[str, Any]]) -> None:
    with st.expander("Columns in this dataset"):
        if not columns:
            st.info("The column summary is not available for this dataset.")
            return
        rows = [
            {
                "Column": c["name"],
                "Type": c["logical_type"],
                "Kind": c["semantic"],
                "Missing %": c["missing_pct"],
                "Smallest": "" if c.get("min") is None else str(c["min"]),
                "Largest": "" if c.get("max") is None else str(c["max"]),
                "Example values": ", ".join(c.get("sample_values", [])),
            }
            for c in columns
        ]
        st.dataframe(rows, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------


def _conversation_key(dataset_id: int) -> str:
    return f"ask_conversation_{dataset_id}"


def _remember(dataset_id: int, question: str, mode: str, response: dict[str, Any]) -> None:
    entry_id = st.session_state.get("ask_next_entry", 1)
    st.session_state["ask_next_entry"] = entry_id + 1
    entries = st.session_state.setdefault(_conversation_key(dataset_id), [])
    entries.append({"id": entry_id, "question": question, "mode": mode, "response": response})
    del entries[:-MAX_CONVERSATION]


def _ask_text(ctx, dataset_id: int, question: str, *, mode: str = "text") -> None:
    with st.spinner("The analyst is working on your question..."):
        response = run_action(
            lambda: analytics_engine.answer_question(ctx, dataset_id, question, input_mode=mode),
            component="ask",
        )
    if response is not None:
        _remember(dataset_id, question.strip(), mode, response)


def _ask_plan(ctx, dataset_id: int, plan: dict[str, Any]) -> None:
    with st.spinner("Running the query..."):
        response = run_action(lambda: analytics_engine.run_plan(ctx, dataset_id, plan), component="ask")
    if response is not None:
        _remember(dataset_id, f"Query builder: {_label(plan['operation'])}", "builder", response)


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------


def _voice_panel(ctx, dataset_id: int) -> None:
    with st.expander("Ask by voice"):
        status = voice_service.engine_status()
        vosk = status["vosk"]
        if vosk["available"]:
            st.success(
                f"Offline recognition is ready for {md_escape(vosk['model_language'])}. "
                "Your recording stays on this computer.",
                icon=":material/check_circle:",
            )
        else:
            st.info(md_escape(vosk["setup_hint"]), icon=":material/mic_off:")
        if status["online"]["enabled"]:
            st.warning(md_escape(status["online"]["warning"]), icon=":material/cloud_upload:")
        languages = status["supported_languages"]
        default = status["default_language"]
        language = st.selectbox(
            "Speech language",
            languages,
            index=languages.index(default) if default in languages else 0,
            key="ask_voice_language",
        )
        audio = st.audio_input("Record your question", key="ask_voice_audio")
        st.caption(
            "Press the microphone button to start and press it again to stop. If nothing records, "
            "allow microphone access when the browser asks, and on Windows open Settings, "
            "Privacy and security, Microphone and turn on access for your browser."
        )
        if st.button("Transcribe", key="ask_voice_transcribe"):
            if audio is None:
                st.warning("Record your question first, then press Transcribe.")
            else:
                result = run_action(
                    lambda: voice_service.transcribe(ctx, audio.getvalue(), language=language),
                    component="voice",
                )
                if result is not None:
                    st.session_state["ask_voice_text"] = result["text"]
                    st.session_state["ask_voice_info"] = (
                        f"Recognized with the {'offline' if result['offline'] else 'online'} engine "
                        f"({result['engine']}), language {result['language']}, "
                        f"{result['duration_seconds']:.1f} seconds."
                    )
        text = st.text_area(
            "Recognized text (edit it before asking)",
            key="ask_voice_text",
            max_chars=1000,
            help="The question you submit is exactly this text.",
        )
        info = st.session_state.get("ask_voice_info")
        if info:
            st.caption(md_escape(info))
        if st.button("Ask this question", key="ask_voice_submit"):
            if not text.strip():
                st.warning("There is no text to ask yet. Record and transcribe, or type the question above.")
            else:
                _ask_text(ctx, dataset_id, text, mode="voice")


# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


def _aggregations_for(logical_type: str) -> list[str]:
    allowed = []
    for agg in analytics_engine.AGGREGATIONS:
        if agg in NUMERIC_ONLY_AGGS and logical_type not in NUMERIC_TYPES:
            continue
        if agg in ORDERED_AGGS and logical_type not in NUMERIC_TYPES | {LogicalType.DATETIME.value}:
            continue
        allowed.append(agg)
    return allowed


def _filter_rows(prefix: str, names: list[str]) -> list[dict[str, Any]]:
    count = int(st.number_input("Number of filters", 0, analytics_engine.MAX_FILTERS, 0, key=f"{prefix}_filter_count"))
    filters = []
    for i in range(count):
        c1, c2, c3 = st.columns([2, 1, 2])
        column = c1.selectbox("Filter column", names, key=f"{prefix}_filter_col_{i}")
        op = c2.selectbox("Operator", list(analytics_engine.FILTER_OPS), key=f"{prefix}_filter_op_{i}")
        item: dict[str, Any] = {"column": column, "op": op, "value": None}
        if op not in VALUE_FREE_OPS:
            raw = c3.text_input(
                "Value",
                key=f"{prefix}_filter_value_{i}",
                help="For in, not in and between, separate values with commas.",
            )
            item["value"] = [v.strip() for v in raw.split(",") if v.strip()] if op in LIST_OPS else raw
        filters.append(item)
    return filters


def _query_builder(ctx, caps, dataset_id: int, columns: list[dict[str, Any]]) -> None:
    with st.expander("Query builder"):
        if not columns:
            st.info("The query builder needs the column summary, which is not available for this dataset.")
            return
        prefix = f"ask_qb_{dataset_id}"
        advanced = caps.has(Feature.ADVANCED_ANALYTICS)
        types = {c["name"]: c["logical_type"] for c in columns}
        names = list(types)
        dates = [n for n, t in types.items() if t == LogicalType.DATETIME.value]
        operations = [
            op
            for op in analytics_engine.OPERATIONS
            if op != "clarify" and (advanced or op not in analytics_engine.ADVANCED_OPERATIONS)
        ]
        st.caption(
            "Build an analysis without typing a question. The plan is checked against this dataset before it runs. "
            "Time series, growth and anomalies need a date column; top, group and compare need a group by column."
        )
        if not advanced:
            feature_unavailable(
                "Correlation, growth and unusual-change operations need the advanced analytics feature."
            )
        operation = st.selectbox("Operation", operations, format_func=_label, key=f"{prefix}_operation")
        metric_columns = st.multiselect(
            "Metric columns", names, max_selections=analytics_engine.MAX_METRICS, key=f"{prefix}_metrics"
        )
        metrics = []
        for column in metric_columns:
            options = _aggregations_for(types[column])
            agg = st.selectbox(
                f"Aggregation for {md_escape(column)}", options, key=f"{prefix}_agg_{column}"
            )
            metrics.append({"column": column, "agg": agg})
        group_by = st.multiselect(
            "Group by", names, max_selections=analytics_engine.MAX_GROUP_BY, key=f"{prefix}_group_by"
        )
        filters = _filter_rows(prefix, names)
        t1, t2 = st.columns(2)
        time_column = t1.selectbox("Time column", [NONE_CHOICE, *dates], key=f"{prefix}_time_column")
        frequency = t2.selectbox(
            "Frequency", list(FREQUENCY_LABELS), format_func=FREQUENCY_LABELS.get, index=2, key=f"{prefix}_frequency"
        )
        o1, o2 = st.columns(2)
        limit = int(
            o1.number_input(
                "Row limit (0 for no limit)",
                0,
                get_settings().result_max_rows,
                0,
                key=f"{prefix}_limit",
            )
        )
        if caps.has(Feature.STANDARD_CHARTS):
            chart_types = [
                "auto",
                *(c for c in llm_service.CHART_TYPES if c != "auto" and (advanced or c != "heatmap")),
            ]
            chart_type = o2.selectbox("Chart type", chart_types, format_func=_label, key=f"{prefix}_chart")
        else:
            chart_type = "auto"
            o2.caption("Charts are not included in your plan, so results are shown as tables.")
        extra = st.multiselect(
            "Columns for describe, correlation, distribution, missing values and duplicates",
            names,
            max_selections=analytics_engine.MAX_COLUMNS,
            key=f"{prefix}_columns",
        )
        if st.button("Run query", key=f"{prefix}_run", type="primary"):
            plan: dict[str, Any] = {"operation": operation}
            if metrics:
                plan["metrics"] = metrics
            if group_by:
                plan["group_by"] = group_by
            if filters:
                plan["filters"] = filters
            if time_column != NONE_CHOICE:
                plan["time_column"], plan["frequency"] = time_column, frequency
            if limit:
                plan["limit"] = limit
            if extra:
                plan["columns"] = extra
            if chart_type != "auto":
                plan["chart"] = {"type": chart_type}
            _ask_plan(ctx, dataset_id, plan)


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def _show_answer(ctx, caps, entry: dict[str, Any]) -> None:
    response = entry["response"]
    key = f"ask_{entry['id']}"
    status = response["status"]
    with st.container(border=True):
        with st.chat_message("user"):
            st.text(entry["question"])
            if entry["mode"] == "voice":
                st.caption("Asked by voice")
        with st.chat_message("assistant"):
            if status == "success":
                st.markdown(md_escape(response["answer_text"]))
            elif status == "clarification":
                st.markdown("**The analyst needs more detail and asks:**")
                st.info(md_escape(response["answer_text"]), icon=":material/help:")
            else:
                st.error(md_escape(response["answer_text"]), icon=":material/error:")
            if response.get("narrative"):
                st.markdown("**AI-written explanation**")
                st.caption(
                    "Written by the local AI model from the computed results. Every number in it was checked "
                    "against the calculations."
                )
                st.markdown(md_escape(response["narrative"]))
            if status == "success":
                render_result(
                    table=response["table"],
                    chart_spec=response["chart"],
                    facts=response["facts"],
                    warnings=response["warnings"],
                    key=key,
                    truncated=response["truncated"],
                    total_rows=response["total_rows"],
                    caps=caps,
                )
            else:
                for warning in response["warnings"]:
                    st.warning(md_escape(warning), icon=":material/warning:")
            verification = response.get("verification")
            parts = [f"Planner: {_planner_label(response)}"]
            if status == "success" and verification:
                if verification.get("verified"):
                    parts.append(f"Numbers verified independently ({verification['checks']} checks)")
                else:
                    parts.append("Numbers not verified")
            st.caption(md_escape(" | ".join(parts)))
            if response.get("plan"):
                with st.expander("How this was calculated"):
                    st.json(response["plan"])
            if status == "success":
                result_actions(
                    ctx, caps, response["history_id"], default_title=entry["question"], key=key
                )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

ctx = require_ctx()
caps = capabilities()
page_header(
    "Ask the analyst",
    "Ask questions about a dataset in plain language, by text or voice. Every number is calculated from your data.",
)
if caps is None or not caps.has(Feature.TEXT_AI_ANALYSIS):
    feature_unavailable("Asking the analyst is not included in your plan.")
    st.stop()

_show_ai_status()
dataset = dataset_selector(ctx, key="ask_dataset")
if dataset is None:
    st.stop()
dataset_id = dataset["dataset_id"]
columns = _schema_columns(ctx, dataset)
_show_schema(columns)

prompt = st.chat_input("Ask a question about this dataset", key="ask_chat_input", max_chars=1000)
if prompt:
    _ask_text(ctx, dataset_id, prompt)

st.markdown("**Example questions**")
st.caption(
    "Examples use common business columns such as revenue, product, region and date. "
    "If your dataset names things differently, the analyst will ask you to rephrase."
)
example_columns = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if example_columns[i % 2].button(example, key=f"ask_example_{i}", width="stretch"):
        _ask_text(ctx, dataset_id, example)

if not caps.has(Feature.VOICE_ANALYSIS):
    feature_unavailable("Voice questions are not included in your plan. Type your question instead.")
elif not caps.can(Permission.USE_VOICE):
    feature_unavailable(
        "Voice questions need the use voice permission. Ask an Organization Admin to grant it, "
        "or type your question instead."
    )
else:
    _voice_panel(ctx, dataset_id)
_query_builder(ctx, caps, dataset_id, columns)

entries = st.session_state.get(_conversation_key(dataset_id), [])
st.subheader("Conversation")
if not entries:
    st.info("No questions yet for this dataset. Type a question below, or press an example.")
else:
    if st.button(
        "Clear conversation",
        key="ask_clear",
        help="Removes the answers from this screen. Your analysis history is kept.",
    ):
        st.session_state[_conversation_key(dataset_id)] = []
        st.rerun()
    st.caption("Newest answer first. Every question is also saved in your analysis history.")
    for entry in reversed(entries):
        _show_answer(ctx, caps, entry)
