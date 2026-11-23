"""Voice routing: speech to text, user edit, then the same analytics pipeline as typed questions.

Spec sections 13, 15 and the "voice routing tests with the microphone mocked" acceptance item.
The recognition seam is the ``vosk`` package that ``services.voice_service`` imports lazily: a fake
module is placed in ``sys.modules`` (with a non-empty model folder so the offline engine counts as
installed). The microphone is replaced by a fake ``sr.Microphone``. The online engine
(``recognize_google``) is replaced by a recorder that fails the test if it is ever called, and every
socket connection is checked so that nothing leaves the loopback interface.
"""

from __future__ import annotations

import io
import json
import socket
import sys
import threading
import types
import wave

import numpy as np
import pytest
import speech_recognition as sr
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select

from config.settings import Settings, set_settings
from core.constants import Permission, Role, UsageMetric
from core.errors import (
    EmptyAudioError,
    FeatureNotAvailableError,
    MicrophoneError,
    PermissionDeniedError,
    SpeechEngineUnavailableError,
    TranscriptionTimeoutError,
    UnsupportedLanguageError,
    VoiceError,
)
from database import session_scope
from database.models import AnalysisHistory, Base, UsageRecord
from services import analytics_engine, voice_service

# What the (fake) engine "hears": a typical recognition slip the user then corrects.
HEARD = "what is the totel revenue"
EDITED = "What is the total revenue?"
SALES_TOTAL = 5840.75  # sum of the conftest sales_frame revenue column


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def make_wav(seconds: float = 1.0, *, amp: float = 0.5, rate: int = 16000) -> bytes:
    """Mono 16-bit PCM WAV holding a sine tone (``amp=0`` gives silence)."""
    t = np.arange(int(seconds * rate)) / rate
    frames = (amp * np.sin(2 * np.pi * 440.0 * t) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return buffer.getvalue()


class FakeVosk(types.ModuleType):
    """Stands in for the ``vosk`` package. ``block`` makes recognition hang until released."""

    def __init__(self, text: str = HEARD) -> None:
        super().__init__("vosk")
        self.text = text
        self.calls = 0
        self.block: threading.Event | None = None
        outer = self

        class Model:
            def __init__(self, path: str) -> None:
                self.path = path

        class KaldiRecognizer:
            def __init__(self, model: object, rate: int) -> None:
                outer.calls += 1

            def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802 - vosk API name
                return False

            def FinalResult(self) -> str:  # noqa: N802 - vosk API name
                if outer.block is not None:
                    outer.block.wait(10)
                return json.dumps({"text": outer.text})

        self.Model = Model
        self.KaldiRecognizer = KaldiRecognizer


@pytest.fixture(autouse=True)
def no_hardware_no_network(monkeypatch):
    """No PyAudio, no online recognition, and no socket connection outside 127.0.0.1."""

    def no_pyaudio():
        raise AttributeError("Could not find PyAudio; check installation")

    class NoMicrophone:
        get_pyaudio = staticmethod(no_pyaudio)

        def __init__(self, *args, **kwargs):
            no_pyaudio()

    monkeypatch.setattr(sr, "Microphone", NoMicrophone)
    monkeypatch.setitem(sys.modules, "vosk", None)  # "import vosk" raises ImportError

    online_calls: list[str] = []

    def online_forbidden(self, audio, language="en-US", **kwargs):
        online_calls.append(language)
        raise AssertionError("the online speech service must not be called")

    monkeypatch.setattr(sr.Recognizer, "recognize_google", online_forbidden)

    remote: list[object] = []
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            remote.append(address)
            raise AssertionError(f"network access attempted: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    voice_service._models.clear()
    yield {"online": online_calls, "remote": remote}
    voice_service._models.clear()
    assert online_calls == [], "online recognition was called"
    assert remote == [], "a non-loopback connection was attempted"


@pytest.fixture
def vosk(monkeypatch, app_settings) -> FakeVosk:
    fake = FakeVosk()
    monkeypatch.setitem(sys.modules, "vosk", fake)
    app_settings.vosk_model_path.mkdir(parents=True, exist_ok=True)
    (app_settings.vosk_model_path / "conf.txt").write_text("model", encoding="utf-8")
    return fake


@pytest.fixture
def sales_id(factory, admin_ctx, sales_frame) -> int:
    return factory.dataset(admin_ctx, sales_frame, name="sales")


def usage_totals(org_id: int) -> dict[str, float]:
    with session_scope() as db:
        totals: dict[str, float] = {}
        for record in db.execute(select(UsageRecord).where(UsageRecord.organization_id == org_id)).scalars():
            totals[record.metric] = totals.get(record.metric, 0.0) + record.quantity
        return totals


def history_rows(org_id: int) -> list[dict]:
    with session_scope() as db:
        rows = db.execute(
            select(AnalysisHistory)
            .where(AnalysisHistory.organization_id == org_id)
            .order_by(AnalysisHistory.id)
        ).scalars()
        return [
            {
                "id": r.id,
                "question": r.question,
                "input_mode": r.input_mode,
                "status": r.status,
                "plan": r.plan_json,
            }
            for r in rows
        ]


def all_text_values() -> list[str]:
    """Every non-null text cell of every table (to prove the raw transcript is never stored)."""
    values: list[str] = []
    with session_scope() as db:
        for table in Base.metadata.sorted_tables:
            text_columns = [c for c in table.columns if c.type.python_type is str]
            if not text_columns:
                continue
            for row in db.execute(select(*text_columns)).all():
                values.extend(v for v in row if isinstance(v, str))
    return values


def comparable(answer: dict) -> dict:
    table = answer["table"]
    return {
        "status": answer["status"],
        "plan": answer["plan"],
        "planner": answer["planner"],
        "facts": answer["facts"],
        "answer_text": answer["answer_text"],
        "table": None if table is None else table.to_dict(orient="list"),
    }


# ---------------------------------------------------------------------------
# Happy path: transcribe, edit, answer through the same engine
# ---------------------------------------------------------------------------


def test_voice_question_runs_the_same_pipeline_as_typed_text(vosk, admin_ctx, org_id, sales_id):
    heard = voice_service.transcribe(admin_ctx, make_wav())
    assert heard["text"] == HEARD
    assert heard["engine"] == "vosk" and heard["offline"] is True and heard["language"] == "en-US"
    assert vosk.calls == 1

    # The UI shows the recognised text; the user corrects it before submitting.
    edited = heard["text"].replace("totel", "total").capitalize() + "?"
    assert edited == EDITED

    voiced = analytics_engine.answer_question(admin_ctx, sales_id, edited, input_mode="voice")
    typed = analytics_engine.answer_question(admin_ctx, sales_id, edited)
    assert voiced["status"] == "success"
    assert comparable(voiced) == comparable(typed)
    assert voiced["facts"][0]["value"] == pytest.approx(SALES_TOTAL)

    rows = history_rows(org_id)
    assert [r["input_mode"] for r in rows] == ["voice", "text"]
    assert rows[0]["question"] == rows[1]["question"] == EDITED
    assert rows[0]["plan"] == rows[1]["plan"]

    totals = usage_totals(org_id)
    # One VOICE_QUERY for the transcription and one for the voice-mode question.
    assert totals[UsageMetric.VOICE_QUERY.value] == 2.0
    assert totals[UsageMetric.ANALYSIS_QUERY.value] == 2.0

    # Neither the recording nor the unedited transcript is stored anywhere.
    assert not any(HEARD in value for value in all_text_values())


def test_analyst_voice_flow_with_grant(vosk, factory, org_id, admin_ctx, sales_id):
    analyst = factory.analyst(org_id)  # defaults include use_voice
    factory.grant(sales_id, analyst.user_id)
    heard = voice_service.transcribe(analyst, make_wav(), language="en-us")
    assert heard["language"] == "en-US"  # language matching is case-insensitive
    answer = analytics_engine.answer_question(analyst, sales_id, EDITED, input_mode="voice")
    assert answer["status"] == "success" and answer["facts"][0]["value"] == pytest.approx(SALES_TOTAL)
    assert history_rows(org_id)[-1]["input_mode"] == "voice"


def test_server_microphone_is_mocked_and_routes_to_the_offline_engine(vosk, monkeypatch, admin_ctx, org_id):
    tone = make_wav(1.0)
    with wave.open(io.BytesIO(tone)) as reader:
        frames = reader.readframes(reader.getnframes())

    class FakeMicrophone:
        opened = 0

        @staticmethod
        def get_pyaudio():
            return object()

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            FakeMicrophone.opened += 1
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(sr, "Microphone", FakeMicrophone)
    monkeypatch.setattr(sr.Recognizer, "listen", lambda self, source, **kw: sr.AudioData(frames, 16000, 2))
    result = voice_service.record_from_server_microphone(admin_ctx, seconds=2)
    assert FakeMicrophone.opened == 1
    assert result["text"] == HEARD and result["offline"] is True
    assert usage_totals(org_id)[UsageMetric.VOICE_QUERY.value] == 1.0


def test_microphone_permission_error_is_reported(vosk, monkeypatch, admin_ctx):
    class DeniedMicrophone:
        @staticmethod
        def get_pyaudio():
            return object()

        def __init__(self, *args, **kwargs):
            raise OSError("access denied by the operating system")

    monkeypatch.setattr(sr, "Microphone", DeniedMicrophone)
    with pytest.raises(MicrophoneError) as info:
        voice_service.record_from_server_microphone(admin_ctx, seconds=2)
    assert "Privacy" in info.value.user_message and "access denied" not in info.value.user_message


# ---------------------------------------------------------------------------
# Permission and plan feature on both steps
# ---------------------------------------------------------------------------


def test_voice_requires_use_voice_permission_on_both_steps(vosk, factory, org_id, sales_id):
    analyst = factory.analyst(org_id, permissions={Permission.EXPORT_DATA.value})
    factory.grant(sales_id, analyst.user_id)
    with pytest.raises(PermissionDeniedError):
        voice_service.transcribe(analyst, make_wav())
    with pytest.raises(PermissionDeniedError):
        analytics_engine.answer_question(analyst, sales_id, EDITED, input_mode="voice")
    assert vosk.calls == 0 and history_rows(org_id) == []
    # The same analyst may still type the question.
    typed = analytics_engine.answer_question(analyst, sales_id, EDITED)
    assert typed["status"] == "success"
    assert [r["input_mode"] for r in history_rows(org_id)] == ["text"]
    assert UsageMetric.VOICE_QUERY.value not in usage_totals(org_id)


def test_voice_requires_the_voice_analysis_plan_feature_on_both_steps(vosk, factory, sales_frame):
    basic_org = factory.organization("Basic Shop", plan="basic")
    admin = factory.org_admin(basic_org)
    dataset = factory.dataset(admin, sales_frame, name="basic sales")
    with pytest.raises(FeatureNotAvailableError):
        voice_service.transcribe(admin, make_wav())
    with pytest.raises(FeatureNotAvailableError):
        analytics_engine.answer_question(admin, dataset, EDITED, input_mode="voice")
    assert vosk.calls == 0 and history_rows(basic_org) == []
    assert analytics_engine.answer_question(admin, dataset, EDITED)["status"] == "success"
    assert UsageMetric.VOICE_QUERY.value not in usage_totals(basic_org)


def test_super_admin_cannot_use_voice(vosk, factory):
    platform = factory.super_admin()
    assert platform.role == Role.SUPER_ADMIN
    with pytest.raises(PermissionDeniedError):
        voice_service.transcribe(platform, make_wav())
    assert vosk.calls == 0


# ---------------------------------------------------------------------------
# Online recognition is off by default
# ---------------------------------------------------------------------------


def test_online_recognition_is_disabled_by_default(app_settings):
    assert Settings().voice_online_recognition_enabled is False
    assert app_settings.voice_online_recognition_enabled is False
    status = voice_service.engine_status()
    assert status["online"]["enabled"] is False and status["online"]["offline"] is False


def test_without_offline_engine_auto_fails_without_any_upload(admin_ctx, org_id, no_hardware_no_network):
    with pytest.raises(SpeechEngineUnavailableError) as info:
        voice_service.transcribe(admin_ctx, make_wav())
    assert "VOICE_ONLINE_RECOGNITION_ENABLED" in info.value.user_message
    with pytest.raises(SpeechEngineUnavailableError):
        voice_service.transcribe(admin_ctx, make_wav(), engine="online")
    assert no_hardware_no_network["online"] == []
    assert UsageMetric.VOICE_QUERY.value not in usage_totals(org_id)


def test_auto_prefers_the_offline_engine_even_when_online_is_enabled(vosk, app_settings, admin_ctx):
    set_settings(app_settings.with_overrides(voice_online_recognition_enabled=True))
    assert voice_service.transcribe(admin_ctx, make_wav())["engine"] == "vosk"


# ---------------------------------------------------------------------------
# Documented errors
# ---------------------------------------------------------------------------


def test_unsupported_language(vosk, admin_ctx, org_id):
    with pytest.raises(UnsupportedLanguageError) as info:
        voice_service.transcribe(admin_ctx, make_wav(), language="fr-FR")
    assert "en-US" in info.value.user_message
    # Supported by the app, but the installed offline model only speaks en-US.
    with pytest.raises(UnsupportedLanguageError):
        voice_service.transcribe(admin_ctx, make_wav(), language="hi-IN")
    assert vosk.calls == 0
    assert UsageMetric.VOICE_QUERY.value not in usage_totals(org_id)


@pytest.mark.parametrize(
    ("audio", "error"),
    [
        (make_wav(1.0, amp=0.0), EmptyAudioError),  # silence
        (make_wav(0.1), EmptyAudioError),  # shorter than VOICE_MIN_SECONDS
        (b"", VoiceError),  # nothing recorded
        (b"RIFF\x00\x00not a wave file", VoiceError),  # unreadable
    ],
    ids=["silent", "too-short", "empty-bytes", "unreadable"],
)
def test_empty_or_unreadable_audio(vosk, admin_ctx, org_id, audio, error):
    with pytest.raises(error):
        voice_service.transcribe(admin_ctx, audio)
    assert vosk.calls == 0
    assert UsageMetric.VOICE_QUERY.value not in usage_totals(org_id)


def test_engine_hearing_nothing_is_empty_audio(vosk, admin_ctx, org_id):
    vosk.text = "   "
    with pytest.raises(EmptyAudioError):
        voice_service.transcribe(admin_ctx, make_wav())
    assert UsageMetric.VOICE_QUERY.value not in usage_totals(org_id)


def test_recognition_timeout(vosk, app_settings, admin_ctx, org_id):
    set_settings(app_settings.with_overrides(voice_transcription_timeout_seconds=1))
    vosk.block = threading.Event()
    try:
        with pytest.raises(TranscriptionTimeoutError):
            voice_service.transcribe(admin_ctx, make_wav())
    finally:
        vosk.block.set()
    assert UsageMetric.VOICE_QUERY.value not in usage_totals(org_id)


def test_history_input_mode_column_accepts_only_known_modes():
    table = sa_inspect(AnalysisHistory).local_table
    checks = [c.sqltext.text for c in table.constraints if hasattr(c, "sqltext")]
    assert any("'voice'" in text and "input_mode" in text for text in checks)
