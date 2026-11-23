"""Voice service: WAV validation, engine routing, offline/online recognition, microphone errors.

No audio hardware, network or real speech engine is used: ``vosk`` is replaced by a
fake module and ``recognize_google`` / ``sr.Microphone`` are monkeypatched.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import types
import wave

import numpy as np
import pytest
import speech_recognition as sr
from sqlalchemy import func, select

from config.settings import set_settings
from core.errors import (
    EmptyAudioError,
    ExternalServiceError,
    FeatureNotAvailableError,
    MicrophoneError,
    PermissionDeniedError,
    SpeechEngineUnavailableError,
    TranscriptionTimeoutError,
    UnsupportedLanguageError,
    ValidationError,
    VoiceError,
)
from database import session_scope
from database.models import AuditLog, UsageRecord
from services import voice_service

TRANSCRIPT = "total revenue by region"


def make_wav(
    seconds: float = 1.0,
    *,
    rate: int = 16000,
    channels: int = 1,
    width: int = 2,
    freq: float = 440.0,
    amp: float = 0.5,
) -> bytes:
    """PCM WAV with a sine tone (``amp`` 0 gives silence)."""
    t = np.arange(int(seconds * rate)) / rate
    wave_data = amp * np.sin(2 * np.pi * freq * t)
    if width == 1:
        raw = (wave_data * 127 + 128).astype(np.uint8)
        frames = np.repeat(raw[:, None], channels, axis=1).tobytes()
    elif width == 2:
        raw = (wave_data * 32767).astype("<i2")
        frames = np.repeat(raw[:, None], channels, axis=1).tobytes()
    elif width == 3:
        ints = (wave_data * (2**23 - 1)).astype("<i4")
        packed = np.repeat(ints[:, None], channels, axis=1).reshape(-1)
        frames = b"".join(int(v).to_bytes(4, "little", signed=True)[:3] for v in packed)
    else:
        raw = (wave_data * (2**31 - 1)).astype("<i4")
        frames = np.repeat(raw[:, None], channels, axis=1).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return buffer.getvalue()


class FakeVosk(types.ModuleType):
    """Stands in for the ``vosk`` package and records how it was used."""

    def __init__(self, text: str = TRANSCRIPT) -> None:
        super().__init__("vosk")
        self.text = text
        self.model_paths: list[str] = []
        self.rates: list[int] = []
        self.fed_bytes: list[int] = []
        outer = self

        class Model:
            def __init__(self, path: str) -> None:
                outer.model_paths.append(path)

        class KaldiRecognizer:
            def __init__(self, model: object, rate: int) -> None:
                outer.rates.append(rate)

            def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802 - vosk API name
                outer.fed_bytes.append(len(data))
                return False

            def FinalResult(self) -> str:  # noqa: N802 - vosk API name
                return json.dumps({"text": outer.text})

        self.Model = Model
        self.KaldiRecognizer = KaldiRecognizer


@pytest.fixture(autouse=True)
def optional_packages_absent(monkeypatch):
    """Tests start as on a machine without vosk and PyAudio, whatever is installed."""
    monkeypatch.setitem(sys.modules, "vosk", None)  # makes "import vosk" raise ImportError

    def no_pyaudio():
        raise AttributeError("Could not find PyAudio; check installation")

    monkeypatch.setattr(sr.Microphone, "get_pyaudio", staticmethod(no_pyaudio))

    class NoPyAudioMicrophone:
        get_pyaudio = staticmethod(no_pyaudio)

        def __init__(self, *args, **kwargs):
            no_pyaudio()

    monkeypatch.setattr(sr, "Microphone", NoPyAudioMicrophone)


@pytest.fixture(autouse=True)
def clean_model_cache():
    voice_service._models.clear()
    yield
    voice_service._models.clear()


@pytest.fixture
def vosk(monkeypatch, app_settings) -> FakeVosk:
    """Fake vosk package plus a non-empty model folder."""
    fake = FakeVosk()
    monkeypatch.setitem(sys.modules, "vosk", fake)
    app_settings.vosk_model_path.mkdir(parents=True)
    (app_settings.vosk_model_path / "conf.txt").write_text("model", encoding="utf-8")
    return fake


@pytest.fixture
def overrides(app_settings):
    def apply(**changes):
        set_settings(app_settings.with_overrides(**changes))

    return apply


@pytest.fixture
def no_network(monkeypatch):
    calls: list[str] = []

    def forbidden(self, audio, language="en-US", **kwargs):
        calls.append(language)
        raise AssertionError("the online speech service must not be called")

    monkeypatch.setattr(sr.Recognizer, "recognize_google", forbidden)
    return calls


def _usage_count(org_id: int) -> int:
    with session_scope() as db:
        return db.execute(
            select(func.count(UsageRecord.id)).where(
                UsageRecord.organization_id == org_id, UsageRecord.metric == "voice_query"
            )
        ).scalar_one()


# ---------------------------------------------------------------------------
# engine_status
# ---------------------------------------------------------------------------


def test_engine_status_without_vosk(app_settings):
    status = voice_service.engine_status()
    assert status["vosk"]["available"] is False
    assert status["vosk"]["package_installed"] is False and status["vosk"]["model_present"] is False
    assert status["vosk"]["offline"] is True and status["vosk"]["model_language"] == "en-US"
    assert (
        "pip install vosk" in status["vosk"]["setup_hint"]
        and "VOSK_MODEL_PATH" in status["vosk"]["setup_hint"]
    )
    assert str(app_settings.vosk_model_path) not in status["vosk"]["setup_hint"]
    assert status["online"]["enabled"] is False and status["online"]["offline"] is False
    assert "Google" in status["online"]["warning"]
    assert status["server_microphone"] == {"available": False}
    assert status["supported_languages"] == ["en-US", "en-IN", "en-GB", "hi-IN"]
    assert status["default_language"] == "en-US"


def test_engine_status_with_vosk_and_model(vosk):
    status = voice_service.engine_status()["vosk"]
    assert status["available"] is True and status["package_installed"] and status["model_present"]
    assert "ready" in status["setup_hint"]


def test_engine_status_package_without_model(monkeypatch):
    monkeypatch.setitem(sys.modules, "vosk", FakeVosk())
    status = voice_service.engine_status()["vosk"]
    assert (
        status["package_installed"] is True
        and status["model_present"] is False
        and status["available"] is False
    )
    assert "pip install" not in status["setup_hint"] and "VOSK_MODEL_PATH" in status["setup_hint"]


def test_engine_status_empty_model_folder_and_no_model_path(vosk, app_settings, overrides):
    for child in app_settings.vosk_model_path.iterdir():
        child.unlink()
    assert voice_service.engine_status()["vosk"]["model_present"] is False
    overrides(vosk_model_path=None)
    assert voice_service.engine_status()["vosk"]["available"] is False


def test_engine_status_reports_microphone_and_online_flag(monkeypatch, overrides):
    monkeypatch.setattr(sr.Microphone, "get_pyaudio", staticmethod(lambda: object()), raising=False)
    overrides(voice_online_recognition_enabled=True)
    status = voice_service.engine_status()
    assert status["server_microphone"]["available"] is True and status["online"]["enabled"] is True


# ---------------------------------------------------------------------------
# transcribe: offline (Vosk)
# ---------------------------------------------------------------------------


def test_transcribe_with_vosk_is_offline_and_audited_without_text(vosk, admin_ctx, org_id):
    result = voice_service.transcribe(admin_ctx, make_wav(1.5))
    assert result == {
        "text": TRANSCRIPT,
        "engine": "vosk",
        "offline": True,
        "language": "en-US",
        "duration_seconds": 1.5,
    }
    assert vosk.rates == [16000] and sum(vosk.fed_bytes) == 1.5 * 16000 * 2
    assert _usage_count(org_id) == 1
    with session_scope() as db:
        row = db.execute(select(AuditLog).where(AuditLog.action == "voice.transcribe")).scalar_one()
        assert row.organization_id == org_id and row.actor_user_id == admin_ctx.user_id
        details = json.loads(row.details_json)
        assert details["engine"] == "vosk" and details["offline"] is True
        assert TRANSCRIPT not in row.details_json and "revenue" not in row.details_json
        usage = db.execute(
            select(UsageRecord.details_json).where(UsageRecord.metric == "voice_query")
        ).scalar_one()
        assert TRANSCRIPT not in usage


def test_vosk_model_is_loaded_once_and_audio_is_resampled(vosk, admin_ctx, app_settings):
    wav = make_wav(1.0, rate=44100, channels=2)
    first = voice_service.transcribe(admin_ctx, wav)
    voice_service.transcribe(admin_ctx, wav, engine="vosk")
    assert vosk.model_paths == [str(app_settings.vosk_model_path)]
    assert first["duration_seconds"] == 1.0
    # 1 second of 44.1 kHz stereo audio is converted to about 16000 mono 16-bit samples.
    per_call = sum(vosk.fed_bytes) / 2
    assert abs(per_call - 16000 * 2) < 200


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_all_pcm_widths_are_accepted(vosk, admin_ctx, width):
    assert voice_service.transcribe(admin_ctx, make_wav(0.8, width=width))["text"] == TRANSCRIPT


def test_transcript_whitespace_is_normalised(vosk, admin_ctx):
    vosk.text = "  total   revenue \n by region  "
    assert voice_service.transcribe(admin_ctx, make_wav())["text"] == TRANSCRIPT


def test_empty_transcript_is_empty_audio(vosk, admin_ctx, org_id):
    vosk.text = "   "
    with pytest.raises(EmptyAudioError):
        voice_service.transcribe(admin_ctx, make_wav())
    assert _usage_count(org_id) == 0


def test_analyst_with_use_voice_can_transcribe(vosk, analyst_ctx):
    assert voice_service.transcribe(analyst_ctx, make_wav())["engine"] == "vosk"


# ---------------------------------------------------------------------------
# Language and engine routing
# ---------------------------------------------------------------------------


def test_unsupported_language(vosk, admin_ctx):
    with pytest.raises(UnsupportedLanguageError):
        voice_service.transcribe(admin_ctx, make_wav(), language="fr-FR")
    with pytest.raises(UnsupportedLanguageError):
        voice_service.transcribe(admin_ctx, make_wav(), language="")  # blank is not a language either


def test_language_is_case_insensitive_and_normalised(vosk, admin_ctx):
    assert voice_service.transcribe(admin_ctx, make_wav(), language="EN-us")["language"] == "en-US"


def test_vosk_supports_only_its_model_language(vosk, admin_ctx):
    with pytest.raises(UnsupportedLanguageError) as info:
        voice_service.transcribe(admin_ctx, make_wav(), language="hi-IN", engine="vosk")
    assert "en-US" in info.value.user_message
    # auto without the online engine reports the same limitation instead of a wrong transcript.
    with pytest.raises(UnsupportedLanguageError):
        voice_service.transcribe(admin_ctx, make_wav(), language="hi-IN")


def test_auto_falls_back_to_online_for_other_languages_when_enabled(vosk, admin_ctx, overrides, monkeypatch):
    overrides(voice_online_recognition_enabled=True)
    seen: list[str] = []
    monkeypatch.setattr(
        sr.Recognizer,
        "recognize_google",
        lambda self, audio, language="en-US", **kw: seen.append(language) or "namaste",
    )
    result = voice_service.transcribe(admin_ctx, make_wav(), language="hi-IN")
    assert result["engine"] == "online" and result["offline"] is False and seen == ["hi-IN"]
    # English still uses the offline engine when it is available.
    assert voice_service.transcribe(admin_ctx, make_wav())["engine"] == "vosk"


def test_no_engine_available_gives_setup_instructions(admin_ctx, no_network):
    with pytest.raises(SpeechEngineUnavailableError) as info:
        voice_service.transcribe(admin_ctx, make_wav())
    message = info.value.user_message
    assert (
        "pip install vosk" in message
        and "VOSK_MODEL_PATH" in message
        and "VOICE_ONLINE_RECOGNITION_ENABLED" in message
    )
    assert no_network == []


def test_explicit_vosk_without_setup_is_unavailable(admin_ctx):
    with pytest.raises(SpeechEngineUnavailableError):
        voice_service.transcribe(admin_ctx, make_wav(), engine="vosk")


def test_online_is_disabled_by_default_and_makes_no_network_call(admin_ctx, no_network):
    with pytest.raises(SpeechEngineUnavailableError) as info:
        voice_service.transcribe(admin_ctx, make_wav(), engine="online")
    assert "VOICE_ONLINE_RECOGNITION_ENABLED" in info.value.user_message
    assert no_network == []


def test_online_enabled_uses_recognize_google(admin_ctx, overrides, monkeypatch):
    overrides(voice_online_recognition_enabled=True)
    calls = []

    def fake(self, audio, language="en-US", **kwargs):
        calls.append((type(audio).__name__, language, len(audio.get_raw_data())))
        return "average sales by category"

    monkeypatch.setattr(sr.Recognizer, "recognize_google", fake)
    result = voice_service.transcribe(admin_ctx, make_wav(1.0), language="en-GB")
    assert result == {
        "text": "average sales by category",
        "engine": "online",
        "offline": False,
        "language": "en-GB",
        "duration_seconds": 1.0,
    }
    assert calls == [("AudioData", "en-GB", 32000)]
    with session_scope() as db:
        assert (
            "average sales"
            not in db.execute(
                select(AuditLog.details_json).where(AuditLog.action == "voice.transcribe")
            ).scalar_one()
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (sr.UnknownValueError(), EmptyAudioError),
        (sr.RequestError("connection failed: secret-host.internal"), ExternalServiceError),
        (RuntimeError("engine blew up at C:\\secret\\path"), VoiceError),
    ],
)
def test_online_errors_are_mapped_and_safe(admin_ctx, overrides, monkeypatch, org_id, error, expected):
    overrides(voice_online_recognition_enabled=True)

    def fail(self, audio, language="en-US", **kwargs):
        raise error

    monkeypatch.setattr(sr.Recognizer, "recognize_google", fail)
    with pytest.raises(expected) as info:
        voice_service.transcribe(admin_ctx, make_wav())
    assert "secret" not in info.value.user_message
    assert _usage_count(org_id) == 0


def test_invalid_engine_name(vosk, admin_ctx):
    with pytest.raises(ValidationError):
        voice_service.transcribe(admin_ctx, make_wav(), engine="cloud")


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------


def test_silence_is_rejected(vosk, admin_ctx, org_id):
    with pytest.raises(EmptyAudioError):
        voice_service.transcribe(admin_ctx, make_wav(1.0, amp=0.0))
    assert _usage_count(org_id) == 0


def test_silence_threshold_boundary(vosk, admin_ctx):
    # A sine of amplitude a has RMS a / sqrt(2): 0.006 -> 0.0042 (silent), 0.01 -> 0.0071 (speech).
    with pytest.raises(EmptyAudioError):
        voice_service.transcribe(admin_ctx, make_wav(1.0, amp=0.006))
    assert voice_service.transcribe(admin_ctx, make_wav(1.0, amp=0.01))["text"] == TRANSCRIPT


def test_too_short_recording(vosk, admin_ctx):
    with pytest.raises(EmptyAudioError):
        voice_service.transcribe(admin_ctx, make_wav(0.2))
    assert voice_service.transcribe(admin_ctx, make_wav(0.5))["text"] == TRANSCRIPT


def test_minimum_duration_boundary_is_inclusive(vosk, admin_ctx, overrides):
    overrides(voice_min_seconds=1.0)
    assert voice_service.transcribe(admin_ctx, make_wav(1.0))["duration_seconds"] == 1.0
    with pytest.raises(EmptyAudioError):
        voice_service.transcribe(admin_ctx, make_wav(0.9))


@pytest.mark.parametrize(
    "data",
    [b"", b"not a wav file at all", b"RIFF\x00\x00\x00\x00WAVE", make_wav(1.0)[:20], bytearray()],
)
def test_corrupt_audio_is_a_voice_error(vosk, admin_ctx, data):
    with pytest.raises(VoiceError) as info:
        voice_service.transcribe(admin_ctx, data)
    assert "could not be read" in info.value.user_message


def test_non_pcm_wav_is_unreadable(vosk, admin_ctx):
    wav = bytearray(make_wav(1.0))
    wav[20:22] = (3).to_bytes(2, "little")  # IEEE float format tag
    with pytest.raises(VoiceError):
        voice_service.transcribe(admin_ctx, bytes(wav))


def test_truncated_data_chunk_is_rejected(vosk, admin_ctx):
    wav = make_wav(1.0)
    with pytest.raises(EmptyAudioError):
        voice_service.transcribe(admin_ctx, wav[:-20000])


def test_too_long_recording(vosk, admin_ctx, overrides):
    overrides(voice_max_seconds=2)
    assert voice_service.transcribe(admin_ctx, make_wav(2.0))["duration_seconds"] == 2.0
    with pytest.raises(VoiceError) as info:
        voice_service.transcribe(admin_ctx, make_wav(2.5))
    assert "too long" in info.value.user_message


def test_oversized_payload_with_small_duration_is_rejected(vosk, admin_ctx, overrides):
    overrides(voice_max_seconds=1)
    big = make_wav(0.9, rate=192000, channels=2, width=4)  # short but far above any sensible bitrate
    with pytest.raises(VoiceError) as info:
        voice_service.transcribe(admin_ctx, big)
    assert "too large" in info.value.user_message


def test_memory_check_runs_before_recognition(vosk, admin_ctx, monkeypatch):
    from services import resource_manager

    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: 1024)
    from core.errors import ResourceLimitError

    with pytest.raises(ResourceLimitError):
        voice_service.transcribe(admin_ctx, make_wav(1.0))


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_slow_engine_times_out(vosk, admin_ctx, overrides, org_id):
    release = threading.Event()

    def slow_model_load(path):
        release.wait(5)

    vosk.Model = slow_model_load
    overrides(voice_transcription_timeout_seconds=0.2)
    try:
        with pytest.raises(TranscriptionTimeoutError):
            voice_service.transcribe(admin_ctx, make_wav())
    finally:
        release.set()
    assert _usage_count(org_id) == 0


# ---------------------------------------------------------------------------
# Server microphone
# ---------------------------------------------------------------------------


class FakeMicrophone:
    opened = 0
    enter_error: BaseException | None = None

    def __init__(self, *args, **kwargs):
        FakeMicrophone.opened += 1
        self.kwargs = kwargs

    def __enter__(self):
        if FakeMicrophone.enter_error:
            raise FakeMicrophone.enter_error
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def microphone(monkeypatch):
    FakeMicrophone.opened = 0
    FakeMicrophone.enter_error = None
    monkeypatch.setattr(sr, "Microphone", FakeMicrophone)
    return FakeMicrophone


def _listen_returning(monkeypatch, wav: bytes | None = None, error: BaseException | None = None):
    captured: dict = {}

    def listen(self, source, timeout=None, phrase_time_limit=None, **kwargs):
        captured.update(timeout=timeout, phrase_time_limit=phrase_time_limit)
        if error:
            raise error
        with wave.open(io.BytesIO(wav), "rb") as reader:
            frames = reader.readframes(reader.getnframes())
            return sr.AudioData(frames, reader.getframerate(), reader.getsampwidth())

    monkeypatch.setattr(sr.Recognizer, "listen", listen)
    return captured


def test_microphone_without_pyaudio_explains_browser_alternative(vosk, admin_ctx):
    with pytest.raises(MicrophoneError) as info:
        voice_service.record_from_server_microphone(admin_ctx)
    assert "PyAudio" in info.value.user_message and "browser" in info.value.user_message


def test_microphone_oserror_gives_windows_hint(vosk, admin_ctx, monkeypatch):
    class Broken:
        def __init__(self, *a, **k):
            raise OSError("no default input device")

    monkeypatch.setattr(sr, "Microphone", Broken)
    with pytest.raises(MicrophoneError) as info:
        voice_service.record_from_server_microphone(admin_ctx)
    assert "Settings > Privacy & security > Microphone" in info.value.user_message
    assert "no default input device" not in info.value.user_message


def test_microphone_open_error_gives_windows_hint(vosk, admin_ctx, microphone):
    microphone.enter_error = OSError(-9996, "Invalid input device")
    with pytest.raises(MicrophoneError) as info:
        voice_service.record_from_server_microphone(admin_ctx)
    assert "allow desktop apps" in info.value.user_message


def test_microphone_wait_timeout_is_empty_audio(vosk, admin_ctx, microphone, monkeypatch):
    _listen_returning(monkeypatch, error=sr.WaitTimeoutError("listening timed out"))
    with pytest.raises(EmptyAudioError):
        voice_service.record_from_server_microphone(admin_ctx, seconds=2)


def test_microphone_success_runs_the_same_pipeline(vosk, admin_ctx, microphone, monkeypatch, org_id):
    captured = _listen_returning(monkeypatch, wav=make_wav(1.0))
    result = voice_service.record_from_server_microphone(admin_ctx, seconds=3)
    assert result["text"] == TRANSCRIPT and result["engine"] == "vosk" and result["offline"] is True
    assert captured == {"timeout": 3, "phrase_time_limit": 3}
    assert microphone.opened == 1 and _usage_count(org_id) == 1
    with session_scope() as db:
        details = json.loads(
            db.execute(
                select(AuditLog.details_json).where(AuditLog.action == "voice.transcribe")
            ).scalar_one()
        )
        assert details["source"] == "server_microphone"


def test_microphone_silence_is_rejected_by_the_shared_pipeline(vosk, admin_ctx, microphone, monkeypatch):
    _listen_returning(monkeypatch, wav=make_wav(1.0, amp=0.0))
    with pytest.raises(EmptyAudioError):
        voice_service.record_from_server_microphone(admin_ctx)


@pytest.mark.parametrize("seconds", [0, -1, 61, True, 2.5, "5"])
def test_microphone_seconds_validation(vosk, admin_ctx, microphone, seconds):
    with pytest.raises(ValidationError):
        voice_service.record_from_server_microphone(admin_ctx, seconds=seconds)
    assert microphone.opened == 0


def test_microphone_not_opened_when_engine_or_language_unusable(admin_ctx, microphone, vosk):
    with pytest.raises(UnsupportedLanguageError):
        voice_service.record_from_server_microphone(admin_ctx, language="fr-FR")
    with pytest.raises(UnsupportedLanguageError):
        voice_service.record_from_server_microphone(admin_ctx, language="hi-IN", engine="vosk")
    with pytest.raises(SpeechEngineUnavailableError):
        voice_service.record_from_server_microphone(admin_ctx, engine="online")
    assert microphone.opened == 0


def test_microphone_engine_unavailable_before_opening(admin_ctx, microphone):
    with pytest.raises(SpeechEngineUnavailableError):
        voice_service.record_from_server_microphone(admin_ctx)
    assert microphone.opened == 0


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


def test_analyst_without_use_voice_is_denied(vosk, factory, org_id, no_network):
    analyst = factory.analyst(org_id, permissions={"export_data"})
    with pytest.raises(PermissionDeniedError):
        voice_service.transcribe(analyst, make_wav())
    with pytest.raises(PermissionDeniedError):
        voice_service.record_from_server_microphone(analyst)
    assert _usage_count(org_id) == 0


def test_basic_plan_is_denied(vosk, factory):
    basic_org = factory.organization(plan="basic")
    admin = factory.org_admin(basic_org)
    with pytest.raises(FeatureNotAvailableError):
        voice_service.transcribe(admin, make_wav())
    with pytest.raises(FeatureNotAvailableError):
        voice_service.record_from_server_microphone(admin)
    assert _usage_count(basic_org) == 0


def test_super_admin_and_suspended_org_are_denied(vosk, factory, admin_ctx, org_id):
    with pytest.raises(PermissionDeniedError):
        voice_service.transcribe(factory.super_admin(), make_wav())
    from database.models import Organization

    with session_scope() as db:
        db.get(Organization, org_id).status = "suspended"
    with pytest.raises(PermissionDeniedError):
        voice_service.transcribe(admin_ctx, make_wav())


def test_authorization_is_checked_before_the_audio(factory, org_id):
    analyst = factory.analyst(org_id, permissions=set())
    with pytest.raises(PermissionDeniedError):
        voice_service.transcribe(analyst, b"garbage", language="fr-FR")
