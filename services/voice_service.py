"""Speech to text for the voice analyst.

Recording happens in the browser (``st.audio_input`` returns a WAV file) or,
optionally, on the server machine's microphone. This module only converts the
audio to text; the UI shows the text for confirmation and editing and then
submits it to the analytics engine as a typed question.

Engines
* ``vosk``: offline, local model, supports only ``VOSK_MODEL_LANGUAGE``. The
  ``vosk`` package is imported lazily and the model is cached per folder.
  (SpeechRecognition's own ``recognize_vosk`` only reads a fixed folder inside
  site-packages, so the package is called directly.)
* ``online``: Google's web speech service through SpeechRecognition. It uploads
  the recording, so it is disabled unless ``VOICE_ONLINE_RECOGNITION_ENABLED``
  is true. Recordings are never uploaded by default.

Silence rule: a recording is "silent" when its RMS level, normalised to full
scale (0.0 to 1.0), is below ``SILENCE_RMS_THRESHOLD`` (0.005, about -46 dBFS).
Recognition runs in a daemon worker thread so a stuck engine cannot freeze the
request; on timeout the worker is abandoned (it cannot be killed) and a model
load that was in progress still finishes and is cached for the next call.

Transcripts and audio are never written to the database or the audit log.
"""

from __future__ import annotations

import io
import json
import logging
import queue
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import speech_recognition as sr

from auth.authorization import AuthContext, authorize
from config.settings import Settings, get_settings
from core.constants import Feature, Permission, UsageMetric
from core.errors import (
    AppError,
    EmptyAudioError,
    ExternalServiceError,
    MicrophoneError,
    SpeechEngineUnavailableError,
    TranscriptionTimeoutError,
    UnsupportedLanguageError,
    ValidationError,
    VoiceError,
)
from database.connection import session_scope
from modules.usage_tracking import record_usage
from services import resource_manager
from services.audit_service import record_audit

logger = logging.getLogger(__name__)

ENGINES = ("auto", "vosk", "online")
SILENCE_RMS_THRESHOLD = 0.005
VOSK_SAMPLE_RATE = 16_000
_VOSK_CHUNK_BYTES = 8_000
# 48 kHz, stereo, 16-bit: the largest raw stream a browser realistically produces.
_MAX_BYTES_PER_SECOND = 48_000 * 2 * 2
_HEADER_ALLOWANCE_BYTES = 1_024 * 1_024

ONLINE_WARNING = "Online recognition sends your recording to Google's speech service over the internet."
_WINDOWS_MIC_HINT = (
    "The microphone could not be opened. On Windows open Settings > Privacy & security > Microphone "
    "and allow desktop apps to access the microphone, then try again."
)
_NO_PYAUDIO_HINT = (
    "Recording from the server's microphone needs the optional PyAudio package, which is not installed. "
    "Recording in the browser works without it."
)

_model_lock = threading.Lock()
_models: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Engine availability
# ---------------------------------------------------------------------------


def _vosk_package_installed() -> bool:
    try:
        import vosk  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken install counts as not installed
        return False
    return True


def _vosk_model_present(settings: Settings) -> bool:
    path = settings.vosk_model_path
    try:
        return path is not None and path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _microphone_available() -> bool:
    try:
        sr.Microphone.get_pyaudio()
    except Exception:  # noqa: BLE001 - PyAudio missing or unusable
        return False
    return True


def _vosk_setup_hint(package_installed: bool, model_present: bool) -> str:
    steps = []
    if not package_installed:
        steps.append("install the package with: pip install vosk")
    if not model_present:
        steps.append(
            "download a Vosk model from the Vosk models page, unpack it and set VOSK_MODEL_PATH "
            "to the unpacked folder"
        )
    if not steps:
        return "Offline recognition is ready."
    return "To enable offline recognition, " + " and ".join(steps) + "."


def engine_status() -> dict[str, Any]:
    """Report which speech engines can be used right now (no tenant data involved)."""
    settings = get_settings()
    package = _vosk_package_installed()
    model = _vosk_model_present(settings)
    return {
        "vosk": {
            "available": package and model,
            "package_installed": package,
            "model_present": model,
            "model_language": settings.vosk_model_language,
            "offline": True,
            "setup_hint": _vosk_setup_hint(package, model),
        },
        "online": {
            "enabled": settings.voice_online_recognition_enabled,
            "offline": False,
            "warning": ONLINE_WARNING,
        },
        "server_microphone": {"available": _microphone_available()},
        "supported_languages": list(settings.voice_supported_languages),
        "default_language": settings.voice_default_language,
    }


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _resolve_language(language: str | None, settings: Settings) -> str:
    chosen = settings.voice_default_language if language is None else language
    supported = {item.casefold(): item for item in settings.voice_supported_languages}
    if not isinstance(chosen, str) or chosen.casefold() not in supported:
        raise UnsupportedLanguageError(
            "The selected language is not supported. Choose one of: "
            + ", ".join(settings.voice_supported_languages)
            + "."
        )
    return supported[chosen.casefold()]


def _languages_match(a: str, b: str) -> bool:
    return a.casefold() == b.casefold()


def _resolve_engine(engine: str, language: str, settings: Settings) -> str:
    """Pick the concrete engine ("vosk" or "online") or raise a descriptive error."""
    if engine not in ENGINES:
        raise ValidationError("The speech engine must be auto, vosk or online.")
    status = engine_status()
    vosk_ready = status["vosk"]["available"]
    online_on = settings.voice_online_recognition_enabled
    vosk_language_ok = _languages_match(language, settings.vosk_model_language)

    if engine == "vosk":
        if not vosk_ready:
            raise SpeechEngineUnavailableError(
                "Offline recognition is not set up. " + status["vosk"]["setup_hint"]
            )
        return _require_vosk_language(language, settings)
    if engine == "online":
        if not online_on:
            raise SpeechEngineUnavailableError(
                "Online recognition is turned off because it uploads recordings. "
                "Set VOICE_ONLINE_RECOGNITION_ENABLED=true to allow it."
            )
        return "online"
    # auto: prefer the offline engine, then the online one when explicitly enabled.
    if vosk_ready and vosk_language_ok:
        return "vosk"
    if online_on:
        return "online"
    if vosk_ready:
        return _require_vosk_language(language, settings)
    raise SpeechEngineUnavailableError(
        "No speech recognition engine is available. "
        + status["vosk"]["setup_hint"]
        + " Alternatively set VOICE_ONLINE_RECOGNITION_ENABLED=true to use the online service, "
        "which uploads recordings."
    )


def _require_vosk_language(language: str, settings: Settings) -> str:
    if not _languages_match(language, settings.vosk_model_language):
        raise UnsupportedLanguageError(
            f"The offline model supports {settings.vosk_model_language} only. "
            "Choose that language or install a model for the language you need."
        )
    return "vosk"


# ---------------------------------------------------------------------------
# Audio inspection
# ---------------------------------------------------------------------------

_UNREADABLE = "The recording could not be read. Record again, or upload a PCM WAV file."


def _decode_samples(raw: bytes, width: int) -> np.ndarray:
    """Convert raw PCM bytes to floats in [-1, 1] (8-bit unsigned, 16/24/32-bit signed)."""
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if width == 3:
        packed = np.frombuffer(raw[: len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        value = np.where(value >= 1 << 23, value - (1 << 24), value)
        return value.astype(np.float64) / float(1 << 23)
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise VoiceError(_UNREADABLE)


def inspect_wav(audio_bytes: bytes, settings: Settings | None = None) -> dict[str, float]:
    """Validate a PCM WAV recording and return ``{"duration_seconds", "rms"}``.

    Raises ``VoiceError`` for unreadable or too long audio and ``EmptyAudioError``
    for audio that is too short or silent.
    """
    settings = settings or get_settings()
    if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
        raise VoiceError(_UNREADABLE)
    try:
        with wave.open(io.BytesIO(bytes(audio_bytes)), "rb") as reader:
            frames, rate = reader.getnframes(), reader.getframerate()
            width, channels = reader.getsampwidth(), reader.getnchannels()
            if rate <= 0 or channels <= 0:
                raise VoiceError(_UNREADABLE)
            duration = frames / rate
            if duration > settings.voice_max_seconds:
                raise VoiceError(
                    f"The recording is too long. Keep it under {settings.voice_max_seconds} seconds."
                )
            if (
                len(audio_bytes)
                > settings.voice_max_seconds * _MAX_BYTES_PER_SECOND + _HEADER_ALLOWANCE_BYTES
            ):
                raise VoiceError("The recording is too large. Record at a normal quality and keep it short.")
            raw = reader.readframes(frames)
    except (wave.Error, EOFError, ValueError) as exc:
        raise VoiceError(_UNREADABLE) from exc
    samples = _decode_samples(raw, width)
    if samples.size == 0 or len(raw) < frames * width * channels:
        # A header that promises more audio than the file holds is a truncated file.
        raise EmptyAudioError("The recording is empty or incomplete. Record again.")
    if duration < settings.voice_min_seconds:
        raise EmptyAudioError("The recording is too short. Speak for at least a moment and try again.")
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms < SILENCE_RMS_THRESHOLD:
        raise EmptyAudioError("No speech was detected in the recording. Check the microphone and try again.")
    return {"duration_seconds": float(duration), "rms": rms}


# ---------------------------------------------------------------------------
# Recognition engines
# ---------------------------------------------------------------------------


def _decode_audio(wav_bytes: bytes) -> sr.AudioData:
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        return sr.Recognizer().record(source)


def _load_vosk_model(vosk_module: Any, model_path: Path) -> Any:
    """Return the cached Vosk model for ``model_path`` (loaded once, guarded by a lock)."""
    key = str(model_path)
    with _model_lock:
        model = _models.get(key)
        if model is None:
            model = vosk_module.Model(key)
            _models[key] = model
        return model


def _recognize_vosk(audio: sr.AudioData) -> str:
    """Offline recognition with the local Vosk model."""
    import vosk

    settings = get_settings()
    if settings.vosk_model_path is None:
        raise SpeechEngineUnavailableError("Offline recognition is not set up. Set VOSK_MODEL_PATH.")
    model = _load_vosk_model(vosk, settings.vosk_model_path)
    recognizer = vosk.KaldiRecognizer(model, VOSK_SAMPLE_RATE)
    raw = audio.get_raw_data(convert_rate=VOSK_SAMPLE_RATE, convert_width=2)
    for start in range(0, len(raw), _VOSK_CHUNK_BYTES):
        recognizer.AcceptWaveform(raw[start : start + _VOSK_CHUNK_BYTES])
    return str(json.loads(recognizer.FinalResult()).get("text", ""))


def _recognize_online(audio: sr.AudioData, language: str) -> str:
    """Online recognition through Google's web speech API (uploads the audio)."""
    return str(sr.Recognizer().recognize_google(audio, language=language))


def _recognize_worker(wav_bytes: bytes, engine: str, language: str) -> str:
    audio = _decode_audio(wav_bytes)
    return _recognize_vosk(audio) if engine == "vosk" else _recognize_online(audio, language)


def _run_with_timeout(task: Callable[[], str], timeout_seconds: float) -> str:
    """Run ``task`` in a daemon thread; raise ``TranscriptionTimeoutError`` when it is too slow."""
    outcome: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            outcome.put(("ok", task()))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            outcome.put(("error", exc))

    thread = threading.Thread(target=target, name="voice-transcription", daemon=True)
    thread.start()
    try:
        kind, value = outcome.get(timeout=timeout_seconds)
    except queue.Empty:
        raise TranscriptionTimeoutError() from None
    if kind == "error":
        raise value
    return value


def _map_engine_error(exc: BaseException) -> AppError:
    """Translate an engine failure into a user-safe application error."""
    if isinstance(exc, AppError):
        return exc
    if isinstance(exc, sr.UnknownValueError):
        return EmptyAudioError("The speech could not be understood. Speak clearly and try again.")
    if isinstance(exc, sr.RequestError):
        logger.warning("Online speech service request failed (%s)", type(exc).__name__)
        return ExternalServiceError(
            "The online speech service could not be reached. "
            "Check the internet connection or use offline recognition."
        )
    logger.error("Speech recognition failed (%s)", type(exc).__name__, exc_info=exc)
    return VoiceError("Speech recognition failed. Try again, or type the question instead.")


def _recognize(wav_bytes: bytes, engine: str, language: str, settings: Settings) -> str:
    try:
        text = _run_with_timeout(
            lambda: _recognize_worker(wav_bytes, engine, language),
            settings.voice_transcription_timeout_seconds,
        )
    except TranscriptionTimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to user-safe errors
        raise _map_engine_error(exc) from exc
    cleaned = " ".join(text.split())
    if not cleaned:
        raise EmptyAudioError("No speech was recognised in the recording. Try again.")
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _authorize_voice(ctx: AuthContext) -> int:
    with session_scope() as db:
        return authorize(db, ctx, permission=Permission.USE_VOICE, feature=Feature.VOICE_ANALYSIS)


def _process(
    ctx: AuthContext, org_id: int, wav_bytes: bytes, language: str, engine: str, source: str
) -> dict[str, Any]:
    """Validate the audio, recognise it and record usage and audit (without the text)."""
    settings = get_settings()
    info = inspect_wav(wav_bytes, settings)
    resource_manager.check_memory(len(wav_bytes) * 16, operation="Speech recognition")
    chosen = _resolve_engine(engine, language, settings)
    text = _recognize(wav_bytes, chosen, language, settings)
    duration = round(info["duration_seconds"], 2)
    offline = chosen == "vosk"
    with session_scope() as db:
        record_usage(
            db,
            org_id,
            ctx.user_id,
            UsageMetric.VOICE_QUERY,
            1.0,
            {"engine": chosen, "language": language, "source": source},
        )
        record_audit(
            db,
            ctx,
            "voice.transcribe",
            target_type="voice",
            details={
                "engine": chosen,
                "offline": offline,
                "language": language,
                "duration_seconds": duration,
                "source": source,
            },
            organization_id=org_id,
        )
    return {
        "text": text,
        "engine": chosen,
        "offline": offline,
        "language": language,
        "duration_seconds": duration,
    }


def transcribe(
    ctx: AuthContext, audio_bytes: bytes, *, language: str | None = None, engine: str = "auto"
) -> dict[str, Any]:
    """Convert a recorded PCM WAV to text.

    Requires ``Permission.USE_VOICE`` and the ``VOICE_ANALYSIS`` plan feature.
    Errors: ``UnsupportedLanguageError`` (language not in the supported list, or
    not the Vosk model language when Vosk is chosen), ``VoiceError`` (unreadable
    or longer than ``VOICE_MAX_SECONDS``), ``EmptyAudioError`` (shorter than
    ``VOICE_MIN_SECONDS``, silent, or nothing recognised),
    ``SpeechEngineUnavailableError``, ``TranscriptionTimeoutError`` and
    ``ExternalServiceError`` (online service unreachable).

    Returns ``{"text", "engine", "offline", "language", "duration_seconds"}``.
    The caller shows the text for confirmation and editing before it is
    submitted as a question. Neither the audio nor the text is stored.
    """
    org_id = _authorize_voice(ctx)
    settings = get_settings()
    chosen_language = _resolve_language(language, settings)
    return _process(ctx, org_id, audio_bytes, chosen_language, engine, "browser")


def _capture_server_audio(seconds: int) -> bytes:
    """Record from the server machine's default microphone and return WAV bytes."""
    try:
        microphone = sr.Microphone(sample_rate=VOSK_SAMPLE_RATE)
    except AttributeError as exc:
        raise MicrophoneError(_NO_PYAUDIO_HINT) from exc
    except OSError as exc:
        raise MicrophoneError(_WINDOWS_MIC_HINT) from exc
    recognizer = sr.Recognizer()
    try:
        with microphone as source:
            audio = recognizer.listen(source, timeout=seconds, phrase_time_limit=seconds)
    except sr.WaitTimeoutError as exc:
        raise EmptyAudioError("No speech was heard before the time ran out. Try again.") from exc
    except OSError as exc:
        raise MicrophoneError(_WINDOWS_MIC_HINT) from exc
    return audio.get_wav_data()


def record_from_server_microphone(
    ctx: AuthContext, *, seconds: int = 5, language: str | None = None, engine: str = "auto"
) -> dict[str, Any]:
    """Record up to ``seconds`` from the server machine's microphone, then transcribe.

    Optional local-machine capture that needs PyAudio; browser recording works
    without it. Same permission, plan feature and result as ``transcribe``.
    Raises ``MicrophoneError`` when PyAudio is missing or the device cannot be
    opened (with the Windows privacy-settings hint) and ``EmptyAudioError`` when
    nobody speaks before the time limit.
    """
    org_id = _authorize_voice(ctx)
    settings = get_settings()
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int)
        or not 1 <= seconds <= settings.voice_max_seconds
    ):
        raise ValidationError(
            f"Recording length must be a whole number of seconds from 1 to {settings.voice_max_seconds}."
        )
    chosen_language = _resolve_language(language, settings)
    _resolve_engine(engine, chosen_language, settings)  # fail before the microphone is opened
    wav_bytes = _capture_server_audio(seconds)
    return _process(ctx, org_id, wav_bytes, chosen_language, engine, "server_microphone")
