"""Application settings loaded from environment variables (and an optional .env file).

Every tunable limit lives here so it is never hardcoded across the code base.
Plan-specific limits (upload size, storage, users...) live in the
``subscription_plans`` table; the values below only seed the default plans
and define global safety ceilings.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from dotenv import load_dotenv

from core.errors import ConfigurationError

BASE_DIR = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _raw(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_str(name: str, default: str | None) -> str | None:
    value = _raw(name)
    return default if value is None else value


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = _raw(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable {name} must be an integer.") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"Environment variable {name} must be >= {minimum}.")
    return parsed


def _env_float(
    name: str, default: float, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    value = _raw(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable {name} must be a number.") from exc
    if minimum is not None and parsed < minimum:
        raise ConfigurationError(f"Environment variable {name} must be >= {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ConfigurationError(f"Environment variable {name} must be <= {maximum}.")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = _raw(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigurationError(f"Environment variable {name} must be true or false.")


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    # Items are separated by "|" so values may contain commas or slashes.
    return tuple(item.strip() for item in value.split("|") if item.strip())


def _env_path(name: str, default: Path | None) -> Path | None:
    value = _raw(name)
    if value is None:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    app_name: str = "Voice-Enabled Data Analyst"
    environment: str = "development"
    base_dir: Path = BASE_DIR
    database_path: Path = BASE_DIR / "data" / "app.db"
    backup_dir: Path = BASE_DIR / "data" / "backups"
    storage_root: Path = BASE_DIR / "storage"
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = "INFO"

    # Authentication and sessions
    session_idle_timeout_minutes: int = 30
    session_absolute_timeout_hours: int = 12
    login_max_failed_attempts: int = 5
    login_attempt_window_minutes: int = 15
    login_lockout_minutes: int = 15
    password_min_length: int = 10
    # Argon2id cost parameters (argon2-cffi defaults = RFC 9106 low-memory profile)
    password_hash_time_cost: int = 3
    password_hash_memory_kib: int = 65536
    password_hash_parallelism: int = 4

    # Default plan limits (seed values for the Basic and Premium plans)
    basic_max_upload_mb: int = 50
    premium_max_upload_mb: int = 500
    # Global ceilings and processing targets (engineering starting points)
    absolute_upload_ceiling_mb: int = 1024
    csv_processing_target_mb: int = 250
    xlsx_processing_target_mb: int = 100
    append_merge_input_target_mb: int = 100
    preview_max_rows: int = 500
    result_max_rows: int = 500
    csv_chunk_rows: int = 50_000
    parquet_batch_rows: int = 50_000
    memory_safety_fraction: float = 0.5
    min_free_disk_mb: int = 500
    excel_max_uncompressed_mb: int = 1024
    excel_max_sheets: int = 200
    operation_timeout_seconds: int = 900
    merge_max_output_rows: int = 20_000_000
    # Widest dataset that import, append and merge may create. Measured on the
    # reference machine: 1,000 columns import in about 9 s and profile in under a
    # minute; width cost is superlinear, so wider files are refused up front.
    max_columns: int = 1000

    # Data typing defaults
    default_missing_placeholders: tuple[str, ...] = (
        "NA",
        "N/A",
        "null",
        "NULL",
        "None",
        "NaN",
        "nan",
        "#N/A",
    )
    default_date_dayfirst: bool = False

    # Local AI (Ollama)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3:8b"
    ollama_fallback_model: str | None = "llama3.2:3b"
    ollama_timeout_seconds: int = 120
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.0
    llm_max_concurrency: int = 1
    llm_queue_wait_seconds: int = 5
    ai_fallback_mode: str = "rules"  # "rules" = labelled rule-based interpreter, "off" = error only
    ai_narrative_enabled: bool = True
    # Optional cloud AI instead of Ollama. Gemini receives only the question and the column
    # names and types (never data values or results), so it also writes no narrative.
    llm_provider: str = "ollama"  # "ollama" (local, default) or "gemini"
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_seconds: int = 60

    # Voice input
    vosk_model_path: Path | None = BASE_DIR / "models" / "vosk"
    vosk_model_language: str = "en-US"
    voice_online_recognition_enabled: bool = False
    voice_default_language: str = "en-US"
    voice_supported_languages: tuple[str, ...] = ("en-US", "en-IN", "en-GB", "hi-IN")
    voice_max_seconds: int = 60
    voice_min_seconds: float = 0.4
    voice_transcription_timeout_seconds: int = 30

    # E-mail (SMTP). Credentials only come from the environment.
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = field(default=None, repr=False)
    smtp_from: str | None = None
    smtp_use_tls: bool = True
    smtp_use_ssl: bool = False
    smtp_timeout_seconds: int = 20

    # Alerts
    alert_default_cooldown_minutes: int = 60

    @property
    def is_test(self) -> bool:
        return self.environment == "test"

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_from)

    def with_overrides(self, **changes: object) -> Settings:
        """Return a copy with selected fields replaced (used by tests and scripts)."""
        valid = {f.name for f in fields(self)}
        unknown = set(changes) - valid
        if unknown:
            raise ConfigurationError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> Settings:
        if load_env_file:
            load_dotenv(BASE_DIR / ".env", override=False)
        d = cls()
        settings = cls(
            app_name=_env_str("APP_NAME", d.app_name) or d.app_name,
            environment=(_env_str("APP_ENV", d.environment) or d.environment).lower(),
            database_path=_env_path("DATABASE_PATH", d.database_path) or d.database_path,
            backup_dir=_env_path("BACKUP_DIR", d.backup_dir) or d.backup_dir,
            storage_root=_env_path("STORAGE_ROOT", d.storage_root) or d.storage_root,
            log_dir=_env_path("LOG_DIR", d.log_dir) or d.log_dir,
            log_level=(_env_str("LOG_LEVEL", d.log_level) or d.log_level).upper(),
            session_idle_timeout_minutes=_env_int(
                "SESSION_IDLE_TIMEOUT_MINUTES", d.session_idle_timeout_minutes, minimum=1
            ),
            session_absolute_timeout_hours=_env_int(
                "SESSION_ABSOLUTE_TIMEOUT_HOURS", d.session_absolute_timeout_hours, minimum=1
            ),
            login_max_failed_attempts=_env_int(
                "LOGIN_MAX_FAILED_ATTEMPTS", d.login_max_failed_attempts, minimum=1
            ),
            login_attempt_window_minutes=_env_int(
                "LOGIN_ATTEMPT_WINDOW_MINUTES", d.login_attempt_window_minutes, minimum=1
            ),
            login_lockout_minutes=_env_int("LOGIN_LOCKOUT_MINUTES", d.login_lockout_minutes, minimum=1),
            password_min_length=_env_int("PASSWORD_MIN_LENGTH", d.password_min_length, minimum=8),
            password_hash_time_cost=_env_int("PASSWORD_HASH_TIME_COST", d.password_hash_time_cost, minimum=1),
            password_hash_memory_kib=_env_int(
                "PASSWORD_HASH_MEMORY_KIB", d.password_hash_memory_kib, minimum=8192
            ),
            password_hash_parallelism=_env_int(
                "PASSWORD_HASH_PARALLELISM", d.password_hash_parallelism, minimum=1
            ),
            basic_max_upload_mb=_env_int("BASIC_MAX_UPLOAD_MB", d.basic_max_upload_mb, minimum=1),
            premium_max_upload_mb=_env_int("PREMIUM_MAX_UPLOAD_MB", d.premium_max_upload_mb, minimum=1),
            absolute_upload_ceiling_mb=_env_int(
                "ABSOLUTE_UPLOAD_CEILING_MB", d.absolute_upload_ceiling_mb, minimum=1
            ),
            csv_processing_target_mb=_env_int(
                "CSV_PROCESSING_TARGET_MB", d.csv_processing_target_mb, minimum=1
            ),
            xlsx_processing_target_mb=_env_int(
                "XLSX_PROCESSING_TARGET_MB", d.xlsx_processing_target_mb, minimum=1
            ),
            append_merge_input_target_mb=_env_int(
                "APPEND_MERGE_INPUT_TARGET_MB", d.append_merge_input_target_mb, minimum=1
            ),
            preview_max_rows=_env_int("PREVIEW_MAX_ROWS", d.preview_max_rows, minimum=1),
            result_max_rows=_env_int("RESULT_MAX_ROWS", d.result_max_rows, minimum=1),
            csv_chunk_rows=_env_int("CSV_CHUNK_ROWS", d.csv_chunk_rows, minimum=1),
            parquet_batch_rows=_env_int("PARQUET_BATCH_ROWS", d.parquet_batch_rows, minimum=1),
            memory_safety_fraction=_env_float(
                "MEMORY_SAFETY_FRACTION", d.memory_safety_fraction, minimum=0.05, maximum=0.95
            ),
            min_free_disk_mb=_env_int("MIN_FREE_DISK_MB", d.min_free_disk_mb, minimum=0),
            excel_max_uncompressed_mb=_env_int(
                "EXCEL_MAX_UNCOMPRESSED_MB", d.excel_max_uncompressed_mb, minimum=1
            ),
            excel_max_sheets=_env_int("EXCEL_MAX_SHEETS", d.excel_max_sheets, minimum=1),
            operation_timeout_seconds=_env_int(
                "OPERATION_TIMEOUT_SECONDS", d.operation_timeout_seconds, minimum=1
            ),
            merge_max_output_rows=_env_int("MERGE_MAX_OUTPUT_ROWS", d.merge_max_output_rows, minimum=1),
            max_columns=_env_int("MAX_COLUMNS", d.max_columns, minimum=1),
            default_missing_placeholders=_env_list(
                "DEFAULT_MISSING_PLACEHOLDERS", d.default_missing_placeholders
            ),
            default_date_dayfirst=_env_bool("DEFAULT_DATE_DAYFIRST", d.default_date_dayfirst),
            ollama_base_url=(_env_str("OLLAMA_BASE_URL", d.ollama_base_url) or d.ollama_base_url).rstrip("/"),
            ollama_model=_env_str("OLLAMA_MODEL", d.ollama_model) or d.ollama_model,
            ollama_fallback_model=_env_str("OLLAMA_FALLBACK_MODEL", d.ollama_fallback_model),
            ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", d.ollama_timeout_seconds, minimum=1),
            ollama_num_ctx=_env_int("OLLAMA_NUM_CTX", d.ollama_num_ctx, minimum=512),
            ollama_temperature=_env_float(
                "OLLAMA_TEMPERATURE", d.ollama_temperature, minimum=0.0, maximum=2.0
            ),
            llm_max_concurrency=_env_int("LLM_MAX_CONCURRENCY", d.llm_max_concurrency, minimum=1),
            llm_queue_wait_seconds=_env_int("LLM_QUEUE_WAIT_SECONDS", d.llm_queue_wait_seconds, minimum=0),
            ai_fallback_mode=(_env_str("AI_FALLBACK_MODE", d.ai_fallback_mode) or d.ai_fallback_mode).lower(),
            ai_narrative_enabled=_env_bool("AI_NARRATIVE_ENABLED", d.ai_narrative_enabled),
            llm_provider=(_env_str("LLM_PROVIDER", d.llm_provider) or d.llm_provider).lower(),
            gemini_api_key=_env_str("GEMINI_API_KEY", None),
            gemini_model=_env_str("GEMINI_MODEL", d.gemini_model) or d.gemini_model,
            gemini_base_url=(_env_str("GEMINI_BASE_URL", d.gemini_base_url) or d.gemini_base_url).rstrip("/"),
            gemini_timeout_seconds=_env_int("GEMINI_TIMEOUT_SECONDS", d.gemini_timeout_seconds, minimum=1),
            vosk_model_path=_env_path("VOSK_MODEL_PATH", d.vosk_model_path),
            vosk_model_language=_env_str("VOSK_MODEL_LANGUAGE", d.vosk_model_language)
            or d.vosk_model_language,
            voice_online_recognition_enabled=_env_bool(
                "VOICE_ONLINE_RECOGNITION_ENABLED", d.voice_online_recognition_enabled
            ),
            voice_default_language=_env_str("VOICE_DEFAULT_LANGUAGE", d.voice_default_language)
            or d.voice_default_language,
            voice_supported_languages=_env_list("VOICE_SUPPORTED_LANGUAGES", d.voice_supported_languages),
            voice_max_seconds=_env_int("VOICE_MAX_SECONDS", d.voice_max_seconds, minimum=1),
            voice_min_seconds=_env_float("VOICE_MIN_SECONDS", d.voice_min_seconds, minimum=0.0),
            voice_transcription_timeout_seconds=_env_int(
                "VOICE_TRANSCRIPTION_TIMEOUT_SECONDS", d.voice_transcription_timeout_seconds, minimum=1
            ),
            smtp_host=_env_str("SMTP_HOST", None),
            smtp_port=_env_int("SMTP_PORT", d.smtp_port, minimum=1),
            smtp_username=_env_str("SMTP_USERNAME", None),
            smtp_password=_env_str("SMTP_PASSWORD", None),
            smtp_from=_env_str("SMTP_FROM", None),
            smtp_use_tls=_env_bool("SMTP_USE_TLS", d.smtp_use_tls),
            smtp_use_ssl=_env_bool("SMTP_USE_SSL", d.smtp_use_ssl),
            smtp_timeout_seconds=_env_int("SMTP_TIMEOUT_SECONDS", d.smtp_timeout_seconds, minimum=1),
            alert_default_cooldown_minutes=_env_int(
                "ALERT_DEFAULT_COOLDOWN_MINUTES", d.alert_default_cooldown_minutes, minimum=0
            ),
        )
        settings.validate()
        return settings

    @classmethod
    def for_testing(cls, root: Path, **overrides: object) -> Settings:
        """Settings that keep every file inside ``root`` (a pytest tmp_path)."""
        base = cls(
            environment="test",
            database_path=root / "data" / "test.db",
            backup_dir=root / "data" / "backups",
            storage_root=root / "storage",
            log_dir=root / "logs",
            log_level="WARNING",
            # Cheap hashing keeps the suite fast; verification logic is identical.
            password_hash_time_cost=1,
            password_hash_memory_kib=1024,
            password_hash_parallelism=1,
            # Unused local port so an accidental real LLM call fails fast.
            ollama_base_url="http://127.0.0.1:9",
            ollama_timeout_seconds=2,
            llm_provider="ollama",
            gemini_api_key=None,
            gemini_base_url="http://127.0.0.1:9",
            gemini_timeout_seconds=2,
            llm_queue_wait_seconds=0,
            vosk_model_path=root / "models" / "vosk",
            smtp_host=None,
            smtp_from=None,
        )
        result = base.with_overrides(**overrides) if overrides else base
        result.validate()
        return result

    def validate(self) -> None:
        if self.ai_fallback_mode not in {"rules", "off"}:
            raise ConfigurationError("AI_FALLBACK_MODE must be 'rules' or 'off'.")
        if self.llm_provider not in {"ollama", "gemini"}:
            raise ConfigurationError("LLM_PROVIDER must be 'ollama' or 'gemini'.")
        if self.smtp_use_tls and self.smtp_use_ssl:
            raise ConfigurationError("SMTP_USE_TLS and SMTP_USE_SSL cannot both be true.")
        if self.voice_default_language not in self.voice_supported_languages:
            raise ConfigurationError("VOICE_DEFAULT_LANGUAGE must be one of VOICE_SUPPORTED_LANGUAGES.")
        if self.basic_max_upload_mb > self.absolute_upload_ceiling_mb or (
            self.premium_max_upload_mb > self.absolute_upload_ceiling_mb
        ):
            raise ConfigurationError("Plan upload limits cannot exceed ABSOLUTE_UPLOAD_CEILING_MB.")


_lock = threading.Lock()
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the active settings (loaded from the environment on first use)."""
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                _settings = Settings.from_env()
    return _settings


def set_settings(settings: Settings) -> None:
    """Replace the active settings (tests and scripts only)."""
    global _settings
    with _lock:
        _settings = settings


def reset_settings() -> None:
    global _settings
    with _lock:
        _settings = None
