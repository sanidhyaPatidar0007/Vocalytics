"""Logging configuration with secret redaction."""

from __future__ import annotations

import logging
import re
import threading
from logging.handlers import RotatingFileHandler

from config.settings import Settings, get_settings

_SECRET_PATTERN = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|authorization)\b(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|\S+)"
)
_configured_for: str | None = None
_lock = threading.Lock()


def redact(text: str) -> str:
    return _SECRET_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)


class RedactingFilter(logging.Filter):
    """Masks values that look like credentials before a record is written."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - never let logging crash the app
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging once per log directory (idempotent)."""
    global _configured_for
    settings = settings or get_settings()
    key = str(settings.log_dir.resolve())
    with _lock:
        if _configured_for == key:
            return
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_vdas_handler", False):
                root.removeHandler(handler)
                handler.close()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler = RotatingFileHandler(
            settings.log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RedactingFilter())
        file_handler._vdas_handler = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)
        root.setLevel(settings.log_level)
        _configured_for = key
