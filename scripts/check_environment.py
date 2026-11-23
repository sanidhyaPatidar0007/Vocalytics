"""Environment diagnostics: Python, packages, memory, disk, folders, database,
Ollama, offline speech model, SMTP and microphone support.

Usage (PowerShell, from the project root):

    python scripts\\check_environment.py

Exit code 0 when no check FAILs (warnings are allowed), 1 otherwise. Secrets are
never printed.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GB = 1024**3

REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "sqlalchemy": "SQLAlchemy",
    "plotly": "plotly",
    "openpyxl": "openpyxl",
    "xlrd": "xlrd",
    "defusedxml": "defusedxml",
    "argon2": "argon2-cffi",
    "dotenv": "python-dotenv",
    "speech_recognition": "SpeechRecognition",
    "psutil": "psutil",
    "requests": "requests",
    "regex": "regex",
}
OPTIONAL_PACKAGES = {"vosk": "vosk (offline speech)", "pyaudio": "PyAudio (server-side microphone)"}


@dataclass
class Check:
    name: str
    status: str  # PASS, WARN, FAIL
    detail: str


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    requirements = ROOT / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;#]+)", line.strip())
            if match:
                pins[match.group(1).lower()] = match.group(2)
    return pins


def check_python() -> Check:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 10):
        return Check("Python", "FAIL", f"{text}; Python 3.10 or newer is required")
    return Check("Python", "PASS", text)


def check_packages() -> list[Check]:
    pins = _pins()
    results = []
    for module, dist in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
            installed = metadata.version(dist)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            results.append(
                Check(
                    f"Package {dist}",
                    "FAIL",
                    f"not importable ({type(exc).__name__}); run pip install -r requirements.txt",
                )
            )
            continue
        pinned = pins.get(dist.lower())
        if pinned and pinned != installed:
            results.append(Check(f"Package {dist}", "WARN", f"{installed} installed, {pinned} pinned"))
        else:
            results.append(Check(f"Package {dist}", "PASS", installed))
    for module, label in OPTIONAL_PACKAGES.items():
        try:
            importlib.import_module(module)
            results.append(Check(f"Optional {label}", "PASS", "installed"))
        except Exception:  # noqa: BLE001
            results.append(Check(f"Optional {label}", "WARN", "not installed (optional)"))
    return results


def check_memory_and_disk(settings) -> list[Check]:  # noqa: ANN001
    import psutil

    results = []
    memory = psutil.virtual_memory()
    status = "PASS" if memory.total >= 8 * GB else "WARN"
    results.append(
        Check(
            "Memory",
            status,
            f"{memory.total / GB:.1f} GB total, {memory.available / GB:.1f} GB available now",
        )
    )
    probe = settings.storage_root if settings.storage_root.exists() else ROOT
    free = shutil.disk_usage(probe).free
    reserve = settings.min_free_disk_mb / 1024
    status = "PASS" if free / GB > max(reserve * 2, 2) else "WARN"
    results.append(Check("Disk space", status, f"{free / GB:.1f} GB free where datasets are stored"))
    return results


def check_folders(settings) -> list[Check]:  # noqa: ANN001
    results = []
    for label, folder in (
        ("Data folder", settings.database_path.parent),
        ("Storage folder", settings.storage_root),
        ("Log folder", settings.log_dir),
    ):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            results.append(Check(label, "PASS", "writable"))
        except OSError as exc:
            results.append(Check(label, "FAIL", f"not writable ({exc.strerror})"))
    return results


def check_env_file() -> Check:
    if (ROOT / ".env").exists():
        return Check(".env file", "PASS", "found")
    return Check(".env file", "WARN", "not found; copy .env.example to .env (defaults are used meanwhile)")


def check_database(settings) -> Check:  # noqa: ANN001
    if not settings.database_path.exists():
        return Check("Database", "WARN", "not created yet; run python scripts\\initialize_database.py")
    from database import get_engine
    from database.migrations import MIGRATIONS, current_version

    version = current_version(get_engine())
    latest = MIGRATIONS[-1].version
    if version < latest:
        return Check(
            "Database", "WARN", f"schema v{version}, v{latest} available; run initialize_database.py"
        )
    return Check("Database", "PASS", f"schema v{version}")


def check_ollama(settings) -> Check:  # noqa: ANN001
    if settings.llm_provider == "gemini":
        if settings.gemini_api_key:
            return Check(
                "AI (Gemini)",
                "PASS",
                f"key set; model {settings.gemini_model}; only questions and column names and types are sent",
            )
        return Check(
            "AI (Gemini)",
            "WARN",
            "LLM_PROVIDER is gemini but GEMINI_API_KEY is empty; get a free key at "
            "https://aistudio.google.com/apikey. Analysis falls back to the rule-based interpreter.",
        )
    import requests

    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
    except Exception:  # noqa: BLE001 - any failure means unavailable
        return Check(
            "Ollama",
            "WARN",
            "not reachable; install from https://ollama.com and run 'ollama serve' "
            f"then 'ollama pull {settings.ollama_model}'. Analysis falls back to the rule-based interpreter.",
        )
    wanted = settings.ollama_model
    present = any(
        name == wanted or name.split(":")[0] == wanted.split(":")[0] and wanted.endswith(":latest")
        for name in models
    )
    present = present or wanted in models
    if present:
        return Check("Ollama", "PASS", f"running; model {wanted} installed")
    return Check(
        "Ollama",
        "WARN",
        f"running, but model {wanted} is missing; run 'ollama pull {wanted}' "
        f"(installed: {', '.join(models) or 'none'})",
    )


def check_vosk(settings) -> Check:  # noqa: ANN001
    path = settings.vosk_model_path
    try:
        importlib.import_module("vosk")
        package = True
    except Exception:  # noqa: BLE001
        package = False
    model = bool(path and path.exists() and any(path.iterdir()))
    if package and model:
        return Check("Offline speech (Vosk)", "PASS", f"model folder found ({settings.vosk_model_language})")
    hint = "pip install -r requirements-optional.txt and unpack a Vosk model into models\\vosk"
    return Check(
        "Offline speech (Vosk)",
        "WARN",
        f"{'package missing' if not package else 'model folder missing'}; {hint}",
    )


def check_smtp(settings) -> Check:  # noqa: ANN001
    if settings.smtp_configured:
        security = (
            "SSL" if settings.smtp_use_ssl else ("STARTTLS" if settings.smtp_use_tls else "no encryption")
        )
        return Check("SMTP", "PASS", f"{settings.smtp_host}:{settings.smtp_port} ({security})")
    return Check(
        "SMTP", "WARN", "not configured; anomaly e-mails will be logged as skipped (set SMTP_* in .env)"
    )


def check_microphone() -> Check:
    try:
        pyaudio = importlib.import_module("pyaudio")
    except Exception:  # noqa: BLE001
        return Check(
            "Microphone",
            "WARN",
            "recording happens in the browser; PyAudio is only needed for server-side capture",
        )
    audio = pyaudio.PyAudio()
    try:
        inputs = [audio.get_device_info_by_index(i) for i in range(audio.get_device_count())]
        count = sum(1 for d in inputs if int(d.get("maxInputChannels", 0)) > 0)
    finally:
        audio.terminate()
    if count:
        return Check("Microphone", "PASS", f"{count} input device(s) visible")
    return Check(
        "Microphone",
        "WARN",
        "no input device visible; check Windows Settings > Privacy & security > Microphone",
    )


def run_checks() -> list[Check]:
    from config.settings import get_settings

    checks = [check_python(), *check_packages()]
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("Configuration", "FAIL", getattr(exc, "user_message", str(exc))))
        return checks
    checks.append(Check("Configuration", "PASS", f"environment '{settings.environment}'"))
    checks.append(check_env_file())
    checks.extend(check_memory_and_disk(settings))
    checks.extend(check_folders(settings))
    checks.append(check_database(settings))
    checks.append(check_ollama(settings))
    checks.append(check_vosk(settings))
    checks.append(check_smtp(settings))
    checks.append(check_microphone())
    return checks


def main() -> int:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    checks = run_checks()
    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"[{check.status:4}] {check.name.ljust(width)}  {check.detail}")
    failures = [c for c in checks if c.status == "FAIL"]
    print(
        f"\n{len(checks)} checks: {len(failures)} failed, {sum(c.status == 'WARN' for c in checks)} warnings."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
