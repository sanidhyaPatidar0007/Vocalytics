"""Run the scheduled anomaly checks once (worker for Windows Task Scheduler).

Every active organization whose plan includes anomaly detection has its active,
due rules checked; new findings are stored and e-mailed (when the plan and the
SMTP settings allow it). Manual rules are never run here.

Register it to run every hour (PowerShell, from any folder; replace <path>):

    schtasks /Create /SC HOURLY /TN "VoiceDataAnalyst Alerts" /TR "<path>\\.venv\\Scripts\\python.exe <path>\\scripts\\run_anomaly_checks.py"

Run it by hand from the project root:

    python scripts\\run_anomaly_checks.py

Remove the task with: schtasks /Delete /TN "VoiceDataAnalyst Alerts" /F

It also retries deleting purged version files that were still open when they were purged.

Exit code 0 means the worker completed (individual rule failures are counted in the
summary line and logged); exit code 1 means the worker itself could not run.
"""  # noqa: E501

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from core.logging_setup import configure_logging  # noqa: E402
from database import init_database  # noqa: E402
from services import anomaly_service, versioning_service  # noqa: E402

logger = logging.getLogger("scripts.run_anomaly_checks")


def _reclaim_storage() -> None:
    """Retry deleting purged version files that were locked when purged. Logs only, never fails the run."""
    try:
        result = versioning_service.reclaim_purged_files()
    except Exception:  # noqa: BLE001 - storage upkeep must not change the worker's exit code
        logger.exception("Reclaiming purged version files failed")
        return
    if result["reclaimed"] or result["still_locked"]:
        logger.info(
            "Purged version files: %s reclaimed, %s still locked", result["reclaimed"], result["still_locked"]
        )


def main(argv: list[str] | None = None) -> int:
    """Run the checks once and print a one-line summary."""
    parser = argparse.ArgumentParser(description="Run the scheduled anomaly checks once and print a summary.")
    parser.parse_args(argv)
    try:
        configure_logging(get_settings())
        init_database(seed=False)
        summary = anomaly_service.run_scheduled_checks()
    except Exception as exc:  # noqa: BLE001 - the exit code reports any worker failure
        logger.exception("Anomaly check run failed")
        print(
            f"Anomaly checks failed: {type(exc).__name__}. See the application log for details.",
            file=sys.stderr,
        )
        return 1
    logger.info("Anomaly check run finished: %s", summary)
    _reclaim_storage()
    print(
        "Anomaly checks finished: "
        f"{summary['rules_run']} rule(s) run, {summary['rules_failed']} failed, "
        f"{summary['rules_not_due']} not due, {summary['new_events']} new event(s), "
        f"{summary['duplicate_suppressed']} duplicate(s) suppressed, "
        f"{summary['cooldown_suppressed']} in cooldown, {summary['notifications_sent']} e-mail(s) sent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
