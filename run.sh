#!/usr/bin/env bash
# Voice-Enabled Data Analyst: one-step setup and start for macOS and Linux
# (Windows: use run.bat). Run "./run.sh", or "bash run.sh"; "./run.sh setup" only sets up.
# The first run needs internet (to download packages) and takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")"

venv_python() {
  if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo .venv/Scripts/python.exe; fi
}

find_python() {
  for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
      "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if [ ! -d .venv ]; then
  echo
  echo "=== First-time setup: creating a private Python environment in .venv ==="
  if ! PYTHON=$(find_python); then
    echo "Python 3.10, 3.11, 3.12 or 3.13 was not found (3.14 is not supported yet)."
    echo "Install Python 3.13 (https://www.python.org/downloads/, or 'brew install python@3.13'), then run ./run.sh again."
    exit 1
  fi
  echo "Using $PYTHON"
  "$PYTHON" -m venv .venv
fi
VENV_PY=$(venv_python)

if [ ! -f .venv/installed.txt ]; then
  echo
  echo "=== Installing packages (a few minutes, only the first time) ==="
  "$VENV_PY" -m pip install --disable-pip-version-check --quiet --upgrade pip
  "$VENV_PY" -m pip install --disable-pip-version-check -r requirements.txt
  echo installed > .venv/installed.txt
fi

[ -f .env ] || cp .env.example .env
"$VENV_PY" scripts/initialize_database.py

if [ ! -f data/.setup-complete ]; then
  echo
  echo "=== Create the Super Admin account (the platform owner who signs in first) ==="
  if [ -n "${SUPER_ADMIN_PASSWORD:-}" ]; then
    "$VENV_PY" scripts/create_super_admin.py --if-missing --from-env
  else
    "$VENV_PY" scripts/create_super_admin.py --if-missing
  fi
  echo
  DEMO=y
  read -r -p "Add a demo company with sample data so you can try the app right away? [Y/n] " DEMO || DEMO=n
  case "${DEMO:-y}" in
    [nN]*) ;;
    *)
      "$VENV_PY" scripts/seed_demo_data.py
      echo "Write down the two demo passwords above. They are shown only once."
      ;;
  esac
  echo done > data/.setup-complete
fi

[ "${1:-}" = "setup" ] && exit 0

echo
echo "=== Starting the app at http://localhost:8501 ==="
echo "Your browser opens in a few seconds. Press Ctrl+C here to stop the app."
(
  sleep 6
  if command -v open >/dev/null 2>&1; then open http://localhost:8501
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open http://localhost:8501
  fi
) >/dev/null 2>&1 &
exec "$VENV_PY" -m streamlit run app.py --server.port 8501
