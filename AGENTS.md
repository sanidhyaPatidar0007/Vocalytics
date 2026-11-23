# Guide for coding agents (Antigravity, Codex, Claude Code and others)

Voice-Enabled Data Analyst: a multi-tenant Streamlit app (SQLite, SQLAlchemy, pandas, Plotly).
Features and settings: `README.md`. Code contracts: `docs/ARCHITECTURE.md`. Requirements:
`docs/SPECIFICATION.md`. What was verified: `docs/VERIFICATION_REPORT.md`.

## Set up and start (the usual request)

* Windows: `run.bat` (in PowerShell: `.\run.bat`). macOS / Linux: `./run.sh`.
* The first run is interactive. It asks for a Super Admin username, e-mail and password (hidden
  input) and whether to add demo data. Either let the user answer in the terminal, or ask the user
  for the three values, set `SUPER_ADMIN_USERNAME`, `SUPER_ADMIN_EMAIL`, `SUPER_ADMIN_PASSWORD`
  and `VDAS_NO_PAUSE=1` in the environment, run `run.bat setup` (or `./run.sh setup`), and answer
  the demo question with `y` or `n` on standard input. Never invent a password, and never write
  one to a file or into the code.
* Then start the server with `run.bat` / `./run.sh` in its own terminal (it keeps running) and
  open http://localhost:8501.
* Manual steps, if a launcher fails: create `.venv` with Python 3.10 to 3.13, run
  `python -m pip install -r requirements.txt`, copy `.env.example` to `.env`, then run
  `python scripts/initialize_database.py`, `python scripts/create_super_admin.py`, optionally
  `python scripts/seed_demo_data.py`, and finally `python -m streamlit run app.py`.
* Python 3.14 is not supported (numpy 2.2.6 has no 3.14 wheels). Keep the pinned versions in
  `requirements.txt`; `regex==2026.9.29` is the newest release with Python 3.10 wheels.

## Test

* Full suite (about 1,900 tests, roughly 12 minutes): `.venv\Scripts\python -m pytest -q` on
  Windows, `.venv/bin/python -m pytest -q` elsewhere. UI tests only: `pytest tests/ui -q`.
* Lint: `pip install -r requirements-dev.txt`, then `ruff check .`.
* Tests use a temporary database and storage folder; they never touch `data/` or `storage/`.

## Rules for changes

* Never commit `.env`, `data/`, `storage/`, `logs/`, `.venv/` or `dist/` (all git-ignored).
* Layering: `views/` and `components/` never query the database; every service takes the
  `AuthContext` first and authorizes before touching tenant data. Records of another organization
  must behave as "not found".
* Escape untrusted text (dataset values, column names, user input, model output) with
  `components.feedback.md_escape` before Markdown and `html.escape` before HTML.
* The language model only proposes a JSON plan that an allowlist validates; never `eval` or
  `exec` model output.
* Line endings are LF (`run.bat` is CRLF); `.gitattributes` enforces this.
* Run the tests for what you changed, and the full suite before calling work done. Report
  failures as they are.

## Optional external software

* Ollama (local language model) is optional: without it the app answers with a clearly labelled
  rule-based interpreter (`AI_FALLBACK_MODE=rules`). Do not install Ollama or pull models (a
  multi-gigabyte download) unless the user asks.
* Gemini instead of Ollama: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env` (README, "Cloud AI
  with Gemini"). Put the key only in `.env`, never in code, docs or commits. The Gemini path is
  schema only by design (question plus column names and types; no values, no results, no
  narrative); keep it that way.
* Offline speech (Vosk), a microphone and SMTP e-mail are optional; see `README.md`.
