# Voice-Enabled Data Analyst SaaS

A multi-tenant, local-first web application for business data. Organizations upload CSV and
Excel files, see exactly what is wrong with them, clean them safely (every change is a new,
restorable version), combine datasets, and ask questions in plain language, by typing or by
voice. Answers are calculated with pandas on the real data; a local Llama model (through
Ollama) only translates the question into a validated analysis plan and explains the result.
By default nothing is sent to cloud AI services. An optional Gemini mode sends only the question
and the column names and types (see "Cloud AI with Gemini").

**Fastest start:** double-click `run.bat` (Windows) or run `./run.sh` (macOS / Linux). It sets
everything up on the first run and opens the app. Step by step: `START_HERE.md`. Coding agents
such as Antigravity: `AGENTS.md`.

* Requirements: `docs/SPECIFICATION.md`
* Architecture and service contracts: `docs/ARCHITECTURE.md`
* Requirement-to-test checklist: `docs/TRACEABILITY.md`
* What was verified for this release, and what needs external software: `docs/VERIFICATION_REPORT.md`

## What each role can do

| Role | Highlights |
|---|---|
| Super Admin | Create, edit, suspend and reactivate organizations (data is always kept); manage Basic and Premium plans, prices, limits and per-plan features; assign plans and see subscription history; platform metrics (organizations, users, datasets, analysis jobs, storage, errors, latency, usage); audit log and error events. |
| Organization Admin | Organization dashboard; create, edit, activate and deactivate analysts; analyst permissions and per-dataset access; uploads with sheet selection; profiling and quality scores; missing values, duplicates, cleaning, append and merge; versions, history and restore; analysis history and usage; anomaly rules, alerts and recipients; exports; organization settings. |
| Data Analyst | Datasets shared with them; quality reports; questions by text and (when permitted) voice; answers with tables and charts; history, saved reports and downloads (when permitted); cleaning when the admin grants it. |

Plan features (stored in the database, editable by the Super Admin):

* Basic: CSV/XLSX upload, profiling, missing-value treatment, duplicate detection, standard cleaning, text AI analysis, standard charts.
* Premium: everything in Basic plus voice analysis, advanced cleaning, append and merge, advanced analytics, anomaly detection, e-mail alerts and saved reports, with higher limits.

Every feature, permission and organization boundary is enforced in the service layer, not only by
hiding pages.

## Quick start (Windows 10/11, PowerShell)

`run.bat` does all of the steps below automatically. To run them by hand:

Requires Python 3.10, 3.11, 3.12 or 3.13 (tested on 3.13 and 3.10; 3.14 is not supported yet
because numpy 2.2.6 has no wheels for it).

```powershell
cd path\to\voice-data-analyst-saas
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # see Troubleshooting if scripts are blocked
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env           # optional: every setting has a safe default
python scripts\check_environment.py   # diagnostics: Python, packages, RAM, disk, Ollama, SMTP...
python scripts\initialize_database.py # creates data\app.db and the Basic/Premium plans
python scripts\create_super_admin.py  # interactive; the password is typed twice, never shown
pytest -q                             # optional: run the automated test suite
streamlit run app.py                  # opens http://localhost:8501
```

Optional demo data (a Premium organization with an admin, an analyst and three sample datasets
that contain typical quality problems; the two passwords are generated and printed once):

```powershell
python scripts\seed_demo_data.py
```

Sign in as the Super Admin to create organizations and their first Organization Admin, or use
the demo accounts printed by the seed script.

## Configuration

All settings come from environment variables or the `.env` file (see `.env.example`, which lists
every variable with its default). The most important ones:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_PATH`, `STORAGE_ROOT`, `LOG_DIR` | `data/app.db`, `storage`, `logs` | Where data lives (relative paths are inside the project folder). |
| `SESSION_IDLE_TIMEOUT_MINUTES` / `SESSION_ABSOLUTE_TIMEOUT_HOURS` | 30 / 12 | Session expiry. |
| `LOGIN_MAX_FAILED_ATTEMPTS`, `LOGIN_LOCKOUT_MINUTES` | 5, 15 | Brute-force protection. |
| `BASIC_MAX_UPLOAD_MB`, `PREMIUM_MAX_UPLOAD_MB` | 50, 500 | Seed values for the default plans (edit plans later in the Super Admin panel). |
| `ABSOLUTE_UPLOAD_CEILING_MB` | 1024 | Hard ceiling over every plan. Also raise `server.maxUploadSize` in `.streamlit/config.toml` if you change it. |
| `CSV_PROCESSING_TARGET_MB`, `XLSX_PROCESSING_TARGET_MB`, `APPEND_MERGE_INPUT_TARGET_MB` | 250, 100, 100 | Above these sizes the memory check becomes strict. |
| `MEMORY_SAFETY_FRACTION` | 0.5 | Share of the RAM available right now that one operation may use. |
| `PREVIEW_MAX_ROWS`, `RESULT_MAX_ROWS` | 500, 500 | Preview and AI result table limits. |
| `MAX_COLUMNS` | 1000 | Widest dataset that an upload or a merge may create. |
| `OPERATION_TIMEOUT_SECONDS` | 900 | Time limit for one long operation (import, cleaning, merge, checks). |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_FALLBACK_MODEL` | `http://127.0.0.1:11434`, `llama3:8b`, `llama3.2:3b` | Local AI. |
| `OLLAMA_NUM_CTX`, `LLM_MAX_CONCURRENCY` | 4096, 1 | Context window and simultaneous AI requests. |
| `LLM_PROVIDER`, `GEMINI_API_KEY`, `GEMINI_MODEL` | `ollama`, empty, `gemini-3.1-flash-lite` | `gemini` uses Google's API, schema only (see "Cloud AI with Gemini"). |
| `AI_FALLBACK_MODE` | `rules` | When Ollama is unavailable: `rules` answers common questions with a clearly labelled rule-based interpreter; `off` shows an error instead. |
| `VOSK_MODEL_PATH`, `VOICE_ONLINE_RECOGNITION_ENABLED` | `models/vosk`, `false` | Offline speech model folder; online recognition is off by default. |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_USE_TLS`, `SMTP_USE_SSL` | empty | E-mail alerts. Leave `SMTP_HOST` empty to disable e-mail. |

Secrets are only read from the environment; none are stored in the code or the database.

## Local AI with Ollama

1. Install Ollama for Windows from https://ollama.com/download and start it (it runs in the
   system tray and listens on `http://127.0.0.1:11434`).
2. Pull a model:
   ```powershell
   ollama pull llama3:8b      # about 4.7 GB, 4-bit quantized; needs roughly 6 GB of free RAM
   ollama pull llama3.2:3b    # about 2 GB; the smaller alternative for busy 16 GB machines
   ```
3. Check it: `ollama list`, or run `python scripts\check_environment.py` (the "Ollama" line), or
   open "Ask the analyst" in the app, which shows whether the local model is running.
4. Choose the model with `OLLAMA_MODEL` (and `OLLAMA_FALLBACK_MODEL`, used automatically when the
   main model is not installed).

How the AI is used safely:

* The model sees only the question, the column names and types, and a few short sample values.
  It never receives the dataset.
* It must answer with a JSON analysis plan. The plan is validated against an allowlist of
  operations (filter, group, aggregate, top-N, time series, growth, describe, correlation, missing
  values, duplicates, distribution, anomalies), existing columns and type rules. Nothing it
  returns is ever executed as code.
* Numbers are computed with pandas and checked again through an independent calculation. An
  AI-written narrative is shown only if every number in it matches the computed results. Time
  results are checked against every calendar period recomputed independently (including which
  periods a sort and limit keep), and growth against the previous calendar period.
* A sum over rows that have no values is shown as not available, never as 0. In time series a
  period with no rows counts as 0, and a period whose rows have no values is not available.
* On timestamp columns, a filter date without a time means the whole day for "equals", "not
  equals", "on or before", "after" and the end of a "between" range.
* When several groups tie for first place the answer names them (up to 5) and adds a warning, also
  when a top-N limit cuts through a tie.
* If Ollama is down, the question is ambiguous, or the reply is malformed, you get a clear
  message, a clarification question, or (with `AI_FALLBACK_MODE=rules`) an answer from the
  rule-based interpreter that is clearly labelled as such. Results are never invented.

Troubleshooting: "Local AI model unavailable" means Ollama is not running or the model is not
pulled (`ollama pull <model>`). Slow first answers are normal while the model loads into memory.
If memory is tight, switch to `llama3.2:3b`. Keep `LLM_MAX_CONCURRENCY=1` on 16 GB machines.

## Cloud AI with Gemini (optional)

No Ollama, or a laptop without the RAM for it? Use Google's Gemini API instead:

1. Create a free API key at https://aistudio.google.com/apikey.
2. In `.env` set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=<your key>` (the key stays in `.env`,
   which is never committed).
3. Restart the app. "Ask the analyst" shows "Gemini is set up".

What leaves the computer ("schema only"):

* Sent to Google: each question, word for word (spoken questions are transcribed on this computer
  first and then sent as text), and the column names and types of the dataset.
* Never sent: data values, sample values, minimums and maximums, row counts and computed results.
  Answers are still calculated here and checked independently. Because results are not sent,
  Gemini writes no narrative; the built-in summary is shown instead.
* Column names can be revealing (for example `patient_diagnosis`), and so can names typed into a
  question ("sales for Acme Ltd").
* Free key: Google's terms for unpaid use let it use what you send to improve its products, and
  people may read it. With billing enabled on the key's project, Google does not use it for that
  but keeps logs for a limited time to detect abuse. For confidential data use Ollama, which keeps
  everything on this computer.

`GEMINI_MODEL` (default `gemini-3.1-flash-lite`, which has a free tier) picks the model. When Gemini
is unreachable, the key is wrong or the free limit is reached, the app says so and answers common
questions with the rule-based interpreter.

## Voice questions

* Recording happens in the browser (the microphone button on "Ask the analyst"). Allow microphone
  access when the browser asks. On Windows also check Settings > Privacy & security > Microphone
  ("Let apps access your microphone" and "Let desktop apps access your microphone").
* The recognized text is shown for you to edit, and only the confirmed text is sent to the same
  analysis pipeline as typed questions.
* Offline recognition (recommended, nothing leaves the machine):
  ```powershell
  pip install -r requirements-optional.txt
  # download a model, for example vosk-model-small-en-us-0.15, from https://alphacephei.com/vosk/models
  # and unzip it so that models\vosk contains the model files (am, conf, graph...)
  ```
  Set `VOSK_MODEL_PATH` and `VOSK_MODEL_LANGUAGE` if you use another folder or language.
* Online recognition (Google's speech service through SpeechRecognition) sends the recording over
  the internet. It is disabled unless you set `VOICE_ONLINE_RECOGNITION_ENABLED=true`, and the app
  shows a warning when it is used. Data analysis itself always stays local.
* Voice is a Premium feature and needs the analyst's "use voice" permission.

## E-mail alerts and the anomaly worker

Anomaly rules (Organization Admin, Premium) detect IQR outliers, z-score outliers, period-over-period
changes, revenue or sales drops, deviations from a historical baseline and sudden increases in
missing or invalid values (blank cells plus values that no longer parse as the column's number or
date type, for example after an append turned a numeric column into text). Findings are worded as
"possible" issues to review, never as fraud. Repeated findings are suppressed, and a cooldown limits
how often a rule e-mails.

SMTP examples for `.env`:

```ini
# Gmail (needs 2-step verification and an App Password)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=your-16-character-app-password
SMTP_FROM=you@gmail.com

# Microsoft 365 / Outlook
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USE_TLS=true
```

Send a test e-mail from Organization > Settings (needs a plan with automated e-mail alerts; at most
5 test e-mails per organization per hour). Without SMTP, alerts are still recorded and the
notification log shows them as "skipped".

The Streamlit app is not a scheduler. "Run check now" works from the Alerts pages; for recurring
checks register the worker with Windows Task Scheduler (replace `<path>`):

```powershell
schtasks /Create /SC HOURLY /TN "VoiceDataAnalyst Alerts" /TR "<path>\.venv\Scripts\python.exe <path>\scripts\run_anomaly_checks.py"
python scripts\run_anomaly_checks.py      # run once by hand
schtasks /Delete /TN "VoiceDataAnalyst Alerts" /F
```

## How data is handled

* Uploads: `.csv`, `.xlsx` and `.xls` only, checked by content (a renamed executable, a
  macro-enabled or zip-bomb workbook, or binary data in a CSV is rejected). CSV encoding, delimiter,
  decimal and thousands separators are detected or configurable. Malformed rows are skipped and
  listed with line numbers (or the file is rejected, if you choose), never dropped silently. For
  Excel you choose the sheet. Files with more columns than `MAX_COLUMNS` (default 1,000) are refused
  before any processing.
* Column types are decided from every value (no sampling):
  * Decimal notation means decimal (`1200.00` stays a float).
  * Whole numbers stay integers, even with missing values.
  * Identifiers with leading zeros, and whole numbers longer than 15 digits, stay text so no digit
    is lost.
  * A decimal becomes a float only when a 64-bit float stores it exactly (a round-trip check, so
    `0.30000000000000004` and Excel formula results qualify) and it is converted with exact
    rounding. Decimals that cannot be stored exactly, or are out of range, stay text and the import
    report says so with examples.
  * Currency amounts become decimals.
  * Dates must use one consistent format; mixed formats are reported.
  * Mostly-numeric columns that contain a few bad values stay text, and the bad values are listed.
  * In Excel files the cell number format decides decimal intent.
* Missing values are blanks, whitespace and your configured placeholders (default `NA`, `N/A`,
  `null`, `None`, `NaN`, `#N/A`). Zero, `False` and categories such as "Unknown" are never treated as
  missing unless you add them.
* The original upload is stored unchanged (read-only, with a SHA-256 checksum). Each cleaning step
  writes a new immutable version (Parquet). History records who did what, the parameters, rows and
  cells changed, and quality before and after. Restoring copies an older version forward. Failed or
  cancelled operations never publish a partial version.
* Saving is short: a new version (or a new dataset) is written, scored and hashed in a staging
  folder first, and only a brief database transaction moves it into place, so a large upload or
  merge does not hold up other users. If two people change the same dataset at the same time, one
  change is saved and the other is told the dataset changed while it ran; a result built from an
  outdated version never becomes current.
* The versions-per-dataset limit is a hard limit: the pinned original and the newest versions are
  kept and older ones (including the one just replaced) are purged. A purged file that Windows still
  has open stays on disk, counts toward the storage quota, and is deleted by a later save or by the
  hourly worker.
* Quality score (0 to 100) = 40% completeness + 20% uniqueness + 25% validity + 15% consistency.
  Only confirmed problems lower it; suspected issues (outliers, spelling variants, possible
  duplicate keys) are shown separately because they may be legitimate. Row hashing treats `0.0` and
  `-0.0` as equal, as pandas does. A dataset with no rows scores 100 and its health check says
  "Empty (no rows)".
* Read-only reports (identifier checks, repeated values, outliers, casing and redundancy) belong
  to the version they were computed on and disappear when the dataset gets a new version.

## Security model

* Tenant isolation: every service call takes the identity from the server-side session and reads
  role, organization status, plan features and permissions fresh from the database. Records of
  another organization behave as "not found".
* Plan features are enforced in the services, not only hidden in the UI. For example, without
  "Standard Plotly charts" results carry no chart (on screen and in exports), and heatmap or
  correlation charts need "Advanced analytics".
* Results: analysis history and saved reports follow current dataset access. Anyone who is not an
  Organization Admin can list, open, export, edit or delete an item (their own included) only while
  its dataset is active and granted to them. Organization Admins see every item of their
  organization.
* Custom regular expressions (pattern validation and constraint rules in Cleaning) run on the
  `regex` engine with a time limit for each value (0.5 s, capped by the operation's time limit). The
  first timeout stops the whole operation, so one pattern cannot freeze the app for other users.
  Nested repeats such as `(a+)+` are refused up front, and values longer than 1,000 characters cannot
  be checked with a custom pattern.
* Sessions: random tokens kept only in server memory; only their SHA-256 hash is stored. Idle and
  absolute expiry, logout revocation, and revocation when a password changes, a user is deactivated
  or an organization is suspended.
* Passwords: Argon2id with a per-password salt, a strength policy, and login lockout after
  repeated failures (also for unknown usernames, without revealing which usernames exist).
* Files: server-generated names only; stored paths are checked against path traversal. Downloads
  require the export permission and the right organization (analysts without it also do not get the
  table toolbar's "Download as CSV" button; anything shown on screen is, by nature, visible to that
  user). Exported CSV/Excel cells and headers that would run as spreadsheet formulas are escaped.
* Network: the server listens on `localhost` only (`.streamlit/config.toml`, `server.address`). To
  use it from other computers on a trusted network, set `address = "0.0.0.0"` and put HTTPS and a
  firewall rule in front of it.
* UI: dataset text is never rendered as Markdown/HTML (it cannot inject links or remote images).
  Users see friendly errors with a reference number; details go to `logs\app.log` with secrets
  redacted. Streamlit usage statistics are disabled.
* Audit: logins, user and plan changes, uploads, every data change, exports and Super Admin actions
  are recorded in the audit log.

## Resource limits and large files

The limits above are engineering starting points, not guarantees. Before expensive work the app
estimates memory from file sizes and Parquet metadata and compares it with a fraction of the RAM
that is free at that moment. It also checks free disk space. If the work does not fit, the
operation is refused with a clear message instead of failing halfway.

* CSV and Excel imports, profiling, exact duplicate detection, cleaning and append are processed in
  bounded batches.
* Statistics (medians, quartiles, duplicate counts) are always exact, never sampled.
* Merges compute the exact output row count before running and refuse oversized results.
* Width: uploads and merge results wider than `MAX_COLUMNS` (default 1,000) are refused. Existing
  datasets are always profiled, in memory-checked batches of columns (1,000 columns profile in about
  5 seconds on the development machine).

Benchmark on your own 16 GB machine before raising limits; these defaults were not benchmarked on
dedicated 16 GB target hardware.

## Tests and code quality

```powershell
pytest -q                    # full suite: services, security, tenant isolation, UI (Streamlit AppTest)
pytest tests\ui -q           # only the UI tests
pip install -r requirements-dev.txt
ruff check .                 # lint
```

The tests use a temporary database and storage folder each time and never touch `data\` or
`storage\`. Ollama, microphones, speech engines and SMTP servers are mocked or replaced by local
stub servers in the tests.

Dependencies are pinned in `requirements.txt`. `regex==2026.9.29` is the newest release that still
ships Python 3.10 wheels; keep it while Python 3.10 is supported.

## Project structure

```
app.py                    Streamlit entry point (sign-in gate, role navigation)
config/settings.py        all settings from environment / .env
core/                     errors, constants, storage (safe paths, atomic writes), logging
database/                 SQLAlchemy models, engine/session, migrations, plan seeding
auth/                     authentication (sessions, lockout), authorization (guards), passwords
modules/                  super_admin, organization_admin, analyst, subscriptions, usage, org settings
services/                 datasets, typing, profiling, missing values, duplicates, cleaning,
                          versioning, append, merge, analytics, LLM, charts, voice, anomalies, e-mail
components/               shared UI pieces (session, navigation, feedback, theme, charts, previews)
views/                    one file per page (platform/, org/, data/, analysis/, alerts/)
scripts/                  initialize_database, create_super_admin, check_environment,
                          seed_demo_data, run_anomaly_checks
tests/                    pytest suite (tests/ui holds the Streamlit AppTest tests)
storage/, logs/, data/    runtime data (git-ignored)
```

## Troubleshooting

* `Activate.ps1 cannot be loaded because running scripts is disabled`: run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
  `.\.venv\Scripts\python.exe -m streamlit run app.py` without activating.
* `Port 8501 is already in use`: `streamlit run app.py --server.port 8502`.
* Upload rejected as too large: plan limit and `ABSOLUTE_UPLOAD_CEILING_MB` both apply; raise the
  plan limit in the Super Admin panel and `server.maxUploadSize` in `.streamlit/config.toml`.
* "Not enough memory": close other programs (Ollama models use several GB), use a smaller file, or
  switch to a smaller model.
* `database is locked`: another process (for example a backup tool or a second app instance) holds
  the SQLite file; keep `data\` out of OneDrive/Dropbox sync folders and exclude it from real-time
  antivirus scanning if it causes delays.
* File access errors under `storage\`: make sure the folder is writable by your user and not inside
  a synced or protected folder.
* Microphone not working: check the browser's site permission and Windows microphone privacy
  settings; recording needs `localhost` or HTTPS.

## Known limitations

* Real Ollama, Gemini, Vosk, microphone and SMTP integrations are covered by tests with mocks and
  local stub servers. They need the external pieces (or a Gemini key) to run for real.
* Recurring anomaly checks need the Task Scheduler worker; the web app does not schedule work.
* `.xls` files do not carry number-format information through `xlrd`, so whole numbers in `.xls`
  files are read as integers.
* Sessions are per browser tab: refreshing the page asks you to sign in again (this keeps the token
  out of the browser entirely).
* Generating new identifiers keeps at most 50,000 old-to-new pairs in the operation history; the
  previous version still holds every original value.
* Comparing one metric across categories and time together (one line per category) is not
  supported yet; the rule-based interpreter says so when it drops the category. A question about
  "duplicate customer records" checks whole-row duplicates, not repeated customer ids.
* Prepared exports stay in the browser session until you sign out, and the Ask page keeps your
  15 latest answers on screen (older ones are in Analysis history).
* The test e-mail limit (5 per hour) is checked before sending, so two requests at the same moment
  can exceed it slightly; tests sent while SMTP is not configured also count.
* Cancel and the operation time limit apply to every change of an existing dataset, but not to the
  final quality scoring of a brand-new dataset (an import, or an append or merge saved as new).
* A file left in the versions folder that no database row references (for example after a crash
  between moving the file and the commit) is not swept automatically.
