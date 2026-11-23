# Verification report

**Status: PASS WITH LIMITATIONS**

Date: 2026-11-18. Final code commit: `e690936`. Commits after it change documentation only
(`README.md`, `docs/TRACEABILITY.md` and this report). The limitations are the parts that need software or
hardware that is not on the build machine (see section 5), not failing checks.

## 1. Starting point

The workspace was empty and no earlier version of the project existed on the machine, so the
application was built from scratch against `docs/SPECIFICATION.md`. The "known defect" named in
the brief (`tests/test_datasets_service.py::test_upload_creates_typed_version_and_preserves_original`,
revenue typed as float) could therefore not be repaired from old code: it was reconstructed as a
contract test together with the regression cases the brief asked for, and it passes:

* `test_type_inference_contract_through_upload` (parametrized): integer-only columns, decimal
  columns written as whole numbers (`1200.00` stays float), nullable integers and floats, numbers
  with thousands separators, currency strings, leading-zero identifiers, 16-digit identifiers and
  mixed text.
* `test_leading_zero_identifiers_keep_their_digits`, `test_xlsx_number_formats_keep_decimal_intent`,
  `test_csv_float_repr_decimals_import_as_float`, `test_import_value_outside_float_range_is_kept_as_text`.
* The original upload stays unchanged (read-only, SHA-256 checked) and every version is
  recoverable: `test_publish_and_restore_round_trip`, `test_tampered_version_file_cannot_be_restored`.

## 2. Automated tests

| Run | Python | Where | Result |
|---|---|---|---|
| Full suite after the last code change | 3.13.12 | project `.venv` | 1,884 passed, 0 failed, 0 skipped (11 min 36 s) |
| Clean install, README quick start (`python -m venv`, `pip install -r requirements.txt`) | 3.13.12 | new folder extracted from `git archive` | 1,884 passed, 0 failed, 0 skipped (11 min 56 s); `pip check`: no broken requirements |
| Clean install | 3.10.21 | new folder extracted from `git archive`, venv created with uv | 1,884 passed, 0 failed, 0 skipped (12 min 28 s) |

Commands: `pytest -q` (full suite) and `ruff check .` (lint: all checks passed).

Each clean install also ran, in order: `scripts/check_environment.py` (30 checks, 0 failed; 7
warnings, all for optional or external pieces: the Vosk package and model, PyAudio, Ollama, SMTP,
the microphone, and the database not created yet), `scripts/initialize_database.py`, `scripts/create_super_admin.py --from-env`, and a
Streamlit AppTest boot of `app.py` (sign-in page, no exception). The SHA-256 of that install's
`data/app.db` and `logs/app.log` was identical before and after the test run, so the tests never
touch real data.

The suite has 1,884 tests: services, security (77), tenant isolation (285 cross-organization
probes), kernel concurrency and migrations, scripts, and 254 Streamlit AppTest UI tests that cover
every page for each role, with plan and permission variations.

## 3. Review and fix round

Five independent reviewers read the finished code: security and tenancy, data correctness,
transactional safety and resources, UI completeness, and specification traceability. They reported
42 findings (5 high, 18 medium, 19 low). An adversarial verifier reproduced each of the 23 medium or
high findings: 22 confirmed and 1 rejected as intended behavior (SEC2, applied anyway as
hardening). One fix round followed, by seven agents with separate files. For each fixed finding the
agent ran its regression test before the fix and saw it fail. The exceptions are ST7 and ST8
(documentation, and a refactor whose new module has its own tests), ST10 (tests only, checked by
mutation: 5 of its 9 tests fail when foreign keys or backups are switched off) and extra guard
tests for behavior that already worked. The lead's follow-up fixes (f83f8a7, e29cb3f, 6a4b779) were
checked by mutation: each new test fails when its fix is removed.

| ID | Finding | Outcome | Commits |
|---|---|---|---|
| SEC1 | User regex could backtrack for hours and freeze every tenant | Fixed: `regex` engine, per-value timeout, GIL released, first timeout aborts | a537d94, 26bd6e6 |
| SEC2 | Revoked analyst could still read old results | Hardened: history and saved reports follow current dataset access | 08daf0f |
| SEC3 | No column limit; wide files took minutes | Fixed: `MAX_COLUMNS` on import, append and merge; profiling batched (1,000 columns: 52 s to 5 s) | 0c93d9e, 91b509d, f0c5f82, c3ebbaf |
| SEC4 | Cell text with a blank line escaped the issue HTML block | Fixed | 4b28a7b, 1462acf |
| SEC5 | Column names rendered as HTML in a Plotly chart | Fixed | 1beab42 |
| SEC6 | Test e-mail without plan feature or rate limit | Fixed (limit check not atomic: known limit) | fe2904d |
| DC1 | 16-17 digit decimals turned columns into text | Fixed: float round trip; also found and fixed a 1-ulp error in pandas parsing | a0d335c, 5d7e996 |
| DC2 | Time results applied `limit` to the wrong end | Fixed | 9af6a45, b3db4bd |
| DC3 | `0.0` and `-0.0` counted as different rows | Fixed | 4421e8c |
| DC4 | Growth results sorted by value failed verification | Fixed | 1a0f3e1, 671f108 |
| DC5 | Mixed ISO dates reported as "kept as text" | Fixed | 0c353a5 |
| DC6 | Ties for first place named only one group | Fixed | 4b4bb1c |
| DC7 | Out-of-range numbers crashed the import | Fixed | a0d335c, 70826dc |
| DC8 | Exact-threshold changes flagged by float error | Fixed | bf44ef0 |
| DC9 | Date-only upper bound dropped the last day | Fixed | e2b9610 |
| DC10 | Sum of no values shown as 0 | Fixed | be64bb8, 7f5c686 |
| DC11 | Empty dataset scored 0 and "critical" | Fixed: 100 and "Empty (no rows)" | 4ed313c, 1a0631d, 212c537 |
| SR1 | Dataset creation held the SQLite write lock for the whole import | Fixed: staging, then a short transaction | 3f7eb15, c597364, 13ed922 |
| SR2 | Concurrent changes raised IntegrityError or overwrote newer data | Fixed: `BEGIN IMMEDIATE` and a re-check | cea1c7a |
| SR3 | Purged files still open on Windows leaked | Fixed: flagged, counted, retried (migration 2) | 224c9c6, 9a95bae, 6a4b779 |
| SR4 | Scoring failure left an orphan version file | Fixed | 5283d33 |
| SR5 | Some operations ignored timeout and cancel | Partly: every operation on an existing dataset honours them; scoring a brand-new dataset (import, append or merge as new) takes no guard (known limit) | 73c3f44, a71b3f8, f83f8a7, e29cb3f |
| SR6 | Original download loaded without a memory check | Fixed | 66797dd |
| SR7 | Retention kept one version too many | Fixed | c202d3a |
| UI1 | Missing-value plan carried over to other columns | Fixed | a5a09bb |
| UI2 | Read-only reports stayed visible after the data changed | Fixed | 56a20d3 |
| UI3 / ST3 | "Standard charts" plan toggle had no effect | Fixed in engine, pages and exports | e51daf9, de0ad58, 6c98430 |
| UI4 | Combine page forgot the chosen dataset | Fixed | 37a31be |
| UI5 | A column named `row` hid the row numbers | Fixed | 1aa5283 |
| UI6 | Account page rendered names as Markdown | Fixed | cc2604b |
| UI7 | Voice section disappeared without explanation | Fixed | b455298 |
| UI8 | Large objects kept in session memory | Partly: uploads no longer re-hashed; exports and 15 answers stay in session (known limit) | 0a4e3d7 |
| ST1 | Cleaning could not be imported on Python 3.10 | Fixed; full suite passes on 3.10.21 | 77296dd |
| ST2 | Duplicate questions answered "no rows to analyse" | Fixed | 9bab711 |
| ST4 | Basic plan could re-draw results as heatmaps | Fixed | 07c17a6 |
| ST5 | No alert for rising invalid data | Fixed: `missing_increase` counts missing or invalid values | 54ab139 |
| ST6 | No metric across categories and time together | Partly: the rule-based answer says when it drops the category (known limit) | 1d400fd |
| ST7 | `docs/TRACEABILITY.md` missing | Fixed | this release |
| ST8 | Architecture text did not match the code | Fixed | 74043d0, b108693 |
| ST9 | Identifier mapping capped at 50,000 pairs | Known limit, now stated on the page | 2887ca0 |
| ST10 | Foreign keys and migration path untested | Tests added | 72a6470 |

Before the review, a live browser check found seven smaller issues (server reachable from the
network, table CSV download offered without export permission, float noise in result tables,
rounded quality label, cramped metric rows, raw chart titles, raw session column names). All were
fixed (9742ec5, 1d99abd). Tests cover the download button, display rounding, the quality label and
chart titles; the localhost binding and the hidden button were re-checked in the browser.

## 4. Checks in a real browser

A local server (scratch database, Chrome driven through DevTools) was used. On the final code
(e690936 and later, docs-only changes since):

* Startup on the existing scratch database (schema 1) ran migration 2 and first wrote the backup
  `app-...-before-v2.db`; the existing datasets still loaded.
* Upload and import of a CSV through the new staged publish: `0.30000000000000004` stored
  bit-exact as float64, `N/A` missing, a nullable integer column typed Int64.
* Missing-value plan preview and apply ("Saved as version 2") through the guarded treatment.
* No tracebacks in the server or app logs during these runs.

The Ask page chart and a custom-pattern cleaning preview were not repeated in the browser on the
final code; they are covered by the test suites that passed on it. The checks below ran earlier,
on code before the fix round (commit 9742ec5 and earlier):

* Sign-in, sign-out notice, forced password change for admin-created accounts (navigation locked
  until changed) for an Organization Admin and an analyst.
* Upload, inspect and import of a CSV (decimal column typed float, integer column typed integer,
  `N/A` treated as missing); the quality profile (score, confirmed and suspected issues, chart).
* "Ask the analyst" with the rule-based fallback (total revenue matched an exact Decimal sum),
  a monthly trend chart, and the voice panel rendering (no real audio injected).
* Missing-value plan preview and apply (new version), then restore of version 1 (new version 3
  equal to version 1 cell for cell; the original file's SHA-256 unchanged).
* The analyst sees only the pages their role and permissions allow.
* During the fix round (after 1d99abd): the server listens on 127.0.0.1 only, and the table
  toolbar's "Download as CSV" button is hidden (`display: none`) for an analyst without the export
  permission and shown for the Organization Admin.
* The only console errors came from a deliberate request for a page that does not exist; the
  server logs contained no passwords.

## 5. Not verified for real (external requirements)

* **Ollama** is not installed on the build machine, so the LLM path ran against a local stub server
  that imitates the Ollama HTTP API in tests, and the app used the clearly labelled rule-based
  fallback. Install Ollama and pull
  `llama3:8b` (or `llama3.2:3b`) to use the AI planner (README "Local AI with Ollama").
* **Offline speech (Vosk)** and a **real microphone**: transcription was tested with generated WAV
  audio and mocked engines; the browser recording widget was only seen rendering. Needs
  `requirements-optional.txt` and a Vosk model folder.
* **SMTP**: e-mail sending was tested with a fake SMTP client (TLS, SSL and every failure step), not
  with a real mail provider.
* **Performance targets** were measured on the development machine (23 GB RAM, 8 cores), not on
  dedicated 16 GB target hardware; treat the limits as starting points (README "Resource limits").
* `create_super_admin.py` interactive prompts were tested with mocked input; `--from-env` ran for
  real in the clean installs.

## 6. Requirement coverage

`docs/TRACEABILITY.md` maps all 78 requirements of the specification to code and tests: 76
implemented and 2 partial (identifier mapping capped at 50,000 pairs; one metric across
categories and time together). Every test it cites exists (checked when the file was generated).
Known limitations are listed in README "Known limitations".

## 7. Handoff update (2026-11-20)

Added for running the project on another laptop:

* `run.bat` (Windows) and `run.sh` (macOS / Linux) set everything up on the first run (Python 3.10
  to 3.13 venv, packages, `.env`, database, Super Admin, optional demo data) and then start the
  app. Tested from a fresh `git archive` copy on Windows: `run.bat` first run, second run and
  start (HTTP 200 on port 8501, health "ok"); `run.sh` first run under Git Bash. Not run on a real
  Mac or Linux machine. Package wheels were checked to resolve for Python 3.10 to 3.13 on Windows,
  macOS (Intel and Apple Silicon) and Linux; Python 3.14 is not supported (no numpy 2.2.6 wheels).
* `START_HERE.md` (step by step) and `AGENTS.md` (read by Antigravity and other coding agents).
* Optional Gemini instead of Ollama (`LLM_PROVIDER=gemini`), schema only: tests with a local stub
  server prove that requests carry the question and column names and types but no sample values,
  ranges, row counts or results, that the key travels only in a header, that the narrative is
  skipped, and that key, quota and outage errors become plain messages. Not exercised against the
  real Gemini API (no key on the build machine).
* Full suite after these changes: 1,897 passed, 0 failed, 0 skipped (Python 3.13.12).

## 8. Release archive

`dist/voice-data-analyst-saas.zip` is built with `git archive` from the commit that contains this
report (folder `voice-data-analyst-saas/`). It holds tracked source, docs and tests only: no
`.env`, no `data/`, `logs/` or `storage/` content, no virtual environment, no caches and no user
data. The two `.gitkeep` files keep the empty `logs/` and `storage/` folders.
