# Architecture and Service Contracts

Requirements: `docs/SPECIFICATION.md`. This file is the contract every module is built against.

## Layers

```
Streamlit (app.py, views/, components/)        UI only: never queries the DB (app.py only runs the
                                               one-time schema bootstrap), never trusts browser ids
        |
modules/   role workflows (super_admin, organization_admin, analyst) + plan/usage rules
services/  data and AI logic (datasets, profiling, cleaning, versions, analytics, voice, alerts)
        |
auth/      AuthContext + authorize() guards (fresh DB checks on every call)
database/  SQLAlchemy models, engine/session, migrations, seeding
core/      errors, constants, storage (paths, atomic writes), time
```

## Kernel (frozen: do not edit; request changes instead)

| Module | Use it for |
|---|---|
| `config/settings.py` | `get_settings()`; every limit comes from here or from the plan row |
| `core/errors.py` | Raise only these. `AppError.user_message` is shown to users, so never put paths, SQL or secrets in it |
| `core/constants.py` | `Role`, `Permission`, `Feature`, `UsageMetric`, `LogicalType`, `PANDAS_DTYPE_FOR` |
| `core/storage.py` | `temporary_directory(org_id)`, `atomic_output`, `atomic_copy`, `resolve_relative`, `sanitize_display_filename`; never build paths from user input |
| `core/timeutil.py` | `timeutil.utcnow()` (naive UTC). Call through the module so tests can patch it |
| `database/connection.py` | `session_scope()` (commit on success, rollback on error); `begin_write(session)` takes SQLite's write lock (`BEGIN IMMEDIATE`) for a check-then-write sequence |
| `database/models.py` | ORM models. Every org-owned row has `organization_id` |
| `auth/authorization.py` | `AuthContext`, `authorize`, `authorize_dataset`, `authorize_super_admin`, `visible_datasets_query`, `has_permission` |
| `auth/password_manager.py` | `hash_password`, `verify_password`, `needs_rehash`, `validate_password_strength` |
| `modules/subscriptions.py` | `get_limits`, `require_feature`, `enforce_*` quota checks, `org_storage_bytes` |
| `modules/usage_tracking.py` | `record_usage`, `monthly_total`, `enforce_monthly_analysis_quota`, `usage_summary` |
| `modules/org_settings.py` | `get_org_settings`, `validate_changes` (placeholders, date order, text placeholder, alert recipients) |
| `services/audit_service.py` | `record_audit(db, ctx, action, ...)` inside your transaction; `record_error(component, exc, ctx=)` |
| `services/resource_manager.py` | `check_memory`, `check_disk`, `estimate_*`, `OperationGuard`, `CancellationToken`, `human_bytes` |
| `services/type_inference.py` | typing contract, `ParseOptions`, `parse_numeric_token`, `numeric_patterns`, `parse_dates`, `DATE_FORMATS`, `BOOLEAN_TRUE/FALSE` |
| `services/frame_utils.py` | `normalize_frame`, `logical_type_for_dtype`, `hash_rows`, `records`, `json_safe`, `escape_spreadsheet_formulas` |
| `services/quality_score.py` | `compute_quality`, `compute_quality_batches`, `QualityAccumulator`, `WEIGHTS` |
| `services/versioning_service.py` | see below |
| `services/dataset_service.py` | upload/import/list/export (already implemented) |

### Versioning API (`services/versioning_service.py`)
* `load_frame(ctx, dataset_id, version_id=None, columns=None, permission=None, feature=None)` authorizes, checks memory and returns typed data.
* `load_frame_with_ref(...)` does the same and also returns a `VersionRef` (dataset_id, version_id, version_number, path, rows, columns, schema, organization_id).
* `load_preview(ctx, dataset_id, version_id=None, max_rows=None)` returns at most PREVIEW_MAX_ROWS rows and reads only the first batch.
* `iter_frame_batches(ctx, dataset_id, ..., batch_rows=None, columns=None)` yields bounded batches (PARQUET_BATCH_ROWS).
* `get_version_ref(ctx, dataset_id, version_id=None, permission=, feature=)`.
* `apply_frame_operation(ctx, dataset_id, operation_type=, parameters=, transform=, permission=CLEAN_DATA, feature=, guard=None)`, where `transform(df) -> (new_df, stats, summary)` and `stats` keys are `rows_added`, `rows_removed`, `rows_affected`, `cells_changed`. It publishes a new version or raises; failures are recorded automatically.
* `apply_streaming_operation(ctx, dataset_id, ..., transform_batch=, guard=)`, where `transform_batch(batch, row_offset) -> (new_batch, stats_delta)`. Use it when the operation is row-local and the data may be large.
* `publish_version(db, ctx, dataset, ..., guard=None)` for custom flows inside your own `session_scope` (append/merge results, restore).
* `create_dataset_with_initial_version(db, ctx, name=, source_type="derived", original_filename=, frame=|parquet_source=)` for new datasets produced by operations (for example a merge saved as a new dataset).
* `list_versions`, `list_operations`, `restore_version`, `verify_original`, `record_failed_operation`.
* Storage dtypes are exactly `Int64`, `float64`, `boolean`, `datetime64[ns]`, `string`. `normalize_frame` enforces them.
* Publish protocol: the version file (and for a new dataset the copy and SHA-256 of the original)
  is written, scored and hashed in an organization staging folder with no database write. Then a
  short `begin_write` transaction re-checks the dataset's current version against the caller's
  snapshot (`ConflictError` "The dataset changed while this operation was running" when it moved),
  the dataset name and the plan limits, moves the files into place and writes the rows. Placed files
  are deleted on any failure (immediately and by the rollback hook), so other tenants' writes never
  wait for a large upload and a result built from an outdated version never becomes current.
* Retention: versions per dataset is a hard limit; pinned versions and the newest ones are kept and
  older ones (including the one just replaced) are purged. A purged file that cannot be deleted
  (open on Windows) is flagged `file_pending_delete` (migration 2), still counted by
  `org_storage_bytes`, retried after later publishes and by `reclaim_purged_files()` in the worker.

## Service conventions (mandatory)
1. A public function takes `ctx: AuthContext` first, opens its own `session_scope()`, and calls `authorize(...)` or `authorize_dataset(...)` before reading or writing any tenant data. Never accept an `organization_id` for tenant actions; use `ctx.organization_id` after `authorize`. Super Admin functions call `authorize_super_admin` and may take an explicit organization id.
2. Return plain data (dicts, dataclasses, DataFrames), never ORM objects (the session is closed).
3. Preview is separate from apply. `preview_*` functions are read-only and return before/after counts, affected rows and sample rows. `apply_*` functions publish a version through the versioning helpers.
4. Destructive actions (dropping columns, deleting rows, rewriting ids) need an explicit `confirm=True` argument and raise `ValidationError` without it.
5. Record every data-changing or security-relevant action with `record_audit`. Versioning records its own audit entries for published versions.
6. Never `eval`/`exec`, never shell out, never read paths derived from user input.
7. Large data: check `resource_manager.check_memory` / `check_disk` before expensive work. Prefer column-subset loads (`columns=[...]`) and batch iteration. Statistics must be exact, and any approximation is labelled.
8. Long loops take an optional `OperationGuard` and call `guard.step(...)` so cancellation and timeouts work. Temporary files only go in `storage.temporary_directory`.
9. User-facing text: plain sentences with no stack traces, paths or SQL. Never use the em-dash character in any text, comment or doc.

## Backend permission / feature matrix
| Operation | Role / permission | Plan feature |
|---|---|---|
| Upload, import, inspect upload | `UPLOAD_DATASETS` | `CSV_XLSX_UPLOAD` |
| Profiling, health check, quality report | dataset access | `DATA_PROFILING` |
| Missing-value analysis and preview | dataset access | `MISSING_VALUES` |
| Missing-value apply | `CLEAN_DATA` | `MISSING_VALUES` |
| Duplicate detection and preview | dataset access | `DUPLICATE_DETECTION` |
| Duplicate removal, composite-key resolution | `CLEAN_DATA` | `DUPLICATE_DETECTION` |
| Identifier generation or rewrite | `CLEAN_DATA` + `confirm=True` | `ADVANCED_CLEANING` |
| Standard cleaning (trim, control chars, rename, empty rows/cols, casing) | `CLEAN_DATA` | `STANDARD_CLEANING` |
| Advanced cleaning (category mapping, date standardization, numeric/type conversion, outlier cap/exclude, pattern/email validation fixes, constraints) | `CLEAN_DATA` | `ADVANCED_CLEANING` |
| Cleaning issue detection (read-only) | dataset access | `STANDARD_CLEANING` |
| Append, merge | `CLEAN_DATA` | `APPEND_MERGE` |
| Restore version | `CLEAN_DATA` | none |
| Export data or results, download original | `EXPORT_DATA` | none |
| Text analysis (chat, query builder) | dataset access + monthly quota | `TEXT_AI_ANALYSIS` |
| Correlation, growth, anomaly questions, heatmaps | dataset access | `ADVANCED_ANALYTICS` |
| Result charts (Ask, history, saved reports and their HTML/JSON exports) | dataset access | `STANDARD_CHARTS` (without it results carry no chart; heatmap and correlation charts also need `ADVANCED_ANALYTICS`) |
| Voice transcription | `USE_VOICE` | `VOICE_ANALYSIS` |
| Saved reports | `SAVE_REPORTS` | `SAVED_REPORTS` |
| Anomaly rules and manual checks | Org Admin | `ANOMALY_DETECTION` |
| E-mail notification of anomalies | Org Admin rule config | `EMAIL_ALERTS` (without it events are recorded but not e-mailed) |
| Test e-mail | Org Admin, at most 5 per organization per hour | `EMAIL_ALERTS` |
| Users, permissions, dataset grants, org settings, org usage | Org Admin | none (user quota applies) |
| Organizations, plans, features, subscriptions, platform metrics | Super Admin | none |

Organization Admins implicitly hold every permission. Analysts hold only granted permissions
(defaults: `export_data`, `use_voice`, `save_reports`) and see only datasets granted to them in
`dataset_access` (an analyst who uploads a dataset is granted access automatically).

Results follow current dataset access: anyone who is not an Org Admin lists, opens, exports,
edits or deletes an analysis history item or a saved report (their own included) only while
its dataset is active and granted to them (`visible_datasets_query`). Rows without a dataset
link are visible to Org Admins only. Org Admins see every item of their organization.

## Data contracts
* Type inference: decimal notation stays float and is never downcast; a decimal becomes float
  only when `type_inference.decimal_is_lossless` holds (float round trip), and it is converted
  with Python `float()` per unique token (exact rounding). Integers with leading zeros or more
  than 15 digits stay text identifiers. Precision fallbacks are reported in the import report.
* Width: inspect, import and merge refuse more than `MAX_COLUMNS` columns; profiling never
  refuses an existing dataset and reads columns in memory-checked batches.
* Quality score: a dataset with no cells scores 100 and its health status is `empty`. Row
  hashing treats `0.0` and `-0.0` as equal, so duplicate counts agree with pandas.
* Anomaly method `missing_increase` measures missing or invalid values per column: null cells
  plus values that fail the column's expected number or date parse (including a column whose
  type changed to text between versions).
* User regular expressions (cleaning pattern validation and constraint rules) compile with the
  `regex` module and match with `timeout=` (0.5 s per value, capped by
  `OperationGuard.remaining()`) and `concurrent=True`; the first timeout aborts the operation.
* LLM provider: `LLM_PROVIDER=ollama` (default, local) or `gemini`. The Gemini path is schema only:
  `llm_service.bounded_schema(..., schema_only=True)` sends column names, types and semantic tags,
  never samples, min/max, missing shares, row counts or results; `narrative_allowed()` turns the
  narrative off; the key goes in the `x-goog-api-key` header, never in a URL or log.
* Analytics verification recomputes every calendar period independently with numpy, checks
  which periods a sort and limit keep, and checks growth against the previous calendar period.
  The anomaly z-score and flag checks reuse `flag_unusual_changes` on independently recomputed
  changes, so they are not a fully independent recomputation. Sums over no values are `None`;
  ties for first place are named and warned about.
* Read-only UI reports carry the version id they were computed on and are ignored once the
  dataset has a newer current version; editor widget keys include dataset, version and column.

## Operation type names (cleaning history)
`import`, `restore`, `missing_values`, `drop_columns`, `remove_duplicates`,
`resolve_key_duplicates`, `generate_ids`, `trim_whitespace`, `remove_control_chars`, `map_categories`,
`standardize_dates`, `convert_numeric`, `convert_types`, `rename_columns`, `remove_empty`,
`normalize_case`, `handle_outliers`, `fix_invalid_values`, `standardize_placeholders`, `append`, `merge`.
Missing-value fills and row or column drops are one `missing_values` operation (the strategy of
each column is in its parameters). `validate_constraints` is read-only and creates no version.
The navigation reads what a user may see through `modules.account.account_summary`; it is a
menu hint only, every service call still authorizes itself.

## UI contract (Streamlit)
* `app.py` bootstraps once per database (`st.cache_resource` keyed by the DB path; no tenant data), validates the session token on every rerun (`components.session.refresh_context`) and builds the menu from `components.navigation.PAGES` + `page_keys_for(capabilities)`.
* Page files live under `views/` exactly at the paths in `components.navigation.PAGES`. Each page starts with:
  ```python
  ctx = require_ctx()          # from components.session; stops the page when signed out
  caps = capabilities()        # from components.navigation; UI hints only, never a security check
  page_header("Title", "One line saying what the page does.")
  ```
* Pages call services/modules only (never the database or files directly) and wrap every call in `components.feedback.run_action(lambda: ..., success="...")`, which shows `AppError.user_message` safely and turns unexpected errors into a reference number.
* Untrusted text: anything that comes from datasets, uploads, users or the LLM is shown with `st.dataframe`, `st.text`, `st.code` or widget options, or passed through `md_escape` before any Markdown-rendering call (`st.markdown`, `st.write`, `st.caption`, `st.info`/`success`/`warning`/`error`, `st.metric` labels). `unsafe_allow_html=True` is only allowed with static HTML/CSS (for example the issue classes in `components.theme`) where every dynamic value is passed through `components.feedback.escape` (HTML escaping). HTML reports escape with `html.escape`.
* Exports are generated only when the user clicks an explicit "Prepare ..." button; the bytes are kept in session state and then offered with `st.download_button`. Never call an export service while rendering, because each call records usage and an audit entry.
* No `st.cache_data` / `st.cache_resource` for tenant data. Per-page state goes in `st.session_state` under keys prefixed with the page name; sign-in and sign-out clear all session state.
* Destructive actions use the preview, then confirm, then apply flow: a Preview button shows the exact effect (counts plus before/after sample), and the Apply button requires `confirm(...)` to be ticked when the service needs `confirm=True`. After apply, show the new version number and quality before/after.
* Long operations pass `progress=Progress("...")` (or a guard) where the service accepts it.
* Downloads use `st.download_button` with bytes returned by authorized service export functions only.
* Features missing from the plan show `feature_unavailable("...")` with a short explanation instead of controls that would fail. Missing permissions behave the same way.
* Every button does exactly what its label says and produces visible feedback. No decorative controls.
* Shared components: `components/dataset_picker.py` (`dataset_selector`), `components/feedback.py`, `components/theme.py` (colors, `score_color`), `components/session.py`, `components/navigation.py` (`capabilities`, `PAGES`).
* UI tests use `tests/ui/conftest.py::make_app(username, page="views/...py")` with users created by `factory.user(org_id, role, username=...)`. Each page needs AppTest coverage for: rendering without exception for each allowed role, its primary actions (forms, buttons, file upload via `app.file_uploader[0].set_value((name, bytes, mime))`), and error states. `st.audio_input` cannot be driven by AppTest; voice is covered by service tests and the live browser check.

## Testing conventions
* Fixtures in `tests/conftest.py`: `factory` (organization, user, ctx, org_admin, analyst, super_admin, dataset, grant), `org_id` (Premium), `admin_ctx`, `analyst_ctx`, `sales_frame`, `app_settings`. Every test gets its own temporary database and storage, and fake memory/disk readings (8 GB / 200 GB).
* Use `monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: small)` to simulate low memory, and `disk_free_provider` for low disk. Never create genuinely large files: lower the settings limits instead (`set_settings(app_settings.with_overrides(...))`).
* No network: mock `requests` / `smtplib` / speech engines, or run a local stub server on 127.0.0.1.
* Every service needs tests for success, validation errors, permission and plan denial, and cross-tenant `NotFoundError`.
