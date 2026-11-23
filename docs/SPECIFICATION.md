# Voice-Enabled Data Analyst SaaS: Product Specification

This is the requirements document the application was built against (copied
from the project brief; section numbers are referenced by `docs/TRACEABILITY.md`).

## 1. Objective
A complete, working, multi-tenant SaaS application that lets organizations upload business
datasets, inspect data quality, clean missing and invalid values, detect and handle duplicates,
append and merge datasets, and analyze real data with natural-language text and voice commands.
Roles: Super Admin, Organization Admin, Data Analyst. Everything visible must work end to end.
Runs locally on Windows (16 GB RAM target) and must be resource aware.

## 2. Technology stack
Windows 10/11; Python 3.10+; Streamlit; SQLite + SQLAlchemy; Pandas + NumPy; Plotly; Ollama with a
configurable Llama model (llama3:8b and a smaller quantized alternative); SpeechRecognition; optional
offline Vosk; openpyxl; smtplib + email.mime; Argon2 or bcrypt; python-dotenv; pytest; Pandera or
robust custom validation; Python logging. No paid AI APIs, cloud LLMs, React, Node.js or MongoDB.

## 3. Target hardware and resource configuration
Initial configurable limits: Basic upload 50 MB per file; Premium 500 MB; CSV processing target 250 MB;
XLSX target 100 MB; combined append/merge input target 100 MB; absolute upload ceiling up to 1 GB only
when the operation, RAM, disk and hardware testing support it; preview max 500 rows; AI result tables
max 500 rows; one AI request at a time; LLM context 4096 tokens by default. These are starting points,
not guarantees. Do not assume all RAM is available. Store limits in environment variables and plan
records. A resource-aware layer checks available memory, disk space, file size, schema complexity,
expected output growth and memory needs, and rejects, defers or batches operations when resources are
insufficient. Never crash intentionally, silently return partial results or corrupt source data.

## 4. Multi-tenant architecture and security
Strict data isolation; organization_id on organization-owned records (users, datasets, versions,
cleaning operations, analysis history, alerts, subscriptions, usage). Authenticate securely; salted
password hashes; RBAC enforced in service functions (not only UI); organization taken from the
authenticated server-side identity, never from browser input; no cross-organization access to datasets,
files, reports or history; plan restrictions enforced on backend operations; parameterized queries / safe
ORM; SQLite foreign keys on; session expiration and secure logout; login-attempt protection; audit log
of security-sensitive and data-changing actions; never expose secrets, credentials or internal paths;
downloads restricted to authorized users; prevent path traversal and filename collisions; never execute
LLM-generated code; treat uploads and dataset contents as untrusted. Super Admin privileges explicit and
auditable.

## 5. Super Admin panel
Create, edit, activate, suspend and deactivate organizations. Create and manage Basic and Premium plans.
Configure plan pricing and billing metadata (no payment gateway). Assign plans to organizations.
Configure user, storage, upload, dataset and usage limits. Enable/disable individual features per plan.
Restrict voice analysis, advanced cleaning, append/merge, predictive analytics and automated alerts by
plan. View subscriptions and renewal metadata. View totals of organizations, users, datasets, analysis
jobs. Monitor storage usage, application errors, processing latency and usage. View audit logs and
subscription history. Suspend an organization without deleting its historical data.
Basic features: CSV/XLSX uploads within limits; data profiling; missing-value treatment; duplicate
detection; standard cleaning; text-based AI analysis; standard Plotly charts.
Premium: everything in Basic plus voice analysis; advanced cleaning; append and merge; advanced business
analytics; predictive anomaly detection; automated email alerts; saved reports and higher limits.
Plans and feature permissions are stored in the database.

## 6. Organization Admin panel
Organization dashboard metrics; create, edit, activate, deactivate and manage analysts; configure analyst
permissions; upload CSV, XLSX and supported XLS; select Excel sheets; preview datasets; view row/column
counts, schema, inferred types, file size, storage usage; view quality scores and cleaning reports; run
automated health checks; configure missing-value strategies; detect and handle duplicates; validate
unique identifiers; append datasets; merge datasets on selected keys; view versions and cleaning
history; restore previous versions; view analysis history and organization usage; configure anomaly
rules and alert recipients; export cleaned datasets and reports. All limited to the user's organization
and permissions.

## 7. Data upload and profiling
On upload inspect: rows and columns; column names and types; numeric, categorical, text, boolean and
datetime fields; missing counts and percentages; exact duplicate rows; repeated values in selected
columns; unique/distinct values; constant and empty columns; inconsistent date formats; invalid numeric
values; leading/trailing whitespace; potentially inconsistent categories; suspicious outliers;
conversion failures; potentially duplicated business keys; dataset quality score. Detailed report with
tables, charts and warnings. Distinguish confirmed problems from suspected issues (a legitimate outlier
or repeated category is not automatically an error). CSV: configurable encoding, delimiter detection,
decimal and thousands separators, malformed-row handling. Excel: sheet selection, workbook validation,
safe parsing limits. Never silently skip invalid rows.

## 8. Missing-value treatment
Detect real nulls/NaN, empty strings, whitespace-only strings and configured placeholders (N/A, NA,
null, None). Admins configure which placeholders mean missing. Zero, False or a category named "Unknown"
are never missing without explicit configuration. Per column show missing count, missing %, type,
recommended strategy, reason and expected impact.
Numeric: mean, median, constant, user-defined value, forward/backward fill when appropriate, leave
unchanged, drop rows with missing values in selected columns, drop columns only after explicit
confirmation. Categorical/text: most frequent value, user-defined replacement, configurable placeholder
such as "Not Provided", forward/backward fill when justified, leave unchanged, drop selected rows.
Datetime: parse supported formats, report invalid dates, explicit user-defined replacement, never invent
dates without explaining. Special cases: all-null columns, very high missing %, mixed types, small
datasets, single non-null value, IDs/labels/sensitive columns.
Rules: offer recommended automatic cleaning and manual control; median for suitable numeric and mode for
suitable categorical by default; never drop columns just for many missing values; never silently alter
identifiers; do not fill target labels or sensitive fields without explicit permission; avoid
ffill/bfill when row order is meaningless; prevent data leakage when data is used for ML evaluation;
preview before destructive operations; before/after missing counts; record changed cells and affected
rows; preserve types; reversible through versioning. Exact statistics on large data (no sample-based
means/medians presented as exact).

## 9. Duplicate detection and unique values
A. Exact duplicate rows: detect, counts and %, preview groups, keep first / keep last / remove all rows of
duplicate groups, preserve order, before/after counts, record removed rows. Large CSVs need a correct
cross-chunk strategy (disk-backed hashing with collision verification or external sorting); per-chunk
checks alone are insufficient.
B. Duplicate values in selected columns (customer_id, email, phone, transaction_id, product_code):
repeated values and frequencies; configurable trimming and case-insensitive comparison; explicit
normalization rules for emails, phones, identifiers; never silently modify stored values; never delete
records just because names, cities or categories repeat.
C. Unique identifier validation: distinguish repeated values from duplicate records; validate whether an
identifier should be unique; generate UUIDs or another configured identifier only when authorized;
check generated IDs do not collide; keep a mapping of changed identifiers; never rewrite customer IDs,
invoice numbers or transaction IDs without explicit authorization.
D. Composite-key duplicates (customer_id + transaction_date, product_id + warehouse_id, invoice_number +
invoice_date): show conflicting records and require a defined resolution policy.
E. Audit and restoration: preserve original data, record duplicate operations, version history, undo via
restore, never destroy the original upload.

## 10. Advanced cleaning
Trim whitespace; remove control characters; normalize category labels with explicit mappings;
standardize date formats; convert numeric strings safely; report failed conversions; remove fully empty
rows and columns after confirmation; identify redundant columns for review; detect inconsistent casing;
configurable numeric separators; outliers by IQR or z-score with retain, cap, exclude or review; validate
email formats and configured patterns; rename columns; convert types; validate column constraints;
preview every destructive operation. Never modify legitimate business values without a visible rule,
preview and operation record.

## 11. Versioning and transactional safety
Keep the original upload, current working version, version numbers, cleaning history, acting user and
timestamp, parameters, changed cells and rows, added/removed rows, before/after dimensions, quality
before/after. Atomic file writes, database transactions, safe temporary storage. On failure keep the
original and never publish partial versions. Versions are genuinely recoverable data. Configurable
retention limits and storage quotas.

## 12. Append and merge
Append: schema validation, match columns by name, report extra/missing columns, explicit unmatched-column
policies, preview, detect possible duplicates after appending, bounded processing.
Merge: inner, left, right, outer; single and composite keys; configurable suffixes. Before executing:
validate keys, inspect key types, detect duplicate keys, estimate output row growth, detect many-to-many,
warn about multiplication, check resources, preview and allow cancellation. Never silently discard
unmatched records or columns. Check plan permissions first.

## 13. Data Analyst panel
Select authorized datasets/sheets; schema and previews; quality reports; natural-language questions by
text and (when permitted) voice; calculated answers, explanations, tables and charts; download results
and cleaned data; recent queries; save analyses and reports; compare metrics across categories and time.
Example questions: total revenue; product with highest sales; monthly revenue trend; region with highest
profit; duplicate customer records; columns with missing values; average sales by category; top 10
products by profit; month-over-month growth; unusual changes in revenue; missing values per column.
Answers come from real calculations on the selected authorized dataset.

## 14. Local AI engine and safe analysis
Ollama with a configurable model. Workflow: question, inspect schema, interpret, structured JSON plan,
validate columns/operations/parameters, execute approved pandas operations, return computed results,
chart if requested, explain in business language. Never eval/exec LLM output. Allowlist: filtering,
sorting, grouping, aggregations, descriptive statistics, correlation, time-series summaries, top-N, chart
specifications. Reject shell commands, imports, file operations, unsafe expressions, unauthorized
queries. No access to files, secrets, environment or OS. Uploaded content is untrusted data, not
instructions. Do not send whole datasets to the LLM; send schema, aggregates and bounded samples.
Validate numerical results independently. Model unavailable, malformed response or ambiguous query:
error or clarification request; never fabricate results.

## 15. Voice input and offline operation
SpeechRecognition microphone interface: start/stop recording, speech to text, show recognized text for
confirmation, let the user edit it, submit confirmed text to the same analytics engine. Handle
microphone permission errors, empty audio, timeouts, unsupported languages. Optional offline Vosk with a
local model. Clearly distinguish offline analysis from speech modes that need internet. Never upload
recordings or dataset contents to external services.

## 16. Charts and reporting
Bar, line, pie and donut, histogram, scatter, box, heatmap, correlation matrix, time series, grouped and
stacked, top-N comparisons, numeric distributions. Chosen from question and data types with manual
override. Validate columns, types and aggregations; handle nulls and empty results; accurate labels,
titles, legends, hover; no misleading charts; downloadable result tables and reports; organization and
dataset permissions applied to every result.

## 17. Anomaly detection and e-mail alerts
Explainable statistics: IQR outliers; configurable z-score thresholds; unexpected daily, weekly, monthly
changes; revenue/sales drops; historical-baseline deviations; sudden increases in missing or invalid
data. Do not call outliers fraud or confirmed issues. Admins configure metrics, thresholds, recipients,
frequency and severity. Anomaly history, duplicate-alert suppression, cooldowns, notification logs, SMTP
via environment variables, clear success/failure status, no hardcoded credentials. Use a separate worker
or Windows Task Scheduler for recurring checks; provide a manual check action.

## 18. Database and file storage
Normalized tables: users, organizations, subscription_plans, subscriptions, plan_features, datasets,
dataset_versions, cleaning_operations, analysis_history, usage_records, anomaly_rules, anomaly_events,
notification_logs, audit_logs. Keys, foreign keys, unique constraints, indexes, timestamps, FK
enforcement, organization-scoped service methods, large files outside SQLite, organization-isolated
server-controlled storage, no loading every dataset when a dashboard opens, initialization and safe
migrations.

## 19. Memory-efficient processing (mandatory)
Chunked CSV reading; avoid needless copies; estimate memory before expensive work; monitor memory and
disk; avoid loading several full datasets; bounded batches and disk-backed temporary processing;
incremental aggregation where valid; exact global statistics when needed; cross-chunk duplicate
detection; preserve precision of financial and identifier fields; check merge expansion before
allocation; timeouts and cancellation; clean temporary files; progress indicators; clear messages when a
task exceeds resources; never present partial results as complete; label any approximation.

## 20. Streamlit interface
Login; role-specific navigation; Super Admin, Org Admin and Analyst dashboards; upload and preview;
profiling; missing values; duplicates; cleaning workspace; append and merge; AI analyst chat; voice;
interactive charts; analysis history; alerts and anomalies; user management; subscription management;
settings and audit logs. Reusable components, forms, session state, tables, cards, progress indicators,
confirmations, friendly errors. Every button performs its advertised action. No decorative controls.
Streamlit caches scoped by organization and dataset; never leak private data through shared caches.

## 21. Configuration and Windows setup
requirements.txt, .env.example, .gitignore, README.md, database initialization script, secure
interactive Super Admin setup (no hardcoded password), environment diagnostics script, Windows
PowerShell commands, Ollama installation/model setup/availability checks/troubleshooting, microphone
permissions, PowerShell execution policy, file access and SMTP setup.

## 22. Automated testing
pytest with isolated databases and temp folders, covering authentication (login, hashing, expiry, roles,
tenant isolation, plan restrictions), missing values, duplicates (including across CSV chunks), cleaning
and versioning (type conversion, whitespace, outlier review, restore, atomic failure recovery), append
and merge (schema mismatch, all joins, duplicate keys, many-to-many, expansion warnings, resource-limit
rejection), AI (valid/invalid plans, invalid columns, wrong operations, division by zero, exact results,
unsafe code rejection, Ollama unavailable, malformed output), files (valid CSV/Excel, corrupted,
unsupported, oversized, empty, malformed), alerts (thresholds, duplicate suppression, missing SMTP, e-mail
failures) and resources (large files, low memory, low disk, temp cleanup, cancellation). Tests never touch
real data.

## Additional acceptance requirements (execution addendum)
* Unit tests for data rules and boundaries; integration tests for database + file storage; authorization
  and cross-tenant tests for every organization-owned resource; Streamlit AppTest smoke tests for
  navigation, forms, callbacks and error states; an end-to-end test upload, profile, clean, version,
  restore, analyze, export; voice routing tests with the microphone mocked; Ollama available/unavailable
  and malformed responses; append/merge row counts, join semantics, duplicate keys, many-to-many; alert
  thresholds, suppression, missing SMTP, SMTP failure; oversized/corrupted files, disk/memory limits,
  cancellation, temp-file cleanup; security tests for path traversal, unsafe file types, unauthorized
  downloads, password handling and session expiry.
* Known defect class: a `revenue` column with whole-valued decimals must be inferred as float, with
  regression tests for integer-only, decimal, nullable numeric, numeric strings, currency strings and
  identifiers with leading zeros, and proof that the original upload is unchanged and versions are
  recoverable (`tests/test_datasets_service.py::test_upload_creates_typed_version_and_preserves_original`).
