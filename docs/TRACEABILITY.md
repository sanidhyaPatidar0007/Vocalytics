# Requirement traceability

Every requirement of docs/SPECIFICATION.md mapped to the code that implements it and the
automated tests that prove it. Statuses: **implemented** (code and tests in place) and
**partial** (works with a stated limit, also listed in README "Known limitations"). Features
that need software outside this repository (Ollama, Vosk, a microphone, SMTP) are implemented
and tested with stubs; see docs/VERIFICATION_REPORT.md section 5.

Summary: 76 implemented, 2 partial.

## Section 3: Target hardware and resource configuration

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R01 | Configurable limits (Basic 50 MB, Premium 500 MB, absolute ceiling) in env vars and plan records | implemented | config/settings.py:Settings; database/seed.py:seed_default_plans; modules/subscriptions.py:enforce_upload_size | test_resource_limits.py::test_plan_upload_limit_is_capped_by_absolute_ceiling; test_datasets_service.py::test_upload_respects_plan_size_limit |
| R02 | CSV 250 MB / XLSX 100 MB / append-merge 100 MB targets drive stricter memory checks | implemented | services/dataset_service.py (csv/xlsx_processing_target_mb); services/append_service.py:estimate_resources; services/merge_service.py | test_resource_limits.py::test_large_csv_above_processing_target_is_rejected_on_low_memory; test_resource_limits.py::test_large_csv_is_accepted_when_memory_allows |
| R03 | Preview max 500 rows; AI result tables max 500 rows | implemented | services/versioning_service.py:load_preview; services/analytics_engine.py:execute_plan | test_analytics.py::test_result_rows_are_capped; test_analyst_module.py::test_get_history_item_reports_truncation |
| R04 | One AI request at a time; LLM context 4096 | implemented | services/llm_service.py:concurrency_gate; config/settings.py (ollama_num_ctx) | test_llm_service.py::test_busy_semaphore_raises_ai_busy; test_analytics.py::test_busy_model_returns_error_status |
| R05 | Resource layer checks memory, disk and size, and rejects or batches; never partial results | implemented | services/resource_manager.py:check_memory, check_disk, estimate_parquet_memory, OperationGuard | test_resource_limits.py::test_low_disk_rejects_imports; test_cleaning.py::test_low_memory_blocks_column_preparation; test_append_merge.py::test_merge_memory_check_is_reported_then_enforced |

## Section 4: Multi-tenant architecture and security

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R06 | organization_id on org-owned records; strict isolation; cross-org access behaves as not found | implemented | database/models.py; auth/authorization.py:authorize, authorize_dataset, visible_datasets_query | test_tenant_isolation.py::test_foreign_ids_are_rejected_and_leave_b_unchanged; test_tenant_isolation.py::test_listings_never_include_other_organizations; test_tenant_isolation.py::test_every_function_taking_an_owned_id_is_in_the_table |
| R07 | Salted password hashes, login-attempt protection, session expiry, secure logout | implemented | auth/password_manager.py:hash_password; auth/authentication.py:login, validate_session, logout | test_auth.py::test_argon2_hashes_are_salted_and_verify; test_auth.py::test_lockout_applies_to_unknown_usernames; test_auth.py::test_idle_timeout_boundary_and_expiry; test_auth.py::test_logout_revokes_and_is_idempotent |
| R08 | RBAC and plan restrictions enforced on backend operations | implemented | modules/subscriptions.py:require_feature; auth/authorization.py:authorize; services/analytics_engine.py:_charts_enabled (STANDARD_CHARTS); components/chart_components.py (ADVANCED_CHART_TYPES need ADVANCED_ANALYTICS); modules/analyst.py:_plan_chart (exports) | test_analytics.py::test_basic_plan_cannot_ask_advanced_questions; test_anomalies.py::test_basic_plan_is_denied_everywhere; none for standard_charts or the chart override; test_analytics.py::test_disabling_standard_charts_on_plan_suppresses_charts; test_analyst_module.py::test_exports_drop_stored_charts_the_plan_no_longer_includes |
| R09 | SQLite foreign keys on | implemented | database/connection.py:_on_connect (PRAGMA foreign_keys=ON; a probe confirms FK violations are rejected) | none found (grep foreign_keys, PRAGMA in tests) |
| R10 | Audit log; no secrets or paths exposed; never execute LLM code; path traversal prevented | implemented | services/audit_service.py:record_audit, record_error; core/storage.py:resolve_relative; services/analytics_engine.py:validate_plan | test_auth.py::test_audit_rows_for_login_failure_and_logout_never_contain_secrets; test_storage.py::test_resolve_relative_refuses_traversal; test_analytics.py::test_no_dynamic_code_execution_in_source |
| R11 | Downloads restricted to authorized users | implemented | services/dataset_service.py:export_dataset, download_original; modules/analyst.py:export_report | test_analyst_module.py::test_export_report_needs_export_permission; test_analyst_module.py::test_export_report_visibility_and_tenant_isolation |

## Section 5: Super Admin panel

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R12 | Create, edit, activate, suspend and deactivate organizations without deleting data | implemented | modules/super_admin.py:create_organization, update_organization, set_organization_status | test_security.py::test_suspended_organization_loses_sessions_and_access; test_authorization.py::test_suspended_org_is_blocked |
| R13 | Create and manage Basic and Premium plans, pricing, billing metadata and limits | implemented | modules/super_admin.py:create_plan, update_plan, set_plan_active; database/models.py:SubscriptionPlan | test_super_admin.py::test_create_plan_stores_every_feature_row; test_super_admin.py::test_create_plan_validation |
| R14 | Assign plans; view subscriptions, renewal metadata and history | implemented | modules/super_admin.py:assign_plan, list_subscriptions; database/models.py:Subscription | test_super_admin.py::test_assign_plan_keeps_history_and_one_active_subscription; ui/test_platform_pages.py::test_assign_plan_and_history_shows_ended_subscription |
| R15 | Enable or disable individual features per plan (stored in the DB) | implemented | modules/super_admin.py:set_plan_feature; modules/subscriptions.py:enabled_features; every toggle is enforced, including standard_charts (engine, result pages and exports) | test_super_admin.py::test_create_plan_without_features_disables_all; none for standard_charts; test_analytics.py::test_disabling_standard_charts_on_plan_suppresses_charts; test_analysis_pages.py::test_chart_tab_respects_standard_charts_plan_feature |
| R16 | Restrict voice, advanced cleaning, append/merge, predictive analytics and alerts by plan | implemented | services/voice_service.py:_authorize_voice; services/cleaning_service.py; services/append_service.py; services/anomaly_service.py:create_rule | test_voice_routing.py::test_voice_requires_the_voice_analysis_plan_feature_on_both_steps; test_duplicates.py::test_basic_plan_cannot_generate_ids; test_anomalies.py::test_basic_plan_is_denied_everywhere |
| R17 | Platform totals, storage, application errors, processing latency, usage; audit and error logs | implemented | modules/super_admin.py:platform_metrics, _latency, list_error_events | test_super_admin.py::test_platform_metrics_match_seeded_data_exactly; test_super_admin.py::test_list_error_events_is_redacted_and_bounded |

## Section 6: Organization Admin panel

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R18 | Org dashboard; create, edit, activate and deactivate analysts; analyst permissions; dataset grants | implemented | modules/organization_admin.py:org_dashboard, create_analyst, update_user, set_user_active, set_user_permissions, grant_dataset_access | test_org_admin.py::test_dashboard_reports_metadata_without_loading_files; test_org_admin.py::test_create_analyst_with_custom_permissions |
| R19 | Upload CSV/XLSX/XLS with sheet selection; preview; schema, types, size and storage usage | implemented | services/dataset_service.py:inspect_upload, import_dataset, list_excel_sheets, storage_summary | test_datasets_service.py::test_xlsx_upload_with_sheet_selection; test_e2e_workflow.py::test_excel_upload_with_sheet_choice_and_profiling |
| R20 | Quality scores, health checks, versions, restore, analysis history, usage, alert config, exports | implemented | services/profiling_service.py:health_check; services/versioning_service.py:restore_version; modules/organization_admin.py:usage_report, org_analysis_history | test_profiling.py::test_health_check_healthy_and_audited; test_versioning.py::test_publish_and_restore_round_trip; test_org_admin.py::test_usage_report_groups_by_calendar_month |

## Section 7: Data upload and profiling

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R21 | Profile rows and columns, types, missing, duplicates, distinct, constant and empty columns, outliers, whitespace, categories, business keys, quality score | implemented | services/profiling_service.py:profile_dataset, repeated_values; services/quality_score.py:compute_quality_batches | test_profiling.py::test_constant_and_empty_columns; test_profiling.py::test_profile_is_consistent_with_quality_score; test_profiling.py::test_repeated_values_with_no_repeats_is_empty |
| R22 | Inconsistent date formats, invalid numerics and conversion failures reported | implemented | services/type_inference.py:infer_types; services/profiling_service.py | test_type_inference.py::test_inconsistent_date_formats_are_reported_not_coerced; test_type_inference.py::test_mostly_numeric_column_reports_failed_conversions |
| R23 | Confirmed and suspected issues kept separate | implemented | services/profiling_service.py:_column_confirmed, _column_suspected | test_profiling.py::test_profile_is_consistent_with_quality_score |
| R24 | CSV encoding, delimiter, decimal and thousands separators, malformed-row handling; never silently skip rows | implemented | services/dataset_service.py:detect_encoding, detect_delimiter, ImportOptions | test_datasets_service.py::test_cp1252_file_is_detected; test_datasets_service.py::test_semicolon_delimiter_and_decimal_comma; test_datasets_service.py::test_report_lists_malformed_rows_instead_of_dropping_silently |
| R25 | Excel sheet selection, workbook validation and safe parsing limits | implemented | services/dataset_service.py:list_excel_sheets, inspect_upload | test_datasets_service.py::test_rejects_invalid_or_unsafe_files; test_datasets_service.py::test_xlsx_number_formats_keep_decimal_intent |

## Section 8: Missing-value treatment

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R26 | Detect nulls, empty and whitespace values and configurable placeholders; zero, False and Unknown are not missing | implemented | services/missing_values.py:missing_mask, analyze_missing; modules/org_settings.py:validate_changes | test_missing_values.py::test_analyze_counts_empty_and_whitespace_and_recommends; test_datasets_service.py::test_placeholders_and_legitimate_values; test_datasets_service.py::test_custom_placeholders_are_configurable |
| R27 | Per column: count, percent, type, recommended strategy, reason, expected impact | implemented | services/missing_values.py:analyze_missing, recommend_plan | test_missing_values.py::test_recommend_plan_only_contains_safe_fills |
| R28 | Numeric, categorical and datetime strategies; ffill only with meaningful order; drop columns only with confirmation | implemented | services/missing_values.py:validate_plans, treat_frame, apply_missing_treatment | test_missing_values.py::test_ffill_requires_meaningful_row_order; test_missing_values.py::test_drop_rows_and_drop_column; test_missing_values.py::test_date_value_fill_and_invalid_dates |
| R29 | Never silently alter identifiers, targets or sensitive fields; prevent ML leakage | implemented | services/missing_values.py:is_sensitive_name, is_target_name, IDENTIFIER_ALLOWED, TreatmentOptions.fit_filter | test_missing_values.py::test_fit_filter_uses_only_training_rows |
| R30 | Preview, before/after counts, changed cells, versioning, exact statistics | implemented | services/missing_values.py:preview_missing_treatment; services/versioning_service.py:apply_frame_operation | test_missing_values.py::test_categorical_mode_fills_empty_and_whitespace |

## Section 9: Duplicate detection and unique values

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R31 | Exact duplicate rows, keep first/last/none, order preserved, cross-chunk hashing with collision verification | implemented | services/duplicate_manager.py:scan_duplicates, find_duplicates_in_chunks, remove_duplicates | test_duplicates.py::test_duplicates_across_csv_chunks_are_found; test_duplicates.py::test_remove_duplicates_keep_options_preserve_order; test_duplicates.py::test_spill_to_disk_gives_identical_results |
| R32 | Value duplicates in selected columns with trimming, case and email/phone/id normalization; stored values untouched | implemented | services/duplicate_manager.py:normalize_values, find_value_duplicates | test_duplicates.py::test_value_duplicates_with_normalization_keep_stored_values; test_duplicates.py::test_value_duplicates_trim_and_case_options |
| R33 | Identifier validation; authorized UUID generation, collision check, mapping of changed ids | partial | services/duplicate_manager.py:validate_identifier, generate_ids (UUIDs, collision check, old-to-new mapping). Limit: the history keeps at most 50,000 pairs (MAPPING_CAP, mapping_truncated) and the page says so; the previous version keeps every original value | test_duplicates.py::test_validate_identifier; test_duplicates.py::test_uuid_ids_in_new_column_leave_original_untouched; ui/test_data_pages.py::test_generate_ids_shows_mapping; test_data_pages.py::test_generate_ids_says_when_the_mapping_is_truncated |
| R34 | Composite-key conflicts with an explicit resolution policy | implemented | services/duplicate_manager.py:find_composite_key_conflicts, resolve_composite_key_duplicates | test_duplicates.py::test_composite_key_conflicts_identical_vs_conflicting; ui/test_data_pages.py::test_composite_keys_resolve_with_policy |
| R35 | Original preserved, operations recorded, undo via restore | implemented | services/versioning_service.py:publish_version, restore_version, verify_original | test_versioning.py::test_publish_and_restore_round_trip; test_datasets_service.py::test_upload_creates_typed_version_and_preserves_original |

## Section 10: Advanced cleaning

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R36 | Trim, control characters, category mapping, date standardization, numeric conversion with separators and failure report | implemented | services/cleaning_service.py:preview_cleaning, apply_cleaning (_SPECS) | test_cleaning.py::test_trim_whitespace_preview_is_exact_and_read_only; test_cleaning.py::test_remove_control_characters; test_cleaning.py::test_map_categories_counts_per_mapping_entry; test_cleaning.py::test_convert_numeric_integers_and_separators; test_cleaning.py::test_convert_numeric_failures |
| R37 | Remove empty rows and columns after confirmation; redundant columns; casing; rename; type conversion | implemented | services/cleaning_service.py:identify_redundant_columns, detect_casing_issues, apply_cleaning | test_cleaning.py::test_remove_empty_requires_confirm_and_removes_rows_and_columns; test_cleaning.py::test_identify_redundant_columns; test_cleaning.py::test_rename_columns_and_validation; test_cleaning.py::test_convert_types_text_to_datetime |
| R38 | Outliers by IQR or z-score with retain, cap, exclude or review; email and pattern validation; column constraints | implemented | services/cleaning_service.py:outlier_report, apply_cleaning (handle_outliers, fix_invalid_values, validate_constraints) | test_cleaning.py::test_outlier_retain_reports_and_never_changes_data; test_cleaning.py::test_outlier_exclude_needs_confirm; test_cleaning.py::test_fix_invalid_emails; test_cleaning.py::test_validate_constraints_report |
| R39 | Cleaning available on every supported Python version (3.10+, spec 2) | implemented | services/cleaning_service.py imports sre_parse when re._parser is missing (Python 3.10); full suite passes on Python 3.10.21 in a clean install (docs/VERIFICATION_REPORT.md) | none (suite only run on 3.13); test_cleaning.py::test_cleaning_service_imports_on_python_310_layout |

## Section 11: Versioning and transactional safety

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R40 | Original kept; version numbers; history with user, parameters, changes, dimensions, quality before and after | implemented | services/versioning_service.py:publish_version, list_versions, list_operations | test_versioning.py::test_publish_and_restore_round_trip |
| R41 | Atomic writes, transactions, never publish partial versions | implemented | core/storage.py:atomic_output; services/versioning_service.py:file_hooks, record_failed_operation | test_storage.py::test_atomic_output_removes_partial_file_on_error; test_append_merge.py::test_merge_row_count_mismatch_saves_nothing |
| R42 | Retention limits and storage quotas | implemented | services/versioning_service.py:_retention_candidates; modules/subscriptions.py:enforce_storage_quota | test_versioning.py::test_retention_purges_files_only_after_commit |

## Section 12: Append and merge

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R43 | Append: schema validation, match by name, extra and missing columns, policies, preview, possible duplicates, bounded processing | implemented | services/append_service.py:analyze_append, preview_append, apply_append, stream_append | test_append_merge.py::test_analyze_append_reports_columns_case_and_type_conflicts; test_append_merge.py::test_union_keeps_all_columns_and_fills_nulls; test_append_merge.py::test_bounded_streaming_gives_identical_output |
| R44 | Merge: all joins, composite keys, suffixes, key validation, duplicate keys, growth estimate, many-to-many, cancellation | implemented | services/merge_service.py:analyze_merge, preview_merge, apply_merge, key_statistics | test_append_merge.py::test_join_types_match_reference_and_exact_estimate; test_append_merge.py::test_custom_suffixes_and_suffix_collisions; test_append_merge.py::test_many_to_many_is_detected_warned_and_needs_confirmation; test_append_merge.py::test_merge_cancellation_and_conflict |

## Section 13: Data Analyst panel

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R45 | Analyst: authorized datasets, previews, quality reports, text and voice questions, downloads, history, saved reports | implemented | modules/analyst.py:list_my_history, save_report, export_report, export_history_result; services/analytics_engine.py:answer_question | test_analyst_module.py::test_list_my_history_returns_only_own_rows_newest_first; test_analyst_module.py::test_save_report_copies_the_analysis |
| R46 | Example questions answered from real calculations | implemented | services/analytics_engine.py:rule_based_plan, build_summary; duplicates answers report 'No duplicate rows were found.' from the rows analysed. Limit: 'duplicate customer records' checks whole-row duplicates | test_analytics.py::test_rule_based_plan_for_example_questions (pins the plan only); test_analytics.py::test_rule_based_answers_are_correct_on_data; test_analytics.py::test_duplicates_with_none_found_does_not_claim_empty_dataset; test_analytics.py::test_rules_answer_for_duplicate_question_without_duplicates |
| R47 | Compare metrics across categories and time | partial | services/analytics_engine.py: one metric over time, or by category, is supported; one line per category over time is not (validate_plan rejects group_by on time_series) and the rule-based interpreter says when it drops the category | test_analytics.py::test_growth_month_over_month (time only); test_analytics.py::test_rules_time_answer_reports_a_dropped_category |

## Section 14: Local AI engine and safe analysis

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R48 | JSON plan, allowlist validation, pandas execution, no eval/exec, only schema and bounded samples sent | implemented | services/analytics_engine.py:validate_plan, execute_plan, get_schema_summary; services/llm_service.py:generate_plan, bounded_schema | test_analytics.py::test_schema_summary_is_bounded_and_sanitized; test_analytics.py::test_no_dynamic_code_execution_in_source; test_analytics.py::test_invalid_plan_twice_is_rejected_and_audited |
| R49 | Independent numeric verification; grounded narrative | implemented | services/analytics_engine.py:verify_result, narrative_is_grounded | test_analytics.py::test_verify_result_catches_corruption_for_each_kind; test_analytics.py::test_narrative_with_hallucinated_number_is_rejected |
| R50 | Model unavailable, malformed reply or ambiguous question: error or clarification, never fabricated results | implemented | services/llm_service.py:check_status, generate_plan; services/analytics_engine.py:_plan_question | test_analytics.py::test_real_unreachable_ollama_falls_back_to_labelled_rules; test_analytics.py::test_malformed_json_retries_once_then_falls_back_to_rules; test_analytics.py::test_clarify_plan_from_model |

## Section 15: Voice input and offline operation

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R51 | Record, transcribe, show and edit the text, submit it to the same engine | implemented | services/voice_service.py:transcribe; views/analysis/ask.py (st.audio_input) | test_voice_routing.py::test_voice_question_runs_the_same_pipeline_as_typed_text |
| R52 | Microphone permission errors, empty audio, timeouts, unsupported languages | implemented | services/voice_service.py:_run_with_timeout, _resolve_language, inspect_wav | test_voice_routing.py::test_microphone_permission_error_is_reported; test_voice_routing.py::test_empty_or_unreadable_audio; test_voice_routing.py::test_recognition_timeout; test_voice_routing.py::test_unsupported_language |
| R53 | Optional offline Vosk; online mode clearly marked and off by default | implemented | services/voice_service.py:engine_status, _resolve_engine | test_voice_routing.py::test_online_recognition_is_disabled_by_default; test_voice_routing.py::test_auto_prefers_the_offline_engine_even_when_online_is_enabled |

## Section 16: Charts and reporting

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R54 | Bar, line, pie/donut, histogram, scatter, box, heatmap, correlation, grouped and stacked; automatic choice with override | implemented | services/chart_service.py:available_chart_types, suggest_chart, validate_chart_spec, build_figure | test_charts.py::test_category_and_numeric_suggests_bar; test_charts.py::test_available_types_include_heatmap_and_correlation_for_a_matrix; test_analytics.py::test_chart_override_and_safety_fallbacks |
| R55 | Nulls and empty results handled, accurate labels, no misleading charts; downloadable results and reports | implemented | services/chart_service.py:build_figure, report_html; modules/analyst.py:export_report | test_charts.py::test_empty_and_all_null_tables_return_none; test_charts.py::test_bar_figure_has_titles_labels_and_a_zero_based_axis; test_analyst_module.py::test_export_report_html_is_escaped_and_offline |
| R56 | Permissions applied to every result (plan features on chart types) | implemented | components/chart_components.py (ADVANCED_CHART_TYPES, caps); modules/analyst.py:_plan_chart (exports) and _dataset_visible (history and saved reports follow current dataset access) | none; test_analyst_module.py::test_exports_drop_stored_charts_the_plan_no_longer_includes; test_analyst_module.py::test_revoking_dataset_access_hides_own_history |

## Section 17: Anomaly detection and e-mail alerts

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R57 | IQR, z-score, period changes, revenue drops, baseline deviation | implemented | services/anomaly_service.py:detect_iqr, detect_zscore, detect_period_change, detect_baseline_deviation | test_anomalies.py::test_iqr_upper_bound_is_exclusive_and_aggregated; test_anomalies.py::test_period_change_flags_both_directions_with_neutral_wording; test_anomalies.py::test_baseline_deviation_uses_previous_six_periods |
| R58 | Sudden increases in missing or invalid data | implemented | services/anomaly_service.py: missing_increase counts missing or invalid values (values failing the column's number or date parse, including a type change to text); services/quality_score.py:count_invalid_text | test_anomalies.py::test_missing_increase_across_two_versions (missing only); test_anomalies.py::test_missing_increase_flags_rise_in_invalid_values_kept_as_text; test_anomalies.py::test_missing_increase_counts_all_non_numeric_values_after_a_type_change_to_text |
| R59 | Configurable metrics, thresholds, recipients, frequency, severity; history, suppression, cooldowns, notification logs | implemented | services/anomaly_service.py:create_rule, update_rule, run_rule, list_events, list_notifications | test_anomalies.py::test_cooldown_suppresses_notifications_until_it_expires; test_anomalies.py::test_concurrent_runs_cannot_duplicate_events_or_alerts; test_anomalies.py::test_recipients_need_the_email_alerts_feature |
| R60 | SMTP from env vars, success/failure status, missing SMTP; worker or Task Scheduler plus manual check | implemented | services/email_service.py:send_email, smtp_status; scripts/run_anomaly_checks.py; services/anomaly_service.py:run_scheduled_checks | test_anomalies.py::test_missing_smtp_logs_skipped_and_keeps_events_new; test_anomalies.py::test_smtp_failure_logs_failed_without_secrets_and_events_stay_new; test_anomalies.py::test_scheduler_runs_only_due_rules |

## Section 18: Database and file storage

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R61 | All named tables with keys, foreign keys, unique constraints, indexes, timestamps | implemented | database/models.py | test_anomalies.py::test_event_fingerprints_are_unique_per_rule_in_the_database |
| R62 | Large files outside SQLite, org-isolated storage, dashboard does not load datasets | implemented | core/storage.py; modules/organization_admin.py:org_dashboard | test_security.py::test_uploaded_file_name_never_reaches_storage_paths; test_org_admin.py::test_dashboard_reports_metadata_without_loading_files |
| R63 | Initialization and safe migrations (backup before upgrade) | implemented | database/__init__.py:init_database; database/migrations/__init__.py:apply_migrations, backup_database | test_scripts.py::test_initialize_database_is_idempotent; none for the backup or upgrade path (grep backup_database, apply_migrations) |

## Section 19: Memory-efficient processing (mandatory)

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R64 | Chunked reading, bounded batches, disk-backed temp processing, exact statistics, cross-chunk duplicates | implemented | services/dataset_service.py; services/versioning_service.py:iter_frame_batches, apply_streaming_operation; services/duplicate_manager.py:scan_duplicates | test_type_inference.py::test_inference_is_identical_for_every_chunk_size; test_cleaning.py::test_streaming_apply_equals_in_memory_result; test_duplicates.py::test_find_exact_duplicates_across_batches |
| R65 | Precision of financial and identifier fields preserved | implemented | services/type_inference.py:infer_types (values with more than 15 digits stay text; probe confirmed) | test_type_inference.py::test_identifiers_and_precision_are_preserved; test_datasets_service.py::test_leading_zero_identifiers_keep_their_digits |
| R66 | Merge expansion checked before allocation; timeouts, cancellation, temp cleanup, progress | implemented | services/merge_service.py:key_statistics; services/resource_manager.py:OperationGuard, CancellationToken; core/storage.py:temporary_directory | test_append_merge.py::test_append_cancellation_publishes_nothing_and_cleans_up; test_duplicates.py::test_find_exact_duplicates_honours_cancellation |

## Section 20: Streamlit interface

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R67 | Login, role navigation, all listed pages, confirmations, friendly errors | implemented | app.py; components/navigation.py:page_keys_for, PAGES; components/feedback.py:run_action | ui/test_shell.py::test_must_change_password_limits_navigation_to_account; ui/test_analysis_pages.py::test_home_renders_for_analyst; ui/test_alert_pages.py::test_rules_page_renders_for_admin_with_the_worker_note |
| R68 | Every button performs its advertised action; no decorative controls | implemented | views/platform/plans.py: every plan feature toggle, including 'Standard Plotly charts', changes behavior (enforced in services and pages) | none; test_analysis_pages.py::test_chart_tab_respects_standard_charts_plan_feature |
| R69 | Caches scoped by organization and dataset | implemented | services/profiling_service.py:_cache_get (keyed by org, version and path) | test_profiling.py::test_cache_never_skips_authorization; test_profiling.py::test_cache_returns_copies_and_follows_new_versions |

## Section 21: Configuration and Windows setup

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R70 | requirements.txt, .env.example, .gitignore, README, init script, interactive Super Admin setup, diagnostics, PowerShell, Ollama, microphone and SMTP docs | implemented | scripts/initialize_database.py; scripts/create_super_admin.py; scripts/check_environment.py; README.md | test_scripts.py::test_create_super_admin_stores_only_a_hash; test_scripts.py::test_environment_checks_report_without_failures |
| R71 | Python 3.10+ support as documented | implemented | Code avoids Python 3.11+ APIs (cleaning_service falls back to sre_parse); regex pinned to 2026.9.29 for 3.10 wheels; full suite passes on 3.10.21 and 3.13.12 in clean installs | none; test_cleaning.py::test_cleaning_service_imports_on_python_310_layout |

## Section 22: Automated testing

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R72 | pytest with isolated DB and temp folders, covering every listed category | implemented | tests/conftest.py:app_settings | test_auth.py::test_lockout_after_max_failures_rejects_correct_password; test_duplicates.py::test_duplicates_across_csv_chunks_are_found; test_datasets_service.py::test_rejects_invalid_or_unsafe_files; test_resource_limits.py::test_low_disk_rejects_imports |

## Section Addendum: Additional acceptance requirements (execution addendum)

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R73 | AppTest smoke tests for navigation, forms, callbacks and error states | implemented | tests/ui/conftest.py:make_app | ui/test_data_pages.py::test_quality_page_renders_for_roles; ui/test_platform_pages.py::test_audit_page_lists_error_events |
| R74 | End-to-end test: upload, profile, clean, version, restore, analyze, export | implemented | tests/test_e2e_workflow.py | test_e2e_workflow.py::test_upload_profile_clean_version_restore_analyze_export |
| R75 | Voice routing with a mocked microphone; Ollama available, unavailable and malformed | implemented | services/voice_service.py:record_from_server_microphone; services/llm_service.py | test_voice_routing.py::test_server_microphone_is_mocked_and_routes_to_the_offline_engine; test_llm_service.py::test_busy_semaphore_raises_ai_busy |
| R76 | Security tests: path traversal, unsafe file types, unauthorized downloads, passwords, session expiry | implemented | core/storage.py:resolve_relative; services/dataset_service.py:inspect_upload | test_storage.py::test_resolve_relative_refuses_traversal; test_security.py::test_unsafe_uploads_are_rejected_without_leftovers; test_auth.py::test_absolute_timeout_ends_an_active_session |
| R77 | Whole-valued revenue decimals inferred as float; regression cases; original unchanged | implemented | services/type_inference.py:infer_types | test_type_inference.py::test_decimal_column_is_never_downcast_even_when_whole; test_type_inference.py::test_logical_types; test_datasets_service.py::test_upload_creates_typed_version_and_preserves_original |

## Section Docs: Documentation

| ID | Requirement | Status | Implementation | Tests |
|---|---|---|---|---|
| R78 | README and SPECIFICATION reference docs/TRACEABILITY.md | implemented | docs/TRACEABILITY.md (this file), referenced from README and docs/SPECIFICATION.md | n/a |

