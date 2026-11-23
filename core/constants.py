"""Shared enumerations and default plan definitions."""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    ANALYST = "analyst"
    # Internal role for background workers. Never stored on a user row and
    # never produced by a login.
    SYSTEM = "system"


USER_ROLES = (Role.SUPER_ADMIN.value, Role.ORG_ADMIN.value, Role.ANALYST.value)


class OrgStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"


class DatasetStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class Permission(str, Enum):
    """Analyst permissions an Organization Admin can grant.

    Organization Admins implicitly hold every permission. Plan features are
    checked separately: a permission never unlocks a feature the plan lacks.
    """

    UPLOAD_DATASETS = "upload_datasets"
    CLEAN_DATA = "clean_data"  # missing values, duplicates, cleaning, append/merge, restore
    EXPORT_DATA = "export_data"
    USE_VOICE = "use_voice"
    SAVE_REPORTS = "save_reports"


ANALYST_DEFAULT_PERMISSIONS = frozenset(
    {Permission.EXPORT_DATA.value, Permission.USE_VOICE.value, Permission.SAVE_REPORTS.value}
)


class Feature(str, Enum):
    """Plan features, stored per plan in the ``plan_features`` table."""

    CSV_XLSX_UPLOAD = "csv_xlsx_upload"
    DATA_PROFILING = "data_profiling"
    MISSING_VALUES = "missing_value_treatment"
    DUPLICATE_DETECTION = "duplicate_detection"
    STANDARD_CLEANING = "standard_cleaning"
    TEXT_AI_ANALYSIS = "text_ai_analysis"
    STANDARD_CHARTS = "standard_charts"
    VOICE_ANALYSIS = "voice_analysis"
    ADVANCED_CLEANING = "advanced_cleaning"
    APPEND_MERGE = "append_merge"
    ADVANCED_ANALYTICS = "advanced_analytics"
    ANOMALY_DETECTION = "anomaly_detection"
    EMAIL_ALERTS = "email_alerts"
    SAVED_REPORTS = "saved_reports"


FEATURE_LABELS: dict[str, str] = {
    Feature.CSV_XLSX_UPLOAD.value: "CSV/XLSX uploads within plan limits",
    Feature.DATA_PROFILING.value: "Data profiling",
    Feature.MISSING_VALUES.value: "Missing-value treatment",
    Feature.DUPLICATE_DETECTION.value: "Duplicate detection",
    Feature.STANDARD_CLEANING.value: "Standard cleaning",
    Feature.TEXT_AI_ANALYSIS.value: "Text-based AI analysis",
    Feature.STANDARD_CHARTS.value: "Standard Plotly charts",
    Feature.VOICE_ANALYSIS.value: "Voice analysis",
    Feature.ADVANCED_CLEANING.value: "Advanced cleaning",
    Feature.APPEND_MERGE.value: "Append and merge datasets",
    Feature.ADVANCED_ANALYTICS.value: "Advanced business analytics",
    Feature.ANOMALY_DETECTION.value: "Predictive anomaly detection",
    Feature.EMAIL_ALERTS.value: "Automated email alerts",
    Feature.SAVED_REPORTS.value: "Saved reports",
}

BASIC_FEATURES = frozenset(
    {
        Feature.CSV_XLSX_UPLOAD.value,
        Feature.DATA_PROFILING.value,
        Feature.MISSING_VALUES.value,
        Feature.DUPLICATE_DETECTION.value,
        Feature.STANDARD_CLEANING.value,
        Feature.TEXT_AI_ANALYSIS.value,
        Feature.STANDARD_CHARTS.value,
    }
)

PREMIUM_FEATURES = BASIC_FEATURES | frozenset(
    {
        Feature.VOICE_ANALYSIS.value,
        Feature.ADVANCED_CLEANING.value,
        Feature.APPEND_MERGE.value,
        Feature.ADVANCED_ANALYTICS.value,
        Feature.ANOMALY_DETECTION.value,
        Feature.EMAIL_ALERTS.value,
        Feature.SAVED_REPORTS.value,
    }
)

ALL_FEATURES = tuple(f.value for f in Feature)


class UsageMetric(str, Enum):
    ANALYSIS_QUERY = "analysis_query"
    VOICE_QUERY = "voice_query"
    UPLOAD_BYTES = "upload_bytes"
    EXPORT = "export"
    CLEANING_OPERATION = "cleaning_operation"
    LLM_CALL = "llm_call"
    ANOMALY_CHECK = "anomaly_check"


class LogicalType(str, Enum):
    """Column types the application reasons about (stored in version schemas)."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"


# pandas dtype used to store each logical type in memory and in Parquet files.
PANDAS_DTYPE_FOR = {
    LogicalType.INTEGER.value: "Int64",
    LogicalType.FLOAT.value: "float64",
    LogicalType.BOOLEAN.value: "boolean",
    LogicalType.DATETIME.value: "datetime64[ns]",
    LogicalType.TEXT.value: "string",
}

SUPPORTED_UPLOAD_EXTENSIONS = (".csv", ".xlsx", ".xls")
