"""Missing-value analysis, recommendations, preview/apply treatment and placeholder handling."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import select

from core.constants import Feature, Permission
from core.errors import FeatureNotAvailableError, NotFoundError, PermissionDeniedError, ValidationError
from database import session_scope
from database.models import PlanFeature, SubscriptionPlan
from services import missing_values as mv
from services import versioning_service


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": pd.array(["C1", "C2", None, "C4", "C5", "C6"], dtype="string"),
            "region": pd.array(["North", "South", "North", None, "", "  "], dtype="string"),
            "revenue": [10.0, None, 30.0, 40.0, None, 20.0],
            "qty": pd.array([2, 3, None, 3, 2, None], dtype="Int64"),
            "active": pd.array([True, None, True, False, True, None], dtype="boolean"),
            "signup": pd.to_datetime(["2024-01-01", None, "2024-01-03", "2024-01-04", None, "2024-01-06"]),
            "notes": pd.array([None] * 6, dtype="string"),
        }
    )


def _versions(ctx, dataset_id) -> int:
    return len(versioning_service.list_versions(ctx, dataset_id))


# ---------------------------------------------------------------------------
# Analysis and recommendations
# ---------------------------------------------------------------------------


def test_analyze_counts_empty_and_whitespace_and_recommends(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    result = mv.analyze_missing(admin_ctx, dataset_id)
    columns = {c["name"]: c for c in result["columns"]}
    assert result["rows"] == 6 and result["small_dataset"]
    assert columns["region"]["missing_count"] == 3  # null, "" and "  "
    assert result["total_missing_cells"] == 1 + 3 + 2 + 2 + 2 + 2 + 6
    assert result["rows_with_missing"] == 6
    assert columns["revenue"]["recommended_strategy"] == "median"
    assert columns["revenue"]["expected_impact"] == {"cells_changed": 2, "rows_removed": 0}
    assert columns["qty"]["recommended_strategy"] == "median" and "half to even" in columns["qty"]["reason"]
    assert columns["region"]["semantic"] == "categorical" and columns["region"]["flags"]["high_missing"]
    assert columns["region"]["recommended_strategy"] == "leave"  # exactly 50 percent missing
    assert columns["active"]["recommended_strategy"] == "mode"
    assert (
        columns["customer_id"]["flags"]["identifier"]
        and columns["customer_id"]["recommended_strategy"] == "leave"
    )
    assert "never altered silently" in columns["customer_id"]["reason"]
    assert (
        columns["signup"]["recommended_strategy"] == "leave"
        and "never invented" in columns["signup"]["reason"]
    )
    notes = columns["notes"]
    assert (
        notes["flags"]["all_null"]
        and notes["flags"]["high_missing"]
        and notes["recommended_strategy"] == "leave"
    )
    assert "never automatic" in notes["reason"]
    assert all(c["flags"]["small_dataset"] for c in result["columns"])


def test_sensitive_target_high_missing_and_mixed_columns_are_left(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "salary": [100.0, None, 300.0, 400.0],
            "date_of_birth": pd.to_datetime(["1990-01-01", None, "1991-01-01", "1992-01-01"]),
            "churn": pd.array([True, None, False, True], dtype="boolean"),
            "score": [1.0, None, None, 4.0],
            "size": pd.array(["10", "two", "3", None], dtype="string"),
            "single": [None, None, 5.0, None],
            "quantity": [1.0, None, 3.0, 4.0],
            "city": pd.array(["Pune", "Delhi", None, "Pune"], dtype="string"),
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    columns = {c["name"]: c for c in mv.analyze_missing(admin_ctx, dataset_id)["columns"]}
    assert columns["salary"]["flags"]["sensitive"] and "explicitly" in columns["salary"]["reason"]
    assert columns["date_of_birth"]["flags"]["sensitive"]
    assert columns["churn"]["flags"]["target_label"] and columns["churn"]["recommended_strategy"] == "leave"
    assert columns["score"]["flags"]["high_missing"] and columns["score"]["recommended_strategy"] == "leave"
    assert columns["size"]["flags"]["mixed_types"] and columns["size"]["recommended_strategy"] == "leave"
    assert columns["single"]["flags"]["single_non_null"]
    # Whole-token matching: "quantity" is not a target "y" and "city" is not sensitive.
    assert (
        not columns["quantity"]["flags"]["target_label"]
        and columns["quantity"]["recommended_strategy"] == "median"
    )
    assert not columns["city"]["flags"]["sensitive"]
    assert columns["city"]["recommended_strategy"] == "mode" and columns["city"]["semantic"] == "categorical"


def test_recommend_plan_only_contains_safe_fills(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    plan = mv.recommend_plan(admin_ctx, dataset_id)
    assert plan == [
        {"column": "revenue", "strategy": "median", "value": None},
        {"column": "qty", "strategy": "median", "value": None},
        {"column": "active", "strategy": "mode", "value": None},
    ]
    result = mv.apply_missing_treatment(admin_ctx, dataset_id, plan)
    assert result["stats"]["cells_changed"] == 2 + 2 + 2


# ---------------------------------------------------------------------------
# Fills and exact statistics
# ---------------------------------------------------------------------------


def test_numeric_mean_and_median_are_exact(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    mean = mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "revenue", "strategy": "mean"}])
    assert mean["columns"][0]["fill_value"] == pytest.approx(25.0)  # (10 + 30 + 40 + 20) / 4
    median = mv.preview_missing_treatment(
        admin_ctx, dataset_id, [{"column": "revenue", "strategy": "median"}]
    )
    column = median["columns"][0]
    assert column["fill_value"] == 25.0 and column["missing_before"] == 2 and column["missing_after"] == 0
    assert median["cells_changed"] == 2 and median["rows_affected"] == 2 and median["rows_removed"] == 0
    assert [s["row_index"] for s in median["sample"]] == [1, 4]
    assert median["sample"][0]["before"] == {"revenue": None} and median["sample"][0]["after"] == {
        "revenue": 25.0
    }


def test_integer_columns_stay_integers_with_half_even_rounding(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    result = mv.apply_missing_treatment(admin_ctx, dataset_id, [{"column": "qty", "strategy": "mean"}])
    assert result["columns"][0]["fill_value"] == 2  # mean 2.5 rounds half to even
    assert any("rounded half to even" in w for w in result["warnings"])
    qty = versioning_service.load_frame(admin_ctx, dataset_id)["qty"]
    assert str(qty.dtype) == "Int64" and qty.tolist() == [2, 3, 2, 3, 2, 2]


def test_categorical_mode_fills_empty_and_whitespace(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    result = mv.apply_missing_treatment(admin_ctx, dataset_id, [{"column": "region", "strategy": "mode"}])
    assert result["columns"][0]["fill_value"] == "North" and result["stats"]["cells_changed"] == 3
    assert versioning_service.load_frame(admin_ctx, dataset_id)["region"].tolist() == [
        "North",
        "South",
        "North",
        "North",
        "North",
        "North",
    ]


def test_mode_tie_uses_first_value_and_warns(admin_ctx, factory):
    frame = pd.DataFrame({"tier": pd.array(["gold", "silver", None, "silver", "gold"], dtype="string")})
    dataset_id = factory.dataset(admin_ctx, frame)
    preview = mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "tier", "strategy": "mode"}])
    assert preview["columns"][0]["fill_value"] == "gold" and "equally frequent" in preview["warnings"][0]


def test_constant_placeholder_and_boolean_fills(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    plans = [
        {"column": "region", "strategy": "placeholder"},
        {"column": "revenue", "strategy": "constant", "value": "0"},
        {"column": "active", "strategy": "constant", "value": "no"},
    ]
    mv.apply_missing_treatment(admin_ctx, dataset_id, plans)
    frame = versioning_service.load_frame(admin_ctx, dataset_id)
    assert frame["region"].tolist()[3:] == ["Not Provided"] * 3
    assert frame["revenue"].tolist() == [10.0, 0.0, 30.0, 40.0, 0.0, 20.0]
    assert frame["active"].tolist() == [True, False, True, False, True, False]


def test_date_value_fill_and_invalid_dates(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    for bad in ("2024-13-45", "not a date", "", None):
        with pytest.raises(ValidationError):
            mv.preview_missing_treatment(
                admin_ctx, dataset_id, [{"column": "signup", "strategy": "date_value", "value": bad}]
            )
    with pytest.raises(ValidationError, match="cannot be used"):
        mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "signup", "strategy": "median"}])
    mv.apply_missing_treatment(
        admin_ctx, dataset_id, [{"column": "signup", "strategy": "date_value", "value": "2024-02-29"}]
    )
    signup = versioning_service.load_frame(admin_ctx, dataset_id)["signup"]
    assert signup.isna().sum() == 0 and signup.iloc[1] == pd.Timestamp("2024-02-29")


def test_ffill_requires_meaningful_row_order(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    plan = [{"column": "signup", "strategy": "ffill"}]
    with pytest.raises(ValidationError, match="row order"):
        mv.preview_missing_treatment(admin_ctx, dataset_id, plan)
    preview = mv.preview_missing_treatment(admin_ctx, dataset_id, plan, row_order_meaningful=True)
    assert preview["columns"][0]["missing_after"] == 0 and preview["cells_changed"] == 2
    leading = mv.preview_missing_treatment(
        admin_ctx, dataset_id, [{"column": "region", "strategy": "bfill"}], row_order_meaningful=True
    )
    assert leading["columns"][0]["missing_after"] == 3 and "remain missing" in leading["warnings"][0]


def test_drop_rows_and_drop_column(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    with pytest.raises(ValidationError, match="Confirm column removal"):
        mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "notes", "strategy": "drop_column"}])
    plans = [{"column": "notes", "strategy": "drop_column"}, {"column": "revenue", "strategy": "drop_rows"}]
    preview = mv.preview_missing_treatment(admin_ctx, dataset_id, plans, confirm_drop_columns=True)
    assert (
        preview["rows_removed"] == 2
        and preview["rows_after"] == 4
        and preview["columns_removed"] == ["notes"]
    )
    assert all(s["removed"] and s["after"] is None for s in preview["sample"])
    result = mv.apply_missing_treatment(admin_ctx, dataset_id, plans, confirm_drop_columns=True)
    frame = versioning_service.load_frame(admin_ctx, dataset_id)
    assert "notes" not in frame.columns and len(frame) == 4 and result["rows"] == 4
    assert versioning_service.list_operations(admin_ctx, dataset_id)[0]["rows_removed"] == 2


# ---------------------------------------------------------------------------
# Protections
# ---------------------------------------------------------------------------


def test_identifier_sensitive_and_target_protections(admin_ctx, factory):
    frame = _frame()
    frame["salary"] = [1.0, None, 3.0, 4.0, 5.0, 6.0]
    frame["label"] = pd.array(["a", None, "b", "a", "b", "a"], dtype="string")
    dataset_id = factory.dataset(admin_ctx, frame)
    with pytest.raises(ValidationError, match="identifier"):
        mv.preview_missing_treatment(
            admin_ctx, dataset_id, [{"column": "customer_id", "strategy": "mode"}], allow_sensitive=True
        )
    constant = mv.preview_missing_treatment(
        admin_ctx, dataset_id, [{"column": "customer_id", "strategy": "constant", "value": "C3"}]
    )
    assert constant["cells_changed"] == 1
    with pytest.raises(ValidationError, match="sensitive"):
        mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "salary", "strategy": "median"}])
    assert (
        mv.preview_missing_treatment(
            admin_ctx, dataset_id, [{"column": "salary", "strategy": "median"}], allow_sensitive=True
        )["cells_changed"]
        == 1
    )
    with pytest.raises(ValidationError, match="target"):
        mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "label", "strategy": "mode"}])
    assert (
        mv.preview_missing_treatment(
            admin_ctx, dataset_id, [{"column": "label", "strategy": "mode"}], allow_target=True
        )["cells_changed"]
        == 1
    )
    assert (
        mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "label", "strategy": "leave"}])[
            "cells_changed"
        ]
        == 0
    )


def test_fit_filter_uses_only_training_rows(admin_ctx, factory):
    frame = pd.DataFrame(
        {
            "split": pd.array(["train", "train", "train", "test", "test"], dtype="string"),
            "amount": [10.0, 20.0, None, 1000.0, None],
        }
    )
    dataset_id = factory.dataset(admin_ctx, frame)
    leaky = mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "amount", "strategy": "mean"}])
    assert leaky["columns"][0]["fill_value"] == pytest.approx(343.3333333)
    fit = {"column": "split", "values": ["train"]}
    result = mv.apply_missing_treatment(
        admin_ctx, dataset_id, [{"column": "amount", "strategy": "mean"}], fit_filter=fit
    )
    assert (
        result["fit_rows"] == 3
        and result["columns"][0]["fit_rows"] == 3
        and result["columns"][0]["fill_value"] == 15.0
    )
    assert versioning_service.load_frame(admin_ctx, dataset_id)["amount"].tolist() == [
        10.0,
        20.0,
        15.0,
        1000.0,
        15.0,
    ]
    with pytest.raises(ValidationError, match="No rows match"):
        mv.preview_missing_treatment(
            admin_ctx,
            dataset_id,
            [{"column": "amount", "strategy": "mean"}],
            fit_filter={"column": "split", "values": ["valid"]},
        )
    with pytest.raises(ValidationError):
        mv.preview_missing_treatment(
            admin_ctx, dataset_id, [{"column": "amount", "strategy": "mean"}], fit_filter={"column": "split"}
        )


@pytest.mark.parametrize(
    "plans",
    [
        [],
        "revenue",
        [{"column": "nope", "strategy": "mean"}],
        [{"column": "region", "strategy": "mean"}],
        [{"column": "active", "strategy": "ffill"}],
        [{"column": "revenue", "strategy": "median"}, {"column": "revenue", "strategy": "mean"}],
        [{"column": "qty", "strategy": "constant", "value": "2.5"}],
        [{"column": "qty", "strategy": "constant", "value": True}],
        [{"column": "revenue", "strategy": "constant", "value": "abc"}],
        [{"column": "revenue", "strategy": "constant"}],
        [{"column": "active", "strategy": "constant", "value": "maybe"}],
        [{"column": "region", "strategy": "constant", "value": "N/A"}],
        [{"column": "region", "strategy": "constant", "value": "   "}],
        [{"column": "notes", "strategy": "mode"}],
        [{"strategy": "mean"}],
    ],
)
def test_plan_validation_errors(admin_ctx, factory, plans):
    dataset_id = factory.dataset(admin_ctx, _frame())
    with pytest.raises(ValidationError):
        mv.preview_missing_treatment(admin_ctx, dataset_id, plans)


def test_preview_is_read_only_and_apply_is_versioned(admin_ctx, factory):
    frame = _frame()
    dataset_id = factory.dataset(admin_ctx, frame)
    before = versioning_service.load_frame(admin_ctx, dataset_id)
    mv.preview_missing_treatment(admin_ctx, dataset_id, [{"column": "revenue", "strategy": "median"}])
    assert _versions(admin_ctx, dataset_id) == 1
    pd.testing.assert_frame_equal(versioning_service.load_frame(admin_ctx, dataset_id), before)
    original = versioning_service.list_versions(admin_ctx, dataset_id)[0]["version_id"]
    result = mv.apply_missing_treatment(admin_ctx, dataset_id, [{"column": "revenue", "strategy": "median"}])
    assert result["version_number"] == 2 and result["stats"] == {
        "cells_changed": 2,
        "rows_affected": 2,
        "rows_removed": 0,
    }
    assert result["quality_after"] >= result["quality_before"]
    pd.testing.assert_frame_equal(
        versioning_service.load_frame(admin_ctx, dataset_id, version_id=original), before
    )
    operation = versioning_service.list_operations(admin_ctx, dataset_id)[0]
    assert operation["operation_type"] == "missing_values" and operation["cells_changed"] == 2
    assert operation["summary"]["columns"][0]["fill_value"] == 25.0
    assert operation["parameters"]["plans"] == [{"column": "revenue", "strategy": "median", "value": None}]


def test_failed_validation_publishes_nothing(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _frame())
    with pytest.raises(ValidationError):
        mv.apply_missing_treatment(admin_ctx, dataset_id, [{"column": "customer_id", "strategy": "mode"}])
    assert _versions(admin_ctx, dataset_id) == 1
    assert versioning_service.list_operations(admin_ctx, dataset_id)[0]["operation_type"] == "import"


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


def _placeholder_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status": pd.array(["ok", "", "  ", "N/A", " n/a ", "Unknown", None], dtype="string"),
            "code": pd.array(["0", "null", "x", "x", "x", "x", "x"], dtype="string"),
            "count": pd.array([0, 1, 2, 3, 4, 5, 6], dtype="Int64"),
            "flag": pd.array([False, True, False, True, False, True, False], dtype="boolean"),
        }
    )


def test_detect_placeholders_uses_configured_tokens_only(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _placeholder_frame())
    result = mv.detect_placeholders(admin_ctx, dataset_id)
    by_column = {c["column"]: c for c in result["columns"]}
    assert set(by_column) == {"status", "code"}  # numbers and booleans are never placeholders
    status = by_column["status"]
    assert status["empty"] == 1 and status["whitespace_only"] == 1 and status["already_missing"] == 1
    assert status["placeholder_matches"] == {"N/A": 2} and status["total"] == 4  # "Unknown" not counted
    assert by_column["code"]["placeholder_matches"] == {"null": 1}  # "0" not counted
    explicit = mv.detect_placeholders(admin_ctx, dataset_id, ["Unknown"])
    assert {c["column"]: c["placeholder_matches"] for c in explicit["columns"]}["status"] == {"Unknown": 1}
    with pytest.raises(ValidationError):
        mv.detect_placeholders(admin_ctx, dataset_id, "N/A")


def test_standardize_placeholders_converts_to_missing(admin_ctx, factory):
    dataset_id = factory.dataset(admin_ctx, _placeholder_frame())
    with pytest.raises(ValidationError):
        mv.standardize_placeholders(admin_ctx, dataset_id, ["N/A"], columns=["count"])
    result = mv.standardize_placeholders(admin_ctx, dataset_id, ["N/A", "Unknown"], columns=["status"])
    assert result["changed"] and result["cells_changed"] == 5
    frame = versioning_service.load_frame(admin_ctx, dataset_id)
    assert frame["status"].isna().tolist() == [False, True, True, True, True, True, True]
    assert frame["code"].tolist()[:2] == ["0", "null"] and frame["count"].tolist()[0] == 0
    operation = versioning_service.list_operations(admin_ctx, dataset_id)[0]
    assert operation["operation_type"] == "standardize_placeholders" and operation["cells_changed"] == 5
    assert not mv.standardize_placeholders(admin_ctx, dataset_id, ["N/A"])[
        "changed"
    ]  # only listed tokens count
    again = mv.standardize_placeholders(admin_ctx, dataset_id, ["NULL"])
    assert (
        again["changed"]
        and again["cells_changed"] == 1
        and again["columns"]["code"]["placeholder_matches"] == {"NULL": 1}
    )
    assert _versions(admin_ctx, dataset_id) == 3


# ---------------------------------------------------------------------------
# Permissions, plans and tenant isolation
# ---------------------------------------------------------------------------

PLAN = [{"column": "revenue", "strategy": "median"}]
READ_CALLS = [
    lambda ctx, d: mv.analyze_missing(ctx, d),
    lambda ctx, d: mv.recommend_plan(ctx, d),
    lambda ctx, d: mv.preview_missing_treatment(ctx, d, PLAN),
    lambda ctx, d: mv.detect_placeholders(ctx, d),
]
WRITE_CALLS = [
    lambda ctx, d: mv.apply_missing_treatment(ctx, d, PLAN),
    lambda ctx, d: mv.standardize_placeholders(ctx, d, ["N/A"]),
]


@pytest.mark.parametrize("call", READ_CALLS + WRITE_CALLS)
def test_other_tenant_gets_not_found(factory, admin_ctx, call):
    dataset_id = factory.dataset(admin_ctx, _frame())
    outsider = factory.org_admin(factory.organization())
    with pytest.raises(NotFoundError):
        call(outsider, dataset_id)


@pytest.mark.parametrize("call", READ_CALLS)
def test_granted_analyst_can_analyze(factory, org_id, admin_ctx, call):
    dataset_id = factory.dataset(admin_ctx, _frame())
    analyst_id = factory.user(org_id, "analyst")
    factory.grant(dataset_id, analyst_id)
    assert call(factory.ctx(analyst_id), dataset_id) is not None


@pytest.mark.parametrize("call", WRITE_CALLS)
def test_analyst_without_clean_permission_is_denied(factory, org_id, admin_ctx, call):
    dataset_id = factory.dataset(admin_ctx, _frame())
    analyst_id = factory.user(org_id, "analyst")
    factory.grant(dataset_id, analyst_id)
    with pytest.raises(PermissionDeniedError):
        call(factory.ctx(analyst_id), dataset_id)
    assert _versions(admin_ctx, dataset_id) == 1


def test_analyst_with_clean_permission_can_apply(factory, org_id, admin_ctx):
    dataset_id = factory.dataset(admin_ctx, _frame())
    analyst_id = factory.user(org_id, "analyst", permissions={Permission.CLEAN_DATA.value})
    factory.grant(dataset_id, analyst_id)
    assert mv.apply_missing_treatment(factory.ctx(analyst_id), dataset_id, PLAN)["version_number"] == 2


@pytest.mark.parametrize("call", READ_CALLS + WRITE_CALLS)
def test_plan_without_missing_value_feature_is_denied(factory, call):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    dataset_id = factory.dataset(ctx, _frame())
    with session_scope() as db:
        plan_id = db.execute(select(SubscriptionPlan.id).where(SubscriptionPlan.code == "basic")).scalar_one()
        feature = db.execute(
            select(PlanFeature).where(
                PlanFeature.plan_id == plan_id, PlanFeature.feature_key == Feature.MISSING_VALUES.value
            )
        ).scalar_one()
        feature.enabled = False
    with pytest.raises(FeatureNotAvailableError):
        call(ctx, dataset_id)


def test_cancel_stops_treatment_between_columns_and_publishes_nothing(admin_ctx, factory, monkeypatch):
    """Regression (SR5 follow-up): the treatment loop checks the guard before each column."""
    from core.errors import OperationCancelledError
    from services.resource_manager import CancellationToken, OperationGuard

    dataset_id = factory.dataset(admin_ctx, _frame())
    token = CancellationToken()
    fills: list[int] = []
    real_fill, real_treat = mv._apply_fill, mv.treat_frame
    guarded = {"active": False}

    def tracking_treat(*args, **kwargs):  # the apply run passes check=; the validation run does not
        guarded["active"] = kwargs.get("check") is not None
        try:
            return real_treat(*args, **kwargs)
        finally:
            guarded["active"] = False

    def fill_then_cancel(*args, **kwargs):
        if guarded["active"]:
            fills.append(1)
            token.cancel()
        return real_fill(*args, **kwargs)

    monkeypatch.setattr(mv, "treat_frame", tracking_treat)
    monkeypatch.setattr(mv, "_apply_fill", fill_then_cancel)
    plans = [{"column": "revenue", "strategy": "median"}, {"column": "region", "strategy": "mode"}]
    with pytest.raises(OperationCancelledError):
        mv.apply_missing_treatment(admin_ctx, dataset_id, plans, guard=OperationGuard(cancel_token=token))
    assert fills == [1]  # the second column was never treated
    assert _versions(admin_ctx, dataset_id) == 1
