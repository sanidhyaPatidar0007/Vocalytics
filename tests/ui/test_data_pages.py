"""AppTest coverage for the data pages (datasets, quality, missing values, duplicates,
cleaning, append and merge, versions) and the shared data components."""

from __future__ import annotations

import io
import sys

import openpyxl
import pandas as pd
import pytest

from core.constants import Permission
from services import cleaning_service, dataset_service, duplicate_manager, versioning_service
from tests.ui.conftest import texts

DATASETS = "views/data/datasets.py"
QUALITY = "views/data/quality.py"
MISSING = "views/data/missing_values.py"
DUPLICATES = "views/data/duplicates.py"
CLEANING = "views/data/cleaning.py"
COMBINE = "views/data/combine.py"
VERSIONS = "views/data/versions.py"

CLEANER = {Permission.CLEAN_DATA.value, Permission.EXPORT_DATA.value}
READER = {Permission.EXPORT_DATA.value}


def ok(app) -> None:
    """No uncaught exception and no unexpected-error banner (run_action / app.py reference numbers)."""
    assert not app.exception, [e.value for e in app.exception]
    unexpected = [e.value for e in app.error if "Reference" in str(e.value)]
    assert not unexpected, unexpected


def errors(app) -> str:
    return "\n".join(str(e.value) for e in app.error)


def by_key(elements, key: str):
    for element in elements:
        if getattr(element, "key", None) == key:
            return element
    raise AssertionError(f"No widget with key {key!r}; keys: {[getattr(e, 'key', None) for e in elements]}")


def labels(elements) -> list[str]:
    return [e.label for e in elements]


def frame_small() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64"),
            "region": pd.array(["North", " South", "North", "East", "South", "North"], dtype="string"),
            "revenue": [100.0, None, 300.0, 400.0, 250.0, 120.0],
        }
    )


def frame_dupes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer": pd.array(["a", "b", "a", "c", "b"], dtype="string"),
            "amount": pd.array([1, 2, 1, 3, 2], dtype="Int64"),
        }
    )


@pytest.fixture
def admin(factory, org_id):
    return factory.ctx(factory.user(org_id, "org_admin", username="dana"))


def analyst(factory, org_id, username: str, permissions, datasets=()):
    user_id = factory.user(org_id, "analyst", username=username, permissions=set(permissions))
    for dataset_id in datasets:
        factory.grant(dataset_id, user_id)
    return factory.ctx(user_id)


def versions_of(ctx, dataset_id):
    return versioning_service.list_versions(ctx, dataset_id)


# ---------------------------------------------------------------------------
# Datasets page
# ---------------------------------------------------------------------------


def test_datasets_page_renders_for_admin_and_analysts(make_app, factory, org_id, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    analyst(factory, org_id, "cleo", CLEANER, [dataset_id])
    analyst(factory, org_id, "rita", READER, [dataset_id])
    app = make_app("dana", page=DATASETS)
    ok(app)
    assert app.title[0].value == "Datasets"
    assert any(b.label == "Archive dataset" for b in app.button)
    for user in ("cleo", "rita"):
        app = make_app(user, page=DATASETS)
        ok(app)
        assert "Archive dataset" not in labels(app.button)


def test_upload_inspect_import_and_list(make_app, admin):
    app = make_app("dana", page=DATASETS)
    content = b"order_id,region,revenue\n1,North,10.5\n2,South,20\n3,East,\n"
    app.file_uploader[0].set_value(("Q1 sales.csv", content, "text/csv"))
    app.run()
    ok(app)
    app.button(key="datasets_inspect").click().run()
    ok(app)
    assert "Import options" in texts(app)
    name_input = [t for t in app.text_input if t.label == "Dataset name"][0]
    assert name_input.value == "Q1 sales"
    name_input.input("Quarter one")
    app.button(key="FormSubmitter:datasets_import_form-Import dataset").click().run()
    ok(app)
    names = [d["name"] for d in dataset_service.list_datasets(admin)]
    assert names == ["Quarter one"]
    assert "Imported Quarter one" in texts(app)
    assert "Import report" in texts(app)


def _workbook() -> bytes:
    book = openpyxl.Workbook()
    first = book.active
    first.title = "Summary"
    first.append(["note"])
    first.append(["ignore me"])
    second = book.create_sheet("Orders")
    second.append(["order_id", "amount"])
    second.append([1, 9.5])
    second.append([2, 7.25])
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_excel_upload_with_sheet_choice(make_app, admin):
    app = make_app("dana", page=DATASETS)
    app.file_uploader[0].set_value(
        ("book.xlsx", _workbook(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    )
    app.run()
    app.button(key="datasets_inspect").click().run()
    ok(app)
    sheet = [s for s in app.selectbox if s.label == "Sheet to import"][0]
    assert sheet.options == ["Summary", "Orders"]
    sheet.select("Orders").run()
    ok(app)
    app.button(key="FormSubmitter:datasets_import_form-Import dataset").click().run()
    ok(app)
    (row,) = dataset_service.list_datasets(admin)
    assert row["rows"] == 2 and row["columns"] == 2
    assert row["sheet_name"] == "Orders"


def test_malformed_csv_shows_warnings(make_app, admin):
    app = make_app("dana", page=DATASETS)
    content = b"a,b\n1,2\n3,4,5\n6,7\n"
    app.file_uploader[0].set_value(("bad.csv", content, "text/csv"))
    app.run()
    app.button(key="datasets_inspect").click().run()
    app.button(key="FormSubmitter:datasets_import_form-Import dataset").click().run()
    ok(app)
    warnings = "\n".join(str(w.value) for w in app.warning)
    assert "malformed row" in warnings
    assert dataset_service.list_datasets(admin)[0]["rows"] == 2


def test_malformed_csv_rejected_when_policy_is_error(make_app, admin):
    app = make_app("dana", page=DATASETS)
    app.file_uploader[0].set_value(("bad.csv", b"a,b\n1,2\n3,4,5\n", "text/csv"))
    app.run()
    app.button(key="datasets_inspect").click().run()
    [r for r in app.radio if r.label.startswith("Rows whose field count")][0].set_value("error")
    app.button(key="FormSubmitter:datasets_import_form-Import dataset").click().run()
    assert not app.exception
    assert errors(app)
    assert dataset_service.list_datasets(admin) == []


def test_upload_denied_without_permission(make_app, factory, org_id, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    analyst(factory, org_id, "rita", READER, [dataset_id])
    app = make_app("rita", page=DATASETS)
    ok(app)
    assert not app.file_uploader
    assert "permission to upload" in texts(app)


def test_export_prepare_then_download(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=DATASETS)
    assert not app.get("download_button")
    app.button(key=f"datasets_prepare_csv_{dataset_id}").click().run()
    ok(app)
    downloads = app.get("download_button")
    assert [d.label.replace("\\", "") for d in downloads] == ["Download sales_v1.csv"]


def test_analyst_without_export_sees_no_export_buttons(make_app, factory, org_id, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    analyst(factory, org_id, "nora", set(), [dataset_id])
    app = make_app("nora", page=DATASETS)
    ok(app)
    assert not [b for b in app.button if b.label.startswith("Prepare")]
    assert "permission to export" in texts(app)


def test_admin_rename_and_archive(make_app, factory, admin):
    factory.dataset(admin, frame_small(), "alpha")
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=DATASETS)
    app.selectbox(key="datasets_detail_select").set_value(dataset_id).run()
    by_key(app.text_input, f"datasets_rename_{dataset_id}").input("sales 2024")
    app.button(key=f"FormSubmitter:datasets_details_form_{dataset_id}-Save details").click().run()
    ok(app)
    assert dataset_service.get_dataset(admin, dataset_id)["name"] == "sales 2024"
    archive = app.button(key=f"datasets_archive_{dataset_id}")
    assert archive.disabled
    by_key(app.checkbox, f"datasets_archive_confirm_{dataset_id}").check().run()
    app.button(key=f"datasets_archive_{dataset_id}").click().run()
    ok(app)
    assert dataset_service.get_dataset(admin, dataset_id)["status"] == "archived"
    picker = app.selectbox(key="datasets_detail_select")
    assert len(picker.options) == 1 and picker.value != dataset_id
    by_key(app.checkbox, "datasets_unarchive_confirm").check().run()
    app.button(key="datasets_unarchive").click().run()
    ok(app)
    assert dataset_service.get_dataset(admin, dataset_id)["status"] == "active"


# ---------------------------------------------------------------------------
# Quality page
# ---------------------------------------------------------------------------


def test_quality_page_renders_for_roles(make_app, factory, org_id, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    analyst(factory, org_id, "cleo", CLEANER, [dataset_id])
    analyst(factory, org_id, "rita", READER, [dataset_id])
    for user in ("dana", "cleo", "rita"):
        app = make_app(user, page=QUALITY)
        ok(app)
        assert app.title[0].value == "Data quality"
        assert "Quality score" in texts(app)


def test_profile_shows_score_and_separated_issues(make_app, factory, admin):
    frame = pd.DataFrame(
        {
            "amount": [10.0, 11.0, 12.0, 10.5, 11.5, 9.5, 10.0, 11.0, 500.0, None],
            "city": pd.array(
                ["Pune", "Pune ", "Delhi", "Delhi", "Pune", "Delhi", "Pune", "Delhi", "Pune", "Delhi"],
                dtype="string",
            ),
        }
    )
    factory.dataset(admin, frame, "metrics")
    app = make_app("dana", page=QUALITY)
    app.button(key="quality_profile").click().run()
    ok(app)
    html = "\n".join(str(m.value) for m in app.markdown)
    assert 'class="vdas-score"' in html
    assert "vdas-issue-confirmed" in html and "vdas-issue-suspected" in html
    assert "Confirmed problems" in texts(app) and "Suspected issues" in texts(app)
    confirmed_block = [
        m.value
        for m in app.markdown
        if 'class="vdas-issue-confirmed"' in str(m.value) and "missing value" in str(m.value)
    ]
    suspected_block = [
        m.value
        for m in app.markdown
        if 'class="vdas-issue-suspected"' in str(m.value) and "outlier" in str(m.value)
    ]
    assert confirmed_block and suspected_block


def test_health_check_and_repeated_values(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=QUALITY)
    app.button(key="quality_health").click().run()
    ok(app)
    assert "Health check" in texts(app)
    by_key(app.multiselect, f"quality_rep_cols_{dataset_id}").select("region").run()
    app.button(key="quality_rep_run").click().run()
    ok(app)
    assert "repeated value" in texts(app)
    shown = app.dataframe[-1].value
    assert "North" in shown["region"].tolist()


def test_repeated_values_requires_a_column(make_app, factory, admin):
    factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=QUALITY)
    app.button(key="quality_rep_run").click().run()
    assert not app.exception
    assert "Select at least one column" in errors(app)


# ---------------------------------------------------------------------------
# Versions page
# ---------------------------------------------------------------------------


def _two_versions(admin, factory):
    from services import cleaning_service

    dataset_id = factory.dataset(admin, frame_small(), "sales")
    cleaning_service.apply_cleaning(admin, dataset_id, "trim_whitespace", {"columns": ["region"]})
    return dataset_id


def test_versions_page_renders_and_restores(make_app, factory, admin):
    dataset_id = _two_versions(admin, factory)
    first = [v for v in versions_of(admin, dataset_id) if v["version_number"] == 1][0]
    app = make_app("dana", page=VERSIONS)
    ok(app)
    by_key(app.selectbox, f"versions_version_{dataset_id}").set_value(first["version_id"]).run()
    ok(app)
    restore = app.button(key=f"versions_restore_{first['version_id']}")
    assert restore.disabled
    by_key(app.checkbox, f"versions_restore_confirm_{first['version_id']}").check().run()
    app.button(key=f"versions_restore_{first['version_id']}").click().run()
    ok(app)
    latest = versions_of(admin, dataset_id)[0]
    assert latest["version_number"] == 3 and latest["operation_type"] == "restore" and latest["is_current"]
    assert "Restored as version 3" in texts(app)


def test_versions_read_only_for_analyst_without_clean_data(make_app, factory, org_id, admin):
    dataset_id = _two_versions(admin, factory)
    analyst(factory, org_id, "rita", READER, [dataset_id])
    app = make_app("rita", page=VERSIONS)
    ok(app)
    first = [v for v in versions_of(admin, dataset_id) if v["version_number"] == 1][0]
    by_key(app.selectbox, f"versions_version_{dataset_id}").set_value(first["version_id"]).run()
    ok(app)
    assert not [b for b in app.button if b.label == "Restore this version"]
    assert "needs the clean data permission" in texts(app)
    app.button(key=f"versions_prepare_{first['version_id']}").click().run()
    ok(app)
    assert [d.label.replace("\\", "") for d in app.get("download_button")] == ["Download sales_v1.csv"]


def test_versions_integrity_and_history(make_app, factory, admin):
    _two_versions(admin, factory)
    app = make_app("dana", page=VERSIONS)
    app.button(key="versions_verify").click().run()
    ok(app)
    assert "no original upload" in texts(app)
    assert "Operation history" in texts(app)
    ops = [d.value for d in app.dataframe if "Operation" in d.value.columns]
    assert ops and "trim_whitespace" in ops[0]["Operation"].tolist()


# ---------------------------------------------------------------------------
# Missing values page
# ---------------------------------------------------------------------------


def test_missing_page_renders_for_admin_and_cleaner(make_app, factory, org_id, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    analyst(factory, org_id, "cleo", CLEANER, [dataset_id])
    for user in ("dana", "cleo"):
        app = make_app(user, page=MISSING)
        ok(app)
        assert app.title[0].value == "Missing values"
        table = app.dataframe[0].value
        assert table.loc[table["Column"] == "revenue", "Missing"].item() == 1


def test_recommended_plan_preview_then_apply_creates_version_2(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=MISSING)
    app.button(key=f"missing_recommend_{dataset_id}").click().run()
    ok(app)
    version_id = versions_of(admin, dataset_id)[0]["version_id"]
    assert by_key(app.selectbox, f"missing_strategy_{dataset_id}_{version_id}_revenue").value == "median"
    assert not [b for b in app.button if b.key == "missing_apply_btn"]
    app.button(key="missing_preview_btn").click().run()
    ok(app)
    assert "Preview" in texts(app)
    assert [v["version_number"] for v in versions_of(admin, dataset_id)] == [1]
    app.button(key="missing_apply_btn").click().run()
    ok(app)
    versions = versions_of(admin, dataset_id)
    assert versions[0]["version_number"] == 2 and versions[0]["operation_type"] == "missing_values"
    assert "Saved as version 2" in texts(app)


def test_missing_manual_constant_requires_valid_value(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=MISSING)
    version_id = versions_of(admin, dataset_id)[0]["version_id"]
    by_key(app.selectbox, f"missing_strategy_{dataset_id}_{version_id}_revenue").set_value("constant").run()
    by_key(app.text_input, f"missing_value_{dataset_id}_{version_id}_revenue").input("not a number").run()
    app.button(key="missing_preview_btn").click().run()
    assert not app.exception
    assert errors(app)
    assert not [b for b in app.button if b.key == "missing_apply_btn"]


def test_missing_drop_rows_needs_confirmation(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=MISSING)
    version_id = versions_of(admin, dataset_id)[0]["version_id"]
    by_key(app.selectbox, f"missing_strategy_{dataset_id}_{version_id}_revenue").set_value("drop_rows").run()
    app.button(key="missing_preview_btn").click().run()
    ok(app)
    assert app.button(key="missing_apply_btn").disabled
    by_key(app.checkbox, f"missing_apply_confirm_{dataset_id}").check().run()
    app.button(key="missing_apply_btn").click().run()
    ok(app)
    assert versions_of(admin, dataset_id)[0]["rows"] == 5


def test_placeholders_detect_and_standardize(make_app, factory, admin):
    frame = pd.DataFrame({"city": pd.array(["Pune", "N/A", "Delhi", "  ", "n/a"], dtype="string")})
    dataset_id = factory.dataset(admin, frame, "cities")
    app = make_app("dana", page=MISSING)
    app.button(key="missing_detect").click().run()
    ok(app)
    convert = [b for b in app.button if b.key == "missing_standardize"]
    assert convert and convert[0].label.startswith("Convert 3 value")
    convert[0].click().run()
    ok(app)
    assert versions_of(admin, dataset_id)[0]["operation_type"] == "standardize_placeholders"


# ---------------------------------------------------------------------------
# Duplicates page
# ---------------------------------------------------------------------------


def test_duplicates_page_renders_for_admin_and_cleaner(make_app, factory, org_id, admin):
    dataset_id = factory.dataset(admin, frame_dupes(), "customers")
    analyst(factory, org_id, "cleo", CLEANER, [dataset_id])
    for user in ("dana", "cleo"):
        app = make_app(user, page=DUPLICATES)
        ok(app)
        assert app.title[0].value == "Duplicates"
        assert [t.label for t in app.tabs] == [
            "Exact duplicates",
            "Repeated values",
            "Identifier check",
            "Composite keys",
            "Generate identifiers",
        ]


def test_duplicate_removal_requires_confirm(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_dupes(), "customers")
    app = make_app("dana", page=DUPLICATES)
    app.button(key="duplicates_exact_find").click().run()
    ok(app)
    assert "2" in [m.value for m in app.metric if m.label == "Duplicate rows"]
    app.button(key="duplicates_removal_preview").click().run()
    ok(app)
    remove = app.button(key="duplicates_remove")
    assert remove.disabled
    assert len(versions_of(admin, dataset_id)) == 1
    by_key(app.checkbox, f"duplicates_remove_confirm_{dataset_id}").check().run()
    app.button(key="duplicates_remove").click().run()
    ok(app)
    latest = versions_of(admin, dataset_id)[0]
    assert latest["version_number"] == 2 and latest["rows"] == 3
    assert "Saved as version 2" in texts(app)


def test_repeated_values_and_identifier_check(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_dupes(), "customers")
    app = make_app("dana", page=DUPLICATES)
    by_key(app.multiselect, f"duplicates_values_cols_{dataset_id}").select("customer").run()
    app.button(key="duplicates_values_find").click().run()
    ok(app)
    assert "not the same as duplicate records" in texts(app).replace("\\", "")
    by_key(app.selectbox, f"duplicates_ident_col_{dataset_id}").set_value("customer").run()
    app.button(key="duplicates_ident_run").click().run()
    ok(app)
    assert "not a valid unique identifier" in texts(app)


def test_composite_keys_resolve_with_policy(make_app, factory, admin):
    frame = pd.DataFrame(
        {
            "customer_id": pd.array([1, 1, 2], dtype="Int64"),
            "day": pd.array(["mon", "mon", "tue"], dtype="string"),
            "amount": [5.0, 6.0, 7.0],
        }
    )
    dataset_id = factory.dataset(admin, frame, "orders")
    app = make_app("dana", page=DUPLICATES)
    by_key(app.multiselect, f"duplicates_keys_cols_{dataset_id}").select("customer_id").select("day").run()
    app.button(key="duplicates_keys_find").click().run()
    ok(app)
    assert "1" in [m.value for m in app.metric if m.label == "Conflicting groups"]
    assert app.button(key="duplicates_keys_resolve").disabled
    by_key(app.selectbox, f"duplicates_keys_policy_{dataset_id}").set_value("keep_last").run()
    by_key(app.checkbox, f"duplicates_keys_confirm_{dataset_id}").check().run()
    app.button(key="duplicates_keys_resolve").click().run()
    ok(app)
    preview = versioning_service.load_preview(admin, dataset_id)
    assert preview["amount"].tolist() == [6.0, 7.0]


def test_generate_ids_shows_mapping(make_app, factory, admin):
    frame = pd.DataFrame(
        {"code": pd.array(["A1", None, "A1"], dtype="string"), "v": pd.array([1, 2, 3], dtype="Int64")}
    )
    dataset_id = factory.dataset(admin, frame, "codes")
    app = make_app("dana", page=DUPLICATES)
    by_key(app.selectbox, f"duplicates_ids_col_{dataset_id}").set_value("code").run()
    by_key(app.radio, f"duplicates_ids_mode_{dataset_id}").set_value("sequential").run()
    by_key(app.text_input, f"duplicates_ids_prefix_{dataset_id}").input("C-").run()
    app.button(key="duplicates_ids_preview").click().run()
    ok(app)
    by_key(app.checkbox, f"duplicates_ids_confirm_{dataset_id}").check().run()
    app.button(key="duplicates_ids_apply").click().run()
    ok(app)
    assert versioning_service.load_preview(admin, dataset_id)["code"].tolist() == ["A1", "C-0001", "C-0002"]
    mapping = [d.value for d in app.dataframe if "New value" in d.value.columns]
    assert mapping and mapping[0]["New value"].tolist() == ["C-0001", "C-0002"]


def test_generate_ids_locked_on_basic_plan(make_app, factory):
    basic = factory.organization("Basic Co", plan="basic")
    ctx = factory.ctx(factory.user(basic, "org_admin", username="bea"))
    factory.dataset(ctx, frame_dupes(), "customers")
    app = make_app("bea", page=DUPLICATES)
    ok(app)
    assert "advanced cleaning, which your plan does not include" in texts(app)
    assert not [b for b in app.button if b.key == "duplicates_ids_preview"]


# ---------------------------------------------------------------------------
# Cleaning page
# ---------------------------------------------------------------------------


def test_cleaning_page_renders_for_admin_and_cleaner(make_app, factory, org_id, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    analyst(factory, org_id, "cleo", CLEANER, [dataset_id])
    for user in ("dana", "cleo"):
        app = make_app(user, page=CLEANING)
        ok(app)
        assert app.title[0].value == "Cleaning"


def test_cleaning_trim_preview_then_apply(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=CLEANING)
    assert app.selectbox(key="cleaning_operation").value == "trim_whitespace"
    by_key(app.multiselect, f"cleaning_trim_whitespace_columns_{dataset_id}").select("region").run()
    app.button(key="cleaning_preview_btn").click().run()
    ok(app)
    assert "1" in [m.value for m in app.metric if m.label == "Cells changed"]
    app.button(key="cleaning_apply_btn").click().run()
    ok(app)
    latest = versions_of(admin, dataset_id)[0]
    assert latest["version_number"] == 2 and latest["operation_type"] == "trim_whitespace"
    assert versioning_service.load_preview(admin, dataset_id)["region"].tolist()[1] == "South"
    assert "Saved as version 2" in texts(app)


def test_cleaning_destructive_operation_needs_confirm(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=CLEANING)
    app.selectbox(key="cleaning_operation").set_value("drop_columns").run()
    by_key(app.multiselect, f"cleaning_drop_columns_columns_{dataset_id}").select("revenue").run()
    app.button(key="cleaning_preview_btn").click().run()
    ok(app)
    assert app.button(key="cleaning_apply_btn").disabled
    by_key(app.checkbox, f"cleaning_confirm_{dataset_id}_drop_columns").check().run()
    app.button(key="cleaning_apply_btn").click().run()
    ok(app)
    assert versions_of(admin, dataset_id)[0]["columns"] == 2


def test_cleaning_rename_input_error_and_constraints_report(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=CLEANING)
    app.selectbox(key="cleaning_operation").set_value("rename_columns").run()
    by_key(app.text_area, f"cleaning_rename_columns_mapping_{dataset_id}").input("no equals sign").run()
    app.button(key="cleaning_preview_btn").click().run()
    assert not app.exception
    assert "old=new" in errors(app)
    app.selectbox(key="cleaning_operation").set_value("validate_constraints").run()
    by_key(app.selectbox, f"cleaning_rule_column_{dataset_id}").set_value("revenue").run()
    by_key(app.checkbox, f"cleaning_rule_notnull_{dataset_id}").check().run()
    by_key(app.text_input, f"cleaning_rule_min_{dataset_id}").input("150").run()
    app.button(key="cleaning_rule_add").click().run()
    app.button(key="cleaning_preview_btn").click().run()
    ok(app)
    assert "violation" in texts(app)
    assert not [b for b in app.button if b.key == "cleaning_apply_btn"]


def test_cleaning_reports(make_app, factory, admin):
    frame = pd.DataFrame(
        {
            "city": pd.array(["Pune", "pune", "Delhi"], dtype="string"),
            "copy": pd.array(["Pune", "pune", "Delhi"], dtype="string"),
            "n": [1.0, 2.0, 50.0],
        }
    )
    factory.dataset(admin, frame, "cities")
    app = make_app("dana", page=CLEANING)
    app.button(key="cleaning_casing").click().run()
    app.button(key="cleaning_redundant").click().run()
    app.button(key="cleaning_outliers").click().run()
    ok(app)
    shown = [d.value for d in app.dataframe]
    assert any("Suggested spelling" in d.columns for d in shown)
    assert any("Finding" in d.columns and "identical" in d["Finding"].tolist() for d in shown)
    assert "Possible outliers" in [m.label for m in app.metric]


def test_basic_plan_locks_advanced_cleaning(make_app, factory):
    basic = factory.organization("Basic Co", plan="basic")
    ctx = factory.ctx(factory.user(basic, "org_admin", username="bea"))
    factory.dataset(ctx, frame_small(), "sales")
    app = make_app("bea", page=CLEANING)
    ok(app)
    operation = app.selectbox(key="cleaning_operation")
    assert any("locked" in o for o in operation.options)
    operation.set_value("map_categories").run()
    ok(app)
    assert "Advanced cleaning is not included in your plan" in texts(app)
    assert not [b for b in app.button if b.key == "cleaning_preview_btn"]


# ---------------------------------------------------------------------------
# Append and merge page
# ---------------------------------------------------------------------------


def _orders_and_more(factory, admin):
    first = pd.DataFrame({"id": pd.array([1, 2], dtype="Int64"), "amount": [10.0, 20.0]})
    second = pd.DataFrame({"id": pd.array([2, 3], dtype="Int64"), "amount": [20.0, 30.0]})
    a = factory.dataset(admin, first, "jan")
    b = factory.dataset(admin, second, "feb")
    return a, b


def _pick(app, key: str, dataset_id: int) -> None:
    app.selectbox(key=key).set_value(dataset_id).run()


def test_combine_page_renders_for_admin_and_cleaner(make_app, factory, org_id, admin):
    a, b = _orders_and_more(factory, admin)
    analyst(factory, org_id, "cleo", CLEANER, [a, b])
    for user in ("dana", "cleo"):
        app = make_app(user, page=COMBINE)
        ok(app)
        assert app.title[0].value == "Append and merge"
        # The two sides start on different datasets; the guard appears only for the same one twice.
        assert "Choose two different datasets" not in texts(app)
        _pick(app, "combine_append_source", app.selectbox(key="combine_append_target").value)
        ok(app)
        assert "Choose two different datasets" in texts(app)


def test_append_preview_then_apply(make_app, factory, admin):
    a, b = _orders_and_more(factory, admin)
    app = make_app("dana", page=COMBINE)
    _pick(app, "combine_append_target", a)
    _pick(app, "combine_append_source", b)
    app.button(key="combine_append_analyze").click().run()
    ok(app)
    assert "4" in [m.value for m in app.metric if m.label == "Rows after appending"]
    app.button(key="combine_append_preview").click().run()
    ok(app)
    assert "1" in [m.value for m in app.metric if m.label == "Possible duplicate rows"]
    assert app.button(key="combine_append_apply").disabled
    app.checkbox(key="combine_append_confirm").check().run()
    app.button(key="combine_append_apply").click().run()
    ok(app)
    latest = versions_of(admin, a)[0]
    assert latest["version_number"] == 2 and latest["rows"] == 4 and latest["operation_type"] == "append"
    assert "Appended and saved as version 2" in texts(app)


def test_merge_analyze_then_apply_as_new_dataset(make_app, factory, admin):
    a, b = _orders_and_more(factory, admin)
    app = make_app("dana", page=COMBINE)
    _pick(app, "combine_merge_left", a)
    _pick(app, "combine_merge_right", b)
    app.multiselect(key=f"combine_merge_left_on_{a}").select("id").run()
    app.multiselect(key=f"combine_merge_right_on_{b}").select("id").run()
    app.button(key="combine_merge_analyze").click().run()
    ok(app)
    assert "1" in [m.value for m in app.metric if m.label == "Output rows (exact)"]
    app.button(key="combine_merge_preview").click().run()
    ok(app)
    app.text_input(key="combine_merge_name").input("jan feb joined").run()
    app.checkbox(key="combine_merge_confirm").check().run()
    app.button(key="combine_merge_apply").click().run()
    ok(app)
    names = {d["name"]: d for d in dataset_service.list_datasets(admin)}
    assert names["jan feb joined"]["rows"] == 1
    assert "new dataset jan feb joined" in texts(app)


def test_merge_with_unequal_keys_is_blocked(make_app, factory, admin):
    a, b = _orders_and_more(factory, admin)
    app = make_app("dana", page=COMBINE)
    _pick(app, "combine_merge_left", a)
    _pick(app, "combine_merge_right", b)
    app.multiselect(key=f"combine_merge_left_on_{a}").select("id").select("amount").run()
    app.multiselect(key=f"combine_merge_right_on_{b}").select("id").run()
    ok(app)
    assert app.button(key="combine_merge_analyze").disabled
    assert "same number of key columns" in texts(app)


def test_combine_not_offered_on_basic_plan(make_app, factory):
    from components.navigation import load_capabilities, page_keys_for

    basic = factory.organization("Basic Co", plan="basic")
    ctx = factory.ctx(factory.user(basic, "org_admin", username="bea"))
    keys = page_keys_for(load_capabilities(ctx))
    assert "combine" not in keys["Data"]
    assert "cleaning" in keys["Data"]


def test_many_to_many_merge_needs_extra_confirmation(make_app, factory, admin):
    left = pd.DataFrame({"k": pd.array([1, 1], dtype="Int64"), "x": [1.0, 2.0]})
    right = pd.DataFrame({"k": pd.array([1, 1], dtype="Int64"), "y": [3.0, 4.0]})
    a = factory.dataset(admin, left, "left side")
    b = factory.dataset(admin, right, "right side")
    app = make_app("dana", page=COMBINE)
    _pick(app, "combine_merge_left", a)
    _pick(app, "combine_merge_right", b)
    app.multiselect(key=f"combine_merge_left_on_{a}").select("k").run()
    app.multiselect(key=f"combine_merge_right_on_{b}").select("k").run()
    app.button(key="combine_merge_analyze").click().run()
    ok(app)
    assert "Many-to-many" in errors(app)
    app.checkbox(key="combine_merge_confirm").check().run()
    assert app.button(key="combine_merge_apply").disabled
    app.checkbox(key="combine_merge_confirm_m2m").check().run()
    app.button(key="combine_merge_apply").click().run()
    ok(app)
    assert max(d["rows"] for d in dataset_service.list_datasets(admin)) == 4


def test_issue_html_escapes_dataset_text(make_app, factory, admin):
    frame = pd.DataFrame({"<img src=x onerror=alert(1)>": [1.0, None, 3.0]})
    factory.dataset(admin, frame, "hostile")
    app = make_app("dana", page=QUALITY)
    app.button(key="quality_profile").click().run()
    ok(app)
    html = "\n".join(str(m.value) for m in app.markdown if "vdas-issue" in str(m.value))
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<img" not in html


def test_versions_restore_as_analyst_with_clean_data(make_app, factory, org_id, admin):
    dataset_id = _two_versions(admin, factory)
    analyst(factory, org_id, "cleo", CLEANER, [dataset_id])
    first = [v for v in versions_of(admin, dataset_id) if v["version_number"] == 1][0]
    app = make_app("cleo", page=VERSIONS)
    ok(app)
    by_key(app.selectbox, f"versions_version_{dataset_id}").set_value(first["version_id"]).run()
    by_key(app.checkbox, f"versions_restore_confirm_{first['version_id']}").check().run()
    app.button(key=f"versions_restore_{first['version_id']}").click().run()
    ok(app)
    assert versions_of(admin, dataset_id)[0]["operation_type"] == "restore"


def test_every_cleaning_operation_form_renders(make_app, factory, admin):
    from services import cleaning_service

    frame = pd.DataFrame(
        {
            "city": pd.array(["Pune", "pune", "Delhi", "Delhi"], dtype="string"),
            "amount": [1.0, 2.0, 3.0, 400.0],
            "qty": pd.array([1, 2, 3, 4], dtype="Int64"),
        }
    )
    factory.dataset(admin, frame, "mixed")
    app = make_app("dana", page=CLEANING)
    for operation in cleaning_service.list_operations():
        app.selectbox(key="cleaning_operation").set_value(operation["name"]).run()
        ok(app)
        assert any(b.key == "cleaning_preview_btn" for b in app.button), operation["name"]


def test_advanced_cleaning_previews(make_app, factory, admin):
    frame = pd.DataFrame(
        {
            "city": pd.array(["Pune", "pune", "Delhi", "Delhi"], dtype="string"),
            "amount": [1.0, 2.0, 3.0, 400.0],
            "qty": pd.array([1, 2, 3, 4], dtype="Int64"),
        }
    )
    dataset_id = factory.dataset(admin, frame, "mixed")
    app = make_app("dana", page=CLEANING)
    app.selectbox(key="cleaning_operation").set_value("map_categories").run()
    by_key(app.text_area, f"cleaning_map_categories_mapping_{dataset_id}").input("pune=Pune").run()
    app.button(key="cleaning_preview_btn").click().run()
    ok(app)
    assert "1" in [m.value for m in app.metric if m.label == "Cells changed"]

    app.selectbox(key="cleaning_operation").set_value("handle_outliers").run()
    by_key(app.selectbox, f"cleaning_handle_outliers_action_{dataset_id}").set_value("cap").run()
    app.button(key="cleaning_preview_btn").click().run()
    ok(app)
    assert "1" in [m.value for m in app.metric if m.label == "Cells changed"]
    app.button(key="cleaning_apply_btn").click().run()
    ok(app)
    assert versions_of(admin, dataset_id)[0]["operation_type"] == "handle_outliers"

    app.selectbox(key="cleaning_operation").set_value("convert_types").run()
    by_key(app.selectbox, f"cleaning_convert_types_column_{dataset_id}").set_value("qty").run()
    by_key(app.selectbox, f"cleaning_convert_types_target_{dataset_id}").set_value("float").run()
    app.button(key="cleaning_preview_btn").click().run()
    ok(app)
    assert "4" in [m.value for m in app.metric if m.label == "Cells changed"]


def test_merge_apply_as_analyst(make_app, factory, org_id, admin):
    a, b = _orders_and_more(factory, admin)
    analyst(factory, org_id, "cleo", CLEANER, [a, b])
    app = make_app("cleo", page=COMBINE)
    _pick(app, "combine_merge_left", a)
    _pick(app, "combine_merge_right", b)
    app.multiselect(key=f"combine_merge_left_on_{a}").select("id").run()
    app.multiselect(key=f"combine_merge_right_on_{b}").select("id").run()
    app.selectbox(key="combine_merge_how").set_value("outer").run()
    app.button(key="combine_merge_analyze").click().run()
    ok(app)
    assert "3" in [m.value for m in app.metric if m.label == "Output rows (exact)"]
    app.checkbox(key="combine_merge_confirm").check().run()
    app.button(key="combine_merge_apply").click().run()
    ok(app)
    assert "new dataset" in texts(app)
    assert any(d["rows"] == 3 for d in dataset_service.list_datasets(admin))


def test_combine_pickers_start_on_different_datasets(make_app, factory, admin):
    """Regression: the second picker on the page no longer shares the remembered choice,
    so append source and merge right side start on a different dataset."""
    a, b = _orders_and_more(factory, admin)
    app = make_app("dana", page=COMBINE)
    ok(app)
    assert app.selectbox(key="combine_append_target").value != app.selectbox(key="combine_append_source").value
    assert app.selectbox(key="combine_merge_left").value != app.selectbox(key="combine_merge_right").value
    assert {
        app.selectbox(key="combine_append_target").value,
        app.selectbox(key="combine_append_source").value,
    } == {a, b}


# ---------------------------------------------------------------------------
# Stale read-only reports, widget state and picker regressions
# ---------------------------------------------------------------------------


def test_identifier_check_is_not_shown_for_an_older_version(make_app, factory, admin):
    frame = pd.DataFrame(
        {
            "id": pd.array([1, 2, 2, 3, 4, 5], dtype="Int64"),
            "v": pd.array([1, 2, 2, 3, 4, 500], dtype="Int64"),
        }
    )
    dataset_id = factory.dataset(admin, frame, "ids")
    app = make_app("dana", page=DUPLICATES)
    by_key(app.selectbox, f"duplicates_ident_col_{dataset_id}").set_value("id").run()
    app.button(key="duplicates_ident_run").click().run()
    ok(app)
    assert "not a valid unique identifier" in texts(app)
    app.button(key="duplicates_exact_find").click().run()
    app.button(key="duplicates_removal_preview").click().run()
    by_key(app.checkbox, f"duplicates_remove_confirm_{dataset_id}").check().run()
    app.button(key="duplicates_remove").click().run()
    ok(app)
    assert versions_of(admin, dataset_id)[0]["version_number"] == 2
    assert "Saved as version 2" in texts(app)
    # The check was computed on version 1; version 2 has unique ids.
    assert "not a valid unique identifier" not in texts(app)


def test_repeated_values_report_is_not_shown_for_an_older_version(make_app, factory, admin):
    frame = pd.DataFrame({"v": pd.array([1, 2, 2, 3, 4, 5], dtype="Int64")})
    dataset_id = factory.dataset(admin, frame, "reps")
    app = make_app("dana", page=DUPLICATES)
    by_key(app.multiselect, f"duplicates_values_cols_{dataset_id}").set_value(["v"]).run()
    app.button(key="duplicates_values_find").click().run()
    ok(app)
    assert [m.value for m in app.metric if m.label == "Repeated values"] == ["1"]
    duplicate_manager.remove_duplicates(admin, dataset_id, keep="first", subset=None, confirm=True)
    assert versions_of(admin, dataset_id)[0]["rows"] == 5
    app.run()
    ok(app)
    assert not [m for m in app.metric if m.label == "Repeated values"]


def test_outlier_report_is_not_shown_for_an_older_version(make_app, factory, admin):
    frame = pd.DataFrame({"v": pd.array([1, 2, 2, 3, 4, 500, 2, 3, 1, 2], dtype="Int64")})
    dataset_id = factory.dataset(admin, frame, "nums")
    app = make_app("dana", page=CLEANING)
    app.button(key="cleaning_outliers").click().run()
    ok(app)
    assert [m.value for m in app.metric if m.label == "Values checked"] == ["10"]
    cleaning_service.apply_cleaning(
        admin,
        dataset_id,
        "handle_outliers",
        {"column": "v", "method": "iqr", "action": "exclude"},
        confirm=True,
    )
    assert versions_of(admin, dataset_id)[0]["rows"] == 9
    app.run()
    ok(app)
    assert not [m for m in app.metric if m.label == "Values checked"]


def test_quality_repeated_values_are_not_shown_for_an_older_version(make_app, factory, admin):
    dataset_id = factory.dataset(admin, frame_small(), "sales")
    app = make_app("dana", page=QUALITY)
    by_key(app.multiselect, f"quality_rep_cols_{dataset_id}").select("region").run()
    app.button(key="quality_rep_run").click().run()
    ok(app)
    assert "repeated value(s)" in texts(app)
    cleaning_service.apply_cleaning(admin, dataset_id, "trim_whitespace", {"columns": ["region"]})
    app.run()
    ok(app)
    assert "repeated value(s)" not in texts(app)


def test_missing_editor_does_not_carry_plan_to_other_columns_after_apply(make_app, factory, admin):
    frame = pd.DataFrame(
        {
            "a": pd.array(["x", None, "z", "w"], dtype="string"),
            "b": [1.0, None, 3.0, 4.0],
            "c": [5.0, None, 7.0, 8.0],
        }
    )
    factory.dataset(admin, frame, "sales")
    app = make_app("dana", page=MISSING)

    def strategies():
        return [s for s in app.selectbox if s.label == "Strategy"]

    def checkbox(label):
        return next(c for c in app.checkbox if c.label == label)

    strategies()[0].set_value("drop_column").run()  # column a
    checkbox("Confirm dropping columns").check().run()
    app.button(key="missing_preview_btn").click().run()
    ok(app)
    assert "Columns that will be removed: a" in texts(app)
    next(c for c in app.checkbox if c.label.startswith("I understand")).check().run()
    app.button(key="missing_apply_btn").click().run()
    ok(app)
    assert "Saved as version 2" in texts(app)

    # Version 2 has columns b and c: nothing chosen for column a may follow them.
    assert [s.value for s in strategies()] == ["leave", "leave"]
    assert checkbox("Confirm dropping columns").value is False
    app.button(key="missing_preview_btn").click().run()
    ok(app)
    assert "Columns that will be removed" not in texts(app)


def test_combine_remembers_the_dataset_the_user_last_chose(make_app, factory, admin):
    """Regression: the untouched merge left picker must not overwrite the shared choice."""
    a, b = _orders_and_more(factory, admin)
    app = make_app("dana", page=COMBINE)
    ok(app)
    first = app.selectbox(key="combine_append_target").value
    other = b if first == a else a
    _pick(app, "combine_append_target", other)
    ok(app)
    assert app.session_state["_vdas_selected_dataset"] == other
    assert app.selectbox(key="combine_merge_left").value == first


def test_sample_tables_keep_row_numbers_when_a_column_is_named_row(make_app, factory, admin):
    frame = pd.DataFrame(
        {
            "row": pd.array([101, 102, 103, 104], dtype="Int64"),
            "name": pd.array([" a", "b ", " c", "d"], dtype="string"),
        }
    )
    dataset_id = factory.dataset(admin, frame, "named_row")
    app = make_app("dana", page=CLEANING)
    by_key(app.multiselect, f"cleaning_trim_whitespace_columns_{dataset_id}").select("name").run()
    app.button(key="cleaning_preview_btn").click().run()
    ok(app)
    samples = [d.value for d in app.dataframe if "name" in d.value.columns and "row" in d.value.columns]
    assert samples
    for table in samples:
        assert table["row"].tolist() == sorted(set(table["row"].tolist()))
        assert not set(table["row"].tolist()) & {101, 102, 103, 104}
    before = samples[0]
    assert before["row (column)"].tolist() == [101, 102, 103]


def test_attached_upload_is_not_read_again_on_every_rerun(make_app, admin, monkeypatch):
    from streamlit.runtime.uploaded_file_manager import UploadedFile

    reads = []
    original = UploadedFile.getvalue

    def counting(self):
        # The AppTest harness itself reads the file on every run; count only reads by the page.
        if sys._getframe(1).f_code.co_filename.endswith("datasets.py"):
            reads.append(self.name)
        return original(self)

    monkeypatch.setattr(UploadedFile, "getvalue", counting)
    app = make_app("dana", page=DATASETS)
    first = b"order_id,region\n1,North\n2,South\n"
    app.file_uploader[0].set_value(("sales.csv", first, "text/csv"))
    app.run()
    ok(app)
    app.run()
    app.run()
    ok(app)
    assert reads == []  # the file is read only when inspecting or importing, never to identify it
    app.button(key="datasets_inspect").click().run()
    ok(app)
    assert reads
    assert "Import options" in texts(app)
    # A different file with the same name and size is a different upload: it must not reuse the inspection.
    app.file_uploader[0].set_value(("sales.csv", b"order_id,region\n7,East\n8,West\n", "text/csv"))
    app.run()
    ok(app)
    assert "Import options" not in texts(app)


def test_generate_ids_says_when_the_mapping_is_truncated(make_app, factory, admin, monkeypatch):
    monkeypatch.setattr(duplicate_manager, "MAPPING_CAP", 1)
    frame = pd.DataFrame(
        {"code": pd.array(["A1", None, None], dtype="string"), "v": pd.array([1, 2, 3], dtype="Int64")}
    )
    dataset_id = factory.dataset(admin, frame, "codes")
    app = make_app("dana", page=DUPLICATES)
    by_key(app.selectbox, f"duplicates_ids_col_{dataset_id}").set_value("code").run()
    by_key(app.radio, f"duplicates_ids_mode_{dataset_id}").set_value("sequential").run()
    app.button(key="duplicates_ids_preview").click().run()
    ok(app)
    by_key(app.checkbox, f"duplicates_ids_confirm_{dataset_id}").check().run()
    app.button(key="duplicates_ids_apply").click().run()
    ok(app)
    page = texts(app)
    assert "Only the first 1 changed identifiers are listed" in page
    assert "previous version still holds the original values" in page
    assert "full list is kept" not in page
