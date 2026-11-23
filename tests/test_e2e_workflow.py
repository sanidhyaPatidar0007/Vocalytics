"""End-to-end workflow through the real services (spec sections 7 to 13, 22 and the acceptance addendum).

Upload a CSV with real problems, profile it, treat missing values, remove duplicates, trim whitespace,
map inconsistent category spellings, inspect the version history, restore version 1, analyze, save
and export a report, export the dataset and download the untouched original. Every expected value is
computed independently with pandas from the data the services return, never hardcoded from a run.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from core import storage
from core.errors import ValidationError
from database import session_scope
from database.models import Dataset
from modules import analyst
from services import (
    analytics_engine,
    cleaning_service,
    dataset_service,
    duplicate_manager,
    missing_values,
    profiling_service,
    versioning_service,
)

# 12 data rows. Problems on purpose:
#   * row 4 has no revenue, row 6 has no quantity (missing values)
#   * row 9 is an exact copy of row 2 (exact duplicate)
#   * " North " and "South " carry leading/trailing whitespace
#   * "north" and "NORTH" are inconsistent spellings of "North"
#   * 98000.00 is a revenue outlier
#   * revenue is written with two decimals ("1200.00"), so it must be typed as float
RAW_CSV = (
    "order_id,region,product,revenue,quantity,order_date\n"
    "1,North,Widget,1200.00,3,2024-01-05\n"
    "2,South,Gadget,980.50,2,2024-01-20\n"
    "3, North ,Widget,1500.00,5,2024-02-03\n"
    "4,East,Gizmo,,1,2024-02-17\n"
    "5,north,Gadget,760.00,2,2024-03-02\n"
    "6,West,Gizmo,1100.00,,2024-03-15\n"
    "7,NORTH,Widget,1250.00,4,2024-04-01\n"
    "8,South ,Gadget,890.00,3,2024-04-12\n"
    "2,South,Gadget,980.50,2,2024-01-20\n"
    "9,East,Widget,98000.00,2,2024-05-02\n"
    "10,West,Gizmo,1010.00,6,2024-05-18\n"
    "11,South,Widget,1180.00,3,2024-06-01\n"
)


def frame(ctx, dataset_id: int, version_id: int | None = None) -> pd.DataFrame:
    return versioning_service.load_frame(ctx, dataset_id, version_id=version_id)


def assert_same(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=True
    )


def issue_types(issues: list[dict]) -> set[tuple[str, str | None]]:
    return {(issue["type"], issue["column"]) for issue in issues}


def test_upload_profile_clean_version_restore_analyze_export(admin_ctx, org_id):
    raw = RAW_CSV.encode("utf-8")

    # ------------------------------------------------------------------ import
    imported = dataset_service.import_dataset(
        admin_ctx, file_name="orders.csv", data=raw, dataset_name="orders"
    )
    ds = imported["dataset_id"]
    assert imported["version_number"] == 1 and imported["rows"] == 12 and imported["columns"] == 6
    detail = dataset_service.get_dataset(admin_ctx, ds)
    types = {c["name"]: c["logical_type"] for c in detail["schema"]}
    assert types == {
        "order_id": "integer",
        "region": "text",
        "product": "text",
        "revenue": "float",  # "1200.00" is a decimal column, never an integer one
        "quantity": "integer",
        "order_date": "datetime",
    }
    v1 = frame(admin_ctx, ds)
    independent = pd.read_csv(io.StringIO(RAW_CSV), keep_default_na=True)
    assert v1["revenue"].tolist() == pytest.approx(independent["revenue"].tolist(), nan_ok=True)
    assert v1["region"].tolist() == independent["region"].tolist()  # whitespace kept as uploaded
    assert v1["order_id"].tolist() == independent["order_id"].tolist()
    assert v1["quantity"].isna().tolist() == independent["quantity"].isna().tolist()
    assert v1["order_date"].tolist() == pd.to_datetime(independent["order_date"]).tolist()
    v1_id = imported["version_id"]

    # --------------------------------------------------------------- profiling
    profile = profiling_service.profile_dataset(admin_ctx, ds)
    confirmed = issue_types(profile["confirmed_issues"])
    suspected = issue_types(profile["suspected_issues"])
    assert ("missing_values", "revenue") in confirmed
    assert ("missing_values", "quantity") in confirmed
    assert ("duplicate_rows", None) in confirmed
    assert ("whitespace", "region") in confirmed
    # Judgement calls are suspected, never confirmed problems.
    assert ("outliers", "revenue") in suspected and ("outliers", "revenue") not in confirmed
    assert ("inconsistent_categories", "region") in suspected
    assert ("inconsistent_categories", "region") not in confirmed
    assert profile["duplicate_rows"]["count"] == int(v1.duplicated().sum()) == 1
    quality_v1 = profile["quality"]["score"]

    # ---------------------------------------------------------- missing values
    plans = [
        {"column": "revenue", "strategy": "median"},
        {"column": "quantity", "strategy": "median"},
    ]
    preview = missing_values.preview_missing_treatment(admin_ctx, ds, plans)
    assert versioning_service.list_versions(admin_ctx, ds)[0]["version_number"] == 1  # preview is read-only
    assert preview["read_only"] is True and preview["cells_changed"] == 2 and preview["rows_removed"] == 0
    fills = {c["column"]: c["fill_value"] for c in preview["columns"]}
    assert fills["revenue"] == pytest.approx(v1["revenue"].median())
    missing_values.apply_missing_treatment(admin_ctx, ds, plans)
    expected = v1.copy()
    expected["revenue"] = expected["revenue"].fillna(v1["revenue"].median())
    q_median = v1["quantity"].dropna().astype("float64").median()
    assert float(q_median).is_integer()  # an odd count of known quantities: no rounding involved
    expected["quantity"] = expected["quantity"].fillna(int(q_median)).astype("Int64")
    v2 = frame(admin_ctx, ds)
    assert_same(v2, expected)
    assert v2.isna().sum().sum() == 0

    # -------------------------------------------------------------- duplicates
    dup_preview = duplicate_manager.preview_remove_duplicates(admin_ctx, ds, keep="first")
    assert dup_preview["rows_removed"] == int(v2.duplicated().sum()) == 1
    assert (dup_preview["rows_before"], dup_preview["rows_after"]) == (12, 11)
    assert len(versioning_service.list_versions(admin_ctx, ds)) == 2  # preview published nothing
    duplicate_manager.remove_duplicates(admin_ctx, ds, keep="first", confirm=True)
    expected = v2.drop_duplicates(keep="first")
    v3 = frame(admin_ctx, ds)
    assert_same(v3, expected)
    assert len(v3) == 11

    # ---------------------------------------------------------------- cleaning
    cleaning_service.apply_cleaning(admin_ctx, ds, "trim_whitespace", {"columns": ["region"]})
    expected = v3.copy()
    expected["region"] = expected["region"].str.strip()
    v4 = frame(admin_ctx, ds)
    assert_same(v4, expected)

    mapping = {"north": "North", "NORTH": "North"}
    cleaning_service.apply_cleaning(admin_ctx, ds, "map_categories", {"column": "region", "mapping": mapping})
    expected = v4.copy()
    expected["region"] = expected["region"].replace(mapping)
    v5 = frame(admin_ctx, ds)
    assert_same(v5, expected)
    assert sorted(v5["region"].unique()) == ["East", "North", "South", "West"]

    # ------------------------------------------------------ versions / history
    versions = versioning_service.list_versions(admin_ctx, ds)
    assert [v["version_number"] for v in versions] == [5, 4, 3, 2, 1]
    assert [v["operation_type"] for v in versions] == [
        "map_categories",
        "trim_whitespace",
        "remove_duplicates",
        "missing_values",
        "import",
    ]
    assert versions[0]["is_current"] and versions[-1]["is_original_import"]
    ops = {op["operation_type"]: op for op in versioning_service.list_operations(admin_ctx, ds)}
    assert {"missing_values", "remove_duplicates", "trim_whitespace", "map_categories"} <= set(ops)
    for name, op in ops.items():
        if name == "import":
            continue
        assert op["status"] == "applied" and op["user"] == admin_ctx.username
        assert op["quality_before"] is not None and op["quality_after"] is not None
        assert op["rows_before"] is not None and op["rows_after"] is not None
    assert ops["missing_values"]["cells_changed"] == 2 and ops["missing_values"]["rows_removed"] == 0
    assert ops["remove_duplicates"]["rows_removed"] == 1
    assert (ops["remove_duplicates"]["rows_before"], ops["remove_duplicates"]["rows_after"]) == (12, 11)
    assert (
        ops["trim_whitespace"]["cells_changed"] == int((v3["region"] != v3["region"].str.strip()).sum()) == 2
    )
    assert ops["map_categories"]["cells_changed"] == 2
    assert ops["missing_values"]["quality_before"] == pytest.approx(quality_v1)
    assert ops["map_categories"]["quality_after"] >= ops["missing_values"]["quality_before"]

    # ------------------------------------------------------- restore version 1
    restored = versioning_service.restore_version(admin_ctx, ds, v1_id)
    assert restored.version_number == 6
    current = frame(admin_ctx, ds)
    assert_same(current, v1)
    assert_same(frame(admin_ctx, ds, version_id=v1_id), v1)  # version 1 itself is untouched
    assert versioning_service.list_versions(admin_ctx, ds)[0]["operation_type"] == "restore"
    assert versioning_service.verify_original(admin_ctx, ds)["intact"] is True

    # ---------------------------------------------------------------- analysis
    answer = analytics_engine.answer_question(admin_ctx, ds, "What is the total revenue?")
    assert answer["status"] == "success" and answer["planner"] == "rules"
    expected_total = float(current["revenue"].sum())  # pandas skips the missing revenue
    assert answer["facts"][0]["value"] == pytest.approx(expected_total, rel=0, abs=1e-9)
    assert float(answer["table"].iloc[0, 0]) == pytest.approx(expected_total, rel=0, abs=1e-9)
    assert answer["verification"]["verified"] is True

    # ------------------------------------------------------- report and export
    report = analyst.save_report(admin_ctx, history_id=answer["history_id"], title="Total revenue")
    stored = analyst.get_report(admin_ctx, report["report_id"])
    assert float(stored["result"].iloc[0, 0]) == pytest.approx(expected_total)
    html, html_name, html_mime = analyst.export_report(admin_ctx, report["report_id"], file_format="html")
    assert html_mime == "text/html" and html_name.endswith(".html")
    assert b"Total revenue" in html
    as_json, _, _ = analyst.export_report(admin_ctx, report["report_id"], file_format="json")
    document = json.loads(as_json)
    assert document["rows"][0][0] == pytest.approx(expected_total)
    as_csv, _, _ = analyst.export_report(admin_ctx, report["report_id"], file_format="csv")
    assert pd.read_csv(io.BytesIO(as_csv), encoding="utf-8-sig").iloc[0, 0] == pytest.approx(expected_total)

    content, name, mime = dataset_service.export_dataset(admin_ctx, ds, file_format="csv")
    assert mime == "text/csv" and name == "orders_v6.csv"
    parsed = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", keep_default_na=True)
    assert list(parsed.columns) == list(current.columns)
    assert parsed["order_id"].tolist() == current["order_id"].tolist()
    assert parsed["region"].tolist() == current["region"].tolist()
    assert parsed["product"].tolist() == current["product"].tolist()
    np.testing.assert_allclose(parsed["revenue"].to_numpy(), current["revenue"].to_numpy(), rtol=0, atol=0)
    assert parsed["quantity"].isna().tolist() == current["quantity"].isna().tolist()
    assert parsed["quantity"].dropna().astype(int).tolist() == current["quantity"].dropna().tolist()
    assert pd.to_datetime(parsed["order_date"]).tolist() == current["order_date"].tolist()

    xlsx, xlsx_name, _ = dataset_service.export_dataset(admin_ctx, ds, version_id=v1_id, file_format="xlsx")
    assert xlsx_name == "orders_v1.xlsx"
    from_xlsx = pd.read_excel(io.BytesIO(xlsx))
    assert from_xlsx["revenue"].tolist() == pytest.approx(v1["revenue"].tolist(), nan_ok=True)

    # --------------------------------------------------------- original upload
    original, original_name = dataset_service.download_original(admin_ctx, ds)
    assert original == raw and original_name == "orders.csv"
    with session_scope() as db:
        row = db.execute(select(Dataset).where(Dataset.id == ds)).scalar_one()
        assert row.original_sha256 == hashlib.sha256(raw).hexdigest()
        assert storage.resolve_relative(row.original_file_path).read_bytes() == raw


def write_workbook(path) -> bytes:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["cover sheet"], "page": [1]}).to_excel(writer, sheet_name="cover", index=False)
        pd.DataFrame(
            {
                "region": ["North", "South", " East", "north", "South", "West", "South"],
                "product": ["A", "B", "C", "A", "B", "C", "B"],
                "revenue": [1200.5, 980.0, 1500.25, None, 980.0, 1100.75, 980.0],
                "units": [3, 2, 5, 1, 2, 4, 2],
            }
        ).to_excel(writer, sheet_name="sales", index=False)
    return path.read_bytes()


def test_excel_upload_with_sheet_choice_and_profiling(admin_ctx, tmp_path):
    data = write_workbook(tmp_path / "book.xlsx")
    inspected = dataset_service.inspect_upload(admin_ctx, file_name="book.xlsx", data=data)
    assert inspected["sheets"] == ["cover", "sales"]
    assert inspected["selected_sheet"] == "cover"  # the first sheet is the default

    options = dataset_service.ImportOptions(sheet_name="sales")
    sales_view = dataset_service.inspect_upload(admin_ctx, file_name="book.xlsx", data=data, options=options)
    assert sales_view["columns"] == ["region", "product", "revenue", "units"]

    imported = dataset_service.import_dataset(
        admin_ctx, file_name="book.xlsx", data=data, dataset_name="book", options=options
    )
    assert imported["rows"] == 7 and imported["columns"] == 4
    assert imported["import_report"]["sheet_name"] == "sales"
    assert imported["import_report"]["sheets_available"] == ["cover", "sales"]
    ds = imported["dataset_id"]
    types = {c["name"]: c["logical_type"] for c in dataset_service.get_dataset(admin_ctx, ds)["schema"]}
    assert types == {"region": "text", "product": "text", "revenue": "float", "units": "integer"}

    loaded = frame(admin_ctx, ds)
    assert loaded["revenue"].sum() == pytest.approx(1200.5 + 980.0 + 1500.25 + 980.0 + 1100.75 + 980.0)

    profile = profiling_service.profile_dataset(admin_ctx, ds)
    confirmed = issue_types(profile["confirmed_issues"])
    suspected = issue_types(profile["suspected_issues"])
    assert ("missing_values", "revenue") in confirmed
    assert ("duplicate_rows", None) in confirmed
    assert profile["duplicate_rows"]["count"] == int(loaded.duplicated().sum()) == 2
    assert ("whitespace", "region") in confirmed
    assert ("inconsistent_categories", "region") in suspected

    original, name = dataset_service.download_original(admin_ctx, ds)
    assert original == data and name == "book.xlsx"

    with pytest.raises(ValidationError) as info:
        dataset_service.import_dataset(
            admin_ctx,
            file_name="book.xlsx",
            data=data,
            dataset_name="book2",
            options=dataset_service.ImportOptions(sheet_name="missing"),
        )
    assert "missing" in info.value.user_message
