"""Upload/import pipeline tests, including the reported type-inference defect.

Defect history: an earlier implementation (not present in this repository)
classified a ``revenue`` column such as 1200.00 / 980.00 / 1500.00 as integer
because every value happened to be whole. The documented contract
(``services/type_inference.py`` rule 3) is that decimal notation always means
a decimal column. These tests pin that contract through the real upload path
and assert on what is reloaded from disk, not on in-memory frames.
"""

from __future__ import annotations

import hashlib
import json
import stat

import pandas as pd
import pytest

from core import storage
from core.errors import ConflictError, FileValidationError, PermissionDeniedError, PlanLimitExceededError
from database import session_scope
from database.models import Dataset, DatasetVersion
from services import dataset_service, versioning_service

SALES_CSV = (
    "order_id,region,product,revenue,quantity,order_date\n"
    "1001,North,Widget,1200.00,3,2024-01-05\n"
    "1002,South,Gadget,980.00,2,2024-01-20\n"
    "1003,North,Widget,1500.00,5,2024-02-03\n"
    "1004,East,Gizmo,300.00,,2024-02-17\n"
    "1005,South,Gadget,760.00,2,2024-03-02\n"
)


def _upload(ctx, text: str, name: str = "sales", **kwargs):
    return dataset_service.import_dataset(
        ctx, file_name=f"{name}.csv", data=text.encode("utf-8"), dataset_name=name, **kwargs
    )


def _schema_types(dataset_id: int) -> dict[str, str]:
    with session_scope() as db:
        dataset = db.get(Dataset, dataset_id)
        version = db.get(DatasetVersion, dataset.current_version_id)
        return {c["name"]: c["logical_type"] for c in json.loads(version.schema_json)}


def test_upload_creates_typed_version_and_preserves_original(admin_ctx):
    raw_bytes = SALES_CSV.encode("utf-8")
    result = dataset_service.import_dataset(
        admin_ctx, file_name="sales.csv", data=raw_bytes, dataset_name="sales"
    )
    assert result["rows"] == 5
    assert result["columns"] == 6
    assert result["version_number"] == 1

    # Types recorded in the version schema.
    types = _schema_types(result["dataset_id"])
    assert types == {
        "order_id": "integer",
        "region": "text",
        "product": "text",
        "revenue": "float",  # whole-valued decimals stay decimal (the reported defect)
        "quantity": "integer",  # a blank value does not turn integers into floats
        "order_date": "datetime",
    }

    # Types and values reloaded from the Parquet file on disk.
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert str(frame["revenue"].dtype) == "float64"
    assert frame["revenue"].tolist() == [1200.0, 980.0, 1500.0, 300.0, 760.0]
    assert str(frame["quantity"].dtype) == "Int64"
    assert frame["quantity"].isna().tolist() == [False, False, False, True, False]
    assert str(frame["order_date"].dtype) == "datetime64[ns]"
    assert frame["region"].tolist() == ["North", "South", "North", "East", "South"]

    # The original upload is stored byte-for-byte, read-only, with a matching checksum.
    with session_scope() as db:
        dataset = db.get(Dataset, result["dataset_id"])
        original_path = storage.resolve_relative(dataset.original_file_path)
        recorded_sha = dataset.original_sha256
    assert original_path.read_bytes() == raw_bytes
    assert recorded_sha == hashlib.sha256(raw_bytes).hexdigest()
    assert not (original_path.stat().st_mode & stat.S_IWRITE)
    downloaded, filename = dataset_service.download_original(admin_ctx, result["dataset_id"])
    assert downloaded == raw_bytes and filename == "sales.csv"

    # Version 1 stays recoverable after a later change, and the original is untouched.
    def drop_first_row(df):
        return df.iloc[1:].reset_index(drop=True), {"rows_removed": 1}, {}

    versioning_service.apply_frame_operation(
        admin_ctx, result["dataset_id"], operation_type="test_drop", parameters={}, transform=drop_first_row
    )
    assert len(versioning_service.load_frame(admin_ctx, result["dataset_id"])) == 4
    versioning_service.restore_version(admin_ctx, result["dataset_id"], result["version_id"])
    restored = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    pd.testing.assert_frame_equal(restored, frame)
    assert original_path.read_bytes() == raw_bytes
    assert versioning_service.verify_original(admin_ctx, result["dataset_id"])["intact"] is True


@pytest.mark.parametrize(
    ("values", "expected_type", "expected_dtype"),
    [
        (["1", "2", "3"], "integer", "Int64"),
        (["1200.00", "980.00", "1500.00"], "float", "float64"),
        (["1200.50", "980", "1500"], "float", "float64"),
        (["1", "", "3"], "integer", "Int64"),
        (["1.5", "", "N/A"], "float", "float64"),
        (['"1,234"', '"12,500"', "7"], "integer", "Int64"),
        (['"$1,200.50"', "$980", "$15"], "float", "float64"),
        (["$1200", "$980", "$15"], "float", "float64"),
        (["00123", "00456", "01234"], "text", "string"),
        (["1234567890123456", "1234567890123457", "1"], "text", "string"),
        (["12", "abc", "14"], "text", "string"),
    ],
    ids=[
        "integer-only",
        "decimal-whole-values",
        "decimal-mixed",
        "nullable-integer",
        "nullable-decimal-with-placeholder",
        "numeric-strings-with-thousands",
        "currency-decimal",
        "currency-whole",
        "identifier-leading-zeros",
        "identifier-over-15-digits",
        "mixed-text",
    ],
)
def test_type_inference_contract_through_upload(admin_ctx, values, expected_type, expected_dtype):
    text = "label,value\n" + "".join(f"r{i},{v}\n" for i, v in enumerate(values))
    result = _upload(admin_ctx, text, name="typing")
    assert _schema_types(result["dataset_id"])["value"] == expected_type
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert str(frame["value"].dtype) == expected_dtype


def test_leading_zero_identifiers_keep_their_digits(admin_ctx):
    result = _upload(admin_ctx, "zip,city\n02134,Boston\n10001,New York\n00501,Holtsville\n", name="zips")
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert frame["zip"].tolist() == ["02134", "10001", "00501"]


def test_report_lists_malformed_rows_instead_of_dropping_silently(admin_ctx):
    text = "a,b,c\n1,2,3\n4,5\n6,7,8\n9,10,11,12\n"
    result = _upload(admin_ctx, text, name="malformed")
    report = result["import_report"]
    assert result["rows"] == 2
    assert report["malformed_rows_skipped"] == 2
    assert [item["line"] for item in report["malformed_row_examples"]] == [3, 5]
    assert any("malformed" in warning for warning in report["warnings"])


def test_malformed_rows_can_reject_the_file(admin_ctx):
    options = dataset_service.ImportOptions(csv=dataset_service.CsvOptions(malformed_rows="error"))
    with pytest.raises(FileValidationError, match="Line 3"):
        _upload(admin_ctx, "a,b,c\n1,2,3\n4,5\n", name="strict", options=options)


def test_semicolon_delimiter_and_decimal_comma(admin_ctx):
    options = dataset_service.ImportOptions(csv=dataset_service.CsvOptions(decimal_separator=","))
    result = _upload(admin_ctx, "item;price\napple;1,50\npear;2,00\n", name="eu", options=options)
    assert result["import_report"]["delimiter"] == ";"
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert frame["price"].tolist() == [1.5, 2.0]


def test_cp1252_file_is_detected(admin_ctx):
    data = "name,city\nJos\xe9,M\xfcnchen\n".encode("cp1252")
    result = dataset_service.import_dataset(admin_ctx, file_name="names.csv", data=data, dataset_name="names")
    assert result["import_report"]["encoding"] == "cp1252"
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert frame["name"].tolist() == ["Jos\xe9"]


def test_placeholders_and_legitimate_values(admin_ctx):
    text = "status,score,flag\nUnknown,0,False\nN/A,NA,\nActive,5,True\n"
    result = _upload(admin_ctx, text, name="placeholders")
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert frame["status"].tolist()[0] == "Unknown"  # a real category, not missing
    assert frame["status"].isna().tolist() == [False, True, False]
    assert frame["score"].tolist()[0] == 0  # zero is a value, not missing
    assert frame["flag"].tolist()[0] is False
    breakdown = result["import_report"]["column_types"]["status"]["missing_breakdown"]
    assert breakdown == {"N/A": 1}


def test_custom_placeholders_are_configurable(admin_ctx):
    options = dataset_service.ImportOptions(missing_tokens=("unknown",))
    result = _upload(admin_ctx, "status\nUnknown\nActive\n", name="custom", options=options)
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert frame["status"].isna().tolist() == [True, False]


def test_duplicate_and_blank_headers_are_renamed_and_reported(admin_ctx):
    result = _upload(admin_ctx, "Revenue,revenue,\n1,2,3\n", name="headers")
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert list(frame.columns) == ["Revenue", "revenue_2", "column_3"]
    reasons = {c["reason"] for c in result["import_report"]["header_changes"]}
    assert {"duplicate header", "blank header"} <= reasons


def test_header_only_file_imports_zero_rows(admin_ctx):
    result = _upload(admin_ctx, "a,b\n", name="empty_rows")
    assert result["rows"] == 0
    assert list(versioning_service.load_frame(admin_ctx, result["dataset_id"]).columns) == ["a", "b"]


@pytest.mark.parametrize(
    ("file_name", "data", "message"),
    [
        ("data.csv", b"", "empty"),
        ("data.exe", b"MZ\x90\x00", "Unsupported file type"),
        ("data.csv", b"MZ\x90\x00binary", "Executable"),
        ("data.csv", b"a,b\n1,\x00\x002\n", "binary"),
        ("data.xlsx", b"not a zip at all", "not a valid .xlsx"),
        ("data.xls", b"plain text pretending", "not a valid .xls"),
        ("data.csv", b"PK\x03\x04rest-of-zip", "Excel workbook"),
    ],
)
def test_rejects_invalid_or_unsafe_files(admin_ctx, file_name, data, message):
    with pytest.raises(FileValidationError, match=message):
        dataset_service.import_dataset(admin_ctx, file_name=file_name, data=data, dataset_name="bad")


def test_xlsx_upload_with_sheet_selection(admin_ctx, tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1, 2]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"region": ["N", "S"], "revenue": [1200.5, 980.0]}).to_excel(
            writer, sheet_name="sales", index=False
        )
    data = path.read_bytes()
    inspected = dataset_service.inspect_upload(admin_ctx, file_name="book.xlsx", data=data)
    assert inspected["sheets"] == ["first", "sales"]
    options = dataset_service.ImportOptions(sheet_name="sales")
    result = dataset_service.import_dataset(
        admin_ctx, file_name="book.xlsx", data=data, dataset_name="book", options=options
    )
    assert result["import_report"]["sheet_name"] == "sales"
    types = _schema_types(result["dataset_id"])
    assert types == {"region": "text", "revenue": "float"}


def test_xlsx_number_formats_keep_decimal_intent(admin_ctx, tmp_path):
    """Excel stores every number as a double and writes 1200.0 as "1200", so in a
    workbook the decimal signal is the cell's number format (decimals, currency,
    percent) or a fractional value. Whole numbers in the General format are
    integers, exactly like an unformatted "1200" in a CSV file."""
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["units", "price", "rate", "amount", "general_whole"])
    for values in [(3, 1200, 1, 1200.5, 1200.0), (2, 980, 2, 980.0, 980.0)]:
        sheet.append(list(values))
    for row in sheet.iter_rows(min_row=2):
        row[1].number_format = '"$"#,##0.00'  # whole numbers formatted as currency
        row[2].number_format = "0.00"  # whole numbers formatted with decimals
    path = tmp_path / "formats.xlsx"
    workbook.save(path)
    result = dataset_service.import_dataset(
        admin_ctx, file_name="formats.xlsx", data=path.read_bytes(), dataset_name="formats"
    )
    assert _schema_types(result["dataset_id"]) == {
        "units": "integer",
        "price": "float",
        "rate": "float",
        "amount": "float",
        "general_whole": "integer",
    }


def test_duplicate_dataset_name_conflicts(admin_ctx):
    _upload(admin_ctx, SALES_CSV, name="sales")
    with pytest.raises(ConflictError):
        _upload(admin_ctx, SALES_CSV, name="SALES")


def test_upload_respects_plan_size_limit(factory):
    org = factory.organization(plan="basic")
    ctx = factory.org_admin(org)
    big = b"a\n" + b"1\n" * (26 * 1024 * 1024)  # ~52 MB, above Basic's 50 MB
    with pytest.raises(PlanLimitExceededError):
        dataset_service.import_dataset(ctx, file_name="big.csv", data=big, dataset_name="big")


def test_analyst_needs_upload_permission(factory, org_id):
    analyst = factory.analyst(org_id)  # default permissions do not include uploads
    with pytest.raises(PermissionDeniedError):
        _upload(analyst, SALES_CSV)


def test_analyst_upload_grants_own_access(factory, org_id):
    analyst = factory.analyst(org_id, permissions={"upload_datasets"})
    result = _upload(analyst, SALES_CSV)
    assert [d["dataset_id"] for d in dataset_service.list_datasets(analyst)] == [result["dataset_id"]]


def test_export_escapes_formula_injection(admin_ctx):
    result = _upload(admin_ctx, 'name,amount\n=HYPERLINK("x"),5\n@SUM(A1),6\nsafe,7\n', name="inject")
    content, filename, mime = dataset_service.export_dataset(
        admin_ctx, result["dataset_id"], file_format="csv"
    )
    text = content.decode("utf-8-sig")
    assert "'=HYPERLINK" in text and "'@SUM" in text
    assert filename == "inject_v1.csv" and mime == "text/csv"
    xlsx, xname, _ = dataset_service.export_dataset(admin_ctx, result["dataset_id"], file_format="xlsx")
    exported = pd.read_excel(__import__("io").BytesIO(xlsx))
    assert exported["name"].tolist()[0] == '\'=HYPERLINK("x")'
    assert xname.endswith(".xlsx")


# --- Regression DC1: float repr decimals (Excel formula results, 0.1 + 0.2) ---


def test_csv_float_repr_decimals_import_as_float(admin_ctx):
    result = _upload(admin_ctx, "item,total\na,59.97\nb,0.30000000000000004\nc,0.7000000000000001\n", name="reprs")
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert str(frame["total"].dtype) == "float64"
    assert frame["total"].tolist() == [59.97, 0.1 + 0.2, 7 * 0.1]
    assert result["import_report"]["warnings"] == []


def test_xlsx_formula_results_import_as_float(admin_ctx):
    import io

    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["item", "qty", "price", "total"])
    for row in [("a", 3, 19.99, 3 * 19.99), ("b", 2, 5.25, 10.5), ("c", 7, 0.1, 7 * 0.1)]:
        sheet.append(list(row))
    buffer = io.BytesIO()
    workbook.save(buffer)
    result = dataset_service.import_dataset(
        admin_ctx, file_name="sales.xlsx", data=buffer.getvalue(), dataset_name="xlsx_totals"
    )
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert str(frame["qty"].dtype) == "Int64"
    assert str(frame["total"].dtype) == "float64"
    assert frame["total"].tolist() == [3 * 19.99, 10.5, 7 * 0.1]


def test_numeric_column_kept_as_text_for_precision_is_warned(admin_ctx):
    result = _upload(admin_ctx, "item,v\na,0.12345678901234567\nb,1.5\n", name="lossy")
    report = result["import_report"]
    assert report["column_types"]["v"]["logical_type"] == "text"
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert frame["v"].tolist() == ["0.12345678901234567", "1.5"]  # every digit kept
    warning = [w for w in report["warnings"] if "'v'" in w]
    assert len(warning) == 1 and "0.12345678901234567" in warning[0] and "text" in warning[0]


# --- Regression DC5: a datetime column is never reported as "kept as text" ---


def test_mixed_iso_date_and_datetime_column_is_not_reported_as_text(admin_ctx):
    result = _upload(admin_ctx, "d,v\n2024-01-15,1\n2024-01-16 10:30:00,2\n2024-02-01,3\n", name="isomix")
    report = result["import_report"]
    assert report["column_types"]["d"]["logical_type"] == "datetime"
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert str(frame["d"].dtype).startswith("datetime64")
    assert not any("'d'" in w for w in report["warnings"])


def test_really_mixed_date_formats_are_still_warned(admin_ctx):
    result = _upload(admin_ctx, "d,v\n15/01/2024,1\n2024-01-16,2\n2024-02-01,3\n", name="datemix")
    report = result["import_report"]
    assert report["column_types"]["d"]["logical_type"] == "text"
    warning = [w for w in report["warnings"] if "'d'" in w]
    assert len(warning) == 1 and "kept as text" in warning[0] and "Inconsistent date formats" in warning[0]


# --- Regression DC7: numbers outside the float range never crash the import ---


@pytest.mark.parametrize("token", ["1e400", "-1e400", "1e-400"])
def test_import_value_outside_float_range_is_kept_as_text(admin_ctx, token):
    result = _upload(admin_ctx, f"id,amt\n1,{token}\n2,5.5\n", name="huge")
    assert result["rows"] == 2
    assert result["import_report"]["column_types"]["amt"]["logical_type"] == "text"
    assert any("'amt'" in w and token in w for w in result["import_report"]["warnings"])
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert frame["amt"].tolist() == [token, "5.5"]


def test_import_accepts_unicode_digits_exactly(admin_ctx):
    result = _upload(admin_ctx, "id,amt\n1,\u0661.\u0665\n2,2.5\n", name="digits")
    frame = versioning_service.load_frame(admin_ctx, result["dataset_id"])
    assert str(frame["amt"].dtype) == "float64" and frame["amt"].tolist() == [1.5, 2.5]


# --- Regression SR6: downloading the original checks memory first ---


def test_download_original_checks_memory_before_reading(admin_ctx, monkeypatch):
    from core.errors import ResourceLimitError
    from services import resource_manager

    result = _upload(admin_ctx, SALES_CSV, name="orig")
    content, _ = dataset_service.download_original(admin_ctx, result["dataset_id"])
    assert content == SALES_CSV.encode("utf-8")
    monkeypatch.setattr(resource_manager, "memory_available_provider", lambda: 100)
    with pytest.raises(ResourceLimitError, match="Downloading the original file"):
        dataset_service.download_original(admin_ctx, result["dataset_id"])
