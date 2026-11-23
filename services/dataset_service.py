"""Dataset upload, import, listing, export and archiving.

Import pipeline (memory bounded, exact):

1. Validate the upload: extension allowlist, size against the plan, magic
   bytes (an .xlsx must be a zip workbook without macros, an .xls an OLE2
   file, a .csv must be text), zip-bomb guard for workbooks.
2. Parse the file once into raw text chunks (CSV via the csv module with
   malformed-row detection; Excel via openpyxl read-only / xlrd) and spool them
   to a temporary all-text Parquet file.
3. Pass 1 over the spool infers every column type exactly
   (``services.type_inference``); pass 2 converts the values and writes the
   typed Parquet file with a fixed schema.
4. The original upload is stored unchanged (read-only, with a SHA-256
   checksum) and the typed file becomes version 1, in one transaction.

Rows are never dropped silently: malformed CSV rows, blank rows and cells
outside the header are counted and listed in the import report.
"""

from __future__ import annotations

import codecs
import csv
import io
import json
import math
import re
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pyarrow as pa
import xlrd
from openpyxl.cell import WriteOnlyCell
from sqlalchemy import func, select

from auth.authorization import AuthContext, authorize, authorize_dataset, visible_datasets_query
from config.settings import get_settings
from core import storage
from core.constants import SUPPORTED_UPLOAD_EXTENSIONS, DatasetStatus, Feature, Permission, Role, UsageMetric
from core.errors import ConflictError, FileValidationError, NotFoundError, ValidationError
from database.connection import session_scope
from database.models import Dataset, DatasetVersion, User
from modules import org_settings, subscriptions, usage_tracking
from services import resource_manager
from services.audit_service import record_audit
from services.frame_utils import escape_spreadsheet_formulas
from services.resource_manager import CancellationToken, ProgressCallback, make_guard
from services.type_inference import (
    INCONSISTENT_DATES_REASON,
    PRECISION_TEXT_REASON,
    ParseOptions,
    arrow_schema,
    convert_frame,
    infer_types,
    normalize_headers,
)
from services.versioning_service import (
    ParquetBatchWriter,
    create_dataset_with_initial_version,
    get_version_ref,
    iter_parquet,
    validate_dataset_name,
)

MB = 1024 * 1024
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"
MAX_CSV_FIELD_CHARS = 10 * 1024 * 1024
MAX_REPORTED_ROWS = 20
EXCEL_MAX_ROWS = 1_048_575
_DECIMAL_NUMBER_FORMAT = re.compile(r"0\.0|0\.#|#\.0|#\.#|[$€£₹¥]|\[\$|%")

csv.field_size_limit(MAX_CSV_FIELD_CHARS)


# ---------------------------------------------------------------------------
# Options and report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CsvOptions:
    encoding: str | None = None  # None = detect (UTF-8, then Windows-1252, then Latin-1)
    delimiter: str | None = None  # None = detect among , ; tab |
    decimal_separator: str = "."
    thousands_separator: str | None = None
    malformed_rows: str = "skip"  # "skip" = skip and report, "error" = reject the file

    def __post_init__(self) -> None:
        if self.malformed_rows not in {"skip", "error"}:
            raise ValidationError("Malformed-row handling must be 'skip' or 'error'.")
        if self.delimiter is not None and self.delimiter not in {",", ";", "\t", "|"}:
            raise ValidationError("Delimiter must be comma, semicolon, tab or pipe.")
        if self.encoding is not None:
            try:
                codecs.lookup(self.encoding)
            except LookupError as exc:
                raise ValidationError(f"Unknown encoding '{self.encoding}'.") from exc


@dataclass(frozen=True)
class ImportOptions:
    csv: CsvOptions = field(default_factory=CsvOptions)
    sheet_name: str | None = None
    missing_tokens: tuple[str, ...] | None = None  # None = organization default
    dayfirst: bool | None = None  # None = organization default

    def to_dict(self) -> dict[str, Any]:
        return {
            "csv": asdict(self.csv),
            "sheet_name": self.sheet_name,
            "missing_tokens": list(self.missing_tokens) if self.missing_tokens is not None else None,
            "dayfirst": self.dayfirst,
        }


@dataclass
class ImportReport:
    file_type: str
    file_size_bytes: int
    encoding: str | None = None
    delimiter: str | None = None
    decimal_separator: str = "."
    thousands_separator: str | None = None
    sheet_name: str | None = None
    sheets_available: list[str] = field(default_factory=list)
    rows_imported: int = 0
    columns: int = 0
    malformed_rows_skipped: int = 0
    malformed_row_examples: list[dict[str, int]] = field(default_factory=list)
    blank_rows_skipped: int = 0
    cells_outside_header_ignored: int = 0
    header_changes: list[dict[str, Any]] = field(default_factory=list)
    missing_tokens: list[str] = field(default_factory=list)
    dayfirst: bool = False
    column_types: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------


def _extension(file_name: str) -> str:
    ext = storage.file_extension(file_name)
    if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise FileValidationError("Unsupported file type. Upload a .csv, .xlsx or .xls file.")
    return ext


def _sniff_content(path: Path, ext: str) -> None:
    with open(path, "rb") as handle:
        head = handle.read(65536)
    if not head:
        raise FileValidationError("The file is empty.")
    if head.startswith(b"MZ") or head.startswith(b"\x7fELF"):
        raise FileValidationError("Executable files are not accepted.")
    if ext == ".xlsx" and not head.startswith(ZIP_MAGIC):
        raise FileValidationError("The file is not a valid .xlsx workbook (it may be corrupted).")
    if ext == ".xls" and not head.startswith(OLE2_MAGIC):
        raise FileValidationError("The file is not a valid .xls workbook (it may be corrupted).")
    if ext == ".csv":
        if head.startswith(ZIP_MAGIC) or head.startswith(OLE2_MAGIC):
            raise FileValidationError(
                "This file is an Excel workbook, not a CSV file. Upload it as .xlsx or .xls."
            )
        if not head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) and b"\x00" in head:
            raise FileValidationError("The file contains binary data and is not a valid CSV file.")


def _validate_xlsx_container(path: Path) -> int:
    """Reject corrupted, macro-enabled or zip-bomb workbooks. Returns uncompressed size."""
    settings = get_settings()
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = {info.filename for info in infos}
    except zipfile.BadZipFile as exc:
        raise FileValidationError("The workbook is corrupted and cannot be opened.") from exc
    if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
        raise FileValidationError("The file is not a valid .xlsx workbook.")
    if any(name.lower().endswith("vbaproject.bin") for name in names):
        raise FileValidationError("Macro-enabled workbooks are not accepted.")
    uncompressed = sum(info.file_size for info in infos)
    compressed = sum(info.compress_size for info in infos) or 1
    if uncompressed > settings.excel_max_uncompressed_mb * MB:
        raise FileValidationError(
            f"The workbook expands to {uncompressed / MB:.0f} MB, above the "
            f"{settings.excel_max_uncompressed_mb} MB safety limit."
        )
    if uncompressed / compressed > 250:
        raise FileValidationError(
            "The workbook is compressed in a suspicious way and was rejected for safety."
        )
    return uncompressed


def _check_processing_resources(ext: str, size: int, uncompressed: int | None) -> None:
    """Reject files the machine cannot safely process right now."""
    settings = get_settings()
    resource_manager.check_disk(size * 4, operation="Importing this file")
    chunk_estimate = resource_manager.estimate_cells_memory(settings.csv_chunk_rows, 200)
    if ext == ".csv":
        if size > settings.csv_processing_target_mb * MB:
            resource_manager.check_memory(
                max(chunk_estimate, size), operation="Importing this large CSV file"
            )
    else:
        target = settings.xlsx_processing_target_mb * MB
        expanded = uncompressed or size * 8
        if size > target:
            resource_manager.check_memory(
                resource_manager.estimate_excel_memory(expanded), operation="Importing this large workbook"
            )
        else:
            resource_manager.check_memory(chunk_estimate, operation="Importing this workbook")


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------


def detect_encoding(path: Path) -> str:
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    for encoding in ("utf-8-sig", "cp1252"):
        decoder = codecs.getincrementaldecoder(encoding)()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(MB), b""):
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_delimiter(sample: str) -> str:
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return ","
    try:
        sniffed = csv.Sniffer().sniff("\n".join(lines[:50]), delimiters=",;\t|").delimiter
        if len(next(csv.reader([lines[0]], delimiter=sniffed))) > 1:
            return sniffed
    except csv.Error:
        pass
    counts = {d: lines[0].count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def _is_blank(tokens: list[Any]) -> bool:
    return all(token is None or not str(token).strip() for token in tokens)


def _csv_rows(path: Path, options: CsvOptions, report: ImportReport) -> Iterator[list[str]]:
    """Yield the header then data rows, validating field counts."""
    encoding = options.encoding or detect_encoding(path)
    report.encoding = encoding
    try:
        with open(path, encoding=encoding, newline="", errors="strict") as handle:
            sample = handle.read(65536)
            delimiter = options.delimiter or detect_delimiter(sample)
            report.delimiter = delimiter
            handle.seek(0)
            reader = csv.reader(handle, delimiter=delimiter)
            header: list[str] | None = None
            for row in reader:
                if header is None:
                    if row and not _is_blank(row):
                        header = row
                        yield header
                    elif row:
                        report.blank_rows_skipped += 1
                    continue
                if not row or _is_blank(row):
                    report.blank_rows_skipped += 1
                    continue
                width = len(header)
                if len(row) > width and _is_blank(row[width:]):
                    row = row[:width]
                if len(row) != width:
                    report.malformed_rows_skipped += 1
                    if len(report.malformed_row_examples) < MAX_REPORTED_ROWS:
                        report.malformed_row_examples.append(
                            {"line": reader.line_num, "fields": len(row), "expected": width}
                        )
                    if options.malformed_rows == "error":
                        raise FileValidationError(
                            f"Line {reader.line_num} has {len(row)} fields but the header has {width}. "
                            "Fix the file or choose to skip malformed rows."
                        )
                    continue
                yield row
            if header is None:
                raise FileValidationError("The file has no header row.")
    except UnicodeDecodeError as exc:
        raise FileValidationError(
            f"The file could not be read with the '{encoding}' encoding. Choose a different encoding."
        ) from exc
    except csv.Error as exc:
        raise FileValidationError(f"The CSV file is malformed and could not be parsed ({exc}).") from exc


# ---------------------------------------------------------------------------
# Excel reading
# ---------------------------------------------------------------------------


def excel_cell_token(
    value: Any, number_format: str | None = None, *, whole_float_as_int: bool = False
) -> str | None:
    """Turn a typed Excel cell into text that keeps its intent (decimal vs integer).

    .xlsx: openpyxl returns ``float`` only when the stored cell text uses decimal
    notation ("1200.0"), so a float is always kept as a decimal, and an int
    formatted with decimals/currency/percent ("0.00", "$#,##0") is decimal too.
    .xls stores every number as a double, so ``whole_float_as_int=True`` treats
    whole values as integers there (number formats are not inspected for .xls).
    """
    if value is None:
        return None
    decimal_format = bool(number_format and _DECIMAL_NUMBER_FORMAT.search(number_format))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value}.0" if decimal_format else str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if whole_float_as_int and value.is_integer() and not decimal_format and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    return str(value)


def list_excel_sheets(path: Path, ext: str) -> list[str]:
    try:
        if ext == ".xlsx":
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
            try:
                return [ws.title for ws in workbook.worksheets]
            finally:
                workbook.close()
        book = xlrd.open_workbook(str(path), on_demand=True)
        try:
            return list(book.sheet_names())
        finally:
            book.release_resources()
    except FileValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - any parser failure means a corrupted workbook
        raise FileValidationError("The workbook is corrupted and cannot be opened.") from exc


def _select_sheet(sheets: list[str], requested: str | None) -> str:
    if not sheets:
        raise FileValidationError("The workbook has no worksheets.")
    if len(sheets) > get_settings().excel_max_sheets:
        raise FileValidationError("The workbook has too many sheets.")
    if requested is None:
        return sheets[0]
    if requested not in sheets:
        raise ValidationError(f"Sheet '{requested}' was not found in the workbook.")
    return requested


def _xlsx_rows(path: Path, sheet: str) -> Iterator[list[str | None]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet]
        if not hasattr(worksheet, "iter_rows"):
            raise ValidationError("The selected sheet does not contain a data table.")
        for row in worksheet.iter_rows():
            yield [excel_cell_token(cell.value, getattr(cell, "number_format", None)) for cell in row]
    finally:
        workbook.close()


def _xls_token(cell: Any, datemode: int) -> str | None:
    kind = cell.ctype
    if kind in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
        return None
    if kind == xlrd.XL_CELL_BOOLEAN:
        return "true" if cell.value else "false"
    if kind == xlrd.XL_CELL_NUMBER:
        return excel_cell_token(float(cell.value), whole_float_as_int=True)
    if kind == xlrd.XL_CELL_DATE:
        try:
            return xlrd.xldate_as_datetime(cell.value, datemode).isoformat(sep=" ")
        except (xlrd.xldate.XLDateError, ValueError, OverflowError):
            return str(cell.value)
    if kind == xlrd.XL_CELL_ERROR:
        return xlrd.error_text_from_code.get(cell.value, "#ERROR")
    return str(cell.value)


def _xls_rows(path: Path, sheet: str) -> Iterator[list[str | None]]:
    try:
        book = xlrd.open_workbook(str(path), on_demand=True)
    except Exception as exc:  # noqa: BLE001
        raise FileValidationError("The workbook is corrupted and cannot be opened.") from exc
    try:
        worksheet = book.sheet_by_name(sheet)
        for index in range(worksheet.nrows):
            yield [_xls_token(cell, book.datemode) for cell in worksheet.row(index)]
    finally:
        book.release_resources()


def _excel_rows(path: Path, ext: str, sheet: str, report: ImportReport) -> Iterator[list[str | None]]:
    """Yield the header then data rows (trimmed/padded to the header width)."""
    rows = _xlsx_rows(path, sheet) if ext == ".xlsx" else _xls_rows(path, sheet)
    header: list[str | None] | None = None
    width = 0
    try:
        for row in rows:
            if header is None:
                if _is_blank(row):
                    report.blank_rows_skipped += 1
                    continue
                last = max(i for i, token in enumerate(row) if token is not None and str(token).strip())
                header = row[: last + 1]
                width = len(header)
                yield header
                continue
            if _is_blank(row):
                report.blank_rows_skipped += 1
                continue
            if len(row) > width:
                extra = row[width:]
                report.cells_outside_header_ignored += sum(
                    1 for token in extra if token is not None and str(token).strip()
                )
                row = row[:width]
            elif len(row) < width:
                row = row + [None] * (width - len(row))
            yield row
    except (FileValidationError, ValidationError):
        raise
    except Exception as exc:  # noqa: BLE001 - parser failure inside a corrupted workbook
        raise FileValidationError("The workbook is corrupted and cannot be read.") from exc
    if header is None:
        raise FileValidationError("The selected sheet is empty.")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _check_column_count(count: int) -> None:
    """Refuse files wider than MAX_COLUMNS before any per-column work."""
    limit = get_settings().max_columns
    if count > limit:
        raise FileValidationError(
            f"The file has {count:,} columns, above the limit of {limit:,} columns. "
            "Remove unused columns or split the file, then upload it again."
        )


def _spool_raw(rows: Iterator[list[Any]], raw_path: Path, report: ImportReport, guard: Any) -> list[str]:
    """Write raw text rows to an all-text Parquet spool. Returns the column names."""
    chunk_rows = get_settings().csv_chunk_rows
    try:
        header = next(rows)
    except StopIteration as exc:
        raise FileValidationError("The file has no header row.") from exc
    _check_column_count(len(header))
    names, changes = normalize_headers(header)
    report.header_changes = changes
    schema = pa.schema([pa.field(name, pa.string()) for name in names])
    buffer: list[list[Any]] = []
    total = 0
    with ParquetBatchWriter(raw_path, schema) as writer:
        for row in rows:
            buffer.append([None if v is None else str(v) for v in row])
            if len(buffer) >= chunk_rows:
                writer.write(pd.DataFrame(buffer, columns=names, dtype=object))
                total += len(buffer)
                buffer = []
                guard.step(0.05, f"Read {total:,} rows")
        if buffer or total == 0:
            writer.write(pd.DataFrame(buffer, columns=names, dtype=object))
            total += len(buffer)
    report.rows_imported = total
    report.columns = len(names)
    return names


def _iter_raw(raw_path: Path) -> Iterator[pd.DataFrame]:
    """Raw spool batches as object columns (str or None)."""
    import pyarrow.parquet as pq

    with open(raw_path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=get_settings().csv_chunk_rows):
            yield batch.to_pandas()


def _import_defaults(db: Any, organization_id: int, options: ImportOptions) -> tuple[tuple[str, ...], bool]:
    defaults = org_settings.get_org_settings(db, organization_id)
    tokens = (
        options.missing_tokens
        if options.missing_tokens is not None
        else tuple(defaults[org_settings.MISSING_PLACEHOLDERS])
    )
    dayfirst = (
        options.dayfirst if options.dayfirst is not None else bool(defaults[org_settings.DATE_DAYFIRST])
    )
    return tuple(tokens), dayfirst


def _one_line(values: list[Any]) -> str:
    """Example values for a warning, each collapsed to a single line."""
    return ", ".join(" ".join(str(v).split()) for v in values)


def _report_warnings(report: ImportReport) -> None:
    if report.malformed_rows_skipped:
        lines = ", ".join(str(item["line"]) for item in report.malformed_row_examples[:10])
        report.warnings.append(
            f"{report.malformed_rows_skipped} malformed row(s) were skipped because their field count did not "
            f"match the header (first lines: {lines})."
        )
    if report.blank_rows_skipped:
        report.warnings.append(f"{report.blank_rows_skipped} completely blank row(s) were skipped.")
    if report.cells_outside_header_ignored:
        report.warnings.append(
            f"{report.cells_outside_header_ignored} non-empty cell(s) to the right of the "
            "last header column were ignored."
        )
    for change in report.header_changes:
        report.warnings.append(
            f"Column {change['position']} header '{change['original']}' was renamed to "
            f"'{change['new']}' ({change['reason']})."
        )
    for name, info in report.column_types.items():
        if info.get("invalid_numeric_count"):
            report.warnings.append(
                f"Column '{name}': {info['invalid_numeric_count']} value(s) are not numbers in a mostly numeric column "
                f"(e.g. {_one_line(info['invalid_numeric_examples'][:3])}); kept as text."
            )
        if PRECISION_TEXT_REASON in info.get("reasons", []):
            report.warnings.append(
                f"Column '{name}': every value is a number, but some have more digits than a 64-bit float "
                "stores exactly or are out of its range "
                f"(e.g. {_one_line(info.get('precision_loss_examples', [])[:3])}), "
                "so the column is kept as text and no digit is lost."
            )
        date_problems = [r for r in info.get("reasons", []) if r.startswith(INCONSISTENT_DATES_REASON)]
        if info.get("invalid_date_count"):
            date_problems.append(
                f"{info['invalid_date_count']} value(s) could not be read as dates "
                f"(e.g. {_one_line(info.get('invalid_date_examples', [])[:3])})."
            )
        if info.get("logical_type") == "text" and date_problems:
            report.warnings.append(
                f"Column '{name}': inconsistent or invalid dates; kept as text. " + " ".join(date_problems)
            )
        if info.get("ambiguous_dates"):
            report.warnings.append(f"Column '{name}': " + " ".join(info.get("reasons", [])))


def inspect_upload(
    ctx: AuthContext,
    *,
    file_name: str,
    data: bytes,
    options: ImportOptions | None = None,
    preview_rows: int = 20,
) -> dict[str, Any]:
    """Validate an upload and return what the import screen needs (sheets, detection, preview)."""
    options = options or ImportOptions()
    display_name = storage.sanitize_display_filename(file_name)
    ext = _extension(display_name)
    if not data:
        raise FileValidationError("The file is empty.")
    with session_scope() as db:
        org_id = authorize(db, ctx, permission=Permission.UPLOAD_DATASETS, feature=Feature.CSV_XLSX_UPLOAD)
        subscriptions.enforce_upload_size(db, org_id, len(data))
    report = ImportReport(file_type=ext[1:], file_size_bytes=len(data))
    with storage.temporary_directory(org_id, prefix="inspect-") as tmp:
        source = tmp / f"upload{ext}"
        source.write_bytes(data)
        _sniff_content(source, ext)
        sheets: list[str] = []
        if ext == ".csv":
            rows = _csv_rows(source, options.csv, report)
        else:
            if ext == ".xlsx":
                _validate_xlsx_container(source)
            sheets = list_excel_sheets(source, ext)
            sheet = _select_sheet(sheets, options.sheet_name)
            report.sheet_name = sheet
            rows = _excel_rows(source, ext, sheet, report)
        preview: list[list[Any]] = []
        try:
            header = next(rows)
            _check_column_count(len(header))
            for row in rows:
                preview.append(row)
                if len(preview) >= preview_rows:
                    break
        except StopIteration as exc:
            raise FileValidationError("The file has no header row.") from exc
        finally:
            rows.close()
        names, changes = normalize_headers(header)
    return {
        "file_name": display_name,
        "file_type": ext[1:],
        "size_bytes": len(data),
        "sheets": sheets,
        "selected_sheet": report.sheet_name,
        "encoding": report.encoding,
        "delimiter": report.delimiter,
        "columns": names,
        "header_changes": changes,
        "preview": pd.DataFrame(
            preview, columns=names[: len(preview[0])] if preview else names, dtype=object
        ),
    }


def import_dataset(
    ctx: AuthContext,
    *,
    file_name: str,
    data: bytes,
    dataset_name: str | None = None,
    description: str = "",
    options: ImportOptions | None = None,
    progress: ProgressCallback | None = None,
    cancel_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """Validate, parse, type and store an uploaded file as a new dataset (version 1)."""
    options = options or ImportOptions()
    guard = make_guard(progress, cancel_token)
    display_name = storage.sanitize_display_filename(file_name)
    ext = _extension(display_name)
    size = len(data or b"")
    if size == 0:
        raise FileValidationError("The file is empty.")
    name = validate_dataset_name(dataset_name or Path(display_name).stem)
    with session_scope() as db:
        org_id = authorize(db, ctx, permission=Permission.UPLOAD_DATASETS, feature=Feature.CSV_XLSX_UPLOAD)
        subscriptions.enforce_upload_size(db, org_id, size)
        subscriptions.enforce_dataset_quota(db, org_id)
        subscriptions.enforce_storage_quota(db, org_id, size * 2)
        missing_tokens, dayfirst = _import_defaults(db, org_id, options)
        clash = db.execute(
            select(Dataset.id).where(
                Dataset.organization_id == org_id, func.lower(Dataset.name) == name.lower()
            )
        ).first()
        if clash:
            raise ConflictError("A dataset with this name already exists in your organization.")

    parse_options = ParseOptions(
        missing_tokens=missing_tokens,
        decimal_separator=options.csv.decimal_separator,
        thousands_separator=options.csv.thousands_separator,
        dayfirst=dayfirst,
    )
    report = ImportReport(
        file_type=ext[1:],
        file_size_bytes=size,
        decimal_separator=parse_options.decimal_separator,
        thousands_separator=parse_options.thousands_separator,
        missing_tokens=list(missing_tokens),
        dayfirst=dayfirst,
    )
    with storage.temporary_directory(org_id, prefix="upload-") as tmp:
        source = tmp / f"upload{ext}"
        source.write_bytes(data)
        _sniff_content(source, ext)
        uncompressed = _validate_xlsx_container(source) if ext == ".xlsx" else None
        _check_processing_resources(ext, size, uncompressed)
        guard.step(0.02, "Reading the file")
        if ext == ".csv":
            rows = _csv_rows(source, options.csv, report)
        else:
            report.sheets_available = list_excel_sheets(source, ext)
            report.sheet_name = _select_sheet(report.sheets_available, options.sheet_name)
            rows = _excel_rows(source, ext, report.sheet_name, report)
        raw_path = tmp / "raw.parquet"
        try:
            columns = _spool_raw(rows, raw_path, report, guard)
        finally:
            rows.close()
        if report.rows_imported == 0:
            report.warnings.append("The file contains a header but no data rows.")
        guard.step(0.4, "Detecting column types")
        inferences = infer_types(_iter_raw(raw_path), parse_options, check=guard.check)
        if not inferences:
            inferences = infer_types(
                [pd.DataFrame({c: pd.Series([], dtype=object) for c in columns})], parse_options
            )
        guard.step(0.6, "Converting values")
        typed_path = tmp / "typed.parquet"
        with ParquetBatchWriter(typed_path, arrow_schema(columns, inferences)) as writer:
            for raw in _iter_raw(raw_path):
                guard.check()
                writer.write(convert_frame(raw, inferences, parse_options, check=guard.check))
        report.column_types = {name: inferences[name].to_dict() for name in columns}
        _report_warnings(report)
        guard.step(0.9, "Saving the dataset")
        with session_scope() as db:
            authorize(db, ctx, permission=Permission.UPLOAD_DATASETS, feature=Feature.CSV_XLSX_UPLOAD)
            dataset, result = create_dataset_with_initial_version(
                db,
                ctx,
                name=name,
                source_type=ext[1:],
                original_filename=display_name,
                parquet_source=typed_path,
                original_source=source,
                description=description,
                sheet_name=report.sheet_name,
                import_options=options.to_dict(),
                import_report=report.to_dict(),
            )
            dataset_id = dataset.id
    guard.step(1.0, "Import complete")
    return {
        "dataset_id": dataset_id,
        "name": name,
        "version_id": result.version_id,
        "version_number": result.version_number,
        "rows": result.rows,
        "columns": result.columns,
        "quality_score": result.quality_score,
        "import_report": report.to_dict(),
    }


# ---------------------------------------------------------------------------
# Listing, details, archiving
# ---------------------------------------------------------------------------


def _dataset_row(dataset: Dataset, version: DatasetVersion | None, creator: str | None) -> dict[str, Any]:
    return {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "source_type": dataset.source_type,
        "original_filename": dataset.original_filename,
        "sheet_name": dataset.sheet_name,
        "status": dataset.status,
        "rows": version.row_count if version else 0,
        "columns": version.column_count if version else 0,
        "current_version": version.version_number if version else None,
        "current_version_id": version.id if version else None,
        "quality_score": version.quality_score if version else None,
        "current_size_bytes": version.size_bytes if version else 0,
        "original_size_bytes": dataset.original_size_bytes,
        "created_by": creator or "system",
        "created_at": dataset.created_at,
        "updated_at": dataset.updated_at,
    }


def list_datasets(ctx: AuthContext, *, include_archived: bool = False) -> list[dict[str, Any]]:
    """Datasets visible to the caller (analysts: only granted ones). Loads no data files."""
    with session_scope() as db:
        authorize(db, ctx)
        if include_archived and ctx.role != Role.ORG_ADMIN:
            include_archived = False
        datasets = (
            db.execute(visible_datasets_query(ctx, include_archived=include_archived).order_by(Dataset.name))
            .scalars()
            .all()
        )
        version_ids = [d.current_version_id for d in datasets if d.current_version_id]
        versions = (
            {
                v.id: v
                for v in db.execute(
                    select(DatasetVersion).where(DatasetVersion.id.in_(version_ids))
                ).scalars()
            }
            if version_ids
            else {}
        )
        creators = (
            dict(
                db.execute(
                    select(User.id, User.username).where(
                        User.id.in_([d.created_by_id for d in datasets if d.created_by_id])
                    )
                ).all()
            )
            if datasets
            else {}
        )
        return [
            _dataset_row(d, versions.get(d.current_version_id), creators.get(d.created_by_id))
            for d in datasets
        ]


def get_dataset(ctx: AuthContext, dataset_id: int) -> dict[str, Any]:
    with session_scope() as db:
        dataset = authorize_dataset(db, ctx, dataset_id, include_archived=ctx.role == Role.ORG_ADMIN)
        version = db.get(DatasetVersion, dataset.current_version_id) if dataset.current_version_id else None
        creator = db.get(User, dataset.created_by_id) if dataset.created_by_id else None
        row = _dataset_row(dataset, version, creator.username if creator else None)
        row["schema"] = json.loads(version.schema_json) if version else []
        row["quality"] = json.loads(version.quality_json) if version and version.quality_json else None
        row["import_report"] = json.loads(dataset.import_report_json or "{}")
        row["import_options"] = json.loads(dataset.import_options_json or "{}")
        row["has_original_file"] = bool(dataset.original_file_path)
        row["version_count"] = db.execute(
            select(func.count(DatasetVersion.id)).where(DatasetVersion.dataset_id == dataset.id)
        ).scalar_one()
        return row


def set_dataset_status(ctx: AuthContext, dataset_id: int, *, archived: bool) -> dict[str, Any]:
    """Archive (hide, keep every file) or restore a dataset. Org Admin only."""
    with session_scope() as db:
        org_id = authorize(db, ctx, roles=(Role.ORG_ADMIN,))
        dataset = authorize_dataset(db, ctx, dataset_id, roles=(Role.ORG_ADMIN,), include_archived=True)
        target = DatasetStatus.ARCHIVED.value if archived else DatasetStatus.ACTIVE.value
        if dataset.status == target:
            return {"dataset_id": dataset.id, "status": dataset.status}
        if not archived:
            subscriptions.enforce_dataset_quota(db, org_id)
        dataset.status = target
        record_audit(
            db,
            ctx,
            "dataset.archive" if archived else "dataset.unarchive",
            target_type="dataset",
            target_id=dataset.id,
        )
        return {"dataset_id": dataset.id, "status": dataset.status}


def update_dataset_details(
    ctx: AuthContext, dataset_id: int, *, name: str | None = None, description: str | None = None
) -> dict[str, Any]:
    with session_scope() as db:
        dataset = authorize_dataset(db, ctx, dataset_id, roles=(Role.ORG_ADMIN,), include_archived=True)
        changes: dict[str, Any] = {}
        if name is not None:
            clean = validate_dataset_name(name)
            clash = db.execute(
                select(Dataset.id).where(
                    Dataset.organization_id == dataset.organization_id,
                    func.lower(Dataset.name) == clean.lower(),
                    Dataset.id != dataset.id,
                )
            ).first()
            if clash:
                raise ConflictError("A dataset with this name already exists in your organization.")
            changes["name"] = (dataset.name, clean)
            dataset.name = clean
        if description is not None:
            dataset.description = description.strip()[:2000]
            changes["description"] = "updated"
        record_audit(db, ctx, "dataset.update", target_type="dataset", target_id=dataset.id, details=changes)
        return {"dataset_id": dataset.id, "name": dataset.name, "description": dataset.description}


def storage_summary(ctx: AuthContext) -> dict[str, Any]:
    with session_scope() as db:
        org_id = authorize(db, ctx)
        limits = subscriptions.get_limits(db, org_id)
        used = subscriptions.org_storage_bytes(db, org_id)
        return {
            "used_bytes": used,
            "quota_bytes": limits.max_storage_bytes,
            "used_fraction": used / limits.max_storage_bytes if limits.max_storage_bytes else 0.0,
            "active_datasets": subscriptions.active_dataset_count(db, org_id),
            "max_datasets": limits.max_datasets,
            "max_upload_mb": limits.max_upload_mb,
        }


# ---------------------------------------------------------------------------
# Exports and downloads (authorized; formula-injection safe)
# ---------------------------------------------------------------------------


def _safe_filename_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "dataset"
    return stem[:80]


def frame_to_csv_bytes(frame: pd.DataFrame, *, escape_formulas: bool = True) -> bytes:
    data = escape_spreadsheet_formulas(frame) if escape_formulas else frame
    return data.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S").encode("utf-8-sig")


def _text_cell(sheet: Any, value: str) -> WriteOnlyCell:
    """A cell that is always stored as text.

    openpyxl turns any string starting with "=" into a formula cell; forcing
    the string type means exported data or headers can never execute.
    """
    cell = WriteOnlyCell(sheet, value=value)
    cell.data_type = "s"
    return cell


def frame_to_xlsx_bytes(
    frame: pd.DataFrame, *, sheet_name: str = "data", escape_formulas: bool = True
) -> bytes:
    if len(frame) > EXCEL_MAX_ROWS:
        raise ValidationError("Excel files are limited to 1,048,575 data rows. Export as CSV instead.")
    data = escape_spreadsheet_formulas(frame) if escape_formulas else frame
    workbook = openpyxl.Workbook(write_only=True)
    sheet = workbook.create_sheet(title=re.sub(r"[\[\]:*?/\\]", "_", sheet_name)[:31] or "data")
    sheet.append([_text_cell(sheet, str(c)) for c in data.columns])
    for row in data.itertuples(index=False, name=None):
        values = []
        for value in row:
            if (
                value is None
                or value is pd.NA
                or value is pd.NaT
                or (isinstance(value, float) and math.isnan(value))
            ):
                values.append(None)
            elif isinstance(value, str):
                values.append(_text_cell(sheet, value))
            elif isinstance(value, pd.Timestamp):
                values.append(value.to_pydatetime())
            elif hasattr(value, "item"):
                values.append(value.item())
            else:
                values.append(value)
        sheet.append(values)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def export_dataset(
    ctx: AuthContext,
    dataset_id: int,
    *,
    version_id: int | None = None,
    file_format: str = "csv",
    escape_formulas: bool = True,
) -> tuple[bytes, str, str]:
    """Export a version as CSV or XLSX. Returns (content, file name, MIME type)."""
    if file_format not in {"csv", "xlsx"}:
        raise ValidationError("Export format must be csv or xlsx.")
    ref = get_version_ref(ctx, dataset_id, version_id, permission=Permission.EXPORT_DATA)
    resource_manager.check_memory(
        resource_manager.estimate_parquet_memory(ref.path) * 2, operation="Exporting this dataset"
    )
    frame = pd.concat(list(iter_parquet(ref.path, get_settings().parquet_batch_rows)), ignore_index=True)
    stem = f"{_safe_filename_stem(ref.dataset_name)}_v{ref.version_number}"
    if file_format == "csv":
        content, filename, mime = (
            frame_to_csv_bytes(frame, escape_formulas=escape_formulas),
            f"{stem}.csv",
            "text/csv",
        )
    else:
        content = frame_to_xlsx_bytes(frame, escape_formulas=escape_formulas)
        filename = f"{stem}.xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    with session_scope() as db:
        authorize(db, ctx, permission=Permission.EXPORT_DATA)
        usage_tracking.record_usage(
            db,
            ref.organization_id,
            ctx.user_id,
            UsageMetric.EXPORT,
            details={"dataset_id": ref.dataset_id, "format": file_format},
        )
        record_audit(
            db,
            ctx,
            "dataset.export",
            target_type="dataset",
            target_id=ref.dataset_id,
            details={"version": ref.version_number, "format": file_format, "rows": len(frame)},
        )
    return content, filename, mime


def download_original(ctx: AuthContext, dataset_id: int) -> tuple[bytes, str]:
    """Return the untouched original upload after an integrity check."""
    with session_scope() as db:
        dataset = authorize_dataset(
            db,
            ctx,
            dataset_id,
            permission=Permission.EXPORT_DATA,
            include_archived=ctx.role == Role.ORG_ADMIN,
        )
        if not dataset.original_file_path:
            raise NotFoundError("This dataset has no original upload (it was derived from other datasets).")
        path = storage.resolve_relative(dataset.original_file_path)
        size = path.stat().st_size if path.exists() else 0
        # The bytes are held in memory (and kept by the UI for the download button).
        resource_manager.check_memory(
            max(size, dataset.original_size_bytes or 0) * 2, operation="Downloading the original file"
        )
        if not path.exists() or storage.sha256_file(path) != dataset.original_sha256:
            raise ConflictError("The stored original file failed its integrity check.")
        content = path.read_bytes()
        record_audit(db, ctx, "dataset.download_original", target_type="dataset", target_id=dataset.id)
        return content, dataset.original_filename
