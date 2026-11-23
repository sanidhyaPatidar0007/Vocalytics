"""Datasets: list, storage, upload and import, details, admin management and exports."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from components.data_preview import (
    flash,
    human_bytes,
    import_report_view,
    preview_table,
    schema_table,
    show_flash,
)
from components.dataset_picker import dataset_selector
from components.feedback import Progress, confirm, feature_unavailable, md_escape, page_header, run_action
from components.navigation import capabilities
from components.session import page_state, require_ctx, set_page_state
from core.constants import Feature, Permission, Role
from modules import org_settings
from services import dataset_service, versioning_service
from services.dataset_service import CsvOptions, ImportOptions

PAGE = "datasets"

ENCODINGS = ["Detect automatically", "utf-8", "utf-8-sig", "cp1252", "latin-1", "utf-16"]
DELIMITERS = {
    "Detect automatically": None,
    "Comma (,)": ",",
    "Semicolon (;)": ";",
    "Tab": "\t",
    "Pipe (|)": "|",
}
THOUSANDS = {"None": None, "Comma (,)": ",", "Period (.)": ".", "Space": " ", "Apostrophe (')": "'"}
DAYFIRST = {"Organization default": None, "Day first (31/12/2024)": True, "Month first (12/31/2024)": False}

ctx = require_ctx()
caps = capabilities()
page_header(
    "Datasets", "Upload CSV or Excel files, review each dataset's schema and quality, and export the data."
)
show_flash(PAGE)
is_admin = ctx.role == Role.ORG_ADMIN


def _storage_summary() -> None:
    summary = run_action(lambda: dataset_service.storage_summary(ctx), component=PAGE)
    if not summary:
        return
    cols = st.columns(3)
    cols[0].metric(
        "Storage used", f"{human_bytes(summary['used_bytes'])} of {human_bytes(summary['quota_bytes'])}"
    )
    cols[1].metric("Active datasets", f"{summary['active_datasets']:,} of {summary['max_datasets']:,}")
    cols[2].metric("Largest upload", f"{summary['max_upload_mb']:,} MB")
    st.progress(
        min(max(float(summary["used_fraction"]), 0.0), 1.0),
        text=f"{summary['used_fraction'] * 100:.1f}% of storage used",
    )


def _datasets_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No datasets yet. Upload a CSV or Excel file below to create your first dataset.")
        return
    table = pd.DataFrame(
        [
            {
                "Name": r["name"],
                "Status": r["status"],
                "Rows": r["rows"],
                "Columns": r["columns"],
                "Quality": None if r["quality_score"] is None else round(float(r["quality_score"]), 1),
                "Current version": r["current_version"],
                "Current size": human_bytes(r["current_size_bytes"]),
                "Original size": human_bytes(r["original_size_bytes"])
                if r.get("original_size_bytes")
                else "-",
                "Created by": r["created_by"],
                "Updated": r["updated_at"],
            }
            for r in rows
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Upload and import
# ---------------------------------------------------------------------------


def _token(uploaded: Any) -> str:
    """Identify the attached file without reading its content (name, upload id and size)."""
    identity = f"{uploaded.name}\0{getattr(uploaded, 'file_id', '')}\0{uploaded.size}"
    return hashlib.sha256(identity.encode("utf-8", "replace")).hexdigest()[:16]


def _org_defaults() -> dict[str, Any] | None:
    """Organization import defaults (placeholders, date order), readable by every member."""
    return run_action(lambda: org_settings.get_import_defaults(ctx), component=PAGE)


def _inspect(token: str, name: str, data: bytes, sheet: str | None) -> None:
    options = ImportOptions(sheet_name=sheet)
    result = run_action(
        lambda: dataset_service.inspect_upload(ctx, file_name=name, data=data, options=options),
        component=PAGE,
    )
    if result is not None:
        set_page_state(PAGE, "inspection", {"token": token, "sheet": sheet, "result": result})


def _inspection_view(result: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown("**What we found in the file**")
        cols = st.columns(4)
        cols[0].metric("File type", str(result["file_type"]).upper())
        cols[1].metric("Size", human_bytes(result["size_bytes"]))
        if result["file_type"] == "csv":
            cols[2].metric("Detected encoding", result.get("encoding") or "-")
            delimiter = {"\t": "tab", ",": "comma", ";": "semicolon", "|": "pipe"}.get(
                result.get("delimiter"), "-"
            )
            cols[3].metric("Detected delimiter", delimiter)
        else:
            cols[2].metric("Sheets", f"{len(result.get('sheets') or [])}")
            cols[3].metric("Columns", f"{len(result.get('columns') or [])}")
        changes = result.get("header_changes") or []
        if changes:
            st.markdown("**Header changes that will be applied**")
            st.dataframe(pd.DataFrame(changes).astype(str), hide_index=True, width="stretch")
        st.markdown("**Raw preview (values exactly as in the file, before typing)**")
        preview_table(result.get("preview"), empty_message="The file has a header but no data rows.")


def _parse_tokens(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.splitlines() if line.strip())


def _import_form(
    token: str, name: str, load: Callable[[], bytes], inspection: dict[str, Any], sheet: str | None
) -> None:
    is_csv = inspection["file_type"] == "csv"
    defaults = _org_defaults()
    with st.form(f"{PAGE}_import_form"):
        st.markdown("**Import options**")
        dataset_name = st.text_input(
            "Dataset name", value=Path(inspection["file_name"]).stem[:200], key=f"{PAGE}_name_{token}"
        )
        description = st.text_area(
            "Description (optional)", key=f"{PAGE}_description_{token}", max_chars=2000
        )
        encoding = delimiter = None
        decimal = "."
        thousands = None
        malformed = "skip"
        if is_csv:
            c1, c2 = st.columns(2)
            encoding = c1.selectbox("Encoding", ENCODINGS, key=f"{PAGE}_encoding_{token}")
            delimiter = c2.selectbox("Delimiter", list(DELIMITERS), key=f"{PAGE}_delimiter_{token}")
            c3, c4 = st.columns(2)
            decimal = c3.selectbox("Decimal separator", [".", ","], key=f"{PAGE}_decimal_{token}")
            thousands = c4.selectbox("Thousands separator", list(THOUSANDS), key=f"{PAGE}_thousands_{token}")
            malformed = st.radio(
                "Rows whose field count does not match the header",
                ["skip", "error"],
                format_func=lambda v: (
                    "Skip them and list them in the import report" if v == "skip" else "Reject the whole file"
                ),
                key=f"{PAGE}_malformed_{token}",
                horizontal=True,
            )
        prefill = "\n".join(defaults.get("missing_placeholders", [])) if defaults else ""
        placeholders = st.text_area(
            "Values that mean missing (one per line)",
            value=prefill,
            key=f"{PAGE}_placeholders_{token}",
            help=(
                "Leave empty to use the organization's configured placeholders. Zero, False "
                "and 'Unknown' are never treated as missing unless listed."
            ),
        )
        dayfirst_label = "Date order"
        if defaults is not None:
            dayfirst_label += (
                f" (organization default: {'day first' if defaults.get('date_dayfirst') else 'month first'})"
            )
        dayfirst = st.selectbox(dayfirst_label, list(DAYFIRST), key=f"{PAGE}_dayfirst_{token}")
        submitted = st.form_submit_button("Import dataset", type="primary")
    if not submitted:
        return

    def build_options() -> ImportOptions:
        csv = CsvOptions(
            encoding=None if not is_csv or encoding == ENCODINGS[0] else encoding,
            delimiter=DELIMITERS.get(delimiter) if is_csv else None,
            decimal_separator=decimal,
            thousands_separator=THOUSANDS.get(thousands) if is_csv else None,
            malformed_rows=malformed,
        )
        tokens = _parse_tokens(placeholders)
        return ImportOptions(
            csv=csv, sheet_name=sheet, missing_tokens=tokens or None, dayfirst=DAYFIRST[dayfirst]
        )

    options = run_action(build_options, component=PAGE)
    if options is None:
        return
    progress = Progress("Importing the file")
    result = run_action(
        lambda: dataset_service.import_dataset(
            ctx,
            file_name=name,
            data=load(),
            dataset_name=dataset_name.strip() or None,
            description=description,
            options=options,
            progress=progress,
        ),
        component=PAGE,
    )
    progress.done()
    if result is not None:
        set_page_state(PAGE, "import_result", result)
        set_page_state(PAGE, "inspection", None)
        flash(
            PAGE,
            f"Imported {md_escape(result['name'])} as version {result['version_number']} with {result['rows']:,} rows.",
        )
        st.rerun()


def _upload_section() -> None:
    st.subheader("Upload a dataset")
    if not caps.has(Feature.CSV_XLSX_UPLOAD):
        feature_unavailable("Your plan does not include file uploads.")
        return
    if not caps.can(Permission.UPLOAD_DATASETS):
        feature_unavailable(
            "You do not have permission to upload datasets. Ask an Organization Admin to grant it."
        )
        return
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], key=f"{PAGE}_file")
    last = page_state(PAGE, "import_result")
    if uploaded is None:
        st.caption("Choose a CSV, XLSX or XLS file, then inspect it before importing.")
    else:
        name = uploaded.name
        token = _token(uploaded)
        if st.button("Inspect file", key=f"{PAGE}_inspect", icon=":material/search:"):
            set_page_state(PAGE, "import_result", None)
            last = None
            _inspect(token, name, uploaded.getvalue(), None)
        stored = page_state(PAGE, "inspection")
        if stored and stored["token"] == token:
            result = stored["result"]
            sheet = None
            if result.get("sheets"):
                sheets = result["sheets"]
                current = result.get("selected_sheet") or sheets[0]
                sheet = st.selectbox(
                    "Sheet to import",
                    sheets,
                    index=sheets.index(current) if current in sheets else 0,
                    key=f"{PAGE}_sheet_{token}",
                )
                if sheet != result.get("selected_sheet"):
                    _inspect(token, name, uploaded.getvalue(), sheet)
                    stored = page_state(PAGE, "inspection")
                    result = stored["result"] if stored and stored["token"] == token else result
            _inspection_view(result)
            _import_form(token, name, uploaded.getvalue, result, sheet)
        elif stored is None or stored["token"] != token:
            st.caption(
                "Inspect the file to check its encoding, delimiter, sheets and first rows before importing."
            )
    if last:
        with st.container(border=True):
            st.markdown(f"**Import report for {md_escape(last['name'])}**")
            cols = st.columns(3)
            cols[0].metric("Rows", f"{last['rows']:,}")
            cols[1].metric("Columns", f"{last['columns']:,}")
            cols[2].metric(
                "Quality score", "-" if last["quality_score"] is None else f"{last['quality_score']:.1f}"
            )
            import_report_view(last["import_report"])


# ---------------------------------------------------------------------------
# Details, management and exports
# ---------------------------------------------------------------------------


def _manage(info: dict[str, Any]) -> None:
    dataset_id = info["dataset_id"]
    with st.form(f"{PAGE}_details_form_{dataset_id}"):
        new_name = st.text_input("Name", value=info["name"], key=f"{PAGE}_rename_{dataset_id}")
        new_description = st.text_area(
            "Description", value=info.get("description") or "", key=f"{PAGE}_desc_{dataset_id}"
        )
        saved = st.form_submit_button("Save details")
    if saved:
        result = run_action(
            lambda: dataset_service.update_dataset_details(
                ctx, dataset_id, name=new_name, description=new_description
            ),
            component=PAGE,
        )
        if result is not None:
            flash(PAGE, f"Saved the details of {md_escape(result['name'])}.")
            st.rerun()
    st.markdown("**Archive**")
    st.caption(
        "Archiving hides the dataset from analysts and pickers. Every version and "
        "the original file are kept, and you can restore it later."
    )
    ok = confirm(
        "I understand that analysts will no longer see this dataset",
        key=f"{PAGE}_archive_confirm_{dataset_id}",
    )
    if st.button(
        "Archive dataset", key=f"{PAGE}_archive_{dataset_id}", disabled=not ok, icon=":material/archive:"
    ):
        result = run_action(
            lambda: dataset_service.set_dataset_status(ctx, dataset_id, archived=True), component=PAGE
        )
        if result is not None:
            flash(PAGE, "The dataset was archived. It is listed under Archived datasets.")
            st.rerun()


def _exports(info: dict[str, Any]) -> None:
    if not caps.can(Permission.EXPORT_DATA):
        feature_unavailable(
            "You do not have permission to export data. Ask an Organization Admin to grant it."
        )
        return
    dataset_id = info["dataset_id"]
    version = info.get("current_version")
    st.caption(
        f"Exports contain the current version (v{version}). Text that a "
        "spreadsheet would run as a formula is made safe."
    )
    exports: dict[str, Any] = page_state(PAGE, "exports", {}) or {}
    cols = st.columns(3)
    actions = {
        "csv": (
            cols[0],
            "Prepare CSV",
            lambda: dataset_service.export_dataset(ctx, dataset_id, file_format="csv"),
        ),
        "xlsx": (
            cols[1],
            "Prepare Excel",
            lambda: dataset_service.export_dataset(ctx, dataset_id, file_format="xlsx"),
        ),
    }
    if info.get("has_original_file"):
        actions["original"] = (
            cols[2],
            "Prepare original file",
            lambda: (*dataset_service.download_original(ctx, dataset_id), "application/octet-stream"),
        )
    for kind, (col, label, call) in actions.items():
        with col:
            if st.button(label, key=f"{PAGE}_prepare_{kind}_{dataset_id}"):
                result = run_action(call, component=PAGE)
                if result is not None:
                    content, filename, mime = result
                    exports[f"{dataset_id}:{kind}"] = {
                        "data": content,
                        "name": filename,
                        "mime": mime,
                        "version": version,
                    }
                    set_page_state(PAGE, "exports", exports)
            prepared = exports.get(f"{dataset_id}:{kind}")
            if prepared and (kind == "original" or prepared["version"] == version):
                st.download_button(
                    f"Download {md_escape(prepared['name'])}",
                    data=prepared["data"],
                    file_name=prepared["name"],
                    mime=prepared["mime"],
                    key=f"{PAGE}_download_{kind}_{dataset_id}",
                    icon=":material/download:",
                )
    if not info.get("has_original_file"):
        cols[2].caption("No original file: this dataset was derived from other datasets.")


def _details_section() -> None:
    st.subheader("Dataset details")
    selected = dataset_selector(ctx, key=f"{PAGE}_detail_select")
    if not selected:
        return
    dataset_id = selected["dataset_id"]
    info = run_action(lambda: dataset_service.get_dataset(ctx, dataset_id), component=PAGE)
    if not info:
        return
    if info.get("description"):
        st.caption(md_escape(info["description"]))
    cols = st.columns(4)
    cols[0].metric("Rows", f"{info['rows']:,}")
    cols[1].metric("Columns", f"{info['columns']:,}")
    cols[2].metric("Current version", f"v{info['current_version']} of {info['version_count']}")
    cols[3].metric("Quality score", "-" if info["quality_score"] is None else f"{info['quality_score']:.1f}")
    names = ["Schema", "Preview", "Import report", "Export"]
    if is_admin:
        names.append("Manage")
    tabs = st.tabs(names)
    with tabs[0]:
        schema_table(info.get("schema"), (info.get("import_report") or {}).get("column_types"))
    with tabs[1]:
        preview = run_action(
            lambda: versioning_service.load_preview(ctx, dataset_id, max_rows=100), component=PAGE
        )
        if preview is not None:
            st.caption(f"First {len(preview):,} rows of version {info['current_version']}.")
            preview_table(preview)
    with tabs[2]:
        import_report_view(info.get("import_report"))
    with tabs[3]:
        _exports(info)
    if is_admin:
        with tabs[4]:
            _manage(info)


def _archived_section(rows: list[dict[str, Any]]) -> None:
    archived = [r for r in rows if r["status"] == "archived"]
    with st.expander(f"Archived datasets ({len(archived)})"):
        if not archived:
            st.caption("No archived datasets.")
            return
        by_id = {r["dataset_id"]: r for r in archived}
        choice = st.selectbox(
            "Archived dataset",
            list(by_id),
            format_func=lambda i: f"{by_id[i]['name']} ({by_id[i]['rows']:,} rows)",
            key=f"{PAGE}_archived_select",
        )
        ok = confirm("Make this dataset visible again", key=f"{PAGE}_unarchive_confirm")
        if st.button("Restore from archive", key=f"{PAGE}_unarchive", disabled=not ok):
            result = run_action(
                lambda: dataset_service.set_dataset_status(ctx, choice, archived=False), component=PAGE
            )
            if result is not None:
                flash(PAGE, "The dataset is active again.")
                st.rerun()


_storage_summary()
st.subheader("Your datasets")
all_rows = (
    run_action(lambda: dataset_service.list_datasets(ctx, include_archived=is_admin), component=PAGE) or []
)
_datasets_table([r for r in all_rows if r["status"] != "archived"])
if is_admin:
    _archived_section(all_rows)
with st.container(border=True):
    _upload_section()
_details_section()
