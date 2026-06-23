from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.rag.document_models import (
    DocumentUnit,
    FileLoadReport,
    IngestionResult,
)


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".csv",
    ".xlsx",
}

TEXT_ENCODINGS_TO_TRY = (
    "utf-8-sig",
    "utf-8",
    "cp874",
    "latin-1",
)

TABULAR_ROWS_PER_UNIT = 50

BLOCKED_NAME_FRAGMENTS = {
    "credential",
    "credentials",
    "secret",
    "secrets",
    "password",
    "passwd",
    "private_key",
    "apikey",
    "api_key",
    "token",
}

BLOCKED_EXACT_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
}


def _normalise_text(text: str) -> str:
    """
    Apply conservative cleanup only.
    Aggressive cleaning will be handled later if needed.
    """

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Keep paragraph structure while removing trailing whitespace.
    lines = [line.rstrip() for line in text.splitlines()]

    # Collapse excessive blank lines.
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)

    return cleaned.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _make_document_id(relative_path: str) -> str:
    """
    Stable identity based on path.
    A file update changes source_sha256 but not document_id.
    """

    return hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()[:20]


def _make_unit_id(
    *,
    source_sha256: str,
    source_path: str,
    unit_type: str,
    unit_index: int,
    selector: str = "",
) -> str:
    payload = {
        "source_sha256": source_sha256,
        "source_path": source_path,
        "unit_type": unit_type,
        "unit_index": unit_index,
        "selector": selector,
    }

    serialised = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialised.encode("utf-8")
    ).hexdigest()


def _relative_path(path: Path, documents_dir: Path) -> str:
    return path.relative_to(documents_dir).as_posix()


def _infer_category(relative_path: str) -> str:
    """
    Infer category from the top-level folder.

    Example:
        developer_support/project_a/readme.md
        -> category = developer_support
    """

    parts = Path(relative_path).parts

    if len(parts) > 1:
        return parts[0]

    return "uncategorized"


def _base_metadata(
    path: Path,
    documents_dir: Path,
) -> dict[str, Any]:
    relative_path = _relative_path(path, documents_dir)
    source_sha256 = _sha256_file(path)

    modified_at = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    ).isoformat()

    return {
        "document_id": _make_document_id(relative_path),
        "title": path.stem.replace("_", " ").replace("-", " ").strip(),
        "source_file": path.name,
        "source_path": relative_path,
        "source_extension": path.suffix.lower(),
        "source_sha256": source_sha256,
        "source_size_bytes": path.stat().st_size,
        "modified_at_utc": modified_at,
        "category": _infer_category(relative_path),
    }


def _is_hidden_path(path: Path, documents_dir: Path) -> bool:
    relative_parts = path.relative_to(documents_dir).parts

    return any(part.startswith(".") for part in relative_parts)


def _looks_sensitive(path: Path) -> bool:
    lower_name = path.name.lower()
    lower_stem = path.stem.lower()

    if lower_name in BLOCKED_EXACT_FILENAMES:
        return True

    return any(
        fragment in lower_stem
        for fragment in BLOCKED_NAME_FRAGMENTS
    )


def discover_files(documents_dir: Path) -> list[Path]:
    if not documents_dir.exists():
        return []

    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file()
    )


def _read_text_with_fallback(path: Path) -> tuple[str, str]:
    errors: list[str] = []

    for encoding in TEXT_ENCODINGS_TO_TRY:
        try:
            return path.read_text(encoding=encoding), encoding

        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "Could not decode text file using supported encodings. "
        + " | ".join(errors),
    )


def _load_text_file(
    path: Path,
    documents_dir: Path,
) -> tuple[list[DocumentUnit], list[str]]:
    metadata = _base_metadata(path, documents_dir)
    raw_text, encoding = _read_text_with_fallback(path)
    text = _normalise_text(raw_text)

    warnings: list[str] = []

    if not text:
        warnings.append("Text file contained no extractable text.")
        return [], warnings

    metadata.update({
        "loader_type": "text",
        "unit_type": "file",
        "unit_index": 0,
        "encoding": encoding,
    })

    unit_id = _make_unit_id(
        source_sha256=metadata["source_sha256"],
        source_path=metadata["source_path"],
        unit_type="file",
        unit_index=0,
    )

    return [
        DocumentUnit(
            unit_id=unit_id,
            text=text,
            metadata=metadata,
        )
    ], warnings


def _load_pdf_file(
    path: Path,
    documents_dir: Path,
) -> tuple[list[DocumentUnit], list[str]]:
    base_metadata = _base_metadata(path, documents_dir)
    reader = PdfReader(str(path))

    units: list[DocumentUnit] = []
    warnings: list[str] = []

    if reader.is_encrypted:
        warnings.append(
            "PDF is encrypted. Text extraction may fail without a password."
        )

    for page_index, page in enumerate(reader.pages):
        extracted_text = page.extract_text() or ""
        text = _normalise_text(extracted_text)

        if not text:
            warnings.append(
                f"Page {page_index + 1} contained no extractable text. "
                "It may be blank or image-based."
            )
            continue

        metadata = {
            **base_metadata,
            "loader_type": "pdf",
            "unit_type": "page",
            "unit_index": page_index,
            "page_number": page_index + 1,
            "page_count": len(reader.pages),
        }

        unit_id = _make_unit_id(
            source_sha256=metadata["source_sha256"],
            source_path=metadata["source_path"],
            unit_type="page",
            unit_index=page_index,
            selector=f"page:{page_index + 1}",
        )

        units.append(
            DocumentUnit(
                unit_id=unit_id,
                text=text,
                metadata=metadata,
            )
        )

    if not units:
        warnings.append(
            "PDF produced zero text units. OCR may be required."
        )

    return units, warnings


def _iter_docx_blocks(
    document: DocumentObject,
) -> Iterator[Paragraph | Table]:
    """
    Yield paragraphs and tables in their original top-level order.
    """

    parent_element = document.element.body

    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)

        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _escape_markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)

    return (
        text
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("|", r"\|")
        .strip()
    )


def _table_rows_to_markdown(
    rows: list[list[Any]],
) -> str:
    if not rows:
        return ""

    normalised_rows = [
        [_escape_markdown_cell(cell) for cell in row]
        for row in rows
    ]

    column_count = max(len(row) for row in normalised_rows)

    padded_rows = [
        row + [""] * (column_count - len(row))
        for row in normalised_rows
    ]

    header = padded_rows[0]
    body = padded_rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * column_count) + " |",
    ]

    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in body
    )

    return "\n".join(lines)


def _docx_heading_prefix(paragraph: Paragraph) -> str:
    style_name = getattr(paragraph.style, "name", "") or ""
    match = re.match(r"Heading\s+(\d+)", style_name, flags=re.IGNORECASE)

    if not match:
        return ""

    level = min(int(match.group(1)), 6)

    return "#" * level + " "


def _load_docx_file(
    path: Path,
    documents_dir: Path,
) -> tuple[list[DocumentUnit], list[str]]:
    metadata = _base_metadata(path, documents_dir)
    document = Document(str(path))

    text_parts: list[str] = []
    paragraph_count = 0
    table_count = 0

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = _normalise_text(block.text)

            if not text:
                continue

            prefix = _docx_heading_prefix(block)
            text_parts.append(prefix + text)
            paragraph_count += 1

        elif isinstance(block, Table):
            rows = [
                [cell.text for cell in row.cells]
                for row in block.rows
            ]

            table_text = _table_rows_to_markdown(rows)

            if table_text:
                text_parts.append("[Table]\n" + table_text)
                table_count += 1

    text = _normalise_text("\n\n".join(text_parts))
    warnings: list[str] = []

    if not text:
        warnings.append("DOCX file contained no extractable text.")
        return [], warnings

    metadata.update({
        "loader_type": "docx",
        "unit_type": "file",
        "unit_index": 0,
        "paragraph_count": paragraph_count,
        "table_count": table_count,
    })

    unit_id = _make_unit_id(
        source_sha256=metadata["source_sha256"],
        source_path=metadata["source_path"],
        unit_type="file",
        unit_index=0,
    )

    return [
        DocumentUnit(
            unit_id=unit_id,
            text=text,
            metadata=metadata,
        )
    ], warnings


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [
        _escape_markdown_cell(column)
        for column in df.columns.tolist()
    ]

    rows = [
        [
            _escape_markdown_cell(value)
            for value in row
        ]
        for row in df.itertuples(
            index=False,
            name=None,
        )
    ]

    return _table_rows_to_markdown(
        [columns, *rows]
    )


def _dataframe_to_units(
    *,
    df: pd.DataFrame,
    base_metadata: dict[str, Any],
    selector: str,
    title: str,
) -> tuple[list[DocumentUnit], list[str]]:
    warnings: list[str] = []
    units: list[DocumentUnit] = []

    if df.empty:
        warnings.append(
            f"{selector} contained no data rows."
        )
        return units, warnings

    df = df.fillna("").astype(str)

    for unit_index, start_index in enumerate(
        range(0, len(df), TABULAR_ROWS_PER_UNIT)
    ):
        end_index = min(
            start_index + TABULAR_ROWS_PER_UNIT,
            len(df),
        )

        frame = df.iloc[start_index:end_index]
        markdown_table = _dataframe_to_markdown(frame)

        text = (
            f"# {title} — Rows {start_index + 1}-{end_index}\n\n"
            f"{markdown_table}"
        )

        metadata = {
            **base_metadata,
            "unit_type": "row_block",
            "unit_index": unit_index,
            "row_start": start_index + 1,
            "row_end": end_index,
            "row_count": len(frame),
            "column_count": len(frame.columns),
            "columns": frame.columns.tolist(),
            "selector": selector,
        }

        unit_id = _make_unit_id(
            source_sha256=metadata["source_sha256"],
            source_path=metadata["source_path"],
            unit_type="row_block",
            unit_index=unit_index,
            selector=selector,
        )

        units.append(
            DocumentUnit(
                unit_id=unit_id,
                text=text,
                metadata=metadata,
            )
        )

    return units, warnings


def _read_csv_with_fallback(
    path: Path,
) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []

    for encoding in TEXT_ENCODINGS_TO_TRY:
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
            )

            return df, encoding

        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        "Could not decode CSV using supported encodings. "
        + " | ".join(errors),
    )


def _load_csv_file(
    path: Path,
    documents_dir: Path,
) -> tuple[list[DocumentUnit], list[str]]:
    base_metadata = _base_metadata(path, documents_dir)
    df, encoding = _read_csv_with_fallback(path)

    base_metadata.update({
        "loader_type": "csv",
        "encoding": encoding,
    })

    return _dataframe_to_units(
        df=df,
        base_metadata=base_metadata,
        selector="csv",
        title=path.stem.replace("_", " ").strip(),
    )


def _load_xlsx_file(
    path: Path,
    documents_dir: Path,
) -> tuple[list[DocumentUnit], list[str]]:
    base_metadata = _base_metadata(path, documents_dir)

    sheets: dict[str, pd.DataFrame] = pd.read_excel(
        path,
        sheet_name=None,
        dtype=str,
        keep_default_na=False,
        engine="openpyxl",
    )

    all_units: list[DocumentUnit] = []
    all_warnings: list[str] = []

    for sheet_name, df in sheets.items():
        sheet_metadata = {
            **base_metadata,
            "loader_type": "xlsx",
            "sheet_name": sheet_name,
        }

        units, warnings = _dataframe_to_units(
            df=df,
            base_metadata=sheet_metadata,
            selector=f"sheet:{sheet_name}",
            title=(
                f"{path.stem.replace('_', ' ').strip()} "
                f"— Sheet: {sheet_name}"
            ),
        )

        all_units.extend(units)
        all_warnings.extend(warnings)

    return all_units, all_warnings


LoaderFunction = Callable[
    [Path, Path],
    tuple[list[DocumentUnit], list[str]],
]


LOADERS_BY_EXTENSION: dict[str, LoaderFunction] = {
    ".txt": _load_text_file,
    ".md": _load_text_file,
    ".markdown": _load_text_file,
    ".pdf": _load_pdf_file,
    ".docx": _load_docx_file,
    ".csv": _load_csv_file,
    ".xlsx": _load_xlsx_file,
}


def load_documents(
    documents_dir: Path,
) -> IngestionResult:
    documents_dir = documents_dir.resolve()
    result = IngestionResult()

    for path in discover_files(documents_dir):
        relative_path = _relative_path(
            path,
            documents_dir,
        )

        extension = path.suffix.lower()

        if _is_hidden_path(path, documents_dir):
            result.file_reports.append(
                FileLoadReport(
                    source_path=relative_path,
                    source_extension=extension,
                    status="skipped_hidden",
                    warnings=[
                        "Hidden path was skipped."
                    ],
                )
            )

            continue

        if _looks_sensitive(path):
            result.file_reports.append(
                FileLoadReport(
                    source_path=relative_path,
                    source_extension=extension,
                    status="skipped_sensitive",
                    warnings=[
                        "Filename looked sensitive and was skipped."
                    ],
                )
            )

            continue

        if extension not in SUPPORTED_EXTENSIONS:
            result.file_reports.append(
                FileLoadReport(
                    source_path=relative_path,
                    source_extension=extension,
                    status="skipped_unsupported",
                    warnings=[
                        "Unsupported file extension."
                    ],
                )
            )

            continue

        loader = LOADERS_BY_EXTENSION[extension]

        try:
            units, warnings = loader(
                path,
                documents_dir,
            )

            result.units.extend(units)

            result.file_reports.append(
                FileLoadReport(
                    source_path=relative_path,
                    source_extension=extension,
                    status=(
                        "loaded"
                        if units
                        else "loaded_empty"
                    ),
                    loader_type=loader.__name__,
                    units_created=len(units),
                    characters_extracted=sum(
                        len(unit.text)
                        for unit in units
                    ),
                    warnings=warnings,
                )
            )

        except Exception as exc:
            result.file_reports.append(
                FileLoadReport(
                    source_path=relative_path,
                    source_extension=extension,
                    status="failed",
                    loader_type=loader.__name__,
                    error=repr(exc),
                )
            )

    return result