from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.config import settings
from app.rag.chunk_models import (
    ChunkingConfig,
    ChunkingResult,
    RetrievalChunk,
)
from app.rag.document_models import DocumentUnit


HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)

TABLE_SEPARATOR_CELL_PATTERN = re.compile(
    r"^:?-{3,}:?$"
)

RECURSIVE_SEPARATORS = [
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",  # Zero-width space: useful in Thai text.
    "\uff0c",  # Fullwidth comma.
    "\u3001",  # Ideographic comma.
    "\uff0e",  # Fullwidth full stop.
    "\u3002",  # Ideographic full stop.
    "",
]


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    title: str


@dataclass(frozen=True, slots=True)
class _Section:
    heading_path: tuple[_Heading, ...]
    body: str


@dataclass(frozen=True, slots=True)
class _Block:
    block_type: str
    text: str


@dataclass(frozen=True, slots=True)
class _Fragment:
    block_type: str
    text: str
    metadata_overrides: dict[str, Any] = field(
        default_factory=dict
    )


def default_chunking_config() -> ChunkingConfig:
    return ChunkingConfig(
        max_chars=settings.chunk_size_chars,
        overlap_chars=settings.chunk_overlap_chars,
        min_chars_for_warning=(
            settings.chunk_min_chars_for_warning
        ),
        max_table_rows_per_chunk=(
            settings.chunk_max_table_rows
        ),
    )


def _normalise_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)

    return cleaned.strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _make_chunk_id(
    *,
    source_unit_id: str,
    chunk_index_within_unit: int,
    text: str,
) -> str:
    payload = {
        "source_unit_id": source_unit_id,
        "chunk_index_within_unit": (
            chunk_index_within_unit
        ),
        "content_sha256": _sha256_text(text),
    }

    serialised = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialised.encode("utf-8")
    ).hexdigest()


def _split_sections_by_heading(
    text: str,
) -> list[_Section]:
    """
    Split Markdown-like content by headings while preserving
    the active heading hierarchy as metadata.

    DOCX headings were already converted to Markdown syntax
    by the Batch 2 loader, so this also works for DOCX units.
    """

    text = _normalise_text(text)

    if not text:
        return []

    sections: list[_Section] = []
    heading_stack: list[_Heading] = []
    body_lines: list[str] = []

    def flush_body() -> None:
        body = _normalise_text(
            "\n".join(body_lines)
        )

        if body:
            sections.append(
                _Section(
                    heading_path=tuple(heading_stack),
                    body=body,
                )
            )

    for line in text.splitlines():
        match = HEADING_PATTERN.match(
            line.strip()
        )

        if not match:
            body_lines.append(line)
            continue

        flush_body()
        body_lines = []

        level = len(match.group(1))
        title = match.group(2).strip()

        heading_stack = [
            heading
            for heading in heading_stack
            if heading.level < level
        ]

        heading_stack.append(
            _Heading(
                level=level,
                title=title,
            )
        )

    flush_body()

    if sections:
        return sections

    return [
        _Section(
            heading_path=tuple(),
            body=text,
        )
    ]


def _render_heading_prefix(
    heading_path: tuple[_Heading, ...],
) -> str:
    return "\n".join(
        f"{'#' * heading.level} {heading.title}"
        for heading in heading_path
    )


def _heading_titles(
    heading_path: tuple[_Heading, ...],
) -> list[str]:
    return [
        heading.title
        for heading in heading_path
    ]


def _is_table_row(line: str) -> bool:
    stripped = line.strip()

    return (
        stripped.startswith("|")
        and stripped.endswith("|")
    )


def _is_table_separator_row(line: str) -> bool:
    if not _is_table_row(line):
        return False

    cells = [
        cell.strip()
        for cell in line.strip().strip("|").split("|")
    ]

    return bool(cells) and all(
        TABLE_SEPARATOR_CELL_PATTERN.fullmatch(cell)
        for cell in cells
    )


def _is_table_start(
    lines: list[str],
    index: int,
) -> bool:
    if index + 1 >= len(lines):
        return False

    return (
        _is_table_row(lines[index])
        and _is_table_separator_row(lines[index + 1])
    )


def _split_body_into_blocks(
    body: str,
) -> list[_Block]:
    """
    Split a section into coarse blocks:
    - normal prose
    - Markdown tables
    - fenced code blocks
    """

    lines = body.splitlines()
    blocks: list[_Block] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        text = _normalise_text(
            "\n".join(buffer)
        )

        if text:
            blocks.append(
                _Block(
                    block_type="text",
                    text=text,
                )
            )

        buffer.clear()

    index = 0

    while index < len(lines):
        stripped = lines[index].strip()

        if stripped.startswith("```"):
            flush_buffer()

            code_lines = [lines[index]]
            index += 1

            while index < len(lines):
                code_lines.append(lines[index])

                if lines[index].strip().startswith("```"):
                    index += 1
                    break

                index += 1

            blocks.append(
                _Block(
                    block_type="code",
                    text="\n".join(code_lines).strip(),
                )
            )

            continue

        if _is_table_start(lines, index):
            flush_buffer()

            table_lines = [
                lines[index],
                lines[index + 1],
            ]

            index += 2

            while (
                index < len(lines)
                and _is_table_row(lines[index])
            ):
                table_lines.append(lines[index])
                index += 1

            blocks.append(
                _Block(
                    block_type="table",
                    text="\n".join(table_lines).strip(),
                )
            )

            continue

        buffer.append(lines[index])
        index += 1

    flush_buffer()

    return blocks


def _make_recursive_splitter(
    *,
    chunk_size: int,
    overlap_chars: int,
) -> RecursiveCharacterTextSplitter:
    safe_overlap = min(
        overlap_chars,
        max(0, chunk_size // 3),
    )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=safe_overlap,
        length_function=len,
        separators=RECURSIVE_SEPARATORS,
        is_separator_regex=False,
    )


def _split_generic_text(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    text = _normalise_text(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    splitter = _make_recursive_splitter(
        chunk_size=max_chars,
        overlap_chars=overlap_chars,
    )

    return [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if chunk.strip()
    ]


def _split_markdown_table(
    text: str,
    *,
    max_chars: int,
    max_rows_per_chunk: int,
) -> list[tuple[str, int | None, int | None]]:
    """
    Split a large Markdown table without splitting rows.

    Repeat the header and separator for every fragment.
    Return row offsets relative to the table body so that
    retrieval chunks can expose accurate source row ranges.
    """

    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip()
    ]

    if (
        len(lines) < 2
        or not _is_table_separator_row(lines[1])
    ):
        return [
            (
                text.strip(),
                None,
                None,
            )
        ]

    header_lines = lines[:2]
    body_rows = lines[2:]

    if not body_rows:
        return [
            (
                "\n".join(header_lines),
                None,
                None,
            )
        ]

    fragments: list[
        tuple[str, int, int]
    ] = []

    current_rows: list[str] = []
    current_start_offset: int | None = None

    def flush_rows() -> None:
        nonlocal current_start_offset

        if (
            not current_rows
            or current_start_offset is None
        ):
            return

        current_end_offset = (
            current_start_offset
            + len(current_rows)
            - 1
        )

        fragments.append(
            (
                "\n".join([
                    *header_lines,
                    *current_rows,
                ]),
                current_start_offset,
                current_end_offset,
            )
        )

        current_rows.clear()
        current_start_offset = None

    for row_offset, row in enumerate(
        body_rows,
        start=1,
    ):
        candidate_rows = [
            *current_rows,
            row,
        ]

        candidate_text = "\n".join([
            *header_lines,
            *candidate_rows,
        ])

        exceeds_length = (
            bool(current_rows)
            and len(candidate_text) > max_chars
        )

        exceeds_row_limit = (
            bool(current_rows)
            and len(candidate_rows)
            > max_rows_per_chunk
        )

        if (
            exceeds_length
            or exceeds_row_limit
        ):
            flush_rows()

        if current_start_offset is None:
            current_start_offset = row_offset

        current_rows.append(row)

    flush_rows()

    return fragments


def _split_code_block(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """
    Preserve code fences. Split only the code body if a fenced
    block is too large, then wrap each piece with the original
    opening and closing fences.
    """

    if len(text) <= max_chars:
        return [text.strip()]

    lines = text.splitlines()

    if not lines:
        return []

    opening = lines[0].strip()

    has_closing_fence = (
        len(lines) >= 2
        and lines[-1].strip().startswith("```")
    )

    closing = (
        lines[-1].strip()
        if has_closing_fence
        else "```"
    )

    body_lines = (
        lines[1:-1]
        if has_closing_fence
        else lines[1:]
    )

    body = "\n".join(body_lines).strip()

    if not body:
        return [text.strip()]

    wrapper_length = (
        len(opening)
        + len(closing)
        + 2
    )

    body_max_chars = max(
        120,
        max_chars - wrapper_length,
    )

    body_parts = _split_generic_text(
        body,
        max_chars=body_max_chars,
        overlap_chars=overlap_chars,
    )

    return [
        "\n".join([
            opening,
            part,
            closing,
        ])
        for part in body_parts
    ]


def _split_block(
    block: _Block,
    *,
    max_chars: int,
    config: ChunkingConfig,
    source_row_start: int | None = None,
) -> list[_Fragment]:
    if block.block_type == "table":
        table_parts = _split_markdown_table(
            block.text,
            max_chars=max_chars,
            max_rows_per_chunk=(
                config.max_table_rows_per_chunk
            ),
        )

        fragments: list[_Fragment] = []

        for (
            part,
            local_row_start,
            local_row_end,
        ) in table_parts:
            metadata_overrides: dict[
                str,
                Any,
            ] = {}

            if (
                source_row_start is not None
                and local_row_start is not None
                and local_row_end is not None
            ):
                metadata_overrides.update({
                    "row_start": (
                        source_row_start
                        + local_row_start
                        - 1
                    ),
                    "row_end": (
                        source_row_start
                        + local_row_end
                        - 1
                    ),
                })

            fragments.append(
                _Fragment(
                    block_type=block.block_type,
                    text=part,
                    metadata_overrides=(
                        metadata_overrides
                    ),
                )
            )

        return fragments

    if block.block_type == "code":
        parts = _split_code_block(
            block.text,
            max_chars=max_chars,
            overlap_chars=(
                config.overlap_chars
            ),
        )

    else:
        parts = _split_generic_text(
            block.text,
            max_chars=max_chars,
            overlap_chars=(
                config.overlap_chars
            ),
        )

    return [
        _Fragment(
            block_type=block.block_type,
            text=part,
        )
        for part in parts
        if part.strip()
    ]


def _assemble_fragments(
    fragments: list[_Fragment],
    *,
    max_chars: int,
) -> list[
    tuple[
        str,
        list[str],
        dict[str, Any],
    ]
]:
    """
    Combine nearby small fragments when safe.

    Two table fragments are intentionally never merged together,
    because each fragment repeats its own table header.

    Metadata overrides allow split table fragments to expose
    accurate row ranges.
    """

    assembled: list[
        tuple[
            str,
            list[str],
            dict[str, Any],
        ]
    ] = []

    current_texts: list[str] = []
    current_types: list[str] = []
    current_metadata_overrides: dict[
        str,
        Any,
    ] = {}

    def flush_current() -> None:
        if not current_texts:
            return

        assembled.append((
            "\n\n".join(
                current_texts
            ).strip(),
            list(
                dict.fromkeys(
                    current_types
                )
            ),
            dict(
                current_metadata_overrides
            ),
        ))

        current_texts.clear()
        current_types.clear()
        current_metadata_overrides.clear()

    for fragment in fragments:
        fragment_text = (
            fragment.text.strip()
        )

        if not fragment_text:
            continue

        if not current_texts:
            current_texts.append(
                fragment_text
            )

            current_types.append(
                fragment.block_type
            )

            current_metadata_overrides.update(
                fragment.metadata_overrides
            )

            continue

        candidate_text = "\n\n".join([
            *current_texts,
            fragment_text,
        ])

        current_has_table = (
            "table" in current_types
        )

        next_is_table = (
            fragment.block_type == "table"
        )

        table_conflict = (
            current_has_table
            and next_is_table
        )

        if (
            len(candidate_text) > max_chars
            or table_conflict
        ):
            flush_current()

        current_texts.append(
            fragment_text
        )

        current_types.append(
            fragment.block_type
        )

        current_metadata_overrides.update(
            fragment.metadata_overrides
        )

    flush_current()

    return assembled


def chunk_document_unit(
    unit: DocumentUnit,
    *,
    config: ChunkingConfig | None = None,
) -> list[RetrievalChunk]:
    config = config or default_chunking_config()

    staged_chunks: list[tuple[str, dict]] = []

    sections = _split_sections_by_heading(
        unit.text
    )

    for section_index, section in enumerate(
        sections
    ):
        heading_prefix = _render_heading_prefix(
            section.heading_path
        )

        prefix_length = len(
            heading_prefix
        )

        body_max_chars = max(
            200,
            config.max_chars
            - prefix_length
            - (2 if heading_prefix else 0),
        )

        blocks = _split_body_into_blocks(
            section.body
        )

        fragments: list[_Fragment] = []

        for block in blocks:
            fragments.extend(
                _split_block(
                    block,
                    max_chars=body_max_chars,
                    config=config,
                    source_row_start=(
                        unit.metadata.get(
                            "row_start"
                        )
                    ),
                )
            )

        assembled_bodies = _assemble_fragments(
            fragments,
            max_chars=body_max_chars,
        )

        for (
            body,
            block_types,
            metadata_overrides,
        ) in assembled_bodies:
            text = (
                f"{heading_prefix}\n\n{body}"
                if heading_prefix
                else body
            ).strip()

            staged_chunks.append((
                text,
                {
                    **unit.metadata,
                    "source_unit_row_start": (
                        unit.metadata.get(
                            "row_start"
                        )
                    ),
                    "source_unit_row_end": (
                        unit.metadata.get(
                            "row_end"
                        )
                    ),
                    **metadata_overrides,
                    "source_unit_id": unit.unit_id,
                    "section_index_within_unit": (
                        section_index
                    ),
                    "heading_path": _heading_titles(
                        section.heading_path
                    ),
                    "heading_prefix": heading_prefix,
                    "heading_path_text": " > ".join(
                        _heading_titles(
                            section.heading_path
                        )
                    ),
                    "block_types": block_types,
                },
            ))

    chunk_count = len(staged_chunks)
    chunks: list[RetrievalChunk] = []

    for chunk_index, (
        text,
        metadata,
    ) in enumerate(staged_chunks):
        content_sha256 = _sha256_text(
            text
        )

        oversized = (
            len(text) > config.max_chars
        )

        small_chunk_warning = (
            len(text)
            < config.min_chars_for_warning
        )

        chunk_id = _make_chunk_id(
            source_unit_id=unit.unit_id,
            chunk_index_within_unit=(
                chunk_index
            ),
            text=text,
        )

        chunks.append(
            RetrievalChunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    **metadata,
                    "chunk_index_within_unit": (
                        chunk_index
                    ),
                    "chunk_count_within_unit": (
                        chunk_count
                    ),
                    "character_count": len(text),
                    "content_sha256": content_sha256,
                    "oversized": oversized,
                    "small_chunk_warning": (
                        small_chunk_warning
                    ),
                },
            )
        )

    return chunks


def chunk_document_units(
    units: Iterable[DocumentUnit],
    *,
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    config = config or default_chunking_config()

    units = list(units)

    result = ChunkingResult(
        source_unit_count=len(units)
    )

    for unit in units:
        chunks = chunk_document_unit(
            unit,
            config=config,
        )

        if not chunks:
            result.warnings.append(
                "No chunks were produced for unit: "
                f"{unit.unit_id}"
            )

        result.chunks.extend(
            chunks
        )

    for chunk in result.chunks:
        if chunk.metadata.get("oversized"):
            result.warnings.append(
                "Oversized chunk: "
                f"{chunk.chunk_id[:12]} "
                f"({len(chunk.text)} chars) "
                f"from {chunk.metadata.get('source_path')}"
            )

    return result