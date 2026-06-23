from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.config import settings
from app.rag.retriever import (
    RetrievalResult,
    RetrievedChunk,
)


@dataclass(frozen=True, slots=True)
class SourceReference:
    source_id: str
    point_id: str
    score: float
    source_path: str
    category: str | None
    page_number: int | None
    sheet_name: str | None
    row_start: int | None
    row_end: int | None
    heading_path: list[str]
    text: str
    truncated: bool

    @property
    def location_label(
        self,
    ) -> str:
        parts: list[str] = []

        if self.page_number is not None:
            parts.append(
                f"page {self.page_number}"
            )

        if self.sheet_name:
            parts.append(
                f"sheet {self.sheet_name}"
            )

        if (
            self.row_start
            is not None
            and self.row_end
            is not None
        ):
            parts.append(
                f"rows "
                f"{self.row_start}"
                f"-"
                f"{self.row_end}"
            )

        if self.heading_path:
            parts.append(
                " > ".join(
                    self.heading_path
                )
            )

        return (
            "; ".join(parts)
            if parts
            else "document"
        )

    @property
    def display_label(
        self,
    ) -> str:
        return (
            f"{self.source_path} "
            f"({self.location_label})"
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            **asdict(
                self
            ),
            "location_label": (
                self.location_label
            ),
            "display_label": (
                self.display_label
            ),
        }


@dataclass(slots=True)
class FormattedContext:
    text: str
    sources: list[
        SourceReference
    ]
    character_count: int
    max_characters: int

    @property
    def allowed_source_ids(
        self,
    ) -> set[str]:
        return {
            source.source_id
            for source in self.sources
        }


def _optional_int(
    value: Any,
) -> int | None:
    if value is None:
        return None

    return int(
        value
    )


def _build_source_reference(
    *,
    source_id: str,
    hit: RetrievedChunk,
    text: str,
    truncated: bool,
) -> SourceReference:
    payload = hit.payload

    heading_path = (
        payload.get(
            "heading_path"
        )
        or []
    )

    return SourceReference(
        source_id=source_id,
        point_id=hit.point_id,
        score=hit.score,
        source_path=str(
            payload.get(
                "source_path",
                "",
            )
        ),
        category=(
            payload.get(
                "category"
            )
        ),
        page_number=(
            _optional_int(
                payload.get(
                    "page_number"
                )
            )
        ),
        sheet_name=(
            payload.get(
                "sheet_name"
            )
        ),
        row_start=(
            _optional_int(
                payload.get(
                    "row_start"
                )
            )
        ),
        row_end=(
            _optional_int(
                payload.get(
                    "row_end"
                )
            )
        ),
        heading_path=list(
            heading_path
        ),
        text=text,
        truncated=truncated,
    )


def _render_source_block(
    source: SourceReference,
) -> str:
    return (
        f"[{source.source_id}]\n"
        f"source_path: "
        f"{source.source_path}\n"
        f"category: "
        f"{source.category}\n"
        f"location: "
        f"{source.location_label}\n"
        f"content:\n"
        f"{source.text}"
    )


def format_retrieval_context(
    retrieval_result: RetrievalResult,
    *,
    max_characters: int = (
        settings
        .rag_context_max_chars
    ),
) -> FormattedContext:
    """
    Convert accepted retrieval hits into explicit source blocks.

    Source IDs are local to one answer:
        S1
        S2
        S3
        ...

    Only blocks that fit inside the context budget are included.
    """

    if max_characters < 500:
        raise ValueError(
            "max_characters must be at least 500."
        )

    sources: list[
        SourceReference
    ] = []

    rendered_blocks: list[
        str
    ] = []

    used_characters = 0

    for source_index, hit in enumerate(
        retrieval_result.accepted_hits,
        start=1,
    ):
        source_id = (
            f"S{source_index}"
        )

        remaining_characters = (
            max_characters
            - used_characters
        )

        if remaining_characters < 250:
            break

        full_text = hit.text.strip()

        temporary_source = (
            _build_source_reference(
                source_id=source_id,
                hit=hit,
                text=full_text,
                truncated=False,
            )
        )

        full_block = (
            _render_source_block(
                temporary_source
            )
        )

        if len(full_block) <= (
            remaining_characters
        ):
            sources.append(
                temporary_source
            )

            rendered_blocks.append(
                full_block
            )

            used_characters += len(
                full_block
            )

            continue

        metadata_overhead = len(
            _render_source_block(
                _build_source_reference(
                    source_id=source_id,
                    hit=hit,
                    text="",
                    truncated=True,
                )
            )
        )

        available_text_chars = max(
            0,
            remaining_characters
            - metadata_overhead
            - 30,
        )

        if available_text_chars < 150:
            break

        truncated_text = (
            full_text[
                :available_text_chars
            ].rstrip()
            + "\n...[truncated]"
        )

        truncated_source = (
            _build_source_reference(
                source_id=source_id,
                hit=hit,
                text=truncated_text,
                truncated=True,
            )
        )

        truncated_block = (
            _render_source_block(
                truncated_source
            )
        )

        sources.append(
            truncated_source
        )

        rendered_blocks.append(
            truncated_block
        )

        used_characters += len(
            truncated_block
        )

        break

    context_text = (
        "\n\n"
        "========================================"
        "\n\n"
    ).join(
        rendered_blocks
    )

    return FormattedContext(
        text=context_text,
        sources=sources,
        character_count=len(
            context_text
        ),
        max_characters=(
            max_characters
        ),
    )