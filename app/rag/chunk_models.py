from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """
    Character-based chunk configuration.

    Character counts are intentionally used for the first implementation
    because they are deterministic and easy to inspect. We can tune the
    limits after measuring retrieval quality with the embedding model.
    """

    max_chars: int = 1600
    overlap_chars: int = 200
    min_chars_for_warning: int = 80
    max_table_rows_per_chunk: int = 20

    def __post_init__(self) -> None:
        if self.max_chars < 200:
            raise ValueError("max_chars must be at least 200.")

        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative.")

        if self.overlap_chars >= self.max_chars:
            raise ValueError(
                "overlap_chars must be smaller than max_chars."
            )

        if self.min_chars_for_warning < 0:
            raise ValueError(
                "min_chars_for_warning must not be negative."
            )

        if self.max_table_rows_per_chunk < 1:
            raise ValueError(
                "max_table_rows_per_chunk must be at least 1."
            )


@dataclass(slots=True)
class RetrievalChunk:
    """
    Final retrieval-ready chunk created from an extracted DocumentUnit.
    """

    chunk_id: str
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        if not self.text:
            raise ValueError("RetrievalChunk text must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ChunkingResult:
    """
    Aggregated output for a collection of extracted document units.
    """

    chunks: list[RetrievalChunk] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    source_unit_count: int = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def source_file_count(self) -> int:
        return len({
            chunk.metadata.get("source_path")
            for chunk in self.chunks
        })

    @property
    def oversized_chunk_count(self) -> int:
        return sum(
            bool(chunk.metadata.get("oversized"))
            for chunk in self.chunks
        )

    @property
    def small_chunk_count(self) -> int:
        return sum(
            bool(chunk.metadata.get("small_chunk_warning"))
            for chunk in self.chunks
        )

    @property
    def character_count(self) -> int:
        return sum(
            len(chunk.text)
            for chunk in self.chunks
        )

    def summary(self) -> dict[str, int]:
        return {
            "source_units": self.source_unit_count,
            "source_files": self.source_file_count,
            "chunks": self.chunk_count,
            "chunk_characters_total": self.character_count,
            "oversized_chunks": self.oversized_chunk_count,
            "small_chunks": self.small_chunk_count,
            "warnings": len(self.warnings),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "warnings": list(self.warnings),
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
        }