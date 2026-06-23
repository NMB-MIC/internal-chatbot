from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentUnit:
    """
    A successfully extracted text unit.

    This is intentionally not a final RAG chunk yet.
    Batch 3 will split these extracted units into retrieval chunks.
    """

    unit_id: str
    text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        self.text = self.text.strip()

        if not self.text:
            raise ValueError("DocumentUnit text must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FileLoadReport:
    """
    Human-readable report for one discovered file.
    """

    source_path: str
    source_extension: str
    status: str
    loader_type: str | None = None
    units_created: int = 0
    characters_extracted: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IngestionResult:
    """
    Aggregated result from loading a directory.
    """

    units: list[DocumentUnit] = field(default_factory=list)
    file_reports: list[FileLoadReport] = field(default_factory=list)

    @property
    def discovered_file_count(self) -> int:
        return len(self.file_reports)

    @property
    def loaded_file_count(self) -> int:
        return sum(
            report.status == "loaded"
            for report in self.file_reports
        )

    @property
    def empty_file_count(self) -> int:
        return sum(
            report.status == "loaded_empty"
            for report in self.file_reports
        )

    @property
    def skipped_file_count(self) -> int:
        return sum(
            report.status.startswith("skipped")
            for report in self.file_reports
        )

    @property
    def failed_file_count(self) -> int:
        return sum(
            report.status == "failed"
            for report in self.file_reports
        )

    @property
    def extracted_unit_count(self) -> int:
        return len(self.units)

    @property
    def extracted_character_count(self) -> int:
        return sum(len(unit.text) for unit in self.units)

    def summary(self) -> dict[str, int]:
        return {
            "discovered_files": self.discovered_file_count,
            "loaded_files": self.loaded_file_count,
            "loaded_empty_files": self.empty_file_count,
            "skipped_files": self.skipped_file_count,
            "failed_files": self.failed_file_count,
            "extracted_units": self.extracted_unit_count,
            "extracted_characters": self.extracted_character_count,
        }