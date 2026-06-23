from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatSession:
    session_id: str
    user_id: str | None
    title: str | None
    summary: str | None
    summarized_until_message_id: int | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    message_id: int
    session_id: str
    role: str
    content: str
    language: str | None
    original_query: str | None
    standalone_query: str | None
    answerable: bool | None
    confidence: str | None
    metadata: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalRunLog:
    retrieval_run_id: int
    session_id: str
    user_message_id: int
    assistant_message_id: int
    original_query: str
    standalone_query: str
    filter_criteria: dict[str, Any]
    raw_hit_count: int
    accepted_hit_count: int
    top_score: float | None
    similarity_threshold: float
    embedding_seconds: float
    search_seconds: float
    total_seconds: float
    llm_called: bool
    thinking_enabled: bool
    llm_metrics: dict[str, Any]
    limitations: list[str]
    retrieval_diagnostics: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalSourceLog:
    retrieval_source_id: int
    retrieval_run_id: int
    source_rank: int
    source_id: str | None
    point_id: str
    accepted: bool
    cited: bool
    score: float
    source_path: str
    category: str | None
    page_number: int | None
    sheet_name: str | None
    row_start: int | None
    row_end: int | None
    heading_path: list[str]
    text_preview: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
