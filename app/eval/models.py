from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExpectedRetrieval:
    raw_hit_count: int | None = None
    accepted_hit_count: int | None = None
    min_raw_hit_count: int | None = None
    min_accepted_hit_count: int | None = None
    max_raw_hit_count: int | None = None
    max_accepted_hit_count: int | None = None
    similarity_threshold: float | None = None
    selected_source_path: str | None = None
    diagnostics_contains: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExpectedRetrieval":
        data = data or {}
        return cls(
            raw_hit_count=data.get("raw_hit_count"),
            accepted_hit_count=data.get("accepted_hit_count"),
            min_raw_hit_count=data.get("min_raw_hit_count"),
            min_accepted_hit_count=data.get("min_accepted_hit_count"),
            max_raw_hit_count=data.get("max_raw_hit_count"),
            max_accepted_hit_count=data.get("max_accepted_hit_count"),
            similarity_threshold=data.get("similarity_threshold"),
            selected_source_path=data.get("selected_source_path"),
            diagnostics_contains=list(data.get("diagnostics_contains", [])),
        )


@dataclass(slots=True)
class EvalCase:
    id: str
    question: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    skip: bool = False
    selected_document: str | None = None
    document_behavior: str | None = None
    session_mode: str | None = None  # isolated | shared
    expected_answerable: bool | None = None
    expected_route: str | None = None
    expected_fragments: list[str] = field(default_factory=list)
    expected_any_fragments: list[list[str]] = field(default_factory=list)
    forbidden_fragments: list[str] = field(default_factory=list)
    min_answer_chars: int | None = None
    max_latency_seconds: float | None = None
    require_valid_markdown: bool = True
    require_sources: bool = True
    expected_retrieval: ExpectedRetrieval = field(default_factory=ExpectedRetrieval)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        any_groups: list[list[str]] = []
        for item in data.get("expected_any_fragments", []):
            if isinstance(item, str):
                any_groups.append([item])
            else:
                any_groups.append(list(item))

        session_mode = data.get("session_mode")
        if session_mode is not None:
            session_mode = str(session_mode).strip().lower()
            if session_mode not in {"isolated", "shared"}:
                raise ValueError(
                    f"Invalid session_mode for case {data.get('id')}: {session_mode!r}. "
                    "Use 'isolated' or 'shared'."
                )

        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            description=str(data.get("description", "")),
            tags=list(data.get("tags", [])),
            skip=bool(data.get("skip", False)),
            selected_document=data.get("selected_document"),
            document_behavior=data.get("document_behavior"),
            session_mode=session_mode,
            expected_answerable=data.get("expected_answerable"),
            expected_route=data.get("expected_route"),
            expected_fragments=list(data.get("expected_fragments", [])),
            expected_any_fragments=any_groups,
            forbidden_fragments=list(data.get("forbidden_fragments", [])),
            min_answer_chars=data.get("min_answer_chars"),
            max_latency_seconds=data.get("max_latency_seconds"),
            require_valid_markdown=bool(data.get("require_valid_markdown", True)),
            require_sources=bool(data.get("require_sources", True)),
            expected_retrieval=ExpectedRetrieval.from_dict(
                data.get("expected_retrieval")
            ),
        )


@dataclass(slots=True)
class EvalSuite:
    name: str
    version: str
    description: str
    default_selected_document: str | None
    default_document_behavior: str | None
    isolate_cases: bool
    cases: list[EvalCase]
    path: Path | None = None

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        path: Path | None = None,
    ) -> "EvalSuite":
        defaults = data.get("defaults", {}) or {}
        isolate_cases = bool(defaults.get("isolate_cases", data.get("isolate_cases", True)))
        default_session_mode = "isolated" if isolate_cases else "shared"

        cases: list[EvalCase] = []
        for raw_case in data.get("cases", []):
            merged = {
                "selected_document": defaults.get("selected_document"),
                "document_behavior": defaults.get("document_behavior"),
                "session_mode": defaults.get("session_mode", default_session_mode),
                **raw_case,
            }
            cases.append(EvalCase.from_dict(merged))

        return cls(
            name=str(data["name"]),
            version=str(data.get("version", "1")),
            description=str(data.get("description", "")),
            default_selected_document=defaults.get("selected_document"),
            default_document_behavior=defaults.get("document_behavior"),
            isolate_cases=isolate_cases,
            cases=cases,
            path=path,
        )


@dataclass(slots=True)
class EvalCaseResult:
    case_id: str
    question: str
    status: str
    passed: bool
    skipped: bool
    latency_seconds: float
    answer: str
    failures: list[str]
    route: str | None
    answerable: bool | None
    cited_source_ids: list[str]
    retrieval_summary: dict[str, Any] | None
    raw_result_keys: list[str]
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "status": self.status,
            "passed": self.passed,
            "skipped": self.skipped,
            "latency_seconds": self.latency_seconds,
            "answer": self.answer,
            "failures": self.failures,
            "route": self.route,
            "answerable": self.answerable,
            "cited_source_ids": self.cited_source_ids,
            "retrieval_summary": self.retrieval_summary,
            "raw_result_keys": self.raw_result_keys,
            "session_id": self.session_id,
        }


@dataclass(slots=True)
class EvalRunReport:
    suite_name: str
    suite_version: str
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    pass_rate: float
    session_id: str | None
    results: list[EvalCaseResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "suite_version": self.suite_version,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "duration_seconds": self.duration_seconds,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "skipped_cases": self.skipped_cases,
            "pass_rate": self.pass_rate,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "results": [result.to_dict() for result in self.results],
        }
