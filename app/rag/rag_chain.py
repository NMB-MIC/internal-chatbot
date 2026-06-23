from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.ollama_client import (
    OllamaClient,
)
from app.llm.prompts import (
    GROUNDED_RAG_REASONING_SYSTEM_PROMPT,
    GROUNDED_RAG_SYSTEM_PROMPT,
    build_grounded_rag_user_prompt,
)
from app.rag.context import (
    FormattedContext,
    SourceReference,
    format_retrieval_context,
)
from app.rag.retriever import (
    QdrantRetriever,
    RetrievalResult,
)
from app.utils.language import (
    LanguageCode,
    detect_language,
)
from app.rag.answer_quality import (
    detect_answer_quality_issue,
)

ConfidenceLabel = Literal[
    "high",
    "medium",
    "low",
]


class GroundedAnswerPayload(
    BaseModel
):
    answerable: bool

    answer: str

    cited_source_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    confidence: ConfidenceLabel

    escalation_recommended: bool = (
        False
    )

    limitations: list[
        str
    ] = Field(
        default_factory=list
    )


@dataclass(slots=True)
class RagAnswerResult:
    query: str
    language: LanguageCode
    answerable: bool
    answer: str
    confidence: ConfidenceLabel
    escalation_recommended: bool
    limitations: list[str]
    cited_sources: list[
        SourceReference
    ]
    available_sources: list[
        SourceReference
    ]
    retrieval_result: RetrievalResult
    formatted_context: (
        FormattedContext
        | None
    )
    llm_called: bool
    llm_metrics: dict[
        str,
        Any,
    ]
    thinking_enabled: bool

    def debug_summary(
        self,
    ) -> dict[str, Any]:
        return {
            "query": self.query,
            "language": self.language,
            "answerable": (
                self.answerable
            ),
            "confidence": (
                self.confidence
            ),
            "escalation_recommended": (
                self
                .escalation_recommended
            ),
            "limitations": (
                self.limitations
            ),
            "cited_source_ids": [
                source.source_id
                for source in (
                    self.cited_sources
                )
            ],
            "available_source_ids": [
                source.source_id
                for source in (
                    self.available_sources
                )
            ],
            "llm_called": (
                self.llm_called
            ),
            "thinking_enabled": (
                self.thinking_enabled
            ),
            "retrieval": (
                self.retrieval_result
                .summary()
            ),
            "llm_metrics": (
                self.llm_metrics
            ),
        }


_CITATION_GROUP_PATTERN = re.compile(
    r"\[((?:S\d+)(?:\s*,\s*S\d+)*)\]"
)

_SOURCE_ID_PATTERN = re.compile(
    r"S\d+"
)

_INCOMPLETE_TRAILING_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "with",
    "that",
    "which",
    "is",
    "are",
    "was",
    "were",
    "by",
    "for",
    "from",
    "in",
    "on",
    "at",
    "as",
    "saw",
    "called",
    "described",
}


def _looks_incomplete_answer(
    answer: str,
) -> bool:
    """
    Reject visibly truncated generated answers.

    This is intentionally conservative. It does not attempt to judge
    factual correctness; it only catches obvious incomplete outputs.
    """

    prepared = (
        answer.strip()
    )

    if not prepared:
        return True

    prepared = re.sub(
        r"""
        \n\n
        (?:
            Sources
            |
            แหล่งข้อมูล
        )
        :
        .*
        $
        """,
        "",
        prepared,
        flags=(
            re.IGNORECASE
            | re.VERBOSE
            | re.DOTALL
        ),
    ).strip()

    prepared = re.sub(
        r"""
        \s*
        \[
            (?:S\d+)
            (?:
                \s*,\s*S\d+
            )*
        \]
        \s*
        $
        """,
        "",
        prepared,
        flags=(
            re.IGNORECASE
            | re.VERBOSE
        ),
    ).strip()

    if len(
        prepared
    ) < 24:
        return True

    if prepared.endswith(
        (
            ":",
            ";",
            ",",
            "-",
            "—",
            "(",
        )
    ):
        return True

    words = re.findall(
        r"[A-Za-z]+",
        prepared.lower(),
    )

    if (
        words
        and words[-1]
        in _INCOMPLETE_TRAILING_WORDS
    ):
        return True

    return False

def _deduplicate_preserve_order(
    values: list[str],
) -> list[str]:
    seen: set[str] = set()

    unique_values: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(
            value
        )

        unique_values.append(
            value
        )

    return unique_values

def _extract_inline_source_ids(
    answer: str,
) -> list[str]:
    source_ids: list[str] = []

    for match in (
        _CITATION_GROUP_PATTERN
        .finditer(
            answer
        )
    ):
        source_ids.extend(
            _SOURCE_ID_PATTERN
            .findall(
                match.group(1)
            )
        )

    return (
        _deduplicate_preserve_order(
            source_ids
        )
    )

def _sanitize_inline_markers(
    answer: str,
    *,
    allowed_source_ids: set[str],
) -> str:
    """
    Remove hallucinated source IDs while preserving valid grouped
    citations such as:
        [S1]
        [S1, S2]
    """

    def replace_marker(
        match: re.Match,
    ) -> str:
        source_ids = (
            _SOURCE_ID_PATTERN
            .findall(
                match.group(1)
            )
        )

        valid_source_ids = (
            _deduplicate_preserve_order([
                source_id
                for source_id in source_ids
                if source_id
                in allowed_source_ids
            ])
        )

        if not valid_source_ids:
            return ""

        return (
            "["
            + ", ".join(
                valid_source_ids
            )
            + "]"
        )

    return (
        _CITATION_GROUP_PATTERN
        .sub(
            replace_marker,
            answer,
        )
        .strip()
    )


def _fallback_answer(
    *,
    language: LanguageCode,
    reason: str,
) -> str:
    if language in {
        "th",
        "mixed",
    }:
        return (
            "ผมยังไม่พบข้อมูลภายในที่น่าเชื่อถือเพียงพอ"
            "สำหรับยืนยันคำตอบนี้ จึงไม่ควรเดาข้อมูลครับ "
            "กรุณาตรวจสอบเอกสารภายในเพิ่มเติม"
            "หรือติดต่อทีมที่รับผิดชอบ"
            f"\n\nรายละเอียด: {reason}"
        )

    return (
        "I could not find sufficiently reliable internal "
        "documentation to confirm this answer, so I should "
        "not guess. Please check the relevant internal "
        "documentation or contact the responsible support team."
        f"\n\nDetails: {reason}"
    )


def _append_sources_if_missing(
    *,
    answer: str,
    source_ids: list[str],
    language: LanguageCode,
) -> str:
    if not source_ids:
        return answer

    inline_source_ids = set(
        _extract_inline_source_ids(
            answer
        )
    )

    if any(
        source_id
        in inline_source_ids
        for source_id in source_ids
    ):
        return answer

    label = (
        "แหล่งข้อมูล"
        if language in {
            "th",
            "mixed",
        }
        else "Sources"
    )

    markers = " ".join(
        f"[{source_id}]"
        for source_id in source_ids
    )

    return (
        f"{answer}\n\n"
        f"{label}: {markers}"
    )


def _answer_quality_issue_reason(
    *,
    answer: str,
    user_question: str,
) -> str | None:
    """
    Combine the Batch 9.3 markdown/completeness detector with the
    older conservative truncation detector.
    """

    quality_issue = (
        detect_answer_quality_issue(
            answer=answer,
            user_question=user_question,
        )
    )

    if quality_issue.failed:
        return quality_issue.reason

    if _looks_incomplete_answer(
        answer
    ):
        return (
            "visibly incomplete or truncated answer"
        )

    return None


def _valid_source_ids_from_payload(
    *,
    payload: GroundedAnswerPayload,
    allowed_source_ids: set[str],
) -> list[str]:
    return (
        _deduplicate_preserve_order([
            source_id
            for source_id in (
                payload
                .cited_source_ids
            )
            if source_id in (
                allowed_source_ids
            )
        ])
    )


def _invalid_source_ids_from_payload(
    *,
    payload: GroundedAnswerPayload,
    allowed_source_ids: set[str],
) -> list[str]:
    return [
        source_id
        for source_id in (
            payload
            .cited_source_ids
        )
        if source_id not in (
            allowed_source_ids
        )
    ]


class GroundedRagAssistant:
    def __init__(
        self,
        *,
        retriever: QdrantRetriever,
        llm_client: OllamaClient,
        context_max_characters: int = (
            settings
            .rag_context_max_chars
        ),
        require_citations: bool = (
            settings
            .rag_require_citations
        ),
    ) -> None:
        self.retriever = retriever

        self.llm_client = (
            llm_client
        )

        self.context_max_characters = (
            context_max_characters
        )

        self.require_citations = (
            require_citations
        )

    def answer(
        self,
        query: str,
        *,
        filter_criteria: (
            dict[str, Any]
            | None
        ) = None,
        think: bool = False,
        similarity_threshold: (
            float
            | None
        ) = None,
        top_k_initial: (
            int
            | None
        ) = None,
        top_k_final: (
            int
            | None
        ) = None,
        context_max_characters: (
            int
            | None
        ) = None,
    ) -> RagAnswerResult:
        language = (
            detect_language(
                query
            )
        )

        retrieval_result = (
            self.retriever
            .retrieve(
                query,
                filter_criteria=(
                    filter_criteria
                ),
                similarity_threshold=(
                    similarity_threshold
                ),
                top_k_initial=(
                    top_k_initial
                ),
                top_k_final=(
                    top_k_final
                ),
            )
        )

        if not (
            retrieval_result
            .has_evidence
        ):
            reason = (
                "No retrieved source exceeded "
                "the configured similarity threshold."
            )

            return RagAnswerResult(
                query=query,
                language=language,
                answerable=False,
                answer=_fallback_answer(
                    language=language,
                    reason=reason,
                ),
                confidence="low",
                escalation_recommended=(
                    True
                ),
                limitations=[
                    reason
                ],
                cited_sources=[],
                available_sources=[],
                retrieval_result=(
                    retrieval_result
                ),
                formatted_context=None,
                llm_called=False,
                llm_metrics={},
                thinking_enabled=(
                    think
                ),
            )

        formatted_context = (
            format_retrieval_context(
                retrieval_result,
                max_characters=(
                    context_max_characters
                    or self
                    .context_max_characters
                ),
            )
        )

        if not (
            formatted_context
            .sources
        ):
            reason = (
                "Retrieved sources could not fit "
                "inside the context budget."
            )

            return RagAnswerResult(
                query=query,
                language=language,
                answerable=False,
                answer=_fallback_answer(
                    language=language,
                    reason=reason,
                ),
                confidence="low",
                escalation_recommended=(
                    True
                ),
                limitations=[
                    reason
                ],
                cited_sources=[],
                available_sources=[],
                retrieval_result=(
                    retrieval_result
                ),
                formatted_context=(
                    formatted_context
                ),
                llm_called=False,
                llm_metrics={},
                thinking_enabled=(
                    think
                ),
            )

        user_prompt = (
            build_grounded_rag_user_prompt(
                query=query,
                language=language,
                context=(
                    formatted_context
                ),
            )
        )

        structured_user_prompt = (
            user_prompt
        )

        reasoning_result = None

        if think:
            reasoning_result = (
                self.llm_client.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                user_prompt
                            ),
                        }
                    ],
                    system_prompt=(
                        GROUNDED_RAG_REASONING_SYSTEM_PROMPT
                    ),
                    temperature=0.0,
                    think=True,
                )
            )

            reasoned_draft = (
                reasoning_result
                .get(
                    "content",
                    "",
                )
                .strip()
            )

            if reasoned_draft:
                structured_user_prompt = f"""
{user_prompt}

Internal draft from a separate reasoning pass:
{reasoned_draft[:4000]}

Use the internal draft only as a helper.
Verify every company-specific statement against the provided
sources before returning the structured answer.
""".strip()

        (
            payload,
            llm_result,
        ) = self.llm_client.chat_json(
            messages=[
                {
                    "role": "user",
                    "content": (
                        structured_user_prompt
                    ),
                }
            ],
            schema=(
                GroundedAnswerPayload
            ),
            system_prompt=(
                GROUNDED_RAG_SYSTEM_PROMPT
            ),
            think=False,
        )

        llm_metrics = (
            llm_result.get(
                "metrics",
                {},
            )
        )

        if reasoning_result is not None:
            llm_metrics = {
                "reasoning_pass": (
                    reasoning_result.get(
                        "metrics",
                        {},
                    )
                ),
                "structured_pass": (
                    llm_metrics
                ),
            }

        allowed_source_ids = (
            formatted_context
            .allowed_source_ids
        )

        initial_safe_answer = (
            _sanitize_inline_markers(
                payload.answer,
                allowed_source_ids=(
                    allowed_source_ids
                ),
            )
        )

        initial_valid_cited_source_ids = (
            _valid_source_ids_from_payload(
                payload=payload,
                allowed_source_ids=(
                    allowed_source_ids
                ),
            )
        )

        repair_reasons: list[str] = []

        if payload.answerable:
            quality_reason = (
                _answer_quality_issue_reason(
                    answer=initial_safe_answer,
                    user_question=query,
                )
            )

            if quality_reason:
                repair_reasons.append(
                    quality_reason
                )

            if (
                self.require_citations
                and not (
                    initial_valid_cited_source_ids
                )
            ):
                repair_reasons.append(
                    "missing valid internal citation"
                )

        if repair_reasons:
            repair_prompt = f"""
{structured_user_prompt}

The previous structured answer was incomplete, malformed, or
failed validation.

Validation problem(s):
{chr(10).join("- " + reason for reason in repair_reasons)}

Previous answer:
{payload.answer}

Previous cited_source_ids:
{payload.cited_source_ids}

Regenerate the structured answer using the same retrieved sources.

Requirements:
- Return a complete answer.
- Use valid citation IDs from the provided source markers only.
- Preserve exact commands, UI labels, filenames, paths, constants,
  Kafka topics, environment variables, and CLI flags exactly as
  written in the sources.
- If sources contain conflicting numeric values or conflicting
  instructions, state the conflict explicitly instead of inventing
  an explanation.
""".strip()

            try:
                (
                    repair_payload,
                    repair_result,
                ) = (
                    self.llm_client
                    .chat_json(
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    repair_prompt
                                ),
                            }
                        ],
                        schema=(
                            GroundedAnswerPayload
                        ),
                        system_prompt=(
                            GROUNDED_RAG_SYSTEM_PROMPT
                        ),
                        think=False,
                    )
                )

                payload = (
                    repair_payload
                )

                llm_metrics[
                    "repair_attempted"
                ] = True

                llm_metrics[
                    "repair_reasons"
                ] = (
                    repair_reasons
                )

                llm_metrics[
                    "repair_metrics"
                ] = (
                    repair_result.get(
                        "metrics",
                        {},
                    )
                )

            except Exception as exc:
                llm_metrics[
                    "repair_attempted"
                ] = True

                llm_metrics[
                    "repair_failed"
                ] = True

                llm_metrics[
                    "repair_reasons"
                ] = (
                    repair_reasons
                )

                llm_metrics[
                    "repair_error"
                ] = (
                    str(exc)
                )

        valid_cited_source_ids = (
            _valid_source_ids_from_payload(
                payload=payload,
                allowed_source_ids=(
                    allowed_source_ids
                ),
            )
        )

        invalid_source_ids = (
            _invalid_source_ids_from_payload(
                payload=payload,
                allowed_source_ids=(
                    allowed_source_ids
                ),
            )
        )

        safe_answer = (
            _sanitize_inline_markers(
                payload.answer,
                allowed_source_ids=(
                    allowed_source_ids
                ),
            )
        )

        limitations = list(
            payload.limitations
        )

        if invalid_source_ids:
            limitations.append(
                "Removed invalid source IDs "
                "returned by the model: "
                + ", ".join(
                    invalid_source_ids
                )
            )

        answerable = (
            payload.answerable
        )

        if (
            self.require_citations
            and answerable
            and not valid_cited_source_ids
        ):
            answerable = False

            limitations.append(
                "The model attempted to answer "
                "without a valid internal citation."
            )

            safe_answer = (
                _fallback_answer(
                    language=language,
                    reason=(
                        "The generated response "
                        "did not contain a valid "
                        "internal citation."
                    ),
                )
            )

        if answerable:
            post_repair_quality_reason = (
                _answer_quality_issue_reason(
                    answer=safe_answer,
                    user_question=query,
                )
            )

            if post_repair_quality_reason:
                answerable = False

                limitations.append(
                    "The generated answer appeared "
                    "visibly incomplete or malformed: "
                    + post_repair_quality_reason
                )

                safe_answer = (
                    _fallback_answer(
                        language=language,
                        reason=(
                            "The generated answer was incomplete "
                            "or malformed after validation."
                        ),
                    )
                )

        if answerable:
            safe_answer = (
                _append_sources_if_missing(
                    answer=safe_answer,
                    source_ids=(
                        valid_cited_source_ids
                    ),
                    language=language,
                )
            )

        elif not safe_answer:
            safe_answer = (
                _fallback_answer(
                    language=language,
                    reason=(
                        "The available sources "
                        "did not explicitly support "
                        "the requested fact."
                    ),
                )
            )

        source_lookup = {
            source.source_id: (
                source
            )
            for source in (
                formatted_context
                .sources
            )
        }

        cited_sources = [
            source_lookup[
                source_id
            ]
            for source_id in (
                valid_cited_source_ids
            )
            if source_id in (
                source_lookup
            )
        ]

        return RagAnswerResult(
            query=query,
            language=language,
            answerable=answerable,
            answer=safe_answer,
            confidence=(
                payload.confidence
                if answerable
                else "low"
            ),
            escalation_recommended=(
                payload
                .escalation_recommended
                or not answerable
            ),
            limitations=limitations,
            cited_sources=(
                cited_sources
            ),
            available_sources=(
                formatted_context
                .sources
            ),
            retrieval_result=(
                retrieval_result
            ),
            formatted_context=(
                formatted_context
            ),
            llm_called=True,
            llm_metrics=(
                llm_metrics
            ),
            thinking_enabled=(
                think
            ),
        )