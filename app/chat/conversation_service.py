from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from pydantic import BaseModel, Field

from app.config import settings
from app.llm.ollama_client import (
    OllamaClient,
)
from app.llm.prompts import (
    FOLLOWUP_REWRITE_SYSTEM_PROMPT,
    build_followup_rewrite_user_prompt,
)
from app.memory.models import (
    ChatMessage,
    ChatSession,
    RetrievalRunLog,
)
from app.memory.sqlite_memory import (
    SqliteChatMemory,
)
from app.rag.rag_chain import (
    GroundedRagAssistant,
    RagAnswerResult,
)
from app.utils.language import (
    detect_language,
)


class FollowupRewritePayload(
    BaseModel
):
    is_followup: bool

    standalone_query: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str


@dataclass(frozen=True, slots=True)
class FollowupResolution:
    original_query: str
    standalone_query: str
    is_followup: bool
    confidence: float
    reason: str
    resolver_called: bool
    llm_metrics: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


@dataclass(slots=True)
class ConversationTurnResult:
    session: ChatSession
    user_message: ChatMessage
    assistant_message: ChatMessage
    original_query: str
    standalone_query: str
    followup_resolution: (
        FollowupResolution
    )
    rag_result: RagAnswerResult
    retrieval_run: RetrievalRunLog

    def debug_summary(
        self,
    ) -> dict[str, Any]:
        return {
            "session_id": (
                self.session
                .session_id
            ),
            "user_message_id": (
                self.user_message
                .message_id
            ),
            "assistant_message_id": (
                self.assistant_message
                .message_id
            ),
            "original_query": (
                self.original_query
            ),
            "standalone_query": (
                self.standalone_query
            ),
            "followup": (
                self.followup_resolution
                .to_dict()
            ),
            "rag": (
                self.rag_result
                .debug_summary()
            ),
            "retrieval_run_id": (
                self.retrieval_run
                .retrieval_run_id
            ),
        }


_CONTEXT_HINT_PATTERN = re.compile(
    r"""
    \bwhat\s+about\b
    |
    \bhow\s+about\b
    |
    \bthat\b
    |
    \bthis\b
    |
    \bit\b
    |
    \bthose\b
    |
    \bsame\b
    |
    \bthen\b
    |
    แล้ว
    |
    อันนี้
    |
    แบบนั้น
    |
    แล้วถ้า
    """,
    flags=(
        re.IGNORECASE
        | re.VERBOSE
    ),
)

_BROAD_STANDALONE_QUERY_PATTERN = re.compile(
    r"""
    ^
    \s*
    (
        which\s+
        (
            project
            |
            projects
            |
            system
            |
            systems
            |
            tool
            |
            tools
            |
            role
            |
            roles
        )
        |
        what\s+
        (
            project
            |
            projects
            |
            system
            |
            systems
            |
            evidence
        )
    )
    \b
    """,
    flags=(
        re.IGNORECASE
        | re.VERBOSE
    ),
)


_EXPLICIT_PRONOUN_PATTERN = re.compile(
    r"""
    \b
    (
        it
        |
        its
        |
        they
        |
        them
        |
        their
        |
        he
        |
        him
        |
        his
        |
        she
        |
        her
        |
        hers
        |
        this
        |
        that
        |
        those
        |
        these
        |
        same
    )
    \b
    |
    แล้ว
    |
    อันนี้
    |
    อันนั้น
    |
    เขา
    |
    มัน
    |
    พวกเขา
    """,
    flags=(
        re.IGNORECASE
        | re.VERBOSE
    ),
)

class FollowupResolver:
    """
    Resolve context-dependent messages into standalone retrieval queries.

    A lightweight heuristic avoids unnecessary LLM calls for clearly
    standalone messages. Ambiguous or short messages are resolved by
    Gemma 4 with strict JSON output and think=False.
    """

    def __init__(
        self,
        *,
        llm_client: OllamaClient,
        max_history_messages: int = (
            settings
            .memory_recent_message_limit
        ),
    ) -> None:
        self.llm_client = (
            llm_client
        )

        self.max_history_messages = (
            max_history_messages
        )

    def _looks_context_dependent(
        self,
        latest_message: str,
        history: Sequence[
            ChatMessage
        ],
    ) -> bool:
        if not history:
            return False

        stripped = (
            latest_message.strip()
        )

        if len(stripped) <= 100:
            return True

        return bool(
            _CONTEXT_HINT_PATTERN
            .search(
                stripped
            )
        )

    def _format_history(
        self,
        history: Sequence[
            ChatMessage
        ],
    ) -> str:
        selected_history = history[
            -self.max_history_messages:
        ]

        if not selected_history:
            return "(none)"

        lines = []

        for message in (
            selected_history
        ):
            lines.append(
                f"{message.role.upper()}: "
                f"{message.content}"
            )

        return "\n\n".join(
            lines
        )

    def resolve(
        self,
        *,
        latest_message: str,
        history: Sequence[
            ChatMessage
        ],
        session_summary: str | None,
    ) -> FollowupResolution:
        prepared_message = (
            latest_message.strip()
        )

        if not prepared_message:
            raise ValueError(
                "latest_message must not be empty."
            )

        if not history:
            return FollowupResolution(
                original_query=(
                    prepared_message
                ),
                standalone_query=(
                    prepared_message
                ),
                is_followup=False,
                confidence=1.0,
                reason=(
                    "No prior conversation "
                    "history exists."
                ),
                resolver_called=False,
                llm_metrics={},
            )
        
        if (
            _BROAD_STANDALONE_QUERY_PATTERN
            .search(
                prepared_message
            )
            and not (
                _EXPLICIT_PRONOUN_PATTERN
                .search(
                    prepared_message
                )
            )
        ):
            return FollowupResolution(
                original_query=(
                    prepared_message
                ),
                standalone_query=(
                    prepared_message
                ),
                is_followup=False,
                confidence=0.90,
                reason=(
                    "Preserved a broad standalone "
                    "query to avoid narrowing an "
                    "ambiguous request prematurely."
                ),
                resolver_called=False,
                llm_metrics={},
            )

        if not self._looks_context_dependent(
            prepared_message,
            history,
        ):
            return FollowupResolution(
                original_query=(
                    prepared_message
                ),
                standalone_query=(
                    prepared_message
                ),
                is_followup=False,
                confidence=0.95,
                reason=(
                    "Message appears "
                    "standalone."
                ),
                resolver_called=False,
                llm_metrics={},
            )

        prompt = (
            build_followup_rewrite_user_prompt(
                latest_message=(
                    prepared_message
                ),
                recent_history_text=(
                    self._format_history(
                        history
                    )
                ),
                session_summary=(
                    session_summary
                ),
            )
        )

        (
            payload,
            llm_result,
        ) = self.llm_client.chat_json(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            schema=(
                FollowupRewritePayload
            ),
            system_prompt=(
                FOLLOWUP_REWRITE_SYSTEM_PROMPT
            ),
            think=False,
        )

        standalone_query = (
            payload
            .standalone_query
            .strip()
            or prepared_message
        )

        return FollowupResolution(
            original_query=(
                prepared_message
            ),
            standalone_query=(
                standalone_query
            ),
            is_followup=(
                payload.is_followup
            ),
            confidence=(
                payload.confidence
            ),
            reason=(
                payload.reason
            ),
            resolver_called=True,
            llm_metrics=(
                llm_result.get(
                    "metrics",
                    {},
                )
            ),
        )


class ConversationRagService:
    """
    Persist a conversational grounded-RAG turn.

    Flow:
        load recent history
        → rewrite follow-up if needed
        → save user message
        → answer rewritten query
        → save assistant message
        → log retrieval run and evidence sources
    """

    def __init__(
        self,
        *,
        memory: SqliteChatMemory,
        followup_resolver: (
            FollowupResolver
        ),
        rag_assistant: (
            GroundedRagAssistant
        ),
    ) -> None:
        self.memory = memory

        self.followup_resolver = (
            followup_resolver
        )

        self.rag_assistant = (
            rag_assistant
        )

    def start_session(
        self,
        *,
        user_id: str | None = None,
        title: str | None = None,
    ) -> ChatSession:
        return (
            self.memory
            .create_session(
                user_id=user_id,
                title=title,
            )
        )

    def ask(
        self,
        *,
        session_id: str,
        user_message: str,
        filter_criteria: (
            dict[str, Any]
            | None
        ) = None,
        think: bool = False,
        similarity_threshold: (
            float
            | None
        ) = None,
        graph_metadata: (
            dict[str, Any]
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
    ) -> ConversationTurnResult:
        session = (
            self.memory
            .get_session(
                session_id
            )
        )

        if session is None:
            raise KeyError(
                f"Unknown session_id: "
                f"{session_id}"
            )
        safe_graph_metadata = dict(
            graph_metadata
            or {}
        )

        recent_history = (
            self.memory
            .load_recent_messages(
                session_id
            )
        )

        followup_resolution = (
            self.followup_resolver
            .resolve(
                latest_message=(
                    user_message
                ),
                history=(
                    recent_history
                ),
                session_summary=(
                    session.summary
                ),
            )
        )

        user_language = (
            detect_language(
                user_message
            )
        )

        stored_user_message = (
            self.memory
            .save_message(
                session_id=(
                    session_id
                ),
                role="user",
                content=(
                    user_message
                ),
                language=(
                    user_language
                ),
                original_query=(
                    followup_resolution
                    .original_query
                ),
                standalone_query=(
                    followup_resolution
                    .standalone_query
                ),
                metadata={
                    **safe_graph_metadata,
                    "followup_resolution": (
                        followup_resolution
                        .to_dict()
                    ),
                },
            )
        )

        rag_result = (
            self.rag_assistant
            .answer(
                (
                    followup_resolution
                    .standalone_query
                ),
                filter_criteria=(
                    filter_criteria
                ),
                think=think,
                similarity_threshold=(
                    similarity_threshold
                ),
                top_k_initial=(
                    top_k_initial
                ),
                top_k_final=(
                    top_k_final
                ),
                context_max_characters=(
                    context_max_characters
                ),
            )
        )

        stored_assistant_message = (
            self.memory
            .save_message(
                session_id=(
                    session_id
                ),
                role="assistant",
                content=(
                    rag_result.answer
                ),
                language=(
                    rag_result.language
                ),
                standalone_query=(
                    followup_resolution
                    .standalone_query
                ),
                answerable=(
                    rag_result
                    .answerable
                ),
                confidence=(
                    rag_result
                    .confidence
                ),
                metadata={
                    **safe_graph_metadata,
                    "escalation_recommended": (
                        rag_result
                        .escalation_recommended
                    ),
                    "limitations": (
                        rag_result
                        .limitations
                    ),
                    "thinking_enabled": (
                        rag_result
                        .thinking_enabled
                    ),
                },
            )
        )

        retrieval_run = (
            self.memory
            .log_rag_result(
                session_id=(
                    session_id
                ),
                user_message_id=(
                    stored_user_message
                    .message_id
                ),
                assistant_message_id=(
                    stored_assistant_message
                    .message_id
                ),
                original_query=(
                    followup_resolution
                    .original_query
                ),
                standalone_query=(
                    followup_resolution
                    .standalone_query
                ),
                filter_criteria=(
                    filter_criteria
                ),
                rag_result=(
                    rag_result
                ),
            )
        )

        refreshed_session = (
            self.memory
            .get_session(
                session_id
            )
        )

        if refreshed_session is None:
            raise RuntimeError(
                "Session disappeared "
                "after turn persistence."
            )

        return ConversationTurnResult(
            session=(
                refreshed_session
            ),
            user_message=(
                stored_user_message
            ),
            assistant_message=(
                stored_assistant_message
            ),
            original_query=(
                followup_resolution
                .original_query
            ),
            standalone_query=(
                followup_resolution
                .standalone_query
            ),
            followup_resolution=(
                followup_resolution
            ),
            rag_result=(
                rag_result
            ),
            retrieval_run=(
                retrieval_run
            ),
        )