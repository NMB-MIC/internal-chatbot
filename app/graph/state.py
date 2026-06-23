from __future__ import annotations

from typing import (
    Any,
    Literal,
    NotRequired,
    TypedDict,
)


RouteName = Literal[
    "greeting",
    "thanks",
    "capability_question",
    "identity",
    "creator",
    "general_chat",
    "document_qa",
    "company_info",
    "admin_support",
    "developer_support",
    "technical_troubleshooting",
    "kafka_iot_support",
    "follow_up",
    "unsupported",
    "human_escalation",
]


DocumentBehavior = Literal[
    "auto",
    "prefer_selected",
    "strict_selected",
]


class InternalSupportGraphState(
    TypedDict,
    total=False,
):
    """
    Shared LangGraph state for one user turn.

    Every graph invocation represents one new user message.
    Persistent conversation history remains in SQLite.
    """

    # Required graph inputs
    session_id: str
    user_message: str

    # Optional caller-provided overrides
    filter_criteria: dict[
        str,
        Any,
    ]

    # Selected-document behavior
    document_scope: str
    document_behavior: (
        DocumentBehavior
    )
    followup_uses_document_scope: bool
    document_scope_used: bool

    # Routing output
    route: RouteName
    route_reason: str
    route_confidence: float
    router_called: bool
    router_metrics: dict[
        str,
        Any,
    ]

    # RAG preparation
    effective_filter_criteria: dict[
        str,
        Any,
    ]
    thinking_enabled: bool
    rag_top_k_initial: int
    rag_top_k_final: int
    rag_similarity_threshold: float
    rag_context_max_characters: int

    # Final answer
    answer: str
    answerable: bool
    confidence: str
    escalation_recommended: bool
    standalone_query: str

    # Persistence references
    user_message_id: int
    assistant_message_id: int
    retrieval_run_id: NotRequired[
        int | None
    ]

    # Diagnostics
    llm_called: bool
    llm_metrics: dict[
        str,
        Any,
    ]
    cited_sources: list[
        dict[
            str,
            Any,
        ]
    ]
