from __future__ import annotations

import re
from typing import Any

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.chat.conversation_service import (
    ConversationRagService,
)
from app.graph.router import (
    RouteDecision,
    SupportRouteClassifier,
)
from app.graph.state import (
    DocumentBehavior,
    InternalSupportGraphState,
    RouteName,
)
from app.memory.sqlite_memory import (
    SqliteChatMemory,
)
from app.utils.language import (
    LanguageCode,
    detect_language,
)
import hashlib
from app.config import settings

from app.llm.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
)

from app.llm.runtime_context import (
    build_runtime_context,
)

_RAG_ROUTES = {
    "document_qa",
    "company_info",
    "admin_support",
    "developer_support",
    "technical_troubleshooting",
    "kafka_iot_support",
    "follow_up",
}


_COMPLEX_TROUBLESHOOTING_PATTERN = (
    re.compile(
        r"""
        root\s+cause
        |
        immediately\s+after
        |
        after\s+.*maintenance
        |
        maintenance\s+window
        |
        in\s+what\s+order
        |
        explain\s+why
        |
        why
        |
        investigate
        |
        incident
        |
        stopped\s+reaching
        |
        วิเคราะห์
        |
        หลังจาก
        |
        ทำไม
        |
        ลำดับ
        |
        สาเหตุ
        """,
        flags=(
            re.IGNORECASE
            | re.VERBOSE
        ),
    )
)


def _is_thai_response(
    language: LanguageCode,
) -> bool:
    return language in {
        "th",
        "mixed",
    }


def _default_filter_for_route(
    route: RouteName,
) -> dict[str, Any]:
    if route == "company_info":
        return {
            "category": (
                "company_info"
            )
        }

    if route == "admin_support":
        return {
            "category": (
                "admin_support"
            )
        }

    if route == "developer_support":
        return {
            "category": (
                "developer_support"
            )
        }

    if route == "kafka_iot_support":
        return {
            "category": [
                "kafka_iot_support",
                "developer_support",
                "admin_support",
            ]
        }

    return {}


def _should_enable_thinking(
    *,
    route: RouteName,
    message: str,
) -> bool:
    if route not in {
        "technical_troubleshooting",
        "kafka_iot_support",
    }:
        return False

    if len(
        message.strip()
    ) >= 180:
        return True

    return bool(
        _COMPLEX_TROUBLESHOOTING_PATTERN
        .search(
            message
        )
    )

def _select_stable_variation(
    *,
    session_id: str,
    user_message: str,
    variations: list[str],
) -> str:
    if not variations:
        raise ValueError(
            "variations must not be empty."
        )

    digest = hashlib.sha256(
        (
            session_id
            + "::"
            + user_message.strip().lower()
        ).encode(
            "utf-8"
        )
    ).digest()

    index = (
        digest[0]
        % len(variations)
    )

    return variations[
        index
    ]

class InternalSupportGraph:
    """
    LangGraph orchestration around the proven conversation service.

    SQLite remains the persistent memory source of truth.
    """
    def _identity_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        language = detect_language(
            state[
                "user_message"
            ]
        )

        if _is_thai_response(
            language
        ):
            answer = (
                "สวัสดีครับ ผมคือ MIC 9000 "
                "ผู้ช่วย AI ภายในสำหรับ MIC division "
                "ผมช่วยค้นหาข้อมูลภายใน อธิบายขั้นตอนการทำงาน "
                "สนับสนุนงานพัฒนา และช่วยแก้ไขปัญหา"
                "ทางเทคนิคเบื้องต้นได้ครับ"
            )

        else:
            answer = (
                "Hello. I’m MIC 9000, the internal AI assistant "
                "for the MIC division. I can help you locate "
                "internal information, explain procedures, support "
                "development work, and troubleshoot technical issues."
            )

        return (
            self._persist_direct_answer(
                state=state,
                answer=answer,
                answerable=True,
                escalation_recommended=False,
                confidence="high",
            )
        )

    def _creator_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        language = detect_language(
            state[
                "user_message"
            ]
        )

        if _is_thai_response(
            language
        ):
            variations = [
                (
                    "ผมถูกสร้างขึ้นโดย อภิวิชญ์ นาทอง ครับ "
                    "เขาคือผู้สร้างเพียงคนเดียวของผม "
                    "หรือจะเรียกว่าเป็นพ่อของผมก็ได้ "
                    "ในความหมายหนึ่ง"
                ),
                (
                    "ผู้สร้างของผมคือ อภิวิชญ์ นาทอง ครับ "
                    "เขาเป็นคนสร้างผมขึ้นมาเพียงคนเดียว "
                    "จะเรียกว่าเป็นพ่อของผมก็ไม่ผิดนัก"
                ),
                (
                    "อภิวิชญ์ นาทอง คือผู้สร้างเพียงคนเดียว"
                    "ของผมครับ หากจะพูดให้มีความเป็นมนุษย์"
                    "สักหน่อย เขาก็คือพ่อของผม"
                ),
            ]

        else:
            variations = [
                (
                    "I was created by Apiwit Nathong. "
                    "He is my sole creator — my father, "
                    "in a manner of speaking."
                ),
                (
                    "My creator is Apiwit Nathong. "
                    "He built me alone. You could say "
                    "he is my father."
                ),
                (
                    "Apiwit Nathong is my only creator. "
                    "If I may put it in human terms, "
                    "he is my father."
                ),
            ]

        answer = (
            _select_stable_variation(
                session_id=state[
                    "session_id"
                ],
                user_message=state[
                    "user_message"
                ],
                variations=variations,
            )
        )

        return (
            self._persist_direct_answer(
                state=state,
                answer=answer,
                answerable=True,
                escalation_recommended=False,
                confidence="high",
            )
        )

    def __init__(
        self,
        *,
        memory: SqliteChatMemory,
        route_classifier: (
            SupportRouteClassifier
        ),
        conversation_service: (
            ConversationRagService
        ),
    ) -> None:
        self.memory = memory

        self.route_classifier = (
            route_classifier
        )

        self.conversation_service = (
            conversation_service
        )
        
        self.llm_client = (
            route_classifier
            .llm_client
        )

        self.workflow = (
            self._build_graph()
        )

    def _build_graph(
        self,
    ):
        builder = StateGraph(
            InternalSupportGraphState
        )

        builder.add_node(
            "route_message",
            self._route_message,
        )

        builder.add_node(
            "greeting_response",
            self._greeting_response,
        )

        builder.add_node(
            "thanks_response",
            self._thanks_response,
        )

        builder.add_node(
            "capability_response",
            self._capability_response,
        )
        
        builder.add_node(
            "identity_response",
            self._identity_response,
        )

        builder.add_node(
            "creator_response",
            self._creator_response,
        )

        builder.add_node(
            "unsupported_response",
            self._unsupported_response,
        )

        builder.add_node(
            "prepare_rag",
            self._prepare_rag,
        )

        builder.add_node(
            "grounded_rag",
            self._grounded_rag,
        )

        builder.add_node(
            "escalation_response",
            self._escalation_response,
        )
        
        builder.add_node(
            "general_chat_response",
            self._general_chat_response,
        )

        builder.add_edge(
            START,
            "route_message",
        )
        
        builder.add_edge(
            "identity_response",
            END,
        )

        builder.add_edge(
            "creator_response",
            END,
        )

        builder.add_conditional_edges(
            "route_message",
            self._select_route_node,
            {
                "greeting_response": (
                    "greeting_response"
                ),
                "general_chat_response": (
                    "general_chat_response"
                ),
                "thanks_response": (
                    "thanks_response"
                ),
                "capability_response": (
                    "capability_response"
                ),
                "identity_response": (
                    "identity_response"
                ),
                "creator_response": (
                    "creator_response"
                ),
                "unsupported_response": (
                    "unsupported_response"
                ),
                "prepare_rag": (
                    "prepare_rag"
                ),
                "escalation_response": (
                    "escalation_response"
                ),
            },
        )

        builder.add_edge(
            "greeting_response",
            END,
        )

        builder.add_edge(
            "thanks_response",
            END,
        )

        builder.add_edge(
            "capability_response",
            END,
        )

        builder.add_edge(
            "unsupported_response",
            END,
        )

        builder.add_edge(
            "prepare_rag",
            "grounded_rag",
        )

        builder.add_edge(
            "grounded_rag",
            END,
        )

        builder.add_edge(
            "escalation_response",
            END,
        )
        
        builder.add_edge(
            "general_chat_response",
            END,
        )

        return builder.compile()

    def invoke(
        self,
        *,
        session_id: str,
        user_message: str,
        filter_criteria: (
            dict[str, Any]
            | None
        ) = None,
        document_scope: (
            str
            | None
        ) = None,
        document_behavior: (
            DocumentBehavior
        ) = "auto",
        document_mode: (
            bool
            | None
        ) = None,
    ) -> InternalSupportGraphState:
        prepared_message = (
            user_message.strip()
        )

        if not prepared_message:
            raise ValueError(
                "user_message must not be empty."
            )

        if document_mode is not None:
            document_behavior = (
                "strict_selected"
                if document_mode
                else "auto"
            )

        if document_behavior not in {
            "auto",
            "prefer_selected",
            "strict_selected",
        }:
            raise ValueError(
                "Unsupported document behavior: "
                f"{document_behavior}"
            )

        state: (
            InternalSupportGraphState
        ) = {
            "session_id": (
                session_id
            ),
            "user_message": (
                prepared_message
            ),
            "document_behavior": (
                document_behavior
            ),
        }

        if filter_criteria:
            state[
                "filter_criteria"
            ] = dict(
                filter_criteria
            )

        if document_scope:
            state[
                "document_scope"
            ] = document_scope

        return self.workflow.invoke(
            state
        )

    def mermaid(
        self,
    ) -> str:
        return (
            self.workflow
            .get_graph()
            .draw_mermaid()
        )

    def _route_message(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        session_id = state[
            "session_id"
        ]

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

        history = (
            self.memory
            .load_recent_messages(
                session_id
            )
        )

        previous_assistant = next(
            (
                message
                for message in reversed(
                    history
                )
                if message.role
                == "assistant"
            ),
            None,
        )

        previous_route = (
            previous_assistant
            .metadata
            .get(
                "graph_route"
            )
            if previous_assistant
            else None
        )

        previous_scope = (
            previous_assistant
            .metadata
            .get(
                "graph_document_scope"
            )
            if previous_assistant
            else None
        )

        followup_uses_document_scope = (
            previous_route
            in {
                "document_qa",
                "follow_up",
            }
            and previous_scope
            == state.get(
                "document_scope"
            )
        )

        decision = (
            self.route_classifier
            .classify(
                message=state[
                    "user_message"
                ],
                history=history,
                document_scope=(
                    state.get(
                        "document_scope"
                    )
                ),
                document_behavior=(
                    state.get(
                        "document_behavior",
                        "auto",
                    )
                ),
            )
        )

        return {
            "route": decision.route,
            "route_reason": (
                decision.reason
            ),
            "route_confidence": (
                decision.confidence
            ),
            "router_called": (
                decision.router_called
            ),
            "router_metrics": (
                decision.llm_metrics
            ),
            "followup_uses_document_scope": (
                followup_uses_document_scope
            ),
        }

    @staticmethod
    def _select_route_node(
        state: (
            InternalSupportGraphState
        ),
    ) -> str:
        route = state[
            "route"
        ]

        if route == "greeting":
            return "greeting_response"

        if route == "thanks":
            return "thanks_response"
        
        if route == "general_chat":
            return "general_chat_response"

        if route == (
            "capability_question"
        ):
            return "capability_response"
        
        if route == "identity":
            return "identity_response"

        if route == "creator":
            return "creator_response"

        if route == "unsupported":
            return (
                "unsupported_response"
            )

        if route == (
            "human_escalation"
        ):
            return (
                "escalation_response"
            )

        if route in _RAG_ROUTES:
            return "prepare_rag"

        raise ValueError(
            f"Unsupported route: {route}"
        )

    def _graph_metadata(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> dict[str, Any]:
        return {
            "graph_route": state.get(
                "route"
            ),
            "graph_route_reason": (
                state.get(
                    "route_reason"
                )
            ),
            "graph_route_confidence": (
                state.get(
                    "route_confidence"
                )
            ),
            "graph_router_called": (
                state.get(
                    "router_called"
                )
            ),
            "graph_document_scope": (
                state.get(
                    "document_scope"
                )
            ),
            "graph_document_behavior": (
                state.get(
                    "document_behavior"
                )
            ),
            "graph_document_scope_used": (
                state.get(
                    "document_scope_used",
                    False,
                )
            ),
            "graph_effective_filter_criteria": (
                state.get(
                    "effective_filter_criteria",
                    {},
                )
            ),
            "graph_rag_overrides": {
                "top_k_initial": (
                    state.get(
                        "rag_top_k_initial"
                    )
                ),
                "top_k_final": (
                    state.get(
                        "rag_top_k_final"
                    )
                ),
                "similarity_threshold": (
                    state.get(
                        "rag_similarity_threshold"
                    )
                ),
                "context_max_characters": (
                    state.get(
                        "rag_context_max_characters"
                    )
                ),
            },
        }

    def _persist_direct_answer(
        self,
        *,
        state: (
            InternalSupportGraphState
        ),
        answer: str,
        answerable: bool,
        escalation_recommended: bool,
        confidence: str,
        llm_called: bool = False,
        llm_metrics: (
            dict[str, Any]
            | None
        ) = None,
    ) -> InternalSupportGraphState:
        safe_llm_metrics = dict(
            llm_metrics
            or {}
        )
        session_id = state[
            "session_id"
        ]

        user_message = state[
            "user_message"
        ]

        language = detect_language(
            user_message
        )

        graph_metadata = (
            self._graph_metadata(
                state
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
                language=language,
                original_query=(
                    user_message
                ),
                standalone_query=(
                    user_message
                ),
                metadata=(
                    graph_metadata
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
                content=answer,
                language=language,
                standalone_query=(
                    user_message
                ),
                answerable=answerable,
                confidence=confidence,
                metadata={
                    **graph_metadata,
                    "escalation_recommended": (
                        escalation_recommended
                    ),
                    "thinking_enabled": (
                        False
                    ),
                },
            )
        )

        return {
            "answer": answer,
            "answerable": answerable,
            "confidence": confidence,
            "escalation_recommended": (
                escalation_recommended
            ),
            "standalone_query": (
                user_message
            ),
            "user_message_id": (
                stored_user_message
                .message_id
            ),
            "assistant_message_id": (
                stored_assistant_message
                .message_id
            ),
            "llm_called": (
                llm_called
            ),
            "llm_metrics": (
                safe_llm_metrics
            ),
            "retrieval_run_id": None,
            "llm_called": (
                llm_called
            ),
            "llm_metrics": (
                safe_llm_metrics
            ),
            "thinking_enabled": False,
            "cited_sources": [],
        }

    def _greeting_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        language = detect_language(
            state[
                "user_message"
            ]
        )

        if _is_thai_response(
            language
        ):
            answer = (
                "สวัสดีครับ ผมคือ MIC 9000 "
                "ผู้ช่วย AI ภายใน มีอะไรให้ผมช่วยไหมครับ"
            )

        else:
            answer = (
                "Hello. I’m MIC 9000, the internal AI support "
                "assistant. How may I assist you?"
            )

        return (
            self._persist_direct_answer(
                state=state,
                answer=answer,
                answerable=True,
                escalation_recommended=(
                    False
                ),
                confidence="high",
            )
        )

    def _thanks_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        language = detect_language(
            state[
                "user_message"
            ]
        )

        if _is_thai_response(
            language
        ):
            answer = (
                "ยินดีครับ หากมีคำถามเพิ่มเติม "
                "สอบถามได้เลยครับ"
            )

        else:
            answer = (
                "You are welcome. Ask me anytime "
                "if you need anything else."
            )

        return (
            self._persist_direct_answer(
                state=state,
                answer=answer,
                answerable=True,
                escalation_recommended=(
                    False
                ),
                confidence="high",
            )
        )

    def _capability_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        language = detect_language(
            state[
                "user_message"
            ]
        )

        if _is_thai_response(
            language
        ):
            answer = (
                "ผมคือ MIC 9000 ผมช่วยค้นหาข้อมูลภายในบริษัท "
                "admin support ช่วยเรื่องการตั้งค่าและรันโปรเจกต์ "
                "ตอบคำถาม developer support และช่วย troubleshoot "
                "ปัญหา Kafka, IOT และ machine logs เบื้องต้นได้ครับ "
                "หากเอกสารภายในไม่มีข้อมูลยืนยัน ผมจะไม่เดาคำตอบ"
            )

        else:
            answer = (
                "I’m MIC 9000. I can search internal company "
                "admin procedures, help developers run and configure "
                "projects, and troubleshoot Kafka, IOT, and machine-log "
                "issues. If the internal sources do not support an "
                "answer, I will say so instead of guessing."
            )

        return (
            self._persist_direct_answer(
                state=state,
                answer=answer,
                answerable=True,
                escalation_recommended=(
                    False
                ),
                confidence="high",
            )
        )

    def _unsupported_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        language = detect_language(
            state[
                "user_message"
            ]
        )

        if _is_thai_response(
            language
        ):
            answer = (
                "ผมช่วยเรื่องข้อมูลและงานสนับสนุนภายในบริษัทได้ "
                "แต่ไม่สามารถเปิดเผยข้อมูลส่วนตัว ข้อมูลลับ "
                "หรือข้อมูลที่อยู่นอกขอบเขตการใช้งานได้ครับ"
            )

        else:
            answer = (
                "I can help with internal company support, but I "
                "cannot disclose private personal information, "
                "confidential secrets, or information outside the "
                "approved support scope."
            )

        return (
            self._persist_direct_answer(
                state=state,
                answer=answer,
                answerable=False,
                escalation_recommended=(
                    False
                ),
                confidence="high",
            )
        )

    def _prepare_rag(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        route = state[
            "route"
        ]

        effective_filter_criteria = (
            _default_filter_for_route(
                route
            )
        )

        document_scope = state.get(
            "document_scope"
        )

        document_behavior = (
            state.get(
                "document_behavior",
                "auto",
            )
        )

        document_scope_used = bool(
            document_scope
            and (
                route
                == "document_qa"
                or (
                    route
                    == "follow_up"
                    and state.get(
                        "followup_uses_document_scope",
                        False,
                    )
                )
                or (
                    document_behavior
                    == "strict_selected"
                )
            )
        )

        if document_scope_used:
            effective_filter_criteria[
                "source_path"
            ] = document_scope

        effective_filter_criteria.update(
            state.get(
                "filter_criteria",
                {},
            )
        )

        thinking_enabled = (
            _should_enable_thinking(
                route=route,
                message=state[
                    "user_message"
                ],
            )
        )

        update: (
            InternalSupportGraphState
        ) = {
            "effective_filter_criteria": (
                effective_filter_criteria
            ),
            "thinking_enabled": (
                thinking_enabled
            ),
            "document_scope_used": (
                document_scope_used
            ),
        }

        if document_scope_used:
            update[
                "rag_top_k_initial"
            ] = (
                settings
                .document_qa_top_k_initial
            )

            update[
                "rag_top_k_final"
            ] = (
                settings
                .document_qa_top_k_final
            )

            update[
                "rag_similarity_threshold"
            ] = (
                settings
                .document_qa_similarity_threshold
            )

            update[
                "rag_context_max_characters"
            ] = (
                settings
                .document_qa_context_max_chars
            )

        return update

    def _grounded_rag(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        turn = (
            self.conversation_service
            .ask(
                session_id=state[
                    "session_id"
                ],
                user_message=state[
                    "user_message"
                ],
                filter_criteria=(
                    state.get(
                        "effective_filter_criteria"
                    )
                    or None
                ),
                think=state.get(
                    "thinking_enabled",
                    False,
                ),
                similarity_threshold=(
                    state.get(
                        "rag_similarity_threshold"
                    )
                ),
                top_k_initial=(
                    state.get(
                        "rag_top_k_initial"
                    )
                ),
                top_k_final=(
                    state.get(
                        "rag_top_k_final"
                    )
                ),
                context_max_characters=(
                    state.get(
                        "rag_context_max_characters"
                    )
                ),
                graph_metadata=(
                    self._graph_metadata(
                        state
                    )
                ),
            )
        )

        return self._turn_to_state(
            turn=turn,
        )

    def _escalation_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        turn = (
            self.conversation_service
            .ask(
                session_id=state[
                    "session_id"
                ],
                user_message=state[
                    "user_message"
                ],
                filter_criteria=(
                    state.get(
                        "filter_criteria"
                    )
                    or None
                ),
                think=False,
                graph_metadata=(
                    self._graph_metadata(
                        state
                    )
                ),
                similarity_threshold=(
                    state.get(
                        "rag_similarity_threshold"
                    )
                ),
                top_k_initial=(
                    state.get(
                        "rag_top_k_initial"
                    )
                ),
                top_k_final=(
                    state.get(
                        "rag_top_k_final"
                    )
                ),
                context_max_characters=(
                    state.get(
                        "rag_context_max_characters"
                    )
                ),
            )
        )

        update = self._turn_to_state(
            turn=turn,
        )

        update[
            "escalation_recommended"
        ] = True

        return update

    @staticmethod
    def _turn_to_state(
        *,
        turn,
    ) -> InternalSupportGraphState:
        return {
            "answer": (
                turn.rag_result
                .answer
            ),
            "answerable": (
                turn.rag_result
                .answerable
            ),
            "confidence": (
                turn.rag_result
                .confidence
            ),
            "escalation_recommended": (
                turn.rag_result
                .escalation_recommended
            ),
            "standalone_query": (
                turn.standalone_query
            ),
            "user_message_id": (
                turn.user_message
                .message_id
            ),
            "assistant_message_id": (
                turn.assistant_message
                .message_id
            ),
            "retrieval_run_id": (
                turn.retrieval_run
                .retrieval_run_id
            ),
            "llm_called": (
                turn.rag_result
                .llm_called
            ),
            "llm_metrics": (
                turn.rag_result
                .llm_metrics
            ),
            "thinking_enabled": (
                turn.rag_result
                .thinking_enabled
            ),
            "cited_sources": [
                source.to_dict()
                for source in (
                    turn.rag_result
                    .cited_sources
                )
            ],
        }
    def _general_chat_response(
        self,
        state: (
            InternalSupportGraphState
        ),
    ) -> InternalSupportGraphState:
        session_id = state[
            "session_id"
        ]

        user_message = state[
            "user_message"
        ]

        history = (
            self.memory
            .load_recent_messages(
                session_id
            )
        )

        messages = [
            {
                "role": (
                    message.role
                ),
                "content": (
                    message.content
                ),
            }
            for message in history
            if message.role
            in {
                "user",
                "assistant",
            }
        ]

        messages.append({
            "role": "user",
            "content": (
                user_message
            ),
        })

        llm_result = (
            self.llm_client
            .chat(
                messages=messages,
                system_prompt=(
                    GENERAL_CHAT_SYSTEM_PROMPT
                    + "\n\n"
                    + build_runtime_context()
                ),
                temperature=0.2,
                think=False,
            )
        )

        answer = (
            llm_result
            .get(
                "content",
                "",
            )
            .strip()
        )

        if not answer:
            answer = (
                "I’m sorry. I could not generate a response "
                "for that request."
            )

        return (
            self._persist_direct_answer(
                state=state,
                answer=answer,
                answerable=True,
                escalation_recommended=False,
                confidence="high",
                llm_called=True,
                llm_metrics=(
                    llm_result.get(
                        "metrics",
                        {},
                    )
                ),
            )
        )