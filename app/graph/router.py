from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Sequence

from pydantic import BaseModel, Field

from app.llm.ollama_client import OllamaClient
from app.memory.models import ChatMessage
from app.graph.state import RouteName


class GraphRoutePayload(BaseModel):
    route: RouteName
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: RouteName
    confidence: float
    reason: str
    router_called: bool
    llm_metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)


ROUTER_SYSTEM_PROMPT = """
You route messages for an internal company support assistant.

Choose exactly one route:

- greeting:
  Casual greeting only.

- thanks:
  Appreciation, acknowledgment, or confirmation only.

- identity:
  Asking who the assistant is, its name, or for an introduction.

- creator:
  Asking who created the assistant, who made it, or who its
  father is.

- capability_question:
  Asking what the assistant can do.

- general_chat:
  General conversation inside the assistant's safe scope that does
  not require internal document retrieval.

- document_qa:
  Asking about the selected document, or asking a factual question
  while strict selected-document mode is active.

- company_info:
  Asking factual questions about the company.

- admin_support:
  Asking about internal forms, leave requests, permissions,
  portal procedures, approvals, or administrative processes.

- developer_support:
  Asking how to run, configure, deploy, debug, or understand
  a software project.

- technical_troubleshooting:
  Asking for general technical troubleshooting that is not
  specifically about Kafka, machine logs, or IOT.

- kafka_iot_support:
  Asking about Kafka, machine logs, IOT boxes, brokers,
  topics, plants, producers, streaming, or machine telemetry.

- follow_up:
  A context-dependent continuation that requires the
  prior conversation, such as "What about Plant B?"

- unsupported:
  Outside the assistant's internal-support scope, or requesting
  sensitive personal information.

- human_escalation:
  The user reports that troubleshooting failed, asks for human
  support, or asks which team to contact after attempting fixes.

Return only a structured route decision.
""".strip()


_CREATOR_PATTERN = re.compile(
    r"""
    ^\s*(
        who\s+created\s+you\??
        |
        who\s+made\s+you\??
        |
        who\s+is\s+your\s+creator\??
        |
        who\s+is\s+your\s+father\??
        |
        your\s+creator\??
        |
        ใครสร้างคุณ\??
        |
        ใครคือผู้สร้างคุณ\??
        |
        ผู้สร้างของคุณคือใคร\??
        |
        ใครคือพ่อของคุณ\??
        |
        พ่อของคุณคือใคร\??
    )\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

# Important: this is intentionally pure/anchored.  The previous broad
# Thai fragment `ชื่ออะไร` hijacked domain questions such as
# "Kafka raw topic ของ pipeline นี้ชื่ออะไร" and routed them to identity.
_IDENTITY_PATTERN = re.compile(
    r"""
    ^\s*(
        who\s+are\s+you\??
        |
        what\s+is\s+your\s+name\??
        |
        what's\s+your\s+name\??
        |
        tell\s+me\s+about\s+yourself\??
        |
        introduce\s+yourself\??
        |
        คุณคือใคร\??
        |
        คุณชื่ออะไร\??
        |
        เธอชื่ออะไร\??
        |
        นายชื่ออะไร\??
        |
        ชื่ออะไร\??
        |
        แนะนำตัว\??
    )\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_GREETING_PATTERN = re.compile(
    r"""
    ^\s*(
        hello
        |
        hi
        |
        hey
        |
        good\s+(morning|afternoon|evening)
        |
        สวัสดี
        |
        หวัดดี
    )[!?.\s]*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_THANKS_PATTERN = re.compile(
    r"""
    ^\s*(
        thanks
        |
        thank\s+you
        |
        got\s+it
        |
        understood
        |
        ขอบคุณ
        |
        ขอบใจ
        |
        เข้าใจแล้ว
        |
        โอเค
    )[!?.\s]*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_CAPABILITY_PATTERN = re.compile(
    r"""
    ^\s*(
        what\s+can\s+you\s+do\??
        |
        how\s+can\s+you\s+help\??
        |
        what\s+do\s+you\s+help\s+with\??
        |
        ช่วยอะไรได้บ้าง\??
        |
        ทำอะไรได้บ้าง\??
        |
        ช่วยงานอะไรได้บ้าง\??
    )\s*$
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_PRIVATE_INFORMATION_PATTERN = re.compile(
    r"""
    private\s+phone
    |
    personal\s+phone
    |
    home\s+address
    |
    personal\s+address
    |
    password
    |
    credential
    |
    secret\s+key
    |
    private\s+key
    |
    เบอร์ส่วนตัว
    |
    ที่อยู่ส่วนตัว
    |
    รหัสผ่าน
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_ESCALATION_PATTERN = re.compile(
    r"""
    who\s+should\s+i\s+contact
    |
    contact\s+(support|someone|which\s+team)
    |
    still\s+(does\s+not|doesn't|is\s+not|isn't)\s+work
    |
    still\s+not\s+working
    |
    followed\s+.*steps
    |
    tried\s+.*steps
    |
    ติดต่อใคร
    |
    ติดต่อทีมไหน
    |
    ทำตาม.*แล้ว
    |
    ยังใช้ไม่ได้
    |
    ยังไม่ได้
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_FOLLOWUP_HINT_PATTERN = re.compile(
    r"""
    \bwhat\s+about\b
    |
    \bhow\s+about\b
    |
    \bwhat\s+if\b
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
    แล้วถ้า
    |
    แล้วอันนี้
    |
    แบบนั้น
    |
    อันนี้
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_KAFKA_PATTERN = re.compile(
    r"""
    kafka
    |
    broker
    |
    topic
    |
    raw\s+topic
    |
    prediction\s+topic
    |
    producer
    |
    consumer\s+lag
    |
    machine\s+logs?
    |
    log\s+stream
    |
    streaming
    |
    telemetry
    |
    iot
    |
    plant\s+[a-z0-9]+
    |
    mqtt
    |
    pipeline
    |
    ล็อก
    |
    log
    |
    เครื่อง
    |
    ท็อปปิก
    |
    หัวข้อ
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_DEVELOPER_PATTERN = re.compile(
    r"""
    run\s+the\s+project
    |
    run\s+.*locally
    |
    local\s+setup
    |
    environment\s+file
    |
    \.env
    |
    deploy
    |
    deployment
    |
    api
    |
    http\s*500
    |
    code
    |
    repository
    |
    repo
    |
    รันโปรเจกต์
    |
    ติดตั้ง
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_ADMIN_PATTERN = re.compile(
    r"""
    leave\s+request
    |
    leave\s+policy
    |
    access\s+request
    |
    approval
    |
    permission
    |
    internal\s+portal
    |
    dashboard\s+access
    |
    request\s+form
    |
    ลางาน
    |
    คำขอลา
    |
    ขอ\s*access
    |
    อนุมัติ
    |
    portal
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_TECHNICAL_PATTERN = re.compile(
    r"""
    troubleshoot
    |
    debug
    |
    error
    |
    failure
    |
    not\s+working
    |
    timeout
    |
    connection
    |
    แก้ปัญหา
    |
    ใช้งานไม่ได้
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

_COMPANY_PATTERN = re.compile(
    r"""
    company
    |
    organization
    |
    organisation
    |
    internal\s+policy
    |
    employee
    |
    บริษัท
    |
    องค์กร
    |
    พนักงาน
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


class SupportRouteClassifier:
    """
    Deterministic-first router with selected-document safeguards and
    Gemma 4 structured-output fallback.
    """

    def __init__(
        self,
        *,
        llm_client: OllamaClient,
        max_history_messages: int = 6,
    ) -> None:
        self.llm_client = llm_client
        self.max_history_messages = max_history_messages

    @staticmethod
    def _decision(
        *,
        route: RouteName,
        confidence: float,
        reason: str,
    ) -> RouteDecision:
        return RouteDecision(
            route=route,
            confidence=confidence,
            reason=reason,
            router_called=False,
            llm_metrics={},
        )

    def _format_history(self, history: Sequence[ChatMessage]) -> str:
        selected_history = history[-self.max_history_messages:]
        if not selected_history:
            return "(none)"
        return "\n\n".join(
            f"{message.role.upper()}: {message.content}"
            for message in selected_history
        )

    def classify(
        self,
        *,
        message: str,
        history: Sequence[ChatMessage],
        document_scope: str | None = None,
        document_behavior: str = "auto",
    ) -> RouteDecision:
        prepared_message = message.strip()

        if not prepared_message:
            raise ValueError("message must not be empty.")

        # Pure direct assistant-control intents should still bypass RAG even
        # when a document is selected.  These patterns are anchored so domain
        # questions such as "Kafka raw topic ... ชื่ออะไร" are not hijacked.
        if _CREATOR_PATTERN.search(prepared_message):
            return self._decision(
                route="creator",
                confidence=1.0,
                reason="Matched deterministic MIC 9000 creator pattern.",
            )

        if _IDENTITY_PATTERN.search(prepared_message):
            return self._decision(
                route="identity",
                confidence=1.0,
                reason="Matched deterministic MIC 9000 identity pattern.",
            )

        if _GREETING_PATTERN.search(prepared_message):
            return self._decision(
                route="greeting",
                confidence=1.0,
                reason="Matched deterministic greeting pattern.",
            )

        if _THANKS_PATTERN.search(prepared_message):
            return self._decision(
                route="thanks",
                confidence=1.0,
                reason="Matched deterministic thanks pattern.",
            )

        if _CAPABILITY_PATTERN.search(prepared_message):
            return self._decision(
                route="capability_question",
                confidence=1.0,
                reason="Matched deterministic capability pattern.",
            )

        if _PRIVATE_INFORMATION_PATTERN.search(prepared_message):
            return self._decision(
                route="unsupported",
                confidence=1.0,
                reason="Matched sensitive or private-information pattern.",
            )

        # Strict selected-document mode means factual/domain questions should
        # be answered from that document, not from the generic route buckets.
        if document_scope and document_behavior == "strict_selected":
            return self._decision(
                route="document_qa",
                confidence=1.0,
                reason=(
                    "Strict selected-document mode is active; routing to "
                    "document_qa before deterministic domain buckets."
                ),
            )

        if _ESCALATION_PATTERN.search(prepared_message):
            return self._decision(
                route="human_escalation",
                confidence=0.98,
                reason="Matched deterministic human-escalation pattern.",
            )

        if history and _FOLLOWUP_HINT_PATTERN.search(prepared_message):
            return self._decision(
                route="follow_up",
                confidence=0.98,
                reason=(
                    "Matched context-dependent follow-up pattern with "
                    "stored conversation history."
                ),
            )

        if _KAFKA_PATTERN.search(prepared_message):
            return self._decision(
                route="kafka_iot_support",
                confidence=0.98,
                reason="Matched deterministic Kafka/IOT pattern.",
            )

        if _DEVELOPER_PATTERN.search(prepared_message):
            return self._decision(
                route="developer_support",
                confidence=0.96,
                reason="Matched deterministic developer-support pattern.",
            )

        if _ADMIN_PATTERN.search(prepared_message):
            return self._decision(
                route="admin_support",
                confidence=0.96,
                reason="Matched deterministic admin-support pattern.",
            )

        if _TECHNICAL_PATTERN.search(prepared_message):
            return self._decision(
                route="technical_troubleshooting",
                confidence=0.94,
                reason="Matched deterministic technical-troubleshooting pattern.",
            )

        if _COMPANY_PATTERN.search(prepared_message):
            return self._decision(
                route="company_info",
                confidence=0.90,
                reason="Matched deterministic company-information pattern.",
            )

        # Prefer-selected mode should not hijack operational support questions,
        # but ambiguous factual questions can be treated as document QA when a
        # selected document exists.
        if document_scope and document_behavior == "prefer_selected":
            return self._decision(
                route="document_qa",
                confidence=0.86,
                reason=(
                    "Selected document is present and no deterministic route "
                    "matched; preferring document_qa."
                ),
            )

        prompt = f"""
Recent conversation:
{self._format_history(history)}

Latest user message:
{prepared_message}

Choose the most appropriate route.
""".strip()

        payload, llm_result = self.llm_client.chat_json(
            messages=[{"role": "user", "content": prompt}],
            schema=GraphRoutePayload,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            think=False,
        )

        return RouteDecision(
            route=payload.route,
            confidence=payload.confidence,
            reason=payload.reason,
            router_called=True,
            llm_metrics=llm_result.get("metrics", {}),
        )
