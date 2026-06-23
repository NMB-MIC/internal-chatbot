from __future__ import annotations

import time
from dataclasses import (
    asdict,
    dataclass,
)
from typing import Any

import requests

from app.chat.conversation_service import (
    ConversationRagService,
    FollowupResolver,
)
from app.config import settings
from app.graph.router import (
    SupportRouteClassifier,
)
from app.graph.workflow import (
    InternalSupportGraph,
)
from app.llm.ollama_client import (
    OllamaClient,
)
from app.memory.sqlite_memory import (
    SqliteChatMemory,
)
from app.rag.embeddings import (
    BgeM3Embedder,
)
from app.rag.rag_chain import (
    GroundedRagAssistant,
)
from app.rag.retriever import (
    QdrantRetriever,
)
from app.rag.vector_store import (
    QdrantVectorStore,
)
from app.services.knowledge_base import (
    KnowledgeBaseService,
)


@dataclass(slots=True)
class BackendBundle:
    memory: SqliteChatMemory
    vector_store: QdrantVectorStore
    embedder: BgeM3Embedder
    llm_client: OllamaClient
    graph: InternalSupportGraph
    knowledge_base: KnowledgeBaseService
    embedding_warmup_seconds: float


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    ollama_ok: bool
    configured_model_available: bool
    qdrant_ok: bool
    collection_exists: bool
    indexed_point_count: int | None
    sqlite_ok: bool
    embedding_ok: bool
    embedding_warmup_seconds: float
    details: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


def build_backend(
    *,
    collection_name: str | None = None,
    warm_embedding: bool = True,
) -> BackendBundle:
    memory = SqliteChatMemory()

    memory.initialize()

    vector_store = (
        QdrantVectorStore(
            collection_name=(
                collection_name
                or settings
                .qdrant_collection_name
            )
        )
    )

    embedder = BgeM3Embedder(
        device=(
            settings
            .embedding_device
        ),
        batch_size=(
            settings
            .embedding_batch_size
        ),
        normalize_embeddings=(
            settings
            .embedding_normalize
        ),
        show_progress_bar=False,
    )

    warmup_seconds = 0.0

    if warm_embedding:
        started_at = (
            time.perf_counter()
        )

        _ = embedder.embed_query(
            "MIC 9000 startup warm-up probe."
        )

        warmup_seconds = (
            time.perf_counter()
            - started_at
        )

    llm_client = OllamaClient()

    retriever = QdrantRetriever(
        embedder=embedder,
        vector_store=(
            vector_store
        ),
    )

    rag_assistant = (
        GroundedRagAssistant(
            retriever=retriever,
            llm_client=llm_client,
        )
    )

    followup_resolver = (
        FollowupResolver(
            llm_client=llm_client,
        )
    )

    conversation_service = (
        ConversationRagService(
            memory=memory,
            followup_resolver=(
                followup_resolver
            ),
            rag_assistant=(
                rag_assistant
            ),
        )
    )

    route_classifier = (
        SupportRouteClassifier(
            llm_client=llm_client,
        )
    )

    graph = InternalSupportGraph(
        memory=memory,
        route_classifier=(
            route_classifier
        ),
        conversation_service=(
            conversation_service
        ),
    )

    knowledge_base = (
        KnowledgeBaseService(
            vector_store=(
                vector_store
            ),
            embedder=embedder,
        )
    )

    return BackendBundle(
        memory=memory,
        vector_store=vector_store,
        embedder=embedder,
        llm_client=llm_client,
        graph=graph,
        knowledge_base=(
            knowledge_base
        ),
        embedding_warmup_seconds=round(
            warmup_seconds,
            4,
        ),
    )


def check_runtime_health(
    backend: BackendBundle,
) -> RuntimeHealth:
    details: dict[
        str,
        Any,
    ] = {}

    ollama_ok = False
    configured_model_available = (
        False
    )

    try:
        response = requests.get(
            (
                settings
                .ollama_base_url
                + "/api/tags"
            ),
            timeout=5,
        )

        response.raise_for_status()

        models = response.json().get(
            "models",
            [],
        )

        model_names = {
            str(
                model.get(
                    "name",
                    "",
                )
            )
            for model in models
        }

        ollama_ok = True

        configured_model_available = (
            settings.ollama_model
            in model_names
        )

        details[
            "ollama_models"
        ] = sorted(
            model_names
        )

    except Exception as exc:
        details[
            "ollama_error"
        ] = repr(
            exc
        )

    qdrant_ok = False
    collection_exists = False
    indexed_point_count: (
        int
        | None
    ) = None

    try:
        qdrant_health = (
            backend
            .vector_store
            .healthcheck()
        )

        qdrant_ok = bool(
            qdrant_health[
                "server_reachable"
            ]
        )

        collection_exists = bool(
            qdrant_health[
                "collection_exists"
            ]
        )

        if collection_exists:
            indexed_point_count = (
                backend
                .vector_store
                .count_points()
            )

        details[
            "qdrant"
        ] = qdrant_health

    except Exception as exc:
        details[
            "qdrant_error"
        ] = repr(
            exc
        )

    sqlite_ok = False

    try:
        pragmas = (
            backend
            .memory
            .inspect_pragmas()
        )

        sqlite_ok = (
            pragmas.get(
                "journal_mode",
                "",
            ).lower()
            == "wal"
            and pragmas.get(
                "foreign_keys"
            )
            == 1
        )

        details[
            "sqlite"
        ] = pragmas

    except Exception as exc:
        details[
            "sqlite_error"
        ] = repr(
            exc
        )

    embedding_ok = False

    try:
        vector = (
            backend
            .embedder
            .embed_query(
                "MIC 9000 health probe."
            )
        )

        embedding_ok = (
            tuple(
                vector.shape
            )
            == (
                1024,
            )
        )

    except Exception as exc:
        details[
            "embedding_error"
        ] = repr(
            exc
        )

    return RuntimeHealth(
        ollama_ok=ollama_ok,
        configured_model_available=(
            configured_model_available
        ),
        qdrant_ok=qdrant_ok,
        collection_exists=(
            collection_exists
        ),
        indexed_point_count=(
            indexed_point_count
        ),
        sqlite_ok=sqlite_ok,
        embedding_ok=embedding_ok,
        embedding_warmup_seconds=(
            backend
            .embedding_warmup_seconds
        ),
        details=details,
    )