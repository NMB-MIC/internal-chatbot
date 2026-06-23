from __future__ import annotations

import time
from dataclasses import (
    asdict,
    dataclass,
)
from typing import Any

from app.config import settings
from app.rag.embeddings import (
    BgeM3Embedder,
)
from app.rag.vector_store import (
    QdrantSearchHit,
    QdrantVectorStore,
)
from app.rag.lexical import (
    LexicalChunkRetriever,
)

@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    embedding_seconds: float
    search_seconds: float
    total_seconds: float

    def to_dict(
        self,
    ) -> dict[str, float]:
        return asdict(
            self
        )


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    requested_top_k_initial: (
        int
        | None
    )

    requested_top_k_final: (
        int
        | None
    )

    requested_similarity_threshold: (
        float
        | None
    )

    effective_top_k_initial: int
    effective_top_k_final: int
    effective_similarity_threshold: (
        float
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    point_id: str
    score: float
    text: str
    payload: dict[str, Any]

    @classmethod
    def from_qdrant_hit(
        cls,
        hit: QdrantSearchHit,
    ) -> "RetrievedChunk":
        return cls(
            point_id=hit.point_id,
            score=hit.score,
            text=hit.text,
            payload=hit.payload,
        )


@dataclass(slots=True)
class RetrievalResult:
    query: str
    raw_hits: list[
        RetrievedChunk
    ]
    accepted_hits: list[
        RetrievedChunk
    ]
    similarity_threshold: float
    top_k_initial: int
    top_k_final: int
    filter_criteria: dict[
        str,
        Any,
    ]
    metrics: RetrievalMetrics
    diagnostics: (
        RetrievalDiagnostics
    )

    @property
    def top_score(
        self,
    ) -> float | None:
        if not self.raw_hits:
            return None

        return (
            self.raw_hits[0]
            .score
        )

    @property
    def has_evidence(
        self,
    ) -> bool:
        return bool(
            self.accepted_hits
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        return {
            "query": self.query,
            "raw_hit_count": len(
                self.raw_hits
            ),
            "accepted_hit_count": len(
                self.accepted_hits
            ),
            "top_score": (
                round(
                    self.top_score,
                    6,
                )
                if self.top_score
                is not None
                else None
            ),
            "similarity_threshold": (
                self
                .similarity_threshold
            ),
            "top_k_initial": (
                self.top_k_initial
            ),
            "top_k_final": (
                self.top_k_final
            ),
            "filter_criteria": (
                self.filter_criteria
            ),
            "metrics": (
                self.metrics
                .to_dict()
            ),
            "diagnostics": (
                self.diagnostics
                .to_dict()
            ),
        }

    def debug_summary(
        self,
    ) -> dict[str, Any]:
        return self.summary()


class QdrantRetriever:
    """
    CPU BGE-M3 query embedding + Qdrant dense retrieval.

    The retriever fetches an initial candidate set and applies the
    score threshold locally so rejected candidates remain visible
    during development and tuning.

    Per-call overrides are supported for document-QA flows while the
    constructor values remain the defaults for ordinary RAG queries.
    """

    def __init__(
        self,
        *,
        embedder: BgeM3Embedder,
        vector_store: (
            QdrantVectorStore
        ),
        top_k_initial: int = (
            settings
            .rag_top_k_initial
        ),
        top_k_final: int = (
            settings
            .rag_top_k_final
        ),
        similarity_threshold: float = (
            settings
            .rag_similarity_threshold
        ),
    ) -> None:
        if top_k_initial < 1:
            raise ValueError(
                "top_k_initial must be positive."
            )

        if top_k_final < 1:
            raise ValueError(
                "top_k_final must be positive."
            )

        if (
            top_k_final
            > top_k_initial
        ):
            raise ValueError(
                "top_k_final must not exceed "
                "top_k_initial."
            )

        self.embedder = embedder

        self.vector_store = (
            vector_store
        )

        self.top_k_initial = (
            top_k_initial
        )

        self.top_k_final = (
            top_k_final
        )

        self.similarity_threshold = (
            similarity_threshold
        )
        
        self.lexical_retriever = (
            LexicalChunkRetriever(
                vector_store=(
                    vector_store
                ),
                max_scroll_points=5000,
                scroll_batch_size=256,
                neighbor_window=1,
            )
        )

        self.hybrid_rrf_k = 60

    def retrieve(
        self,
        query: str,
        *,
        filter_criteria: (
            dict[str, Any]
            | None
        ) = None,
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
    ) -> RetrievalResult:
        prepared_query = (
            query.strip()
        )

        if not prepared_query:
            raise ValueError(
                "Query must not be empty."
            )

        effective_top_k_initial = (
            self.top_k_initial
            if top_k_initial
            is None
            else top_k_initial
        )

        effective_top_k_final = (
            self.top_k_final
            if top_k_final
            is None
            else top_k_final
        )

        effective_similarity_threshold = (
            self
            .similarity_threshold
            if similarity_threshold
            is None
            else similarity_threshold
        )

        if (
            effective_top_k_initial
            < 1
        ):
            raise ValueError(
                "effective_top_k_initial "
                "must be positive."
            )

        if (
            effective_top_k_final
            < 1
        ):
            raise ValueError(
                "effective_top_k_final "
                "must be positive."
            )

        if (
            effective_top_k_final
            > effective_top_k_initial
        ):
            raise ValueError(
                "effective_top_k_final "
                "must not exceed "
                "effective_top_k_initial."
            )

        criteria = dict(
            filter_criteria
            or {}
        )

        query_filter = (
            self.vector_store
            .build_match_filter(
                **criteria
            )
        )

        retrieval_started_at = (
            time.perf_counter()
        )

        embedding_started_at = (
            time.perf_counter()
        )

        query_vector = (
            self.embedder
            .embed_query(
                prepared_query
            )
        )

        embedding_seconds = (
            time.perf_counter()
            - embedding_started_at
        )

        search_started_at = (
            time.perf_counter()
        )

        qdrant_hits = (
            self.vector_store
            .search(
                query_vector=(
                    query_vector
                ),
                limit=(
                    effective_top_k_initial
                ),
                query_filter=(
                    query_filter
                ),
            )
        )

        search_seconds = (
            time.perf_counter()
            - search_started_at
        )

        raw_hits = [
            RetrievedChunk
            .from_qdrant_hit(
                hit
            )
            for hit in qdrant_hits
        ]
        
        if self._should_use_hybrid(
            filter_criteria=criteria
        ):
            lexical_hits = (
                self.lexical_retriever
                .search(
                    query=(
                        prepared_query
                    ),
                    query_filter=(
                        query_filter
                    ),
                    limit=max(
                        effective_top_k_initial,
                        32,
                    ),
                    include_neighbors=True,
                )
            )

            raw_hits = (
                self
                ._fuse_dense_and_lexical(
                    dense_hits=raw_hits,
                    lexical_hits=lexical_hits,
                    top_k_initial=(
                        effective_top_k_initial
                    ),
                )
            )

        accepted_hits = [
            hit
            for hit in raw_hits
            if (
                hit.score
                >= effective_similarity_threshold
            )
        ][
            :effective_top_k_final
        ]

        total_seconds = (
            time.perf_counter()
            - retrieval_started_at
        )

        diagnostics = (
            RetrievalDiagnostics(
                requested_top_k_initial=(
                    top_k_initial
                ),
                requested_top_k_final=(
                    top_k_final
                ),
                requested_similarity_threshold=(
                    similarity_threshold
                ),
                effective_top_k_initial=(
                    effective_top_k_initial
                ),
                effective_top_k_final=(
                    effective_top_k_final
                ),
                effective_similarity_threshold=(
                    effective_similarity_threshold
                ),
            )
        )

        return RetrievalResult(
            query=prepared_query,
            raw_hits=raw_hits,
            accepted_hits=(
                accepted_hits
            ),
            similarity_threshold=(
                effective_similarity_threshold
            ),
            top_k_initial=(
                effective_top_k_initial
            ),
            top_k_final=(
                effective_top_k_final
            ),
            filter_criteria=criteria,
            metrics=RetrievalMetrics(
                embedding_seconds=round(
                    embedding_seconds,
                    4,
                ),
                search_seconds=round(
                    search_seconds,
                    4,
                ),
                total_seconds=round(
                    total_seconds,
                    4,
                ),
            ),
            diagnostics=(
                diagnostics
            ),
        )
        
    @staticmethod
    def _should_use_hybrid(
        *,
        filter_criteria: dict[str, Any],
    ) -> bool:
        # Keep ordinary internal-support retrieval conservative.
        # Use hybrid mainly for selected-document QA where exact strings
        # such as commands and UI labels matter.
        return bool(
            filter_criteria.get(
                "source_path"
            )
        )

    def _fuse_dense_and_lexical(
        self,
        *,
        dense_hits: list[
            RetrievedChunk
        ],
        lexical_hits: list[Any],
        top_k_initial: int,
    ) -> list[RetrievedChunk]:
        dense_rank = {
            hit.point_id: rank
            for rank, hit in enumerate(
                dense_hits,
                start=1,
            )
        }

        lexical_rank = {
            hit.point_id: rank
            for rank, hit in enumerate(
                lexical_hits,
                start=1,
            )
        }

        chunk_by_id: dict[
            str,
            RetrievedChunk,
        ] = {
            hit.point_id: hit
            for hit in dense_hits
        }

        for lexical_hit in lexical_hits:
            existing = chunk_by_id.get(
                lexical_hit.point_id
            )

            lexical_score = max(
                0.30,
                min(
                    0.99,
                    0.35
                    + float(
                        lexical_hit.score
                    ),
                ),
            )

            if existing is None:
                payload = dict(
                    lexical_hit.payload
                    or {}
                )

                payload[
                    "retrieval_match_type"
                ] = "lexical"

                payload[
                    "lexical_matched_terms"
                ] = list(
                    lexical_hit
                    .matched_terms
                )

                chunk_by_id[
                    lexical_hit.point_id
                ] = RetrievedChunk(
                    point_id=(
                        lexical_hit.point_id
                    ),
                    score=(
                        lexical_score
                    ),
                    text=(
                        lexical_hit.text
                    ),
                    payload=payload,
                )

            elif lexical_score > existing.score:
                payload = dict(
                    existing.payload
                )

                payload[
                    "retrieval_match_type"
                ] = "hybrid"

                payload[
                    "lexical_matched_terms"
                ] = list(
                    lexical_hit
                    .matched_terms
                )

                chunk_by_id[
                    existing.point_id
                ] = RetrievedChunk(
                    point_id=existing.point_id,
                    score=lexical_score,
                    text=existing.text,
                    payload=payload,
                )

        fused_rows = []

        for point_id, chunk in (
            chunk_by_id.items()
        ):
            score = 0.0

            if point_id in dense_rank:
                score += (
                    1.0
                    / (
                        self.hybrid_rrf_k
                        + dense_rank[
                            point_id
                        ]
                    )
                )

            if point_id in lexical_rank:
                score += (
                    1.0
                    / (
                        self.hybrid_rrf_k
                        + lexical_rank[
                            point_id
                        ]
                    )
                )

            # Stable tie-breaker:
            # Prefer higher chunk score after RRF.
            fused_rows.append(
                (
                    score,
                    chunk.score,
                    point_id,
                    chunk,
                )
            )

        fused_rows.sort(
            key=lambda row: (
                row[0],
                row[1],
            ),
            reverse=True,
        )

        return [
            row[3]
            for row in fused_rows[
                :top_k_initial
            ]
        ]
