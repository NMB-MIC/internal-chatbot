from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from app.config import settings
from app.rag.chunk_models import RetrievalChunk
from app.rag.embeddings import EmbeddedChunks


# Stable namespace for deterministic Qdrant UUID generation.
_QDRANT_POINT_NAMESPACE = uuid.UUID(
    "922127d7-5c34-47bb-a627-8a2c1df2b56e"
)


@dataclass(frozen=True, slots=True)
class QdrantIndexReport:
    collection_name: str
    input_chunks: int
    indexed_points: int
    skipped_exact_duplicates: int
    duplicate_groups: int
    vector_dimension: int
    upsert_batch_size: int
    elapsed_seconds: float
    point_count_after_upsert: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(frozen=True, slots=True)
class QdrantSnapshotReport:
    enabled: bool
    created: bool
    collection_name: str
    snapshot_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QdrantSearchHit:
    point_id: str
    score: float
    text: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "score": self.score,
            "text": self.text,
            "payload": self.payload,
        }


def _json_safe(
    value: Any,
) -> Any:
    """
    Convert common Python and NumPy values into JSON-safe values
    suitable for Qdrant payloads.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    return str(value)


def _omit_none_values(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value is not None
    }


def _chunk_to_point_id(
    chunk: RetrievalChunk,
) -> str:
    """
    Qdrant supports UUID point IDs.

    Retrieval chunk IDs are SHA-256 hex strings, so we convert each
    chunk ID into a stable UUID. Re-indexing the same chunk generates
    the same Qdrant point ID.
    """

    return str(
        uuid.uuid5(
            _QDRANT_POINT_NAMESPACE,
            chunk.chunk_id,
        )
    )


def _batched(
    items: Sequence[Any],
    *,
    batch_size: int,
) -> Iterable[
    Sequence[Any]
]:
    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    for start_index in range(
        0,
        len(items),
        batch_size,
    ):
        yield items[
            start_index:
            start_index + batch_size
        ]


def _content_sha256(
    chunk: RetrievalChunk,
) -> str:
    value = chunk.metadata.get(
        "content_sha256"
    )

    if not value:
        raise ValueError(
            "Chunk is missing content_sha256 metadata."
        )

    return str(value)


def _group_chunks_by_exact_content(
    chunks: Sequence[RetrievalChunk],
) -> dict[
    str,
    list[RetrievalChunk],
]:
    grouped: dict[
        str,
        list[RetrievalChunk],
    ] = {}

    for chunk in chunks:
        grouped.setdefault(
            _content_sha256(
                chunk
            ),
            [],
        ).append(
            chunk
        )

    return grouped


def _select_representative_chunks(
    chunks: Sequence[RetrievalChunk],
) -> tuple[
    list[RetrievalChunk],
    dict[
        str,
        list[RetrievalChunk],
    ],
]:
    grouped = (
        _group_chunks_by_exact_content(
            chunks
        )
    )

    representatives = [
        grouped_chunks[0]
        for grouped_chunks in grouped.values()
    ]

    return (
        representatives,
        grouped,
    )


def _build_payload(
    *,
    representative: RetrievalChunk,
    grouped_occurrences: Sequence[
        RetrievalChunk
    ],
) -> dict[str, Any]:
    duplicate_chunks = list(
        grouped_occurrences[1:]
    )

    duplicate_chunk_ids = [
        chunk.chunk_id
        for chunk in duplicate_chunks
    ]

    occurrence_source_paths = sorted({
        str(
            chunk.metadata.get(
                "source_path",
                "",
            )
        )
        for chunk in grouped_occurrences
    })

    metadata = _json_safe(
        representative.metadata
    )

    payload = {
        **metadata,
        "chunk_id": (
            representative.chunk_id
        ),
        "text": representative.text,
        "duplicate_count": len(
            duplicate_chunks
        ),
        "duplicate_chunk_ids": (
            duplicate_chunk_ids
        ),
        "occurrence_source_paths": (
            occurrence_source_paths
        ),
    }

    return _omit_none_values(
        payload
    )


class QdrantVectorStore:
    """
    Qdrant-backed dense vector store.

    The collection uses one named dense vector so that a sparse
    vector can be added later for hybrid retrieval.
    """

    KEYWORD_INDEX_FIELDS = (
        "category",
        "source_path",
        "source_extension",
        "loader_type",
        "document_id",
        "unit_type",
        "sheet_name",
        "source_sha256",
    )

    INTEGER_INDEX_FIELDS = (
        "page_number",
        "row_start",
        "row_end",
        "chunk_index_within_unit",
    )

    def __init__(
        self,
        *,
        url: str = (
            settings.qdrant_url
        ),
        collection_name: str = (
            settings
            .qdrant_collection_name
        ),
        api_key: str | None = (
            settings.qdrant_api_key
        ),
        timeout_seconds: int = (
            settings
            .qdrant_timeout_seconds
        ),
        dense_vector_name: str = (
            settings
            .qdrant_dense_vector_name
        ),
    ) -> None:
        self.url = url.rstrip("/")
        self.collection_name = (
            collection_name
        )
        self.api_key = api_key
        self.timeout_seconds = (
            timeout_seconds
        )
        self.dense_vector_name = (
            dense_vector_name
        )

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=self.timeout_seconds,
        )

    def healthcheck(
        self,
    ) -> dict[str, Any]:
        collections = (
            self.client
            .get_collections()
            .collections
        )

        return {
            "url": self.url,
            "server_reachable": True,
            "collection_name": (
                self.collection_name
            ),
            "collection_exists": (
                self.collection_exists()
            ),
            "available_collections": [
                collection.name
                for collection in collections
            ],
        }

    def collection_exists(
        self,
    ) -> bool:
        collection_names = {
            collection.name
            for collection in (
                self.client
                .get_collections()
                .collections
            )
        }

        return (
            self.collection_name
            in collection_names
        )

    def delete_collection(
        self,
    ) -> bool:
        if not self.collection_exists():
            return False

        self.client.delete_collection(
            collection_name=(
                self.collection_name
            ),
        )

        return True

    def create_collection(
        self,
        *,
        vector_dimension: int,
        recreate: bool = False,
    ) -> None:
        if vector_dimension < 1:
            raise ValueError(
                "vector_dimension must be positive."
            )

        exists = (
            self.collection_exists()
        )

        if exists and recreate:
            self.delete_collection()
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=(
                    self.collection_name
                ),
                vectors_config={
                    self.dense_vector_name: (
                        models.VectorParams(
                            size=(
                                vector_dimension
                            ),
                            distance=(
                                models
                                .Distance
                                .COSINE
                            ),
                        )
                    )
                },
            )

        self.create_payload_indexes()

    def create_payload_indexes(
        self,
    ) -> None:
        """
        Create indexes before ingestion for fields used by filters.
        """

        for field_name in (
            self.KEYWORD_INDEX_FIELDS
        ):
            self.client.create_payload_index(
                collection_name=(
                    self.collection_name
                ),
                field_name=field_name,
                field_schema=(
                    models
                    .PayloadSchemaType
                    .KEYWORD
                ),
                wait=True,
            )

        for field_name in (
            self.INTEGER_INDEX_FIELDS
        ):
            self.client.create_payload_index(
                collection_name=(
                    self.collection_name
                ),
                field_name=field_name,
                field_schema=(
                    models
                    .PayloadSchemaType
                    .INTEGER
                ),
                wait=True,
            )

    def get_collection_info(
        self,
    ) -> Any:
        return self.client.get_collection(
            collection_name=(
                self.collection_name
            ),
        )

    def count_points(
        self,
        *,
        query_filter: (
            models.Filter
            | None
        ) = None,
    ) -> int:
        result = self.client.count(
            collection_name=(
                self.collection_name
            ),
            count_filter=query_filter,
            exact=True,
        )

        return int(
            result.count
        )

    def upsert_embedded_chunks(
        self,
        *,
        embedded_chunks: EmbeddedChunks,
        all_chunks: Sequence[
            RetrievalChunk
        ]
        | None = None,
        batch_size: int = (
            settings
            .qdrant_upsert_batch_size
        ),
    ) -> QdrantIndexReport:
        """
        Upsert unique representative chunks.

        all_chunks:
            Include the full pre-deduplicated chunk list so duplicate
            occurrence references can be attached to representative
            payloads.

        embedded_chunks:
            Must contain one vector per representative chunk.
        """

        started_at = (
            time.perf_counter()
        )

        representatives = list(
            embedded_chunks.chunks
        )

        vectors = np.asarray(
            embedded_chunks.vectors,
            dtype=np.float32,
        )

        if len(representatives) != len(
            vectors
        ):
            raise ValueError(
                "Representative chunk count and vector count do not match."
            )

        original_chunks = list(
            all_chunks
            if all_chunks is not None
            else representatives
        )

        grouped_occurrences = (
            _group_chunks_by_exact_content(
                original_chunks
            )
        )

        representative_hashes = [
            _content_sha256(chunk)
            for chunk in representatives
        ]

        if len(set(
            representative_hashes
        )) != len(
            representative_hashes
        ):
            raise ValueError(
                "embedded_chunks still contains exact duplicates."
            )

        points: list[
            models.PointStruct
        ] = []

        for (
            representative,
            vector,
        ) in zip(
            representatives,
            vectors,
            strict=True,
        ):
            content_sha256 = (
                _content_sha256(
                    representative
                )
            )

            occurrences = (
                grouped_occurrences
                .get(
                    content_sha256,
                    [
                        representative
                    ],
                )
            )

            point_id = _chunk_to_point_id(
                representative
            )

            payload = _build_payload(
                representative=(
                    representative
                ),
                grouped_occurrences=(
                    occurrences
                ),
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        self
                        .dense_vector_name: (
                            vector.tolist()
                        )
                    },
                    payload=payload,
                )
            )

        for batch in _batched(
            points,
            batch_size=batch_size,
        ):
            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=list(batch),
                wait=True,
            )

        point_count = (
            self.count_points()
        )

        duplicate_groups = sum(
            len(group) > 1
            for group in (
                grouped_occurrences
                .values()
            )
        )

        skipped_duplicates = sum(
            max(
                0,
                len(group) - 1,
            )
            for group in (
                grouped_occurrences
                .values()
            )
        )

        elapsed_seconds = (
            time.perf_counter()
            - started_at
        )

        return QdrantIndexReport(
            collection_name=(
                self.collection_name
            ),
            input_chunks=len(
                original_chunks
            ),
            indexed_points=len(
                points
            ),
            skipped_exact_duplicates=(
                skipped_duplicates
            ),
            duplicate_groups=(
                duplicate_groups
            ),
            vector_dimension=int(
                vectors.shape[1]
            ),
            upsert_batch_size=(
                batch_size
            ),
            elapsed_seconds=round(
                elapsed_seconds,
                4,
            ),
            point_count_after_upsert=(
                point_count
            ),
        )

    def create_snapshot_report(
        self,
        *,
        enabled: bool = True,
    ) -> QdrantSnapshotReport:
        if not enabled:
            return QdrantSnapshotReport(
                enabled=False,
                created=False,
                collection_name=(
                    self.collection_name
                ),
                error="snapshot disabled",
            )

        if not self.collection_exists():
            return QdrantSnapshotReport(
                enabled=True,
                created=False,
                collection_name=(
                    self.collection_name
                ),
                error="collection does not exist",
            )

        try:
            snapshot = (
                self.client
                .create_snapshot(
                    collection_name=(
                        self.collection_name
                    )
                )
            )

            snapshot_name = (
                getattr(
                    snapshot,
                    "name",
                    None,
                )
                or getattr(
                    snapshot,
                    "snapshot_name",
                    None,
                )
                or str(snapshot)
            )

            return QdrantSnapshotReport(
                enabled=True,
                created=True,
                collection_name=(
                    self.collection_name
                ),
                snapshot_name=(
                    snapshot_name
                ),
            )

        except Exception as exc:
            return QdrantSnapshotReport(
                enabled=True,
                created=False,
                collection_name=(
                    self.collection_name
                ),
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    def rebuild_collection(
        self,
        *,
        embedded_chunks: EmbeddedChunks,
        all_chunks: Sequence[
            RetrievalChunk
        ]
        | None = None,
        batch_size: int = (
            settings
            .qdrant_upsert_batch_size
        ),
    ) -> QdrantIndexReport:
        self.create_collection(
            vector_dimension=(
                embedded_chunks
                .vector_dimension
            ),
            recreate=True,
        )

        return self.upsert_embedded_chunks(
            embedded_chunks=(
                embedded_chunks
            ),
            all_chunks=all_chunks,
            batch_size=batch_size,
        )

    @staticmethod
    def build_match_filter(
        **criteria: Any,
    ) -> models.Filter | None:
        """
        Build an AND filter.

        Scalar:
            category="admin_support"

        List:
            category=[
                "admin_support",
                "company_info",
            ]
        """

        must_conditions: list[
            models.FieldCondition
        ] = []

        for (
            field_name,
            expected_value,
        ) in criteria.items():
            if expected_value is None:
                continue

            if isinstance(
                expected_value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):
                values = list(
                    expected_value
                )

                if not values:
                    continue

                match = models.MatchAny(
                    any=values
                )

            else:
                match = models.MatchValue(
                    value=expected_value
                )

            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=match,
                )
            )

        if not must_conditions:
            return None

        return models.Filter(
            must=must_conditions
        )

    def search(
        self,
        *,
        query_vector: np.ndarray,
        limit: int = 5,
        score_threshold: (
            float
            | None
        ) = None,
        query_filter: (
            models.Filter
            | None
        ) = None,
    ) -> list[QdrantSearchHit]:
        """
        Search with the modern universal query API.

        A compatibility fallback is included for older qdrant-client
        versions that still expose search() but not query_points().
        """

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        ).reshape(
            -1
        )

        if hasattr(
            self.client,
            "query_points",
        ):
            response = (
                self.client
                .query_points(
                    collection_name=(
                        self
                        .collection_name
                    ),
                    query=(
                        vector.tolist()
                    ),
                    using=(
                        self
                        .dense_vector_name
                    ),
                    query_filter=(
                        query_filter
                    ),
                    limit=limit,
                    score_threshold=(
                        score_threshold
                    ),
                    with_payload=True,
                    with_vectors=False,
                )
            )

            scored_points = (
                response.points
            )

        else:
            scored_points = (
                self.client
                .search(
                    collection_name=(
                        self
                        .collection_name
                    ),
                    query_vector=(
                        models.NamedVector(
                            name=(
                                self
                                .dense_vector_name
                            ),
                            vector=(
                                vector.tolist()
                            ),
                        )
                    ),
                    query_filter=(
                        query_filter
                    ),
                    limit=limit,
                    score_threshold=(
                        score_threshold
                    ),
                    with_payload=True,
                    with_vectors=False,
                )
            )

        hits: list[
            QdrantSearchHit
        ] = []

        for point in scored_points:
            payload = dict(
                point.payload
                or {}
            )

            hits.append(
                QdrantSearchHit(
                    point_id=str(
                        point.id
                    ),
                    score=float(
                        point.score
                    ),
                    text=str(
                        payload.get(
                            "text",
                            "",
                        )
                    ),
                    payload=payload,
                )
            )

        return hits

    def scroll_payloads(
        self,
        *,
        query_filter: (
            models.Filter
            | None
        ) = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        points, _ = self.client.scroll(
            collection_name=(
                self.collection_name
            ),
            scroll_filter=(
                query_filter
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                "point_id": str(
                    point.id
                ),
                **dict(
                    point.payload
                    or {}
                ),
            }
            for point in points
        ]