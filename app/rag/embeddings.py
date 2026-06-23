from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.rag.chunk_models import RetrievalChunk


@dataclass(frozen=True, slots=True)
class EmbeddingStats:
    input_count: int
    vector_dimension: int
    elapsed_seconds: float
    texts_per_second: float
    device: str
    normalized: bool
    batch_size: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class EmbeddedChunks:
    chunks: list[RetrievalChunk]
    vectors: np.ndarray
    stats: EmbeddingStats

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(
                "Embedded chunk vectors must be a 2D matrix."
            )

        if len(self.chunks) != len(self.vectors):
            raise ValueError(
                "Chunk count and vector count must match."
            )

    @property
    def vector_dimension(self) -> int:
        if self.vectors.size == 0:
            return 0

        return int(
            self.vectors.shape[1]
        )


@dataclass(frozen=True, slots=True)
class ExactDuplicateGroup:
    content_sha256: str
    representative_chunk_id: str
    duplicate_chunk_ids: tuple[str, ...]
    source_paths: tuple[str, ...]

    @property
    def duplicate_count(self) -> int:
        return len(
            self.duplicate_chunk_ids
        )

    def to_dict(self) -> dict:
        return {
            "content_sha256": self.content_sha256,
            "representative_chunk_id": (
                self.representative_chunk_id
            ),
            "duplicate_chunk_ids": (
                self.duplicate_chunk_ids
            ),
            "source_paths": self.source_paths,
            "duplicate_count": (
                self.duplicate_count
            ),
        }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _normalise_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    if matrix.size == 0:
        return matrix

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    safe_norms = np.where(
        norms == 0,
        1.0,
        norms,
    )

    return matrix / safe_norms


def cosine_similarity_matrix(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between two vector matrices.

    Input shapes:
        left_vectors  -> [left_count, dimension]
        right_vectors -> [right_count, dimension]

    Output shape:
        [left_count, right_count]
    """

    left = np.asarray(
        left_vectors,
        dtype=np.float32,
    )

    right = np.asarray(
        right_vectors,
        dtype=np.float32,
    )

    if left.ndim == 1:
        left = left.reshape(
            1,
            -1,
        )

    if right.ndim == 1:
        right = right.reshape(
            1,
            -1,
        )

    if left.shape[1] != right.shape[1]:
        raise ValueError(
            "Vector dimensions must match."
        )

    return (
        _normalise_rows(left)
        @ _normalise_rows(right).T
    )


def find_exact_duplicate_groups(
    chunks: Iterable[RetrievalChunk],
) -> list[ExactDuplicateGroup]:
    """
    Find chunks whose final retrieval text is exactly identical.

    The first chunk in each group becomes the representative.
    We will later index only representatives while preserving
    references to duplicate sources.
    """

    grouped_chunks: dict[
        str,
        list[RetrievalChunk],
    ] = {}

    for chunk in chunks:
        content_sha256 = (
            chunk.metadata.get(
                "content_sha256"
            )
            or _sha256_text(
                chunk.text
            )
        )

        grouped_chunks.setdefault(
            content_sha256,
            [],
        ).append(
            chunk
        )

    duplicate_groups: list[
        ExactDuplicateGroup
    ] = []

    for (
        content_sha256,
        grouped,
    ) in grouped_chunks.items():
        if len(grouped) <= 1:
            continue

        representative = grouped[0]
        duplicates = grouped[1:]

        duplicate_groups.append(
            ExactDuplicateGroup(
                content_sha256=(
                    content_sha256
                ),
                representative_chunk_id=(
                    representative.chunk_id
                ),
                duplicate_chunk_ids=tuple(
                    chunk.chunk_id
                    for chunk in duplicates
                ),
                source_paths=tuple(
                    sorted({
                        str(
                            chunk.metadata.get(
                                "source_path",
                                "",
                            )
                        )
                        for chunk in grouped
                    })
                ),
            )
        )

    return sorted(
        duplicate_groups,
        key=lambda group: (
            -group.duplicate_count,
            group.content_sha256,
        ),
    )


def select_unique_chunks(
    chunks: Iterable[RetrievalChunk],
) -> tuple[
    list[RetrievalChunk],
    list[str],
]:
    """
    Keep the first exact-text occurrence of each chunk.

    Returns:
        unique_chunks
        skipped_duplicate_chunk_ids
    """

    unique_chunks: list[
        RetrievalChunk
    ] = []

    skipped_duplicate_ids: list[
        str
    ] = []

    seen_hashes: set[str] = set()

    for chunk in chunks:
        content_sha256 = (
            chunk.metadata.get(
                "content_sha256"
            )
            or _sha256_text(
                chunk.text
            )
        )

        if content_sha256 in seen_hashes:
            skipped_duplicate_ids.append(
                chunk.chunk_id
            )

            continue

        seen_hashes.add(
            content_sha256
        )

        unique_chunks.append(
            chunk
        )

    return (
        unique_chunks,
        skipped_duplicate_ids,
    )


class BgeM3Embedder:
    """
    Lazy-loading dense embedding wrapper for BAAI/bge-m3.

    The first model access downloads the model if it is not
    already present in the local Hugging Face cache.
    """

    def __init__(
        self,
        *,
        model_name: str = (
            settings.embedding_model_name
        ),
        device: str = (
            settings.embedding_device
        ),
        batch_size: int = (
            settings.embedding_batch_size
        ),
        normalize_embeddings: bool = (
            settings.embedding_normalize
        ),
        show_progress_bar: bool = (
            settings.embedding_show_progress
        ),
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = (
            normalize_embeddings
        )
        self.show_progress_bar = (
            show_progress_bar
        )

        self._model: (
            SentenceTransformer
            | None
        ) = None

    @property
    def model(
        self,
    ) -> SentenceTransformer:
        if self._model is None:
            self._model = (
                SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
            )

        return self._model

    @property
    def vector_dimension(self) -> int:
        dimension = (
            self
            .model
            .get_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "Embedding dimension could not be determined."
            )

        return int(
            dimension
        )

    @property
    def max_sequence_length(self) -> int:
        return int(
            self.model.max_seq_length
        )

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: (
            bool
            | None
        ) = None,
    ) -> tuple[
        np.ndarray,
        EmbeddingStats,
    ]:
        prepared_texts = [
            text.strip()
            for text in texts
            if text.strip()
        ]

        effective_batch_size = (
            batch_size
            or self.batch_size
        )

        effective_show_progress = (
            self.show_progress_bar
            if show_progress_bar is None
            else show_progress_bar
        )

        if not prepared_texts:
            empty_vectors = np.empty(
                (
                    0,
                    self.vector_dimension,
                ),
                dtype=np.float32,
            )

            stats = EmbeddingStats(
                input_count=0,
                vector_dimension=(
                    self.vector_dimension
                ),
                elapsed_seconds=0.0,
                texts_per_second=0.0,
                device=self.device,
                normalized=(
                    self.normalize_embeddings
                ),
                batch_size=(
                    effective_batch_size
                ),
            )

            return (
                empty_vectors,
                stats,
            )

        started_at = time.perf_counter()

        vectors = self.model.encode(
            prepared_texts,
            batch_size=(
                effective_batch_size
            ),
            show_progress_bar=(
                effective_show_progress
            ),
            convert_to_numpy=True,
            normalize_embeddings=(
                self.normalize_embeddings
            ),
            device=self.device,
        )

        elapsed_seconds = (
            time.perf_counter()
            - started_at
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        stats = EmbeddingStats(
            input_count=len(
                prepared_texts
            ),
            vector_dimension=int(
                vectors.shape[1]
            ),
            elapsed_seconds=round(
                elapsed_seconds,
                4,
            ),
            texts_per_second=round(
                len(prepared_texts)
                / elapsed_seconds,
                4,
            ),
            device=self.device,
            normalized=(
                self.normalize_embeddings
            ),
            batch_size=(
                effective_batch_size
            ),
        )

        return (
            vectors,
            stats,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        vectors, _ = self.encode_texts(
            [query],
            show_progress_bar=False,
        )

        return vectors[0]

    def embed_chunks(
        self,
        chunks: Sequence[
            RetrievalChunk
        ],
        *,
        batch_size: int | None = None,
        show_progress_bar: (
            bool
            | None
        ) = None,
    ) -> EmbeddedChunks:
        vectors, stats = (
            self.encode_texts(
                [
                    chunk.text
                    for chunk in chunks
                ],
                batch_size=batch_size,
                show_progress_bar=(
                    show_progress_bar
                ),
            )
        )

        return EmbeddedChunks(
            chunks=list(
                chunks
            ),
            vectors=vectors,
            stats=stats,
        )

    def rank_texts(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> list[dict]:
        if not documents:
            return []

        query_vector = (
            self.embed_query(
                query
            )
        )

        document_vectors, _ = (
            self.encode_texts(
                documents,
                show_progress_bar=False,
            )
        )

        scores = (
            cosine_similarity_matrix(
                query_vector,
                document_vectors,
            )[0]
        )

        ranked_indices = (
            np.argsort(
                -scores
            )
        )

        if top_k is not None:
            ranked_indices = (
                ranked_indices[:top_k]
            )

        return [
            {
                "rank": rank,
                "document_index": int(
                    index
                ),
                "score": float(
                    scores[index]
                ),
                "document": (
                    documents[index]
                ),
            }
            for rank, index in enumerate(
                ranked_indices,
                start=1,
            )
        ]

    def unload(
        self,
    ) -> None:
        self._model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()