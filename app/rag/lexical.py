from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


_TEXT_PAYLOAD_KEYS = (
    "text",
    "chunk_text",
    "content",
    "page_content",
)

_CHUNK_ORDER_KEYS = (
    "chunk_index",
    "chunk_no",
    "chunk_number",
    "source_chunk_index",
    "unit_index",
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "should", "the",
    "to", "what", "when", "where", "which", "who", "why", "with",
    "does", "do", "did", "data", "value", "values",
}

_TECH_TOKEN_PATTERN = re.compile(
    r"""
    --?[A-Za-z0-9_./:-]+
    |
    [A-Za-z0-9_./:-]+
    |
    [ก-๙]+
    """,
    flags=re.VERBOSE,
)

_QUOTED_OR_CODE_PATTERN = re.compile(
    r"""
    `([^`]+)`
    |
    "([^"]+)"
    |
    '([^']+)'
    """,
    flags=re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    point_id: str
    text: str
    payload: dict[str, Any]
    score: float
    matched_terms: tuple[str, ...]
    rank_reason: str


@dataclass(frozen=True, slots=True)
class _CorpusChunk:
    point_id: str
    text: str
    payload: dict[str, Any]
    order_key: int | None


def _normalize(text: str) -> str:
    return (
        text.lower()
        .replace("\u200b", "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .strip()
    )


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TECH_TOKEN_PATTERN.findall(text)
        if token.strip()
    ]


def _contentful_tokens(text: str) -> list[str]:
    return [
        token
        for token in _tokens(text)
        if len(token) >= 2 and token not in _STOPWORDS
    ]


def _extract_phrases(query: str) -> list[str]:
    """
    Extract exact phrases and add small domain-aware expansions for
    internal technical runbooks.

    These expansions are intentionally conservative. They only add exact
    artifacts that commonly appear in questions but may be phrased
    differently from the document text.
    """

    query_norm = _normalize(query)

    phrases: list[str] = []

    for match in _QUOTED_OR_CODE_PATTERN.finditer(query):
        phrase = next(
            group
            for group in match.groups()
            if group
        ).strip()

        if phrase:
            phrases.append(phrase.lower())

    for token in _tokens(query):
        token_norm = token.lower()

        if (
            "." in token_norm
            or "_" in token_norm
            or "/" in token_norm
            or ":" in token_norm
            or token_norm.startswith("--")
            or token.isupper()
        ):
            phrases.append(token_norm)

    # UI labels from the Streamlit dashboard section.
    if "replay" in query_norm:
        phrases.extend(
            [
                "earliest (replay)",
                "buffer_hours=87600",
                "buffer_hours",
            ]
        )

    if "live" in query_norm:
        phrases.extend(
            [
                "latest (live)",
                "buffer_hours=24",
                "buffer_hours",
            ]
        )

    # Health endpoint wording often asks "should report", while the
    # runbook line contains JSON fields such as num_features.
    if (
        "health" in query_norm
        or "endpoint" in query_norm
        or "num features" in query_norm
        or "num_features" in query_norm
        or "model input features" in query_norm
        or "total model input" in query_norm
    ):
        phrases.extend(
            [
                "/health",
                "num_features",
                "feature_version",
                "feats_39_behavioral_v2",
            ]
        )

    # Historical replay / shadow mode exact command.
    if (
        "shadow" in query_norm
        or "historical" in query_norm
        or "replay" in query_norm
        or "replays" in query_norm
    ):
        phrases.extend(
            [
                "python replay.py",
                "--input",
                "--bootstrap localhost:9092",
                "--topic iot.machine.status.raw",
                "--sleep 0.0",
            ]
        )

    # Common troubleshooting exact strings.
    if "mqtt" in query_norm and (
        "not receiving" in query_norm
        or "message" in query_norm
        or "messages" in query_norm
        or "check" in query_norm
    ):
        phrases.extend(
            [
                "docker logs mqtt_to_ml_kafka -f",
                "mqtt_broker",
            ]
        )

    if (
        "pod" in query_norm
        and (
            "crashing" in query_norm
            or "logs" in query_norm
            or "log" in query_norm
        )
    ):
        phrases.extend(
            [
                "kubectl -n ml logs",
                "deployment/alert-eta-service",
                "--tail=50",
            ]
        )

    return list(dict.fromkeys(phrases))


def _payload_text(payload: dict[str, Any]) -> str:
    for key in _TEXT_PAYLOAD_KEYS:
        value = payload.get(key)

        if isinstance(value, str) and value.strip():
            return value

    return ""


def _chunk_order_key(payload: dict[str, Any]) -> int | None:
    for key in _CHUNK_ORDER_KEYS:
        value = payload.get(key)

        if value is None:
            continue

        try:
            return int(value)

        except (TypeError, ValueError):
            continue

    return None


def _get_client_and_collection(vector_store: Any) -> tuple[Any, str]:
    client = (
        getattr(vector_store, "client", None)
        or getattr(vector_store, "qdrant_client", None)
        or getattr(vector_store, "_client", None)
    )

    collection_name = (
        getattr(vector_store, "collection_name", None)
        or getattr(vector_store, "_collection_name", None)
    )

    if client is None:
        raise AttributeError(
            "Could not find Qdrant client on vector_store."
        )

    if not collection_name:
        raise AttributeError(
            "Could not find collection_name on vector_store."
        )

    return client, str(collection_name)


class LexicalChunkRetriever:
    """
    Lightweight exact-token retriever over the Qdrant payload corpus.

    This is a backstop for technical docs where exact strings matter:
    commands, filenames, Kafka topics, env vars, constants, UI labels,
    CLI flags, and error messages.
    """

    def __init__(
        self,
        *,
        vector_store: Any,
        max_scroll_points: int = 5000,
        scroll_batch_size: int = 256,
        neighbor_window: int = 1,
    ) -> None:
        self.vector_store = vector_store
        self.max_scroll_points = max_scroll_points
        self.scroll_batch_size = scroll_batch_size
        self.neighbor_window = neighbor_window

    def _load_corpus(
        self,
        *,
        query_filter: Any,
    ) -> list[_CorpusChunk]:
        client, collection_name = _get_client_and_collection(
            self.vector_store
        )

        chunks: list[_CorpusChunk] = []
        offset = None

        while len(chunks) < self.max_scroll_points:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=min(
                    self.scroll_batch_size,
                    self.max_scroll_points - len(chunks),
                ),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                payload = dict(record.payload or {})
                text = _payload_text(payload)

                if not text:
                    continue

                chunks.append(
                    _CorpusChunk(
                        point_id=str(record.id),
                        text=text,
                        payload=payload,
                        order_key=_chunk_order_key(payload),
                    )
                )

            if next_offset is None:
                break

            offset = next_offset

        return chunks

    def _score_chunk(
        self,
        *,
        query: str,
        chunk: _CorpusChunk,
    ) -> tuple[float, tuple[str, ...], str]:
        query_tokens = _contentful_tokens(query)
        query_terms = set(query_tokens)

        chunk_text_norm = _normalize(chunk.text)
        chunk_terms = set(_contentful_tokens(chunk.text))

        matched_terms = sorted(query_terms & chunk_terms)

        if not query_terms:
            overlap_score = 0.0
        else:
            # Query-side coverage matters more than full chunk length for
            # long procedural chunks.
            overlap_score = 0.45 * (
                len(matched_terms)
                / max(len(query_terms), 1)
            )

            # Keep a small normalization term so very noisy chunks do not
            # dominate solely by containing one common token.
            overlap_score += 0.10 * (
                len(matched_terms)
                / math.sqrt(
                    len(query_terms) * max(len(chunk_terms), 1)
                )
            )

        phrase_bonus = 0.0
        matched_phrases: list[str] = []

        for phrase in _extract_phrases(query):
            phrase_norm = _normalize(phrase)

            if phrase_norm and phrase_norm in chunk_text_norm:
                matched_phrases.append(phrase_norm)

                # Exact technical artifacts should be strong enough to
                # enter the fused candidate list even when dense retrieval
                # prefers a conceptual chunk.
                if (
                    phrase_norm.startswith("--")
                    or "_" in phrase_norm
                    or "/" in phrase_norm
                    or "." in phrase_norm
                    or "(" in phrase_norm
                    or ")" in phrase_norm
                    or phrase_norm.isdigit()
                ):
                    phrase_bonus += 0.65
                else:
                    phrase_bonus += 0.45

        command_bonus = 0.0

        command_tokens = [
            token
            for token in query_tokens
            if (
                "." in token
                or "/" in token
                or token.startswith("--")
            )
        ]

        if command_tokens:
            matched_command_tokens = [
                token
                for token in command_tokens
                if token in chunk_text_norm
            ]

            command_bonus = 0.10 * len(matched_command_tokens)

        score = min(
            1.0,
            overlap_score + phrase_bonus + command_bonus,
        )

        matched = tuple(
            list(matched_terms)
            + matched_phrases
        )

        reason = (
            "lexical exact/term match"
            if matched
            else "no lexical match"
        )

        return score, matched, reason

    def _with_neighbors(
        self,
        *,
        corpus: list[_CorpusChunk],
        selected: list[LexicalCandidate],
    ) -> list[LexicalCandidate]:
        if self.neighbor_window < 1:
            return selected

        selected_by_id = {
            candidate.point_id: candidate
            for candidate in selected
        }

        by_source: dict[str, list[_CorpusChunk]] = {}

        for chunk in corpus:
            if chunk.order_key is None:
                continue

            source_path = str(
                chunk.payload.get("source_path", "")
            )

            by_source.setdefault(source_path, []).append(chunk)

        for source_chunks in by_source.values():
            source_chunks.sort(
                key=lambda item: (
                    item.order_key
                    if item.order_key is not None
                    else 10**12
                )
            )

            index_by_id = {
                chunk.point_id: index
                for index, chunk in enumerate(source_chunks)
            }

            for candidate in list(selected):
                if candidate.point_id not in index_by_id:
                    continue

                center = index_by_id[candidate.point_id]

                start = max(
                    0,
                    center - self.neighbor_window,
                )

                end = min(
                    len(source_chunks),
                    center + self.neighbor_window + 1,
                )

                for neighbor in source_chunks[start:end]:
                    if neighbor.point_id in selected_by_id:
                        continue

                    selected_by_id[neighbor.point_id] = (
                        LexicalCandidate(
                            point_id=neighbor.point_id,
                            text=neighbor.text,
                            payload=neighbor.payload,
                            score=0.30,
                            matched_terms=(),
                            rank_reason=(
                                "neighbor of lexical match"
                            ),
                        )
                    )

        return list(selected_by_id.values())

    def search(
        self,
        *,
        query: str,
        query_filter: Any,
        limit: int = 32,
        include_neighbors: bool = True,
    ) -> list[LexicalCandidate]:
        prepared_query = query.strip()

        if not prepared_query:
            return []

        corpus = self._load_corpus(
            query_filter=query_filter
        )

        candidates: list[LexicalCandidate] = []

        for chunk in corpus:
            score, matched, reason = self._score_chunk(
                query=prepared_query,
                chunk=chunk,
            )

            if score <= 0:
                continue

            candidates.append(
                LexicalCandidate(
                    point_id=chunk.point_id,
                    text=chunk.text,
                    payload=chunk.payload,
                    score=score,
                    matched_terms=matched,
                    rank_reason=reason,
                )
            )

        candidates.sort(
            key=lambda item: (
                item.score,
                len(item.matched_terms),
            ),
            reverse=True,
        )

        candidates = candidates[:limit]

        if include_neighbors:
            candidates = self._with_neighbors(
                corpus=corpus,
                selected=candidates,
            )

            candidates.sort(
                key=lambda item: item.score,
                reverse=True,
            )

        return candidates[:limit]