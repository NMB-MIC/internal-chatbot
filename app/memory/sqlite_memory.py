from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from app.config import settings
from app.memory.models import (
    ChatMessage,
    ChatSession,
    RetrievalRunLog,
    RetrievalSourceLog,
)
from app.rag.rag_chain import (
    RagAnswerResult,
)


def _utc_now() -> str:
    return (
        datetime
        .now(timezone.utc)
        .isoformat(
            timespec="milliseconds"
        )
    )


def _json_dumps(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _json_loads(
    value: str | None,
    *,
    default: Any,
) -> Any:
    if not value:
        return default

    return json.loads(
        value
    )


def _optional_bool(
    value: Any,
) -> bool | None:
    if value is None:
        return None

    return bool(
        value
    )


def _optional_int(
    value: Any,
) -> int | None:
    if value is None:
        return None

    return int(
        value
    )


def _optional_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    return float(
        value
    )


def _ensure_column(
    connection: sqlite3.Connection,
    *,
    table_name: str,
    column_name: str,
    declaration: str,
) -> None:
    """
    Apply a simple additive SQLite migration.

    Table and column names are internal constants,
    never user-provided values.
    """

    existing_columns = {
        str(
            row["name"]
        )
        for row in (
            connection
            .execute(
                f"PRAGMA table_info("
                f"{table_name}"
                f");"
            )
            .fetchall()
        )
    }

    if column_name in (
        existing_columns
    ):
        return

    connection.execute(
        f"ALTER TABLE "
        f"{table_name} "
        f"ADD COLUMN "
        f"{column_name} "
        f"{declaration};"
    )


class SqliteChatMemory:
    """
    Persistent local chat memory.

    A fresh SQLite connection is opened per operation.
    Connections are not shared across threads.
    """

    def __init__(
        self,
        *,
        db_path: Path = (
            settings.sqlite_db_path
        ),
        timeout_seconds: int = (
            settings
            .memory_sqlite_timeout_seconds
        ),
        recent_message_limit: int = (
            settings
            .memory_recent_message_limit
        ),
        summary_trigger_messages: int = (
            settings
            .memory_summary_trigger_messages
        ),
        text_preview_chars: int = (
            settings
            .memory_text_preview_chars
        ),
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        if recent_message_limit < 1:
            raise ValueError(
                "recent_message_limit must be positive."
            )

        if summary_trigger_messages < 1:
            raise ValueError(
                "summary_trigger_messages must be positive."
            )

        if text_preview_chars < 50:
            raise ValueError(
                "text_preview_chars must be at least 50."
            )

        self.db_path = Path(
            db_path
        )

        self.timeout_seconds = (
            timeout_seconds
        )

        self.recent_message_limit = (
            recent_message_limit
        )

        self.summary_trigger_messages = (
            summary_trigger_messages
        )

        self.text_preview_chars = (
            text_preview_chars
        )

    @contextmanager
    def _connection(
        self,
    ) -> Iterator[
        sqlite3.Connection
    ]:
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            self.db_path,
            timeout=self.timeout_seconds,
        )

        connection.row_factory = (
            sqlite3.Row
        )

        connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        connection.execute(
            "PRAGMA busy_timeout = 10000;"
        )

        try:
            yield connection
            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def initialize(
        self,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "PRAGMA journal_mode = WAL;"
            )

            connection.execute(
                "PRAGMA synchronous = NORMAL;"
            )

            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_schema (
                    schema_version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    summary TEXT,
                    summarized_until_message_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL
                        CHECK (
                            role IN (
                                'system',
                                'user',
                                'assistant'
                            )
                        ),
                    content TEXT NOT NULL,
                    language TEXT,
                    original_query TEXT,
                    standalone_query TEXT,
                    answerable INTEGER,
                    confidence TEXT,
                    metadata_json TEXT NOT NULL
                        DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (
                        session_id
                    )
                    REFERENCES chat_sessions (
                        session_id
                    )
                    ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_chat_messages_session_id_message_id
                ON chat_messages (
                    session_id,
                    message_id
                );

                CREATE TABLE IF NOT EXISTS retrieval_runs (
                    retrieval_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message_id INTEGER NOT NULL,
                    assistant_message_id INTEGER NOT NULL,
                    original_query TEXT NOT NULL,
                    standalone_query TEXT NOT NULL,
                    filter_criteria_json TEXT NOT NULL,
                    raw_hit_count INTEGER NOT NULL,
                    accepted_hit_count INTEGER NOT NULL,
                    top_score REAL,
                    similarity_threshold REAL NOT NULL,
                    embedding_seconds REAL NOT NULL,
                    search_seconds REAL NOT NULL,
                    total_seconds REAL NOT NULL,
                    llm_called INTEGER NOT NULL,
                    thinking_enabled INTEGER NOT NULL,
                    llm_metrics_json TEXT NOT NULL,
                    limitations_json TEXT NOT NULL,
                    retrieval_diagnostics_json TEXT NOT NULL
                        DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (
                        session_id
                    )
                    REFERENCES chat_sessions (
                        session_id
                    )
                    ON DELETE CASCADE,
                    FOREIGN KEY (
                        user_message_id
                    )
                    REFERENCES chat_messages (
                        message_id
                    )
                    ON DELETE CASCADE,
                    FOREIGN KEY (
                        assistant_message_id
                    )
                    REFERENCES chat_messages (
                        message_id
                    )
                    ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_retrieval_runs_session_id_run_id
                ON retrieval_runs (
                    session_id,
                    retrieval_run_id
                );

                CREATE TABLE IF NOT EXISTS retrieval_sources (
                    retrieval_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    retrieval_run_id INTEGER NOT NULL,
                    source_rank INTEGER NOT NULL,
                    source_id TEXT,
                    point_id TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    cited INTEGER NOT NULL,
                    score REAL NOT NULL,
                    source_path TEXT NOT NULL,
                    category TEXT,
                    page_number INTEGER,
                    sheet_name TEXT,
                    row_start INTEGER,
                    row_end INTEGER,
                    heading_path_json TEXT NOT NULL,
                    text_preview TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (
                        retrieval_run_id
                    )
                    REFERENCES retrieval_runs (
                        retrieval_run_id
                    )
                    ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_retrieval_sources_run_id_rank
                ON retrieval_sources (
                    retrieval_run_id,
                    source_rank
                );
                """
            )

            _ensure_column(
                connection,
                table_name=(
                    "retrieval_runs"
                ),
                column_name=(
                    "retrieval_diagnostics_json"
                ),
                declaration=(
                    "TEXT NOT NULL "
                    "DEFAULT '{}'"
                ),
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO memory_schema (
                    schema_version,
                    applied_at
                )
                VALUES (?, ?);
                """,
                (
                    1,
                    _utc_now(),
                ),
            )

    def inspect_pragmas(
        self,
    ) -> dict[str, Any]:
        with self._connection() as connection:
            journal_mode = (
                connection
                .execute(
                    "PRAGMA journal_mode;"
                )
                .fetchone()[0]
            )

            foreign_keys = (
                connection
                .execute(
                    "PRAGMA foreign_keys;"
                )
                .fetchone()[0]
            )

            synchronous = (
                connection
                .execute(
                    "PRAGMA synchronous;"
                )
                .fetchone()[0]
            )

            busy_timeout = (
                connection
                .execute(
                    "PRAGMA busy_timeout;"
                )
                .fetchone()[0]
            )

        return {
            "journal_mode": (
                journal_mode
            ),
            "foreign_keys": int(
                foreign_keys
            ),
            "synchronous": int(
                synchronous
            ),
            "busy_timeout": int(
                busy_timeout
            ),
        }

    def list_table_names(
        self,
    ) -> list[str]:
        with self._connection() as connection:
            rows = (
                connection
                .execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    ORDER BY name;
                    """
                )
                .fetchall()
            )

        return [
            str(
                row["name"]
            )
            for row in rows
        ]

    def _require_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> None:
        row = (
            connection
            .execute(
                """
                SELECT session_id
                FROM chat_sessions
                WHERE session_id = ?;
                """,
                (
                    session_id,
                ),
            )
            .fetchone()
        )

        if row is None:
            raise KeyError(
                f"Unknown session_id: "
                f"{session_id}"
            )

    def create_session(
        self,
        *,
        user_id: str | None = None,
        title: str | None = None,
        session_id: str | None = None,
    ) -> ChatSession:
        created_at = (
            _utc_now()
        )

        resolved_session_id = (
            session_id
            or str(
                uuid.uuid4()
            )
        )

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (
                    session_id,
                    user_id,
                    title,
                    summary,
                    summarized_until_message_id,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, NULL, NULL, ?, ?);
                """,
                (
                    resolved_session_id,
                    user_id,
                    title,
                    created_at,
                    created_at,
                ),
            )

        session = self.get_session(
            resolved_session_id
        )

        if session is None:
            raise RuntimeError(
                "Session creation failed."
            )

        return session

    def get_session(
        self,
        session_id: str,
    ) -> ChatSession | None:
        with self._connection() as connection:
            row = (
                connection
                .execute(
                    """
                    SELECT
                        session_id,
                        user_id,
                        title,
                        summary,
                        summarized_until_message_id,
                        created_at,
                        updated_at
                    FROM chat_sessions
                    WHERE session_id = ?;
                    """,
                    (
                        session_id,
                    ),
                )
                .fetchone()
            )

        if row is None:
            return None

        return ChatSession(
            session_id=str(
                row["session_id"]
            ),
            user_id=row["user_id"],
            title=row["title"],
            summary=row["summary"],
            summarized_until_message_id=(
                _optional_int(
                    row[
                        "summarized_until_message_id"
                    ]
                )
            ),
            created_at=str(
                row["created_at"]
            ),
            updated_at=str(
                row["updated_at"]
            ),
        )

    def list_sessions(
        self,
        *,
        limit: int = 20,
    ) -> list[ChatSession]:
        with self._connection() as connection:
            rows = (
                connection
                .execute(
                    """
                    SELECT
                        session_id,
                        user_id,
                        title,
                        summary,
                        summarized_until_message_id,
                        created_at,
                        updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT ?;
                    """,
                    (
                        limit,
                    ),
                )
                .fetchall()
            )

        return [
            ChatSession(
                session_id=str(
                    row["session_id"]
                ),
                user_id=row["user_id"],
                title=row["title"],
                summary=row["summary"],
                summarized_until_message_id=(
                    _optional_int(
                        row[
                            "summarized_until_message_id"
                        ]
                    )
                ),
                created_at=str(
                    row["created_at"]
                ),
                updated_at=str(
                    row["updated_at"]
                ),
            )
            for row in rows
        ]

    def save_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        language: str | None = None,
        original_query: str | None = None,
        standalone_query: str | None = None,
        answerable: bool | None = None,
        confidence: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        if role not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError(
                f"Unsupported role: {role}"
            )

        prepared_content = (
            content.strip()
        )

        if not prepared_content:
            raise ValueError(
                "Message content must not be empty."
            )

        created_at = (
            _utc_now()
        )

        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            cursor = (
                connection
                .execute(
                    """
                    INSERT INTO chat_messages (
                        session_id,
                        role,
                        content,
                        language,
                        original_query,
                        standalone_query,
                        answerable,
                        confidence,
                        metadata_json,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        session_id,
                        role,
                        prepared_content,
                        language,
                        original_query,
                        standalone_query,
                        (
                            None
                            if answerable
                            is None
                            else int(
                                answerable
                            )
                        ),
                        confidence,
                        _json_dumps(
                            metadata
                            or {}
                        ),
                        created_at,
                    ),
                )
            )

            message_id = int(
                cursor.lastrowid
            )

            connection.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE session_id = ?;
                """,
                (
                    created_at,
                    session_id,
                ),
            )

        message = self.get_message(
            message_id
        )

        if message is None:
            raise RuntimeError(
                "Message insertion failed."
            )

        return message

    def get_message(
        self,
        message_id: int,
    ) -> ChatMessage | None:
        with self._connection() as connection:
            row = (
                connection
                .execute(
                    """
                    SELECT
                        message_id,
                        session_id,
                        role,
                        content,
                        language,
                        original_query,
                        standalone_query,
                        answerable,
                        confidence,
                        metadata_json,
                        created_at
                    FROM chat_messages
                    WHERE message_id = ?;
                    """,
                    (
                        message_id,
                    ),
                )
                .fetchone()
            )

        if row is None:
            return None

        return self._row_to_message(
            row
        )

    def _row_to_message(
        self,
        row: sqlite3.Row,
    ) -> ChatMessage:
        return ChatMessage(
            message_id=int(
                row["message_id"]
            ),
            session_id=str(
                row["session_id"]
            ),
            role=str(
                row["role"]
            ),
            content=str(
                row["content"]
            ),
            language=row["language"],
            original_query=(
                row["original_query"]
            ),
            standalone_query=(
                row["standalone_query"]
            ),
            answerable=(
                _optional_bool(
                    row["answerable"]
                )
            ),
            confidence=(
                row["confidence"]
            ),
            metadata=(
                _json_loads(
                    row["metadata_json"],
                    default={},
                )
            ),
            created_at=str(
                row["created_at"]
            ),
        )

    def load_recent_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
    ) -> list[ChatMessage]:
        effective_limit = (
            limit
            or self.recent_message_limit
        )

        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            rows = (
                connection
                .execute(
                    """
                    SELECT
                        message_id,
                        session_id,
                        role,
                        content,
                        language,
                        original_query,
                        standalone_query,
                        answerable,
                        confidence,
                        metadata_json,
                        created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?;
                    """,
                    (
                        session_id,
                        effective_limit,
                    ),
                )
                .fetchall()
            )

        return [
            self._row_to_message(
                row
            )
            for row in reversed(
                rows
            )
        ]

    def load_all_messages(
        self,
        session_id: str,
    ) -> list[ChatMessage]:
        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            rows = (
                connection
                .execute(
                    """
                    SELECT
                        message_id,
                        session_id,
                        role,
                        content,
                        language,
                        original_query,
                        standalone_query,
                        answerable,
                        confidence,
                        metadata_json,
                        created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY message_id ASC;
                    """,
                    (
                        session_id,
                    ),
                )
                .fetchall()
            )

        return [
            self._row_to_message(
                row
            )
            for row in rows
        ]

    def count_messages(
        self,
        session_id: str,
    ) -> int:
        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            row = (
                connection
                .execute(
                    """
                    SELECT COUNT(*) AS message_count
                    FROM chat_messages
                    WHERE session_id = ?;
                    """,
                    (
                        session_id,
                    ),
                )
                .fetchone()
            )

        return int(
            row["message_count"]
        )

    def update_session_summary(
        self,
        *,
        session_id: str,
        summary: str,
        summarized_until_message_id: int,
    ) -> ChatSession:
        prepared_summary = (
            summary.strip()
        )

        if not prepared_summary:
            raise ValueError(
                "Summary must not be empty."
            )

        updated_at = (
            _utc_now()
        )

        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            connection.execute(
                """
                UPDATE chat_sessions
                SET
                    summary = ?,
                    summarized_until_message_id = ?,
                    updated_at = ?
                WHERE session_id = ?;
                """,
                (
                    prepared_summary,
                    summarized_until_message_id,
                    updated_at,
                    session_id,
                ),
            )

        session = self.get_session(
            session_id
        )

        if session is None:
            raise RuntimeError(
                "Session summary update failed."
            )

        return session

    def summary_needed(
        self,
        session_id: str,
    ) -> bool:
        session = self.get_session(
            session_id
        )

        if session is None:
            raise KeyError(
                f"Unknown session_id: "
                f"{session_id}"
            )

        after_message_id = (
            session
            .summarized_until_message_id
            or 0
        )

        with self._connection() as connection:
            row = (
                connection
                .execute(
                    """
                    SELECT COUNT(*) AS unsummarized_count
                    FROM chat_messages
                    WHERE
                        session_id = ?
                        AND message_id > ?;
                    """,
                    (
                        session_id,
                        after_message_id,
                    ),
                )
                .fetchone()
            )

        return (
            int(
                row[
                    "unsummarized_count"
                ]
            )
            >= self
            .summary_trigger_messages
        )

    def log_rag_result(
        self,
        *,
        session_id: str,
        user_message_id: int,
        assistant_message_id: int,
        original_query: str,
        standalone_query: str,
        filter_criteria: (
            dict[str, Any]
            | None
        ),
        rag_result: RagAnswerResult,
    ) -> RetrievalRunLog:
        created_at = (
            _utc_now()
        )

        retrieval = (
            rag_result
            .retrieval_result
        )

        metrics = retrieval.metrics

        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            cursor = (
                connection
                .execute(
                    """
                    INSERT INTO retrieval_runs (
                        session_id,
                        user_message_id,
                        assistant_message_id,
                        original_query,
                        standalone_query,
                        filter_criteria_json,
                        raw_hit_count,
                        accepted_hit_count,
                        top_score,
                        similarity_threshold,
                        embedding_seconds,
                        search_seconds,
                        total_seconds,
                        llm_called,
                        thinking_enabled,
                        llm_metrics_json,
                        limitations_json,
                        retrieval_diagnostics_json,
                        created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?
                    );
                    """,
                    (
                        session_id,
                        user_message_id,
                        assistant_message_id,
                        original_query,
                        standalone_query,
                        _json_dumps(
                            filter_criteria
                            or {}
                        ),
                        len(
                            retrieval.raw_hits
                        ),
                        len(
                            retrieval
                            .accepted_hits
                        ),
                        retrieval.top_score,
                        retrieval
                        .similarity_threshold,
                        metrics
                        .embedding_seconds,
                        metrics.search_seconds,
                        metrics.total_seconds,
                        int(
                            rag_result.llm_called
                        ),
                        int(
                            rag_result
                            .thinking_enabled
                        ),
                        _json_dumps(
                            rag_result
                            .llm_metrics
                        ),
                        _json_dumps(
                            rag_result
                            .limitations
                        ),
                        _json_dumps(
                            retrieval
                            .diagnostics
                            .to_dict()
                        ),
                        created_at,
                    ),
                )
            )

            retrieval_run_id = int(
                cursor.lastrowid
            )

            accepted_point_ids = {
                hit.point_id
                for hit in (
                    retrieval
                    .accepted_hits
                )
            }

            source_id_by_point_id = {
                source.point_id: (
                    source.source_id
                )
                for source in (
                    rag_result
                    .available_sources
                )
            }

            cited_point_ids = {
                source.point_id
                for source in (
                    rag_result
                    .cited_sources
                )
            }

            for source_rank, hit in enumerate(
                retrieval.raw_hits,
                start=1,
            ):
                payload = hit.payload

                heading_path = (
                    payload.get(
                        "heading_path"
                    )
                    or []
                )

                connection.execute(
                    """
                    INSERT INTO retrieval_sources (
                        retrieval_run_id,
                        source_rank,
                        source_id,
                        point_id,
                        accepted,
                        cited,
                        score,
                        source_path,
                        category,
                        page_number,
                        sheet_name,
                        row_start,
                        row_end,
                        heading_path_json,
                        text_preview,
                        created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?
                    );
                    """,
                    (
                        retrieval_run_id,
                        source_rank,
                        source_id_by_point_id
                        .get(
                            hit.point_id
                        ),
                        hit.point_id,
                        int(
                            hit.point_id
                            in accepted_point_ids
                        ),
                        int(
                            hit.point_id
                            in cited_point_ids
                        ),
                        hit.score,
                        str(
                            payload.get(
                                "source_path",
                                "",
                            )
                        ),
                        payload.get(
                            "category"
                        ),
                        payload.get(
                            "page_number"
                        ),
                        payload.get(
                            "sheet_name"
                        ),
                        payload.get(
                            "row_start"
                        ),
                        payload.get(
                            "row_end"
                        ),
                        _json_dumps(
                            heading_path
                        ),
                        hit.text[
                            :self
                            .text_preview_chars
                        ],
                        created_at,
                    ),
                )

        run = self.get_retrieval_run(
            retrieval_run_id
        )

        if run is None:
            raise RuntimeError(
                "Retrieval-run logging failed."
            )

        return run

    def get_retrieval_run(
        self,
        retrieval_run_id: int,
    ) -> RetrievalRunLog | None:
        with self._connection() as connection:
            row = (
                connection
                .execute(
                    """
                    SELECT
                        retrieval_run_id,
                        session_id,
                        user_message_id,
                        assistant_message_id,
                        original_query,
                        standalone_query,
                        filter_criteria_json,
                        raw_hit_count,
                        accepted_hit_count,
                        top_score,
                        similarity_threshold,
                        embedding_seconds,
                        search_seconds,
                        total_seconds,
                        llm_called,
                        thinking_enabled,
                        llm_metrics_json,
                        limitations_json,
                        retrieval_diagnostics_json,
                        created_at
                    FROM retrieval_runs
                    WHERE retrieval_run_id = ?;
                    """,
                    (
                        retrieval_run_id,
                    ),
                )
                .fetchone()
            )

        if row is None:
            return None

        return self._row_to_retrieval_run(
            row
        )

    def list_retrieval_runs(
        self,
        session_id: str,
    ) -> list[RetrievalRunLog]:
        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            rows = (
                connection
                .execute(
                    """
                    SELECT
                        retrieval_run_id,
                        session_id,
                        user_message_id,
                        assistant_message_id,
                        original_query,
                        standalone_query,
                        filter_criteria_json,
                        raw_hit_count,
                        accepted_hit_count,
                        top_score,
                        similarity_threshold,
                        embedding_seconds,
                        search_seconds,
                        total_seconds,
                        llm_called,
                        thinking_enabled,
                        llm_metrics_json,
                        limitations_json,
                        retrieval_diagnostics_json,
                        created_at
                    FROM retrieval_runs
                    WHERE session_id = ?
                    ORDER BY retrieval_run_id ASC;
                    """,
                    (
                        session_id,
                    ),
                )
                .fetchall()
            )

        return [
            self._row_to_retrieval_run(
                row
            )
            for row in rows
        ]

    def _row_to_retrieval_run(
        self,
        row: sqlite3.Row,
    ) -> RetrievalRunLog:
        return RetrievalRunLog(
            retrieval_run_id=int(
                row["retrieval_run_id"]
            ),
            session_id=str(
                row["session_id"]
            ),
            user_message_id=int(
                row["user_message_id"]
            ),
            assistant_message_id=int(
                row[
                    "assistant_message_id"
                ]
            ),
            original_query=str(
                row["original_query"]
            ),
            standalone_query=str(
                row["standalone_query"]
            ),
            filter_criteria=(
                _json_loads(
                    row[
                        "filter_criteria_json"
                    ],
                    default={},
                )
            ),
            raw_hit_count=int(
                row["raw_hit_count"]
            ),
            accepted_hit_count=int(
                row["accepted_hit_count"]
            ),
            top_score=(
                _optional_float(
                    row["top_score"]
                )
            ),
            similarity_threshold=float(
                row[
                    "similarity_threshold"
                ]
            ),
            embedding_seconds=float(
                row["embedding_seconds"]
            ),
            search_seconds=float(
                row["search_seconds"]
            ),
            total_seconds=float(
                row["total_seconds"]
            ),
            llm_called=bool(
                row["llm_called"]
            ),
            thinking_enabled=bool(
                row[
                    "thinking_enabled"
                ]
            ),
            llm_metrics=(
                _json_loads(
                    row[
                        "llm_metrics_json"
                    ],
                    default={},
                )
            ),
            limitations=(
                _json_loads(
                    row[
                        "limitations_json"
                    ],
                    default=[],
                )
            ),
            retrieval_diagnostics=(
                _json_loads(
                    row[
                        "retrieval_diagnostics_json"
                    ],
                    default={},
                )
            ),
            created_at=str(
                row["created_at"]
            ),
        )

    def list_retrieval_sources(
        self,
        retrieval_run_id: int,
    ) -> list[
        RetrievalSourceLog
    ]:
        with self._connection() as connection:
            rows = (
                connection
                .execute(
                    """
                    SELECT
                        retrieval_source_id,
                        retrieval_run_id,
                        source_rank,
                        source_id,
                        point_id,
                        accepted,
                        cited,
                        score,
                        source_path,
                        category,
                        page_number,
                        sheet_name,
                        row_start,
                        row_end,
                        heading_path_json,
                        text_preview,
                        created_at
                    FROM retrieval_sources
                    WHERE retrieval_run_id = ?
                    ORDER BY source_rank ASC;
                    """,
                    (
                        retrieval_run_id,
                    ),
                )
                .fetchall()
            )

        return [
            RetrievalSourceLog(
                retrieval_source_id=int(
                    row[
                        "retrieval_source_id"
                    ]
                ),
                retrieval_run_id=int(
                    row[
                        "retrieval_run_id"
                    ]
                ),
                source_rank=int(
                    row["source_rank"]
                ),
                source_id=(
                    row["source_id"]
                ),
                point_id=str(
                    row["point_id"]
                ),
                accepted=bool(
                    row["accepted"]
                ),
                cited=bool(
                    row["cited"]
                ),
                score=float(
                    row["score"]
                ),
                source_path=str(
                    row["source_path"]
                ),
                category=(
                    row["category"]
                ),
                page_number=(
                    _optional_int(
                        row[
                            "page_number"
                        ]
                    )
                ),
                sheet_name=(
                    row["sheet_name"]
                ),
                row_start=(
                    _optional_int(
                        row["row_start"]
                    )
                ),
                row_end=(
                    _optional_int(
                        row["row_end"]
                    )
                ),
                heading_path=(
                    _json_loads(
                        row[
                            "heading_path_json"
                        ],
                        default=[],
                    )
                ),
                text_preview=str(
                    row["text_preview"]
                ),
                created_at=str(
                    row["created_at"]
                ),
            )
            for row in rows
        ]

    def delete_session(
        self,
        session_id: str,
    ) -> bool:
        with self._connection() as connection:
            cursor = (
                connection
                .execute(
                    """
                    DELETE FROM chat_sessions
                    WHERE session_id = ?;
                    """,
                    (
                        session_id,
                    ),
                )
            )

        return (
            cursor.rowcount > 0
        )
        
    def rename_session(
        self,
        *,
        session_id: str,
        title: str,
    ) -> ChatSession:
        prepared_title = (
            title.strip()
        )

        if not prepared_title:
            raise ValueError(
                "Session title must not be empty."
            )

        updated_at = (
            _utc_now()
        )

        with self._connection() as connection:
            self._require_session(
                connection,
                session_id,
            )

            connection.execute(
                """
                UPDATE chat_sessions
                SET
                    title = ?,
                    updated_at = ?
                WHERE session_id = ?;
                """,
                (
                    prepared_title,
                    updated_at,
                    session_id,
                ),
            )

        session = self.get_session(
            session_id
        )

        if session is None:
            raise RuntimeError(
                "Session rename failed."
            )

        return session