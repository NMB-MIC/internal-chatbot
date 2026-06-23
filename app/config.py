from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Internal RAG Chatbot")
    app_env: str = os.getenv("APP_ENV", "local")
    debug: bool = _get_bool("DEBUG", True)

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    ).rstrip("/")

    ollama_model: str = os.getenv(
        "OLLAMA_MODEL",
        "gemma4:26b",
    )

    ollama_temperature: float = float(
        os.getenv("OLLAMA_TEMPERATURE", "0.2")
    )

    ollama_num_ctx: int = int(
        os.getenv("OLLAMA_NUM_CTX", "8192")
    )

    ollama_keep_alive: str = os.getenv(
        "OLLAMA_KEEP_ALIVE",
        "30m",
    )

    ollama_request_timeout_seconds: int = int(
        os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "300")
    )

    ollama_think: bool = _get_bool("OLLAMA_THINK", False)

    qdrant_url: str = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333",
    ).rstrip("/")

    qdrant_collection_name: str = os.getenv(
        "QDRANT_COLLECTION_NAME",
        "internal_documents",
    )
    
    qdrant_api_key: str | None = (
        os.getenv(
            "QDRANT_API_KEY",
            "",
        ).strip()
        or None
    )

    qdrant_timeout_seconds: int = int(
        os.getenv(
            "QDRANT_TIMEOUT_SECONDS",
            "30",
        )
    )

    qdrant_upsert_batch_size: int = int(
        os.getenv(
            "QDRANT_UPSERT_BATCH_SIZE",
            "64",
        )
    )

    qdrant_dense_vector_name: str = os.getenv(
        "QDRANT_DENSE_VECTOR_NAME",
        "dense",
    )

    documents_dir: Path = PROJECT_ROOT / os.getenv(
        "DOCUMENTS_DIR",
        "data/documents",
    )

    processed_dir: Path = PROJECT_ROOT / os.getenv(
        "PROCESSED_DIR",
        "data/processed",
    )

    sqlite_db_path: Path = PROJECT_ROOT / os.getenv(
        "SQLITE_DB_PATH",
        "data/sqlite/chat_history.db",
    )
    
    chunk_size_chars: int = int(
        os.getenv("CHUNK_SIZE_CHARS", "1600")
    )

    chunk_overlap_chars: int = int(
        os.getenv("CHUNK_OVERLAP_CHARS", "200")
    )

    chunk_min_chars_for_warning: int = int(
        os.getenv("CHUNK_MIN_CHARS_FOR_WARNING", "80")
    )

    chunk_max_table_rows: int = int(
        os.getenv("CHUNK_MAX_TABLE_ROWS", "20")
    )
    
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-m3",
    )

    embedding_device: str = os.getenv(
        "EMBEDDING_DEVICE",
        "cpu",
    )

    embedding_batch_size: int = int(
        os.getenv(
            "EMBEDDING_BATCH_SIZE",
            "8",
        )
    )

    embedding_normalize: bool = _get_bool(
        "EMBEDDING_NORMALIZE",
        True,
    )

    embedding_show_progress: bool = _get_bool(
        "EMBEDDING_SHOW_PROGRESS",
        True,
    )
    
    rag_top_k_initial: int = int(
        os.getenv(
            "TOP_K_INITIAL",
            "8",
        )
    )

    rag_top_k_final: int = int(
        os.getenv(
            "TOP_K_FINAL",
            "4",
        )
    )

    rag_similarity_threshold: float = float(
        os.getenv(
            "SIMILARITY_THRESHOLD",
            "0.35",
        )
    )

    rag_context_max_chars: int = int(
        os.getenv(
            "RAG_CONTEXT_MAX_CHARS",
            "7000",
        )
    )

    rag_require_citations: bool = _get_bool(
        "RAG_REQUIRE_CITATIONS",
        True,
    )
    
    memory_sqlite_timeout_seconds: int = int(
        os.getenv(
            "MEMORY_SQLITE_TIMEOUT_SECONDS",
            "10",
        )
    )

    memory_recent_message_limit: int = int(
        os.getenv(
            "MEMORY_RECENT_MESSAGE_LIMIT",
            "8",
        )
    )

    memory_summary_trigger_messages: int = int(
        os.getenv(
            "MEMORY_SUMMARY_TRIGGER_MESSAGES",
            "20",
        )
    )

    memory_text_preview_chars: int = int(
        os.getenv(
            "MEMORY_TEXT_PREVIEW_CHARS",
            "500",
        )
    )
    
    app_name: str = os.getenv(
        "APP_NAME",
        "MIC 9000",
    )

    app_subtitle: str = os.getenv(
        "APP_SUBTITLE",
        "Internal AI Support",
    )

    division_name: str = os.getenv(
        "DIVISION_NAME",
        "Manufacturing Improvement Yokoten Center",
    )

    kb_staging_dir: Path = PROJECT_ROOT / os.getenv(
        "KB_STAGING_DIR",
        "data/staging",
    )

    kb_upload_max_mb: int = int(
        os.getenv(
            "KB_UPLOAD_MAX_MB",
            "50",
        )
    )

    kb_default_category: str = os.getenv(
        "KB_DEFAULT_CATEGORY",
        "uncategorized",
    )

    ui_session_title_max_chars: int = int(
        os.getenv(
            "UI_SESSION_TITLE_MAX_CHARS",
            "48",
        )
    )

    ui_session_list_limit: int = int(
        os.getenv(
            "UI_SESSION_LIST_LIMIT",
            "30",
        )
    )
    
    document_qa_top_k_initial: int = int(
        os.getenv(
            "DOCUMENT_QA_TOP_K_INITIAL",
            "24",
        )
    )

    document_qa_top_k_final: int = int(
        os.getenv(
            "DOCUMENT_QA_TOP_K_FINAL",
            "8",
        )
    )

    document_qa_similarity_threshold: float = float(
        os.getenv(
            "DOCUMENT_QA_SIMILARITY_THRESHOLD",
            "0.28",
        )
    )

    document_qa_context_max_chars: int = int(
        os.getenv(
            "DOCUMENT_QA_CONTEXT_MAX_CHARS",
            "12000",
        )
    )
    
    document_scope_default_behavior: str = os.getenv(
        "DOCUMENT_SCOPE_DEFAULT_BEHAVIOR",
        "prefer_selected",
    )


    mic_index_mode: str = os.getenv(
        "MIC_INDEX_MODE",
        "development",
    ).strip().lower()

    index_manifest_dir: Path = PROJECT_ROOT / os.getenv(
        "INDEX_MANIFEST_DIR",
        "storage/index_manifests",
    )

    index_quarantine_dir: Path = PROJECT_ROOT / os.getenv(
        "INDEX_QUARANTINE_DIR",
        "storage/index_quarantine",
    )

    index_snapshot_enabled: bool = _get_bool(
        "INDEX_SNAPSHOT_ENABLED",
        True,
    )

    index_allow_secret_documents: bool = _get_bool(
        "INDEX_ALLOW_SECRET_DOCUMENTS",
        False,
    )

    index_quarantine_synthetic_in_production: bool = _get_bool(
        "INDEX_QUARANTINE_SYNTHETIC_IN_PRODUCTION",
        True,
    )

    index_quarantine_external_books_in_production: bool = _get_bool(
        "INDEX_QUARANTINE_EXTERNAL_BOOKS_IN_PRODUCTION",
        True,
    )

    index_stale_document_days: int = int(
        os.getenv(
            "INDEX_STALE_DOCUMENT_DAYS",
            "365",
        )
    )

    index_safety_scan_max_chars: int = int(
        os.getenv(
            "INDEX_SAFETY_SCAN_MAX_CHARS",
            "200000",
        )
    )

    index_fail_on_rejected_files: bool = _get_bool(
        "INDEX_FAIL_ON_REJECTED_FILES",
        False,
    )


settings = Settings()