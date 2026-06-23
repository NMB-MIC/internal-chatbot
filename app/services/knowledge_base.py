from __future__ import annotations

import re
import shutil
from dataclasses import (
    asdict,
    dataclass,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from app.config import settings
from app.rag.chunking import (
    chunk_document_units,
    default_chunking_config,
)
from app.rag.embeddings import (
    BgeM3Embedder,
    select_unique_chunks,
)
from app.rag.loaders import (
    SUPPORTED_EXTENSIONS,
    load_documents,
)
from app.rag.vector_store import (
    QdrantIndexReport,
    QdrantVectorStore,
)
from app.services.index_manifest import (
    write_index_manifest,
)
from app.services.index_safety import (
    apply_quarantine_actions,
    make_rebuild_id,
    scan_document_tree,
    summarize_assessments,
    utc_now_iso,
)


class UploadedFileLike(
    Protocol
):
    name: str

    def getvalue(
        self,
    ) -> bytes:
        ...


@dataclass(frozen=True, slots=True)
class StagedFileInfo:
    relative_path: str
    category: str
    filename: str
    extension: str
    size_bytes: int
    modified_at_utc: str

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )


@dataclass(frozen=True, slots=True)
class RebuildReport:
    mode: str
    rebuild_id: str
    started_at_utc: str
    completed_at_utc: str
    manifest_path: str | None
    snapshot: dict[str, Any]
    document_safety: dict[str, Any]
    promoted_files: tuple[str, ...]
    discovered_files: int
    loaded_files: int
    skipped_files: int
    failed_files: int
    extracted_units: int
    all_chunks: int
    unique_chunks: int
    skipped_exact_duplicates: int
    vector_dimension: int
    qdrant: QdrantIndexReport

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            **asdict(
                self
            ),
            "qdrant": (
                self.qdrant
                .to_dict()
            ),
        }


_SAFE_FILENAME_PATTERN = re.compile(
    r"[^A-Za-z0-9ก-๙._() -]+"
)

_SAFE_CATEGORY_PATTERN = re.compile(
    r"[^A-Za-z0-9ก-๙_-]+"
)

_BLOCKED_NAME_FRAGMENTS = {
    "credential",
    "credentials",
    "secret",
    "secrets",
    "password",
    "passwd",
    "private_key",
    "apikey",
    "api_key",
    "token",
}


def _utc_now() -> str:
    return (
        datetime
        .now(timezone.utc)
        .isoformat(
            timespec="seconds"
        )
    )


def _sanitize_filename(
    filename: str,
) -> str:
    basename = Path(
        filename
    ).name.strip()

    if not basename:
        raise ValueError(
            "Uploaded filename is empty."
        )

    if basename.startswith("."):
        raise ValueError(
            "Hidden filenames are not allowed."
        )

    cleaned = (
        _SAFE_FILENAME_PATTERN
        .sub(
            "_",
            basename,
        )
        .strip(" .")
    )

    if not cleaned:
        raise ValueError(
            "Uploaded filename became empty "
            "after sanitization."
        )

    return cleaned


def _sanitize_category(
    category: str,
) -> str:
    cleaned = (
        _SAFE_CATEGORY_PATTERN
        .sub(
            "_",
            category.strip(),
        )
        .strip("_")
    )

    return (
        cleaned
        or settings
        .kb_default_category
    )


def _looks_sensitive(
    filename: str,
) -> bool:
    stem = (
        Path(filename)
        .stem
        .lower()
    )

    return any(
        fragment in stem
        for fragment in (
            _BLOCKED_NAME_FRAGMENTS
        )
    )


def _unique_destination(
    path: Path,
) -> Path:
    if not path.exists():
        return path

    timestamp = (
        datetime
        .now(timezone.utc)
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    return path.with_name(
        f"{path.stem}_{timestamp}"
        f"{path.suffix}"
    )


class KnowledgeBaseService:
    """
    Developer-only manual knowledge-base workflow.

    Upload:
        browser
        → staging directory only

    Rebuild:
        promote staged files
        → load documents
        → chunk
        → deduplicate
        → embed
        → rebuild Qdrant collection
    """

    def __init__(
        self,
        *,
        vector_store: QdrantVectorStore,
        embedder: BgeM3Embedder,
        documents_dir: Path = (
            settings.documents_dir
        ),
        staging_dir: Path = (
            settings.kb_staging_dir
        ),
        upload_max_mb: int = (
            settings.kb_upload_max_mb
        ),
    ) -> None:
        self.vector_store = (
            vector_store
        )

        self.embedder = embedder

        self.documents_dir = Path(
            documents_dir
        )

        self.staging_dir = Path(
            staging_dir
        )

        self.upload_max_bytes = (
            upload_max_mb
            * 1024
            * 1024
        )

        self.documents_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.staging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def supported_extensions(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                SUPPORTED_EXTENSIONS
            )
        )

    def stage_uploaded_files(
        self,
        uploaded_files: Iterable[
            UploadedFileLike
        ],
        *,
        category: str,
    ) -> list[
        StagedFileInfo
    ]:
        safe_category = (
            _sanitize_category(
                category
            )
        )

        category_dir = (
            self.staging_dir
            / safe_category
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        staged: list[
            StagedFileInfo
        ] = []

        for uploaded_file in (
            uploaded_files
        ):
            filename = (
                _sanitize_filename(
                    uploaded_file.name
                )
            )

            extension = (
                Path(filename)
                .suffix
                .lower()
            )

            if extension not in (
                SUPPORTED_EXTENSIONS
            ):
                raise ValueError(
                    "Unsupported file extension: "
                    f"{extension or '(none)'}"
                )

            if _looks_sensitive(
                filename
            ):
                raise ValueError(
                    "Filename appears sensitive "
                    f"and cannot be staged: "
                    f"{filename}"
                )

            content = (
                uploaded_file
                .getvalue()
            )

            if len(content) > (
                self.upload_max_bytes
            ):
                raise ValueError(
                    f"File exceeds the "
                    f"{self.upload_max_bytes // 1024 // 1024}"
                    f" MB limit: {filename}"
                )

            destination = (
                _unique_destination(
                    category_dir
                    / filename
                )
            )

            destination.write_bytes(
                content
            )

            staged.append(
                self._to_staged_info(
                    destination
                )
            )

        return staged

    def _to_staged_info(
        self,
        path: Path,
    ) -> StagedFileInfo:
        relative_path = (
            path
            .relative_to(
                self.staging_dir
            )
            .as_posix()
        )

        return StagedFileInfo(
            relative_path=(
                relative_path
            ),
            category=(
                path
                .relative_to(
                    self.staging_dir
                )
                .parts[0]
            ),
            filename=path.name,
            extension=(
                path.suffix.lower()
            ),
            size_bytes=(
                path.stat()
                .st_size
            ),
            modified_at_utc=(
                datetime
                .fromtimestamp(
                    path.stat()
                    .st_mtime,
                    tz=timezone.utc,
                )
                .isoformat(
                    timespec="seconds"
                )
            ),
        )

    def list_staged_files(
        self,
    ) -> list[
        StagedFileInfo
    ]:
        if not self.staging_dir.exists():
            return []

        return [
            self._to_staged_info(
                path
            )
            for path in sorted(
                self.staging_dir
                .rglob("*")
            )
            if path.is_file()
        ]

    def clear_staging(
        self,
    ) -> int:
        files = (
            self.list_staged_files()
        )

        if self.staging_dir.exists():
            shutil.rmtree(
                self.staging_dir
            )

        self.staging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return len(files)

    def promote_staged_files(
        self,
    ) -> list[str]:
        promoted: list[str] = []

        for staged_file in (
            self.list_staged_files()
        ):
            source = (
                self.staging_dir
                / staged_file
                .relative_path
            )

            destination = (
                _unique_destination(
                    self.documents_dir
                    / staged_file
                    .relative_path
                )
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(source),
                str(destination),
            )

            promoted.append(
                destination
                .relative_to(
                    self.documents_dir
                )
                .as_posix()
            )

        self.clear_staging()

        return promoted

    def rebuild(
        self,
        *,
        promote_staged: bool = True,
        batch_size: int | None = None,
    ) -> RebuildReport:
        started_at_utc = (
            utc_now_iso()
        )

        rebuild_id = (
            make_rebuild_id()
        )

        mode = (
            settings
            .mic_index_mode
            .strip()
            .lower()
        )

        if mode not in {
            "development",
            "production",
        }:
            raise ValueError(
                "MIC_INDEX_MODE must be either "
                "'development' or 'production'."
            )

        promoted_files = (
            self.promote_staged_files()
            if promote_staged
            else []
        )

        safety_assessments = (
            scan_document_tree(
                documents_dir=(
                    self.documents_dir
                ),
                supported_extensions=(
                    SUPPORTED_EXTENSIONS
                ),
                mode=mode,
                stale_days=(
                    settings
                    .index_stale_document_days
                ),
                allow_secret_documents=(
                    settings
                    .index_allow_secret_documents
                ),
                quarantine_synthetic_in_production=(
                    settings
                    .index_quarantine_synthetic_in_production
                ),
                quarantine_external_books_in_production=(
                    settings
                    .index_quarantine_external_books_in_production
                ),
                max_scan_chars=(
                    settings
                    .index_safety_scan_max_chars
                ),
            )
        )

        if any(
            item.status
            in {
                "rejected",
                "quarantined",
            }
            for item in safety_assessments
        ):
            safety_assessments = (
                apply_quarantine_actions(
                    assessments=(
                        safety_assessments
                    ),
                    documents_dir=(
                        self.documents_dir
                    ),
                    quarantine_dir=(
                        settings
                        .index_quarantine_dir
                    ),
                    rebuild_id=(
                        rebuild_id
                    ),
                )
            )

        safety_summary = (
            summarize_assessments(
                mode=mode,
                assessments=(
                    safety_assessments
                ),
            )
        )

        if (
            settings
            .index_fail_on_rejected_files
            and safety_summary
            .rejected_files
            > 0
        ):
            manifest = {
                "mode": mode,
                "rebuild_id": rebuild_id,
                "started_at_utc": (
                    started_at_utc
                ),
                "completed_at_utc": (
                    utc_now_iso()
                ),
                "promoted_files": (
                    promoted_files
                ),
                "document_safety": (
                    safety_summary
                    .to_dict()
                ),
                "error": (
                    "Rejected files were detected "
                    "and INDEX_FAIL_ON_REJECTED_FILES "
                    "is enabled."
                ),
            }

            manifest_path = (
                write_index_manifest(
                    manifest_dir=(
                        settings
                        .index_manifest_dir
                    ),
                    rebuild_id=(
                        rebuild_id
                    ),
                    manifest=manifest,
                )
            )

            raise ValueError(
                "Rejected files were detected. "
                f"See manifest: {manifest_path}"
            )

        snapshot_report = (
            self.vector_store
            .create_snapshot_report(
                enabled=(
                    settings
                    .index_snapshot_enabled
                )
            )
        )

        ingestion = load_documents(
            self.documents_dir
        )

        chunking = (
            chunk_document_units(
                ingestion.units,
                config=(
                    default_chunking_config()
                ),
            )
        )

        safety_by_source_path = {
            item.relative_path: item
            for item in (
                safety_assessments
            )
        }

        all_chunks = []

        for chunk in (
            chunking.chunks
        ):
            source_path = str(
                chunk.metadata.get(
                    "source_path",
                    "",
                )
            )

            safety = (
                safety_by_source_path
                .get(
                    source_path
                )
            )

            if safety is not None:
                chunk.metadata.update({
                    "index_rebuild_id": (
                        rebuild_id
                    ),
                    "index_mode": (
                        mode
                    ),
                    "index_file_status": (
                        safety.status
                    ),
                    "index_file_sha256": (
                        safety.sha256
                    ),
                    "index_file_warnings": (
                        safety.warnings
                    ),
                    "index_file_reasons": (
                        safety.reasons
                    ),
                })

            all_chunks.append(
                chunk
            )

        (
            unique_chunks,
            skipped_duplicate_ids,
        ) = select_unique_chunks(
            all_chunks
        )

        if not unique_chunks:
            completed_at_utc = (
                utc_now_iso()
            )

            manifest = {
                "mode": mode,
                "rebuild_id": rebuild_id,
                "started_at_utc": (
                    started_at_utc
                ),
                "completed_at_utc": (
                    completed_at_utc
                ),
                "promoted_files": (
                    promoted_files
                ),
                "snapshot": (
                    snapshot_report
                    .to_dict()
                ),
                "document_safety": (
                    safety_summary
                    .to_dict()
                ),
                "error": (
                    "No indexable document chunks "
                    "were produced."
                ),
            }

            manifest_path = (
                write_index_manifest(
                    manifest_dir=(
                        settings
                        .index_manifest_dir
                    ),
                    rebuild_id=(
                        rebuild_id
                    ),
                    manifest=manifest,
                )
            )

            raise ValueError(
                "No indexable document chunks "
                f"were produced. See manifest: {manifest_path}"
            )

        embedded = (
            self.embedder
            .embed_chunks(
                unique_chunks,
                batch_size=(
                    batch_size
                ),
                show_progress_bar=False,
            )
        )

        qdrant_report = (
            self.vector_store
            .rebuild_collection(
                embedded_chunks=(
                    embedded
                ),
                all_chunks=(
                    all_chunks
                ),
                batch_size=(
                    batch_size
                    or settings
                    .qdrant_upsert_batch_size
                ),
            )
        )

        completed_at_utc = (
            utc_now_iso()
        )

        report_without_manifest = {
            "mode": mode,
            "rebuild_id": (
                rebuild_id
            ),
            "started_at_utc": (
                started_at_utc
            ),
            "completed_at_utc": (
                completed_at_utc
            ),
            "manifest_path": None,
            "snapshot": (
                snapshot_report
                .to_dict()
            ),
            "document_safety": (
                safety_summary
                .to_dict()
            ),
            "promoted_files": (
                promoted_files
            ),
            "discovered_files": (
                ingestion
                .discovered_file_count
            ),
            "loaded_files": (
                ingestion
                .loaded_file_count
            ),
            "skipped_files": (
                ingestion
                .skipped_file_count
            ),
            "failed_files": (
                ingestion
                .failed_file_count
            ),
            "extracted_units": (
                ingestion
                .extracted_unit_count
            ),
            "all_chunks": len(
                all_chunks
            ),
            "unique_chunks": len(
                unique_chunks
            ),
            "skipped_exact_duplicates": len(
                skipped_duplicate_ids
            ),
            "vector_dimension": (
                embedded
                .vector_dimension
            ),
            "qdrant": (
                qdrant_report
                .to_dict()
            ),
        }

        manifest_path = (
            write_index_manifest(
                manifest_dir=(
                    settings
                    .index_manifest_dir
                ),
                rebuild_id=(
                    rebuild_id
                ),
                manifest=(
                    report_without_manifest
                ),
            )
        )

        return RebuildReport(
            mode=mode,
            rebuild_id=(
                rebuild_id
            ),
            started_at_utc=(
                started_at_utc
            ),
            completed_at_utc=(
                completed_at_utc
            ),
            manifest_path=(
                manifest_path
                .as_posix()
            ),
            snapshot=(
                snapshot_report
                .to_dict()
            ),
            document_safety=(
                safety_summary
                .to_dict()
            ),
            promoted_files=tuple(
                promoted_files
            ),
            discovered_files=(
                ingestion
                .discovered_file_count
            ),
            loaded_files=(
                ingestion
                .loaded_file_count
            ),
            skipped_files=(
                ingestion
                .skipped_file_count
            ),
            failed_files=(
                ingestion
                .failed_file_count
            ),
            extracted_units=(
                ingestion
                .extracted_unit_count
            ),
            all_chunks=len(
                all_chunks
            ),
            unique_chunks=len(
                unique_chunks
            ),
            skipped_exact_duplicates=len(
                skipped_duplicate_ids
            ),
            vector_dimension=(
                embedded
                .vector_dimension
            ),
            qdrant=(
                qdrant_report
            ),
        )

    def list_document_paths(
        self,
    ) -> list[str]:
        if not (
            self.documents_dir
            .exists()
        ):
            return []

        return [
            path
            .relative_to(
                self.documents_dir
            )
            .as_posix()
            for path in sorted(
                self.documents_dir
                .rglob("*")
            )
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]