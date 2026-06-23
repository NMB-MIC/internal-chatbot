from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, settings


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    rebuild_id: str | None = None
    mode: str | None = None
    manifest_path: str | None = None
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    indexed_points: int | None = None
    accepted_files: int | None = None
    quarantined_files: int | None = None
    rejected_files: int | None = None
    warned_files: int | None = None
    secret_hits: int | None = None
    synthetic_fixture_hits: int | None = None
    external_book_hits: int | None = None
    stale_hits: int | None = None
    snapshot_enabled: bool | None = None
    snapshot_created: bool | None = None
    snapshot_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeDiagnosticsSnapshot:
    generated_at_utc: str
    app: dict[str, Any]
    environment: dict[str, Any]
    qdrant: dict[str, Any]
    knowledge_base: dict[str, Any]
    latest_manifest: ManifestSummary
    manifest_inventory: list[dict[str, Any]] = field(default_factory=list)
    consistency: dict[str, Any] = field(default_factory=dict)
    selected_document: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "app": self.app,
            "environment": self.environment,
            "qdrant": self.qdrant,
            "knowledge_base": self.knowledge_base,
            "latest_manifest": self.latest_manifest.to_dict(),
            "manifest_inventory": self.manifest_inventory,
            "consistency": self.consistency,
            "selected_document": self.selected_document,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_project_path(value: str | Path | None, default_relative: str) -> Path:
    if value is None:
        path = PROJECT_ROOT / default_relative
    else:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

    return path


def _json_load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "_manifest_load_error": repr(exc),
            "_manifest_path": path.as_posix(),
        }


def _manifest_dir() -> Path:
    return _resolve_project_path(
        getattr(settings, "index_manifest_dir", None),
        "storage/index_manifests",
    )


def _list_manifest_paths(limit: int = 20) -> list[Path]:
    manifest_dir = _manifest_dir()

    if not manifest_dir.exists():
        return []

    paths = [
        path
        for path in manifest_dir.glob("*.json")
        if path.is_file()
    ]

    paths.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return paths[:limit]


def _summarize_manifest(path: Path) -> ManifestSummary:
    payload = _json_load(path)

    if "_manifest_load_error" in payload:
        return ManifestSummary(
            manifest_path=path.as_posix(),
            error=str(payload["_manifest_load_error"]),
        )

    safety = payload.get("document_safety") or {}
    qdrant = payload.get("qdrant") or {}
    snapshot = payload.get("snapshot") or {}

    return ManifestSummary(
        rebuild_id=payload.get("rebuild_id"),
        mode=payload.get("mode"),
        manifest_path=path.as_posix(),
        started_at_utc=payload.get("started_at_utc"),
        completed_at_utc=payload.get("completed_at_utc"),
        indexed_points=qdrant.get("indexed_points"),
        accepted_files=safety.get("accepted_files"),
        quarantined_files=safety.get("quarantined_files"),
        rejected_files=safety.get("rejected_files"),
        warned_files=safety.get("warned_files"),
        secret_hits=safety.get("secret_hits"),
        synthetic_fixture_hits=safety.get("synthetic_fixture_hits"),
        external_book_hits=safety.get("external_book_hits"),
        stale_hits=safety.get("stale_hits"),
        snapshot_enabled=snapshot.get("enabled"),
        snapshot_created=snapshot.get("created"),
        snapshot_name=snapshot.get("snapshot_name"),
        error=None,
    )


def _manifest_inventory(limit: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in _list_manifest_paths(limit=limit):
        summary = _summarize_manifest(path)
        rows.append({
            "rebuild_id": summary.rebuild_id,
            "mode": summary.mode,
            "indexed_points": summary.indexed_points,
            "accepted_files": summary.accepted_files,
            "quarantined_files": summary.quarantined_files,
            "rejected_files": summary.rejected_files,
            "snapshot_created": summary.snapshot_created,
            "started_at_utc": summary.started_at_utc,
            "manifest_path": summary.manifest_path,
            "error": summary.error,
        })

    return rows


def _safe_call(description: str, fn: Any) -> dict[str, Any]:
    try:
        return {
            "ok": True,
            "value": fn(),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "value": None,
            "error": f"{description}: {exc!r}",
        }


def _qdrant_summary(backend: Any) -> dict[str, Any]:
    vector_store = getattr(backend, "vector_store", None)

    if vector_store is None:
        knowledge_base = getattr(backend, "knowledge_base", None)
        vector_store = getattr(knowledge_base, "vector_store", None)

    if vector_store is None:
        return {
            "ok": False,
            "error": "No vector_store found on backend or knowledge_base.",
        }

    health = _safe_call(
        "qdrant healthcheck",
        vector_store.healthcheck,
    )

    count = _safe_call(
        "qdrant count_points",
        vector_store.count_points,
    )

    return {
        "ok": bool(health["ok"] and count["ok"]),
        "url": getattr(vector_store, "url", None),
        "collection_name": getattr(vector_store, "collection_name", None),
        "dense_vector_name": getattr(vector_store, "dense_vector_name", None),
        "health": health["value"],
        "point_count": count["value"],
        "error": health["error"] or count["error"],
    }


def _knowledge_base_summary(backend: Any) -> dict[str, Any]:
    knowledge_base = getattr(backend, "knowledge_base", None)

    if knowledge_base is None:
        return {
            "ok": False,
            "error": "No knowledge_base found on backend.",
        }

    documents_result = _safe_call(
        "list_document_paths",
        knowledge_base.list_document_paths,
    )

    staged_result = _safe_call(
        "list_staged_files",
        knowledge_base.list_staged_files,
    )

    documents = documents_result["value"] or []
    staged_files = staged_result["value"] or []

    categories = Counter(
        str(path).split("/", 1)[0]
        if "/" in str(path)
        else "uncategorized"
        for path in documents
    )

    extensions = Counter(
        Path(str(path)).suffix.lower() or "(none)"
        for path in documents
    )

    return {
        "ok": bool(documents_result["ok"] and staged_result["ok"]),
        "documents_dir": getattr(knowledge_base, "documents_dir", None).as_posix()
        if getattr(knowledge_base, "documents_dir", None)
        else None,
        "staging_dir": getattr(knowledge_base, "staging_dir", None).as_posix()
        if getattr(knowledge_base, "staging_dir", None)
        else None,
        "document_count": len(documents),
        "staged_file_count": len(staged_files),
        "categories": dict(sorted(categories.items())),
        "extensions": dict(sorted(extensions.items())),
        "documents": documents,
        "error": documents_result["error"] or staged_result["error"],
    }


def build_runtime_diagnostics_snapshot(
    backend: Any,
    *,
    selected_document: str | None = None,
    manifest_limit: int = 10,
) -> RuntimeDiagnosticsSnapshot:
    manifest_paths = _list_manifest_paths(limit=manifest_limit)

    latest_manifest = (
        _summarize_manifest(manifest_paths[0])
        if manifest_paths
        else ManifestSummary(
            manifest_path=_manifest_dir().as_posix(),
            error="No index manifests found.",
        )
    )

    qdrant = _qdrant_summary(backend)
    knowledge_base = _knowledge_base_summary(backend)

    active_point_count = qdrant.get("point_count")
    manifest_point_count = latest_manifest.indexed_points

    consistency = {
        "active_index_matches_latest_manifest": (
            active_point_count is not None
            and manifest_point_count is not None
            and int(active_point_count) == int(manifest_point_count)
        ),
        "active_point_count": active_point_count,
        "latest_manifest_indexed_points": manifest_point_count,
        "selected_document_exists": (
            selected_document in set(knowledge_base.get("documents") or [])
            if selected_document
            else None
        ),
        "manifest_dir": _manifest_dir().as_posix(),
    }

    return RuntimeDiagnosticsSnapshot(
        generated_at_utc=_utc_now(),
        app={
            "app_name": getattr(settings, "app_name", None),
            "app_env": getattr(settings, "app_env", None),
            "debug": getattr(settings, "debug", None),
            "ollama_model": getattr(settings, "ollama_model", None),
            "embedding_model_name": getattr(settings, "embedding_model_name", None),
            "document_scope_default_behavior": getattr(settings, "document_scope_default_behavior", None),
        },
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "project_root": PROJECT_ROOT.as_posix(),
            "mic_index_mode": getattr(settings, "mic_index_mode", None),
            "manifest_dir": _manifest_dir().as_posix(),
        },
        qdrant=qdrant,
        knowledge_base=knowledge_base,
        latest_manifest=latest_manifest,
        manifest_inventory=_manifest_inventory(limit=manifest_limit),
        consistency=consistency,
        selected_document=selected_document,
    )
