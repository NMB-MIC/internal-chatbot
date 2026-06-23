from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path.cwd().resolve()


def default_manifest_dir() -> Path:
    try:
        from app.config import settings  # type: ignore

        return Path(settings.index_manifest_dir)
    except Exception:
        return project_root() / "storage" / "index_manifests"


def default_documents_dir() -> Path:
    try:
        from app.config import settings  # type: ignore

        return Path(settings.documents_dir)
    except Exception:
        return project_root() / "data" / "documents"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )


def find_manifest(
    *,
    manifest: str | None,
    rebuild_id: str | None,
    manifest_dir: Path | None = None,
) -> Path:
    if manifest:
        path = Path(manifest).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path

    root = manifest_dir or default_manifest_dir()
    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Manifest directory not found: {root}")

    if rebuild_id and rebuild_id != "latest":
        path = root / f"{rebuild_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found for rebuild_id={rebuild_id}: {path}")
        return path

    manifests = sorted(
        root.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    if not manifests:
        raise FileNotFoundError(f"No manifests found in {root}")

    return manifests[0]


def manifest_summary(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    safety = manifest.get("document_safety", {}) or {}
    snapshot = manifest.get("snapshot", {}) or {}
    qdrant = manifest.get("qdrant", {}) or {}
    return {
        "path": str(path),
        "rebuild_id": manifest.get("rebuild_id"),
        "mode": manifest.get("mode"),
        "started_at_utc": manifest.get("started_at_utc"),
        "completed_at_utc": manifest.get("completed_at_utc"),
        "accepted_files": safety.get("accepted_files"),
        "quarantined_files": safety.get("quarantined_files"),
        "rejected_files": safety.get("rejected_files"),
        "secret_hits": safety.get("secret_hits"),
        "synthetic_fixture_hits": safety.get("synthetic_fixture_hits"),
        "external_book_hits": safety.get("external_book_hits"),
        "snapshot_created": snapshot.get("created"),
        "snapshot_name": snapshot.get("snapshot_name"),
        "collection_name": qdrant.get("collection_name") or snapshot.get("collection_name"),
        "indexed_points": qdrant.get("indexed_points"),
        "point_count_after_upsert": qdrant.get("point_count_after_upsert"),
    }
