from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _index_cli_common import (
    default_manifest_dir,
    dump_json,
    find_manifest,
    load_json,
)
from restore_quarantined_files import restore_items


def _snapshot_url(
    *,
    qdrant_url: str,
    collection_name: str,
    snapshot_name: str,
) -> str:
    return (
        qdrant_url.rstrip("/")
        + f"/collections/{collection_name}/snapshots/{snapshot_name}"
    )


def _load_settings() -> Any:
    from app.config import settings  # type: ignore

    return settings


def _attempt_qdrant_snapshot_recover(
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    settings = _load_settings()

    from qdrant_client import QdrantClient  # type: ignore

    snapshot = manifest.get("snapshot") or {}
    qdrant = manifest.get("qdrant") or {}

    collection_name = (
        snapshot.get("collection_name")
        or qdrant.get("collection_name")
        or settings.qdrant_collection_name
    )
    snapshot_name = snapshot.get("snapshot_name")

    if not snapshot.get("created") or not snapshot_name:
        raise ValueError("Manifest does not contain a created Qdrant snapshot.")

    location = _snapshot_url(
        qdrant_url=settings.qdrant_url,
        collection_name=collection_name,
        snapshot_name=snapshot_name,
    )

    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )

    if not hasattr(client, "recover_snapshot"):
        raise RuntimeError(
            "Installed qdrant-client does not expose recover_snapshot(). "
            "Use the manual snapshot URL plan from this script."
        )

    result = client.recover_snapshot(
        collection_name=collection_name,
        location=location,
        wait=True,
    )

    return {
        "collection_name": collection_name,
        "snapshot_name": snapshot_name,
        "snapshot_location": location,
        "recover_result": str(result),
    }


def _run_rebuild(
    *,
    index_mode: str | None,
) -> dict[str, Any]:
    env = os.environ.copy()

    if index_mode:
        env["MIC_INDEX_MODE"] = index_mode

    command = [
        sys.executable,
        "scripts/rebuild_index.py",
    ]

    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    return {
        "command": command,
        "index_mode": env.get("MIC_INDEX_MODE"),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch 10.1 rollback helper. Can restore quarantined files, "
            "attempt Qdrant snapshot recovery, and/or rebuild the index."
        )
    )
    parser.add_argument("--manifest", default=None, help="Manifest JSON to roll back from.")
    parser.add_argument("--rebuild-id", default="latest", help="Manifest rebuild_id, or 'latest'.")
    parser.add_argument("--manifest-dir", default=None, help="Manifest directory.")
    parser.add_argument(
        "--action",
        choices=("plan", "restore-files", "recover-snapshot", "rebuild", "restore-files-and-rebuild"),
        default="plan",
        help="Rollback action to perform.",
    )
    parser.add_argument(
        "--restore-status",
        choices=("quarantined", "rejected", "all"),
        default="quarantined",
        help="File statuses to restore when action restores files.",
    )
    parser.add_argument("--documents-dir", default=None, help="Target data/documents directory.")
    parser.add_argument("--overwrite-documents", action="store_true", help="Overwrite documents during restore.")
    parser.add_argument("--move", action="store_true", help="Move instead of copy from quarantine.")
    parser.add_argument(
        "--rebuild-index-mode",
        choices=("development", "production"),
        default=None,
        help="Set MIC_INDEX_MODE when action runs a rebuild.",
    )
    parser.add_argument("--yes", action="store_true", help="Actually perform destructive actions.")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).expanduser().resolve() if args.manifest_dir else default_manifest_dir()
    manifest_path = find_manifest(
        manifest=args.manifest,
        rebuild_id=args.rebuild_id,
        manifest_dir=manifest_dir,
    )
    manifest = load_json(manifest_path)

    snapshot = manifest.get("snapshot") or {}
    qdrant = manifest.get("qdrant") or {}

    plan = {
        "manifest_path": str(manifest_path),
        "rebuild_id": manifest.get("rebuild_id"),
        "manifest_mode": manifest.get("mode"),
        "action": args.action,
        "snapshot": snapshot,
        "qdrant": {
            "collection_name": qdrant.get("collection_name") or snapshot.get("collection_name"),
            "indexed_points": qdrant.get("indexed_points"),
            "point_count_after_upsert": qdrant.get("point_count_after_upsert"),
        },
        "notes": [
            "A manifest snapshot is the collection state before that rebuild ran.",
            "For example, the production rebuild manifest's snapshot can restore the previous dev index.",
            "File restore copies from quarantine by default so the audit trail remains intact.",
        ],
    }

    if args.action == "plan":
        print(dump_json(plan))
        return

    if not args.yes:
        plan["dry_run"] = True
        plan["message"] = "No changes made. Re-run with --yes to execute this action."
        print(dump_json(plan))
        return

    result: dict[str, Any] = {
        **plan,
        "dry_run": False,
        "steps": [],
    }

    if args.action in {"restore-files", "restore-files-and-rebuild"}:
        from _index_cli_common import default_documents_dir

        documents_dir = Path(args.documents_dir).expanduser().resolve() if args.documents_dir else default_documents_dir()
        restore_report = restore_items(
            manifest_path=manifest_path,
            documents_dir=documents_dir,
            status=args.restore_status,
            dry_run=False,
            overwrite=args.overwrite_documents,
            move=args.move,
        )
        result["steps"].append({
            "step": "restore_files",
            "report": restore_report,
        })

    if args.action == "recover-snapshot":
        try:
            recover_report = _attempt_qdrant_snapshot_recover(
                manifest=manifest,
            )
            result["steps"].append({
                "step": "recover_snapshot",
                "report": recover_report,
            })
        except Exception as exc:
            result["steps"].append({
                "step": "recover_snapshot",
                "error": f"{type(exc).__name__}: {exc}",
            })
            result["manual_snapshot_url_hint"] = (
                "If Qdrant recovery failed, open/download the snapshot URL and recover it "
                "using the Qdrant REST API or dashboard."
            )

    if args.action in {"rebuild", "restore-files-and-rebuild"}:
        rebuild_report = _run_rebuild(
            index_mode=args.rebuild_index_mode,
        )
        result["steps"].append({
            "step": "rebuild_index",
            "report": rebuild_report,
        })

    print(dump_json(result))


if __name__ == "__main__":
    main()
