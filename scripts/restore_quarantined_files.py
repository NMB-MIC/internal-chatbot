from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from _index_cli_common import (
    default_documents_dir,
    default_manifest_dir,
    dump_json,
    find_manifest,
    load_json,
)


def _selected_statuses(status: str) -> set[str]:
    if status == "all":
        return {"quarantined", "rejected"}
    return {status}


def _iter_restore_items(
    manifest: dict[str, Any],
    *,
    statuses: set[str],
) -> list[dict[str, Any]]:
    files = (
        (manifest.get("document_safety") or {})
        .get("files", [])
    )

    items: list[dict[str, Any]] = []

    for item in files:
        if item.get("status") not in statuses:
            continue

        quarantined_to = item.get("quarantined_to")
        relative_path = item.get("relative_path")

        if not quarantined_to or not relative_path:
            continue

        items.append(item)

    return items


def restore_items(
    *,
    manifest_path: Path,
    documents_dir: Path,
    status: str,
    dry_run: bool,
    overwrite: bool,
    move: bool,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    statuses = _selected_statuses(status)
    items = _iter_restore_items(
        manifest,
        statuses=statuses,
    )

    restored: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    documents_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in items:
        source = Path(str(item["quarantined_to"])).expanduser()
        destination = documents_dir / str(item["relative_path"])

        action = "move" if move else "copy"

        if not source.exists():
            skipped.append({
                "relative_path": item.get("relative_path"),
                "reason": "quarantine source missing",
                "source": str(source),
                "destination": str(destination),
            })
            continue

        if destination.exists() and not overwrite:
            skipped.append({
                "relative_path": item.get("relative_path"),
                "reason": "destination exists; pass --overwrite to replace",
                "source": str(source),
                "destination": str(destination),
            })
            continue

        restored.append({
            "relative_path": item.get("relative_path"),
            "status": item.get("status"),
            "action": "dry-run" if dry_run else action,
            "source": str(source),
            "destination": str(destination),
        })

        if dry_run:
            continue

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists() and overwrite:
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        if move:
            shutil.move(
                str(source),
                str(destination),
            )
        else:
            shutil.copy2(
                source,
                destination,
            )

    return {
        "manifest_path": str(manifest_path),
        "rebuild_id": manifest.get("rebuild_id"),
        "mode": manifest.get("mode"),
        "documents_dir": str(documents_dir),
        "status_filter": status,
        "dry_run": dry_run,
        "move": move,
        "overwrite": overwrite,
        "matched_items": len(items),
        "restored_items": len(restored),
        "skipped_items": len(skipped),
        "restored": restored,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore MIC 9000 files that Batch 10 moved into index quarantine."
    )
    parser.add_argument("--manifest", default=None, help="Path to a manifest JSON file.")
    parser.add_argument("--rebuild-id", default="latest", help="Manifest rebuild_id, or 'latest'.")
    parser.add_argument("--manifest-dir", default=None, help="Manifest directory.")
    parser.add_argument("--documents-dir", default=None, help="Target data/documents directory.")
    parser.add_argument(
        "--status",
        choices=("quarantined", "rejected", "all"),
        default="quarantined",
        help="Which quarantine statuses to restore.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    parser.add_argument("--move", action="store_true", help="Move instead of copy from quarantine.")
    parser.add_argument("--yes", action="store_true", help="Actually restore. Without this, the script is dry-run only.")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).expanduser().resolve() if args.manifest_dir else default_manifest_dir()
    manifest_path = find_manifest(
        manifest=args.manifest,
        rebuild_id=args.rebuild_id,
        manifest_dir=manifest_dir,
    )
    documents_dir = Path(args.documents_dir).expanduser().resolve() if args.documents_dir else default_documents_dir()

    report = restore_items(
        manifest_path=manifest_path,
        documents_dir=documents_dir,
        status=args.status,
        dry_run=not args.yes,
        overwrite=args.overwrite,
        move=args.move,
    )

    print(dump_json(report))

    if not args.yes:
        print("\nDry run only. Re-run with --yes to restore files.")


if __name__ == "__main__":
    main()
