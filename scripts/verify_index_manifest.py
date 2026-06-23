from __future__ import annotations

import argparse
from pathlib import Path

from _index_cli_common import (
    default_manifest_dir,
    dump_json,
    find_manifest,
    load_json,
)


def _status_lookup(manifest: dict) -> dict[str, str]:
    files = (manifest.get("document_safety") or {}).get("files", [])
    return {
        str(item.get("relative_path")): str(item.get("status"))
        for item in files
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assert key properties of a MIC 9000 index manifest."
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--rebuild-id", default="latest")
    parser.add_argument("--manifest-dir", default=None)
    parser.add_argument("--expect-mode", choices=("development", "production"), default=None)
    parser.add_argument("--expect-indexed-points", type=int, default=None)
    parser.add_argument("--require-snapshot", action="store_true")
    parser.add_argument("--require-accepted", action="append", default=[])
    parser.add_argument("--require-quarantined", action="append", default=[])
    parser.add_argument("--require-rejected", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).expanduser().resolve() if args.manifest_dir else default_manifest_dir()
    manifest_path = find_manifest(
        manifest=args.manifest,
        rebuild_id=args.rebuild_id,
        manifest_dir=manifest_dir,
    )
    manifest = load_json(manifest_path)
    status_by_path = _status_lookup(manifest)

    failures: list[str] = []

    if args.expect_mode and manifest.get("mode") != args.expect_mode:
        failures.append(f"Expected mode={args.expect_mode}, got {manifest.get('mode')}")

    indexed_points = (manifest.get("qdrant") or {}).get("indexed_points")
    if args.expect_indexed_points is not None and indexed_points != args.expect_indexed_points:
        failures.append(
            f"Expected indexed_points={args.expect_indexed_points}, got {indexed_points}"
        )

    snapshot = manifest.get("snapshot") or {}
    if args.require_snapshot and not snapshot.get("created"):
        failures.append("Expected snapshot.created=true")

    for relative_path in args.require_accepted:
        if status_by_path.get(relative_path) != "accepted":
            failures.append(f"Expected accepted: {relative_path}; got {status_by_path.get(relative_path)}")

    for relative_path in args.require_quarantined:
        if status_by_path.get(relative_path) != "quarantined":
            failures.append(f"Expected quarantined: {relative_path}; got {status_by_path.get(relative_path)}")

    for relative_path in args.require_rejected:
        if status_by_path.get(relative_path) != "rejected":
            failures.append(f"Expected rejected: {relative_path}; got {status_by_path.get(relative_path)}")

    report = {
        "manifest_path": str(manifest_path),
        "rebuild_id": manifest.get("rebuild_id"),
        "mode": manifest.get("mode"),
        "indexed_points": indexed_points,
        "snapshot_created": snapshot.get("created"),
        "failure_count": len(failures),
        "failures": failures,
    }

    print(dump_json(report))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
