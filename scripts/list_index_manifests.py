from __future__ import annotations

import argparse
from pathlib import Path

from _index_cli_common import (
    default_manifest_dir,
    dump_json,
    load_json,
    manifest_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List MIC 9000 index rebuild manifests."
    )
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="Directory containing manifest JSON files. Defaults to settings.index_manifest_dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of manifests to display.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a compact table.",
    )
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir).expanduser().resolve() if args.manifest_dir else default_manifest_dir()

    if not manifest_dir.exists():
        raise FileNotFoundError(f"Manifest directory does not exist: {manifest_dir}")

    paths = sorted(
        manifest_dir.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[: max(args.limit, 1)]

    rows = [
        manifest_summary(
            load_json(path),
            path,
        )
        for path in paths
    ]

    if args.json:
        print(dump_json(rows))
        return

    if not rows:
        print(f"No manifests found in {manifest_dir}")
        return

    header = (
        "rebuild_id",
        "mode",
        "indexed",
        "accepted",
        "quarantined",
        "rejected",
        "snapshot",
        "started_at_utc",
    )

    print(" | ".join(header))
    print("-" * 120)

    for row in rows:
        print(
            " | ".join(
                str(value)
                for value in (
                    row.get("rebuild_id"),
                    row.get("mode"),
                    row.get("indexed_points"),
                    row.get("accepted_files"),
                    row.get("quarantined_files"),
                    row.get("rejected_files"),
                    "yes" if row.get("snapshot_created") else "no",
                    row.get("started_at_utc"),
                )
            )
        )


if __name__ == "__main__":
    main()
