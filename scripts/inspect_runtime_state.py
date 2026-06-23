from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.runtime import build_backend
from app.services.runtime_diagnostics import build_runtime_diagnostics_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect MIC 9000 runtime/index diagnostics."
    )
    parser.add_argument(
        "--selected-document",
        default=None,
        help="Optional selected document path to validate against the current KB.",
    )
    parser.add_argument(
        "--manifest-limit",
        type=int,
        default=10,
        help="Number of recent manifests to include.",
    )
    parser.add_argument(
        "--no-warm-embedding",
        action="store_true",
        help="Avoid explicit embedding warmup while building the backend.",
    )
    args = parser.parse_args()

    backend = build_backend(
        warm_embedding=not args.no_warm_embedding,
    )

    snapshot = build_runtime_diagnostics_snapshot(
        backend,
        selected_document=args.selected_document,
        manifest_limit=args.manifest_limit,
    )

    print(json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        indent=2,
    ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
