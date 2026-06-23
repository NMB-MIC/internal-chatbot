from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from app.services.runtime import build_backend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the MIC 9000 Qdrant knowledge-base index."
    )

    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not promote staged files before rebuilding.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding/upsert batch size override.",
    )

    args = parser.parse_args()

    backend = build_backend(
        warm_embedding=True,
    )

    report = (
        backend
        .knowledge_base
        .rebuild(
            promote_staged=(
                not args.no_promote
            ),
            batch_size=(
                args.batch_size
            ),
        )
    )

    print(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
