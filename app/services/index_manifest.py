from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_index_manifest(
    *,
    manifest_dir: Path,
    rebuild_id: str,
    manifest: dict[str, Any],
) -> Path:
    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        manifest_dir
        / f"{rebuild_id}.json"
    )

    path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    return path
