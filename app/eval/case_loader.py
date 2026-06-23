from __future__ import annotations

import json
from pathlib import Path

from app.eval.models import EvalSuite


def load_eval_suite(path: str | Path) -> EvalSuite:
    suite_path = Path(path).expanduser().resolve()

    if not suite_path.exists():
        raise FileNotFoundError(f"Eval suite not found: {suite_path}")

    with suite_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return EvalSuite.from_dict(data, path=suite_path)


def discover_eval_suites(directory: str | Path) -> list[Path]:
    suite_dir = Path(directory).expanduser().resolve()

    if not suite_dir.exists():
        return []

    return sorted(suite_dir.glob("*.json"))
