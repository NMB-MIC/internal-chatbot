from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.case_loader import discover_eval_suites, load_eval_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List MIC 9000 eval suites.")
    parser.add_argument("--dir", default="eval_suites", help="Eval suite directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = discover_eval_suites(args.dir)

    if not paths:
        print("No eval suites found.")
        return 0

    print("suite | version | cases | path")
    print("-" * 100)
    for path in paths:
        suite = load_eval_suite(path)
        print(f"{suite.name} | {suite.version} | {len(suite.cases)} | {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
