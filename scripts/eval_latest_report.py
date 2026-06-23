from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print summary of latest eval report.")
    parser.add_argument("--dir", default="eval_reports", help="Eval report directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = Path(args.dir)
    reports = sorted(report_dir.glob("*.json"), reverse=True)
    if not reports:
        print("No JSON eval reports found.")
        return 1

    path = reports[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "path": str(path),
        "suite_name": data.get("suite_name"),
        "started_at_utc": data.get("started_at_utc"),
        "total_cases": data.get("total_cases"),
        "passed_cases": data.get("passed_cases"),
        "failed_cases": data.get("failed_cases"),
        "skipped_cases": data.get("skipped_cases"),
        "pass_rate": data.get("pass_rate"),
        "failed_case_ids": [
            result.get("case_id")
            for result in data.get("results", [])
            if not result.get("passed") and not result.get("skipped")
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
