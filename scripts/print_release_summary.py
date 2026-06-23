#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a readable MIC 9000 release manifest summary.")
    parser.add_argument("manifest")
    args = parser.parse_args()

    path = Path(args.manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    latest = data.get("latest_manifest") or {}
    evals = data.get("eval_reports") or {}

    print(f"MIC 9000 Release: {data.get('version')} ({data.get('release_stage')})")
    print(f"Release ID: {data.get('release_id')}")
    print(f"Gate: {data.get('release_gate')}")
    if data.get("gate_failures"):
        print("Failures:")
        for failure in data["gate_failures"]:
            print(f"- {failure}")
    print()
    print("Index")
    print(f"- rebuild_id: {latest.get('rebuild_id')}")
    print(f"- mode: {latest.get('mode')}")
    print(f"- indexed_points: {latest.get('indexed_points')}")
    print(f"- accepted_files: {latest.get('accepted_files')}")
    print(f"- quarantined_files: {latest.get('quarantined_files')}")
    print(f"- rejected_files: {latest.get('rejected_files')}")
    print(f"- snapshot_created: {latest.get('snapshot_created')}")
    print()
    print("Eval suites")
    for name, report in evals.items():
        print(f"- {name}: {report.get('passed_cases')}/{report.get('total_cases')} passed, failed={report.get('failed_cases')}")
    return 0 if data.get("release_gate") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
