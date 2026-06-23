from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.eval.case_loader import load_eval_suite
from app.eval.reporting import write_json_report, write_markdown_report
from app.eval.runner import EvalRunner
from app.services.runtime import build_backend


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a MIC 9000 evaluation suite."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Path to eval suite JSON file.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Only run this case ID. Can be passed multiple times.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Only run cases with this tag. Can be passed multiple times.",
    )
    parser.add_argument(
        "--user-id",
        default="mic9000-eval",
        help="Eval user ID for memory sessions.",
    )
    parser.add_argument(
        "--session-title",
        default=None,
        help="Optional session title. Defaults to suite name.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_reports",
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument(
        "--no-warm-embedding",
        action="store_true",
        help="Pass warm_embedding=False to build_backend().",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failed case.",
    )
    parser.add_argument(
        "--print-answers",
        action="store_true",
        help="Print each answer to stdout while running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    suite = load_eval_suite(args.suite)
    session_title = args.session_title or f"Eval: {suite.name}"

    backend = build_backend(
        warm_embedding=not args.no_warm_embedding,
    )

    runner = EvalRunner(
        backend=backend,
        user_id=args.user_id,
        session_title=session_title,
    )

    report = runner.run(
        suite,
        case_ids=set(args.case_id) if args.case_id else None,
        tags=set(args.tag) if args.tag else None,
        fail_fast=args.fail_fast,
    )

    output_dir = Path(args.output_dir)
    basename = f"{_timestamp()}_{_safe_name(suite.name)}"
    json_path = output_dir / f"{basename}.json"
    md_path = output_dir / f"{basename}.md"

    write_json_report(report, json_path)
    write_markdown_report(report, md_path)

    summary = {
        "suite_name": report.suite_name,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "skipped_cases": report.skipped_cases,
        "pass_rate": report.pass_rate,
        "session_id": report.session_id,
        "json_report": str(json_path),
        "markdown_report": str(md_path),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.print_answers:
        for result in report.results:
            print("=" * 100)
            print(result.status, result.case_id)
            print(result.question)
            print("-" * 100)
            print(result.answer)
            if result.failures:
                print("FAILURES:")
                for failure in result.failures:
                    print("-", failure)

    return 0 if report.failed_cases == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
