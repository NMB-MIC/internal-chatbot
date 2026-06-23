from __future__ import annotations

import json
from pathlib import Path

from app.eval.models import EvalRunReport


def write_json_report(report: EvalRunReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def write_markdown_report(report: EvalRunReport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# MIC 9000 Eval Report — {report.suite_name}")
    lines.append("")
    lines.append(f"- Suite version: `{report.suite_version}`")
    lines.append(f"- Started: `{report.started_at_utc}`")
    lines.append(f"- Completed: `{report.completed_at_utc}`")
    lines.append(f"- Duration: `{report.duration_seconds:.3f}s`")
    lines.append(f"- Session ID: `{report.session_id}`")
    lines.append(f"- Isolated cases: `{report.metadata.get('isolate_cases')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total | {report.total_cases} |")
    lines.append(f"| Passed | {report.passed_cases} |")
    lines.append(f"| Failed | {report.failed_cases} |")
    lines.append(f"| Skipped | {report.skipped_cases} |")
    lines.append(f"| Pass rate | {report.pass_rate:.2%} |")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Status | Case | Session | Latency | Failures |")
    lines.append("|---|---|---|---:|---|")

    for result in report.results:
        failures = "<br>".join(result.failures) if result.failures else ""
        lines.append(
            f"| {result.status} | `{result.case_id}` | "
            f"`{result.session_id or ''}` | "
            f"{result.latency_seconds:.3f}s | {failures} |"
        )

    lines.append("")
    lines.append("## Failed case details")
    lines.append("")

    failed_results = [result for result in report.results if not result.passed and not result.skipped]
    if not failed_results:
        lines.append("No failed cases.")
    else:
        for result in failed_results:
            lines.append(f"### {result.case_id}")
            lines.append("")
            lines.append("**Question**")
            lines.append("")
            lines.append(result.question)
            lines.append("")
            lines.append("**Failures**")
            lines.append("")
            for failure in result.failures:
                lines.append(f"- {failure}")
            lines.append("")
            lines.append("**Answer**")
            lines.append("")
            lines.append("```text")
            lines.append(result.answer)
            lines.append("```")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
