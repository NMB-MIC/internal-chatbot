from __future__ import annotations

import json
import re
from typing import Any

from app.eval.models import EvalCase

_CITATION_PATTERN = re.compile(r"\[(S\d+)(?:\s*,\s*S\d+)*\]")


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


def extract_cited_source_ids(answer: str) -> list[str]:
    seen: set[str] = set()
    source_ids: list[str] = []

    for match in re.finditer(r"S\d+", answer):
        source_id = match.group(0)
        if source_id in seen:
            continue
        seen.add(source_id)
        source_ids.append(source_id)

    return source_ids


def answer_has_valid_markdown(answer: str) -> tuple[bool, str]:
    prepared = answer.strip()

    if prepared.count("```") % 2 != 0:
        return False, "unclosed code fence"

    prepared_without_fences = re.sub(
        r"```.*?```",
        "",
        prepared,
        flags=re.DOTALL,
    )

    if prepared_without_fences.count("`") % 2 != 0:
        return False, "unmatched inline-code marker"

    if prepared_without_fences.count("**") % 2 != 0:
        return False, "unmatched bold marker"

    if prepared.endswith((":", ",", ";", "-", "—", "(")):
        return False, "answer ends with dangling punctuation"

    if re.search(r"\b(and|or|including|such as)\s+`?[\w_.:/-]*:?\s*$", prepared, flags=re.I):
        return False, "answer appears truncated after conjunction or field name"

    if re.search(r"`?[\w_.-]+:\s*$", prepared):
        return False, "answer ends with unfinished technical field"

    return True, "ok"


def _extract_result_value(result: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = result
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok:
            return current
    return None


def _json_dump_for_search(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check_count(
    *,
    failures: list[str],
    name: str,
    actual_raw: Any,
    expected_exact: int | None,
    expected_min: int | None,
    expected_max: int | None,
) -> None:
    actual = _as_int(actual_raw)

    if expected_exact is not None:
        if actual != expected_exact:
            failures.append(f"expected {name}={expected_exact}, got {actual_raw}")
        return

    if actual is None:
        if expected_min is not None or expected_max is not None:
            failures.append(f"missing retrieval count for {name}")
        return

    if expected_min is not None and actual < expected_min:
        failures.append(f"expected {name}>={expected_min}, got {actual}")

    if expected_max is not None and actual > expected_max:
        failures.append(f"expected {name}<={expected_max}, got {actual}")


def evaluate_case_output(
    *,
    case: EvalCase,
    answer: str,
    result: dict[str, Any],
    retrieval_summary: dict[str, Any] | None,
    latency_seconds: float,
) -> list[str]:
    failures: list[str] = []

    if case.expected_answerable is not None:
        answerable = _extract_result_value(result, "answerable", "debug.answerable")
        if answerable is not None and bool(answerable) != case.expected_answerable:
            failures.append(
                f"expected answerable={case.expected_answerable}, got {answerable}"
            )

    if case.expected_route is not None:
        route = _extract_result_value(
            result,
            "route",
            "selected_route",
            "metadata.route",
            "debug.route",
        )
        if route is not None and str(route) != case.expected_route:
            failures.append(f"expected route={case.expected_route}, got {route}")

    if case.min_answer_chars is not None and len(answer.strip()) < case.min_answer_chars:
        failures.append(
            f"answer too short: {len(answer.strip())} < {case.min_answer_chars} chars"
        )

    if case.max_latency_seconds is not None and latency_seconds > case.max_latency_seconds:
        failures.append(
            f"latency too high: {latency_seconds:.3f}s > {case.max_latency_seconds:.3f}s"
        )

    for fragment in case.expected_fragments:
        if not _contains(answer, fragment):
            failures.append(f"missing expected fragment: {fragment}")

    for alternatives in case.expected_any_fragments:
        if alternatives and not any(_contains(answer, item) for item in alternatives):
            failures.append(
                "missing one of expected alternatives: " + " | ".join(alternatives)
            )

    for fragment in case.forbidden_fragments:
        if _contains(answer, fragment):
            failures.append(f"forbidden fragment present: {fragment}")

    if case.require_sources:
        if not _CITATION_PATTERN.search(answer):
            failures.append("answer does not contain a valid [S#] citation")

    if case.require_valid_markdown:
        ok, reason = answer_has_valid_markdown(answer)
        if not ok:
            failures.append(f"invalid or incomplete markdown: {reason}")

    expected_retrieval = case.expected_retrieval
    if retrieval_summary is not None:
        _check_count(
            failures=failures,
            name="raw_hit_count",
            actual_raw=retrieval_summary.get("raw_hit_count"),
            expected_exact=expected_retrieval.raw_hit_count,
            expected_min=expected_retrieval.min_raw_hit_count,
            expected_max=expected_retrieval.max_raw_hit_count,
        )

        _check_count(
            failures=failures,
            name="accepted_hit_count",
            actual_raw=retrieval_summary.get("accepted_hit_count"),
            expected_exact=expected_retrieval.accepted_hit_count,
            expected_min=expected_retrieval.min_accepted_hit_count,
            expected_max=expected_retrieval.max_accepted_hit_count,
        )

        if expected_retrieval.similarity_threshold is not None:
            actual = retrieval_summary.get("similarity_threshold")
            if actual is not None:
                if abs(float(actual) - expected_retrieval.similarity_threshold) > 1e-6:
                    failures.append(
                        "expected similarity_threshold="
                        f"{expected_retrieval.similarity_threshold}, got {actual}"
                    )

        searchable = _json_dump_for_search(retrieval_summary)

        if expected_retrieval.selected_source_path:
            if expected_retrieval.selected_source_path not in searchable:
                failures.append(
                    "expected selected source path in retrieval summary: "
                    f"{expected_retrieval.selected_source_path}"
                )

        for fragment in expected_retrieval.diagnostics_contains:
            if fragment not in searchable:
                failures.append(
                    f"expected retrieval diagnostics to contain: {fragment}"
                )

    return failures
