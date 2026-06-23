from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from app.eval.assertions import (
    evaluate_case_output,
    extract_cited_source_ids,
)
from app.eval.models import EvalCaseResult, EvalRunReport, EvalSuite


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_to_dict(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): _safe_to_dict(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_safe_to_dict(v) for v in value]

    if hasattr(value, "to_dict"):
        try:
            return _safe_to_dict(value.to_dict())
        except Exception:
            return str(value)

    if hasattr(value, "debug_summary"):
        try:
            return _safe_to_dict(value.debug_summary())
        except Exception:
            return str(value)

    if hasattr(value, "__dict__"):
        try:
            return _safe_to_dict(vars(value))
        except Exception:
            return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _get_answer(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("answer", "assistant_message", "message", "content"):
            value = result.get(key)
            if isinstance(value, str):
                return value

    for key in ("answer", "assistant_message", "message", "content"):
        value = getattr(result, key, None)
        if isinstance(value, str):
            return value

    return str(result)


def _get_route(result: dict[str, Any]) -> str | None:
    for path in (
        ("route",),
        ("selected_route",),
        ("metadata", "route"),
        ("debug", "route"),
    ):
        current: Any = result
        ok = True
        for part in path:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current is not None:
            return str(current)
    return None


def _get_answerable(result: dict[str, Any]) -> bool | None:
    value = result.get("answerable")
    if isinstance(value, bool):
        return value
    debug = result.get("debug")
    if isinstance(debug, dict) and isinstance(debug.get("answerable"), bool):
        return debug["answerable"]
    return None


def _list_retrieval_runs(memory: Any, session_id: str) -> list[Any]:
    if not hasattr(memory, "list_retrieval_runs"):
        return []

    try:
        return list(memory.list_retrieval_runs(session_id))
    except TypeError:
        return list(memory.list_retrieval_runs(session_id=session_id))
    except Exception:
        return []


def _retrieval_run_summary(run: Any) -> dict[str, Any]:
    data = _safe_to_dict(run)

    if isinstance(data, dict):
        summary = dict(data)
    else:
        summary = {"repr": str(run)}

    for key in (
        "raw_hit_count",
        "accepted_hit_count",
        "similarity_threshold",
        "standalone_query",
        "retrieval_diagnostics",
    ):
        if key not in summary and hasattr(run, key):
            summary[key] = _safe_to_dict(getattr(run, key))

    return summary


class EvalRunner:
    def __init__(
        self,
        *,
        backend: Any,
        user_id: str = "mic9000-eval",
        session_title: str = "MIC 9000 eval run",
    ) -> None:
        self.backend = backend
        self.user_id = user_id
        self.session_title = session_title

    def _create_session(self, *, title_suffix: str | None = None) -> Any:
        memory = self.backend.memory
        title = self.session_title
        if title_suffix:
            title = f"{title} — {title_suffix}"

        return memory.create_session(
            user_id=self.user_id,
            title=title,
        )

    @staticmethod
    def _session_id_from_session(session: Any) -> str:
        session_id = getattr(session, "session_id", None) or getattr(session, "id", None)
        if session_id is None:
            raise RuntimeError("Could not resolve created session_id.")
        return str(session_id)

    def _invoke_case(self, *, session_id: str, case: Any) -> Any:
        kwargs: dict[str, Any] = {
            "session_id": session_id,
            "user_message": case.question,
        }

        if case.selected_document is not None:
            kwargs["document_scope"] = case.selected_document

        if case.document_behavior is not None:
            kwargs["document_behavior"] = case.document_behavior

        return self.backend.graph.invoke(**kwargs)

    def run(
        self,
        suite: EvalSuite,
        *,
        case_ids: set[str] | None = None,
        tags: set[str] | None = None,
        fail_fast: bool = False,
    ) -> EvalRunReport:
        started_at = _utc_now()
        started_perf = time.perf_counter()

        shared_session = self._create_session(title_suffix=suite.name)
        shared_session_id = self._session_id_from_session(shared_session)

        results: list[EvalCaseResult] = []
        case_session_ids: dict[str, str] = {}

        for case in suite.cases:
            if case_ids and case.id not in case_ids:
                continue

            if tags and not (set(case.tags) & tags):
                continue

            if case.session_mode == "shared":
                session_id = shared_session_id
            else:
                case_session = self._create_session(title_suffix=f"{suite.name}:{case.id}")
                session_id = self._session_id_from_session(case_session)

            case_session_ids[case.id] = session_id

            if case.skip:
                results.append(
                    EvalCaseResult(
                        case_id=case.id,
                        question=case.question,
                        status="SKIP",
                        passed=True,
                        skipped=True,
                        latency_seconds=0.0,
                        answer="",
                        failures=[],
                        route=None,
                        answerable=None,
                        cited_source_ids=[],
                        retrieval_summary=None,
                        raw_result_keys=[],
                        session_id=session_id,
                    )
                )
                continue

            runs_before = _list_retrieval_runs(self.backend.memory, session_id)
            before_count = len(runs_before)

            case_started = time.perf_counter()
            raw_result = self._invoke_case(session_id=session_id, case=case)
            latency_seconds = time.perf_counter() - case_started

            result_dict = _safe_to_dict(raw_result)
            if not isinstance(result_dict, dict):
                result_dict = {"result": result_dict}

            answer = _get_answer(raw_result)

            runs_after = _list_retrieval_runs(self.backend.memory, session_id)
            retrieval_summary = None
            if len(runs_after) > before_count:
                retrieval_summary = _retrieval_run_summary(runs_after[-1])
            elif runs_after:
                retrieval_summary = _retrieval_run_summary(runs_after[-1])

            failures = evaluate_case_output(
                case=case,
                answer=answer,
                result=result_dict,
                retrieval_summary=retrieval_summary,
                latency_seconds=latency_seconds,
            )

            passed = not failures

            result = EvalCaseResult(
                case_id=case.id,
                question=case.question,
                status="PASS" if passed else "FAIL",
                passed=passed,
                skipped=False,
                latency_seconds=round(latency_seconds, 4),
                answer=answer,
                failures=failures,
                route=_get_route(result_dict),
                answerable=_get_answerable(result_dict),
                cited_source_ids=extract_cited_source_ids(answer),
                retrieval_summary=retrieval_summary,
                raw_result_keys=sorted(result_dict.keys()),
                session_id=session_id,
            )

            results.append(result)

            if fail_fast and not passed:
                break

        completed_at = _utc_now()
        duration_seconds = time.perf_counter() - started_perf

        total_cases = len(results)
        skipped_cases = sum(result.skipped for result in results)
        failed_cases = sum((not result.passed) and (not result.skipped) for result in results)
        passed_cases = sum(result.passed and not result.skipped for result in results)
        executable_cases = max(1, total_cases - skipped_cases)

        report_session_id = (
            "isolated_per_case" if suite.isolate_cases else shared_session_id
        )

        return EvalRunReport(
            suite_name=suite.name,
            suite_version=suite.version,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            duration_seconds=round(duration_seconds, 4),
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            skipped_cases=skipped_cases,
            pass_rate=passed_cases / executable_cases,
            session_id=report_session_id,
            results=results,
            metadata={
                "suite_path": str(suite.path) if suite.path else None,
                "user_id": self.user_id,
                "session_title": self.session_title,
                "isolate_cases": suite.isolate_cases,
                "shared_session_id": shared_session_id,
                "case_session_ids": case_session_ids,
            },
        )
