from __future__ import annotations

import re
from dataclasses import (
    dataclass,
)


_DANGLING_ENDINGS = (
    ":",
    "-",
    "•",
    "*",
    ",",
    ";",
    "(",
    "[",
    "{",
)


@dataclass(
    frozen=True,
    slots=True,
)
class AnswerQualityIssue:
    failed: bool
    reason: str


def detect_answer_quality_issue(
    *,
    answer: str,
    user_question: str,
) -> AnswerQualityIssue:
    prepared = (
        answer
        or ""
    ).strip()

    if not prepared:
        return AnswerQualityIssue(
            failed=True,
            reason="empty answer",
        )

    if len(prepared) < 20:
        return AnswerQualityIssue(
            failed=True,
            reason="suspiciously short answer",
        )

    # Broken bold / inline code markers.
    if (
        prepared.count("**")
        % 2
        != 0
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="unmatched bold markdown marker",
        )

    if (
        prepared.count("`")
        % 2
        != 0
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="unmatched inline-code marker",
        )

    if (
        prepared.count("```")
        % 2
        != 0
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="unclosed code fence",
        )

    if prepared.endswith(
        _DANGLING_ENDINGS
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="answer ends with dangling punctuation",
        )

    last_line = (
        prepared
        .splitlines()[-1]
        .strip()
    )

    if re.match(
        r"^[-*•]\s*$",
        last_line,
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="dangling bullet item",
        )

    # Multi-part question but answer only starts one side.
    lower_question = (
        user_question
        .lower()
    )

    lower_answer = (
        prepared
        .lower()
    )

    if (
        "respectively"
        in lower_question
        and (
            "live"
            in lower_question
            and "replay"
            in lower_question
        )
        and not (
            "live"
            in lower_answer
            and "replay"
            in lower_answer
        )
    ):
        return AnswerQualityIssue(
            failed=True,
            reason=(
                "multi-part live/replay answer appears incomplete"
            ),
        )

    if (
        "respectively"
        in lower_question
        and len(
            prepared
            .split()
        )
        < 12
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="multi-part answer too short",
        )
    
        # Incomplete inline-code span such as:
    # "... reports `num_features: 39` and `feature_version:"
    if (
        prepared.count("`")
        % 2
        != 0
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="unmatched inline-code marker",
        )

    # Common truncation pattern: answer ends after saying "and"
    # or after introducing a field name.
    if re.search(
        r"\b(and|or|including|such as)\s+`?[\w_.:/-]*:?\s*$",
        prepared,
        flags=re.IGNORECASE,
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="answer appears truncated after conjunction or field name",
        )

    # Ends with an unfinished technical field.
    if re.search(
        r"`?[\w_.-]+:\s*$",
        prepared,
    ):
        return AnswerQualityIssue(
            failed=True,
            reason="answer ends with unfinished technical field",
        )

    return AnswerQualityIssue(
        failed=False,
        reason="ok",
    )