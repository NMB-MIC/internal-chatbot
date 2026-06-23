from __future__ import annotations

import re
from typing import Literal


LanguageCode = Literal[
    "en",
    "th",
    "mixed",
    "other",
]


_THAI_PATTERN = re.compile(
    r"[\u0E00-\u0E7F]"
)

_LATIN_PATTERN = re.compile(
    r"[A-Za-z]"
)


def detect_language(
    text: str,
) -> LanguageCode:
    """
    Lightweight deterministic language detection.

    Technical questions often mix Thai with English terms such as:
        Kafka
        topic
        machine logs
        API

    The LLM prompt later receives this label as a response-language hint.
    """

    has_thai = bool(
        _THAI_PATTERN.search(
            text
        )
    )

    has_latin = bool(
        _LATIN_PATTERN.search(
            text
        )
    )

    if has_thai and has_latin:
        return "mixed"

    if has_thai:
        return "th"

    if has_latin:
        return "en"

    return "other"