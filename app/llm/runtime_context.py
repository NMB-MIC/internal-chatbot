from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


LOCAL_TIMEZONE_NAME = (
    "Asia/Bangkok"
)

LOCAL_TIMEZONE = ZoneInfo(
    LOCAL_TIMEZONE_NAME
)


def local_now() -> datetime:
    """
    Return the current timezone-aware local datetime
    for MIC 9000.
    """

    return datetime.now(
        LOCAL_TIMEZONE
    )


def build_runtime_context() -> str:
    """
    Build a dynamic prompt block for time-sensitive reasoning.

    This function is called for every relevant LLM request rather
    than once at module import time. A long-running Streamlit process
    therefore receives the correct date after midnight without
    requiring a restart.
    """

    now = local_now()

    return (
        "Runtime context:\n"
        f"- Current local date: "
        f"{now:%Y-%m-%d}\n"
        f"- Current local time: "
        f"{now:%H:%M:%S}\n"
        f"- Timezone: "
        f"{LOCAL_TIMEZONE_NAME}\n"
        "- Treat dates before the current local date "
        "as past dates.\n"
        "- Treat dates after the current local date "
        "as future dates.\n"
        "- When a source document explicitly states a date, "
        "preserve the source value and interpret its timing "
        "relative to this runtime context."
    )