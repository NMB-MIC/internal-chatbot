from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.memory.sqlite_memory import (  # noqa: E402
    SqliteChatMemory,
)


UI_USER_ID = "local-user"

# Set this to False before sharing the export if retrieved
# source previews contain confidential internal information.
INCLUDE_SOURCE_PREVIEWS = True


def utc_now() -> str:
    return (
        datetime
        .now(timezone.utc)
        .isoformat(
            timespec="seconds"
        )
    )


def safe_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def export_ui_chats(
    *,
    latest_only: bool = False,
    session_id: str | None = None,
    session_title: str | None = None,
) -> tuple[
    Path,
    Path,
]:
    memory = SqliteChatMemory()

    memory.initialize()

    sessions = [
        session
        for session in (
            memory
            .list_sessions(
                limit=500,
            )
        )
        if session.user_id
        == UI_USER_ID
    ]

    if session_id:
        sessions = [
            session
            for session in sessions
            if session.session_id
            == session_id
        ]

    if session_title:
        sessions = [
            session
            for session in sessions
            if session.title
            == session_title
        ]

    if latest_only:
        sessions = sessions[:1]

    exported_sessions = []

    for session in sessions:
        messages = [
            message.to_dict()
            for message in (
                memory
                .load_all_messages(
                    session.session_id
                )
            )
        ]

        retrieval_runs = []

        for run in (
            memory
            .list_retrieval_runs(
                session.session_id
            )
        ):
            run_data = (
                run.to_dict()
            )

            sources = []

            for source in (
                memory
                .list_retrieval_sources(
                    run.retrieval_run_id
                )
            ):
                source_data = (
                    source.to_dict()
                )

                if not (
                    INCLUDE_SOURCE_PREVIEWS
                ):
                    source_data[
                        "text_preview"
                    ] = (
                        "[preview omitted]"
                    )

                sources.append(
                    source_data
                )

            run_data[
                "sources"
            ] = sources

            retrieval_runs.append(
                run_data
            )

        exported_sessions.append({
            "session": (
                session.to_dict()
            ),
            "messages": messages,
            "retrieval_runs": (
                retrieval_runs
            ),
        })

    payload = {
        "exported_at_utc": (
            utc_now()
        ),
        "ui_user_id": (
            UI_USER_ID
        ),
        "include_source_previews": (
            INCLUDE_SOURCE_PREVIEWS
        ),
        "session_count": len(
            exported_sessions
        ),
        "sessions": (
            exported_sessions
        ),
    }

    output_dir = (
        PROJECT_ROOT
        / "exports"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = (
        datetime
        .now(timezone.utc)
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    json_path = (
        output_dir
        / (
            "mic9000_ui_chats_"
            f"{timestamp}.json"
        )
    )

    markdown_path = (
        output_dir
        / (
            "mic9000_ui_chats_"
            f"{timestamp}.md"
        )
    )

    json_path.write_text(
        safe_json(
            payload
        ),
        encoding="utf-8",
    )

    markdown_lines = [
        "# MIC 9000 UI Chat Export",
        "",
        (
            f"Exported at: "
            f"{payload['exported_at_utc']}"
        ),
        (
            f"Sessions: "
            f"{payload['session_count']}"
        ),
        "",
    ]

    for exported_session in (
        exported_sessions
    ):
        session = (
            exported_session[
                "session"
            ]
        )

        markdown_lines.extend([
            "---",
            "",
            (
                "## "
                + (
                    session.get(
                        "title"
                    )
                    or "Untitled chat"
                )
            ),
            "",
            (
                f"Session ID: "
                f"`{session['session_id']}`"
            ),
            (
                f"Updated at: "
                f"`{session['updated_at']}`"
            ),
            "",
            "### Conversation",
            "",
        ])

        for message in (
            exported_session[
                "messages"
            ]
        ):
            markdown_lines.extend([
                (
                    "#### "
                    + message[
                        "role"
                    ].upper()
                    + " · message "
                    + str(
                        message[
                            "message_id"
                        ]
                    )
                ),
                "",
                message[
                    "content"
                ],
                "",
                "<details>",
                (
                    "<summary>"
                    "Stored metadata"
                    "</summary>"
                ),
                "",
                "```json",
                safe_json({
                    "language": (
                        message.get(
                            "language"
                        )
                    ),
                    "original_query": (
                        message.get(
                            "original_query"
                        )
                    ),
                    "standalone_query": (
                        message.get(
                            "standalone_query"
                        )
                    ),
                    "answerable": (
                        message.get(
                            "answerable"
                        )
                    ),
                    "confidence": (
                        message.get(
                            "confidence"
                        )
                    ),
                    "metadata": (
                        message.get(
                            "metadata"
                        )
                    ),
                }),
                "```",
                "",
                "</details>",
                "",
            ])

        markdown_lines.extend([
            "### Retrieval runs",
            "",
        ])

        for run in (
            exported_session[
                "retrieval_runs"
            ]
        ):
            markdown_lines.extend([
                (
                    "#### Retrieval run "
                    + str(
                        run[
                            "retrieval_run_id"
                        ]
                    )
                ),
                "",
                (
                    "**Original query:** "
                    + run[
                        "original_query"
                    ]
                ),
                "",
                (
                    "**Standalone query:** "
                    + run[
                        "standalone_query"
                    ]
                ),
                "",
                "```json",
                safe_json({
                    "raw_hit_count": (
                        run[
                            "raw_hit_count"
                        ]
                    ),
                    "accepted_hit_count": (
                        run[
                            "accepted_hit_count"
                        ]
                    ),
                    "top_score": (
                        run[
                            "top_score"
                        ]
                    ),
                    "similarity_threshold": (
                        run[
                            "similarity_threshold"
                        ]
                    ),
                    "embedding_seconds": (
                        run[
                            "embedding_seconds"
                        ]
                    ),
                    "search_seconds": (
                        run[
                            "search_seconds"
                        ]
                    ),
                    "total_seconds": (
                        run[
                            "total_seconds"
                        ]
                    ),
                    "llm_called": (
                        run[
                            "llm_called"
                        ]
                    ),
                    "thinking_enabled": (
                        run[
                            "thinking_enabled"
                        ]
                    ),
                    "limitations": (
                        run[
                            "limitations"
                        ]
                    ),
                    "retrieval_diagnostics": (
                        run.get(
                            "retrieval_diagnostics",
                            {},
                        )
                    ),
                }),
                "```",
                "",
            ])

            for source in (
                run[
                    "sources"
                ]
            ):
                markdown_lines.extend([
                    (
                        "##### Rank "
                        + str(
                            source[
                                "source_rank"
                            ]
                        )
                        + " · "
                        + (
                            source.get(
                                "source_id"
                            )
                            or "retrieved"
                        )
                    ),
                    "",
                    (
                        f"- Path: "
                        f"`{source['source_path']}`"
                    ),
                    (
                        f"- Score: "
                        f"`{source['score']}`"
                    ),
                    (
                        f"- Accepted: "
                        f"`{source['accepted']}`"
                    ),
                    (
                        f"- Cited: "
                        f"`{source['cited']}`"
                    ),
                    "",
                    "```text",
                    source[
                        "text_preview"
                    ],
                    "```",
                    "",
                ])

    markdown_path.write_text(
        "\n".join(
            markdown_lines
        ),
        encoding="utf-8",
    )

    return (
        json_path,
        markdown_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export MIC 9000 UI "
            "conversation diagnostics."
        )
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help=(
            "Export only the latest "
            "local UI session."
        ),
    )

    parser.add_argument(
        "--session-id",
        default=None,
        help=(
            "Export one exact "
            "session ID."
        ),
    )

    parser.add_argument(
        "--session-title",
        default=None,
        help=(
            "Export sessions with "
            "this exact title."
        ),
    )

    args = parser.parse_args()

    (
        json_output,
        markdown_output,
    ) = export_ui_chats(
        latest_only=(
            args.latest
        ),
        session_id=(
            args.session_id
        ),
        session_title=(
            args.session_title
        ),
    )

    print(
        "JSON export:"
    )

    print(
        json_output
    )

    print(
        "\nMarkdown export:"
    )

    print(
        markdown_output
    )
