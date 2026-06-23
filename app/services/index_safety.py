from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TEXT_SCAN_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".sql",
    ".env",
    ".ini",
    ".cfg",
    ".toml",
    ".xml",
}


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN\s+(?:RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE KEY-----",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "openai_like_secret",
        re.compile(
            r"\bsk-[A-Za-z0-9_\-]{12,}\b",
        ),
    ),
    (
        "slack_token",
        re.compile(
            r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b",
        ),
    ),
    (
        "github_pat",
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b",
        ),
    ),
    (
        "aws_access_key_id",
        re.compile(
            r"\bAKIA[0-9A-Z]{16}\b",
        ),
    ),
    (
        "aws_secret_access_key_assignment",
        re.compile(
            r"\bAWS_SECRET_ACCESS_KEY\b\s*[:=]\s*[\"']?[^\"'\s]{12,}",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "password_assignment",
        re.compile(
            r"\b(?:password|passwd|pwd)\b\s*[:=]\s*[\"']?[^\"'\s]{4,}",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "api_key_assignment",
        re.compile(
            r"\b(?:api[_-]?key|secret|token)\b\s*[:=]\s*[\"']?[^\"'\s]{8,}",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "credentialed_database_url",
        re.compile(
            r"\b(?:postgresql|postgres|mysql|mongodb|redis)://[^/\s:@]+:[^@\s]+@",
            flags=re.IGNORECASE,
        ),
    ),
)


SENSITIVE_FILENAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credential_filename",
        re.compile(
            r"\bcredentials?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "password_filename",
        re.compile(
            r"\b(?:password|passwd|pwd)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "secret_filename",
        re.compile(
            r"\bsecrets?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "token_filename",
        re.compile(
            r"\btokens?\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "api_key_filename",
        re.compile(
            r"\bapi\s*key\b|\bapikey\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "private_key_filename",
        re.compile(
            r"\bprivate\s*key\b",
            flags=re.IGNORECASE,
        ),
    ),
)


SYNTHETIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "batch_sample_filename",
        re.compile(
            r"\bbatch\s*\d+\s*sample\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "synthetic_keyword",
        re.compile(
            r"\bsynthetic\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "fixture_keyword",
        re.compile(
            r"\bfixture\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "mock_data_keyword",
        re.compile(
            r"\bmock(?:\s|-|_)?data\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "test_document_keyword",
        re.compile(
            r"\btest(?:\s|-|_)?document\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "demo_support_article",
        re.compile(
            r"\bdemo(?:\s|-|_)?support(?:\s|-|_)?article\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "lorem_ipsum",
        re.compile(
            r"\blorem\s+ipsum\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "example_only",
        re.compile(
            r"\bexample(?:\s|-|_)?only\b",
            flags=re.IGNORECASE,
        ),
    ),
)


EXTERNAL_BOOK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "blood_meridian",
        re.compile(
            r"\bblood\s*meridian\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "cormac_mccarthy",
        re.compile(
            r"\bcormac\s*mccarthy\b|\bmccarthy\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "external_book_filename",
        re.compile(
            r"\b(novel|book|ebook|fiction)\b",
            flags=re.IGNORECASE,
        ),
    ),
)


STALE_PATH_PATTERN = re.compile(
    r"\b(?:old|deprecated|archive|archived|legacy|superseded)\b",
    flags=re.IGNORECASE,
)


STALE_CONTENT_PATTERN = re.compile(
    r"\b(?:deprecated|superseded|no longer used|archived|legacy version)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class FileSafetyAssessment:
    relative_path: str
    category: str
    extension: str
    size_bytes: int
    modified_at_utc: str
    sha256: str
    status: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    secret_hits: list[str] = field(default_factory=list)
    synthetic_hits: list[str] = field(default_factory=list)
    external_book_hits: list[str] = field(default_factory=list)
    stale_hits: list[str] = field(default_factory=list)
    quarantined_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DocumentSafetySummary:
    mode: str
    accepted_files: int
    quarantined_files: int
    rejected_files: int
    warned_files: int
    secret_hits: int
    synthetic_fixture_hits: int
    external_book_hits: int
    stale_hits: int
    files: list[FileSafetyAssessment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "accepted_files": self.accepted_files,
            "quarantined_files": self.quarantined_files,
            "rejected_files": self.rejected_files,
            "warned_files": self.warned_files,
            "secret_hits": self.secret_hits,
            "synthetic_fixture_hits": self.synthetic_fixture_hits,
            "external_book_hits": self.external_book_hits,
            "stale_hits": self.stale_hits,
            "files": [
                item.to_dict()
                for item in self.files
            ],
        }


def utc_now_iso() -> str:
    return (
        datetime
        .now(timezone.utc)
        .isoformat(timespec="seconds")
    )


def make_rebuild_id() -> str:
    stamp = (
        datetime
        .now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
    )

    return f"{stamp}_{hashlib.sha1(stamp.encode('utf-8')).hexdigest()[:8]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_obj:
        for block in iter(
            lambda: file_obj.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _safe_relative_path(
    *,
    path: Path,
    root: Path,
) -> str:
    return (
        path
        .relative_to(root)
        .as_posix()
    )


def _file_modified_at(path: Path) -> str:
    return (
        datetime
        .fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )
        .isoformat(timespec="seconds")
    )


def _read_text_for_scan(
    path: Path,
    *,
    max_chars: int,
) -> str:
    if path.suffix.lower() not in TEXT_SCAN_EXTENSIONS:
        return ""

    try:
        raw = path.read_bytes()[: max_chars * 4]
        return raw.decode("utf-8", errors="ignore")[:max_chars]

    except OSError:
        return ""


def _scan_patterns(
    *,
    text: str,
    path_text: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]],
) -> list[str]:
    hits: list[str] = []

    scan_target = (
        path_text
        + "\n"
        + text
    )

    for name, pattern in patterns:
        if pattern.search(scan_target):
            hits.append(name)

    return sorted(set(hits))


def _scan_sensitive_filename(
    *,
    path_text: str,
) -> list[str]:
    hits: list[str] = []

    for name, pattern in SENSITIVE_FILENAME_PATTERNS:
        if pattern.search(path_text):
            hits.append(name)

    return sorted(set(hits))


def assess_document_file(
    *,
    path: Path,
    documents_dir: Path,
    mode: str,
    stale_days: int,
    allow_secret_documents: bool,
    quarantine_synthetic_in_production: bool,
    quarantine_external_books_in_production: bool,
    max_scan_chars: int,
) -> FileSafetyAssessment:
    relative_path = _safe_relative_path(
        path=path,
        root=documents_dir,
    )

    parts = Path(relative_path).parts
    category = parts[0] if parts else "uncategorized"
    path_text = relative_path.replace("_", " ").replace("-", " ")
    scan_text = _read_text_for_scan(
        path,
        max_chars=max_scan_chars,
    )

    secret_hits = _scan_patterns(
        text=scan_text,
        path_text=path_text,
        patterns=SECRET_PATTERNS,
    )

    filename_secret_hits = _scan_sensitive_filename(
        path_text=path_text,
    )

    secret_hits = sorted(set([
        *secret_hits,
        *filename_secret_hits,
    ]))

    synthetic_hits = _scan_patterns(
        text=scan_text,
        path_text=path_text,
        patterns=SYNTHETIC_PATTERNS,
    )

    external_book_hits = _scan_patterns(
        text=scan_text,
        path_text=path_text,
        patterns=EXTERNAL_BOOK_PATTERNS,
    )

    stale_hits: list[str] = []

    age_seconds = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )
    ).total_seconds()

    if stale_days > 0 and age_seconds > stale_days * 86400:
        stale_hits.append("mtime_older_than_threshold")

    if STALE_PATH_PATTERN.search(path_text):
        stale_hits.append("stale_path_keyword")

    if scan_text and STALE_CONTENT_PATTERN.search(scan_text):
        stale_hits.append("stale_content_keyword")

    reasons: list[str] = []
    warnings: list[str] = []
    status = "accepted"

    production = mode == "production"

    if secret_hits:
        if allow_secret_documents:
            warnings.append("secret-like content detected but override is enabled")
        else:
            status = "rejected"
            reasons.append("secret-like content detected")

    if status == "accepted" and production and synthetic_hits and quarantine_synthetic_in_production:
        status = "quarantined"
        reasons.append("synthetic/test fixture detected in production mode")

    if status == "accepted" and production and external_book_hits and quarantine_external_books_in_production:
        status = "quarantined"
        reasons.append("external book/stress-test document detected in production mode")

    if synthetic_hits and not production:
        warnings.append("synthetic/test fixture marker detected")

    if external_book_hits and not production:
        warnings.append("external book/stress-test marker detected")

    if stale_hits:
        warnings.append("document appears stale or legacy")

    return FileSafetyAssessment(
        relative_path=relative_path,
        category=category,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        modified_at_utc=_file_modified_at(path),
        sha256=sha256_file(path),
        status=status,
        reasons=reasons,
        warnings=warnings,
        secret_hits=secret_hits,
        synthetic_hits=synthetic_hits,
        external_book_hits=external_book_hits,
        stale_hits=sorted(set(stale_hits)),
    )


def apply_quarantine_actions(
    *,
    assessments: list[FileSafetyAssessment],
    documents_dir: Path,
    quarantine_dir: Path,
    rebuild_id: str,
) -> list[FileSafetyAssessment]:
    updated: list[FileSafetyAssessment] = []

    for assessment in assessments:
        if assessment.status == "accepted":
            updated.append(assessment)
            continue

        source = documents_dir / assessment.relative_path

        if not source.exists():
            updated.append(assessment)
            continue

        destination = (
            quarantine_dir
            / rebuild_id
            / assessment.status
            / assessment.relative_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.move(
            str(source),
            str(destination),
        )

        updated.append(
            FileSafetyAssessment(
                **{
                    **assessment.to_dict(),
                    "quarantined_to": (
                        destination
                        .as_posix()
                    ),
                }
            )
        )

    return updated


def summarize_assessments(
    *,
    mode: str,
    assessments: list[FileSafetyAssessment],
) -> DocumentSafetySummary:
    return DocumentSafetySummary(
        mode=mode,
        accepted_files=sum(
            item.status == "accepted"
            for item in assessments
        ),
        quarantined_files=sum(
            item.status == "quarantined"
            for item in assessments
        ),
        rejected_files=sum(
            item.status == "rejected"
            for item in assessments
        ),
        warned_files=sum(
            bool(item.warnings)
            for item in assessments
        ),
        secret_hits=sum(
            bool(item.secret_hits)
            for item in assessments
        ),
        synthetic_fixture_hits=sum(
            bool(item.synthetic_hits)
            for item in assessments
        ),
        external_book_hits=sum(
            bool(item.external_book_hits)
            for item in assessments
        ),
        stale_hits=sum(
            bool(item.stale_hits)
            for item in assessments
        ),
        files=assessments,
    )


def scan_document_tree(
    *,
    documents_dir: Path,
    supported_extensions: Iterable[str],
    mode: str,
    stale_days: int,
    allow_secret_documents: bool,
    quarantine_synthetic_in_production: bool,
    quarantine_external_books_in_production: bool,
    max_scan_chars: int,
) -> list[FileSafetyAssessment]:
    supported = {
        extension.lower()
        for extension in supported_extensions
    }

    if not documents_dir.exists():
        return []

    assessments: list[FileSafetyAssessment] = []

    for path in sorted(documents_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in supported
        ):
            continue

        assessments.append(
            assess_document_file(
                path=path,
                documents_dir=documents_dir,
                mode=mode,
                stale_days=stale_days,
                allow_secret_documents=allow_secret_documents,
                quarantine_synthetic_in_production=quarantine_synthetic_in_production,
                quarantine_external_books_in_production=quarantine_external_books_in_production,
                max_scan_chars=max_scan_chars,
            )
        )

    return assessments