#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


def load_latest_manifest(manifest_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    manifests = sorted(manifest_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in manifests:
        try:
            return json.loads(path.read_text(encoding="utf-8")), path
        except Exception:
            continue
    return None, None


def find_latest_eval_report(eval_reports_dir: Path, suite_name: str) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = sorted(eval_reports_dir.glob(f"*_{suite_name}.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            return json.loads(path.read_text(encoding="utf-8")), path
        except Exception:
            continue
    return None, None


def git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def summarize_manifest(manifest: dict[str, Any] | None, path: Path | None) -> dict[str, Any]:
    if not manifest:
        return {"error": "latest manifest not found"}
    safety = manifest.get("document_safety") or {}
    snapshot = manifest.get("snapshot") or {}
    qdrant = manifest.get("qdrant") or {}
    return {
        "rebuild_id": manifest.get("rebuild_id"),
        "mode": manifest.get("mode"),
        "manifest_path": str(path) if path else manifest.get("manifest_path"),
        "started_at_utc": manifest.get("started_at_utc"),
        "completed_at_utc": manifest.get("completed_at_utc"),
        "indexed_points": qdrant.get("indexed_points") or manifest.get("indexed_points"),
        "point_count_after_upsert": qdrant.get("point_count_after_upsert"),
        "accepted_files": safety.get("accepted_files"),
        "quarantined_files": safety.get("quarantined_files"),
        "rejected_files": safety.get("rejected_files"),
        "warned_files": safety.get("warned_files"),
        "secret_hits": safety.get("secret_hits"),
        "synthetic_fixture_hits": safety.get("synthetic_fixture_hits"),
        "external_book_hits": safety.get("external_book_hits"),
        "snapshot_enabled": snapshot.get("enabled"),
        "snapshot_created": snapshot.get("created"),
        "snapshot_name": snapshot.get("snapshot_name"),
    }


def summarize_eval(report: dict[str, Any] | None, path: Path | None) -> dict[str, Any]:
    if not report:
        return {"found": False}
    return {
        "found": True,
        "report_path": str(path) if path else None,
        "suite_name": report.get("suite_name"),
        "total_cases": report.get("total_cases"),
        "passed_cases": report.get("passed_cases"),
        "failed_cases": report.get("failed_cases"),
        "skipped_cases": report.get("skipped_cases"),
        "pass_rate": report.get("pass_rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a MIC 9000 release manifest.")
    parser.add_argument("--env-file", default=".env.production")
    parser.add_argument("--output-dir", default="storage/releases")
    parser.add_argument("--version", default="1.0.0-rc1")
    parser.add_argument("--ui-runtime", default="streamlit")
    args = parser.parse_args()

    project_root = Path.cwd()
    env_file = Path(args.env_file)
    env = parse_env_file(env_file)
    for key, value in env.items():
        os.environ.setdefault(key, value)

    manifest_dir = Path(env.get("INDEX_MANIFEST_DIR", "storage/index_manifests"))
    eval_reports_dir = Path("eval_reports")
    latest_manifest, latest_manifest_path = load_latest_manifest(manifest_dir)

    suite_names = [
        "runbook_regression",
        "document_mode_smoke",
        "thai_runbook_smoke",
    ]
    evals: dict[str, Any] = {}
    for suite in suite_names:
        report, path = find_latest_eval_report(eval_reports_dir, suite)
        evals[suite] = summarize_eval(report, path)

    manifest_summary = summarize_manifest(latest_manifest, latest_manifest_path)

    eval_failures = [name for name, item in evals.items() if item.get("failed_cases") not in (0, None) or not item.get("found")]
    release_gate = "pass"
    gate_failures: list[str] = []
    if manifest_summary.get("mode") != "production":
        release_gate = "fail"
        gate_failures.append("latest manifest is not production mode")
    if manifest_summary.get("snapshot_created") is not True:
        release_gate = "fail"
        gate_failures.append("latest manifest snapshot was not created")
    if eval_failures:
        release_gate = "fail"
        gate_failures.append("one or more eval suites failed or are missing: " + ", ".join(eval_failures))

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_id = f"mic9000_release_{now}"
    release_manifest = {
        "release_id": release_id,
        "version": args.version,
        "release_stage": "release-candidate",
        "ui_runtime": args.ui_runtime,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "env_file": str(env_file),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(project_root),
        "app_env": env.get("APP_ENV"),
        "mic_index_mode": env.get("MIC_INDEX_MODE"),
        "latest_manifest": manifest_summary,
        "eval_reports": evals,
        "release_gate": release_gate,
        "gate_failures": gate_failures,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{release_id}.json"
    output_path.write_text(json.dumps(release_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "release_gate": release_gate,
        "release_id": release_id,
        "output_path": str(output_path),
        "gate_failures": gate_failures,
    }, indent=2, ensure_ascii=False))
    return 0 if release_gate == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
