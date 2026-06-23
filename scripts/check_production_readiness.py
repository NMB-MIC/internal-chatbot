#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            values[key] = val
            os.environ.setdefault(key, val)
    return values


def run_json(cmd: list[str]) -> tuple[dict | None, str | None, int]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip(), proc.returncode
    out = proc.stdout.strip()
    try:
        return json.loads(out), None, 0
    except Exception as exc:
        return None, f"failed to parse JSON: {exc}; stdout={out[:500]}", 1


def latest_manifest(manifest_dir: Path) -> dict | None:
    manifests = sorted(manifest_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        return None
    try:
        return json.loads(manifests[0].read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="MIC 9000 production readiness checker")
    parser.add_argument("--env-file", default=".env.production")
    parser.add_argument("--expect-mode", default="production", choices=["production", "development", "any"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    load_env_file(root / args.env_file)
    manifest_dir = Path(os.getenv("INDEX_MANIFEST_DIR", "storage/index_manifests"))

    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # Static env checks
    add("APP_ENV production", os.getenv("APP_ENV", "production") == "production", os.getenv("APP_ENV", "unset"))
    add("MIC_SECURITY_ENABLED true", os.getenv("MIC_SECURITY_ENABLED", "").lower() in {"1", "true", "yes", "on"}, os.getenv("MIC_SECURITY_ENABLED", "unset"))
    add("MIC_SECURITY_PUBLIC_TRACE false", os.getenv("MIC_SECURITY_PUBLIC_TRACE", "false").lower() not in {"1", "true", "yes", "on"}, os.getenv("MIC_SECURITY_PUBLIC_TRACE", "unset"))
    add("MIC_SECURITY_PUBLIC_DIAGNOSTICS false", os.getenv("MIC_SECURITY_PUBLIC_DIAGNOSTICS", "false").lower() not in {"1", "true", "yes", "on"}, os.getenv("MIC_SECURITY_PUBLIC_DIAGNOSTICS", "unset"))
    add("MIC_DISPLAY_UNLOCK_HINTS false", os.getenv("MIC_DISPLAY_UNLOCK_HINTS", "false").lower() not in {"1", "true", "yes", "on"}, os.getenv("MIC_DISPLAY_UNLOCK_HINTS", "unset"))
    add("MIC_ADMIN_ACTIONS_ENABLED false", os.getenv("MIC_ADMIN_ACTIONS_ENABLED", "false").lower() not in {"1", "true", "yes", "on"}, os.getenv("MIC_ADMIN_ACTIONS_ENABLED", "unset"))
    add("developer token configured", bool(os.getenv("MIC_DEVELOPER_TOKEN")), "set" if os.getenv("MIC_DEVELOPER_TOKEN") else "missing")
    add("admin token configured", bool(os.getenv("MIC_ADMIN_TOKEN")), "set" if os.getenv("MIC_ADMIN_TOKEN") else "missing")

    m = latest_manifest(manifest_dir)
    if not m:
        add("latest manifest exists", False, str(manifest_dir))
    else:
        mode = m.get("mode")
        points = (m.get("qdrant") or {}).get("indexed_points")
        if points is None:
            points = m.get("indexed_points")
        add("latest manifest exists", True, m.get("rebuild_id", "unknown"))
        if args.expect_mode != "any":
            add(f"latest manifest mode is {args.expect_mode}", mode == args.expect_mode, str(mode))
        add("latest manifest has snapshot", bool((m.get("snapshot") or {}).get("created")), str((m.get("snapshot") or {}).get("snapshot_name")))
        add("latest manifest has indexed points", isinstance(points, int) and points > 0, str(points))
        ds = m.get("document_safety") or {}
        if args.expect_mode == "production":
            add("production accepted files > 0", int(ds.get("accepted_files", 0)) > 0, str(ds.get("accepted_files")))
            add("production rejected files == 0", int(ds.get("rejected_files", 0)) == 0, str(ds.get("rejected_files")))

    if not args.skip_runtime and (root / "scripts" / "inspect_runtime_state.py").exists():
        runtime, err, code = run_json([sys.executable, "scripts/inspect_runtime_state.py", "--no-warm-embedding"])
        if runtime is None:
            add("runtime diagnostics JSON", False, err or f"returncode {code}")
        else:
            cons = runtime.get("consistency") or {}
            latest = runtime.get("latest_manifest") or {}
            qdrant = runtime.get("qdrant") or {}
            add("qdrant ok", bool(qdrant.get("ok")), str(qdrant.get("error")))
            add("active index matches latest manifest", bool(cons.get("active_index_matches_latest_manifest")), json.dumps(cons, ensure_ascii=False))
            if args.expect_mode != "any":
                add(f"runtime latest manifest mode is {args.expect_mode}", latest.get("mode") == args.expect_mode, str(latest.get("mode")))
    elif not args.skip_runtime:
        add("runtime diagnostics script exists", False, "scripts/inspect_runtime_state.py missing")

    failures = [c for c in checks if not c["ok"]]
    report = {
        "env_file": args.env_file,
        "expect_mode": args.expect_mode,
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures and args.strict:
        return 1
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
