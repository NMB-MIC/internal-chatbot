#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
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


def http_ok(url: str, timeout: float = 2.0) -> tuple[bool, str | None]:
    try:
        req = Request(url, headers={"User-Agent": "mic9000-preflight"})
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 500, f"HTTP {resp.status}"
    except HTTPError as exc:
        return True, f"HTTP {exc.code}"
    except URLError as exc:
        return False, str(exc.reason)
    except Exception as exc:  # pragma: no cover
        return False, repr(exc)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def command_available(name: str) -> bool:
    return shutil.which(name) is not None


def truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="MIC 9000 Streamlit ops preflight checker")
    parser.add_argument("--env-file", default=".env.production")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd()
    env = load_env_file(project_root / args.env_file)

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
    docs_dir = Path(os.getenv("DOCUMENTS_DIR", "data/documents"))
    sqlite_db = Path(os.getenv("SQLITE_DB_PATH", "data/sqlite/chat_history.db"))
    manifest_dir = Path(os.getenv("INDEX_MANIFEST_DIR", "storage/index_manifests"))
    backup_dir = Path(os.getenv("MIC_BACKUP_DIR", "storage/backups"))
    ops_bundle_dir = Path(os.getenv("MIC_OPS_BUNDLE_DIR", "storage/ops_bundles"))

    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    add("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    add("streamlit command", command_available("streamlit"), shutil.which("streamlit") or "missing")
    add("streamlit module", module_available("streamlit"), "installed" if module_available("streamlit") else "missing")
    add("qdrant_client module", module_available("qdrant_client"), "installed" if module_available("qdrant_client") else "missing")
    add("sentence_transformers module", module_available("sentence_transformers"), "installed" if module_available("sentence_transformers") else "missing")
    add("streamlit_app.py exists", (project_root / "streamlit_app.py").exists(), "")
    add("chainlit_app.py not required", True, "Streamlit UI is active; Chainlit checks removed")
    add("eval suites exist", (project_root / "eval_suites").exists(), "")
    add("documents dir exists", docs_dir.exists(), str(docs_dir))
    add("documents dir has files", any(docs_dir.rglob("*")) if docs_dir.exists() else False, str(docs_dir))
    add("sqlite parent exists", sqlite_db.parent.exists(), str(sqlite_db.parent))
    add("manifest dir exists", manifest_dir.exists(), str(manifest_dir))
    add("manifest dir has manifests", any(manifest_dir.glob("*.json")) if manifest_dir.exists() else False, str(manifest_dir))
    backup_dir.mkdir(parents=True, exist_ok=True)
    ops_bundle_dir.mkdir(parents=True, exist_ok=True)
    add("backup dir writable", os.access(backup_dir, os.W_OK), str(backup_dir))
    add("ops bundle dir writable", os.access(ops_bundle_dir, os.W_OK), str(ops_bundle_dir))

    q_ok, q_detail = http_ok(f"{qdrant_url}/collections")
    add("qdrant reachable", q_ok, f"{qdrant_url}/collections :: {q_detail}")

    security_enabled = truthy(os.getenv("MIC_SECURITY_ENABLED"))
    add("security enabled", security_enabled, os.getenv("MIC_SECURITY_ENABLED", "unset"))
    add("developer token configured", bool(os.getenv("MIC_DEVELOPER_TOKEN")), "set" if os.getenv("MIC_DEVELOPER_TOKEN") else "missing")
    add("admin token configured", bool(os.getenv("MIC_ADMIN_TOKEN")), "set" if os.getenv("MIC_ADMIN_TOKEN") else "missing")
    add("public trace disabled", not truthy(os.getenv("MIC_SECURITY_PUBLIC_TRACE", "false")), os.getenv("MIC_SECURITY_PUBLIC_TRACE", "unset"))
    add("public diagnostics disabled", not truthy(os.getenv("MIC_SECURITY_PUBLIC_DIAGNOSTICS", "false")), os.getenv("MIC_SECURITY_PUBLIC_DIAGNOSTICS", "unset"))
    add("unlock hints disabled", not truthy(os.getenv("MIC_DISPLAY_UNLOCK_HINTS", "false")), os.getenv("MIC_DISPLAY_UNLOCK_HINTS", "unset"))

    failures = [c for c in checks if not c["ok"]]
    report = {
        "project_root": str(project_root),
        "env_file": args.env_file,
        "ui_runtime": "streamlit",
        "loaded_env_keys": sorted(env.keys()),
        "failure_count": len(failures),
        "checks": checks,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures and args.strict:
        return 1
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
