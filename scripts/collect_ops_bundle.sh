#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env.production}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

OUT_DIR="${MIC_OPS_BUNDLE_DIR:-storage/ops_bundles}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d)"
OUT="$OUT_DIR/mic9000_ops_bundle_$TS.tar.gz"
mkdir -p "$OUT_DIR"

safe_run() {
  local name="$1"; shift
  { "$@" > "$WORK/$name.out" 2> "$WORK/$name.err"; echo $? > "$WORK/$name.code"; } || true
}

safe_run security_config python scripts/check_security_config.py
safe_run runtime_state python scripts/inspect_runtime_state.py --no-warm-embedding
safe_run manifests python scripts/list_index_manifests.py
safe_run preflight python scripts/ops_preflight.py --env-file "$ENV_FILE"

cp -r eval_reports "$WORK/eval_reports" 2>/dev/null || true
cp -r storage/index_manifests "$WORK/index_manifests" 2>/dev/null || true
cp "${MIC_CHAINLIT_LOG:-logs/chainlit.log}" "$WORK/chainlit.log" 2>/dev/null || true

cat > "$WORK/README.txt" <<EOF
MIC 9000 ops bundle
created_at_utc=$TS
project=$(pwd)
Secrets and env files are intentionally excluded.
EOF

tar -czf "$OUT" -C "$WORK" .
rm -rf "$WORK"
echo "$OUT"
