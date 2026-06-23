#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env.production}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

BACKUP_DIR="${MIC_BACKUP_DIR:-storage/backups}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/mic9000_backup_$TS.tar.gz"
mkdir -p "$BACKUP_DIR"

TMP_MANIFEST="$(mktemp)"
cat > "$TMP_MANIFEST" <<EOF
MIC 9000 backup
created_at_utc=$TS
project=$(pwd)
index_mode=${MIC_INDEX_MODE:-unknown}
includes=data/documents data/sqlite storage/index_manifests storage/index_quarantine eval_reports .chainlit chainlit.md public
excludes=*.env .env.production .env.security secrets tokens qdrant snapshots by default
EOF

tar -czf "$OUT" \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*.snapshot' \
  --exclude='storage/backups' \
  --exclude='storage/ops_bundles' \
  --transform="s|^$TMP_MANIFEST$|BACKUP_MANIFEST.txt|" \
  "$TMP_MANIFEST" \
  data/documents \
  data/sqlite \
  storage/index_manifests \
  storage/index_quarantine \
  eval_reports \
  .chainlit \
  chainlit.md \
  public 2>/tmp/mic9000_backup_tar.err || {
    cat /tmp/mic9000_backup_tar.err >&2
    rm -f "$TMP_MANIFEST"
    exit 1
  }
rm -f "$TMP_MANIFEST"

echo "$OUT"
