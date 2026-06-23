#!/usr/bin/env bash
set -euo pipefail

BACKUP="${1:-}"
CONFIRM="${2:-}"

if [[ -z "$BACKUP" || ! -f "$BACKUP" ]]; then
  echo "Usage: bash scripts/restore_mic9000_backup.sh <backup.tar.gz> [--yes]" >&2
  exit 1
fi

if [[ "$CONFIRM" != "--yes" ]]; then
  echo "Dry run. Archive contents preview:"
  tar -tzf "$BACKUP" | sed -n '1,80p'
  echo
  echo "Re-run with --yes to restore. A pre-restore backup will be created first."
  exit 0
fi

mkdir -p storage/backups
PRE="storage/backups/pre_restore_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
echo "Creating pre-restore backup: $PRE"
tar -czf "$PRE" \
  --exclude='storage/backups' \
  --exclude='storage/ops_bundles' \
  data/documents data/sqlite storage/index_manifests storage/index_quarantine eval_reports .chainlit chainlit.md public 2>/dev/null || true

echo "Restoring $BACKUP into $(pwd)"
tar -xzf "$BACKUP"
echo "Restore complete. Pre-restore backup: $PRE"
