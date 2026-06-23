#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env.production}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

echo "== Load environment =="
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo "== Compile release scripts =="
python -m py_compile scripts/freeze_release_manifest.py scripts/print_release_summary.py

echo "== Run production release checks =="
bash scripts/run_release_checks.sh "$ENV_FILE"

echo "== Freeze release manifest =="
python scripts/freeze_release_manifest.py --env-file "$ENV_FILE" --version "1.0.0-rc1" --ui-runtime "streamlit"

LATEST_RELEASE_MANIFEST="$(ls -t storage/releases/mic9000_release_*.json | head -n 1)"

echo "== Release summary =="
python scripts/print_release_summary.py "$LATEST_RELEASE_MANIFEST"

echo "FINAL RELEASE CHECK: PASS"
