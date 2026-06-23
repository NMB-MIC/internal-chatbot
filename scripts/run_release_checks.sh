#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env.production}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

STRICT="${MIC_RELEASE_CHECK_STRICT:-true}"
STRICT_FLAG=""
if [[ "$STRICT" == "true" || "$STRICT" == "1" ]]; then
  STRICT_FLAG="--strict"
fi

mkdir -p logs eval_reports storage/backups storage/ops_bundles

echo "== Compile Python files =="
COMPILE_TARGETS=(
  "streamlit_app.py"
  "scripts/ops_preflight.py"
  "scripts/check_production_readiness.py"
)

if compgen -G "app/ui/*.py" > /dev/null; then
  COMPILE_TARGETS+=(app/ui/*.py)
fi

if compgen -G "app/security/*.py" > /dev/null; then
  COMPILE_TARGETS+=(app/security/*.py)
fi

if [[ -f "scripts/check_security_config.py" ]]; then
  COMPILE_TARGETS+=("scripts/check_security_config.py")
fi

python -m py_compile "${COMPILE_TARGETS[@]}"

echo "== Ops preflight =="
python scripts/ops_preflight.py --env-file "$ENV_FILE" $STRICT_FLAG

if [[ -f "scripts/check_security_config.py" ]]; then
  echo "== Security config =="
  python scripts/check_security_config.py
else
  echo "== Security config =="
  echo "scripts/check_security_config.py not found; skipping legacy Chainlit security display checker."
fi

echo "== Runtime diagnostics =="
python scripts/inspect_runtime_state.py --no-warm-embedding > logs/release_runtime_state.json
cat logs/release_runtime_state.json

echo "== Eval suites =="
python scripts/run_eval_suite.py --suite eval_suites/runbook_regression.json --no-warm-embedding
python scripts/run_eval_suite.py --suite eval_suites/document_mode_smoke.json --no-warm-embedding
python scripts/run_eval_suite.py --suite eval_suites/thai_runbook_smoke.json --no-warm-embedding

echo "== Production readiness =="
python scripts/check_production_readiness.py --env-file "$ENV_FILE" --expect-mode "${MIC_INDEX_MODE:-production}" $STRICT_FLAG

echo "== Release checks complete =="
