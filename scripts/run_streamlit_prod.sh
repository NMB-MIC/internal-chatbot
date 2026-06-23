#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-.env.production}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

HOST="${STREAMLIT_HOST:-${HOST:-0.0.0.0}}"
PORT="${STREAMLIT_PORT:-${PORT:-8501}}"

exec streamlit run streamlit_app.py \
  --server.address "$HOST" \
  --server.port "$PORT"
