#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: expected repo-root interpreter at $PYTHON_BIN" >&2
  echo "The repo-root .venv is missing or broken. Repair ${REPO_ROOT}/.venv first." >&2
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_openai_compatible_models.py" "$@"
