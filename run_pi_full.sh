#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
cd "$PROJECT_DIR"

if [ -x "$VENV_PYTHON" ]; then
  exec env USE_CPP_BRIDGE=0 "$VENV_PYTHON" -u main.py
fi

exec env USE_CPP_BRIDGE=0 python3 -u main.py
