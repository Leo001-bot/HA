#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
cd "$PROJECT_DIR"

if [ -x "$VENV_PYTHON" ]; then
  exec env STT_BACKEND=cpp HA_STT_THREADS=1 "$VENV_PYTHON" -u main.py
fi

exec env STT_BACKEND=cpp HA_STT_THREADS=1 python3 -u main.py