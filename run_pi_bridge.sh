#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
cd "$PROJECT_DIR"

if [ ! -x "$PROJECT_DIR/cpp/build/hearing_aid_realtime" ]; then
  echo "C++ binary not found. Run ./setup_pi.sh first."
  exit 1
fi

if [ -x "$VENV_PYTHON" ]; then
  exec env USE_CPP_BRIDGE=1 "$VENV_PYTHON" -u main.py
fi

exec env USE_CPP_BRIDGE=1 python3 -u main.py
