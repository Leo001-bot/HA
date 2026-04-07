#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -x "$PROJECT_DIR/cpp/build/hearing_aid_realtime" ]; then
  echo "C++ binary not found. Run ./setup_pi.sh first."
  exit 1
fi

USE_CPP_BRIDGE=1 python3 -u main.py
