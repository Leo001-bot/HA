#!/usr/bin/env bash
set -euo pipefail

# C/C++ STACK LAUNCHER (Raspberry Pi)
# This builds and runs the standalone C++ realtime engine.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/cpp"

cmake -S . -B build
cmake --build build -j4

"$SCRIPT_DIR/cpp/build/hearing_aid_realtime" "$SCRIPT_DIR/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
