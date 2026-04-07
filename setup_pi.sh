#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$PROJECT_DIR/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"

echo "[0/5] Updating repository..."
if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	git -C "$PROJECT_DIR" pull --ff-only
else
	echo "Not a git repository; skipping git pull."
fi

echo "[1/5] Installing system packages..."
sudo apt update
sudo apt install -y git cmake build-essential pkg-config portaudio19-dev libportaudio2 python3-pip python3-venv python3-dev

echo "[2/5] Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install flask flask-socketio python-socketio python-engineio numpy sounddevice

echo "[3/5] Ensuring model files are present..."
mkdir -p "$MODEL_DIR"
if ! find "$MODEL_DIR" -maxdepth 1 -type f \( -name 'encoder*.onnx' -o -name 'decoder*.onnx' -o -name 'joiner*.onnx' \) | grep -q .; then
	tmp_archive="$(mktemp /tmp/sherpa-model.XXXXXX.tar.bz2)"
	if command -v wget >/dev/null 2>&1; then
		wget -O "$tmp_archive" "$MODEL_URL"
	else
		curl -L -o "$tmp_archive" "$MODEL_URL"
	fi
	tar -xjf "$tmp_archive" -C "$PROJECT_DIR/models"
	rm -f "$tmp_archive"
fi

echo "[4/5] Building C++ engine..."
cd "$PROJECT_DIR/cpp"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo "[5/5] Starting web UI + C++ bridge..."
cd "$PROJECT_DIR"
exec env USE_CPP_BRIDGE=1 python3 -u main.py
