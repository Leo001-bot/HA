#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
MODEL_DIR="$PROJECT_DIR/models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"

if [ "${EUID}" -eq 0 ]; then
  echo "Please run without sudo: ./setup_pi.sh"
  echo "The script already uses sudo for apt commands and keeps Python deps in a local virtualenv."
  exit 1
fi

echo "[0/6] Updating repository..."
if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	git -C "$PROJECT_DIR" pull --ff-only
else
	echo "Not a git repository; skipping git pull."
fi

echo "[1/6] Installing system packages..."
sudo apt update
sudo apt install -y git cmake build-essential pkg-config portaudio19-dev libportaudio2 python3-pip python3-venv python3-dev

echo "[2/6] Creating virtual environment..."
if [ ! -x "$VENV_PYTHON" ]; then
	python3 -m venv "$VENV_DIR"
fi

echo "[3/6] Installing Python packages into virtual environment..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install flask flask-socketio python-socketio python-engineio numpy sounddevice

echo "[4/6] Ensuring model files are present..."
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

echo "[5/6] Building C++ engine..."
cd "$PROJECT_DIR/cpp"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo "[6/6] Starting web UI + C++ bridge..."
cd "$PROJECT_DIR"
exec env USE_CPP_BRIDGE=1 "$VENV_PYTHON" -u main.py
