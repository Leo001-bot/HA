#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/4] Installing system packages..."
sudo apt update
sudo apt install -y git cmake build-essential pkg-config portaudio19-dev libportaudio2 python3-pip python3-venv python3-dev

echo "[2/4] Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install flask flask-socketio python-socketio python-engineio numpy sounddevice

echo "[3/4] Building C++ engine..."
cd "$PROJECT_DIR/cpp"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

echo "[4/4] Setup complete."
echo "Run the web UI + C++ bridge with:"
echo "  cd '$PROJECT_DIR'"
echo "  USE_CPP_BRIDGE=1 python3 -u main.py"
