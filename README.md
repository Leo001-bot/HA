# Hearing Aid Project

A hearing-aid prototype with a Python web UI/server and an optional C++ real-time audio engine.

## What Runs Where

- Python: Flask web server, Socket.IO UI, configuration, and bridge controller
- C++: real-time audio processing and optional sherpa-onnx C API STT
- Bridge mode: Python launches the compiled C++ engine and forwards STT text to the web UI
- Compression controls: threshold, ratio, makeup gain, AGC target, AGC max gain are exposed in the UI
- Pi launch modes: `bridge` (C++ engine) or `python` (full Python DSP modules)
- Pi hybrid mode: Python audio processing + sliders, C++ stdin STT only (`STT_BACKEND=cpp`)

## Repository Layout

- `main.py` - Python entry point
- `server.py` - Flask/Socket.IO server
- `audio_io.py` - Python audio I/O path
- `processing.py` - Python DSP modules
- `stt.py` - Python STT backends and helpers
- `cpp_bridge.py` - Python launcher/monitor for the C++ engine
- `cpp/` - C++ realtime engine and build files
- `templates/index.html` - Web UI
- `models/` - Sherpa-ONNX model bundle
- `setup_pi.sh` - One-click Raspberry Pi setup/update/build/run script
- `run_pi_bridge.sh` - Run-only Raspberry Pi launcher

## Prerequisites

### Raspberry Pi / Ubuntu Linux

```bash
sudo apt update
sudo apt install -y git cmake build-essential pkg-config portaudio19-dev libportaudio2 python3-pip python3-venv python3-dev
```

### Python packages

```bash
python3 -m pip install flask flask-socketio python-socketio python-engineio numpy sounddevice
```

## Quick Start - Raspberry Pi

### One-click setup and launch

From the repository root:

```bash
cd ~/桌面/HA
chmod +x setup_pi.sh
./setup_pi.sh
```

Important: run `setup_pi.sh` without `sudo`. The script uses `sudo` only for apt commands and installs Python packages in a local `.venv`.
If you want the full Python modules to affect audio on Pi, use `HA_RUN_MODE=python ./setup_pi.sh` or `./run_pi_full.sh`.
If you want Python processing/sliders but C++ STT, use `HA_RUN_MODE=hybrid ./setup_pi.sh` or `./run_pi_hybrid.sh`.

`setup_pi.sh` will:

1. `git pull` the latest changes
2. install system packages
3. create/use local Python virtual environment (`.venv`) and install Python packages
4. check/download the model bundle if missing
5. build the C++ engine
6. start the Python server + C++ bridge

To keep all Python DSP modules active on Pi, launch with:

```bash
HA_RUN_MODE=python ./setup_pi.sh
```

### Run only the server + bridge after setup

```bash
cd ~/桌面/HA
chmod +x run_pi_bridge.sh
./run_pi_bridge.sh
```

Then open:

```text
http://localhost:5000
```

### Run full Python DSP mode on Pi

If you want the sliders and processing modules to directly affect output audio, run:

```bash
cd ~/桌面/HA
chmod +x run_pi_full.sh
./run_pi_full.sh
```

### Run hybrid mode on Pi

Python keeps audio processing and UI sliders, while C++ handles STT only:

```bash
cd ~/桌面/HA
chmod +x run_pi_hybrid.sh
./run_pi_hybrid.sh
```

## Quick Start - Linux Virtual Machine

If you already have the repository and model files in the VM:

```bash
cd ~/桌面/HA
USE_CPP_BRIDGE=1 python3 -u main.py
```

Bridge start command (copy/paste):

```bash
USE_CPP_BRIDGE=1 python3 -u main.py
```

If you only want the native C++ engine:

```bash
cd ~/桌面/HA
./run_cpp_vm.sh
```

## First-Time Clone

If the repository is not yet on the machine:

```bash
cd ~
git clone https://github.com/Leo001-bot/HA.git
cd HA
```

If the repo is stored on the desktop instead:

```bash
cd ~/桌面/HA
```

## Model Files

The C++ engine expects this model bundle:

```text
models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
```

It must contain at least:

- `tokens.txt`
- `encoder*.onnx`
- `decoder*.onnx`
- `joiner*.onnx`

## Notes

- The web UI and server are still Python.
- In bridge mode, the compiled C++ engine handles real-time audio processing and STT.
- The Python bridge forwards C++ STT transcripts into the existing UI chatbox.
- When compression settings change in the UI, the C++ bridge restarts so the new profile takes effect.
- On Raspberry Pi, the project uses conservative audio defaults for stability.

## License

See upstream dependency licenses in their respective projects.
