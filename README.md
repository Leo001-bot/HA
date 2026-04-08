# Hearing Aid Project

A hearing-aid prototype with a Python web UI/server and an optional C++ real-time audio engine.

## What Runs Where

- Python: Flask web server, Socket.IO UI, configuration, and bridge controller
- C++: real-time audio processing and optional sherpa-onnx C API STT
- Bridge mode: Python launches the compiled C++ engine and forwards STT text to the web UI

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

`setup_pi.sh` will:

1. `git pull` the latest changes
2. install system packages
3. install Python packages
4. check/download the model bundle if missing
5. build the C++ engine
6. start the Python server + C++ bridge

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
- On Raspberry Pi, the project uses conservative audio defaults for stability.

## License

See upstream dependency licenses in their respective projects.
