# Stack Labels

## Python Stack Files
- main.py: Main runtime loop (audio processing + server startup)
- server.py: Flask/Socket.IO web server and API
- audio_io.py: PortAudio-based input/output bridge in Python
- processing.py: DSP modules (NR, compressor, EQ, limiter)
- stt.py: STT backends (Python backend + optional ALSA/external backend)
- config.py: Runtime configuration and validation
- templates/index.html: Web UI
- run_python_app.bat: Launcher for Python app (Windows)

Run command:
- python main.py

## C/C++ Stack Files
- cpp/hearing_aid_realtime.cpp: Standalone realtime audio + optional sherpa-onnx C API STT
- cpp/CMakeLists.txt: Build file for C++ target
- run_cpp_realtime_pi.sh: Build+run helper for Raspberry Pi/Linux

Run command:
- ./run_cpp_realtime_pi.sh

## Important
- Python and C/C++ stacks are currently separate runtimes.
- The C/C++ binary is not yet connected to Flask/Socket.IO APIs in server.py.
