@echo off
setlocal

REM PYTHON + C++ BRIDGE LAUNCHER
REM Starts the Flask web server and launches the native C++ engine in bridge mode.

cd /d %~dp0
set USE_CPP_BRIDGE=1
python main.py

endlocal
