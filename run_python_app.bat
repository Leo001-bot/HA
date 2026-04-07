@echo off
setlocal

REM PYTHON STACK LAUNCHER
REM This starts the existing Flask + Audio + STT Python application.

cd /d %~dp0
python main.py

endlocal
