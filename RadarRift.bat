@echo off
title RadarRift
cd /d "%~dp0"

REM Use pythonw to launch without a console window (GUI-only).
REM Falls back to python if pythonw is not available.
pythonw main.py
if errorlevel 1 python main.py
