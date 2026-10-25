@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" run.py check --profile main
if errorlevel 1 exit /b 1
pause
