@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" analysis\analyze_saved_predictions.py
if errorlevel 1 exit /b 1
pause
