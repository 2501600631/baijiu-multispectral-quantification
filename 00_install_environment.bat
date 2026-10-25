@echo off
cd /d "%~dp0"
for %%V in (3.12 3.11 3.10 3.9) do (
  py -%%V -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    py -%%V -m venv .venv
    goto :install
  )
)
echo Install Python 3.9 through 3.12 and run this launcher again.
exit /b 1
:install
if not exist ".venv\Scripts\python.exe" exit /b 1
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
pause
