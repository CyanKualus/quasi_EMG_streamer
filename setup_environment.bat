@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Ready. Open launch_quasi_EMG_processing.bat.
pause
exit /b 0
:failed
echo Setup failed. Python 3.12 and network access are required.
pause
exit /b 1
