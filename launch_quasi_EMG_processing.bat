@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_environment.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 (
    echo quasi_EMG_processing failed to start.
    pause
)
endlocal
