@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" skeleton.py
) else (
    py -3 main.py
)

echo.
pause
