@echo off
rem Double-click to open the Zet interface. The first run sets everything up (a few minutes).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Setting up Zet for the first time...
  python -m venv .venv || (echo Python 3.10 or newer is needed: https://www.python.org/downloads/ & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --quiet -e . || (echo Installing Zet failed. & pause & exit /b 1)
)
".venv\Scripts\python.exe" -c "import zet.ui" 2>nul || ".venv\Scripts\python.exe" -m pip install --quiet -e .
echo Zet is starting. Keep this window open while you use Zet; close it to stop.
".venv\Scripts\python.exe" -m zet.ui
pause
