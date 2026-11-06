@echo off
setlocal

rem Always resolve imports from this project directory, not the caller's cwd.
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
rem Interpreter: set PHENOME_PYTHON to a conda env python, else PATH is used.
if not defined PHENOME_PYTHON set "PHENOME_PYTHON=python"

where %PHENOME_PYTHON% >nul 2>nul
if errorlevel 1 (
  echo Error: Python was not found: %PHENOME_PYTHON%
  echo Set PHENOME_PYTHON to your interpreter, e.g. set PHENOME_PYTHON=C:\envs\bean\python.exe
  pause
  exit /b 1
)

for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do set "PHENOME_PORT_PID=%%P"
if defined PHENOME_PORT_PID (
  echo Port 8000 is already in use by PID %PHENOME_PORT_PID%.
  echo Stop that service first, then run this file again.
  pause
  exit /b 1
)

echo Starting PhenoMe Web at http://0.0.0.0:8000
echo Press Ctrl+C to stop the service.
"%PHENOME_PYTHON%" -m uvicorn web.app:app --host 0.0.0.0 --port 8000
set "PHENOME_EXIT_CODE=%ERRORLEVEL%"

if not "%PHENOME_EXIT_CODE%"=="0" pause
exit /b %PHENOME_EXIT_CODE%
