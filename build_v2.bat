@echo off
chcp 65001 >nul
echo PhenoMe Windows build script
echo =======================================

REM Sanity check: must run from the repository root
if not exist "main.py" (
    echo ERROR: run this script from the directory containing main.py
    pause
    exit /b 1
)

REM Step 1: dependency smoke test
echo.
echo [1/4] Checking dependencies...
python test_build.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: dependency smoke test reported issues; continuing anyway.
    echo.
)

REM Step 2: make sure PyInstaller is installed
echo.
echo [2/4] Checking PyInstaller...
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo PyInstaller not found, installing...
    pip install pyinstaller
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: failed to install PyInstaller
        pause
        exit /b 1
    )
) else (
    echo PyInstaller already installed.
)

REM Step 3: clean previous build artifacts
echo.
echo [3/4] Cleaning previous build output...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"

REM Step 4: build with PyInstaller
echo.
echo [4/4] Building executable (this can take a while)...
python -m PyInstaller --clean PhenoMe.spec

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build finished.
    echo Executable: dist\PhenoMe\PhenoMe.exe
    if exist "dist\PhenoMe\PhenoMe.exe" (
        for %%A in ("dist\PhenoMe\PhenoMe.exe") do (
            set /a "filesize=%%~zA / 1048576"
        )
        if defined filesize echo Size: %filesize% MB
    ) else (
        echo WARNING: dist\PhenoMe\PhenoMe.exe not found — check the log above.
    )
    echo.
    echo Next steps:
    echo   1. Run dist\PhenoMe\PhenoMe.exe to smoke-test the build
    echo   2. Copy the whole dist\PhenoMe folder to deploy on other machines
) else (
    echo.
    echo BUILD FAILED — see the PyInstaller output above.
)

echo.
pause >nul
