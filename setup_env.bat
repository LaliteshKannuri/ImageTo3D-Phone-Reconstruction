@echo off
REM ============================================================
REM Phone Reconstruction Environment Setup Script
REM This script creates a fresh virtual environment with all
REM dependencies and PYTHONPATH configuration
REM ============================================================

echo.
echo ============================================================
echo  Phone Reconstruction Environment Setup
echo ============================================================
echo.

REM Step 1: Remove old environment if it exists
if exist venv (
    echo [1/4] Removing old virtual environment...
    rmdir /s /q venv
    echo       Done!
) else (
    echo [1/4] No existing environment found, creating fresh...
)

echo.

REM Step 2: Create new virtual environment
echo [2/4] Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo       Done!
echo.

REM Step 3: Install dependencies
echo [3/4] Installing dependencies from requirements.txt...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo       Done!
echo.

REM Step 4: Configure PYTHONPATH in activate.bat
echo [4/4] Configuring PYTHONPATH for Depth-Anything-V2...

REM Check if PYTHONPATH is already configured
findstr /C:"PYTHONPATH=src" venv\Scripts\activate.bat >nul
if errorlevel 1 (
    REM Add PYTHONPATH configuration before :END label
    powershell -Command "(Get-Content venv\Scripts\activate.bat) -replace ':END', 'rem Set PYTHONPATH for Depth-Anything-V2`r`nset PYTHONPATH=src;src\Depth-Anything-V2;%%PYTHONPATH%%`r`n`r`n:END' | Set-Content venv\Scripts\activate.bat"
    echo       PYTHONPATH configured!
) else (
    echo       PYTHONPATH already configured!
)

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo Your environment is ready to use. To activate it:
echo.
echo   venv\Scripts\activate.bat
echo.
echo Then run reconstruction:
echo.
echo   python scripts\run_reconstruction.py -i data/input/test1.jpg
echo.
echo ============================================================
pause
