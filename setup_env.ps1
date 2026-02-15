# ============================================================
# Phone Reconstruction Environment Setup Script (PowerShell)
# This script creates a fresh virtual environment with all
# dependencies and PYTHONPATH configuration
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Phone Reconstruction Environment Setup (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Remove old environment if it exists
if (Test-Path "venv") {
    Write-Host "[1/4] Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
    Write-Host "      Done!" -ForegroundColor Green
} else {
    Write-Host "[1/4] No existing environment found, creating fresh..." -ForegroundColor Yellow
}

Write-Host ""

# Step 2: Create new virtual environment
Write-Host "[2/4] Creating new virtual environment..." -ForegroundColor Yellow
python -m venv venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "      Done!" -ForegroundColor Green
Write-Host ""

# Step 3: Install dependencies
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "      Done!" -ForegroundColor Green
Write-Host ""

# Step 4: Configure PYTHONPATH in activate.bat for cmd users
Write-Host "[4/4] Configuring PYTHONPATH for Depth-Anything-V2..." -ForegroundColor Yellow

$activateBat = Get-Content "venv\Scripts\activate.bat" -Raw
if ($activateBat -notmatch "PYTHONPATH=src") {
    $activateBat = $activateBat -replace ':END', "rem Set PYTHONPATH for Depth-Anything-V2`r`nset PYTHONPATH=src;src\Depth-Anything-V2;%PYTHONPATH%`r`n`r`n:END"
    Set-Content "venv\Scripts\activate.bat" $activateBat
    Write-Host "      PYTHONPATH configured in activate.bat!" -ForegroundColor Green
} else {
    Write-Host "      PYTHONPATH already configured!" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your environment is ready to use. To activate it:" -ForegroundColor White
Write-Host ""
Write-Host "  .\activate_env.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Then run reconstruction:" -ForegroundColor White
Write-Host ""
Write-Host "  python scripts\run_reconstruction.py -i data/input/test1.jpg" -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Read-Host "Press Enter to exit"
