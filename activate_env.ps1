# Activate virtual environment and set PYTHONPATH for PowerShell
# Use this instead of calling Activate.ps1 directly

# Activate the virtual environment
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# Set PYTHONPATH for Depth-Anything-V2
$env:PYTHONPATH = "src;src\Depth-Anything-V2;" + $env:PYTHONPATH

Write-Host ""
Write-Host "✓ Virtual environment activated!" -ForegroundColor Green
Write-Host "✓ PYTHONPATH configured for Depth-Anything-V2" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Cyan
Write-Host "  python scripts\run_reconstruction.py -i data/input/test1.jpg" -ForegroundColor Yellow
Write-Host ""
