# PowerShell Environment Guide

This guide is for using **PowerShell** with the phone reconstruction system.

## Quick Start

### First Time Setup
```powershell
cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction
.\setup_env.ps1
```

### Daily Usage
```powershell
# Navigate to project
cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction

# Activate environment (do this every time you open PowerShell)
.\activate_env.ps1

# Run reconstruction
python scripts\run_reconstruction.py -i data/input/test1.jpg -o data/output
```

---

## What Was Fixed

### ✅ Activate.ps1 Signature Issue
**Problem:** The original `Activate.ps1` had custom code after the digital signature block, which PowerShell security policy blocks.

**Solution:** Removed the problematic line from `Activate.ps1` and created `activate_env.ps1` which:
1. Calls the clean `Activate.ps1`
2. Sets PYTHONPATH for Depth-Anything-V2

---

## Files Overview

| File | Purpose |
|------|---------|
| `setup_env.ps1` | Creates fresh environment (run once) |
| `activate_env.ps1` | Activates environment with PYTHONPATH (run every session) |
| `venv\Scripts\Activate.ps1` | Standard venv activation (fixed) |

---

## Execution Policy Issue?

If you get "cannot be loaded because running scripts is disabled", run this **once** as Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or run scripts with bypass:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

---

## Verify Setup

```powershell
# Check if loguru is installed
pip show loguru

# Check PYTHONPATH
$env:PYTHONPATH

# Should show: src;src\Depth-Anything-V2;...
```

---

## Complete Workflow

```powershell
# === ONE TIME SETUP ===
cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction
.\setup_env.ps1

# === EVERY DAY ===
cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction
.\activate_env.ps1

# Run your scripts
python scripts\run_reconstruction.py -i data/input/test1.jpg -o data/output
python scripts\run_reconstruction.py -i data/input/test2.jpg -o data/output

# Deactivate when done (optional)
deactivate
```

---

## Tips

### Add to PowerShell Profile
To auto-navigate to project when opening PowerShell:

```powershell
# Edit profile
notepad $PROFILE

# Add this line:
cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction
```

### Create Alias
```powershell
# Add to $PROFILE
function Activate-PhoneReconstruction {
    cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction
    .\activate_env.ps1
}
Set-Alias reconstruct Activate-PhoneReconstruction
```

Then just type `reconstruct` to activate!

---

## Troubleshooting

### "Activate.ps1 cannot be loaded"
Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### "Module not found"
Make sure you ran `.\activate_env.ps1` (not just `Activate.ps1`)

### "PYTHONPATH not set"
Use `.\activate_env.ps1` instead of calling `Activate.ps1` directly

### Need fresh environment?
Just run `.\setup_env.ps1` again - it will delete and recreate everything
