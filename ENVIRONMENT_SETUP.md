# Environment Setup Guide

This guide shows you how to create a fresh, working environment for the phone reconstruction system.

## Option 1: Automated Setup (Recommended)

Run the setup script which handles everything automatically:

```cmd
setup_env.bat
```

This will:
1. ✅ Remove old virtual environment (if exists)
2. ✅ Create new virtual environment
3. ✅ Install all dependencies from requirements.txt
4. ✅ Configure PYTHONPATH for Depth-Anything-V2

**Time:** ~2-5 minutes depending on internet speed

---

## Option 2: Manual Setup

### Step 1: Create Virtual Environment

```cmd
python -m venv venv
```

### Step 2: Activate Environment

```cmd
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure PYTHONPATH

Edit `venv\Scripts\activate.bat` and add these lines **before** the `:END` label:

```batch
rem Set PYTHONPATH for Depth-Anything-V2
set PYTHONPATH=src;src\Depth-Anything-V2;%PYTHONPATH%
```

**Example:**
```batch
set PATH=%VIRTUAL_ENV%\Scripts;%PATH%
set VIRTUAL_ENV_PROMPT=(venv) 

rem Set PYTHONPATH for Depth-Anything-V2
set PYTHONPATH=src;src\Depth-Anything-V2;%PYTHONPATH%

:END
```

---

## How to Activate Environment

### Every Time You Open a New Terminal:

```cmd
cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction
venv\Scripts\activate.bat
```

You'll see `(venv)` in your prompt, meaning the environment is active.

---

## Verify Setup

After activating, test that everything works:

```cmd
# Check if packages are installed
pip show loguru

# Check if Depth-Anything-V2 is accessible
python -c "import sys; print('src' in sys.path or 'src' in str(sys.path))"

# Run reconstruction
python scripts\run_reconstruction.py -i data/input/test1.jpg -o data/output
```

---

## Understanding What Was Fixed

### Issue 1: Missing Dependencies
**Problem:** Packages like `loguru`, `click` were not installed

**Solution:** `pip install -r requirements.txt` installs all required packages

### Issue 2: Depth-Anything-V2 Not Found
**Problem:** Python couldn't find the module in `src\Depth-Anything-V2\`

**Solution:** Set `PYTHONPATH=src;src\Depth-Anything-V2;%PYTHONPATH%` in activate.bat

### Issue 3: PowerShell Execution Policy
**Problem:** `Activate.ps1` was blocked by security policy

**Solution:** Use `activate.bat` instead (cmd.exe), which doesn't have this restriction

---

## Troubleshooting

### "python: command not found"
Make sure Python is installed and in your PATH

### "pip install fails"
Try:
```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### "Still can't import depth_anything_v2"
Check PYTHONPATH is set after activation:
```cmd
echo %PYTHONPATH%
```
Should show: `src;src\Depth-Anything-V2;...`

### Need to start over?
Just run `setup_env.bat` again - it will delete and recreate everything

---

## Quick Reference

| Action | Command |
|--------|---------|
| **Setup new environment** | `setup_env.bat` |
| **Activate environment** | `venv\Scripts\activate.bat` |
| **Deactivate environment** | `deactivate` |
| **Install new package** | `pip install package_name` |
| **Update requirements** | `pip freeze > requirements.txt` |
| **Run reconstruction** | `python scripts\run_reconstruction.py -i data/input/test1.jpg` |

---

## Daily Workflow

```cmd
# 1. Navigate to project
cd c:\Users\sushe\OneDrive\Documents\Review\phone_reconstruction

# 2. Activate environment (do this every time you open a new terminal)
venv\Scripts\activate.bat

# 3. Run your scripts
python scripts\run_reconstruction.py -i data/input/test1.jpg -o data/output

# 4. When done, deactivate (optional)
deactivate
```

---

## Why Virtual Environments?

- ✅ **Isolated dependencies** - Don't interfere with other Python projects
- ✅ **Reproducible** - Same packages on any machine
- ✅ **Easy to reset** - Just delete `venv` folder and recreate
- ✅ **Custom configuration** - Set PYTHONPATH and other env vars per project
