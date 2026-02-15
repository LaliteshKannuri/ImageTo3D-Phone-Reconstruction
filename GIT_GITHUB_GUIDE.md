# Git & GitHub Setup Guide for Phone Reconstruction

Complete guide to set up version control with Git and sync with your GitHub account.

## Quick Setup

### 1. Configure Git (First Time Only)

```bash
# Set your GitHub username
git config --global user.name "YourGitHubUsername"

# Set your GitHub email
git config --global user.email "your.email@example.com"
```

### 2. Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click **+** → **New repository**
3. Name it: `phone_reconstruction`
4. **Don't** initialize with README (we already have code)
5. Click **Create repository**

### 3. Connect to GitHub

```bash
# Add GitHub as remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/phone_reconstruction.git

# Verify remote
git remote -v
```

### 4. Create First Checkpoint

```bash
# Check what files will be committed
git status

# Add all files
git add .

# Create checkpoint with message
git commit -m "Initial commit: Phone reconstruction project setup"

# Push to GitHub
git push -u origin main
```

---

## Daily Workflow: Creating Checkpoints

### Create a Checkpoint (Save Your Work)

```bash
# Option 1: Quick commit (commits all tracked changes)
git commit -am "Description of what you changed"

# Option 2: More control
git add .                              # Stage all changes
git commit -m "Description of changes" # Create checkpoint
```

### Push to GitHub (Backup Online)

```bash
git push
```

### Example Workflow

```bash
# After making changes to your code
git status                                    # See what changed
git commit -am "Fixed depth estimation bug"   # Create checkpoint
git push                                      # Backup to GitHub
```

---

## Rolling Back Changes

### View History

```bash
# See all checkpoints
git log --oneline

# Output example:
# abc1234 Fixed depth estimation bug
# def5678 Added new feature
# 789ghij Initial commit
```

### Roll Back Options

#### 1. **Undo Last Commit (Keep Changes)**
```bash
git reset --soft HEAD~1
```
Your changes stay in working directory, uncommitted.

#### 2. **Undo Last Commit (Discard Changes)**
```bash
git reset --hard HEAD~1
```
⚠️ **Warning:** This deletes all changes!

#### 3. **Go Back to Specific Checkpoint**
```bash
# Get commit hash from git log
git log --oneline

# Go back to that checkpoint
git reset --hard abc1234  # Replace with actual hash
```

#### 4. **Create New Branch from Old Checkpoint**
```bash
git checkout -b experiment abc1234
```
This lets you explore old code without losing current work.

---

## Useful Commands

| Command | What It Does |
|---------|-------------|
| `git status` | See what files changed |
| `git log --oneline` | View checkpoint history |
| `git diff` | See exact changes |
| `git commit -am "message"` | Quick checkpoint |
| `git push` | Backup to GitHub |
| `git pull` | Download from GitHub |
| `git checkout filename` | Undo changes to one file |

---

## .gitignore Explained

Your `.gitignore` already excludes:
- ✅ `venv/` - Virtual environment (too large)
- ✅ `models/*.pth` - Model files (too large for GitHub)
- ✅ `data/output/*` - Generated files
- ✅ `__pycache__/` - Python cache

**This is good!** These files don't need to be in version control.

---

## Complete Example Workflow

```bash
# === SETUP (ONCE) ===
git config --global user.name "sushe"
git config --global user.email "your.email@gmail.com"
git remote add origin https://github.com/sushe/phone_reconstruction.git
git add .
git commit -m "Initial commit"
git push -u origin main

# === DAILY WORK ===
# ... make changes to code ...

# Create checkpoint
git commit -am "Improved screw detection accuracy"

# Backup to GitHub
git push

# === ROLL BACK IF NEEDED ===
git log --oneline                    # Find checkpoint
git reset --hard abc1234             # Go back to it
```

---

## Authentication with GitHub

### Option 1: Personal Access Token (Recommended)

1. GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token
3. Select scopes: `repo` (full control)
4. Copy the token
5. When pushing, use token as password:
   ```
   Username: your-github-username
   Password: ghp_xxxxxxxxxxxx (your token)
   ```

### Option 2: SSH Key

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

Then use SSH remote:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/phone_reconstruction.git
```

---

## Troubleshooting

### "fatal: not a git repository"
```bash
git init
```

### "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/phone_reconstruction.git
```

### Push rejected (diverged)
```bash
git pull --rebase
git push
```

### Large files blocking push
Add them to `.gitignore`, then:
```bash
git rm --cached filename
git commit -m "Remove large file"
```

---

## Pro Tips

### 1. Commit Often
Small, frequent checkpoints are better than one big commit.

### 2. Write Good Messages
```bash
# Good
git commit -m "Fix depth normalization bug in depth_estimator.py"

# Bad
git commit -m "fixes"
```

### 3. Branch for Experiments
```bash
git checkout -b new-feature
# ... experiment ...
git checkout main  # Go back to stable code
```

### 4. Check Before Committing
```bash
git status    # What changed?
git diff      # Exact changes?
git add .     # Stage changes
git commit    # Commit
```
