# Detailed Setup Guide - 60 Days Advanced Data and AI

Complete setup instructions for the 60-day bootcamp.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Python Installation](#python-installation)
3. [Docker Installation](#docker-installation)
4. [Git Installation](#git-installation)
5. [Project Setup](#project-setup)
6. [Package Installation](#package-installation)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

### Knowledge Prerequisites
- ✅ Completed **100 Days of Data and AI** bootcamp
- ✅ Comfortable with Python programming
- ✅ Basic SQL knowledge
- ✅ Understanding of data structures and algorithms
- ✅ Familiarity with command line/terminal

### System Requirements
- **OS**: macOS, Linux, or Windows 10/11
- **RAM**: 16GB minimum, 32GB recommended
- **Disk Space**: 20GB+ free space
- **Internet**: Stable connection for downloads

### Accounts (Optional but Recommended)
- GitHub account (for version control)
- AWS account (for Days 22-23, 55, 59)
- Docker Hub account (optional)

---

## Python Installation

### Check Current Version

```bash
python3 --version
```

If you see `Python 3.11.x` or higher, skip to [Docker Installation](#docker-installation).

### macOS

**Option 1: Homebrew (Recommended)**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Verify
python3 --version
```

**Option 2: Official Installer**
1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. Verify installation

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip

# Verify
python3.11 --version
```

### Windows

1. Download Python 3.11+ from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer
3. ✅ **IMPORTANT**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify in Command Prompt:
   ```cmd
   python --version
   ```

---

## Docker Installation

Docker is required for running databases, Kafka, Airflow, and other services.

### Check if Installed

```bash
docker --version
docker ps
```

If both commands work, skip to [Git Installation](#git-installation).

### macOS

1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Install the .dmg file
3. Start Docker Desktop
4. Verify:
   ```bash
   docker --version
   docker ps
   ```

### Linux

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in, then verify
docker --version
docker ps
```

### Windows

1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Install the executable
3. Start Docker Desktop
4. Verify in PowerShell:
   ```powershell
   docker --version
   docker ps
   ```

**Note**: Windows requires WSL 2. Docker Desktop will guide you through setup.

---

## Git Installation

### Check if Installed

```bash
git --version
```

If installed, skip to [Project Setup](#project-setup).

### macOS

```bash
# Using Homebrew
brew install git

# Verify
git --version
```

### Linux

```bash
# Ubuntu/Debian
sudo apt install git

# Verify
git --version
```

### Windows

1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Run the installer (use default settings)
3. Verify in Command Prompt:
   ```cmd
   git --version
   ```

---

## Project Setup

### Option 1: Fork and Clone (Recommended)

**Why fork?** Track your progress and build your portfolio!

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/60-days-advanced-data-ai.git
   cd 60-days-advanced-data-ai
   ```

3. **Configure Git** (if first time):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### Option 2: Download ZIP

1. Go to the GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract to your preferred location
4. Open terminal in that folder

---

## Package Installation

### Create Virtual Environment

**Why?** Isolates project dependencies from system Python.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### Install Packages

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- **Core**: numpy, pandas, matplotlib, requests
- **Data Engineering**: psycopg2, pymongo, redis, apache-airflow, dbt-core
- **ML/AI**: torch, transformers, scikit-learn
- **Quality**: great-expectations, pytest
- **Cloud**: boto3, google-cloud-storage

**Note**: Installation may take 5-10 minutes.

### Verify Installation

```bash
python3 tools/test_setup.py
```

You should see all checks pass.

---

## Verification

### Quick Test

```bash
# Test Python
python3 -c "import numpy, pandas, torch; print('✅ Core packages work')"

# Test Docker
docker run hello-world

# Test Git
git --version
```

### Full Verification

```bash
python3 tools/test_setup.py
```

Expected output:
```
============================================================
60 DAYS ADVANCED DATA AND AI - SETUP VERIFICATION
============================================================

✅ Python version: 3.11.x
✅ Docker is running
✅ numpy installed
✅ pandas installed
✅ matplotlib installed
✅ requests installed
✅ psycopg2 installed
✅ pymongo installed
✅ redis installed
✅ torch installed
✅ transformers installed

============================================================
✅ ALL CHECKS PASSED (11/11)

You're ready to start Day 1!
Navigate to: days/day-01-postgresql-advanced/
```

---

## Troubleshooting

### Python Issues

**"python: command not found"**
- Try `python3` instead
- Restart terminal after installation
- Check PATH environment variable

**"Permission denied"**
- Use `python3 -m pip install` instead of `pip install`
- Don't use `sudo` with pip (use virtual environment)

### Docker Issues

**"Cannot connect to Docker daemon"**
- Start Docker Desktop
- Wait for Docker to fully start (check system tray)
- Try `docker ps` again

**"Docker not found"**
- Restart terminal after installation
- Check Docker Desktop is installed
- Verify PATH includes Docker

### Virtual Environment Issues

**"venv not activating"**
- Make sure you're in project directory
- Try creating new venv: `python3 -m venv venv`
- On Windows, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**"Package not found after installation"**
- Verify venv is activated (see `(venv)` in prompt)
- Try: `pip list` to see installed packages
- Reinstall: `pip install -r requirements.txt`

### Git Issues

**"Permission denied (publickey)"**
- Use HTTPS instead of SSH for cloning
- Or set up SSH keys: [GitHub SSH Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

**"Git not found"**
- Restart terminal after installation
- Check PATH includes Git

---

## Next Steps

Once setup is complete:

1. ✅ Verify all checks pass
2. 📖 Read [QUICKSTART.md](../QUICKSTART.md)
3. 🚀 Start Day 1: `cd days/day-01-postgresql-advanced`
4. 📚 Review [CURRICULUM.md](./CURRICULUM.md) for overview

---

## Additional Resources

- **Python**: [python.org/downloads](https://www.python.org/downloads/)
- **Docker**: [docs.docker.com/get-docker](https://docs.docker.com/get-docker/)
- **Git**: [git-scm.com/downloads](https://git-scm.com/downloads/)
- **VS Code**: [code.visualstudio.com](https://code.visualstudio.com/)
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

---

## Getting Help

If you're still stuck:
1. Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
2. Search for your error message online
3. Ask in community forums
4. Review official documentation

---

**Setup complete? Start learning!** 🚀

```bash
cd days/day-01-postgresql-advanced
open README.md
```
