# Quick Start Guide - 60 Days Advanced Data and AI

Get started in 10 minutes! 🚀

## Prerequisites

Before starting this bootcamp, you should have:
- ✅ Completed **100 Days of Data and AI** bootcamp (or equivalent)
- ✅ Python 3.11+ installed
- ✅ Docker installed and running
- ✅ Git installed
- ✅ 20GB+ free disk space
- ✅ AWS account (optional for Days 22-23, 55, 59)
- ✅ GitHub account (recommended for tracking progress)

> **📝 Note**: This is an **advanced** bootcamp. If you're new to data engineering or AI, start with the 100 Days bootcamp first.

---

## Step 0: Fork the Repository (Recommended)

**Why fork?** Track your progress, build your portfolio, and practice Git!

### Option A: Fork (Recommended)

1. **Go to**: https://github.com/YOUR-ORG/60-days-advanced-data-ai
2. **Click "Fork"** button (top right)
3. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/60-days-advanced-data-ai.git
   cd 60-days-advanced-data-ai
   ```

### Option B: Download (No Git)

If you don't want to use Git, download the ZIP file from GitHub.

> 💡 **Tip**: Using Git lets you commit your solutions daily and build a portfolio!

---

## Step 1: Install Prerequisites (5 minutes)

### Python 3.11+

Check if installed:
```bash
python3 --version  # Should show 3.11 or higher
```

If not installed:
- **Mac**: `brew install python@3.11`
- **Linux**: `sudo apt install python3.11 python3-pip`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### Docker

Check if installed:
```bash
docker --version
docker ps  # Should not error
```

If not installed:
- **Mac/Windows**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: Follow [official guide](https://docs.docker.com/engine/install/)

### Git

Check if installed:
```bash
git --version
```

If not installed:
- **Mac**: `brew install git`
- **Linux**: `sudo apt install git`
- **Windows**: Download from [git-scm.com](https://git-scm.com/)

---

## Step 2: Setup Environment (3 minutes)

### Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Install Core Packages

```bash
# Install required packages
pip install -r requirements.txt

# Or install core packages manually:
pip install numpy pandas matplotlib requests psycopg2-binary pymongo redis \
    apache-airflow dbt-core great-expectations feast boto3 torch transformers
```

---

## Step 3: Verify Setup (2 minutes)

Run the setup verification script:

```bash
python3 tools/test_setup.py
```

You should see:
```
✅ Python version: 3.11.x
✅ Docker is running
✅ Core packages installed
✅ All checks passed! Ready to start Day 1
```

If you see errors, check [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)

---

## Step 4: Start Learning! (1 minute)

Navigate to Day 1:

```bash
cd days/day-01-postgresql-advanced
```

Open `README.md` in your editor or browser and start learning!

---

## Daily Routine

### Before Starting Each Day:

```bash
# Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Navigate to the day's folder
cd days/day-XX-topic-name
```

### Follow This Pattern:

1. **📖 Read** the `README.md` (15 min)
   - Learn the concepts
   - Study the code examples

2. **💻 Code** the `exercise.py` or `exercise.sql` (40 min)
   - Complete the TODO exercises
   - Try before looking at solutions
   - Run and test your code

3. **✅ Check** the `solution.py` or `solution.sql` (if stuck)
   - Compare your approach
   - Learn from the examples

4. **🎯 Quiz** with `quiz.md` (5 min)
   - Test your understanding
   - Review if needed

### When Done for the Day:

```bash
# Commit your work (if using Git)
git add days/day-XX-topic-name/
git commit -m "Complete Day XX: Topic Name"
git push origin main

# Deactivate virtual environment
deactivate
```

---

## Project Days (Extra Time Needed)

These days are major projects and may take 1.5-2 hours:
- **Day 7**: CDC pipeline
- **Day 14**: Governed platform with Airflow
- **Day 24**: Production pipeline with Airflow + dbt + quality
- **Day 32**: ML model with feature store
- **Day 39**: MLOps pipeline
- **Day 47**: Prompting system
- **Day 54**: RAG system
- **Day 60**: Capstone - Full production system

Don't rush these - they're where you consolidate your learning!

---

## Recommended Tools

### Code Editor (Choose One):

**VS Code** (Recommended)
- Download: [code.visualstudio.com](https://code.visualstudio.com/)
- Install extensions: Python, Docker, Kubernetes
- Free and feature-rich

**PyCharm Professional**
- Download: [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)
- Full-featured IDE for data/ML
- Free for students

**Or use any text editor** + terminal!

---

## Learning Tips

✅ **Code along** - Type examples yourself, don't copy-paste  
✅ **Take breaks** - Split days across sessions if needed  
✅ **Practice more** - Try variations of exercises  
✅ **Use Git** - Commit daily to track progress  
✅ **Build portfolio** - Showcase your work on GitHub  
✅ **Ask questions** - Use community resources  
✅ **Be patient** - Advanced topics take time!

---

## Troubleshooting

### "Python not found"
- Restart terminal after installation
- Try `python3` instead of `python`
- Check PATH environment variable

### "Docker not running"
- Start Docker Desktop
- Check `docker ps` works
- Restart Docker if needed

### Virtual environment not activating
- Make sure you're in the project folder
- Try `python3 -m venv venv` again
- On Windows: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Package installation fails
```bash
# Upgrade pip first
python3 -m pip install --upgrade pip

# Then try again
pip install -r requirements.txt
```

### Still stuck?
1. Check [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
2. Check [docs/SETUP.md](./docs/SETUP.md)
3. Google the error message
4. Ask in community forums

---

## Daily Workflow with Git (Recommended)

### Each Day:

1. **Start the day**:
   ```bash
   cd 60-days-advanced-data-ai
   source venv/bin/activate
   cd days/day-XX-topic-name
   ```

2. **Complete the exercises**

3. **Commit your work**:
   ```bash
   git add days/day-XX-topic-name/
   git commit -m "Complete Day XX: Topic Name"
   git push origin main
   ```

### Sample Commit Messages:
```bash
git commit -m "Complete Day 1: PostgreSQL Advanced"
git commit -m "Complete Day 12: Airflow Basics"
git commit -m "Complete Day 24: Production Pipeline Project"
git commit -m "Complete Day 60: Capstone Production System"
```

### Track Your Progress:
- Your GitHub profile shows daily commits (green squares!)
- Employers can see your learning journey
- You have a backup of all your work
- Build your data engineering portfolio

👉 **See [docs/GIT_SETUP.md](./docs/GIT_SETUP.md) for detailed Git workflow**

---

## What's Next?

After completing all 60 days, you'll be ready for:
- Building production data pipelines
- Deploying ML systems to AWS
- Implementing complete MLOps workflows
- Fine-tuning and serving LLMs
- Building enterprise RAG systems
- Managing infrastructure with Kubernetes and Terraform
- Senior data engineering and ML engineering roles

---

## Need More Help?

- 📖 **Detailed Setup**: [docs/SETUP.md](./docs/SETUP.md)
- 📚 **Curriculum Overview**: [docs/CURRICULUM.md](./docs/CURRICULUM.md)
- 🆘 **Troubleshooting**: [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
- 🔄 **Migration Guide**: [docs/MIGRATION_GUIDE.md](./docs/MIGRATION_GUIDE.md)
- 🏠 **Main README**: [README.md](./README.md)

---

**Ready? Let's start with Day 1!** 🚀

```bash
cd days/day-01-postgresql-advanced
open README.md  # or 'code README.md' for VS Code
```

Master advanced data and AI skills! 💪
