# 60 Days of Advanced Data and AI

An advanced 60-day program (1 hour/day) for production-ready Data Engineering, MLOps, and GenAI skills.

> **📝 Prerequisites**: Complete **100 Days of Data and AI** bootcamp first. This is an advanced program building on those foundations.

## Overview

**Duration**: 60 days × 1 hour/day = 60 hours  
**Prerequisites**: Completed **100 Days of Data and AI** or equivalent  
**Outcome**: Production-ready skills for senior data/ML engineering roles

**What's Included**:
- 📚 60 comprehensive lessons with production patterns
- 💻 180+ hands-on exercises
- ✅ 300+ quiz questions
- 🎯 8 major projects (Days 7, 14, 24, 32, 39, 47, 54, 60)
- 🏗️ Real-world production systems

---

## 🚀 Quick Start

**New to this bootcamp?** Get started in 10 minutes:

1. **Fork this repository** (recommended for portfolio)
2. **Clone**: `git clone https://github.com/YOUR-USERNAME/60-days-advanced-data-ai.git`
3. **Setup**: Follow [QUICKSTART.md](./QUICKSTART.md)
4. **Start**: `cd days/day-01-postgresql-advanced`

👉 **See [QUICKSTART.md](./QUICKSTART.md) for step-by-step setup**

---

## Program Structure

### Phase 1: Production Data Engineering (Days 1-14)
Advanced data systems, governance, and orchestration basics.

### Phase 2: Data Orchestration & Quality (Days 15-24) ⭐ NEW
Production Airflow, dbt, data quality, observability, and AWS data services.

### Phase 3: Advanced ML & MLOps (Days 25-39)
Feature stores, production ML, advanced techniques, and complete MLOps.

### Phase 4: Advanced GenAI & LLMs (Days 40-53)
LLM internals, fine-tuning, and production RAG.

### Phase 5: Infrastructure & DevOps (Days 54-60)
Cloud infrastructure, Kubernetes, monitoring, and capstone.

---

## What You'll Learn

### Advanced Data Engineering
- PostgreSQL advanced (indexing, query optimization)
- NoSQL databases (MongoDB, Redis)
- Data warehouses (Snowflake specifics)
- CDC (Change Data Capture) with Debezium
- Advanced Kafka (partitions, replication)
- Data catalogs (Datahub, Amundsen)
- Data lineage tracking
- Data privacy (GDPR, PII handling)
- Access control (RBAC, row-level security)

### Data Orchestration & Quality ⭐ NEW
- Apache Airflow (basics to production patterns)
- dbt (models, tests, macros, snapshots)
- Data quality in production (Great Expectations)
- Data observability (monitoring, anomaly detection)
- Testing strategies for data pipelines
- AWS Glue & Data Catalog
- AWS Kinesis & streaming
- Production pipeline integration

### Advanced ML & MLOps
- Feature stores (Feast) ⭐ MOVED HERE
- Advanced feature engineering
- Time series forecasting (ARIMA, Prophet)
- Anomaly detection techniques
- Recommendation systems
- Ensemble methods
- Model explainability (SHAP, LIME)
- Model serving at scale (vLLM, TGI)
- A/B testing for ML
- Model versioning (DVC)
- CI/CD for ML (GitHub Actions)
- Feature monitoring and drift
- AutoML (H2O, AutoGluon)

### Advanced GenAI & LLMs
- Transformer architecture deep dive
- Attention mechanisms in detail
- Tokenization (BPE, WordPiece)
- LLM training stages
- Advanced prompt engineering (DSPy)
- Prompt security (injection attacks)
- Fine-tuning deep dive (full fine-tuning, LoRA, QLoRA)
- RLHF and DPO
- Quantization (GPTQ, AWQ, GGUF)
- LLM serving optimization
- Advanced RAG (hybrid search, re-ranking)
- RAG evaluation (RAGAS in depth)

### Infrastructure & DevOps
- AWS deep dive (SageMaker, ECS, Lambda)
- Kubernetes for ML/Data
- Terraform for data infrastructure
- Monitoring (Prometheus, Grafana)
- Cost optimization strategies

---

## Weekly Breakdown

### Week 1-2: Production Data Engineering (Days 1-14)
Master advanced data systems, governance, and orchestration basics.

### Week 3-4: Data Orchestration & Quality (Days 15-24) ⭐ NEW
Production Airflow, dbt, data quality, and AWS data services.

### Week 5-6: Advanced ML & MLOps (Days 25-39)
Feature stores, ML techniques, and complete MLOps pipelines.

### Week 7-8: Advanced GenAI & LLMs (Days 40-53)
Deep dive into LLM internals, fine-tuning, and production RAG.

### Week 9: Infrastructure & Capstone (Days 54-60)
Cloud infrastructure, monitoring, and final production system.

---

## 📊 Daily Structure

**Regular Days** (1 hour):
- **📖 Theory** (15 min) - Production patterns and concepts
- **💻 Exercise** (40 min) - Hands-on implementation
- **✅ Quiz** (5 min) - Test understanding

**Project Days** (1.5-2 hours):
- Days 7, 14, 24, 32, 39, 47, 54, 60
- Build complete production systems
- Integrate multiple technologies

---

## 🎯 Major Projects

| Day | Project | Technologies |
|-----|---------|-------------|
| 7 | Real-time CDC pipeline | Debezium, Kafka, PostgreSQL |
| 14 | Governed data platform | Airflow, Data Catalog, Lineage |
| 24 | Production pipeline ⭐ | Airflow + dbt + Great Expectations |
| 32 | ML model with features | Feast, MLflow, Production ML |
| 39 | Complete MLOps pipeline | Airflow, DVC, CI/CD, Monitoring |
| 47 | Advanced prompting system | DSPy, LLMs, Prompt Engineering |
| 54 | Production RAG system | RAG, Vector DB, Orchestration |
| 60 | **Capstone** - Full system | Everything integrated |

**Result**: 6/8 projects use orchestration (vs 1/8 in old structure)

---

## Prerequisites

### Required Knowledge (from 100 Days)
- Data formats and table formats
- Spark, Kafka, Airflow basics
- FastAPI fundamentals
- ML and PyTorch basics
- LLM and RAG basics
- Docker and Git basics

### Required Software
- Everything from 100 Days bootcamp
- Kubernetes (minikube or Docker Desktop)
- Terraform
- AWS CLI
- Additional 20GB+ disk space

### Recommended
- AWS account with credits
- GPU for LLM fine-tuning
- 32GB+ RAM

---

## How to Use This Bootcamp

### Sequential Approach (Recommended)
Complete days 1-50 in order after finishing 100 Days bootcamp.

### Track-Based Approach
Focus on specific areas:
- **Data Engineering**: Days 1-14
- **Orchestration & Quality**: Days 15-24 ⭐ NEW
- **MLOps**: Days 25-39
- **GenAI**: Days 40-53
- **Infrastructure**: Days 54-60

### Project-Driven Approach
Pick a comprehensive project (1-9) and complete relevant days.

---

## 📁 Project Structure

```
60-days-advanced-data-ai/
├── README.md                    # Start here
├── QUICKSTART.md                # 10-minute setup guide
├── requirements.txt             # Python packages
│
├── docs/                        # 📚 Documentation
│   ├── CURRICULUM.md            # Day-by-day breakdown
│   ├── SETUP.md                 # Detailed setup guide
│   ├── TROUBLESHOOTING.md       # Common issues & fixes
│   ├── GIT_SETUP.md            # Git workflow guide
│   ├── MIGRATION_GUIDE.md       # From 50 to 60 days
│   └── archive/                 # Historical docs
│
├── tools/                       # 🛠️ Utilities
│   ├── test_setup.py            # Verify installation
│   └── verify_structure.py      # Check structure
│
├── data/                        # 📊 Data files
│   ├── raw/                     # Original data
│   └── processed/               # Processed data
│
├── resources/                   # 📖 Additional resources
│
└── days/                        # 📖 60 Daily Lessons
    ├── day-01-postgresql-advanced/
    │   ├── README.md            # Lesson
    │   ├── exercise.sql         # Practice
    │   ├── solution.sql         # Solutions
    │   └── quiz.md              # Quiz
    ├── day-12-airflow-basics/
    ├── day-24-project-production-pipeline/
    └── day-60-capstone-production-system/
```

---

## 💡 How to Use Each Day

1. **📖 Read** the lesson (README.md) - 15 minutes
2. **💻 Code** the exercises (exercise.py/sql) - 40 minutes
3. **✅ Check** solutions if stuck (solution.py/sql)
4. **🎯 Quiz** yourself (quiz.md) - 5 minutes
5. **📝 Commit** your work (if using Git)

---

## 🎓 After Completion

Upon finishing 60 Days Advanced, you'll be ready to:

✅ **Orchestrate** data pipelines with Airflow and dbt ⭐  
✅ **Implement** data quality and observability ⭐  
✅ **Deploy** production systems to AWS  
✅ **Build** complete MLOps pipelines  
✅ **Fine-tune** and serve LLMs at scale  
✅ **Create** enterprise-grade RAG systems  
✅ **Manage** infrastructure with Kubernetes and Terraform  
✅ **Apply** for senior data/ML engineering roles

---

## Folder Structure

## 📝 Track Your Progress with Git

**Recommended**: Fork this repository and commit your solutions daily!

### Benefits:
- 🎯 **Portfolio** - Show employers your learning journey
- 💾 **Backup** - Never lose your work
- 📈 **Motivation** - See your progress with GitHub's green squares
- 🛠️ **Git Practice** - Learn version control alongside data engineering

### Quick Start:
1. **Fork** this repository on GitHub
2. **Clone** your fork
3. **Complete** each day's exercises
4. **Commit** daily: `git commit -m "Complete Day X: Topic"`
5. **Push** to GitHub: `git push origin main`

👉 **See [docs/GIT_SETUP.md](./docs/GIT_SETUP.md) for complete workflow**

---

## 💡 Tips for Success

✅ **Code along** - Type examples yourself, don't copy-paste  
✅ **Take breaks** - Split days across sessions if needed  
✅ **Practice more** - Try variations of exercises  
✅ **Use Git** - Commit daily to track progress  
✅ **Build portfolio** - Showcase your work on GitHub  
✅ **Ask questions** - Use community resources  
✅ **Be patient** - Advanced topics take time!  
✅ **Focus on projects** - They consolidate your learning

---

## 📚 Documentation & Resources

- 🚀 **[QUICKSTART.md](./QUICKSTART.md)** - Get started in 10 minutes
- 📖 **[docs/CURRICULUM.md](./docs/CURRICULUM.md)** - Complete day-by-day breakdown
- 🔧 **[docs/SETUP.md](./docs/SETUP.md)** - Detailed setup instructions
- 🆘 **[docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)** - Common issues & solutions
- 🔄 **[docs/MIGRATION_GUIDE.md](./docs/MIGRATION_GUIDE.md)** - Migrating from 50-day version
- 🤝 **[docs/GIT_SETUP.md](./docs/GIT_SETUP.md)** - Git workflow guide

---

## 🆘 Getting Help

- Check [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) for common issues
- Review [docs/SETUP.md](./docs/SETUP.md) for detailed setup
- Search for error messages online
- Ask in community forums
- Review official documentation

---

## 🎯 Learning Path

```
100 Days of Data and AI
         ↓
Build standalone projects
         ↓
60 Days of Advanced Data and AI  ← You are here
         ↓
Build comprehensive projects
         ↓
Production deployment
         ↓
Senior roles
```

---

## Key Differences from 100 Days

| Aspect | 100 Days | 50 Days Advanced |
|--------|----------|------------------|
| Focus | Fundamentals | Production |
| Depth | Broad coverage | Deep dives |
| Projects | Standalone | Comprehensive |
| Deployment | Local | Cloud (AWS) |
| Scale | Small datasets | Production scale |
| Complexity | Simple | Complex systems |

---

## Learning Path

```
100 Days of Data and AI
         ↓
Build standalone projects (data-engineering + ai-development)
         ↓
60 Days of Advanced Data and AI
         ↓
Build comprehensive projects (1-9)
         ↓
Production deployment
```

---

## Getting Started

1. **Complete 100 Days bootcamp first**
2. **Build 2-3 standalone projects**
3. **Set up AWS account**
4. **Install Kubernetes (minikube)**
5. **Start Day 1**

---

## Community & Support

- **Discord**: Advanced channel
- **GitHub Discussions**: Technical deep dives
- **Monthly Office Hours**: Production Q&A
- **Project Reviews**: Get feedback on comprehensive projects

---

## Certification Path (Optional)

After completing both bootcamps:
- AWS Certified Solutions Architect
- AWS Certified Machine Learning Specialty
- Kubernetes certifications (CKA, CKAD)
- Databricks certifications

---

## Next Steps

After completing this bootcamp:
1. Build all 9 comprehensive projects
2. Deploy to production
3. Contribute to open source
4. Build your own products
5. Apply for senior roles

---

## License

MIT License - Free to use and modify

---

## Acknowledgments

Built to support the [Data & AI Portfolio](../) projects.
Advanced track for production-ready skills.
