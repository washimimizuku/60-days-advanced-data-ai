# 60 Days of Advanced Data and AI - Curriculum

## Phase 1: Production Data Engineering (Days 1-14)

### Week 1: Advanced Data Systems (Days 1-7)
- Day 1: PostgreSQL advanced - Indexing and query optimization
- Day 2: NoSQL databases - MongoDB patterns
- Day 3: NoSQL databases - Redis for caching
- Day 4: Data warehouses - Snowflake specifics
- Day 5: CDC (Change Data Capture) - Debezium
- Day 6: Advanced Kafka - Partitions and replication
- Day 7: **Project** - Real-time CDC pipeline

### Week 2: Governance & Orchestration Basics (Days 8-14)
- Day 8: Data catalogs - Datahub, Amundsen
- Day 9: Data lineage tracking
- Day 10: Data privacy - GDPR, PII handling
- Day 11: Access control - RBAC, row-level security
- Day 12: Apache Airflow basics - DAGs, operators, scheduling
- Day 13: dbt basics - Models, sources, tests
- Day 14: **Project** - Governed data platform with Airflow

---

## Phase 2: Data Orchestration & Quality (Days 15-24)

### Week 3: Production Orchestration (Days 15-21)
- Day 15: Airflow production patterns - Dynamic DAGs, task groups
- Day 16: Airflow at scale - Executors, monitoring, error handling
- Day 17: dbt deep dive - Materializations, incremental models
- Day 18: dbt advanced - Macros, packages, snapshots (SCD Type 2)
- Day 19: Data quality in production - Great Expectations automation
- Day 20: Data observability - Monitoring, anomaly detection, alerting
- Day 21: Testing strategies for data pipelines

### Week 4: AWS Data Services (Days 22-24)
- Day 22: AWS Glue & Data Catalog - Crawlers, ETL jobs, Athena
- Day 23: AWS Kinesis & streaming - Data Streams, Firehose, Analytics
- Day 24: **Project** - Production pipeline with Airflow + dbt + quality

---

## Phase 3: Advanced ML & MLOps (Days 25-39)

### Week 5: Advanced ML (Days 25-31)
- Day 25: Feature stores - Feast, feature engineering at scale
- Day 26: Advanced feature engineering techniques
- Day 27: Time series forecasting - ARIMA, Prophet
- Day 28: Anomaly detection techniques
- Day 29: Recommendation systems
- Day 30: Ensemble methods - Stacking, blending
- Day 31: Model explainability - SHAP, LIME

### Week 6: Production MLOps (Days 32-39)
- Day 32: **Project** - Production ML model with feature store
- Day 33: Model serving at scale - vLLM, TGI
- Day 34: A/B testing for ML models
- Day 35: Model versioning - DVC
- Day 36: CI/CD for ML - GitHub Actions
- Day 37: Feature monitoring and drift detection
- Day 38: AutoML - H2O, AutoGluon
- Day 39: **Project** - Complete MLOps pipeline with Airflow

---

## Phase 4: Advanced GenAI & LLMs (Days 40-53)

### Week 7: LLM Deep Dive (Days 40-46)
- Day 40: **Checkpoint** - ML systems review
- Day 41: Transformer architecture deep dive
- Day 42: Attention mechanisms in detail
- Day 43: Tokenization - BPE, WordPiece
- Day 44: LLM training stages
- Day 45: Advanced prompt engineering - DSPy
- Day 46: Prompt security - Injection attacks

### Week 8: Production LLMs (Days 47-53)
- Day 47: **Project** - Advanced prompting system
- Day 48: Fine-tuning deep dive - LoRA, QLoRA
- Day 49: RLHF and DPO
- Day 50: Quantization - GPTQ, AWQ, GGUF
- Day 51: LLM serving optimization
- Day 52: Advanced RAG - Hybrid search, re-ranking
- Day 53: RAG evaluation - RAGAS in depth

---

## Phase 5: Infrastructure & DevOps (Days 54-60)

### Week 9: Cloud & Infrastructure (Days 54-60)
- Day 54: **Project** - Production RAG system with orchestration
- Day 55: AWS deep dive - SageMaker, ECS, EMR
- Day 56: Kubernetes for ML/Data workloads
- Day 57: Terraform for data infrastructure
- Day 58: Monitoring - Prometheus, Grafana, data observability
- Day 59: Cost optimization strategies - AWS, compute, storage
- Day 60: **Capstone Project** - Full production system

---

## Key Changes from Previous Version

### Orchestration Moved Earlier ⭐
- **Before**: Airflow Day 51, dbt Day 53
- **After**: Airflow Day 12, dbt Day 13
- **Impact**: All projects after Day 14 use proper orchestration

### Feature Stores Moved to ML ⭐
- **Before**: Day 12 (too early)
- **After**: Day 25 (with ML section)
- **Impact**: Introduced when ML context is understood

### AWS Data Services Grouped ⭐
- **Before**: Split across Days 45 and 58-59
- **After**: Days 22-23 (with data engineering)
- **Impact**: Logical grouping with data systems

### Better Project Progression ⭐
- Day 7: CDC pipeline (no orchestration yet)
- Day 14: Governed platform with Airflow basics
- Day 24: Full pipeline with Airflow + dbt + quality
- Day 32: ML model with feature store
- Day 39: MLOps pipeline with Airflow
- Day 47: Prompting system
- Day 54: RAG system with orchestration
- Day 60: Full production system

---

## Learning Progression

### Days 1-14: Foundation
Learn data systems, governance, and orchestration basics

### Days 15-24: Production Patterns
Master Airflow, dbt, data quality, and AWS data services

### Days 25-39: ML & MLOps
Advanced ML techniques with production orchestration

### Days 40-53: GenAI & LLMs
LLM internals, fine-tuning, and RAG systems

### Days 54-60: Infrastructure
Cloud infrastructure, monitoring, and final capstone

---

## Projects Timeline

| Day | Project | Uses Orchestration |
|-----|---------|-------------------|
| 7 | CDC pipeline | No (learning basics) |
| 14 | Governed platform | Yes (Airflow basics) |
| 24 | Production pipeline | Yes (Airflow + dbt) |
| 32 | ML model | Yes (with feature store) |
| 39 | MLOps pipeline | Yes (full orchestration) |
| 47 | Prompting system | Optional |
| 54 | RAG system | Yes (orchestrated) |
| 60 | Full production system | Yes (everything) |

**Result**: 6 out of 8 projects use orchestration (vs. 1 out of 8 before)
