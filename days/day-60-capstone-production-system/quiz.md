# Day 60: Capstone Production System - Knowledge Quiz

## Instructions
Answer all questions to test your understanding of the complete Intelligent Customer Analytics Platform and production system integration concepts.

**Time Limit**: 15 minutes  
**Passing Score**: 80% (24/30 questions)

---

## Section 1: System Architecture & Integration (10 questions)

### Question 1
What is the primary purpose of the Intelligent Customer Analytics Platform capstone project?

A) To demonstrate individual technology skills  
B) To integrate all 60 days of curriculum into a production-ready system  
C) To build a simple customer database  
D) To create a basic web application  

**Answer**: B

**Explanation**: The capstone project integrates all technologies from the 60-day curriculum (data engineering, MLOps, GenAI, infrastructure) into a cohesive production system.

---

### Question 2
Which architectural pattern is used for handling real-time and batch data processing?

A) Microservices Architecture  
B) Event-Driven Architecture  
C) Lambda Architecture  
D) Layered Architecture  

**Answer**: C

**Explanation**: Lambda Architecture combines speed layer (real-time), batch layer (historical), and serving layer (combined views) for comprehensive data processing.

---

### Question 3
What are the three main layers in the system architecture?

A) Presentation, Business, Data  
B) Speed, Batch, Serving  
C) Data, Processing, Serving  
D) Input, Transform, Output  

**Answer**: C

**Explanation**: The architecture consists of Data Layer (sources), Processing Layer (Kafka, Airflow, dbt), and Serving Layer (APIs, models, features).

---

### Question 4
Which databases are used in the data layer and for what purposes?

A) Only PostgreSQL for all data  
B) PostgreSQL (customers), MongoDB (events), Redis (caching)  
C) MySQL (customers), Cassandra (events), Memcached (caching)  
D) Oracle (customers), Neo4j (events), Hazelcast (caching)  

**Answer**: B

**Explanation**: PostgreSQL stores structured customer data, MongoDB handles event/session data, and Redis provides real-time caching and feature serving.

---

### Question 5
What is the role of Debezium in the system architecture?

A) Data transformation  
B) Change Data Capture (CDC)  
C) Model serving  
D) API gateway  

**Answer**: B

**Explanation**: Debezium captures changes from PostgreSQL and streams them to Kafka for real-time data synchronization across the platform.

---

### Question 6
Which orchestration tools are integrated in the platform?

A) Only Airflow  
B) Airflow and dbt  
C) Airflow, dbt, and Great Expectations  
D) Jenkins and GitLab CI  

**Answer**: C

**Explanation**: Airflow orchestrates workflows, dbt handles data transformations, and Great Expectations ensures data quality validation.

---

### Question 7
What is the purpose of the feature store (Feast) in the ML pipeline?

A) Store ML models  
B) Centralized feature management and serving  
C) Data warehouse storage  
D) API documentation  

**Answer**: B

**Explanation**: Feast provides centralized feature management, serving both online (real-time) and offline (batch) features for ML models.

---

### Question 8
Which GenAI technique is used for customer insights generation?

A) Fine-tuning  
B) Prompt engineering only  
C) RAG (Retrieval-Augmented Generation)  
D) Model quantization  

**Answer**: C

**Explanation**: RAG combines vector database retrieval with LLM generation to provide contextual customer insights based on business knowledge.

---

### Question 9
What monitoring stack is used for observability?

A) ELK Stack  
B) Prometheus and Grafana  
C) Splunk  
D) New Relic  

**Answer**: B

**Explanation**: Prometheus collects metrics from the application, and Grafana provides visualization dashboards for monitoring system health and performance.

---

### Question 10
Which infrastructure tools are used for deployment and management?

A) Docker only  
B) Kubernetes only  
C) Docker, Kubernetes, and Terraform  
D) VMware and Ansible  

**Answer**: C

**Explanation**: Docker containerizes applications, Kubernetes orchestrates containers, and Terraform manages infrastructure as code.

---

## Section 2: Data Engineering & Processing (8 questions)

### Question 11
What is the primary benefit of using CDC (Change Data Capture) with Debezium?

A) Faster queries  
B) Real-time data synchronization  
C) Data compression  
D) Schema validation  

**Answer**: B

**Explanation**: CDC captures database changes in real-time, enabling immediate data synchronization across systems without batch processing delays.

---

### Question 12
How does the system handle both real-time and batch processing requirements?

A) Separate systems for each  
B) Lambda architecture with speed and batch layers  
C) Only real-time processing  
D) Only batch processing  

**Answer**: B

**Explanation**: Lambda architecture processes real-time data through Kafka Streams (speed layer) and historical data through Airflow+dbt (batch layer).

---

### Question 13
What is the role of dbt in the data pipeline?

A) Data ingestion  
B) Data transformation and testing  
C) Data visualization  
D) Data storage  

**Answer**: B

**Explanation**: dbt handles SQL-based data transformations, includes built-in testing capabilities, and generates documentation for data models.

---

### Question 14
Which data quality tool is integrated for validation?

A) Apache Griffin  
B) Great Expectations  
C) Deequ  
D) Monte Carlo  

**Answer**: B

**Explanation**: Great Expectations provides data quality validation, profiling, and monitoring with integration into the Airflow pipeline.

---

### Question 15
How are customer features cached for real-time serving?

A) PostgreSQL  
B) MongoDB  
C) Redis  
D) Elasticsearch  

**Answer**: C

**Explanation**: Redis provides low-latency caching for frequently accessed customer features, enabling sub-100ms API response times.

---

### Question 16
What format is recommended for storing large-scale analytics data?

A) CSV files  
B) JSON files  
C) Parquet files  
D) XML files  

**Answer**: C

**Explanation**: Parquet provides columnar storage with compression and efficient querying for analytics workloads.

---

### Question 17
How does the system ensure data lineage tracking?

A) Manual documentation  
B) DataHub integration  
C) Custom scripts  
D) Database logs  

**Answer**: B

**Explanation**: DataHub provides automated data lineage tracking, cataloging, and metadata management across the entire data pipeline.

---

### Question 18
What is the purpose of implementing CQRS (Command Query Responsibility Segregation)?

A) Security enhancement  
B) Separate read/write models for optimal performance  
C) Data backup  
D) User authentication  

**Answer**: B

**Explanation**: CQRS separates read and write operations, allowing optimization of each for their specific use cases and improving overall system performance.

---

## Section 3: ML & GenAI Integration (7 questions)

### Question 19
Which ML framework is used for experiment tracking and model registry?

A) Weights & Biases  
B) Neptune  
C) MLflow  
D) Comet  

**Answer**: C

**Explanation**: MLflow provides experiment tracking, model registry, and model serving capabilities integrated throughout the ML pipeline.

---

### Question 20
What is the primary ML use case demonstrated in the platform?

A) Image classification  
B) Customer churn prediction  
C) Natural language processing  
D) Time series forecasting  

**Answer**: B

**Explanation**: The platform focuses on customer churn prediction as the main ML use case, with additional recommendation scoring and segmentation.

---

### Question 21
How are ML model features served in production?

A) Direct database queries  
B) Feature store (Feast) with online serving  
C) API calls to external services  
D) File-based lookups  

**Answer**: B

**Explanation**: Feast feature store provides both online (real-time) and offline (batch) feature serving with consistent feature definitions.

---

### Question 22
What GenAI technique enhances the recommendation system?

A) Model fine-tuning  
B) Prompt engineering with DSPy  
C) Model quantization  
D) Transfer learning  

**Answer**: B

**Explanation**: DSPy provides systematic prompt optimization for generating better customer insights and recommendations.

---

### Question 23
Which vector database is used for the RAG system?

A) Pinecone  
B) Weaviate  
C) ChromaDB  
D) Milvus  

**Answer**: C

**Explanation**: ChromaDB provides vector storage and similarity search capabilities for the RAG-based customer insights system.

---

### Question 24
How does the system handle A/B testing for ML models?

A) Manual deployment  
B) Gradual rollout with traffic splitting  
C) Blue-green deployment  
D) Canary releases only  

**Answer**: B

**Explanation**: The platform implements A/B testing with gradual traffic splitting to safely evaluate new model versions against existing ones.

---

### Question 25
What is the purpose of model monitoring and drift detection?

A) Cost optimization  
B) Detect performance degradation over time  
C) Security compliance  
D) Data backup  

**Answer**: B

**Explanation**: Model monitoring detects when model performance degrades due to data drift, concept drift, or other factors, triggering retraining.

---

## Section 4: Production & Operations (5 questions)

### Question 26
Which auto-scaling mechanisms are implemented in Kubernetes?

A) Only Horizontal Pod Autoscaler (HPA)  
B) Only Vertical Pod Autoscaler (VPA)  
C) Both HPA and VPA  
D) Manual scaling only  

**Answer**: C

**Explanation**: The system uses both HPA (horizontal scaling based on CPU/memory/custom metrics) and VPA (vertical scaling for right-sizing resources).

---

### Question 27
What is the target API response time for ML predictions?

A) < 50ms  
B) < 100ms  
C) < 500ms  
D) < 1000ms  

**Answer**: B

**Explanation**: The system targets sub-100ms p95 latency for ML predictions to ensure responsive user experience.

---

### Question 28
Which cost optimization strategies are implemented?

A) Reserved instances only  
B) Spot instances only  
C) Spot instances, auto-scaling, and resource right-sizing  
D) Manual resource management  

**Answer**: C

**Explanation**: The platform uses multiple cost optimization strategies including spot instances for batch workloads, auto-scaling policies, and VPA for right-sizing.

---

### Question 29
What security measures are implemented in production?

A) Basic authentication only  
B) HTTPS, JWT authentication, RBAC, and encryption  
C) No security measures  
D) Password-based authentication only  

**Answer**: B

**Explanation**: Production security includes HTTPS/TLS, JWT token authentication, role-based access control (RBAC), and end-to-end encryption.

---

### Question 30
How is system observability achieved?

A) Log files only  
B) Metrics only  
C) Metrics, logs, and distributed tracing  
D) Manual monitoring  

**Answer**: C

**Explanation**: Comprehensive observability includes Prometheus metrics, structured logging, and distributed tracing for complete system visibility.

---

## Answer Key

1. B - Integration of all curriculum technologies
2. C - Lambda Architecture for real-time and batch processing
3. C - Data, Processing, Serving layers
4. B - PostgreSQL, MongoDB, Redis for different purposes
5. B - Change Data Capture (CDC)
6. C - Airflow, dbt, and Great Expectations
7. B - Centralized feature management and serving
8. C - RAG (Retrieval-Augmented Generation)
9. B - Prometheus and Grafana
10. C - Docker, Kubernetes, and Terraform
11. B - Real-time data synchronization
12. B - Lambda architecture implementation
13. B - Data transformation and testing
14. B - Great Expectations for data quality
15. C - Redis for feature caching
16. C - Parquet for analytics storage
17. B - DataHub for lineage tracking
18. B - Separate read/write models for performance
19. C - MLflow for ML lifecycle management
20. B - Customer churn prediction
21. B - Feature store with online serving
22. B - Prompt engineering with DSPy
23. C - ChromaDB vector database
24. B - Gradual rollout with traffic splitting
25. B - Detect performance degradation
26. C - Both HPA and VPA auto-scaling
27. B - Sub-100ms API response time
28. C - Multiple cost optimization strategies
29. B - Comprehensive security measures
30. C - Metrics, logs, and distributed tracing

---

## Scoring Guide

- **30/30 (100%)**: Exceptional - Complete mastery of production system integration
- **27-29 (90-97%)**: Excellent - Strong understanding with minor gaps
- **24-26 (80-87%)**: Good - Solid grasp of core concepts
- **21-23 (70-77%)**: Satisfactory - Basic understanding, review recommended
- **Below 21 (<70%)**: Needs Improvement - Significant review required

---

## Key Takeaways

After completing this quiz, you should understand:

✅ **System Architecture**: How to integrate multiple technologies into a cohesive platform  
✅ **Data Engineering**: Real-time and batch processing with proper orchestration  
✅ **ML Operations**: Feature stores, model serving, and monitoring in production  
✅ **GenAI Integration**: RAG systems and prompt optimization for business insights  
✅ **Production Deployment**: Kubernetes, monitoring, security, and cost optimization  
✅ **Operational Excellence**: Observability, auto-scaling, and reliability patterns  

**Congratulations on completing the 60 Days Advanced Data and AI curriculum!** 🎉

You now have the knowledge and skills to build production-ready data and AI systems that can handle enterprise-scale workloads and deliver real business value.