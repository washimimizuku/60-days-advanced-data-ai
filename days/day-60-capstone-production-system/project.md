# Day 60: Capstone Project - Intelligent Customer Analytics Platform

## Project Overview

Build a comprehensive **Intelligent Customer Analytics Platform** that integrates all technologies and concepts from the 60-day curriculum. This production-ready system demonstrates mastery of data engineering, MLOps, GenAI, and cloud infrastructure through a real-world e-commerce analytics use case.

**Business Context**: TechCommerce needs a scalable platform to process customer data, predict behavior, generate personalized recommendations, and provide AI-powered insights to business teams.

## Learning Objectives

By completing this capstone project, you will demonstrate:
- **System Integration**: Combining 15+ technologies into a cohesive production platform
- **Architecture Design**: Making informed decisions about scalability, reliability, and cost
- **Production Deployment**: Implementing monitoring, security, and operational best practices
- **Business Value**: Translating technical capabilities into measurable business outcomes
- **End-to-End Ownership**: Managing the complete data and AI lifecycle

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Processing Layer │───▶│  Serving Layer  │
│                 │    │                  │    │                 │
│ • PostgreSQL    │    │ • Kafka Streams  │    │ • Feature Store │
│ • MongoDB       │    │ • Airflow DAGs   │    │ • ML Models     │
│ • Redis         │    │ • dbt Transform  │    │ • RAG System    │
│ • Web Events    │    │ • Spark Jobs     │    │ • APIs          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │ Infrastructure   │
                       │                  │
                       │ • Kubernetes     │
                       │ • Terraform      │
                       │ • Monitoring     │
                       │ • Cost Mgmt      │
                       └──────────────────┘
```

### Technology Stack Integration

#### Data Layer (Days 1-11)
- **PostgreSQL**: Customer profiles, transactions, product catalog
- **MongoDB**: Event data, user sessions, content metadata
- **Redis**: Real-time caching, session management, feature serving
- **Debezium**: Change data capture for real-time sync
- **Kafka**: Event streaming and data distribution

#### Orchestration Layer (Days 12-24)
- **Airflow**: Workflow orchestration with dynamic DAGs
- **dbt**: Data transformations with testing and documentation
- **Great Expectations**: Data quality validation and monitoring
- **DataHub**: Data cataloging and lineage tracking

#### ML Layer (Days 25-39)
- **Feast**: Feature store for ML feature management
- **MLflow**: Experiment tracking and model registry
- **Scikit-learn/XGBoost**: ML models for prediction
- **FastAPI**: Model serving with auto-scaling

#### GenAI Layer (Days 41-54)
- **LangChain**: RAG system for customer insights
- **OpenAI/Hugging Face**: LLM integration for text generation
- **ChromaDB**: Vector database for semantic search
- **DSPy**: Systematic prompt optimization

#### Infrastructure Layer (Days 55-60)
- **AWS EKS**: Kubernetes cluster management
- **Terraform**: Infrastructure as code
- **Prometheus/Grafana**: Monitoring and alerting
- **AWS Cost Explorer**: Cost optimization and FinOps

## Implementation Phases

### Phase 1: Data Infrastructure Setup (25 minutes)

#### 1.1 Infrastructure Provisioning
```bash
# Initialize Terraform infrastructure
cd infrastructure/
terraform init
terraform plan -var-file="production.tfvars"
terraform apply

# Verify Kubernetes cluster
kubectl get nodes
kubectl get namespaces
```

#### 1.2 Data Sources Configuration
```python
# PostgreSQL setup with sample data
import psycopg2
from faker import Faker
import pandas as pd

def setup_customer_database():
    """Initialize PostgreSQL with customer data"""
    conn = psycopg2.connect(
        host="postgres.customer-analytics.local",
        database="customers",
        user="analytics_user",
        password="secure_password"
    )
    
    # Create tables with proper indexing
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id UUID PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            first_name VARCHAR(100),
            last_name VARCHAR(100),
            registration_date TIMESTAMP,
            segment VARCHAR(50),
            lifetime_value DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX idx_customers_segment ON customers(segment);
        CREATE INDEX idx_customers_ltv ON customers(lifetime_value);
        CREATE INDEX idx_customers_registration ON customers(registration_date);
    """)
    
    # Insert sample data
    fake = Faker()
    for _ in range(10000):
        cursor.execute("""
            INSERT INTO customers (customer_id, email, first_name, last_name, 
                                 registration_date, segment, lifetime_value)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            fake.uuid4(),
            fake.email(),
            fake.first_name(),
            fake.last_name(),
            fake.date_time_between(start_date='-2y', end_date='now'),
            fake.random_element(['Premium', 'Standard', 'Basic']),
            fake.random_number(digits=4, fix_len=False)
        ))
    
    conn.commit()
    conn.close()
```

#### 1.3 CDC and Streaming Setup
```python
# Debezium connector configuration
debezium_config = {
    "name": "customer-postgres-connector",
    "config": {
        "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
        "database.hostname": "postgres.customer-analytics.local",
        "database.port": "5432",
        "database.user": "debezium_user",
        "database.password": "debezium_password",
        "database.dbname": "customers",
        "database.server.name": "customer-analytics",
        "table.include.list": "public.customers,public.transactions,public.events",
        "plugin.name": "pgoutput",
        "slot.name": "debezium_slot",
        "publication.name": "debezium_publication"
    }
}

# Kafka consumer for real-time processing
from kafka import KafkaConsumer
import json

def process_customer_events():
    """Process real-time customer events from Kafka"""
    consumer = KafkaConsumer(
        'customer-analytics.public.customers',
        'customer-analytics.public.transactions',
        bootstrap_servers=['kafka:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    for message in consumer:
        event = message.value
        
        # Update Redis cache for real-time features
        if event['op'] in ['c', 'u']:  # Create or Update
            update_realtime_features(event['after'])
        
        # Send to feature store
        update_feature_store(event)
```

#### 1.4 Deliverables for Phase 1
- [ ] Terraform infrastructure deployed to AWS
- [ ] PostgreSQL, MongoDB, Redis instances running
- [ ] Debezium CDC capturing changes to Kafka
- [ ] Sample data loaded (10K customers, 100K transactions)
- [ ] Basic monitoring dashboards operational

### Phase 2: ML Pipeline Implementation (25 minutes)

#### 2.1 Feature Store Setup
```python
# Feast feature store configuration
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String, UnixTimestamp
from datetime import timedelta

# Define entities
customer = Entity(
    name="customer_id",
    description="Customer identifier",
    value_type=String
)

# Historical features from data warehouse
customer_features_source = FileSource(
    path="s3://customer-analytics-features/customer_features.parquet",
    timestamp_field="event_timestamp"
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_order_value", dtype=Float64),
        Field(name="days_since_last_purchase", dtype=Int64),
        Field(name="preferred_category", dtype=String),
        Field(name="churn_probability", dtype=Float64),
    ],
    source=customer_features_source,
    ttl=timedelta(days=30)
)

# Real-time features from Redis
from feast.infra.online_stores.redis import RedisOnlineStoreConfig

redis_config = RedisOnlineStoreConfig(
    connection_string="redis://redis.customer-analytics.local:6379"
)

# Apply feature store configuration
fs = FeatureStore(repo_path=".")
fs.apply([customer, customer_features])
```

#### 2.2 ML Model Training Pipeline
```python
# Customer churn prediction model
import mlflow
import mlflow.sklearn
from sklearn.ensemble import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ChurnPredictionModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.feature_store = FeatureStore(repo_path=".")
    
    def train(self, training_data_path: str):
        """Train churn prediction model with MLflow tracking"""
        
        with mlflow.start_run(run_name="churn_prediction_training"):
            # Load features from feature store
            features = self.feature_store.get_historical_features(
                entity_df=pd.read_parquet(training_data_path),
                features=[
                    "customer_features:age",
                    "customer_features:total_purchases",
                    "customer_features:avg_order_value",
                    "customer_features:days_since_last_purchase"
                ]
            ).to_df()
            
            # Prepare training data
            X = features.drop(['customer_id', 'churn_label'], axis=1)
            y = features['churn_label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Log metrics and model
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(self.model, "churn_model")
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "model_uri": mlflow.get_artifact_uri("churn_model")
            }
```

#### 2.3 Model Serving Infrastructure
```python
# FastAPI model serving with auto-scaling
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
from typing import List

app = FastAPI(title="Customer Analytics API", version="1.0.0")

class PredictionRequest(BaseModel):
    customer_ids: List[str]

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    recommendation_score: float
    segment: str

# Load model from MLflow
model_uri = "models:/churn_prediction/production"
churn_model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict/churn", response_model=List[PredictionResponse])
async def predict_churn(request: PredictionRequest):
    """Predict customer churn probability"""
    try:
        # Get features from feature store
        feature_store = FeatureStore(repo_path=".")
        features = feature_store.get_online_features(
            features=[
                "customer_features:age",
                "customer_features:total_purchases",
                "customer_features:avg_order_value",
                "customer_features:days_since_last_purchase"
            ],
            entity_rows=[{"customer_id": cid} for cid in request.customer_ids]
        ).to_dict()
        
        # Make predictions
        predictions = []
        for customer_id in request.customer_ids:
            customer_features = [
                features[f"customer_features:age"][customer_id],
                features[f"customer_features:total_purchases"][customer_id],
                features[f"customer_features:avg_order_value"][customer_id],
                features[f"customer_features:days_since_last_purchase"][customer_id]
            ]
            
            churn_prob = churn_model.predict_proba([customer_features])[0][1]
            
            predictions.append(PredictionResponse(
                customer_id=customer_id,
                churn_probability=float(churn_prob),
                recommendation_score=calculate_recommendation_score(customer_features),
                segment=determine_segment(customer_features)
            ))
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": churn_model is not None}
```

#### 2.4 Deliverables for Phase 2
- [ ] Feature store deployed with customer features
- [ ] ML models trained and registered in MLflow
- [ ] Model serving API deployed to Kubernetes
- [ ] A/B testing framework configured
- [ ] Model monitoring dashboards created

### Phase 3: GenAI Integration (25 minutes)

#### 3.1 RAG System for Customer Insights
```python
# Customer insights RAG system
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

class CustomerInsightRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.1, max_tokens=500)
        self.vectorstore = None
        self.qa_chain = None
        self.setup_knowledge_base()
    
    def setup_knowledge_base(self):
        """Initialize vector database with customer knowledge"""
        
        # Load customer behavior patterns and business rules
        knowledge_data = pd.DataFrame([
            {
                "content": "High-value customers (LTV > $5000) typically purchase premium products and respond well to exclusive offers and early access to new products.",
                "category": "customer_segmentation"
            },
            {
                "content": "Customers with churn probability > 0.7 should receive retention campaigns including personalized discounts and customer success outreach.",
                "category": "churn_prevention"
            },
            {
                "content": "Customers who haven't purchased in 30+ days but have high engagement scores are good candidates for re-engagement campaigns.",
                "category": "reactivation"
            },
            {
                "content": "Premium segment customers prefer email communication, while Standard segment responds better to SMS and push notifications.",
                "category": "communication_preferences"
            }
        ])
        
        # Create documents and split text
        loader = DataFrameLoader(knowledge_data, page_content_column="content")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./customer_knowledge_base"
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
        )
    
    def generate_customer_insights(self, customer_data: dict) -> dict:
        """Generate personalized customer insights using RAG"""
        
        # Prepare customer context
        context = f"""
        Customer Profile Analysis:
        - Customer ID: {customer_data['customer_id']}
        - Segment: {customer_data['segment']}
        - Lifetime Value: ${customer_data['lifetime_value']}
        - Churn Probability: {customer_data['churn_probability']:.2%}
        - Days Since Last Purchase: {customer_data['days_since_last_purchase']}
        - Average Order Value: ${customer_data['avg_order_value']}
        - Total Purchases: {customer_data['total_purchases']}
        """
        
        # Generate insights using RAG
        insights_query = f"""
        Based on this customer profile, provide specific actionable insights and recommendations:
        {context}
        
        Please provide:
        1. Customer segment analysis
        2. Risk assessment and retention strategies
        3. Personalization opportunities
        4. Next best actions for customer success team
        """
        
        insights = self.qa_chain.run(insights_query)
        
        # Generate personalized recommendations
        recommendations_query = f"""
        Based on the customer profile, suggest 3 specific product recommendations and marketing strategies:
        {context}
        """
        
        recommendations = self.qa_chain.run(recommendations_query)
        
        return {
            "customer_id": customer_data['customer_id'],
            "insights": insights,
            "recommendations": recommendations,
            "confidence_score": self.calculate_confidence_score(customer_data),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def calculate_confidence_score(self, customer_data: dict) -> float:
        """Calculate confidence score based on data completeness"""
        required_fields = ['lifetime_value', 'churn_probability', 'total_purchases']
        available_fields = sum(1 for field in required_fields if customer_data.get(field) is not None)
        return available_fields / len(required_fields)
```

#### 3.2 Prompt Engineering with DSPy
```python
# Systematic prompt optimization with DSPy
import dspy
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate

class CustomerInsightGenerator(dspy.Signature):
    """Generate actionable customer insights based on profile data"""
    
    customer_profile = dspy.InputField(desc="Customer profile with behavioral and transactional data")
    business_context = dspy.InputField(desc="Business rules and segmentation guidelines")
    insights = dspy.OutputField(desc="Actionable insights with specific recommendations")

class OptimizedInsightSystem(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_insights = dspy.ChainOfThought(CustomerInsightGenerator)
    
    def forward(self, customer_profile, business_context):
        prediction = self.generate_insights(
            customer_profile=customer_profile,
            business_context=business_context
        )
        return dspy.Prediction(insights=prediction.insights)

# Optimize prompts with few-shot examples
def optimize_customer_insights_prompts():
    """Optimize prompts using DSPy with training examples"""
    
    # Training examples
    training_examples = [
        dspy.Example(
            customer_profile="High LTV customer ($8000), Premium segment, low churn risk (0.1), frequent purchaser",
            business_context="Premium customers prefer exclusive offers and early access",
            insights="Recommend VIP program enrollment, exclusive product previews, and premium customer success manager assignment"
        ),
        dspy.Example(
            customer_profile="Medium LTV customer ($2000), Standard segment, high churn risk (0.8), 45 days since last purchase",
            business_context="High churn risk customers need immediate retention intervention",
            insights="Initiate retention campaign with personalized discount, customer success outreach, and product recommendation based on purchase history"
        )
    ]
    
    # Configure optimizer
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=16)
    optimizer = BootstrapFewShot(metric=insight_quality_metric, **config)
    
    # Optimize the system
    optimized_system = optimizer.compile(OptimizedInsightSystem(), trainset=training_examples)
    
    return optimized_system

def insight_quality_metric(example, pred, trace=None):
    """Evaluate quality of generated insights"""
    # Check for actionability, specificity, and business relevance
    insights = pred.insights.lower()
    
    actionable_keywords = ['recommend', 'initiate', 'contact', 'offer', 'schedule']
    specific_keywords = ['discount', 'product', 'campaign', 'program', 'manager']
    
    actionable_score = sum(1 for keyword in actionable_keywords if keyword in insights)
    specific_score = sum(1 for keyword in specific_keywords if keyword in insights)
    
    return (actionable_score + specific_score) / 10  # Normalize to 0-1
```

#### 3.3 GenAI API Integration
```python
# GenAI-powered customer insights API
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio

app = FastAPI(title="Customer Insights GenAI API", version="1.0.0")

class InsightRequest(BaseModel):
    customer_id: str
    include_recommendations: bool = True
    context: Optional[str] = None

class InsightResponse(BaseModel):
    customer_id: str
    insights: str
    recommendations: str
    confidence_score: float
    processing_time_ms: int

# Initialize RAG system
rag_system = CustomerInsightRAG()
optimized_dspy_system = optimize_customer_insights_prompts()

@app.post("/insights/generate", response_model=InsightResponse)
async def generate_customer_insights(request: InsightRequest, background_tasks: BackgroundTasks):
    """Generate AI-powered customer insights"""
    start_time = time.time()
    
    try:
        # Get customer data from feature store and database
        customer_data = await get_customer_profile(request.customer_id)
        
        # Generate insights using RAG
        rag_insights = rag_system.generate_customer_insights(customer_data)
        
        # Enhance with DSPy-optimized prompts
        enhanced_insights = optimized_dspy_system(
            customer_profile=format_customer_profile(customer_data),
            business_context=request.context or "Standard business rules apply"
        )
        
        # Combine and format results
        final_insights = combine_insights(rag_insights, enhanced_insights)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Log for monitoring
        background_tasks.add_task(log_insight_generation, request.customer_id, processing_time)
        
        return InsightResponse(
            customer_id=request.customer_id,
            insights=final_insights['insights'],
            recommendations=final_insights['recommendations'],
            confidence_score=final_insights['confidence_score'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

@app.get("/insights/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_system": rag_system is not None,
        "dspy_system": optimized_dspy_system is not None
    }
```

#### 3.4 Deliverables for Phase 3
- [ ] RAG system deployed with customer knowledge base
- [ ] DSPy prompt optimization implemented
- [ ] GenAI insights API deployed to Kubernetes
- [ ] Vector database (ChromaDB) operational
- [ ] LLM integration with rate limiting and monitoring

### Phase 4: Production Deployment & Monitoring (15 minutes)

#### 4.1 Kubernetes Deployment
```yaml
# Complete Kubernetes deployment manifests
apiVersion: v1
kind: Namespace
metadata:
  name: customer-analytics
  labels:
    name: customer-analytics
    environment: production

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: customer-analytics
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: ml-api
        image: customer-analytics/ml-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow:5000"
        - name: FEAST_REPO_PATH
          value: "/app/feature_repo"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genai-api
  namespace: customer-analytics
spec:
  replicas: 2
  selector:
    matchLabels:
      app: genai-api
  template:
    metadata:
      labels:
        app: genai-api
    spec:
      containers:
      - name: genai-api
        image: customer-analytics/genai-api:latest
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        - name: CHROMA_HOST
          value: "chromadb:8000"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

---
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
  namespace: customer-analytics
spec:
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: customer-analytics-ingress
  namespace: customer-analytics
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.customer-analytics.com
    secretName: customer-analytics-tls
  rules:
  - host: api.customer-analytics.com
    http:
      paths:
      - path: /ml
        pathType: Prefix
        backend:
          service:
            name: ml-api-service
            port:
              number: 80
      - path: /insights
        pathType: Prefix
        backend:
          service:
            name: genai-api-service
            port:
              number: 80
```

#### 4.2 Monitoring and Observability
```yaml
# Prometheus monitoring configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: customer-analytics
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "customer_analytics_rules.yml"
    
    scrape_configs:
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
    
    - job_name: 'ml-models'
      static_configs:
      - targets: ['ml-api:8080']
      metrics_path: '/metrics'
      scrape_interval: 30s
    
    - job_name: 'genai-api'
      static_configs:
      - targets: ['genai-api:8080']
      metrics_path: '/metrics'
      scrape_interval: 30s

---
# Grafana dashboard for customer analytics
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-customer-analytics
  namespace: customer-analytics
data:
  customer-analytics.json: |
    {
      "dashboard": {
        "title": "Customer Analytics Platform",
        "panels": [
          {
            "title": "API Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total[5m])",
                "legendFormat": "{{method}} {{endpoint}}"
              }
            ]
          },
          {
            "title": "Model Prediction Latency",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m]))",
                "legendFormat": "95th percentile"
              }
            ]
          },
          {
            "title": "Feature Store Cache Hit Rate",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(feature_store_cache_hits_total[5m]) / rate(feature_store_requests_total[5m])",
                "legendFormat": "Cache Hit Rate"
              }
            ]
          },
          {
            "title": "GenAI Insight Generation Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(genai_insights_generated_total[5m])",
                "legendFormat": "Insights per second"
              }
            ]
          }
        ]
      }
    }
```

#### 4.3 Cost Optimization Configuration
```python
# Automated cost optimization with Kubernetes HPA and VPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
  namespace: customer-analytics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"

---
# Vertical Pod Autoscaler for right-sizing
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-api-vpa
  namespace: customer-analytics
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: ml-api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```

#### 4.4 Deliverables for Phase 4
- [ ] Complete system deployed to Kubernetes
- [ ] Prometheus and Grafana monitoring operational
- [ ] Auto-scaling configured (HPA and VPA)
- [ ] SSL certificates and ingress configured
- [ ] Cost optimization policies active
- [ ] Alerting rules configured for critical metrics

## Final Deliverables

### 1. Complete System Architecture
```
customer-analytics-platform/
├── infrastructure/
│   ├── terraform/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── modules/
│   └── kubernetes/
│       ├── namespaces/
│       ├── deployments/
│       ├── services/
│       └── monitoring/
├── data-pipeline/
│   ├── airflow/
│   │   ├── dags/
│   │   └── plugins/
│   ├── dbt/
│   │   ├── models/
│   │   ├── tests/
│   │   └── macros/
│   └── kafka/
│       └── connectors/
├── ml-platform/
│   ├── feature-store/
│   ├── models/
│   ├── serving/
│   └── monitoring/
├── genai-system/
│   ├── rag/
│   ├── prompts/
│   ├── vector-db/
│   └── api/
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   └── alerting/
└── docs/
    ├── architecture.md
    ├── deployment.md
    ├── api-docs.md
    └── runbooks/
```

### 2. API Documentation
Complete OpenAPI specifications for all endpoints with examples and authentication.

### 3. Monitoring Dashboards
- **System Health**: Infrastructure metrics, service availability
- **ML Performance**: Model accuracy, prediction latency, drift detection
- **GenAI Metrics**: Insight generation rate, LLM token usage, quality scores
- **Business KPIs**: Customer engagement, revenue impact, cost per prediction

### 4. Deployment Guide
Step-by-step instructions for deploying the entire system from scratch.

### 5. Architecture Decision Records (ADRs)
Documentation of key architectural decisions and trade-offs made during implementation.

## Success Criteria

### Technical Requirements ✅
- [ ] **System Availability**: > 99.9% uptime with health checks
- [ ] **API Performance**: < 100ms p95 latency for ML predictions
- [ ] **Data Freshness**: < 5 minutes for real-time features
- [ ] **Scalability**: Auto-scaling from 2 to 10 replicas based on load
- [ ] **Security**: HTTPS, authentication, input validation
- [ ] **Monitoring**: Comprehensive metrics and alerting

### Business Requirements ✅
- [ ] **Customer Insights**: Generate actionable insights for 10K+ customers
- [ ] **Churn Prediction**: > 85% accuracy on test dataset
- [ ] **Personalization**: Deliver relevant recommendations
- [ ] **Cost Efficiency**: < $0.10 per customer per month
- [ ] **Operational Excellence**: Automated deployment and monitoring

### Integration Requirements ✅
- [ ] **Data Sources**: PostgreSQL, MongoDB, Redis integrated
- [ ] **Streaming**: Real-time CDC with Kafka and Debezium
- [ ] **Orchestration**: Airflow DAGs with dbt transformations
- [ ] **ML Pipeline**: Feature store, model training, serving
- [ ] **GenAI**: RAG system with LLM integration
- [ ] **Infrastructure**: Kubernetes, Terraform, monitoring

## Evaluation Rubric

### Architecture Design (25 points)
- **Excellent (23-25)**: Comprehensive system design with clear component interactions, proper separation of concerns, and scalability considerations
- **Good (18-22)**: Well-designed system with minor gaps in architecture or documentation
- **Satisfactory (13-17)**: Basic system design with some architectural issues
- **Needs Improvement (0-12)**: Incomplete or poorly designed architecture

### Implementation Quality (25 points)
- **Excellent (23-25)**: Production-ready code with proper error handling, logging, testing, and documentation
- **Good (18-22)**: High-quality implementation with minor issues
- **Satisfactory (13-17)**: Functional implementation with some quality issues
- **Needs Improvement (0-12)**: Incomplete or low-quality implementation

### Technology Integration (25 points)
- **Excellent (23-25)**: Seamless integration of 10+ technologies with proper configuration and optimization
- **Good (18-22)**: Good integration with minor configuration issues
- **Satisfactory (13-17)**: Basic integration with some missing components
- **Needs Improvement (0-12)**: Poor or incomplete technology integration

### Production Readiness (25 points)
- **Excellent (23-25)**: Complete deployment with monitoring, security, auto-scaling, and operational procedures
- **Good (18-22)**: Well-deployed system with minor operational gaps
- **Satisfactory (13-17)**: Basic deployment with some production concerns
- **Needs Improvement (0-12)**: Incomplete or non-production-ready deployment

## Resources and References

### Architecture Patterns
- [Microservices Architecture](https://microservices.io/)
- [Event-Driven Architecture](https://martinfowler.com/articles/201701-event-driven.html)
- [Lambda Architecture](http://lambda-architecture.net/)

### Technology Documentation
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [dbt Documentation](https://docs.getdbt.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Feast Feature Store](https://docs.feast.dev/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)

### Industry Examples
- [Netflix ML Platform](https://netflixtechblog.com/machine-learning-platform-at-netflix-8dc2d2c0b0b4)
- [Uber's Michelangelo](https://eng.uber.com/michelangelo-machine-learning-platform/)
- [Airbnb's Data Platform](https://medium.com/airbnb-engineering/data-infrastructure-at-airbnb-8adfb34f169c)
- [Spotify's ML Platform](https://engineering.atspotify.com/2019/12/13/the-winding-road-to-better-machine-learning-infrastructure-through-tensorflow-extended-and-kubeflow/)

## Congratulations! 🎉

You've successfully completed the 60 Days Advanced Data and AI bootcamp by building a comprehensive production system that integrates:

- **15+ Technologies** across data engineering, ML, GenAI, and infrastructure
- **Production-Grade Architecture** with scalability, monitoring, and cost optimization
- **End-to-End Pipeline** from raw data to AI-powered business insights
- **Real-World Application** solving actual business problems

**You're now ready to tackle the most challenging data and AI engineering roles in the industry!**

### Next Steps in Your Career
1. **Showcase Your Work**: Add this capstone to your portfolio and LinkedIn
2. **Pursue Certifications**: AWS, GCP, or Azure ML certifications
3. **Join Communities**: Engage with data engineering and MLOps communities
4. **Continuous Learning**: Stay current with emerging technologies and patterns
5. **Mentor Others**: Share your knowledge and help others on their journey

**Welcome to the elite tier of data and AI engineers!** 🚀
