# Day 24: Project - Production Pipeline with Quality & Monitoring

## 🎯 Project Overview

**Company**: DataFlow Enterprises  
**Role**: Senior Data Platform Engineer  
**Challenge**: Build a comprehensive production data platform integrating all Phase 2 technologies  
**Stakeholders**: CTO, Head of Data, DevOps Team, Data Scientists, Business Analysts  

### Business Context

DataFlow Enterprises is a rapidly growing e-commerce company processing millions of customer interactions daily. The company needs a robust, scalable data platform that can handle both batch and real-time processing while ensuring data quality, providing comprehensive observability, and supporting advanced analytics and machine learning initiatives.

**Current Challenges**:
- Fragmented data processing across multiple tools
- Inconsistent data quality and validation processes
- Limited observability into pipeline health and performance
- Manual deployment and testing processes
- Difficulty scaling to handle growing data volumes
- Need for real-time insights alongside batch processing

**Success Criteria**:
- Process 10M+ records daily with 99.9% reliability
- Sub-hour data freshness for critical business metrics
- Automated data quality validation with comprehensive reporting
- Complete observability with proactive alerting
- Seamless integration of batch and streaming workloads
- Comprehensive testing coverage enabling confident deployments

---

## 🏗️ Architecture Design

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DataFlow Production Data Platform                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Data Sources   │    │   Processing    │    │    Outputs      │         │
│  │                 │    │     Layer       │    │                 │         │
│  │ • PostgreSQL    │───▶│                 │───▶│ • Data Warehouse│         │
│  │ • REST APIs     │    │ • Airflow       │    │ • Analytics DB  │         │
│  │ • CSV Files     │    │ • dbt           │    │ • Dashboards    │         │
│  │ • Kafka Streams │    │ • Great Expect. │    │ • ML Features   │         │
│  │ • Kinesis       │    │ • AWS Glue      │    │ • Real-time API │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Quality & Governance Layer                       │   │
│  │                                                                     │   │
│  │  Data Quality:              Testing:               Observability:   │   │
│  │  • Great Expectations       • Unit Tests          • Prometheus      │   │
│  │  • Custom Validators        • Integration Tests   • Grafana         │   │
│  │  • Schema Evolution         • End-to-End Tests    • ELK Stack       │   │
│  │  • Anomaly Detection        • Performance Tests   • Jaeger Tracing  │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Infrastructure & Deployment                      │   │
│  │                                                                     │   │
│  │  • Docker Containers: Consistent deployment environments           │   │
│  │  • Kubernetes: Orchestration and scaling (optional)                │   │
│  │  • AWS Services: Glue, Kinesis, S3, RDS                           │   │
│  │  • CI/CD Pipeline: Automated testing and deployment                │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose | Integration Points |
|-------|------------|---------|-------------------|
| **Orchestration** | Apache Airflow | Workflow management | dbt, Great Expectations, AWS Glue |
| **Transformation** | dbt | Data modeling & testing | Airflow DAGs, Great Expectations |
| **Quality** | Great Expectations | Data validation | Airflow tasks, dbt tests |
| **Streaming** | AWS Kinesis | Real-time processing | Airflow monitoring, Lambda functions |
| **Serverless ETL** | AWS Glue | Scalable batch processing | Airflow triggers, S3 integration |
| **Observability** | Prometheus + Grafana | Metrics & dashboards | All pipeline components |
| **Logging** | ELK Stack | Centralized logging | Airflow, dbt, custom applications |
| **Tracing** | Jaeger | Distributed tracing | Complex workflow debugging |
| **Testing** | pytest + custom | Comprehensive testing | CI/CD pipeline integration |

---

## 📋 Implementation Phases

### Phase 1: Infrastructure Setup (30 minutes)

#### Step 1.1: Project Structure Creation
```bash
# Create comprehensive project structure
mkdir dataflow-production-platform
cd dataflow-production-platform

# Create organized directory structure
mkdir -p {
  airflow/{dags,plugins,config,logs},
  dbt/{models/{staging,intermediate,marts},tests,macros,seeds,profiles},
  great_expectations/{expectations,checkpoints,plugins},
  monitoring/{prometheus,grafana/{dashboards,datasources},alerts},
  logging/{elasticsearch,kibana},
  aws/{glue_scripts,lambda_functions,cloudformation},
  tests/{unit,integration,end_to_end},
  docker/{airflow,monitoring,databases},
  docs/{architecture,runbooks,api},
  scripts/{deployment,utilities,data_generation}
}
```

#### Step 1.2: Docker Infrastructure Setup
Create `docker-compose.yml` with comprehensive services:

```yaml
version: '3.8'
services:
  # Core Airflow Services
  airflow-webserver:
    build: ./docker/airflow
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
      - AIRFLOW__CELERY__RESULT_BACKEND=redis://redis:6379/0
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./dbt:/opt/dbt
      - ./great_expectations:/opt/great_expectations
      - ./aws:/opt/aws
    ports:
      - "8080:8080"
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  airflow-scheduler:
    build: ./docker/airflow
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
      - AIRFLOW__CELERY__RESULT_BACKEND=redis://redis:6379/0
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./dbt:/opt/dbt
      - ./great_expectations:/opt/great_expectations
    depends_on:
      - postgres
      - redis
    command: scheduler

  airflow-worker:
    build: ./docker/airflow
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__CELERY__RESULT_BACKEND=redis://redis:6379/0
      - AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./dbt:/opt/dbt
      - ./great_expectations:/opt/great_expectations
    depends_on:
      - postgres
      - redis
    command: celery worker

  # Data Infrastructure
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
      - POSTGRES_MULTIPLE_DATABASES=dataflow_warehouse,great_expectations
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init-multiple-databases.sh:/docker-entrypoint-initdb.d/init-multiple-databases.sh
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus

  # Logging Stack
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch
    volumes:
      - ./logging/kibana/config:/usr/share/kibana/config

  # Distributed Tracing
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger collector
      - "6831:6831/udp"  # Jaeger agent
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
  elasticsearch_data:
```

#### Step 1.3: Configuration Files Setup
Create essential configuration files:
- Airflow configuration with production settings
- dbt profiles for multiple environments
- Great Expectations data context
- Monitoring configurations and dashboards

### Phase 2: Data Pipeline Implementation (60 minutes)

#### Step 2.1: Advanced Airflow DAGs (25 minutes)

**Main Production Pipeline DAG**:
```python
# airflow/dags/production_data_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
from airflow.sensors.filesystem import FileSensor
from airflow_dbt.operators.dbt_operator import DbtRunOperator, DbtTestOperator
from great_expectations_provider.operators.great_expectations import GreatExpectationsOperator

# DAG configuration
default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=2)
}

dag = DAG(
    'production_data_pipeline',
    default_args=default_args,
    description='Production data pipeline with comprehensive quality and monitoring',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,
    tags=['production', 'phase2-integration', 'critical']
)

# Data Ingestion Task Group
with TaskGroup('data_ingestion', dag=dag) as ingestion_group:
    
    # File sensor for batch data
    wait_for_files = FileSensor(
        task_id='wait_for_source_files',
        filepath='/opt/data/incoming/',
        fs_conn_id='data_source_conn',
        poke_interval=300,
        timeout=3600
    )
    
    # Multiple source ingestion
    data_sources = ['customers', 'transactions', 'products', 'events']
    ingestion_tasks = []
    
    for source in data_sources:
        ingest_task = PythonOperator(
            task_id=f'ingest_{source}',
            python_callable=ingest_data_source,
            op_kwargs={
                'source': source,
                'target_table': f'raw_{source}',
                'connection_id': f'{source}_conn'
            }
        )
        ingestion_tasks.append(ingest_task)
    
    wait_for_files >> ingestion_tasks

# Data Quality Validation Task Group
with TaskGroup('quality_validation', dag=dag) as quality_group:
    
    # Great Expectations validation
    validate_customers = GreatExpectationsOperator(
        task_id='validate_customers_data',
        expectation_suite_name='customers_suite',
        batch_request_file='/opt/great_expectations/batch_requests/customers.json',
        data_context_root_dir='/opt/great_expectations'
    )
    
    validate_transactions = GreatExpectationsOperator(
        task_id='validate_transactions_data',
        expectation_suite_name='transactions_suite',
        batch_request_file='/opt/great_expectations/batch_requests/transactions.json',
        data_context_root_dir='/opt/great_expectations'
    )
    
    # Custom data quality checks
    custom_quality_check = PythonOperator(
        task_id='custom_quality_validation',
        python_callable=run_custom_quality_checks
    )
    
    [validate_customers, validate_transactions] >> custom_quality_check

# dbt Transformations Task Group
with TaskGroup('transformations', dag=dag) as transform_group:
    
    # dbt staging models
    dbt_staging = DbtRunOperator(
        task_id='dbt_run_staging',
        dir='/opt/dbt',
        profiles_dir='/opt/dbt/profiles',
        select='tag:staging'
    )
    
    # dbt intermediate models
    dbt_intermediate = DbtRunOperator(
        task_id='dbt_run_intermediate',
        dir='/opt/dbt',
        profiles_dir='/opt/dbt/profiles',
        select='tag:intermediate'
    )
    
    # dbt marts models
    dbt_marts = DbtRunOperator(
        task_id='dbt_run_marts',
        dir='/opt/dbt',
        profiles_dir='/opt/dbt/profiles',
        select='tag:marts'
    )
    
    # dbt tests
    dbt_test = DbtTestOperator(
        task_id='dbt_test_all',
        dir='/opt/dbt',
        profiles_dir='/opt/dbt/profiles'
    )
    
    dbt_staging >> dbt_intermediate >> dbt_marts >> dbt_test

# AWS Integration Task Group
with TaskGroup('aws_integration', dag=dag) as aws_group:
    
    # Trigger AWS Glue job
    trigger_glue_job = PythonOperator(
        task_id='trigger_glue_etl',
        python_callable=trigger_aws_glue_job,
        op_kwargs={
            'job_name': 'dataflow_advanced_etl',
            'arguments': {
                '--source_database': 'dataflow_warehouse',
                '--target_s3_path': 's3://dataflow-processed/analytics/'
            }
        }
    )
    
    # Process Kinesis streams
    process_streams = PythonOperator(
        task_id='process_kinesis_streams',
        python_callable=process_real_time_streams,
        op_kwargs={
            'stream_names': ['customer_events', 'transaction_events']
        }
    )
    
    [trigger_glue_job, process_streams]

# Monitoring and Alerting
monitoring_task = PythonOperator(
    task_id='update_monitoring_metrics',
    python_callable=update_pipeline_metrics,
    dag=dag
)

# Define comprehensive dependencies
ingestion_group >> quality_group >> transform_group >> aws_group >> monitoring_task
```

**Monitoring and Alerting DAG**:
```python
# airflow/dags/monitoring_and_alerting.py
dag = DAG(
    'monitoring_and_alerting',
    default_args=default_args,
    description='Comprehensive monitoring and alerting for data platform',
    schedule_interval=timedelta(minutes=15),
    catchup=False,
    tags=['monitoring', 'alerting', 'operations']
)

# System health checks
system_health = PythonOperator(
    task_id='check_system_health',
    python_callable=check_system_health,
    dag=dag
)

# Data freshness monitoring
data_freshness = PythonOperator(
    task_id='monitor_data_freshness',
    python_callable=monitor_data_freshness,
    dag=dag
)

# Quality metrics collection
quality_metrics = PythonOperator(
    task_id='collect_quality_metrics',
    python_callable=collect_quality_metrics,
    dag=dag
)

# Performance monitoring
performance_monitoring = PythonOperator(
    task_id='monitor_performance',
    python_callable=monitor_pipeline_performance,
    dag=dag
)

# Alert generation
generate_alerts = PythonOperator(
    task_id='generate_alerts',
    python_callable=generate_operational_alerts,
    dag=dag
)

[system_health, data_freshness, quality_metrics, performance_monitoring] >> generate_alerts
```
#### Step 2.2: Advanced dbt Models (20 minutes)

**dbt Project Structure**:
```
dbt/
├── dbt_project.yml
├── profiles/
│   └── profiles.yml
├── models/
│   ├── staging/
│   │   ├── _sources.yml
│   │   ├── stg_customers.sql
│   │   ├── stg_transactions.sql
│   │   └── stg_products.sql
│   ├── intermediate/
│   │   ├── int_customer_metrics.sql
│   │   ├── int_transaction_analysis.sql
│   │   └── int_product_performance.sql
│   └── marts/
│       ├── dim_customers.sql
│       ├── fct_transactions.sql
│       └── customer_360.sql
├── tests/
│   ├── generic/
│   └── singular/
├── macros/
│   ├── generate_schema_name.sql
│   ├── data_quality_checks.sql
│   └── business_logic.sql
└── seeds/
    └── reference_data.csv
```

**Advanced Staging Model Example**:
```sql
-- models/staging/stg_customers.sql
{{ config(
    materialized='view',
    tags=['staging', 'customers'],
    meta={
        'owner': 'data_engineering_team',
        'description': 'Cleaned and standardized customer data'
    }
) }}

with source_data as (
    select * from {{ source('raw', 'customers') }}
),

cleaned_data as (
    select
        customer_id,
        {{ dbt_utils.generate_surrogate_key(['customer_id', 'email']) }} as customer_key,
        
        -- Data cleaning and standardization
        trim(lower(email)) as email,
        trim(first_name) as first_name,
        trim(last_name) as last_name,
        
        -- Date standardization
        cast(registration_date as date) as registration_date,
        cast(last_login_date as timestamp) as last_login_date,
        
        -- Business logic
        case
            when customer_segment is null then 'unknown'
            when customer_segment = '' then 'unknown'
            else lower(trim(customer_segment))
        end as customer_segment,
        
        -- Data quality indicators
        case
            when email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$' then true
            else false
        end as email_is_valid,
        
        -- Audit fields
        current_timestamp as processed_at,
        '{{ run_started_at }}' as dbt_run_started_at
        
    from source_data
    where customer_id is not null
),

final as (
    select
        *,
        -- Row quality score
        {{ calculate_row_quality_score([
            'customer_id',
            'email_is_valid',
            'first_name',
            'last_name'
        ]) }} as row_quality_score
        
    from cleaned_data
)

select * from final

-- Data quality tests
{{ config(
    post_hook="insert into {{ ref('data_quality_log') }} 
               select 'stg_customers', count(*), avg(row_quality_score), current_timestamp 
               from {{ this }}"
) }}
```

**Advanced Mart Model Example**:
```sql
-- models/marts/customer_360.sql
{{ config(
    materialized='table',
    indexes=[
        {'columns': ['customer_id'], 'unique': True},
        {'columns': ['customer_segment', 'value_tier']},
        {'columns': ['last_transaction_date']}
    ],
    tags=['marts', 'customer_analytics', 'critical'],
    meta={
        'owner': 'analytics_team',
        'description': 'Comprehensive customer 360 view for analytics and ML',
        'sla_hours': 2
    }
) }}

with customer_base as (
    select * from {{ ref('stg_customers') }}
    where email_is_valid = true
),

transaction_metrics as (
    select
        customer_id,
        count(*) as total_transactions,
        sum(transaction_amount) as total_spent,
        avg(transaction_amount) as avg_transaction_amount,
        min(transaction_date) as first_transaction_date,
        max(transaction_date) as last_transaction_date,
        
        -- Advanced analytics
        {{ dbt_utils.datediff('min(transaction_date)', 'max(transaction_date)', 'day') }} as customer_lifetime_days,
        
        -- Recency, Frequency, Monetary analysis
        {{ dbt_utils.datediff('max(transaction_date)', 'current_date', 'day') }} as days_since_last_transaction,
        
        -- Seasonal analysis
        count(case when extract(quarter from transaction_date) = 4 then 1 end) as q4_transactions,
        sum(case when extract(quarter from transaction_date) = 4 then transaction_amount end) as q4_spending
        
    from {{ ref('stg_transactions') }}
    group by customer_id
),

product_preferences as (
    select
        customer_id,
        {{ dbt_utils.listagg('product_category', ', ', "order by purchase_count desc") }} as top_categories,
        count(distinct product_category) as category_diversity
    from (
        select
            customer_id,
            product_category,
            count(*) as purchase_count,
            row_number() over (partition by customer_id order by count(*) desc) as category_rank
        from {{ ref('stg_transactions') }} t
        join {{ ref('stg_products') }} p on t.product_id = p.product_id
        group by customer_id, product_category
    ) ranked_categories
    where category_rank <= 3
    group by customer_id
),

final as (
    select
        cb.customer_id,
        cb.customer_key,
        cb.email,
        cb.first_name,
        cb.last_name,
        cb.registration_date,
        cb.customer_segment,
        
        -- Transaction metrics
        coalesce(tm.total_transactions, 0) as total_transactions,
        coalesce(tm.total_spent, 0) as total_spent,
        coalesce(tm.avg_transaction_amount, 0) as avg_transaction_amount,
        tm.first_transaction_date,
        tm.last_transaction_date,
        coalesce(tm.customer_lifetime_days, 0) as customer_lifetime_days,
        coalesce(tm.days_since_last_transaction, 999) as days_since_last_transaction,
        
        -- Customer value segmentation
        case
            when tm.total_spent > 10000 then 'high_value'
            when tm.total_spent > 1000 then 'medium_value'
            when tm.total_spent > 0 then 'low_value'
            else 'no_purchases'
        end as value_tier,
        
        -- Customer lifecycle stage
        case
            when tm.days_since_last_transaction <= 30 then 'active'
            when tm.days_since_last_transaction <= 90 then 'at_risk'
            when tm.days_since_last_transaction <= 365 then 'dormant'
            else 'churned'
        end as lifecycle_stage,
        
        -- Product preferences
        pp.top_categories,
        coalesce(pp.category_diversity, 0) as category_diversity,
        
        -- Seasonal behavior
        coalesce(tm.q4_transactions, 0) as q4_transactions,
        coalesce(tm.q4_spending, 0) as q4_spending,
        
        -- Audit fields
        current_timestamp as updated_at,
        '{{ run_started_at }}' as dbt_run_started_at
        
    from customer_base cb
    left join transaction_metrics tm on cb.customer_id = tm.customer_id
    left join product_preferences pp on cb.customer_id = pp.customer_id
)

select * from final
```

#### Step 2.3: Great Expectations Integration (15 minutes)

**Comprehensive Expectation Suite**:
```python
# great_expectations/expectations/customers_suite.py
import great_expectations as ge
from great_expectations.core.expectation_suite import ExpectationSuite

def create_customers_expectation_suite():
    """Create comprehensive expectation suite for customer data"""
    
    suite = ExpectationSuite(expectation_suite_name="customers_suite")
    
    # Table-level expectations
    suite.add_expectation(
        ge.expectations.ExpectTableRowCountToBeBetween(
            min_value=10000,
            max_value=50000000,
            meta={"notes": "Customer table should have reasonable row count"}
        )
    )
    
    # Column existence and types
    expected_columns = [
        "customer_id", "email", "first_name", "last_name", 
        "registration_date", "customer_segment"
    ]
    
    for column in expected_columns:
        suite.add_expectation(
            ge.expectations.ExpectColumnToExist(column=column)
        )
    
    # Data completeness expectations
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToNotBeNull(
            column="customer_id",
            meta={"notes": "Customer ID is required for all records"}
        )
    )
    
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeUnique(
            column="customer_id",
            meta={"notes": "Customer ID must be unique"}
        )
    )
    
    # Data validity expectations
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToMatchRegex(
            column="email",
            regex=r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
            meta={"notes": "Email must be valid format"}
        )
    )
    
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="customer_segment",
            value_set=["premium", "standard", "basic", "unknown"],
            meta={"notes": "Customer segment must be valid value"}
        )
    )
    
    # Statistical expectations
    suite.add_expectation(
        ge.expectations.ExpectColumnMeanToBeBetween(
            column="days_since_registration",
            min_value=0,
            max_value=3650,  # 10 years
            meta={"notes": "Average customer age should be reasonable"}
        )
    )
    
    # Custom business rule expectations
    suite.add_expectation(
        ge.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="last_login_date",
            column_B="registration_date",
            meta={"notes": "Last login must be after registration"}
        )
    )
    
    return suite

def create_transactions_expectation_suite():
    """Create comprehensive expectation suite for transaction data"""
    
    suite = ExpectationSuite(expectation_suite_name="transactions_suite")
    
    # Volume expectations
    suite.add_expectation(
        ge.expectations.ExpectTableRowCountToBeBetween(
            min_value=100000,
            max_value=100000000,
            meta={"notes": "Transaction volume should be within expected range"}
        )
    )
    
    # Required fields
    required_fields = ["transaction_id", "customer_id", "product_id", "transaction_amount", "transaction_date"]
    
    for field in required_fields:
        suite.add_expectation(
            ge.expectations.ExpectColumnValuesToNotBeNull(column=field)
        )
    
    # Business logic expectations
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="transaction_amount",
            min_value=0.01,
            max_value=100000.00,
            meta={"notes": "Transaction amounts should be positive and reasonable"}
        )
    )
    
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeBetween(
            column="transaction_date",
            min_value="2020-01-01",
            max_value="2030-12-31",
            meta={"notes": "Transaction dates should be within business range"}
        )
    )
    
    # Referential integrity
    suite.add_expectation(
        ge.expectations.ExpectColumnValuesToBeInSet(
            column="customer_id",
            value_set=None,  # Will be populated from customer table
            meta={"notes": "All customer IDs must exist in customer table"}
        )
    )
    
    return suite
```

### Phase 3: AWS Integration & Streaming (30 minutes)

#### Step 3.1: AWS Glue ETL Jobs (15 minutes)

**Advanced Glue ETL Script**:
```python
# aws/glue_scripts/advanced_analytics_etl.py
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Initialize Glue context
args = getResolvedOptions(sys.argv, [
    'JOB_NAME', 'source_database', 'target_s3_path', 
    'enable_metrics', 'enable_continuous_cloudwatch_log'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

def advanced_customer_analytics():
    """Advanced customer analytics with ML feature engineering"""
    
    # Read from Data Catalog
    customers_dyf = glueContext.create_dynamic_frame.from_catalog(
        database=args['source_database'],
        table_name="customer_360",
        transformation_ctx="customers_dyf"
    )
    
    transactions_dyf = glueContext.create_dynamic_frame.from_catalog(
        database=args['source_database'],
        table_name="fct_transactions",
        transformation_ctx="transactions_dyf"
    )
    
    # Convert to DataFrames for complex operations
    customers_df = customers_dyf.toDF()
    transactions_df = transactions_dyf.toDF()
    
    # Advanced feature engineering
    customer_features = customers_df.select(
        "customer_id",
        "customer_segment",
        "total_spent",
        "total_transactions",
        "days_since_last_transaction",
        
        # RFM Analysis
        F.when(F.col("days_since_last_transaction") <= 30, 5)
         .when(F.col("days_since_last_transaction") <= 60, 4)
         .when(F.col("days_since_last_transaction") <= 90, 3)
         .when(F.col("days_since_last_transaction") <= 180, 2)
         .otherwise(1).alias("recency_score"),
        
        F.when(F.col("total_transactions") >= 50, 5)
         .when(F.col("total_transactions") >= 20, 4)
         .when(F.col("total_transactions") >= 10, 3)
         .when(F.col("total_transactions") >= 5, 2)
         .otherwise(1).alias("frequency_score"),
        
        F.when(F.col("total_spent") >= 10000, 5)
         .when(F.col("total_spent") >= 5000, 4)
         .when(F.col("total_spent") >= 1000, 3)
         .when(F.col("total_spent") >= 500, 2)
         .otherwise(1).alias("monetary_score")
    ).withColumn(
        "rfm_score",
        F.col("recency_score") + F.col("frequency_score") + F.col("monetary_score")
    ).withColumn(
        "customer_value_segment",
        F.when(F.col("rfm_score") >= 12, "champions")
         .when(F.col("rfm_score") >= 9, "loyal_customers")
         .when(F.col("rfm_score") >= 6, "potential_loyalists")
         .otherwise("at_risk")
    )
    
    # Time-based aggregations
    transaction_trends = transactions_df.groupBy("customer_id").agg(
        F.count("*").alias("total_transactions"),
        F.sum("transaction_amount").alias("total_revenue"),
        F.avg("transaction_amount").alias("avg_order_value"),
        F.stddev("transaction_amount").alias("order_value_std"),
        F.min("transaction_date").alias("first_purchase_date"),
        F.max("transaction_date").alias("last_purchase_date"),
        
        # Seasonal patterns
        F.sum(F.when(F.quarter("transaction_date") == 1, F.col("transaction_amount")).otherwise(0)).alias("q1_revenue"),
        F.sum(F.when(F.quarter("transaction_date") == 2, F.col("transaction_amount")).otherwise(0)).alias("q2_revenue"),
        F.sum(F.when(F.quarter("transaction_date") == 3, F.col("transaction_amount")).otherwise(0)).alias("q3_revenue"),
        F.sum(F.when(F.quarter("transaction_date") == 4, F.col("transaction_amount")).otherwise(0)).alias("q4_revenue"),
        
        # Purchase patterns
        F.countDistinct("product_category").alias("category_diversity"),
        F.avg(F.datediff(F.lead("transaction_date").over(
            Window.partitionBy("customer_id").orderBy("transaction_date")
        ), F.col("transaction_date"))).alias("avg_days_between_purchases")
    )
    
    # Join features
    final_features = customer_features.join(
        transaction_trends, 
        on="customer_id", 
        how="left"
    ).withColumn(
        "processing_timestamp", 
        F.current_timestamp()
    ).withColumn(
        "data_version",
        F.lit("v1.0")
    )
    
    # Convert back to DynamicFrame
    final_dyf = DynamicFrame.fromDF(final_features, glueContext, "final_dyf")
    
    # Write to S3 with partitioning
    glueContext.write_dynamic_frame.from_options(
        frame=final_dyf,
        connection_type="s3",
        connection_options={
            "path": args['target_s3_path'],
            "partitionKeys": ["customer_value_segment", "customer_segment"]
        },
        format="parquet",
        format_options={
            "compression": "snappy"
        },
        transformation_ctx="final_write"
    )
    
    # Write summary statistics
    summary_stats = final_features.agg(
        F.count("*").alias("total_customers"),
        F.avg("rfm_score").alias("avg_rfm_score"),
        F.sum("total_revenue").alias("total_platform_revenue")
    ).withColumn("analysis_date", F.current_date())
    
    summary_dyf = DynamicFrame.fromDF(summary_stats, glueContext, "summary_dyf")
    
    glueContext.write_dynamic_frame.from_options(
        frame=summary_dyf,
        connection_type="s3",
        connection_options={
            "path": f"{args['target_s3_path']}/summary_stats/"
        },
        format="parquet",
        transformation_ctx="summary_write"
    )

# Execute ETL
advanced_customer_analytics()
job.commit()
```

#### Step 3.2: Kinesis Streaming Integration (15 minutes)

**Real-time Stream Processing**:
```python
# aws/lambda_functions/kinesis_stream_processor.py
import json
import base64
import boto3
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')
sns = boto3.client('sns')

# Configuration
REAL_TIME_METRICS_TABLE = 'real_time_customer_metrics'
ALERT_TOPIC_ARN = 'arn:aws:sns:region:account:data-platform-alerts'

def lambda_handler(event, context):
    """Process Kinesis stream records for real-time analytics"""
    
    processed_records = []
    metrics = {
        'records_processed': 0,
        'high_value_transactions': 0,
        'anomalies_detected': 0,
        'processing_errors': 0
    }
    
    for record in event['Records']:
        try:
            # Decode Kinesis record
            payload = json.loads(base64.b64decode(record['kinesis']['data']).decode('utf-8'))
            
            # Process transaction event
            if payload.get('event_type') == 'transaction':
                result = process_transaction_event(payload)
                processed_records.append(result)
                
                # Update metrics
                metrics['records_processed'] += 1
                
                if result.get('is_high_value'):
                    metrics['high_value_transactions'] += 1
                
                if result.get('is_anomaly'):
                    metrics['anomalies_detected'] += 1
                    send_anomaly_alert(result)
            
            # Process customer event
            elif payload.get('event_type') == 'customer_update':
                result = process_customer_event(payload)
                processed_records.append(result)
                metrics['records_processed'] += 1
                
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
            metrics['processing_errors'] += 1
    
    # Send metrics to CloudWatch
    send_cloudwatch_metrics(metrics)
    
    logger.info(f"Processed {metrics['records_processed']} records, "
               f"{metrics['anomalies_detected']} anomalies detected")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed_records': len(processed_records),
            'metrics': metrics
        })
    }

def process_transaction_event(payload):
    """Process real-time transaction event"""
    
    transaction_amount = float(payload.get('amount', 0))
    customer_id = payload.get('customer_id')
    
    # Real-time feature calculation
    features = {
        'customer_id': customer_id,
        'transaction_amount': transaction_amount,
        'timestamp': datetime.now().isoformat(),
        'is_high_value': transaction_amount > 1000,
        'is_anomaly': False
    }
    
    # Anomaly detection (simplified)
    if transaction_amount > 10000:
        features['is_anomaly'] = True
        features['anomaly_type'] = 'high_amount'
    
    # Update real-time metrics in DynamoDB
    try:
        table = dynamodb.Table(REAL_TIME_METRICS_TABLE)
        table.put_item(Item=features)
    except Exception as e:
        logger.error(f"Error updating DynamoDB: {str(e)}")
    
    return features

def process_customer_event(payload):
    """Process real-time customer event"""
    
    customer_id = payload.get('customer_id')
    event_type = payload.get('sub_type', 'unknown')
    
    features = {
        'customer_id': customer_id,
        'event_type': event_type,
        'timestamp': datetime.now().isoformat()
    }
    
    # Process different customer events
    if event_type == 'profile_update':
        features['profile_updated'] = True
    elif event_type == 'login':
        features['last_login'] = datetime.now().isoformat()
    
    return features

def send_anomaly_alert(anomaly_data):
    """Send real-time anomaly alert"""
    
    message = {
        'alert_type': 'transaction_anomaly',
        'customer_id': anomaly_data.get('customer_id'),
        'transaction_amount': anomaly_data.get('transaction_amount'),
        'anomaly_type': anomaly_data.get('anomaly_type'),
        'timestamp': anomaly_data.get('timestamp')
    }
    
    try:
        sns.publish(
            TopicArn=ALERT_TOPIC_ARN,
            Message=json.dumps(message),
            Subject='Real-time Transaction Anomaly Detected'
        )
    except Exception as e:
        logger.error(f"Error sending alert: {str(e)}")

def send_cloudwatch_metrics(metrics):
    """Send custom metrics to CloudWatch"""
    
    try:
        cloudwatch.put_metric_data(
            Namespace='DataPlatform/RealTime',
            MetricData=[
                {
                    'MetricName': 'RecordsProcessed',
                    'Value': metrics['records_processed'],
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'HighValueTransactions',
                    'Value': metrics['high_value_transactions'],
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'AnomaliesDetected',
                    'Value': metrics['anomalies_detected'],
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'ProcessingErrors',
                    'Value': metrics['processing_errors'],
                    'Unit': 'Count'
                }
            ]
        )
    except Exception as e:
        logger.error(f"Error sending CloudWatch metrics: {str(e)}")
```

### Phase 4: Testing & Quality Assurance (30 minutes)

#### Step 4.1: Comprehensive Testing Suite (20 minutes)

**Unit Tests for Pipeline Components**:
```python
# tests/unit/test_data_processing.py
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from airflow.models import DagBag
from scripts.data_processing import DataProcessor, CustomerAnalytics

class TestDataProcessor:
    """Unit tests for data processing components"""
    
    @pytest.fixture
    def sample_customer_data(self):
        """Generate sample customer data for testing"""
        return pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'email': ['user1@test.com', 'user2@test.com', 'invalid-email', 'user4@test.com', 'user5@test.com'],
            'first_name': ['John', 'Jane', 'Bob', '', 'Alice'],
            'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
            'registration_date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
            'customer_segment': ['premium', 'standard', 'basic', 'premium', 'standard']
        })
    
    @pytest.fixture
    def sample_transaction_data(self):
        """Generate sample transaction data for testing"""
        return pd.DataFrame({
            'transaction_id': [1, 2, 3, 4, 5],
            'customer_id': [1, 1, 2, 3, 4],
            'product_id': [101, 102, 103, 104, 105],
            'transaction_amount': [100.0, 250.0, 75.0, 500.0, 1200.0],
            'transaction_date': ['2023-06-01', '2023-06-02', '2023-06-01', '2023-06-03', '2023-06-01']
        })
    
    def test_email_validation(self, sample_customer_data):
        """Test email validation logic"""
        processor = DataProcessor()
        
        validated_data = processor.validate_emails(sample_customer_data)
        
        # Check that invalid emails are flagged
        assert validated_data.loc[2, 'email_is_valid'] == False
        assert validated_data.loc[0, 'email_is_valid'] == True
        assert validated_data.loc[1, 'email_is_valid'] == True
    
    def test_customer_segmentation(self, sample_transaction_data):
        """Test customer value segmentation logic"""
        analytics = CustomerAnalytics()
        
        customer_metrics = analytics.calculate_customer_metrics(sample_transaction_data)
        
        # Test segmentation logic
        high_value_customers = customer_metrics[customer_metrics['value_segment'] == 'high_value']
        assert len(high_value_customers) > 0
        
        # Customer 4 should be high value (1200.0 transaction)
        customer_4_segment = customer_metrics[customer_metrics['customer_id'] == 4]['value_segment'].iloc[0]
        assert customer_4_segment == 'high_value'
    
    def test_data_quality_scoring(self, sample_customer_data):
        """Test data quality scoring algorithm"""
        processor = DataProcessor()
        
        scored_data = processor.calculate_quality_scores(sample_customer_data)
        
        # Check that quality scores are calculated
        assert 'quality_score' in scored_data.columns
        assert scored_data['quality_score'].min() >= 0
        assert scored_data['quality_score'].max() <= 1
        
        # Row with missing first_name should have lower quality score
        empty_name_score = scored_data[scored_data['first_name'] == '']['quality_score'].iloc[0]
        complete_row_score = scored_data[scored_data['first_name'] == 'John']['quality_score'].iloc[0]
        assert empty_name_score < complete_row_score
    
    def test_rfm_analysis(self, sample_transaction_data):
        """Test RFM analysis calculations"""
        analytics = CustomerAnalytics()
        
        rfm_scores = analytics.calculate_rfm_scores(sample_transaction_data)
        
        # Check RFM components exist
        required_columns = ['recency_score', 'frequency_score', 'monetary_score', 'rfm_score']
        for col in required_columns:
            assert col in rfm_scores.columns
        
        # Check score ranges
        assert rfm_scores['recency_score'].min() >= 1
        assert rfm_scores['recency_score'].max() <= 5
        assert rfm_scores['frequency_score'].min() >= 1
        assert rfm_scores['frequency_score'].max() <= 5
        assert rfm_scores['monetary_score'].min() >= 1
        assert rfm_scores['monetary_score'].max() <= 5

class TestAirflowDAGs:
    """Test Airflow DAG structure and logic"""
    
    def test_dag_loading(self):
        """Test that DAGs load without errors"""
        dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
        
        # Check for import errors
        assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
        
        # Check that expected DAGs exist
        expected_dags = ['production_data_pipeline', 'monitoring_and_alerting']
        for dag_id in expected_dags:
            assert dag_id in dagbag.dag_ids, f"Missing DAG: {dag_id}"
    
    def test_production_pipeline_structure(self):
        """Test production pipeline DAG structure"""
        dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
        dag = dagbag.get_dag('production_data_pipeline')
        
        # Check task groups exist
        expected_task_groups = ['data_ingestion', 'quality_validation', 'transformations', 'aws_integration']
        task_group_ids = [tg.group_id for tg in dag.task_groups.values()]
        
        for group_id in expected_task_groups:
            assert group_id in task_group_ids, f"Missing task group: {group_id}"
        
        # Check dependencies
        ingestion_tasks = [task for task in dag.tasks if 'ingest_' in task.task_id]
        assert len(ingestion_tasks) > 0, "No ingestion tasks found"
    
    def test_dag_scheduling(self):
        """Test DAG scheduling configuration"""
        dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
        
        production_dag = dagbag.get_dag('production_data_pipeline')
        monitoring_dag = dagbag.get_dag('monitoring_and_alerting')
        
        # Check scheduling intervals
        assert production_dag.schedule_interval == '@hourly'
        assert str(monitoring_dag.schedule_interval) == '0:15:00'  # 15 minutes
        
        # Check other important settings
        assert production_dag.max_active_runs == 1
        assert production_dag.catchup == False
```

**Integration Tests**:
```python
# tests/integration/test_pipeline_integration.py
import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import patch, Mock
import great_expectations as ge

class TestPipelineIntegration:
    """Integration tests for complete pipeline flow"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_end_to_end_data_flow(self, temp_data_dir):
        """Test complete data flow from ingestion to output"""
        
        # Create test data files
        customers_data = pd.DataFrame({
            'customer_id': range(1, 101),
            'email': [f'user{i}@test.com' for i in range(1, 101)],
            'first_name': [f'User{i}' for i in range(1, 101)],
            'last_name': ['Test'] * 100,
            'registration_date': ['2023-01-01'] * 100,
            'customer_segment': ['standard'] * 100
        })
        
        transactions_data = pd.DataFrame({
            'transaction_id': range(1, 501),
            'customer_id': [i % 100 + 1 for i in range(500)],
            'product_id': [i % 50 + 1 for i in range(500)],
            'transaction_amount': [100.0 + (i % 100) for i in range(500)],
            'transaction_date': ['2023-06-01'] * 500
        })
        
        # Save test data
        customers_file = os.path.join(temp_data_dir, 'customers.csv')
        transactions_file = os.path.join(temp_data_dir, 'transactions.csv')
        
        customers_data.to_csv(customers_file, index=False)
        transactions_data.to_csv(transactions_file, index=False)
        
        # Test data ingestion
        from scripts.data_ingestion import DataIngestionManager
        ingestion_manager = DataIngestionManager()
        
        ingested_customers = ingestion_manager.ingest_csv_file(customers_file, 'customers')
        ingested_transactions = ingestion_manager.ingest_csv_file(transactions_file, 'transactions')
        
        assert len(ingested_customers) == 100
        assert len(ingested_transactions) == 500
        
        # Test data quality validation
        from scripts.quality_validation import QualityValidator
        validator = QualityValidator()
        
        customer_quality_results = validator.validate_dataframe(ingested_customers, 'customers')
        transaction_quality_results = validator.validate_dataframe(ingested_transactions, 'transactions')
        
        assert customer_quality_results['overall_success'] == True
        assert transaction_quality_results['overall_success'] == True
        
        # Test data transformation
        from scripts.data_transformation import DataTransformer
        transformer = DataTransformer()
        
        customer_360 = transformer.create_customer_360_view(ingested_customers, ingested_transactions)
        
        assert len(customer_360) == 100
        assert 'total_spent' in customer_360.columns
        assert 'value_segment' in customer_360.columns
    
    def test_great_expectations_integration(self, temp_data_dir):
        """Test Great Expectations integration with pipeline"""
        
        # Create Great Expectations context
        context = ge.get_context(context_root_dir=temp_data_dir)
        
        # Create test datasource
        datasource_config = {
            "name": "test_datasource",
            "class_name": "Datasource",
            "module_name": "great_expectations.datasource",
            "execution_engine": {
                "module_name": "great_expectations.execution_engine",
                "class_name": "PandasExecutionEngine"
            },
            "data_connectors": {
                "default_inferred_data_connector_name": {
                    "class_name": "InferredAssetFilesystemDataConnector",
                    "base_directory": temp_data_dir,
                    "default_regex": {
                        "group_names": ["data_asset_name"],
                        "pattern": "(.*)\\.csv"
                    }
                }
            }
        }
        
        context.add_datasource(**datasource_config)
        
        # Create test data
        test_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['test1@example.com', 'test2@example.com', 'test3@example.com'],
            'transaction_amount': [100.0, 200.0, 300.0]
        })
        
        test_file = os.path.join(temp_data_dir, 'test_data.csv')
        test_data.to_csv(test_file, index=False)
        
        # Create and run expectation suite
        suite = context.create_expectation_suite("test_suite")
        validator = context.get_validator(
            batch_request={
                "datasource_name": "test_datasource",
                "data_connector_name": "default_inferred_data_connector_name",
                "data_asset_name": "test_data"
            },
            expectation_suite_name="test_suite"
        )
        
        # Add expectations
        validator.expect_column_to_exist("customer_id")
        validator.expect_column_values_to_not_be_null("customer_id")
        validator.expect_column_values_to_be_unique("customer_id")
        
        # Run validation
        results = validator.validate()
        
        assert results.success == True
    
    def test_monitoring_integration(self):
        """Test monitoring and alerting integration"""
        
        from scripts.monitoring import MonitoringManager
        
        # Mock Prometheus and Grafana clients
        with patch('prometheus_client.CollectorRegistry') as mock_registry, \
             patch('scripts.monitoring.grafana_client') as mock_grafana:
            
            monitoring_manager = MonitoringManager()
            
            # Test metrics collection
            metrics = monitoring_manager.collect_pipeline_metrics()
            
            assert 'pipeline_runs_total' in metrics
            assert 'data_quality_score' in metrics
            assert 'processing_time_seconds' in metrics
            
            # Test alert generation
            alerts = monitoring_manager.check_alert_conditions()
            
            # Should return list of alerts (empty or with alerts)
            assert isinstance(alerts, list)
```
#### Step 4.2: Performance and Load Testing (10 minutes)

**Performance Testing Suite**:
```python
# tests/performance/test_pipeline_performance.py
import pytest
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import memory_profiler

class TestPipelinePerformance:
    """Performance and load testing for data pipeline"""
    
    def generate_large_dataset(self, num_records: int) -> pd.DataFrame:
        """Generate large dataset for performance testing"""
        
        np.random.seed(42)  # For reproducible results
        
        return pd.DataFrame({
            'customer_id': np.random.randint(1, num_records // 10, num_records),
            'transaction_id': range(1, num_records + 1),
            'product_id': np.random.randint(1, 1000, num_records),
            'transaction_amount': np.random.uniform(10, 1000, num_records),
            'transaction_date': pd.date_range('2023-01-01', periods=num_records, freq='1min'),
            'customer_segment': np.random.choice(['premium', 'standard', 'basic'], num_records)
        })
    
    @pytest.mark.performance
    def test_large_dataset_processing(self):
        """Test processing performance with large datasets"""
        
        # Test with 1M records
        large_dataset = self.generate_large_dataset(1_000_000)
        
        from scripts.data_processing import DataProcessor
        processor = DataProcessor()
        
        # Measure processing time
        start_time = time.time()
        processed_data = processor.process_transactions(large_dataset)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 300, f"Processing took {processing_time:.2f}s, exceeds 5-minute SLA"
        assert len(processed_data) == len(large_dataset), "Data loss during processing"
        
        # Memory usage check
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        assert memory_usage < 2048, f"Memory usage {memory_usage:.2f}MB exceeds 2GB limit"
    
    @pytest.mark.performance
    def test_concurrent_processing(self):
        """Test concurrent processing capabilities"""
        
        datasets = [self.generate_large_dataset(100_000) for _ in range(5)]
        
        from scripts.data_processing import DataProcessor
        processor = DataProcessor()
        
        # Test concurrent processing
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(processor.process_transactions, dataset) 
                for dataset in datasets
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = [processor.process_transactions(dataset) for dataset in datasets]
        sequential_time = time.time() - start_time
        
        # Concurrent should be faster than sequential
        assert concurrent_time < sequential_time, "Concurrent processing not faster than sequential"
        assert len(results) == len(sequential_results), "Concurrent processing lost data"
    
    @pytest.mark.performance
    @memory_profiler.profile
    def test_memory_efficiency(self):
        """Test memory efficiency of data processing"""
        
        # Process data in chunks to test memory management
        chunk_size = 100_000
        total_records = 1_000_000
        
        from scripts.data_processing import DataProcessor
        processor = DataProcessor()
        
        processed_chunks = []
        
        for i in range(0, total_records, chunk_size):
            chunk_data = self.generate_large_dataset(min(chunk_size, total_records - i))
            processed_chunk = processor.process_transactions(chunk_data)
            processed_chunks.append(len(processed_chunk))
            
            # Force garbage collection
            import gc
            gc.collect()
        
        total_processed = sum(processed_chunks)
        assert total_processed == total_records, "Data loss in chunked processing"
    
    @pytest.mark.performance
    def test_database_performance(self):
        """Test database query performance"""
        
        from scripts.database_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        # Test large query performance
        start_time = time.time()
        
        query = """
        SELECT 
            customer_id,
            COUNT(*) as transaction_count,
            SUM(transaction_amount) as total_spent,
            AVG(transaction_amount) as avg_spent
        FROM transactions 
        WHERE transaction_date >= '2023-01-01'
        GROUP BY customer_id
        HAVING COUNT(*) > 10
        ORDER BY total_spent DESC
        LIMIT 10000
        """
        
        results = db_manager.execute_query(query)
        query_time = time.time() - start_time
        
        # Query should complete within reasonable time
        assert query_time < 30, f"Query took {query_time:.2f}s, exceeds 30s limit"
        assert len(results) > 0, "Query returned no results"
    
    @pytest.mark.performance
    def test_api_response_times(self):
        """Test API endpoint response times"""
        
        import requests
        
        # Test various API endpoints
        endpoints = [
            '/api/customers/metrics',
            '/api/transactions/summary',
            '/api/quality/status',
            '/api/pipeline/health'
        ]
        
        base_url = 'http://localhost:8080'  # Airflow webserver
        
        for endpoint in endpoints:
            start_time = time.time()
            
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                # API should respond quickly
                assert response_time < 5, f"Endpoint {endpoint} took {response_time:.2f}s"
                assert response.status_code == 200, f"Endpoint {endpoint} returned {response.status_code}"
                
            except requests.exceptions.RequestException as e:
                pytest.skip(f"API endpoint {endpoint} not available: {str(e)}")

class TestScalabilityLimits:
    """Test system scalability limits"""
    
    @pytest.mark.scalability
    def test_maximum_throughput(self):
        """Test maximum data throughput"""
        
        from scripts.data_processing import DataProcessor
        processor = DataProcessor()
        
        # Test increasing data volumes
        volumes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
        processing_times = []
        
        for volume in volumes:
            dataset = pd.DataFrame({
                'id': range(volume),
                'value': np.random.random(volume)
            })
            
            start_time = time.time()
            processor.process_simple_data(dataset)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            # Calculate throughput (records per second)
            throughput = volume / processing_time
            print(f"Volume: {volume:,}, Time: {processing_time:.2f}s, Throughput: {throughput:,.0f} records/sec")
        
        # Throughput should scale reasonably
        min_throughput = min([volumes[i] / processing_times[i] for i in range(len(volumes))])
        assert min_throughput > 1000, f"Minimum throughput {min_throughput:.0f} records/sec too low"
    
    @pytest.mark.scalability
    def test_concurrent_user_load(self):
        """Test system under concurrent user load"""
        
        import threading
        import queue
        
        def simulate_user_request(request_queue, result_queue):
            """Simulate user making API requests"""
            while not request_queue.empty():
                try:
                    request_id = request_queue.get_nowait()
                    
                    # Simulate API call
                    start_time = time.time()
                    # Mock API processing
                    time.sleep(0.1)  # Simulate processing time
                    response_time = time.time() - start_time
                    
                    result_queue.put({
                        'request_id': request_id,
                        'response_time': response_time,
                        'success': True
                    })
                    
                except queue.Empty:
                    break
        
        # Test with increasing concurrent users
        user_counts = [10, 50, 100, 200]
        
        for user_count in user_counts:
            request_queue = queue.Queue()
            result_queue = queue.Queue()
            
            # Add requests to queue
            for i in range(user_count * 10):  # 10 requests per user
                request_queue.put(i)
            
            # Start concurrent users
            threads = []
            start_time = time.time()
            
            for _ in range(user_count):
                thread = threading.Thread(
                    target=simulate_user_request,
                    args=(request_queue, result_queue)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())
            
            success_rate = sum(1 for r in results if r['success']) / len(results)
            avg_response_time = sum(r['response_time'] for r in results) / len(results)
            
            print(f"Users: {user_count}, Success Rate: {success_rate:.2%}, "
                  f"Avg Response Time: {avg_response_time:.3f}s")
            
            # System should handle load gracefully
            assert success_rate > 0.95, f"Success rate {success_rate:.2%} too low for {user_count} users"
            assert avg_response_time < 1.0, f"Response time {avg_response_time:.3f}s too high"
```

### Phase 5: Monitoring & Deployment (30 minutes)

#### Step 5.1: Comprehensive Monitoring Setup (20 minutes)

**Prometheus Configuration**:
```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Airflow metrics
  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow-webserver:8080']
    metrics_path: '/admin/metrics'
    scrape_interval: 30s

  # PostgreSQL metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']

  # Custom application metrics
  - job_name: 'data-pipeline'
    static_configs:
      - targets: ['data-pipeline-metrics:8000']
    scrape_interval: 15s

  # System metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # Redis metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis_exporter:9121']
```

**Grafana Dashboard Configuration**:
```json
{
  "dashboard": {
    "id": null,
    "title": "DataFlow Production Pipeline Dashboard",
    "tags": ["dataflow", "production", "pipeline"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Pipeline Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(airflow_dag_run_success_total[5m]) / rate(airflow_dag_run_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 95},
                {"color": "green", "value": 99}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Data Quality Score",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(data_quality_score)",
            "legendFormat": "Quality Score"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.8},
                {"color": "green", "value": 0.95}
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Processing Volume",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(records_processed_total[5m]))",
            "legendFormat": "Records/sec"
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(pipeline_errors_total[5m]))",
            "legendFormat": "Errors/sec"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

**Alert Rules**:
```yaml
# monitoring/prometheus/rules/pipeline_alerts.yml
groups:
  - name: pipeline_alerts
    rules:
      - alert: PipelineFailureRate
        expr: rate(airflow_dag_run_failed_total[5m]) / rate(airflow_dag_run_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High pipeline failure rate detected"
          description: "Pipeline failure rate is {{ $value | humanizePercentage }} over the last 5 minutes"

      - alert: DataQualityDegradation
        expr: avg(data_quality_score) < 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Data quality score below threshold"
          description: "Average data quality score is {{ $value | humanize }} (threshold: 0.8)"

      - alert: ProcessingLatency
        expr: histogram_quantile(0.95, rate(processing_duration_seconds_bucket[5m])) > 300
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High processing latency detected"
          description: "95th percentile processing time is {{ $value | humanizeDuration }}"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "PostgreSQL database is not responding"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"
```

#### Step 5.2: Deployment Automation (10 minutes)

**CI/CD Pipeline Configuration**:
```yaml
# .github/workflows/deploy_pipeline.yml
name: Deploy DataFlow Production Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  DOCKER_REGISTRY: your-registry.com
  PROJECT_NAME: dataflow-pipeline

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        DATABASE_URL: postgresql://postgres:test@localhost:5432/test_db
    
    - name: Run data quality tests
      run: |
        pytest tests/quality/ -v
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.DOCKER_REGISTRY }}
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}
    
    - name: Build and push Docker images
      run: |
        docker buildx build --platform linux/amd64,linux/arm64 \
          -t ${{ env.DOCKER_REGISTRY }}/${{ env.PROJECT_NAME }}-airflow:${{ github.sha }} \
          -t ${{ env.DOCKER_REGISTRY }}/${{ env.PROJECT_NAME }}-airflow:latest \
          --push ./docker/airflow
        
        docker buildx build --platform linux/amd64,linux/arm64 \
          -t ${{ env.DOCKER_REGISTRY }}/${{ env.PROJECT_NAME }}-monitoring:${{ github.sha }} \
          -t ${{ env.DOCKER_REGISTRY }}/${{ env.PROJECT_NAME }}-monitoring:latest \
          --push ./docker/monitoring

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        # Update docker-compose with new image tags
        sed -i "s|image: .*airflow.*|image: ${{ env.DOCKER_REGISTRY }}/${{ env.PROJECT_NAME }}-airflow:${{ github.sha }}|g" docker-compose.staging.yml
        
        # Deploy to staging environment
        scp docker-compose.staging.yml user@staging-server:/opt/dataflow/
        ssh user@staging-server "cd /opt/dataflow && docker-compose -f docker-compose.staging.yml up -d"
    
    - name: Run smoke tests
      run: |
        # Wait for services to be ready
        sleep 60
        
        # Run smoke tests against staging
        pytest tests/smoke/ --base-url=https://staging.dataflow.com -v

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        # Blue-green deployment strategy
        sed -i "s|image: .*airflow.*|image: ${{ env.DOCKER_REGISTRY }}/${{ env.PROJECT_NAME }}-airflow:${{ github.sha }}|g" docker-compose.prod.yml
        
        # Deploy to production
        scp docker-compose.prod.yml user@prod-server:/opt/dataflow/
        ssh user@prod-server "cd /opt/dataflow && ./deploy-blue-green.sh"
    
    - name: Verify production deployment
      run: |
        # Health checks
        curl -f https://prod.dataflow.com/health || exit 1
        
        # Run production verification tests
        pytest tests/production/ --base-url=https://prod.dataflow.com -v
```

---

## 📦 Deliverables

### 1. Complete Production Codebase
```
dataflow-production-platform/
├── README.md                          # Comprehensive project documentation
├── docker-compose.yml                 # Local development environment
├── docker-compose.staging.yml         # Staging environment
├── docker-compose.prod.yml           # Production environment
├── .env.example                       # Environment variables template
├── requirements.txt                   # Python dependencies
├── requirements-test.txt              # Testing dependencies
│
├── airflow/
│   ├── dags/
│   │   ├── production_data_pipeline.py
│   │   ├── monitoring_and_alerting.py
│   │   └── data_quality_validation.py
│   ├── plugins/
│   │   ├── custom_operators.py
│   │   └── custom_sensors.py
│   └── config/
│       └── airflow.cfg
│
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles/
│   ├── models/
│   │   ├── staging/
│   │   ├── intermediate/
│   │   └── marts/
│   ├── tests/
│   ├── macros/
│   └── seeds/
│
├── great_expectations/
│   ├── great_expectations.yml
│   ├── expectations/
│   ├── checkpoints/
│   └── plugins/
│
├── aws/
│   ├── glue_scripts/
│   ├── lambda_functions/
│   └── cloudformation/
│
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   └── alerts/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── performance/
│   └── end_to_end/
│
├── scripts/
│   ├── deployment/
│   ├── utilities/
│   └── data_generation/
│
├── docs/
│   ├── architecture/
│   ├── runbooks/
│   └── api/
│
└── .github/
    └── workflows/
        └── deploy_pipeline.yml
```

### 2. Documentation Package
- **Architecture Documentation**: Complete system design and integration patterns
- **Deployment Guide**: Step-by-step deployment for all environments
- **Operations Runbook**: Incident response and troubleshooting procedures
- **API Documentation**: Complete API specifications and usage examples
- **Performance Benchmarks**: Baseline performance metrics and optimization guides

### 3. Monitoring & Observability
- **Grafana Dashboards**: Executive, operational, and technical dashboards
- **Prometheus Alerts**: Comprehensive alerting rules for all system components
- **Log Analysis**: ELK stack configuration for centralized logging
- **Distributed Tracing**: Jaeger setup for complex workflow debugging

### 4. Testing Suite
- **Unit Tests**: 90%+ code coverage for all components
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Load testing and scalability validation
- **Quality Tests**: Data quality and business rule validation

---

## ✅ Success Criteria

### Functional Requirements ✅
- [ ] **Complete Pipeline**: Processes data from ingestion to analytics-ready output
- [ ] **Multi-Technology Integration**: Seamlessly combines Airflow, dbt, Great Expectations, AWS services
- [ ] **Quality Assurance**: Automated validation with comprehensive reporting
- [ ] **Real-time Processing**: Kinesis streaming integration with batch processing
- [ ] **Monitoring**: Complete observability with proactive alerting
- [ ] **Testing**: Comprehensive test coverage enabling confident deployments

### Performance Requirements ✅
- [ ] **Throughput**: Processes 10M+ records daily (>115 records/second average)
- [ ] **Latency**: Sub-hour data freshness for critical business metrics
- [ ] **Reliability**: 99.9% uptime with automatic failover and recovery
- [ ] **Scalability**: Handles 10x data growth without architectural changes
- [ ] **Efficiency**: Optimal resource utilization with cost monitoring

### Operational Requirements ✅
- [ ] **Deployment**: Automated CI/CD pipeline with blue-green deployment
- [ ] **Monitoring**: Real-time visibility into all system components
- [ ] **Alerting**: Proactive notification of issues and anomalies
- [ ] **Documentation**: Complete operational procedures and troubleshooting guides
- [ ] **Security**: Comprehensive security controls and audit trails

---

## 🎯 Evaluation Rubric

### Architecture & Integration (30 points)
- **Excellent (27-30)**: Seamless integration of all Phase 2 technologies with production-grade architecture
- **Good (21-26)**: Solid integration with most technologies working together effectively
- **Satisfactory (15-20)**: Basic integration with core functionality working
- **Needs Improvement (0-14)**: Poor integration or missing critical components

### Implementation Quality (25 points)
- **Excellent (23-25)**: Production-ready code with comprehensive error handling and optimization
- **Good (18-22)**: Well-implemented solution following best practices
- **Satisfactory (13-17)**: Functional implementation with basic quality standards
- **Needs Improvement (0-12)**: Poor code quality or incomplete implementation

### Testing & Quality (25 points)
- **Excellent (23-25)**: Comprehensive testing suite with high coverage and quality validation
- **Good (18-22)**: Good testing coverage with quality checks implemented
- **Satisfactory (13-17)**: Basic testing with some quality validation
- **Needs Improvement (0-12)**: Insufficient testing or poor quality controls

### Monitoring & Operations (20 points)
- **Excellent (18-20)**: Complete observability with proactive monitoring and alerting
- **Good (14-17)**: Good monitoring setup with basic alerting
- **Satisfactory (10-13)**: Basic monitoring with limited observability
- **Needs Improvement (0-9)**: Poor or missing monitoring capabilities

---

## 🚀 Extension Challenges

### Advanced Features (Optional)
1. **Machine Learning Integration**: Add ML model training and serving to the pipeline
2. **Multi-Cloud Deployment**: Deploy across AWS, Azure, and GCP with unified management
3. **Advanced Analytics**: Implement real-time OLAP cubes and advanced analytics
4. **Cost Optimization**: Add comprehensive cost tracking and optimization features
5. **Disaster Recovery**: Implement cross-region backup and disaster recovery

### Enterprise Enhancements
1. **Security Hardening**: Implement advanced security controls and compliance features
2. **Performance Optimization**: Add advanced caching, query optimization, and resource management
3. **API Gateway**: Create comprehensive API layer for external integrations
4. **Data Mesh Architecture**: Implement domain-driven data architecture patterns
5. **Advanced Governance**: Add comprehensive data governance and compliance automation

---

## 📞 Support & Resources

### Getting Help
- **Architecture Questions**: Review the comprehensive architecture documentation
- **Implementation Issues**: Check the troubleshooting guides and runbooks
- **Performance Problems**: Use the monitoring dashboards and performance guides
- **Quality Issues**: Consult the data quality validation procedures

### Additional Resources
- **Phase 2 Integration Patterns**: Review all previous days for integration examples
- **Production Best Practices**: [DataOps Best Practices Guide](https://dataops.org/)
- **Monitoring Strategies**: [Observability Engineering](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
- **Testing Approaches**: [Testing Data Pipelines](https://www.oreilly.com/library/view/fundamentals-of-data/9781492108344/)

---

**Project Duration**: 2 hours  
**Difficulty Level**: ⭐⭐⭐⭐⭐ (Advanced Integration Project)  
**Team Size**: 1-2 people  
**Prerequisites**: Completion of Days 15-23 (Phase 2)

**Ready to build enterprise-grade production data platforms!** 🚀

**This project represents the culmination of Phase 2 and demonstrates advanced data engineering skills through comprehensive technology integration, production-grade implementation, and enterprise-level operational practices.**