# Day 8: Data Catalogs - DataHub, Amundsen

## 📖 Learning Objectives

By the end of today, you will:
- Understand the critical role of data catalogs in modern data architecture
- Compare and contrast DataHub vs Amundsen capabilities and use cases
- Implement metadata ingestion and discovery workflows
- Build data lineage tracking and impact analysis systems
- Apply data governance and quality monitoring through catalogs
- Deploy production-ready data catalog solutions

---

## Theory

### What Are Data Catalogs?

Data catalogs are centralized metadata management systems that help organizations discover, understand, and govern their data assets. Think of them as "Google for your data" - they provide search, documentation, lineage, and governance capabilities across all data sources.

**Key Problems Data Catalogs Solve**:
- **Data Discovery**: "What data do we have and where is it?"
- **Data Understanding**: "What does this dataset contain and how is it structured?"
- **Data Lineage**: "Where does this data come from and where is it used?"
- **Data Quality**: "Can I trust this data for my analysis?"
- **Data Governance**: "Who owns this data and what are the access policies?"

### Why Data Catalogs Matter

#### The Data Discovery Problem

```
Without Data Catalog:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Team   │    │ Analytics   │    │ ML Team     │
│             │    │ Team        │    │             │
│ "We have    │    │ "Where is   │    │ "What data  │
│ this data   │    │ customer    │    │ can I use   │
│ somewhere"  │    │ data?"      │    │ for this    │
│             │    │             │    │ model?"     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ❌ Silos, Duplication,
                       Wasted Time
```

```
With Data Catalog:
┌─────────────────────────────────────────────────────────────┐
│                    Data Catalog                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Discovery   │  │ Lineage     │  │ Governance  │        │
│  │ & Search    │  │ & Impact    │  │ & Quality   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
       │                   │                   │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Team   │    │ Analytics   │    │ ML Team     │
│ ✅ Publishes│    │ ✅ Finds    │    │ ✅ Discovers│
│ metadata    │    │ trusted     │    │ features    │
│             │    │ datasets    │    │ quickly     │
└─────────────┘    └─────────────┘    └─────────────┘
```

#### Business Impact

**Without Data Catalog**:
- Data scientists spend 80% of time finding and preparing data
- Duplicate data pipelines and inconsistent metrics
- Compliance violations due to unknown data usage
- Poor data quality due to lack of visibility

**With Data Catalog**:
- 60% reduction in time to find relevant data
- Consistent metrics and reduced duplication
- Automated compliance and governance
- Proactive data quality monitoring

### DataHub vs Amundsen: Comprehensive Comparison

#### DataHub (LinkedIn)

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                      DataHub                                │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Frontend  │  │   Backend   │  │  Metadata   │        │
│  │   (React)   │  │   (Java)    │  │  Service    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   GraphQL   │  │    Kafka    │  │  Elasticsearch│       │
│  │     API     │  │   Events    │  │   Search     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**Key Strengths**:
- **Real-time Updates**: Kafka-based event streaming for immediate metadata updates
- **Rich API**: GraphQL API for programmatic access and integrations
- **Extensible**: Plugin architecture for custom metadata models
- **Lineage**: Advanced column-level lineage tracking
- **Active Development**: Strong LinkedIn backing and community

**Use Cases**:
- Large-scale enterprises with complex data ecosystems
- Organizations needing real-time metadata updates
- Teams requiring extensive API integrations
- Companies with custom metadata requirements

#### Amundsen (Lyft)

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                     Amundsen                                │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Frontend   │  │  Metadata   │  │   Search    │        │
│  │  (Flask)    │  │  Service    │  │  Service    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Neo4j    │  │ Elasticsearch│  │   Apache    │        │
│  │   Graph     │  │   Search     │  │   Airflow   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

**Key Strengths**:
- **Graph Database**: Neo4j provides powerful relationship queries
- **User Experience**: Clean, intuitive interface focused on data discovery
- **Simplicity**: Easier to deploy and maintain
- **Community**: Strong open-source community and documentation
- **Proven**: Battle-tested at Lyft and other major companies

**Use Cases**:
- Medium to large organizations focused on data discovery
- Teams prioritizing user experience and ease of use
- Organizations with existing Neo4j expertise
- Companies wanting proven, stable solutions

#### Feature Comparison

| Feature | DataHub | Amundsen |
|---------|---------|----------|
| **Real-time Updates** | ✅ Kafka-based | ❌ Batch-based |
| **API Richness** | ✅ GraphQL + REST | ⚠️ REST only |
| **Lineage Granularity** | ✅ Column-level | ⚠️ Table-level |
| **Graph Capabilities** | ⚠️ Limited | ✅ Neo4j native |
| **Deployment Complexity** | ⚠️ Higher | ✅ Lower |
| **Customization** | ✅ Highly extensible | ⚠️ Limited |
| **Community** | ✅ Very active | ✅ Active |
| **Enterprise Features** | ✅ Advanced | ⚠️ Basic |

### Core Data Catalog Concepts

#### 1. Metadata Management

**Types of Metadata**:

```python
# Technical Metadata
{
    "dataset": "user_events",
    "schema": {
        "user_id": {"type": "bigint", "nullable": false},
        "event_type": {"type": "varchar", "nullable": false},
        "timestamp": {"type": "timestamp", "nullable": false},
        "properties": {"type": "json", "nullable": true}
    },
    "location": "s3://data-lake/events/user_events/",
    "format": "parquet",
    "partitions": ["year", "month", "day"],
    "size_bytes": 1073741824,
    "row_count": 10000000
}

# Business Metadata
{
    "dataset": "user_events",
    "description": "User interaction events from web and mobile apps",
    "owner": "analytics-team@company.com",
    "domain": "user_behavior",
    "tags": ["pii", "gdpr", "critical"],
    "sla": {
        "freshness": "< 1 hour",
        "quality": "> 99%"
    },
    "usage": "Used for user segmentation and product analytics"
}

# Operational Metadata
{
    "dataset": "user_events",
    "last_updated": "2024-01-15T10:30:00Z",
    "update_frequency": "hourly",
    "data_quality_score": 0.98,
    "popularity_score": 0.85,
    "access_count_30d": 1250,
    "unique_users_30d": 45
}
```

#### 2. Data Lineage

**Table-Level Lineage**:
```
Raw Data → Staging → Processed → Analytics
    │         │          │          │
    │         │          │          └─→ Dashboard
    │         │          └─→ ML Features
    │         └─→ Data Quality Checks
    └─→ Archive
```

**Column-Level Lineage**:
```sql
-- Source: raw_events.user_id
-- Transformation: CAST(user_id AS STRING)
-- Destination: processed_events.user_identifier

SELECT 
    CAST(user_id AS STRING) as user_identifier,  -- Lineage tracked
    event_type,
    DATE(timestamp) as event_date               -- Lineage tracked
FROM raw_events
WHERE timestamp >= '2024-01-01'
```

#### 3. Data Discovery

**Search Capabilities**:
- **Full-text search**: Search across dataset names, descriptions, column names
- **Faceted search**: Filter by owner, domain, tags, data source
- **Semantic search**: Find datasets by business concepts
- **Popularity ranking**: Surface most-used datasets first

**Example Search Queries**:
```
"customer revenue"          → Find revenue-related customer datasets
owner:analytics-team        → Find datasets owned by analytics team
tag:pii AND domain:finance  → Find PII data in finance domain
updated:last-week           → Find recently updated datasets
```

#### 4. Data Quality Integration

**Quality Metrics**:
```python
quality_metrics = {
    "completeness": 0.98,      # % of non-null values
    "uniqueness": 0.95,        # % of unique values where expected
    "validity": 0.99,          # % of values matching expected format
    "consistency": 0.97,       # % of values consistent across sources
    "timeliness": 0.92,        # % of data arriving within SLA
    "accuracy": 0.94           # % of values matching source of truth
}
```

**Quality Rules**:
```yaml
# Great Expectations integration
quality_rules:
  - expectation: expect_column_values_to_not_be_null
    column: user_id
    threshold: 0.99
  
  - expectation: expect_column_values_to_be_in_set
    column: event_type
    value_set: ["click", "view", "purchase", "signup"]
    threshold: 1.0
  
  - expectation: expect_column_values_to_match_regex
    column: email
    regex: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
    threshold: 0.95
```

### DataHub Implementation Deep Dive

#### 1. Architecture Components

**Metadata Service (GMS)**:
```java
@RestController
@RequestMapping("/entities")
public class EntityController {
    
    @Autowired
    private EntityService entityService;
    
    @GetMapping("/{entityType}/{entityUrn}")
    public ResponseEntity<EntityResponse> getEntity(
            @PathVariable String entityType,
            @PathVariable String entityUrn) {
        
        Entity entity = entityService.getEntity(entityUrn);
        return ResponseEntity.ok(EntityResponse.builder()
            .entity(entity)
            .aspects(entity.getAspects())
            .build());
    }
    
    @PostMapping("/{entityType}")
    public ResponseEntity<Void> ingestMetadata(
            @PathVariable String entityType,
            @RequestBody MetadataChangeProposal mcp) {
        
        entityService.ingestProposal(mcp);
        return ResponseEntity.ok().build();
    }
}
```

**Metadata Ingestion**:
```python
# DataHub Python SDK
from datahub.emitter.mce_builder import make_data_platform_urn, make_dataset_urn
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import DatasetPropertiesClass

# Create emitter
emitter = DatahubRestEmitter(gms_server="http://localhost:8080")

# Create dataset metadata
dataset_urn = make_dataset_urn(platform="postgres", name="ecommerce.users")

dataset_properties = DatasetPropertiesClass(
    description="User account information including profile and preferences",
    customProperties={
        "owner": "data-team@company.com",
        "domain": "customer",
        "criticality": "high"
    }
)

# Emit metadata
mcp = MetadataChangeProposalWrapper(
    entityType="dataset",
    entityUrn=dataset_urn,
    aspectName="datasetProperties",
    aspect=dataset_properties
)

emitter.emit_mcp(mcp)
```

#### 2. Custom Metadata Models

**Define Custom Aspects**:
```json
{
  "type": "record",
  "name": "DataQualityMetrics",
  "namespace": "com.company.metadata",
  "fields": [
    {
      "name": "completenessScore",
      "type": "double",
      "doc": "Percentage of complete records"
    },
    {
      "name": "validityScore", 
      "type": "double",
      "doc": "Percentage of valid records"
    },
    {
      "name": "lastQualityCheck",
      "type": "long",
      "doc": "Timestamp of last quality assessment"
    },
    {
      "name": "qualityRules",
      "type": {
        "type": "array",
        "items": "string"
      },
      "doc": "List of applied quality rules"
    }
  ]
}
```

#### 3. Advanced Lineage Tracking

**SQL Parser Integration**:
```python
from datahub.utilities.sql_parser import SqlLineageParser

# Parse SQL for lineage
sql = """
INSERT INTO analytics.user_metrics
SELECT 
    u.user_id,
    u.email,
    COUNT(o.order_id) as order_count,
    SUM(o.total_amount) as total_spent
FROM ecommerce.users u
LEFT JOIN ecommerce.orders o ON u.user_id = o.user_id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.user_id, u.email
"""

parser = SqlLineageParser(sql)
lineage = parser.get_lineage()

# lineage.inputs: ["ecommerce.users", "ecommerce.orders"]
# lineage.output: "analytics.user_metrics"
# lineage.column_lineage: {
#     "analytics.user_metrics.user_id": ["ecommerce.users.user_id"],
#     "analytics.user_metrics.email": ["ecommerce.users.email"],
#     "analytics.user_metrics.order_count": ["ecommerce.orders.order_id"],
#     "analytics.user_metrics.total_spent": ["ecommerce.orders.total_amount"]
# }
```

### Amundsen Implementation Deep Dive

#### 1. Neo4j Graph Model

**Node Types**:
```cypher
// Table node
CREATE (t:Table {
    key: 'postgres://ecommerce.users',
    name: 'users',
    schema: 'ecommerce',
    database: 'postgres',
    description: 'User account information'
})

// Column nodes
CREATE (c1:Column {
    key: 'postgres://ecommerce.users/user_id',
    name: 'user_id',
    type: 'bigint',
    description: 'Unique user identifier'
})

CREATE (c2:Column {
    key: 'postgres://ecommerce.users/email',
    name: 'email', 
    type: 'varchar',
    description: 'User email address'
})

// Relationships
CREATE (t)-[:COLUMN]->(c1)
CREATE (t)-[:COLUMN]->(c2)
```

**Lineage Relationships**:
```cypher
// Upstream/Downstream relationships
MATCH (source:Table {key: 'postgres://raw.events'})
MATCH (target:Table {key: 'postgres://processed.user_events'})
CREATE (source)-[:UPSTREAM_OF]->(target)
CREATE (target)-[:DOWNSTREAM_OF]->(source)
```

#### 2. Metadata Extraction

**Custom Extractor**:
```python
from pyhocon import ConfigFactory
from amundsen_databuilder.extractor.base_extractor import Extractor
from amundsen_databuilder.models.table_metadata import TableMetadata, ColumnMetadata

class CustomDatabaseExtractor(Extractor):
    """Extract metadata from custom database"""
    
    def init(self, conf: ConfigFactory) -> None:
        self.connection_string = conf.get_string('connection_string')
        self.database_name = conf.get_string('database_name')
    
    def extract(self) -> Iterator[TableMetadata]:
        """Extract table and column metadata"""
        
        # Connect to database
        conn = create_connection(self.connection_string)
        
        # Query table metadata
        tables_query = """
        SELECT 
            table_schema,
            table_name,
            table_comment
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        """
        
        for table_row in conn.execute(tables_query):
            schema_name = table_row['table_schema']
            table_name = table_row['table_name']
            description = table_row['table_comment']
            
            # Query column metadata
            columns_query = """
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_comment
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """
            
            columns = []
            for col_row in conn.execute(columns_query, [schema_name, table_name]):
                column = ColumnMetadata(
                    name=col_row['column_name'],
                    description=col_row['column_comment'],
                    col_type=col_row['data_type'],
                    sort_order=len(columns)
                )
                columns.append(column)
            
            # Create table metadata
            table_metadata = TableMetadata(
                database=self.database_name,
                cluster='default',
                schema=schema_name,
                name=table_name,
                description=description,
                columns=columns
            )
            
            yield table_metadata
```

#### 3. Search Integration

**Elasticsearch Indexing**:
```python
from amundsen_databuilder.publisher.elasticsearch_publisher import ElasticsearchPublisher

# Configure Elasticsearch publisher
elasticsearch_config = ConfigFactory.from_dict({
    'publisher.elasticsearch.host': 'localhost:9200',
    'publisher.elasticsearch.index': 'table_search_index',
    'publisher.elasticsearch.doc_type': '_doc'
})

publisher = ElasticsearchPublisher()
publisher.init(elasticsearch_config)

# Publish table metadata to Elasticsearch
for table_metadata in extracted_tables:
    publisher.publish_record(table_metadata)
```

### Production Deployment Patterns

#### 1. DataHub Production Setup

**Docker Compose Configuration**:
```yaml
version: '3.8'
services:
  # Elasticsearch for search
  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  # MySQL for metadata storage
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_DATABASE: datahub
      MYSQL_USER: datahub
      MYSQL_PASSWORD: datahub
      MYSQL_ROOT_PASSWORD: datahub
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

  # Kafka for real-time updates
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"

  # DataHub GMS (backend)
  datahub-gms:
    image: linkedin/datahub-gms:v0.11.0
    environment:
      - EBEAN_DATASOURCE_URL=jdbc:mysql://mysql:3306/datahub
      - EBEAN_DATASOURCE_USERNAME=datahub
      - EBEAN_DATASOURCE_PASSWORD=datahub
      - KAFKA_BOOTSTRAP_SERVER=kafka:9092
      - ELASTICSEARCH_HOST=elasticsearch
      - ELASTICSEARCH_PORT=9200
    ports:
      - "8080:8080"
    depends_on:
      - mysql
      - elasticsearch
      - kafka

  # DataHub Frontend
  datahub-frontend:
    image: linkedin/datahub-frontend-react:v0.11.0
    environment:
      - DATAHUB_GMS_HOST=datahub-gms
      - DATAHUB_GMS_PORT=8080
    ports:
      - "9002:9002"
    depends_on:
      - datahub-gms

volumes:
  elasticsearch_data:
  mysql_data:
```

#### 2. Amundsen Production Setup

**Kubernetes Deployment**:
```yaml
# amundsen-frontend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: amundsen-frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: amundsen-frontend
  template:
    metadata:
      labels:
        app: amundsen-frontend
    spec:
      containers:
      - name: frontend
        image: amundsendev/amundsen-frontend:latest
        ports:
        - containerPort: 5000
        env:
        - name: METADATA_SERVICE_URL
          value: "http://amundsen-metadata:5002"
        - name: SEARCH_SERVICE_URL
          value: "http://amundsen-search:5001"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"

---
apiVersion: v1
kind: Service
metadata:
  name: amundsen-frontend
spec:
  selector:
    app: amundsen-frontend
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

#### 3. High Availability Considerations

**DataHub HA Setup**:
```yaml
# Multiple GMS instances behind load balancer
datahub-gms:
  image: linkedin/datahub-gms:latest
  deploy:
    replicas: 3
    update_config:
      parallelism: 1
      delay: 30s
    restart_policy:
      condition: on-failure
  environment:
    - EBEAN_DATASOURCE_URL=jdbc:mysql://mysql-cluster:3306/datahub
    - KAFKA_BOOTSTRAP_SERVER=kafka-cluster:9092
    - ELASTICSEARCH_HOST=elasticsearch-cluster
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

**Database Clustering**:
```yaml
# MySQL cluster for DataHub
mysql-primary:
  image: mysql:8.0
  environment:
    MYSQL_REPLICATION_MODE: master
    MYSQL_REPLICATION_USER: replicator
    MYSQL_REPLICATION_PASSWORD: replicator_password

mysql-secondary:
  image: mysql:8.0
  environment:
    MYSQL_REPLICATION_MODE: slave
    MYSQL_REPLICATION_USER: replicator
    MYSQL_REPLICATION_PASSWORD: replicator_password
    MYSQL_MASTER_HOST: mysql-primary
```

### Integration Patterns

#### 1. CI/CD Integration

**Automated Metadata Updates**:
```yaml
# .github/workflows/metadata-update.yml
name: Update Data Catalog
on:
  push:
    paths:
      - 'sql/**'
      - 'dbt/**'
      - 'airflow/dags/**'

jobs:
  update-catalog:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Extract DBT Metadata
      run: |
        dbt docs generate
        python scripts/extract_dbt_metadata.py
    
    - name: Update DataHub
      run: |
        datahub ingest -c datahub_config.yml
    
    - name: Validate Lineage
      run: |
        python scripts/validate_lineage.py
```

#### 2. Data Quality Integration

**Great Expectations Integration**:
```python
from great_expectations.core.batch import RuntimeBatchRequest
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.metadata.schema_classes import DatasetPropertiesClass

def update_quality_metrics(dataset_name: str, validation_results: dict):
    """Update data catalog with quality metrics"""
    
    # Calculate quality scores
    quality_score = calculate_overall_score(validation_results)
    
    # Create dataset URN
    dataset_urn = make_dataset_urn(platform="postgres", name=dataset_name)
    
    # Update custom properties with quality metrics
    properties = DatasetPropertiesClass(
        customProperties={
            "data_quality_score": str(quality_score),
            "last_quality_check": str(datetime.now().isoformat()),
            "quality_rules_passed": str(validation_results['success_count']),
            "quality_rules_failed": str(validation_results['failure_count'])
        }
    )
    
    # Emit to DataHub
    mcp = MetadataChangeProposalWrapper(
        entityType="dataset",
        entityUrn=dataset_urn,
        aspectName="datasetProperties", 
        aspect=properties
    )
    
    emitter.emit_mcp(mcp)
```

#### 3. Airflow Integration

**Automatic Lineage Extraction**:
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datahub_airflow_plugin.entities import Dataset, Task

def extract_and_transform():
    """ETL task with automatic lineage tracking"""
    
    # Define inputs and outputs for lineage
    inputs = [
        Dataset("postgres", "raw.user_events"),
        Dataset("postgres", "raw.user_profiles")
    ]
    
    outputs = [
        Dataset("postgres", "processed.user_analytics")
    ]
    
    # Your ETL logic here
    result = perform_etl()
    
    return result

dag = DAG(
    'user_analytics_pipeline',
    default_args=default_args,
    schedule_interval='@hourly'
)

# Task with automatic lineage tracking
etl_task = PythonOperator(
    task_id='extract_and_transform',
    python_callable=extract_and_transform,
    dag=dag,
    # DataHub plugin automatically extracts lineage
    inlets=[
        Dataset("postgres", "raw.user_events"),
        Dataset("postgres", "raw.user_profiles")
    ],
    outlets=[
        Dataset("postgres", "processed.user_analytics")
    ]
)
```

### Monitoring and Observability

#### 1. Catalog Health Metrics

**Key Metrics to Track**:
```python
# Metadata freshness
metadata_freshness_hours = {
    "postgres_tables": 2,
    "s3_datasets": 6, 
    "kafka_topics": 1,
    "dbt_models": 4
}

# Usage metrics
catalog_usage_metrics = {
    "daily_active_users": 150,
    "searches_per_day": 2500,
    "dataset_views_per_day": 1800,
    "lineage_queries_per_day": 400
}

# Data quality coverage
quality_coverage = {
    "tables_with_quality_checks": 0.75,
    "columns_with_descriptions": 0.60,
    "datasets_with_owners": 0.85,
    "datasets_with_tags": 0.70
}
```

#### 2. Alerting Rules

**Prometheus Alerts**:
```yaml
groups:
- name: data_catalog
  rules:
  - alert: MetadataIngestionFailed
    expr: datahub_ingestion_success == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "DataHub metadata ingestion failed"
      description: "Metadata ingestion for {{ $labels.source }} has been failing for 5 minutes"

  - alert: CatalogSearchDown
    expr: up{job="amundsen-search"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Amundsen search service is down"
      
  - alert: LowMetadataFreshness
    expr: (time() - datahub_last_ingestion_timestamp) > 86400
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Metadata is stale"
      description: "{{ $labels.source }} metadata hasn't been updated in over 24 hours"
```

### Best Practices

#### 1. Metadata Governance

**Ownership Model**:
```yaml
# Data ownership configuration
ownership_rules:
  - pattern: "finance.*"
    owner: "finance-team@company.com"
    domain: "finance"
    
  - pattern: "marketing.*"
    owner: "marketing-team@company.com" 
    domain: "marketing"
    
  - pattern: "*.user_*"
    tags: ["pii", "gdpr"]
    retention_policy: "7_years"
```

**Tagging Strategy**:
```python
# Standardized tag taxonomy
TAG_TAXONOMY = {
    "data_classification": ["public", "internal", "confidential", "restricted"],
    "data_type": ["pii", "financial", "operational", "analytical"],
    "compliance": ["gdpr", "ccpa", "sox", "hipaa"],
    "quality": ["verified", "experimental", "deprecated"],
    "domain": ["finance", "marketing", "product", "engineering"]
}
```

#### 2. User Adoption Strategies

**Onboarding Workflow**:
```python
def onboard_new_dataset(dataset_info: dict):
    """Standardized dataset onboarding process"""
    
    # 1. Validate required metadata
    required_fields = ["description", "owner", "domain", "tags"]
    for field in required_fields:
        if field not in dataset_info:
            raise ValueError(f"Missing required field: {field}")
    
    # 2. Apply governance policies
    apply_governance_policies(dataset_info)
    
    # 3. Generate data profile
    profile = generate_data_profile(dataset_info["location"])
    
    # 4. Create catalog entry
    create_catalog_entry(dataset_info, profile)
    
    # 5. Set up monitoring
    setup_quality_monitoring(dataset_info)
    
    # 6. Notify stakeholders
    notify_stakeholders(dataset_info)
```

#### 3. Performance Optimization

**Search Optimization**:
```python
# Elasticsearch index optimization for Amundsen
{
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "table_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "name": {
                "type": "text",
                "analyzer": "table_analyzer",
                "boost": 3.0
            },
            "description": {
                "type": "text", 
                "analyzer": "table_analyzer",
                "boost": 2.0
            },
            "column_names": {
                "type": "text",
                "analyzer": "table_analyzer",
                "boost": 1.5
            }
        }
    }
}
```

---

## 💻 Hands-On Exercise

See `exercise.py` for hands-on practice with data catalog implementation.

**What you'll build**:
1. Set up DataHub locally using Docker Compose
2. Implement custom metadata extractors for PostgreSQL and CSV files
3. Create data lineage tracking between datasets
4. Build data quality integration with metrics
5. Set up search and discovery workflows
6. Implement governance policies and validation

**Prerequisites**:
- Docker and Docker Compose installed
- Python 3.8+ with pip
- 8GB+ RAM available for Docker
- Review `SETUP.md` for detailed instructions

**Expected time**: 45 minutes

**Quick Start**:
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Start DataHub (see SETUP.md for complete docker-compose.yml)
docker-compose up -d

# Run exercise
python exercise.py
```

---

## 📚 Resources

- [DataHub Documentation](https://datahubproject.io/docs/)
- [Amundsen Documentation](https://www.amundsen.io/amundsen/)
- [Data Catalog Best Practices](https://www.oreilly.com/library/view/data-catalogs-for/9781492055914/)
- [Metadata Management Patterns](https://martinfowler.com/articles/data-monolith-to-mesh.html)
- [Data Discovery at Scale](https://eng.uber.com/databook/)

---

## 🎯 Key Takeaways

- **Data catalogs solve critical discovery and governance challenges** in modern data architectures
- **DataHub excels in real-time updates and API richness** while Amundsen focuses on user experience
- **Metadata management requires both technical and business context** for maximum value
- **Automated lineage tracking** reduces manual effort and improves accuracy
- **Integration with existing tools** (Airflow, dbt, Great Expectations) is essential
- **User adoption depends on ease of use** and clear value proposition
- **Governance policies must be enforced** through the catalog, not just documented
- **Search and discovery capabilities** are the primary user-facing features
- **Quality metrics integration** provides trust and confidence in data assets
- **Production deployment requires** high availability and performance considerations

---

## 🚀 What's Next?

Tomorrow (Day 9), you'll learn about **Data Lineage Tracking** - diving deeper into automated lineage extraction, impact analysis, and building comprehensive data flow documentation across your entire data ecosystem.

**Preview**: Data lineage provides:
- End-to-end data flow visualization
- Impact analysis for changes
- Root cause analysis for data issues
- Compliance and audit trails

---

## ✅ Before Moving On

- [ ] Understand the role and value of data catalogs
- [ ] Can compare DataHub vs Amundsen capabilities
- [ ] Know how to implement metadata extraction
- [ ] Understand data lineage concepts and implementation
- [ ] Can integrate catalogs with existing data tools
- [ ] Complete the exercise in `exercise.py`
- [ ] Review the solution in `solution.py`
- [ ] Take the quiz in `quiz.md`

**Time**: ~1 hour | **Difficulty**: ⭐⭐⭐ (Intermediate)

## 📋 Additional Resources

### Setup & Configuration
- [SETUP.md](./SETUP.md) - Complete DataHub setup guide
- [requirements.txt](./requirements.txt) - Python dependencies
- [.env.example](./.env.example) - Environment configuration template

### Sample Data
- [sample_data/](./sample_data/) - CSV files for testing metadata extraction
- Sample PostgreSQL schema in exercise examples

### Documentation
- [DataHub Quickstart](https://datahubproject.io/docs/quickstart)
- [DataHub Python SDK](https://datahubproject.io/docs/metadata-ingestion/as-a-library)
- [Amundsen Documentation](https://www.amundsen.io/amundsen/)

Ready to organize and govern your data assets! 🚀
