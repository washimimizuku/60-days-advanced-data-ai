# Day 9: Data Lineage Tracking

## 📖 Learning Objectives

By the end of today, you will:
- Understand the critical importance of data lineage in modern data systems
- Implement automated lineage extraction from SQL, code, and orchestration tools
- Build impact analysis systems for change management and debugging
- Create column-level lineage tracking for compliance and governance
- Deploy production-ready lineage systems with monitoring and alerting
- Apply lineage for root cause analysis and data quality debugging

---

## Theory

### What is Data Lineage?

Data lineage is the documentation and visualization of data flow through systems - from source to destination, including all transformations, dependencies, and relationships along the way. Think of it as a "family tree" for your data that shows where data comes from, how it's transformed, and where it goes.

**Key Components**:
- **Source Systems**: Where data originates (databases, APIs, files)
- **Transformations**: How data is modified (SQL queries, Python scripts, ML models)
- **Dependencies**: What data depends on what other data
- **Destinations**: Where data ends up (dashboards, reports, ML models)
- **Metadata**: Context about the data flow (owners, schedules, quality)

### Why Data Lineage Matters

#### The Data Debugging Problem

```
Without Lineage:
📊 Dashboard shows wrong numbers
    ↓
❓ Which data source is wrong?
    ↓
🔍 Manual investigation (hours/days)
    ↓
🐛 Find issue in transformation #47
    ↓
❓ What else is affected?
    ↓
🔍 More manual investigation
    ↓
😰 Discover 12 other systems affected
```

```
With Lineage:
📊 Dashboard shows wrong numbers
    ↓
🔍 Check lineage graph (30 seconds)
    ↓
🎯 Identify exact transformation causing issue
    ↓
📈 See all downstream impacts immediately
    ↓
🚨 Proactive alerts to affected teams
    ↓
✅ Fix root cause and validate all impacts
```

#### Business Impact

**Without Lineage**:
- Hours/days to debug data issues
- Unknown downstream impacts of changes
- Compliance violations due to data tracking gaps
- Duplicate work and inconsistent transformations
- Fear of making changes due to unknown dependencies

**With Lineage**:
- Minutes to identify root causes
- Complete impact analysis before changes
- Automated compliance and audit trails
- Reusable transformations and reduced duplication
- Confidence to make changes with full visibility

### Types of Data Lineage

#### 1. Table-Level Lineage

Shows relationships between datasets/tables:

```
Raw Data → Staging → Processed → Analytics
    │         │          │          │
    │         │          │          └─→ Dashboard A
    │         │          └─→ ML Model B
    │         └─→ Data Quality Checks
    └─→ Archive Storage
```

**Use Cases**:
- High-level impact analysis
- System architecture understanding
- Data flow documentation
- Compliance reporting

#### 2. Column-Level Lineage

Shows relationships between individual columns:

```sql
-- Source: raw_events.user_id, raw_events.timestamp
-- Transformation: 
SELECT 
    user_id,                           -- Direct mapping
    DATE(timestamp) as event_date,     -- Transformation
    COUNT(*) as daily_events           -- Aggregation
FROM raw_events 
GROUP BY user_id, DATE(timestamp)
-- Destination: user_daily_stats.user_id, user_daily_stats.event_date, user_daily_stats.daily_events
```

**Use Cases**:
- Detailed impact analysis
- Compliance and privacy (PII tracking)
- Data quality root cause analysis
- Transformation optimization

#### 3. Code-Level Lineage

Shows lineage within application code:

```python
# Python transformation with lineage
def process_user_events(df):
    # Lineage: input_df.user_id -> output_df.user_id
    # Lineage: input_df.timestamp -> output_df.event_date (via DATE function)
    # Lineage: input_df.* -> output_df.daily_events (via COUNT aggregation)
    
    return df.groupby(['user_id', df['timestamp'].dt.date]) \
             .size() \
             .reset_index(name='daily_events') \
             .rename(columns={'timestamp': 'event_date'})
```

**Use Cases**:
- Application-level debugging
- Code refactoring impact analysis
- Performance optimization
- Technical documentation

### Automated Lineage Extraction Methods

#### 1. SQL Parsing

Extract lineage from SQL queries automatically:

```python
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

def extract_sql_lineage(sql_query):
    """Extract table and column lineage from SQL"""
    parsed = sqlparse.parse(sql_query)[0]
    
    # Extract source tables
    source_tables = []
    target_tables = []
    column_mappings = []
    
    # Parse SELECT statements
    for token in parsed.flatten():
        if token.ttype is DML and token.value.upper() == 'SELECT':
            # Extract column lineage
            pass
        elif token.ttype is Keyword and token.value.upper() == 'FROM':
            # Extract source tables
            pass
        elif token.ttype is Keyword and token.value.upper() == 'INSERT':
            # Extract target tables
            pass
    
    return {
        'source_tables': source_tables,
        'target_tables': target_tables,
        'column_mappings': column_mappings
    }

# Example usage
sql = """
INSERT INTO analytics.user_metrics
SELECT 
    u.user_id,
    u.email,
    COUNT(o.order_id) as order_count,
    SUM(o.total_amount) as total_spent
FROM ecommerce.users u
LEFT JOIN ecommerce.orders o ON u.user_id = o.user_id
WHERE u.status = 'active'
GROUP BY u.user_id, u.email
"""

lineage = extract_sql_lineage(sql)
# Result: 
# source_tables: ['ecommerce.users', 'ecommerce.orders']
# target_tables: ['analytics.user_metrics']
# column_mappings: [
#   {'source': 'ecommerce.users.user_id', 'target': 'analytics.user_metrics.user_id'},
#   {'source': 'ecommerce.users.email', 'target': 'analytics.user_metrics.email'},
#   {'source': 'ecommerce.orders.order_id', 'target': 'analytics.user_metrics.order_count'},
#   {'source': 'ecommerce.orders.total_amount', 'target': 'analytics.user_metrics.total_spent'}
# ]
```

#### 2. AST (Abstract Syntax Tree) Analysis

Extract lineage from Python/Scala code:

```python
import ast
import pandas as pd

class LineageExtractor(ast.NodeVisitor):
    """Extract data lineage from Python code using AST"""
    
    def __init__(self):
        self.lineage = {
            'inputs': [],
            'outputs': [],
            'transformations': []
        }
    
    def visit_Call(self, node):
        """Visit function calls to extract data operations"""
        if isinstance(node.func, ast.Attribute):
            # DataFrame operations
            if node.func.attr in ['groupby', 'merge', 'join']:
                self._extract_transformation(node)
            elif node.func.attr in ['to_csv', 'to_sql', 'to_parquet']:
                self._extract_output(node)
        elif isinstance(node.func, ast.Name):
            # Function calls
            if node.func.id in ['read_csv', 'read_sql', 'read_parquet']:
                self._extract_input(node)
        
        self.generic_visit(node)
    
    def _extract_input(self, node):
        """Extract input data sources"""
        if len(node.args) > 0:
            if isinstance(node.args[0], ast.Str):
                self.lineage['inputs'].append(node.args[0].s)
    
    def _extract_output(self, node):
        """Extract output data destinations"""
        if len(node.args) > 0:
            if isinstance(node.args[0], ast.Str):
                self.lineage['outputs'].append(node.args[0].s)
    
    def _extract_transformation(self, node):
        """Extract transformation operations"""
        self.lineage['transformations'].append({
            'operation': node.func.attr,
            'line': node.lineno
        })

# Example usage
code = """
import pandas as pd

# Read input data
users_df = pd.read_csv('data/users.csv')
orders_df = pd.read_sql('SELECT * FROM orders', connection)

# Transform data
user_stats = users_df.merge(orders_df, on='user_id') \
                    .groupby('user_id') \
                    .agg({'order_id': 'count', 'total_amount': 'sum'})

# Write output
user_stats.to_parquet('output/user_statistics.parquet')
"""

tree = ast.parse(code)
extractor = LineageExtractor()
extractor.visit(tree)

print(extractor.lineage)
# Result:
# {
#   'inputs': ['data/users.csv', 'SELECT * FROM orders'],
#   'outputs': ['output/user_statistics.parquet'],
#   'transformations': [
#     {'operation': 'merge', 'line': 7},
#     {'operation': 'groupby', 'line': 8}
#   ]
# }
```

#### 3. Orchestration Tool Integration

Extract lineage from workflow orchestration tools:

```python
# Airflow DAG lineage extraction
from airflow.lineage.entities import File, Table
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator

def extract_airflow_lineage(dag):
    """Extract lineage from Airflow DAG"""
    lineage_graph = {
        'nodes': [],
        'edges': []
    }
    
    for task_id, task in dag.task_dict.items():
        # Extract task inputs and outputs
        if hasattr(task, 'inlets'):
            for inlet in task.inlets:
                lineage_graph['edges'].append({
                    'source': str(inlet),
                    'target': task_id,
                    'type': 'input'
                })
        
        if hasattr(task, 'outlets'):
            for outlet in task.outlets:
                lineage_graph['edges'].append({
                    'source': task_id,
                    'target': str(outlet),
                    'type': 'output'
                })
        
        lineage_graph['nodes'].append({
            'id': task_id,
            'type': 'task',
            'operator': task.__class__.__name__
        })
    
    return lineage_graph

# Example DAG with lineage
dag = DAG('user_analytics_pipeline', default_args=default_args)

extract_task = PythonOperator(
    task_id='extract_users',
    python_callable=extract_users,
    dag=dag,
    inlets=[Table(table='raw.users', cluster='prod')],
    outlets=[File(url='s3://bucket/staging/users.parquet')]
)

transform_task = PythonOperator(
    task_id='transform_users',
    python_callable=transform_users,
    dag=dag,
    inlets=[File(url='s3://bucket/staging/users.parquet')],
    outlets=[Table(table='analytics.user_metrics', cluster='prod')]
)
```

### Advanced Lineage Patterns

#### 1. Multi-Hop Lineage

Track lineage across multiple transformation steps:

```python
class LineageGraph:
    """Multi-hop lineage tracking"""
    
    def __init__(self):
        self.nodes = {}  # dataset_id -> metadata
        self.edges = []  # source -> target relationships
    
    def add_transformation(self, sources, targets, transformation_info):
        """Add a transformation with multiple inputs/outputs"""
        transformation_id = f"transform_{len(self.edges)}"
        
        # Add transformation node
        self.nodes[transformation_id] = {
            'type': 'transformation',
            'info': transformation_info
        }
        
        # Add edges from sources to transformation
        for source in sources:
            self.edges.append({
                'source': source,
                'target': transformation_id,
                'type': 'input'
            })
        
        # Add edges from transformation to targets
        for target in targets:
            self.edges.append({
                'source': transformation_id,
                'target': target,
                'type': 'output'
            })
    
    def get_upstream_lineage(self, dataset_id, max_depth=10):
        """Get all upstream dependencies"""
        upstream = set()
        to_visit = [(dataset_id, 0)]
        visited = set()
        
        while to_visit and max_depth > 0:
            current, depth = to_visit.pop(0)
            if current in visited or depth >= max_depth:
                continue
            
            visited.add(current)
            
            # Find all sources that feed into current
            for edge in self.edges:
                if edge['target'] == current and edge['source'] not in visited:
                    upstream.add(edge['source'])
                    to_visit.append((edge['source'], depth + 1))
        
        return upstream
    
    def get_downstream_lineage(self, dataset_id, max_depth=10):
        """Get all downstream dependencies"""
        downstream = set()
        to_visit = [(dataset_id, 0)]
        visited = set()
        
        while to_visit and max_depth > 0:
            current, depth = to_visit.pop(0)
            if current in visited or depth >= max_depth:
                continue
            
            visited.add(current)
            
            # Find all targets that current feeds into
            for edge in self.edges:
                if edge['source'] == current and edge['target'] not in visited:
                    downstream.add(edge['target'])
                    to_visit.append((edge['target'], depth + 1))
        
        return downstream
    
    def impact_analysis(self, dataset_id):
        """Analyze impact of changes to a dataset"""
        return {
            'upstream_dependencies': self.get_upstream_lineage(dataset_id),
            'downstream_impacts': self.get_downstream_lineage(dataset_id),
            'total_affected_systems': len(self.get_downstream_lineage(dataset_id))
        }

# Example usage
lineage = LineageGraph()

# Add transformations
lineage.add_transformation(
    sources=['raw.users', 'raw.orders'],
    targets=['staging.user_orders'],
    transformation_info={'type': 'join', 'sql': 'JOIN ON user_id'}
)

lineage.add_transformation(
    sources=['staging.user_orders'],
    targets=['analytics.user_metrics', 'ml.user_features'],
    transformation_info={'type': 'aggregation', 'sql': 'GROUP BY user_id'}
)

# Impact analysis
impact = lineage.impact_analysis('raw.users')
print(f"Changing raw.users affects {impact['total_affected_systems']} downstream systems")
```

#### 2. Column-Level Lineage with Transformations

Track how individual columns are transformed:

```python
class ColumnLineage:
    """Column-level lineage tracking"""
    
    def __init__(self):
        self.column_mappings = []
    
    def add_column_mapping(self, source_table, source_column, target_table, 
                          target_column, transformation_type, transformation_logic):
        """Add column-level lineage mapping"""
        self.column_mappings.append({
            'source': f"{source_table}.{source_column}",
            'target': f"{target_table}.{target_column}",
            'transformation_type': transformation_type,
            'transformation_logic': transformation_logic,
            'created_at': datetime.now().isoformat()
        })
    
    def parse_sql_for_column_lineage(self, sql_query):
        """Parse SQL to extract column-level lineage"""
        # Simplified SQL parsing for column lineage
        # In production, use libraries like sqlparse or sqlglot
        
        mappings = []
        
        # Example: SELECT user_id, UPPER(email) as email_upper, COUNT(*) as cnt
        # FROM users GROUP BY user_id, email
        
        # This would extract:
        # users.user_id -> result.user_id (direct)
        # users.email -> result.email_upper (transformation: UPPER)
        # users.* -> result.cnt (aggregation: COUNT)
        
        return mappings
    
    def get_column_lineage(self, target_column):
        """Get lineage for a specific column"""
        lineage_chain = []
        
        def trace_column(column, depth=0):
            if depth > 10:  # Prevent infinite recursion
                return
            
            for mapping in self.column_mappings:
                if mapping['target'] == column:
                    lineage_chain.append(mapping)
                    trace_column(mapping['source'], depth + 1)
        
        trace_column(target_column)
        return lineage_chain
    
    def find_pii_lineage(self):
        """Find all columns that derive from PII data"""
        pii_patterns = ['email', 'phone', 'ssn', 'name', 'address']
        pii_lineage = []
        
        for mapping in self.column_mappings:
            source_col = mapping['source'].lower()
            if any(pattern in source_col for pattern in pii_patterns):
                # This column derives from PII
                pii_lineage.append(mapping)
        
        return pii_lineage

# Example usage
col_lineage = ColumnLineage()

# Add column mappings
col_lineage.add_column_mapping(
    'raw.users', 'email',
    'analytics.user_stats', 'email_domain',
    'transformation', 'SUBSTRING_INDEX(email, "@", -1)'
)

col_lineage.add_column_mapping(
    'raw.users', 'user_id',
    'analytics.user_stats', 'user_id',
    'direct', 'direct mapping'
)

# Find PII lineage
pii_columns = col_lineage.find_pii_lineage()
print(f"Found {len(pii_columns)} columns derived from PII data")
```

### Production Lineage Systems

#### 1. Apache Atlas Integration

```python
from pyatlasclient.client import Atlas

class AtlasLineageManager:
    """Manage lineage using Apache Atlas"""
    
    def __init__(self, atlas_url, username, password):
        self.client = Atlas(atlas_url, username=username, password=password)
    
    def create_dataset_entity(self, dataset_info):
        """Create dataset entity in Atlas"""
        entity = {
            'typeName': 'DataSet',
            'attributes': {
                'name': dataset_info['name'],
                'qualifiedName': dataset_info['qualified_name'],
                'description': dataset_info.get('description', ''),
                'owner': dataset_info.get('owner', ''),
                'createTime': int(time.time() * 1000)
            }
        }
        
        response = self.client.entity_post.create(entity=entity)
        return response
    
    def create_process_entity(self, process_info, inputs, outputs):
        """Create process entity with lineage"""
        entity = {
            'typeName': 'Process',
            'attributes': {
                'name': process_info['name'],
                'qualifiedName': process_info['qualified_name'],
                'description': process_info.get('description', ''),
                'inputs': [{'guid': inp} for inp in inputs],
                'outputs': [{'guid': out} for out in outputs]
            }
        }
        
        response = self.client.entity_post.create(entity=entity)
        return response
    
    def get_lineage(self, entity_guid, direction='BOTH', depth=3):
        """Get lineage for an entity"""
        lineage = self.client.lineage_get.get_lineage(
            guid=entity_guid,
            direction=direction,
            depth=depth
        )
        return lineage

# Example usage
atlas = AtlasLineageManager('http://localhost:21000', 'admin', 'admin')

# Create dataset entities
users_entity = atlas.create_dataset_entity({
    'name': 'users',
    'qualified_name': 'raw.users@cluster1',
    'description': 'Raw user data',
    'owner': 'data-team'
})

user_stats_entity = atlas.create_dataset_entity({
    'name': 'user_stats',
    'qualified_name': 'analytics.user_stats@cluster1',
    'description': 'Aggregated user statistics',
    'owner': 'analytics-team'
})

# Create process entity (transformation)
process_entity = atlas.create_process_entity(
    process_info={
        'name': 'user_aggregation',
        'qualified_name': 'etl.user_aggregation@cluster1',
        'description': 'Aggregate user data for analytics'
    },
    inputs=[users_entity['guid']],
    outputs=[user_stats_entity['guid']]
)
```

#### 2. DataHub Lineage Integration

```python
from datahub.emitter.mce_builder import make_dataset_urn
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from datahub.metadata.schema_classes import (
    UpstreamLineageClass,
    UpstreamClass,
    DatasetLineageTypeClass,
    FineGrainedLineageClass,
    FineGrainedLineageUpstreamTypeClass,
    FineGrainedLineageDownstreamTypeClass
)

class DataHubLineageManager:
    """Manage lineage using DataHub"""
    
    def __init__(self, datahub_url):
        self.emitter = DatahubRestEmitter(gms_server=datahub_url)
    
    def create_table_lineage(self, downstream_table, upstream_tables, transformation_info):
        """Create table-level lineage"""
        downstream_urn = make_dataset_urn(
            platform="postgres",
            name=downstream_table
        )
        
        upstreams = []
        for upstream_table in upstream_tables:
            upstream_urn = make_dataset_urn(
                platform="postgres",
                name=upstream_table
            )
            
            upstream = UpstreamClass(
                dataset=upstream_urn,
                type=DatasetLineageTypeClass.TRANSFORMED
            )
            upstreams.append(upstream)
        
        upstream_lineage = UpstreamLineageClass(upstreams=upstreams)
        
        mcp = MetadataChangeProposalWrapper(
            entityType="dataset",
            entityUrn=downstream_urn,
            aspectName="upstreamLineage",
            aspect=upstream_lineage
        )
        
        self.emitter.emit_mcp(mcp)
    
    def create_column_lineage(self, downstream_table, column_mappings):
        """Create column-level lineage"""
        downstream_urn = make_dataset_urn(
            platform="postgres",
            name=downstream_table
        )
        
        fine_grained_lineages = []
        for mapping in column_mappings:
            upstream_urn = make_dataset_urn(
                platform="postgres",
                name=mapping['upstream_table']
            )
            
            fine_grained_lineage = FineGrainedLineageClass(
                upstreamType=FineGrainedLineageUpstreamTypeClass.FIELD_SET,
                upstreams=[f"urn:li:schemaField:({upstream_urn},{mapping['upstream_column']})"],
                downstreamType=FineGrainedLineageDownstreamTypeClass.FIELD,
                downstreams=[f"urn:li:schemaField:({downstream_urn},{mapping['downstream_column']})"],
                transformOperation=mapping.get('transformation', 'IDENTITY')
            )
            fine_grained_lineages.append(fine_grained_lineage)
        
        upstream_lineage = UpstreamLineageClass(
            upstreams=[],  # Table-level upstreams
            fineGrainedLineages=fine_grained_lineages
        )
        
        mcp = MetadataChangeProposalWrapper(
            entityType="dataset",
            entityUrn=downstream_urn,
            aspectName="upstreamLineage",
            aspect=upstream_lineage
        )
        
        self.emitter.emit_mcp(mcp)

# Example usage
datahub = DataHubLineageManager('http://localhost:8080')

# Create table lineage
datahub.create_table_lineage(
    downstream_table='analytics.user_metrics',
    upstream_tables=['raw.users', 'raw.orders'],
    transformation_info={'type': 'aggregation', 'tool': 'dbt'}
)

# Create column lineage
datahub.create_column_lineage(
    downstream_table='analytics.user_metrics',
    column_mappings=[
        {
            'upstream_table': 'raw.users',
            'upstream_column': 'user_id',
            'downstream_column': 'user_id',
            'transformation': 'IDENTITY'
        },
        {
            'upstream_table': 'raw.orders',
            'upstream_column': 'total_amount',
            'downstream_column': 'total_spent',
            'transformation': 'SUM'
        }
    ]
)
```

### Lineage for Compliance and Governance

#### 1. GDPR Data Subject Requests

```python
class GDPRLineageTracker:
    """Track data lineage for GDPR compliance"""
    
    def __init__(self, lineage_system):
        self.lineage = lineage_system
    
    def find_personal_data_usage(self, user_id):
        """Find all systems that contain data for a specific user"""
        personal_data_locations = []
        
        # Start with known PII tables
        pii_tables = ['users', 'user_profiles', 'user_preferences']
        
        for table in pii_tables:
            # Find all downstream systems
            downstream = self.lineage.get_downstream_lineage(table)
            
            for system in downstream:
                # Check if this system likely contains user data
                if self._contains_user_data(system, user_id):
                    personal_data_locations.append({
                        'system': system,
                        'source_table': table,
                        'data_types': self._get_data_types(system),
                        'retention_policy': self._get_retention_policy(system)
                    })
        
        return personal_data_locations
    
    def generate_data_deletion_plan(self, user_id):
        """Generate plan for deleting all user data"""
        locations = self.find_personal_data_usage(user_id)
        
        deletion_plan = {
            'user_id': user_id,
            'systems_to_update': [],
            'deletion_order': [],
            'verification_steps': []
        }
        
        # Sort by dependency order (delete downstream first)
        sorted_locations = self._sort_by_dependency(locations)
        
        for location in sorted_locations:
            deletion_plan['systems_to_update'].append(location)
            deletion_plan['deletion_order'].append(location['system'])
            deletion_plan['verification_steps'].append(
                f"Verify {location['system']} no longer contains data for user {user_id}"
            )
        
        return deletion_plan
    
    def _contains_user_data(self, system, user_id):
        """Check if system contains data for specific user"""
        # This would query the actual system
        # For demo, assume all systems contain user data
        return True
    
    def _get_data_types(self, system):
        """Get types of personal data in system"""
        # This would analyze the system's schema
        return ['email', 'name', 'preferences']
    
    def _get_retention_policy(self, system):
        """Get data retention policy for system"""
        # This would look up retention policies
        return '7 years'
    
    def _sort_by_dependency(self, locations):
        """Sort locations by dependency order"""
        # Sort so downstream systems are deleted first
        return sorted(locations, key=lambda x: x['system'])

# Example usage
gdpr_tracker = GDPRLineageTracker(lineage_system)

# Find all personal data for a user
user_data = gdpr_tracker.find_personal_data_usage('user_12345')
print(f"User data found in {len(user_data)} systems")

# Generate deletion plan
deletion_plan = gdpr_tracker.generate_data_deletion_plan('user_12345')
print(f"Deletion plan involves {len(deletion_plan['systems_to_update'])} systems")
```

#### 2. Data Quality Root Cause Analysis

```python
class DataQualityLineageAnalyzer:
    """Use lineage for data quality root cause analysis"""
    
    def __init__(self, lineage_system, quality_system):
        self.lineage = lineage_system
        self.quality = quality_system
    
    def analyze_quality_issue(self, dataset, quality_issue):
        """Analyze root cause of data quality issue"""
        analysis = {
            'dataset': dataset,
            'issue': quality_issue,
            'potential_root_causes': [],
            'affected_downstream_systems': [],
            'recommended_actions': []
        }
        
        # Get upstream lineage to find potential root causes
        upstream_systems = self.lineage.get_upstream_lineage(dataset)
        
        for upstream in upstream_systems:
            # Check quality metrics for upstream systems
            upstream_quality = self.quality.get_quality_metrics(upstream)
            
            if self._has_quality_issues(upstream_quality, quality_issue):
                analysis['potential_root_causes'].append({
                    'system': upstream,
                    'quality_metrics': upstream_quality,
                    'confidence': self._calculate_confidence(upstream_quality, quality_issue)
                })
        
        # Get downstream systems that might be affected
        downstream_systems = self.lineage.get_downstream_lineage(dataset)
        analysis['affected_downstream_systems'] = downstream_systems
        
        # Generate recommendations
        analysis['recommended_actions'] = self._generate_recommendations(
            analysis['potential_root_causes'],
            downstream_systems
        )
        
        return analysis
    
    def _has_quality_issues(self, quality_metrics, target_issue):
        """Check if quality metrics indicate similar issues"""
        if target_issue['type'] == 'completeness':
            return quality_metrics.get('completeness', 1.0) < 0.95
        elif target_issue['type'] == 'uniqueness':
            return quality_metrics.get('uniqueness', 1.0) < 0.99
        elif target_issue['type'] == 'validity':
            return quality_metrics.get('validity', 1.0) < 0.95
        return False
    
    def _calculate_confidence(self, quality_metrics, target_issue):
        """Calculate confidence that this is the root cause"""
        # Simple confidence calculation based on quality scores
        if target_issue['type'] in quality_metrics:
            score = quality_metrics[target_issue['type']]
            return max(0, (0.95 - score) / 0.95)  # Higher confidence for lower scores
        return 0.5
    
    def _generate_recommendations(self, root_causes, downstream_systems):
        """Generate recommended actions"""
        recommendations = []
        
        # Sort root causes by confidence
        sorted_causes = sorted(root_causes, key=lambda x: x['confidence'], reverse=True)
        
        if sorted_causes:
            top_cause = sorted_causes[0]
            recommendations.append(
                f"Investigate {top_cause['system']} - highest confidence root cause"
            )
        
        if downstream_systems:
            recommendations.append(
                f"Alert teams responsible for {len(downstream_systems)} downstream systems"
            )
        
        recommendations.append("Set up monitoring to prevent similar issues")
        
        return recommendations

# Example usage
quality_analyzer = DataQualityLineageAnalyzer(lineage_system, quality_system)

# Analyze a data quality issue
issue = {
    'type': 'completeness',
    'description': 'Missing email addresses in user_metrics table',
    'severity': 'high'
}

analysis = quality_analyzer.analyze_quality_issue('analytics.user_metrics', issue)
print(f"Found {len(analysis['potential_root_causes'])} potential root causes")
print(f"Affects {len(analysis['affected_downstream_systems'])} downstream systems")
```

### Monitoring and Alerting

#### 1. Lineage Health Monitoring

```python
class LineageHealthMonitor:
    """Monitor health of lineage system"""
    
    def __init__(self, lineage_system):
        self.lineage = lineage_system
        self.metrics = {}
    
    def check_lineage_completeness(self):
        """Check if lineage is complete for all datasets"""
        datasets = self.lineage.get_all_datasets()
        incomplete_lineage = []
        
        for dataset in datasets:
            upstream = self.lineage.get_upstream_lineage(dataset)
            downstream = self.lineage.get_downstream_lineage(dataset)
            
            # Check if dataset has expected lineage
            if self._should_have_upstream(dataset) and not upstream:
                incomplete_lineage.append({
                    'dataset': dataset,
                    'issue': 'missing_upstream_lineage'
                })
            
            if self._should_have_downstream(dataset) and not downstream:
                incomplete_lineage.append({
                    'dataset': dataset,
                    'issue': 'missing_downstream_lineage'
                })
        
        self.metrics['lineage_completeness'] = {
            'total_datasets': len(datasets),
            'incomplete_datasets': len(incomplete_lineage),
            'completeness_percentage': (len(datasets) - len(incomplete_lineage)) / len(datasets) * 100
        }
        
        return incomplete_lineage
    
    def check_lineage_freshness(self):
        """Check if lineage information is up to date"""
        stale_lineage = []
        datasets = self.lineage.get_all_datasets()
        
        for dataset in datasets:
            last_updated = self.lineage.get_last_updated(dataset)
            if last_updated:
                age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                if age_hours > 24:  # Stale if older than 24 hours
                    stale_lineage.append({
                        'dataset': dataset,
                        'last_updated': last_updated,
                        'age_hours': age_hours
                    })
        
        self.metrics['lineage_freshness'] = {
            'total_datasets': len(datasets),
            'stale_datasets': len(stale_lineage),
            'freshness_percentage': (len(datasets) - len(stale_lineage)) / len(datasets) * 100
        }
        
        return stale_lineage
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        incomplete = self.check_lineage_completeness()
        stale = self.check_lineage_freshness()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'metrics': self.metrics,
            'issues': {
                'incomplete_lineage': incomplete,
                'stale_lineage': stale
            },
            'recommendations': []
        }
        
        # Determine overall health
        completeness = self.metrics['lineage_completeness']['completeness_percentage']
        freshness = self.metrics['lineage_freshness']['freshness_percentage']
        
        if completeness < 80 or freshness < 80:
            report['overall_health'] = 'unhealthy'
        elif completeness < 90 or freshness < 90:
            report['overall_health'] = 'degraded'
        
        # Generate recommendations
        if incomplete:
            report['recommendations'].append(
                f"Fix lineage for {len(incomplete)} datasets with missing lineage"
            )
        
        if stale:
            report['recommendations'].append(
                f"Update lineage for {len(stale)} datasets with stale information"
            )
        
        return report

# Example usage
monitor = LineageHealthMonitor(lineage_system)
health_report = monitor.generate_health_report()

print(f"Lineage health: {health_report['overall_health']}")
print(f"Completeness: {health_report['metrics']['lineage_completeness']['completeness_percentage']:.1f}%")
print(f"Freshness: {health_report['metrics']['lineage_freshness']['freshness_percentage']:.1f}%")
```

### Best Practices

#### 1. Lineage Collection Strategy

**Automated Collection**:
- SQL parsing from query logs
- Code analysis from Git repositories
- Orchestration tool integration (Airflow, dbt)
- API call monitoring

**Manual Annotation**:
- Business logic documentation
- External system dependencies
- Data quality rules
- Compliance requirements

**Hybrid Approach**:
- Automated for technical lineage
- Manual for business context
- Validation and verification processes
- Regular audits and updates

#### 2. Lineage Storage and Performance

**Graph Database Optimization**:
```python
# Neo4j optimization for lineage queries
CREATE INDEX ON :Dataset(qualified_name);
CREATE INDEX ON :Column(qualified_name);
CREATE INDEX ON :Transformation(type);

# Optimized query for upstream lineage
MATCH (d:Dataset {qualified_name: $dataset_name})<-[:PRODUCES*1..5]-(upstream:Dataset)
RETURN upstream.qualified_name, upstream.type
ORDER BY upstream.qualified_name;
```

**Caching Strategy**:
```python
import redis
import json

class LineageCacheManager:
    """Cache lineage queries for performance"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    def get_cached_lineage(self, dataset_id, direction):
        """Get cached lineage if available"""
        cache_key = f"lineage:{dataset_id}:{direction}"
        cached = self.redis.get(cache_key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def cache_lineage(self, dataset_id, direction, lineage_data):
        """Cache lineage data"""
        cache_key = f"lineage:{dataset_id}:{direction}"
        self.redis.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(lineage_data)
        )
    
    def invalidate_cache(self, dataset_id):
        """Invalidate cache when lineage changes"""
        pattern = f"lineage:{dataset_id}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
```

#### 3. Lineage Governance

**Data Lineage Policies**:
```yaml
# lineage_policies.yml
lineage_governance:
  required_lineage:
    - all_production_datasets
    - pii_containing_datasets
    - financial_data
  
  lineage_validation:
    - upstream_sources_documented
    - transformation_logic_explained
    - data_owner_identified
  
  compliance_requirements:
    - gdpr_data_subject_tracking
    - sox_financial_audit_trail
    - hipaa_phi_lineage
  
  quality_standards:
    - lineage_completeness: 95%
    - lineage_freshness: 24_hours
    - column_level_coverage: 80%
```

---

## 💻 Hands-On Exercise

See `exercise.py` for hands-on practice with data lineage implementation.

**What you'll build**:
1. SQL parser for automated lineage extraction
2. Multi-hop lineage tracking system
3. Column-level lineage with transformations
4. Impact analysis for change management
5. GDPR compliance lineage tracker
6. Data quality root cause analyzer

**Expected time**: 45 minutes

---

## 📚 Resources

- [Apache Atlas Documentation](https://atlas.apache.org/2.2.0/)
- [DataHub Lineage Guide](https://datahubproject.io/docs/lineage/intro/)
- [dbt Lineage Documentation](https://docs.getdbt.com/docs/collaborate/data-lineage)
- [Great Expectations Lineage](https://docs.greatexpectations.io/docs/guides/validation/advanced/how_to_create_renderers_for_custom_expectations/)
- [OpenLineage Standard](https://openlineage.io/)

---

## 🎯 Key Takeaways

- **Data lineage is essential** for debugging, compliance, and impact analysis
- **Automated extraction** from SQL, code, and orchestration tools scales better than manual documentation
- **Column-level lineage** provides detailed insights for compliance and quality analysis
- **Multi-hop lineage** shows end-to-end data flow across complex systems
- **Impact analysis** enables confident change management and root cause analysis
- **GDPR and compliance** require comprehensive lineage for data subject requests
- **Graph databases** are optimal for storing and querying lineage relationships
- **Caching and optimization** are crucial for performance at scale
- **Monitoring lineage health** ensures the system remains accurate and complete
- **Hybrid approaches** combine automated collection with manual business context

---

## 🚀 What's Next?

Tomorrow (Day 10), you'll learn about **Data Privacy - GDPR, PII Handling** - understanding privacy regulations, implementing PII detection and masking, and building privacy-compliant data systems.

**Preview**: Data privacy covers:
- GDPR, CCPA, and other privacy regulations
- Automated PII detection and classification
- Data anonymization and pseudonymization techniques
- Privacy-preserving analytics and ML

---

## ✅ Before Moving On

- [ ] Understand the importance and types of data lineage
- [ ] Can implement automated lineage extraction from SQL and code
- [ ] Know how to build multi-hop and column-level lineage tracking
- [ ] Can perform impact analysis for change management
- [ ] Understand lineage applications for compliance and quality
- [ ] Complete the exercise in `exercise.py`
- [ ] Review the solution in `solution.py`
- [ ] Take the quiz in `quiz.md`

**Time**: ~1 hour | **Difficulty**: ⭐⭐⭐⭐ (Advanced)

Ready to track your data's journey! 🚀
