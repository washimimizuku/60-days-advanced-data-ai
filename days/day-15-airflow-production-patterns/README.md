# Day 15: Airflow Production Patterns - Dynamic DAGs, Task Groups, Advanced Orchestration

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** dynamic DAG generation patterns for scalable workflow management
- **Implement** task groups for better organization and reusability
- **Apply** advanced branching and conditional execution patterns
- **Configure** production-ready resource management with pools and priorities
- **Build** enterprise-grade Airflow workflows with proper error handling and monitoring

---

## Theory

### Moving Beyond Basic Airflow

While Day 12 covered Airflow fundamentals, production environments require sophisticated patterns to manage complexity, ensure reliability, and maintain scalability. Enterprise Airflow deployments often manage hundreds of DAGs processing terabytes of data daily.

**Production Challenges**:
- **Scale**: Managing 100+ similar DAGs without code duplication
- **Complexity**: Organizing workflows with 50+ tasks
- **Reliability**: Handling failures gracefully in distributed systems
- **Resource Management**: Preventing resource contention
- **Maintainability**: Keeping workflows understandable and debuggable

### Dynamic DAG Generation

Dynamic DAG generation is the practice of creating DAGs programmatically from configurations, eliminating code duplication and enabling data-driven workflow creation.

#### Why Dynamic DAGs?

**Before Dynamic DAGs** (Anti-pattern):
```python
# pipeline_customers.py - 200 lines
dag_customers = DAG('process_customers', ...)
extract_customers = PythonOperator(...)
transform_customers = PythonOperator(...)
# ... 50 more lines

# pipeline_orders.py - 200 lines (95% identical)
dag_orders = DAG('process_orders', ...)
extract_orders = PythonOperator(...)
transform_orders = PythonOperator(...)
# ... 50 more lines

# pipeline_products.py - 200 lines (95% identical)
# ... and so on for 20 more entities
```

**With Dynamic DAGs** (Best Practice):
```python
# dynamic_etl_dags.py - 50 lines total
ENTITIES = [
    {'name': 'customers', 'source': 'postgres', 'schedule': '@daily'},
    {'name': 'orders', 'source': 'api', 'schedule': '@hourly'},
    {'name': 'products', 'source': 'sftp', 'schedule': '@weekly'},
    # ... 20 entities defined in 20 lines
]

def create_etl_dag(entity_config):
    """Generate ETL DAG from configuration"""
    # Single implementation handles all entities
    # 30 lines of reusable logic

# Generate all DAGs
for config in ENTITIES:
    dag_id = f"etl_{config['name']}"
    globals()[dag_id] = create_etl_dag(config)
```

#### Advanced Dynamic DAG Patterns

**1. Database-Driven DAG Generation**:
```python
import psycopg2
from airflow.models import Variable

def get_pipeline_configs():
    """Load pipeline configurations from database"""
    conn = psycopg2.connect(Variable.get("DB_CONN"))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT pipeline_name, source_system, target_table, 
               schedule_interval, is_active, config_json
        FROM pipeline_configurations 
        WHERE is_active = true
    """)
    
    configs = []
    for row in cursor.fetchall():
        config = {
            'name': row[0],
            'source': row[1],
            'target': row[2],
            'schedule': row[3],
            'config': json.loads(row[5])
        }
        configs.append(config)
    
    return configs

# Generate DAGs from database
for config in get_pipeline_configs():
    dag_id = f"db_driven_{config['name']}"
    globals()[dag_id] = create_pipeline_dag(config)
```

**2. API-Driven DAG Generation**:
```python
import requests
from airflow.models import Variable

def get_api_endpoints():
    """Load API endpoints to create monitoring DAGs"""
    api_config_url = Variable.get("API_CONFIG_URL")
    response = requests.get(api_config_url)
    
    endpoints = response.json()['endpoints']
    
    monitoring_configs = []
    for endpoint in endpoints:
        config = {
            'name': endpoint['name'],
            'url': endpoint['url'],
            'method': endpoint.get('method', 'GET'),
            'expected_status': endpoint.get('expected_status', 200),
            'timeout': endpoint.get('timeout', 30),
            'schedule': endpoint.get('schedule', '*/5 * * * *')  # Every 5 minutes
        }
        monitoring_configs.append(config)
    
    return monitoring_configs

# Generate API monitoring DAGs
for config in get_api_endpoints():
    dag_id = f"api_monitor_{config['name']}"
    globals()[dag_id] = create_api_monitoring_dag(config)
```

**3. Template-Based DAG Generation**:
```python
from jinja2 import Template
import yaml

def create_dag_from_template(template_path, config_path):
    """Generate DAG from Jinja2 template and YAML config"""
    
    # Load template
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Render template
    template = Template(template_content)
    dag_code = template.render(**config)
    
    # Execute generated code
    exec(dag_code, globals())

# Generate DAGs from templates
template_configs = [
    ('templates/etl_template.py', 'configs/customers_etl.yml'),
    ('templates/etl_template.py', 'configs/orders_etl.yml'),
    ('templates/ml_template.py', 'configs/recommendation_ml.yml'),
]

for template_path, config_path in template_configs:
    create_dag_from_template(template_path, config_path)
```

### Task Groups: Organizing Complex Workflows

Task Groups replace the deprecated SubDAGs, providing visual organization and logical grouping of related tasks without the complexity and performance issues of SubDAGs.

#### Task Group Benefits

**Visual Organization**:
```python
# Without Task Groups - Flat structure
extract_customers >> transform_customers >> load_customers >> validate_customers
extract_orders >> transform_orders >> load_orders >> validate_orders
extract_products >> transform_products >> load_products >> validate_products
# UI shows 12 separate tasks - hard to understand

# With Task Groups - Hierarchical structure
with TaskGroup('customers_etl') as customers_group:
    extract >> transform >> load >> validate

with TaskGroup('orders_etl') as orders_group:
    extract >> transform >> load >> validate

with TaskGroup('products_etl') as products_group:
    extract >> transform >> load >> validate

# UI shows 3 collapsible groups - much cleaner
```

#### Advanced Task Group Patterns

**1. Reusable Task Group Factory**:
```python
def create_etl_task_group(group_id, entity_config):
    """Factory function to create standardized ETL task groups"""
    
    with TaskGroup(group_id=group_id) as group:
        
        # Extract task
        extract = PythonOperator(
            task_id='extract',
            python_callable=extract_data,
            op_kwargs={
                'source_type': entity_config['source_type'],
                'source_config': entity_config['source_config'],
                'entity': entity_config['entity']
            }
        )
        
        # Data quality check
        quality_check = PythonOperator(
            task_id='quality_check',
            python_callable=validate_data_quality,
            op_kwargs={
                'entity': entity_config['entity'],
                'quality_rules': entity_config['quality_rules']
            }
        )
        
        # Transform task
        transform = PythonOperator(
            task_id='transform',
            python_callable=transform_data,
            op_kwargs={
                'entity': entity_config['entity'],
                'transformations': entity_config['transformations']
            }
        )
        
        # Load task
        load = PythonOperator(
            task_id='load',
            python_callable=load_data,
            op_kwargs={
                'target_type': entity_config['target_type'],
                'target_config': entity_config['target_config'],
                'entity': entity_config['entity']
            }
        )
        
        # Validation task
        validate = PythonOperator(
            task_id='validate',
            python_callable=validate_load,
            op_kwargs={
                'entity': entity_config['entity'],
                'validation_queries': entity_config['validation_queries']
            }
        )
        
        # Define dependencies within group
        extract >> quality_check >> transform >> load >> validate
    
    return group

# Usage in DAG
with DAG('multi_entity_etl', ...) as dag:
    
    # Create task groups for different entities
    customers_etl = create_etl_task_group('customers_etl', CUSTOMERS_CONFIG)
    orders_etl = create_etl_task_group('orders_etl', ORDERS_CONFIG)
    products_etl = create_etl_task_group('products_etl', PRODUCTS_CONFIG)
    
    # Cross-group dependencies
    start = DummyOperator(task_id='start')
    end = DummyOperator(task_id='end')
    
    start >> [customers_etl, products_etl] >> orders_etl >> end
```

**2. Nested Task Groups**:
```python
def create_ml_pipeline_task_group(group_id, model_config):
    """Create ML pipeline with nested task groups"""
    
    with TaskGroup(group_id=group_id) as ml_group:
        
        # Data preparation task group
        with TaskGroup('data_preparation') as data_prep:
            extract_features = PythonOperator(
                task_id='extract_features',
                python_callable=extract_features_func
            )
            
            clean_data = PythonOperator(
                task_id='clean_data',
                python_callable=clean_data_func
            )
            
            split_data = PythonOperator(
                task_id='split_data',
                python_callable=split_data_func
            )
            
            extract_features >> clean_data >> split_data
        
        # Model training task group
        with TaskGroup('model_training') as training:
            train_model = PythonOperator(
                task_id='train_model',
                python_callable=train_model_func
            )
            
            validate_model = PythonOperator(
                task_id='validate_model',
                python_callable=validate_model_func
            )
            
            hyperparameter_tuning = PythonOperator(
                task_id='hyperparameter_tuning',
                python_callable=tune_hyperparameters_func
            )
            
            train_model >> validate_model >> hyperparameter_tuning
        
        # Model deployment task group
        with TaskGroup('model_deployment') as deployment:
            register_model = PythonOperator(
                task_id='register_model',
                python_callable=register_model_func
            )
            
            deploy_model = PythonOperator(
                task_id='deploy_model',
                python_callable=deploy_model_func
            )
            
            test_endpoint = PythonOperator(
                task_id='test_endpoint',
                python_callable=test_endpoint_func
            )
            
            register_model >> deploy_model >> test_endpoint
        
        # Define cross-group dependencies
        data_prep >> training >> deployment
    
    return ml_group
```

### Advanced Branching and Conditional Logic

Production workflows often require complex decision-making based on data conditions, external factors, or business rules.

#### Branching Patterns

**1. Data-Driven Branching**:
```python
def choose_processing_branch(**context):
    """Choose processing path based on data volume"""
    ti = context['task_instance']
    
    # Get data volume from previous task
    data_info = ti.xcom_pull(task_ids='check_data_volume')
    record_count = data_info['record_count']
    data_size_mb = data_info['size_mb']
    
    # Decision logic based on data characteristics
    if record_count > 10_000_000 or data_size_mb > 1000:
        return 'large_data_processing'
    elif record_count > 1_000_000 or data_size_mb > 100:
        return 'medium_data_processing'
    else:
        return 'small_data_processing'

branch_task = BranchPythonOperator(
    task_id='choose_processing_path',
    python_callable=choose_processing_branch
)

# Different processing paths
large_processing = PythonOperator(
    task_id='large_data_processing',
    python_callable=process_large_dataset,
    pool='large_processing_pool',  # Use dedicated resources
    priority_weight=10
)

medium_processing = PythonOperator(
    task_id='medium_data_processing',
    python_callable=process_medium_dataset,
    pool='standard_pool',
    priority_weight=5
)

small_processing = PythonOperator(
    task_id='small_data_processing',
    python_callable=process_small_dataset,
    pool='standard_pool',
    priority_weight=1
)
```

**2. Time-Based Branching**:
```python
def choose_schedule_branch(**context):
    """Choose different processing based on time"""
    execution_date = context['execution_date']
    
    # Different processing on weekends
    if execution_date.weekday() >= 5:  # Saturday or Sunday
        return 'weekend_processing'
    
    # Different processing during business hours
    if 9 <= execution_date.hour <= 17:
        return 'business_hours_processing'
    else:
        return 'off_hours_processing'

time_branch = BranchPythonOperator(
    task_id='choose_schedule_branch',
    python_callable=choose_schedule_branch
)
```

**3. External System Branching**:
```python
def check_external_system_branch(**context):
    """Branch based on external system availability"""
    import requests
    
    try:
        # Check if external API is available
        response = requests.get('https://api.external-system.com/health', timeout=10)
        
        if response.status_code == 200:
            system_status = response.json()['status']
            
            if system_status == 'healthy':
                return 'use_external_api'
            elif system_status == 'degraded':
                return 'use_cached_data'
            else:
                return 'skip_processing'
        else:
            return 'use_cached_data'
            
    except requests.RequestException:
        return 'use_cached_data'

external_check = BranchPythonOperator(
    task_id='check_external_system',
    python_callable=check_external_system_branch
)
```

### Resource Management with Pools and Priorities

Production Airflow deployments must carefully manage resources to prevent system overload and ensure critical tasks get priority.

#### Pool Configuration

**1. Creating Pools**:
```python
# In Airflow UI or via CLI
# airflow pools set large_processing_pool 2 "Pool for large data processing"
# airflow pools set standard_pool 10 "Standard processing pool"
# airflow pools set critical_pool 5 "Critical business processes"

# Using pools in tasks
heavy_task = PythonOperator(
    task_id='heavy_processing',
    python_callable=process_large_dataset,
    pool='large_processing_pool',  # Limit to 2 concurrent
    priority_weight=10,            # High priority
    execution_timeout=timedelta(hours=2)
)

standard_task = PythonOperator(
    task_id='standard_processing',
    python_callable=process_standard_dataset,
    pool='standard_pool',          # Limit to 10 concurrent
    priority_weight=5,             # Medium priority
    execution_timeout=timedelta(minutes=30)
)
```

**2. Dynamic Pool Assignment**:
```python
def get_pool_for_task(data_size_mb):
    """Dynamically assign pool based on data size"""
    if data_size_mb > 1000:
        return 'large_processing_pool'
    elif data_size_mb > 100:
        return 'medium_processing_pool'
    else:
        return 'standard_pool'

def create_processing_task(task_id, data_config):
    """Create task with appropriate pool"""
    return PythonOperator(
        task_id=task_id,
        python_callable=process_data,
        pool=get_pool_for_task(data_config['size_mb']),
        priority_weight=data_config.get('priority', 5),
        op_kwargs={'config': data_config}
    )
```

### Advanced Sensor Patterns

Sensors are crucial for coordinating with external systems and handling dependencies in production environments.

#### Production Sensor Patterns

**1. Smart File Sensor with Validation**:
```python
class ValidatedFileSensor(FileSensor):
    """File sensor that validates file content"""
    
    def __init__(self, min_file_size=0, max_age_hours=24, **kwargs):
        super().__init__(**kwargs)
        self.min_file_size = min_file_size
        self.max_age_hours = max_age_hours
    
    def poke(self, context):
        """Check if file exists and meets criteria"""
        if not super().poke(context):
            return False
        
        import os
        from datetime import datetime, timedelta
        
        # Check file size
        file_size = os.path.getsize(self.filepath)
        if file_size < self.min_file_size:
            self.log.info(f"File too small: {file_size} < {self.min_file_size}")
            return False
        
        # Check file age
        file_mtime = datetime.fromtimestamp(os.path.getmtime(self.filepath))
        max_age = datetime.now() - timedelta(hours=self.max_age_hours)
        
        if file_mtime < max_age:
            self.log.warning(f"File too old: {file_mtime} < {max_age}")
            return False
        
        return True

# Usage
wait_for_valid_file = ValidatedFileSensor(
    task_id='wait_for_data_file',
    filepath='/data/input/{{ ds }}/data.csv',
    min_file_size=1024,  # At least 1KB
    max_age_hours=2,     # No older than 2 hours
    poke_interval=60,    # Check every minute
    timeout=3600,        # Timeout after 1 hour
    mode='reschedule'    # Don't block worker
)
```

**2. Database Record Sensor**:
```python
from airflow.sensors.base import BaseSensorOperator
from airflow.hooks.postgres_hook import PostgresHook

class DatabaseRecordSensor(BaseSensorOperator):
    """Wait for specific records in database"""
    
    def __init__(self, sql, postgres_conn_id='postgres_default', **kwargs):
        super().__init__(**kwargs)
        self.sql = sql
        self.postgres_conn_id = postgres_conn_id
    
    def poke(self, context):
        """Check if records exist"""
        hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        
        # Execute query and check result
        result = hook.get_first(self.sql)
        
        if result and result[0] > 0:
            self.log.info(f"Found {result[0]} records")
            return True
        else:
            self.log.info("No records found yet")
            return False

# Usage
wait_for_new_orders = DatabaseRecordSensor(
    task_id='wait_for_new_orders',
    sql="""
        SELECT COUNT(*) 
        FROM orders 
        WHERE created_date = '{{ ds }}' 
        AND status = 'new'
    """,
    poke_interval=300,  # Check every 5 minutes
    timeout=7200,       # Timeout after 2 hours
    mode='reschedule'
)
```

### Production Error Handling and Monitoring

**1. Custom Failure Callbacks**:
```python
def task_failure_alert(context):
    """Send alert when task fails"""
    import requests
    
    task_instance = context['task_instance']
    dag_id = context['dag'].dag_id
    task_id = task_instance.task_id
    execution_date = context['execution_date']
    
    # Send Slack notification
    slack_webhook = Variable.get("SLACK_WEBHOOK_URL")
    
    message = {
        "text": f"🚨 Airflow Task Failed",
        "attachments": [
            {
                "color": "danger",
                "fields": [
                    {"title": "DAG", "value": dag_id, "short": True},
                    {"title": "Task", "value": task_id, "short": True},
                    {"title": "Execution Date", "value": str(execution_date), "short": True},
                    {"title": "Log URL", "value": task_instance.log_url, "short": False}
                ]
            }
        ]
    }
    
    requests.post(slack_webhook, json=message)

def dag_success_callback(context):
    """Send notification when DAG succeeds"""
    # Implementation for success notifications
    pass

# Apply to DAG
dag = DAG(
    'production_pipeline',
    default_args={
        'on_failure_callback': task_failure_alert,
        'on_success_callback': dag_success_callback,
    },
    # ... other DAG parameters
)
```

**2. Circuit Breaker Pattern**:
```python
class CircuitBreakerOperator(PythonOperator):
    """Operator with circuit breaker pattern"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=300, **kwargs):
        super().__init__(**kwargs)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
    
    def execute(self, context):
        """Execute with circuit breaker logic"""
        from airflow.models import Variable
        import json
        from datetime import datetime, timedelta
        
        # Get circuit breaker state
        cb_key = f"circuit_breaker_{self.dag_id}_{self.task_id}"
        cb_state = Variable.get(cb_key, default_var='{"failures": 0, "last_failure": null, "state": "closed"}')
        cb_data = json.loads(cb_state)
        
        # Check if circuit is open
        if cb_data['state'] == 'open':
            last_failure = datetime.fromisoformat(cb_data['last_failure'])
            if datetime.now() - last_failure < timedelta(seconds=self.recovery_timeout):
                raise Exception("Circuit breaker is OPEN - skipping execution")
            else:
                cb_data['state'] = 'half-open'
        
        try:
            # Execute the actual task
            result = super().execute(context)
            
            # Reset circuit breaker on success
            cb_data = {"failures": 0, "last_failure": None, "state": "closed"}
            Variable.set(cb_key, json.dumps(cb_data))
            
            return result
            
        except Exception as e:
            # Increment failure count
            cb_data['failures'] += 1
            cb_data['last_failure'] = datetime.now().isoformat()
            
            # Open circuit if threshold reached
            if cb_data['failures'] >= self.failure_threshold:
                cb_data['state'] = 'open'
            
            Variable.set(cb_key, json.dumps(cb_data))
            raise e
```

---

## 💻 Hands-On Exercise (40 minutes)

Build a production-grade data processing platform with advanced Airflow patterns.

**Scenario**: You're the Lead Data Engineer at "TechCorp", a SaaS company processing customer data from multiple sources. You need to build a scalable, maintainable pipeline system that can handle varying data volumes and adapt to changing business requirements.

**Requirements**:
1. **Dynamic DAG Generation**: Create pipelines for multiple data sources from configuration
2. **Task Groups**: Organize ETL processes with reusable task groups
3. **Intelligent Branching**: Route data based on volume and business rules
4. **Resource Management**: Use pools to manage computational resources
5. **Advanced Sensors**: Wait for external dependencies with validation
6. **Error Handling**: Implement comprehensive error handling and alerting

See `exercise.py` for starter code and detailed requirements.

---

## 📚 Resources

- **Airflow Best Practices**: [airflow.apache.org/docs/apache-airflow/stable/best-practices.html](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- **Dynamic DAG Generation**: [airflow.apache.org/docs/apache-airflow/stable/howto/dynamic-dag-generation.html](https://airflow.apache.org/docs/apache-airflow/stable/howto/dynamic-dag-generation.html)
- **Task Groups**: [airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#taskgroups](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#taskgroups)
- **Pools**: [airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/pools.html](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/pools.html)
- **Sensors**: [airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/sensors.html)

---

## 🎯 Key Takeaways

- **Dynamic DAG generation** eliminates code duplication and enables data-driven workflow creation
- **Task groups** provide visual organization and reusable patterns without SubDAG complexity
- **Advanced branching** enables intelligent routing based on data, time, and external conditions
- **Resource management** with pools and priorities prevents system overload
- **Production sensors** handle external dependencies with proper validation and error handling
- **Error handling patterns** ensure reliable operation and proper alerting
- **Circuit breaker patterns** prevent cascading failures in distributed systems
- **Monitoring and observability** are essential for production Airflow deployments

---

## 🚀 What's Next?

Tomorrow (Day 16), you'll learn **Airflow at Scale** - advanced deployment patterns, executors, monitoring, and enterprise-grade operational practices.

**Preview**: You'll explore distributed executors (Celery, Kubernetes), advanced monitoring with Prometheus and Grafana, auto-scaling strategies, and enterprise deployment patterns. This builds on today's production patterns to handle truly massive scale!

---

## ✅ Before Moving On

- [ ] Understand dynamic DAG generation patterns and use cases
- [ ] Can create and organize workflows with task groups
- [ ] Know how to implement intelligent branching logic
- [ ] Can configure pools and priorities for resource management
- [ ] Understand advanced sensor patterns for external dependencies
- [ ] Can implement production error handling and monitoring
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced Production Patterns)

Ready to build enterprise-grade Airflow workflows! 🚀