# Day 19: Data Quality in Production - Frameworks, Monitoring, Automation

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** comprehensive data quality frameworks for production environments
- **Implement** automated validation and monitoring systems at enterprise scale
- **Build** data quality metrics, alerting, and incident response workflows
- **Integrate** quality checks with orchestration tools (Airflow, dbt) and CI/CD pipelines
- **Design** data contracts and SLAs for reliable data products

---

## Theory

### Production Data Quality Framework

Building on the data governance concepts from Days 8-11, production data quality requires systematic approaches to ensure data reliability, accuracy, and trustworthiness at scale.

#### 1. Data Quality Dimensions

**Completeness**:
```python
# Example: Completeness validation
def validate_completeness(df, required_columns, threshold=0.95):
    """Validate that required columns meet completeness threshold"""
    completeness_results = {}
    
    for column in required_columns:
        non_null_ratio = df[column].count() / len(df)
        completeness_results[column] = {
            'completeness_ratio': non_null_ratio,
            'passes_threshold': non_null_ratio >= threshold,
            'missing_count': df[column].isnull().sum()
        }
    
    return completeness_results
```

**Accuracy**:
```python
# Example: Accuracy validation with business rules
def validate_accuracy(df, validation_rules):
    """Validate data accuracy against business rules"""
    accuracy_results = {}
    
    for rule_name, rule_func in validation_rules.items():
        try:
            valid_records = df.apply(rule_func, axis=1)
            accuracy_ratio = valid_records.sum() / len(df)
            
            accuracy_results[rule_name] = {
                'accuracy_ratio': accuracy_ratio,
                'invalid_count': (~valid_records).sum(),
                'sample_invalid_records': df[~valid_records].head(5).to_dict('records')
            }
        except Exception as e:
            accuracy_results[rule_name] = {'error': str(e)}
    
    return accuracy_results
```

**Consistency**:
```python
# Example: Cross-system consistency validation
def validate_consistency(source_df, target_df, key_columns, value_columns):
    """Validate consistency between source and target systems"""
    
    # Join on key columns
    merged = source_df.merge(
        target_df, 
        on=key_columns, 
        how='inner', 
        suffixes=('_source', '_target')
    )
    
    consistency_results = {}
    
    for column in value_columns:
        source_col = f"{column}_source"
        target_col = f"{column}_target"
        
        if source_col in merged.columns and target_col in merged.columns:
            matches = merged[source_col] == merged[target_col]
            consistency_ratio = matches.sum() / len(merged)
            
            consistency_results[column] = {
                'consistency_ratio': consistency_ratio,
                'mismatch_count': (~matches).sum(),
                'total_compared': len(merged)
            }
    
    return consistency_results
```

**Timeliness**:
```python
# Example: Data freshness validation
def validate_timeliness(df, timestamp_column, max_age_hours=24):
    """Validate data timeliness/freshness"""
    
    current_time = datetime.now()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Calculate age of each record
    df['age_hours'] = (current_time - df[timestamp_column]).dt.total_seconds() / 3600
    
    fresh_records = df['age_hours'] <= max_age_hours
    freshness_ratio = fresh_records.sum() / len(df)
    
    return {
        'freshness_ratio': freshness_ratio,
        'stale_count': (~fresh_records).sum(),
        'avg_age_hours': df['age_hours'].mean(),
        'max_age_hours': df['age_hours'].max(),
        'oldest_records': df.nlargest(5, 'age_hours')[['age_hours', timestamp_column]].to_dict('records')
    }
```

#### 2. Great Expectations Framework

**Expectation Suites**:
```python
# Example: Comprehensive expectation suite
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration

def create_customer_data_suite():
    """Create comprehensive expectation suite for customer data"""
    
    suite = gx.ExpectationSuite(expectation_suite_name="customer_data_quality")
    
    # Completeness expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": "customer_id"}
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "customer_id",
                "mostly": 1.0  # 100% completeness required
            }
        )
    )
    
    # Uniqueness expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "customer_id"}
        )
    )
    
    # Format expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "email",
                "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "mostly": 0.95  # Allow 5% invalid emails
            }
        )
    )
    
    # Range expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "age",
                "min_value": 0,
                "max_value": 120,
                "mostly": 0.99
            }
        )
    )
    
    # Business rule expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "subscription_status",
                "value_set": ["active", "inactive", "suspended", "cancelled"]
            }
        )
    )
    
    return suite
```

**Checkpoints and Automation**:
```python
# Example: Automated checkpoint configuration
def create_automated_checkpoint():
    """Create automated checkpoint for production validation"""
    
    checkpoint_config = {
        "name": "customer_data_checkpoint",
        "config_version": 1.0,
        "template_name": None,
        "module_name": "great_expectations.checkpoint",
        "class_name": "Checkpoint",
        "run_name_template": "%Y%m%d-%H%M%S-customer-data-validation",
        "expectation_suite_name": "customer_data_quality",
        "batch_request": {
            "datasource_name": "production_datasource",
            "data_connector_name": "default_inferred_data_connector",
            "data_asset_name": "customer_data",
            "batch_spec_passthrough": {
                "reader_method": "read_csv",
                "reader_options": {"header": 0}
            }
        },
        "action_list": [
            {
                "name": "store_validation_result",
                "action": {
                    "class_name": "StoreValidationResultAction"
                }
            },
            {
                "name": "store_evaluation_params",
                "action": {
                    "class_name": "StoreEvaluationParametersAction"
                }
            },
            {
                "name": "update_data_docs",
                "action": {
                    "class_name": "UpdateDataDocsAction",
                    "site_names": ["local_site"]
                }
            },
            {
                "name": "slack_notification",
                "action": {
                    "class_name": "SlackNotificationAction",
                    "slack_webhook": "${SLACK_WEBHOOK_URL}",
                    "notify_on": "failure",
                    "renderer": {
                        "module_name": "great_expectations.render.renderer.slack_renderer",
                        "class_name": "SlackRenderer"
                    }
                }
            }
        ],
        "evaluation_parameters": {},
        "runtime_configuration": {},
        "validations": []
    }
    
    return checkpoint_config
```

#### 3. Data Contracts and SLAs

**Data Contract Definition**:
```yaml
# Example: Data contract specification
data_contract:
  name: "customer_data_v1"
  version: "1.2.0"
  description: "Customer data contract for analytics and ML"
  
  owner:
    team: "data-platform"
    contact: "data-platform@company.com"
  
  consumers:
    - team: "analytics"
      use_case: "customer_segmentation"
    - team: "ml-platform"
      use_case: "churn_prediction"
  
  schema:
    columns:
      - name: "customer_id"
        type: "string"
        nullable: false
        unique: true
        description: "Unique customer identifier"
        
      - name: "email"
        type: "string"
        nullable: false
        unique: true
        format: "email"
        description: "Customer email address"
        
      - name: "created_at"
        type: "timestamp"
        nullable: false
        description: "Account creation timestamp"
        
      - name: "subscription_tier"
        type: "string"
        nullable: false
        enum: ["free", "basic", "premium", "enterprise"]
        description: "Current subscription level"
  
  quality_requirements:
    completeness:
      customer_id: 1.0
      email: 1.0
      created_at: 1.0
      subscription_tier: 0.95
    
    uniqueness:
      customer_id: 1.0
      email: 0.99
    
    freshness:
      max_age_hours: 4
      update_frequency: "hourly"
    
    accuracy:
      email_format_valid: 0.95
      subscription_tier_valid: 1.0
  
  sla:
    availability: 99.9
    latency_p95_ms: 500
    error_rate_max: 0.1
    
  monitoring:
    alerts:
      - condition: "completeness < 0.95"
        severity: "high"
        notification: ["slack", "email"]
      
      - condition: "freshness > 6 hours"
        severity: "critical"
        notification: ["pagerduty", "slack", "email"]
```

#### 4. Integration with Orchestration Tools

**Airflow Integration**:
```python
# Example: Great Expectations in Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow_providers_great_expectations.operators.great_expectations import GreatExpectationsOperator

def create_data_quality_dag():
    """Create Airflow DAG with integrated data quality checks"""
    
    dag = DAG(
        'data_quality_pipeline',
        default_args={
            'owner': 'data-platform',
            'start_date': datetime(2024, 1, 1),
            'retries': 2,
            'retry_delay': timedelta(minutes=5)
        },
        schedule_interval='@hourly',
        catchup=False,
        tags=['data_quality', 'production']
    )
    
    # Data extraction task
    extract_data = PythonOperator(
        task_id='extract_customer_data',
        python_callable=extract_customer_data,
        dag=dag
    )
    
    # Data quality validation
    validate_data_quality = GreatExpectationsOperator(
        task_id='validate_customer_data_quality',
        checkpoint_name='customer_data_checkpoint',
        data_context_root_dir='/opt/airflow/great_expectations',
        fail_task_on_validation_failure=True,
        dag=dag
    )
    
    # Data processing (only if quality checks pass)
    process_data = PythonOperator(
        task_id='process_customer_data',
        python_callable=process_customer_data,
        dag=dag
    )
    
    # Quality metrics collection
    collect_quality_metrics = PythonOperator(
        task_id='collect_quality_metrics',
        python_callable=collect_and_store_quality_metrics,
        dag=dag
    )
    
    # Define dependencies
    extract_data >> validate_data_quality >> process_data >> collect_quality_metrics
    
    return dag
```

**dbt Integration**:
```sql
-- Example: dbt test for data quality
-- tests/generic/test_data_freshness.sql
{% test data_freshness(model, column_name, max_age_hours=24) %}

  with freshness_check as (
    select
      {{ column_name }},
      extract(epoch from (current_timestamp - {{ column_name }})) / 3600 as age_hours
    from {{ model }}
  ),
  
  stale_records as (
    select *
    from freshness_check
    where age_hours > {{ max_age_hours }}
  )
  
  select * from stale_records

{% endtest %}

-- models/schema.yml
version: 2

models:
  - name: dim_customers
    tests:
      - data_freshness:
          column_name: updated_at
          max_age_hours: 6
    columns:
      - name: customer_id
        tests:
          - unique
          - not_null
      - name: email
        tests:
          - unique
          - not_null
          - dbt_expectations.expect_column_values_to_match_regex:
              regex: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
```

#### 5. Advanced Quality Monitoring

**Real-time Quality Metrics**:
```python
# Example: Real-time quality monitoring system
class DataQualityMonitor:
    """Real-time data quality monitoring system"""
    
    def __init__(self, metrics_store, alert_manager):
        self.metrics_store = metrics_store
        self.alert_manager = alert_manager
        self.quality_thresholds = self._load_quality_thresholds()
    
    def monitor_data_quality(self, dataset_name, validation_results):
        """Monitor data quality and trigger alerts if needed"""
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(validation_results)
        
        # Store metrics for trending
        self._store_quality_metrics(dataset_name, quality_metrics)
        
        # Check thresholds and trigger alerts
        self._check_quality_thresholds(dataset_name, quality_metrics)
        
        return quality_metrics
    
    def _calculate_quality_metrics(self, validation_results):
        """Calculate comprehensive quality metrics"""
        
        total_expectations = len(validation_results.results)
        successful_expectations = sum(1 for result in validation_results.results if result.success)
        
        metrics = {
            'overall_success_rate': successful_expectations / total_expectations if total_expectations > 0 else 0,
            'total_expectations': total_expectations,
            'successful_expectations': successful_expectations,
            'failed_expectations': total_expectations - successful_expectations,
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Calculate dimension-specific metrics
        dimension_metrics = {}
        for result in validation_results.results:
            expectation_type = result.expectation_config.expectation_type
            dimension = self._map_expectation_to_dimension(expectation_type)
            
            if dimension not in dimension_metrics:
                dimension_metrics[dimension] = {'total': 0, 'successful': 0}
            
            dimension_metrics[dimension]['total'] += 1
            if result.success:
                dimension_metrics[dimension]['successful'] += 1
        
        # Calculate success rates by dimension
        for dimension, counts in dimension_metrics.items():
            success_rate = counts['successful'] / counts['total'] if counts['total'] > 0 else 0
            metrics[f'{dimension}_success_rate'] = success_rate
        
        return metrics
    
    def _check_quality_thresholds(self, dataset_name, quality_metrics):
        """Check quality metrics against thresholds and trigger alerts"""
        
        thresholds = self.quality_thresholds.get(dataset_name, {})
        
        for metric_name, metric_value in quality_metrics.items():
            if metric_name.endswith('_success_rate'):
                threshold = thresholds.get(metric_name, 0.95)  # Default 95% threshold
                
                if metric_value < threshold:
                    alert = {
                        'dataset': dataset_name,
                        'metric': metric_name,
                        'value': metric_value,
                        'threshold': threshold,
                        'severity': self._determine_alert_severity(metric_value, threshold),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.alert_manager.send_alert(alert)
    
    def _determine_alert_severity(self, value, threshold):
        """Determine alert severity based on how far below threshold"""
        
        deviation = (threshold - value) / threshold
        
        if deviation >= 0.2:  # 20% or more below threshold
            return 'critical'
        elif deviation >= 0.1:  # 10-20% below threshold
            return 'high'
        elif deviation >= 0.05:  # 5-10% below threshold
            return 'medium'
        else:
            return 'low'
```

#### 6. Data Quality Dashboards and Reporting

**Quality Dashboard Configuration**:
```python
# Example: Data quality dashboard configuration
def create_quality_dashboard():
    """Create comprehensive data quality dashboard"""
    
    dashboard_config = {
        "dashboard": {
            "title": "Production Data Quality Dashboard",
            "tags": ["data_quality", "production"],
            "time": {"from": "now-24h", "to": "now"},
            "refresh": "1m",
            
            "panels": [
                {
                    "title": "Overall Data Quality Score",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "avg(data_quality_overall_success_rate)",
                            "legendFormat": "Quality Score"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 80},
                                    {"color": "green", "value": 95}
                                ]
                            }
                        }
                    }
                },
                
                {
                    "title": "Quality Trends by Dataset",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "data_quality_success_rate by (dataset)",
                            "legendFormat": "{{ dataset }}"
                        }
                    ]
                },
                
                {
                    "title": "Failed Expectations by Type",
                    "type": "piechart",
                    "targets": [
                        {
                            "expr": "sum by (expectation_type) (data_quality_failed_expectations)",
                            "legendFormat": "{{ expectation_type }}"
                        }
                    ]
                },
                
                {
                    "title": "Data Freshness Status",
                    "type": "table",
                    "targets": [
                        {
                            "expr": "data_freshness_hours by (dataset)",
                            "format": "table"
                        }
                    ]
                }
            ]
        }
    }
    
    return dashboard_config
```

---

## 💻 Hands-On Exercise (40 minutes)

Build a comprehensive production data quality system with automated validation, monitoring, and alerting.

**Scenario**: You're the Data Quality Engineer at "QualityFirst Corp", a data-driven company processing millions of customer records daily. You need to implement a robust data quality framework that ensures data reliability across all systems.

**Requirements**:
1. **Quality Framework**: Implement comprehensive data quality validation
2. **Automated Monitoring**: Build real-time quality monitoring and alerting
3. **Integration**: Connect quality checks with Airflow and dbt workflows
4. **Data Contracts**: Define and enforce data contracts with SLAs
5. **Dashboards**: Create quality monitoring dashboards and reports
6. **Incident Response**: Implement automated incident response workflows

**Data Sources**:
- Customer transaction data (high volume, real-time)
- Product catalog (frequent updates)
- User behavior events (streaming data)
- Financial reporting data (critical accuracy requirements)

See `exercise.py` for starter code and detailed requirements.

### 🐳 Development Environment Setup

**Quick Start with Docker**:
```bash
# Start the complete development environment
./setup.sh

# Access the Great Expectations container
docker-compose exec great_expectations bash

# Run data quality validations
python exercise.py

# View Grafana dashboard
open http://localhost:3000  # admin/admin
```

**Manual Setup**:
1. Install Great Expectations and dependencies
2. Configure PostgreSQL database
3. Set up Great Expectations context
4. Load sample data for testing

**Infrastructure Included**:
- 🐳 **Docker Compose**: Complete development environment
- 🗄️ **PostgreSQL**: Sample database with test data
- 📦 **Great Expectations**: Pre-configured context and datasources
- 📊 **Grafana**: Data quality monitoring dashboard
- 🧪 **Sample Data**: Customer, transaction, and product data
- 🔧 **Setup Script**: Automated environment initialization

---

## 📚 Resources

- **Great Expectations**: [docs.greatexpectations.io](https://docs.greatexpectations.io/)
- **dbt Testing**: [docs.getdbt.com/docs/build/tests](https://docs.getdbt.com/docs/build/tests)
- **Data Quality Patterns**: [martinfowler.com/articles/data-quality.html](https://martinfowler.com/articles/data-quality.html)
- **Airflow Data Quality**: [airflow.apache.org/docs/apache-airflow-providers-great-expectations](https://airflow.apache.org/docs/apache-airflow-providers-great-expectations)
- **Data Contracts**: [datacontract.com](https://datacontract.com/)
- **Monte Carlo**: [docs.getmontecarlo.com](https://docs.getmontecarlo.com/) - Data observability platform

---

## 🎯 Key Takeaways

- **Comprehensive validation** across all data quality dimensions (completeness, accuracy, consistency, timeliness)
- **Automated monitoring** with real-time alerting and incident response
- **Integration with orchestration** tools ensures quality checks are part of every pipeline
- **Data contracts** provide clear expectations and SLAs for data consumers
- **Proactive quality management** prevents downstream issues and builds trust
- **Quality metrics and dashboards** enable continuous improvement and transparency
- **Shift-left approach** catches quality issues early in the data pipeline
- **Business impact focus** prioritizes quality checks based on downstream usage

---

## 🚀 What's Next?

Tomorrow (Day 20), you'll learn **Data Observability** - comprehensive monitoring, alerting, and incident response for data systems.

**Preview**: You'll explore advanced observability patterns, implement comprehensive monitoring systems, and build automated incident response workflows that provide complete visibility into data system health and performance!

---

## ✅ Before Moving On

- [ ] Understand comprehensive data quality frameworks and dimensions
- [ ] Can implement automated validation with Great Expectations
- [ ] Know how to integrate quality checks with orchestration tools
- [ ] Understand data contracts and SLA management
- [ ] Can build quality monitoring dashboards and alerting
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced Production Data Quality)

Ready to ensure data reliability at scale! 🚀
