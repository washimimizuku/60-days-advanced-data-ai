"""
Day 19: Data Quality in Production - Exercise

Build a comprehensive production data quality system with automated validation, monitoring, and alerting.

Scenario:
You're the Data Quality Engineer at "QualityFirst Corp", a data-driven company 
processing millions of customer records daily. You need to implement a robust 
data quality framework that ensures data reliability across all systems.

Business Context:
- Processing 10M+ customer transactions daily
- Real-time data feeds from multiple sources
- Critical financial reporting requirements
- Regulatory compliance obligations (SOX, GDPR)
- Multiple downstream consumers (analytics, ML, reporting)

Your Task:
Build a comprehensive data quality system with validation, monitoring, and automation.

Requirements:
1. Comprehensive data quality validation framework
2. Real-time monitoring and alerting system
3. Integration with Airflow and dbt workflows
4. Data contracts with SLAs and enforcement
5. Quality dashboards and reporting
6. Automated incident response workflows
"""

import os
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from great_expectations.checkpoint import Checkpoint

# =============================================================================
# DATA QUALITY FRAMEWORK IMPLEMENTATION
# =============================================================================

class QualityFirstDataQualityFramework:
    """Comprehensive data quality framework for QualityFirst Corp"""
    
    def __init__(self):
        self.context = None
        self.quality_thresholds = self._load_quality_thresholds()
        self.data_contracts = self._load_data_contracts()
        
    def initialize_great_expectations(self):
        """Initialize Great Expectations context"""
        
        # TODO: Initialize Great Expectations context
        # Set up data context, datasources, and stores
        
        context_config = {
            "config_version": 3.0,
            "datasources": {
                "production_datasource": {
                    "class_name": "Datasource",
                    "execution_engine": {
                        "class_name": "PandasExecutionEngine"
                    },
                    "data_connectors": {
                        "default_inferred_data_connector": {
                            "class_name": "InferredAssetFilesystemDataConnector",
                            "base_directory": "/data/production",
                            "default_regex": {
                                "group_names": ["data_asset_name"],
                                "pattern": "(.*)\\.csv"
                            }
                        }
                    }
                }
            },
            "stores": {
                "expectations_store": {
                    "class_name": "ExpectationsStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "/opt/great_expectations/expectations/"
                    }
                },
                "validations_store": {
                    "class_name": "ValidationsStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "/opt/great_expectations/validations/"
                    }
                },
                "evaluation_parameter_store": {
                    "class_name": "EvaluationParameterStore"
                },
                "checkpoint_store": {
                    "class_name": "CheckpointStore",
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "/opt/great_expectations/checkpoints/"
                    }
                }
            },
            "expectations_store_name": "expectations_store",
            "validations_store_name": "validations_store",
            "evaluation_parameter_store_name": "evaluation_parameter_store",
            "checkpoint_store_name": "checkpoint_store",
            "data_docs_sites": {
                "local_site": {
                    "class_name": "SiteBuilder",
                    "show_how_to_buttons": True,
                    "store_backend": {
                        "class_name": "TupleFilesystemStoreBackend",
                        "base_directory": "/opt/great_expectations/data_docs/"
                    },
                    "site_index_builder": {
                        "class_name": "DefaultSiteIndexBuilder"
                    }
                }
            }
        }
        
        self.context = gx.get_context(project_config=context_config)
        return self.context
    
    def _load_quality_thresholds(self):
        """Load quality thresholds for datasets"""
        return {
            "customer_data": {"overall_success_rate": 0.95},
            "transaction_data": {"overall_success_rate": 0.99}
        }
    
    def _load_data_contracts(self):
        """Load data contracts"""
        return {"customer_data_v1": {"version": "1.0.0"}}

# =============================================================================
# EXPECTATION SUITES CREATION
# =============================================================================

def create_customer_data_expectations():
    """Create comprehensive expectation suite for customer data"""
    
    # TODO: Create detailed expectation suite for customer data
    # Include completeness, uniqueness, format, range, and business rule validations
    
    suite = gx.ExpectationSuite(expectation_suite_name="customer_data_quality_v1")
    
    # Completeness expectations
    critical_columns = ["customer_id", "email", "created_at", "subscription_status"]
    for column in critical_columns:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_to_exist",
                kwargs={"column": column}
            )
        )
        
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": column,
                    "mostly": 1.0 if column in ["customer_id", "email"] else 0.95
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
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={
                "column": "email",
                "mostly": 0.99  # Allow for some duplicates due to data entry errors
            }
        )
    )
    
    # Format expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "email",
                "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "mostly": 0.95
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "customer_id",
                "regex": r"^CUST_[0-9]{8}$"  # Format: CUST_12345678
            }
        )
    )
    
    # Range and business rule expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "subscription_status",
                "value_set": ["active", "inactive", "suspended", "cancelled", "trial"]
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "age",
                "min_value": 13,  # Minimum age for account creation
                "max_value": 120,
                "mostly": 0.99
            }
        )
    )
    
    # Temporal expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "created_at",
                "min_value": "2020-01-01",  # Company founding date
                "max_value": datetime.now().strftime("%Y-%m-%d"),
                "parse_strings_as_datetimes": True
            }
        )
    )
    
    return suite

def create_transaction_data_expectations():
    """Create expectation suite for transaction data"""
    
    # TODO: Create comprehensive expectations for financial transaction data
    # Focus on accuracy, completeness, and regulatory compliance
    
    suite = gx.ExpectationSuite(expectation_suite_name="transaction_data_quality_v1")
    
    # Critical completeness (100% required for financial data)
    critical_columns = ["transaction_id", "customer_id", "amount", "currency", "timestamp", "status"]
    for column in critical_columns:
        suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={
                    "column": column,
                    "mostly": 1.0  # 100% completeness required for financial data
                }
            )
        )
    
    # Uniqueness
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_unique",
            kwargs={"column": "transaction_id"}
        )
    )
    
    # Financial validation
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "amount",
                "min_value": 0.01,  # Minimum transaction amount
                "max_value": 100000.00,  # Maximum single transaction
                "mostly": 0.999  # Allow for rare large transactions
            }
        )
    )
    
    # Currency validation
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "currency",
                "value_set": ["USD", "EUR", "GBP", "CAD", "AUD"]  # Supported currencies
            }
        )
    )
    
    # Status validation
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "status",
                "value_set": ["pending", "completed", "failed", "cancelled", "refunded"]
            }
        )
    )
    
    # Referential integrity
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "customer_id",
                "regex": r"^CUST_[0-9]{8}$"
            }
        )
    )
    
    return suite

# =============================================================================
# CHECKPOINT CONFIGURATION
# =============================================================================

def create_automated_checkpoints():
    """Create automated checkpoints for production validation"""
    
    # TODO: Create comprehensive checkpoint configurations
    # Include validation, alerting, and documentation updates
    
    checkpoints = {}
    
    # Customer data checkpoint
    checkpoints["customer_data_checkpoint"] = {
        "name": "customer_data_checkpoint",
        "config_version": 1.0,
        "template_name": None,
        "module_name": "great_expectations.checkpoint",
        "class_name": "Checkpoint",
        "run_name_template": "%Y%m%d-%H%M%S-customer-data-validation",
        "expectation_suite_name": "customer_data_quality_v1",
        "batch_request": {
            "datasource_name": "production_datasource",
            "data_connector_name": "default_inferred_data_connector",
            "data_asset_name": "customer_data"
        },
        "action_list": [
            {
                "name": "store_validation_result",
                "action": {
                    "class_name": "StoreValidationResultAction"
                }
            },
            {
                "name": "update_data_docs",
                "action": {
                    "class_name": "UpdateDataDocsAction"
                }
            },
            {
                "name": "slack_notification_on_failure",
                "action": {
                    "class_name": "SlackNotificationAction",
                    "slack_webhook": "${SLACK_WEBHOOK_URL}",
                    "notify_on": "failure",
                    "renderer": {
                        "module_name": "great_expectations.render.renderer.slack_renderer",
                        "class_name": "SlackRenderer"
                    }
                }
            },
            {
                "name": "email_notification_critical",
                "action": {
                    "class_name": "EmailAction",
                    "notify_on": "failure",
                    "notify_with": ["data-quality@qualityfirst.com", "oncall@qualityfirst.com"],
                    "renderer": {
                        "module_name": "great_expectations.render.renderer.email_renderer",
                        "class_name": "EmailRenderer"
                    }
                }
            }
        ]
    }
    
    # Transaction data checkpoint (more critical)
    checkpoints["transaction_data_checkpoint"] = {
        "name": "transaction_data_checkpoint",
        "config_version": 1.0,
        "template_name": None,
        "module_name": "great_expectations.checkpoint",
        "class_name": "Checkpoint",
        "run_name_template": "%Y%m%d-%H%M%S-transaction-data-validation",
        "expectation_suite_name": "transaction_data_quality_v1",
        "batch_request": {
            "datasource_name": "production_datasource",
            "data_connector_name": "default_inferred_data_connector",
            "data_asset_name": "transaction_data"
        },
        "action_list": [
            {
                "name": "store_validation_result",
                "action": {
                    "class_name": "StoreValidationResultAction"
                }
            },
            {
                "name": "update_data_docs",
                "action": {
                    "class_name": "UpdateDataDocsAction"
                }
            },
            {
                "name": "pagerduty_notification",
                "action": {
                    "class_name": "PagerdutyAlertAction",
                    "api_key": "${PAGERDUTY_API_KEY}",
                    "routing_key": "${PAGERDUTY_ROUTING_KEY}",
                    "notify_on": "failure"
                }
            },
            {
                "name": "slack_notification_urgent",
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
        ]
    }
    
    return checkpoints

# =============================================================================
# AIRFLOW INTEGRATION
# =============================================================================

def create_data_quality_dag():
    """Create Airflow DAG with integrated data quality checks"""
    
    # TODO: Create comprehensive Airflow DAG with quality validation
    # Include pre-processing validation, post-processing validation, and quality metrics collection
    
    dag_code = '''
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow_providers_great_expectations.operators.great_expectations import GreatExpectationsOperator
from airflow.operators.email import EmailOperator

default_args = {
    'owner': 'data-quality-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['data-quality@qualityfirst.com']
}

dag = DAG(
    'production_data_quality_pipeline',
    default_args=default_args,
    description='Production data quality validation pipeline',
    schedule_interval='@hourly',
    catchup=False,
    tags=['data_quality', 'production', 'critical']
)

# Data extraction and staging
extract_customer_data = PythonOperator(
    task_id='extract_customer_data',
    python_callable=extract_customer_data_function,
    dag=dag
)

extract_transaction_data = PythonOperator(
    task_id='extract_transaction_data',
    python_callable=extract_transaction_data_function,
    dag=dag
)

# Pre-processing quality validation
validate_customer_data_raw = GreatExpectationsOperator(
    task_id='validate_customer_data_raw',
    checkpoint_name='customer_data_checkpoint',
    data_context_root_dir='/opt/great_expectations',
    fail_task_on_validation_failure=True,
    dag=dag
)

validate_transaction_data_raw = GreatExpectationsOperator(
    task_id='validate_transaction_data_raw',
    checkpoint_name='transaction_data_checkpoint',
    data_context_root_dir='/opt/great_expectations',
    fail_task_on_validation_failure=True,
    dag=dag
)

# Data processing (only proceeds if validation passes)
process_customer_data = PythonOperator(
    task_id='process_customer_data',
    python_callable=process_customer_data_function,
    dag=dag
)

process_transaction_data = PythonOperator(
    task_id='process_transaction_data',
    python_callable=process_transaction_data_function,
    dag=dag
)

# Post-processing quality validation
validate_processed_data = GreatExpectationsOperator(
    task_id='validate_processed_data',
    checkpoint_name='processed_data_checkpoint',
    data_context_root_dir='/opt/great_expectations',
    fail_task_on_validation_failure=True,
    dag=dag
)

# Quality metrics collection and reporting
collect_quality_metrics = PythonOperator(
    task_id='collect_quality_metrics',
    python_callable=collect_quality_metrics_function,
    dag=dag
)

# Data quality report generation
generate_quality_report = PythonOperator(
    task_id='generate_quality_report',
    python_callable=generate_quality_report_function,
    dag=dag
)

# Define task dependencies
[extract_customer_data, extract_transaction_data] >> [validate_customer_data_raw, validate_transaction_data_raw]
[validate_customer_data_raw, validate_transaction_data_raw] >> [process_customer_data, process_transaction_data]
[process_customer_data, process_transaction_data] >> validate_processed_data
validate_processed_data >> collect_quality_metrics >> generate_quality_report
'''
    
    return dag_code

# Missing function implementations for Airflow DAG
def extract_customer_data_function():
    """Extract customer data"""
    print("Extracting customer data...")

def extract_transaction_data_function():
    """Extract transaction data"""
    print("Extracting transaction data...")

def process_customer_data_function():
    """Process customer data"""
    print("Processing customer data...")

def process_transaction_data_function():
    """Process transaction data"""
    print("Processing transaction data...")

def collect_quality_metrics_function():
    """Collect quality metrics"""
    print("Collecting quality metrics...")

def generate_quality_report_function():
    """Generate quality report"""
    print("Generating quality report...")

# =============================================================================
# DATA CONTRACTS IMPLEMENTATION
# =============================================================================

def create_data_contracts():
    """Create comprehensive data contracts with SLAs"""
    
    # TODO: Define detailed data contracts for all critical datasets
    # Include schema definitions, quality requirements, and SLAs
    
    contracts = {}
    
    # Customer data contract
    contracts["customer_data_v1"] = {
        "contract_version": "1.2.0",
        "name": "customer_data_contract",
        "description": "Customer data contract for analytics and ML workloads",
        
        "owner": {
            "team": "data-platform",
            "contact": "data-platform@qualityfirst.com",
            "slack": "#data-platform"
        },
        
        "consumers": [
            {
                "team": "analytics",
                "use_case": "customer_segmentation",
                "contact": "analytics@qualityfirst.com"
            },
            {
                "team": "ml-platform",
                "use_case": "churn_prediction",
                "contact": "ml-platform@qualityfirst.com"
            },
            {
                "team": "marketing",
                "use_case": "campaign_targeting",
                "contact": "marketing@qualityfirst.com"
            }
        ],
        
        "schema": {
            "format": "parquet",
            "columns": [
                {
                    "name": "customer_id",
                    "type": "string",
                    "nullable": False,
                    "unique": True,
                    "format": "CUST_[0-9]{8}",
                    "description": "Unique customer identifier"
                },
                {
                    "name": "email",
                    "type": "string",
                    "nullable": False,
                    "unique": True,
                    "format": "email",
                    "description": "Customer email address"
                },
                {
                    "name": "first_name",
                    "type": "string",
                    "nullable": False,
                    "description": "Customer first name"
                },
                {
                    "name": "last_name",
                    "type": "string",
                    "nullable": False,
                    "description": "Customer last name"
                },
                {
                    "name": "age",
                    "type": "integer",
                    "nullable": True,
                    "min_value": 13,
                    "max_value": 120,
                    "description": "Customer age in years"
                },
                {
                    "name": "subscription_status",
                    "type": "string",
                    "nullable": False,
                    "enum": ["active", "inactive", "suspended", "cancelled", "trial"],
                    "description": "Current subscription status"
                },
                {
                    "name": "created_at",
                    "type": "timestamp",
                    "nullable": False,
                    "description": "Account creation timestamp"
                },
                {
                    "name": "updated_at",
                    "type": "timestamp",
                    "nullable": False,
                    "description": "Last profile update timestamp"
                }
            ]
        },
        
        "quality_requirements": {
            "completeness": {
                "customer_id": 1.0,
                "email": 1.0,
                "first_name": 0.95,
                "last_name": 0.95,
                "subscription_status": 1.0,
                "created_at": 1.0,
                "updated_at": 1.0
            },
            "uniqueness": {
                "customer_id": 1.0,
                "email": 0.99
            },
            "accuracy": {
                "email_format_valid": 0.95,
                "age_range_valid": 0.99,
                "subscription_status_valid": 1.0
            },
            "freshness": {
                "max_age_hours": 4,
                "update_frequency": "hourly"
            }
        },
        
        "sla": {
            "availability": 99.9,
            "latency_p95_ms": 500,
            "error_rate_max": 0.1,
            "data_quality_score_min": 0.95
        },
        
        "monitoring": {
            "alerts": [
                {
                    "condition": "completeness < 0.95",
                    "severity": "high",
                    "notification": ["slack", "email"]
                },
                {
                    "condition": "freshness > 6 hours",
                    "severity": "critical",
                    "notification": ["pagerduty", "slack", "email"]
                },
                {
                    "condition": "uniqueness < 0.99",
                    "severity": "medium",
                    "notification": ["slack"]
                }
            ],
            "dashboard_url": "https://monitoring.qualityfirst.com/customer-data",
            "runbook_url": "https://docs.qualityfirst.com/runbooks/customer-data"
        }
    }
    
    return contracts

# =============================================================================
# QUALITY MONITORING SYSTEM
# =============================================================================

class QualityMonitoringSystem:
    """Real-time data quality monitoring and alerting system"""
    
    def __init__(self):
        self.metrics_store = None  # Initialize with actual metrics store
        self.alert_manager = None  # Initialize with actual alert manager
        
    def monitor_data_quality(self, dataset_name, validation_results):
        """Monitor data quality in real-time and trigger alerts"""
        
        # TODO: Implement comprehensive quality monitoring
        # Calculate metrics, store trends, and trigger alerts based on thresholds
        
        quality_metrics = self._calculate_quality_metrics(validation_results)
        self._store_quality_metrics(dataset_name, quality_metrics)
        self._check_quality_thresholds(dataset_name, quality_metrics)
        
        return quality_metrics
    
    def _calculate_quality_metrics(self, validation_results):
        """Calculate comprehensive quality metrics from validation results"""
        
        total_expectations = len(validation_results.results)
        successful_expectations = sum(1 for result in validation_results.results if result.success)
        
        return {
            'overall_success_rate': successful_expectations / total_expectations if total_expectations > 0 else 0,
            'total_expectations': total_expectations,
            'successful_expectations': successful_expectations,
            'failed_expectations': total_expectations - successful_expectations,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _store_quality_metrics(self, dataset_name, quality_metrics):
        """Store quality metrics for trending"""
        print(f"Storing quality metrics for {dataset_name}: {quality_metrics}")
    
    def _check_quality_thresholds(self, dataset_name, quality_metrics):
        """Check quality thresholds and trigger alerts"""
        if quality_metrics['overall_success_rate'] < 0.95:
            print(f"Quality alert for {dataset_name}: {quality_metrics['overall_success_rate']:.2f} < 0.95")
    
    def generate_quality_dashboard_config(self):
        """Generate configuration for quality monitoring dashboard"""
        
        return {
            "dashboard": {
                "title": "Data Quality Dashboard",
                "panels": [
                    {"title": "Overall Quality Score", "type": "stat"},
                    {"title": "Quality Trends", "type": "timeseries"},
                    {"title": "Failed Expectations", "type": "table"}
                ]
            }
        }

# =============================================================================
# EXERCISE INSTRUCTIONS
# =============================================================================

def print_exercise_instructions():
    """Print detailed exercise instructions"""
    
    print("🎯 Data Quality in Production Exercise - QualityFirst Corp")
    print("=" * 65)
    
    print("\n📋 REQUIREMENTS:")
    print("1. Implement comprehensive data quality validation framework")
    print("2. Build real-time monitoring and alerting system")
    print("3. Integrate quality checks with Airflow and dbt workflows")
    print("4. Define and enforce data contracts with SLAs")
    print("5. Create quality monitoring dashboards and reports")
    print("6. Implement automated incident response workflows")
    
    print("\n🏗️ SYSTEM ARCHITECTURE:")
    print("""
    QualityFirst Data Quality System:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Data Sources                                 │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │  Customer   │  │Transaction  │  │  Product    │             │
    │  │    Data     │  │    Data     │  │  Catalog    │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                Quality Validation Layer                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │Great        │  │   dbt       │  │  Custom     │             │
    │  │Expectations │  │   Tests     │  │ Validators  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                Orchestration Layer                              │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │   Airflow   │  │    dbt      │  │   CI/CD     │             │
    │  │    DAGs     │  │  Pipeline   │  │  Pipeline   │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │              Monitoring & Alerting                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │  Grafana    │  │    Slack    │  │ PagerDuty   │             │
    │  │ Dashboard   │  │   Alerts    │  │   Alerts    │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("• Comprehensive validation covering all quality dimensions")
    print("• Real-time monitoring with appropriate alerting thresholds")
    print("• Seamless integration with existing orchestration tools")
    print("• Clear data contracts with enforced SLAs")
    print("• Actionable dashboards and quality reports")
    print("• Automated incident response and escalation")
    
    print("\n🚀 GETTING STARTED:")
    print("1. Set up Great Expectations context and datasources")
    print("2. Create comprehensive expectation suites for each dataset")
    print("3. Configure automated checkpoints with alerting")
    print("4. Integrate quality checks into Airflow DAGs")
    print("5. Define data contracts with quality requirements")
    print("6. Build monitoring dashboards and alerting rules")
    print("7. Test the complete quality validation pipeline")

if __name__ == "__main__":
    print_exercise_instructions()
    
    print("\n" + "="*65)
    print("🎯 Ready to build production data quality systems!")
    print("Complete the TODOs above to create a comprehensive quality framework.")
    print("="*65)
