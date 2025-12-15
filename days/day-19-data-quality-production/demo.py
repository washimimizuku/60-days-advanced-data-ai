#!/usr/bin/env python3
"""
Day 19: Data Quality in Production - Working Demo

This script demonstrates the complete data quality framework with real validations.
"""

import os
import pandas as pd
import great_expectations as gx
from great_expectations.core.expectation_configuration import ExpectationConfiguration
from sqlalchemy import create_engine
from exercise import (
    QualityFirstDataQualityFramework,
    create_customer_data_expectations,
    create_transaction_data_expectations,
    QualityMonitoringSystem
)

def main():
    """Run complete data quality validation demo"""
    
    print("🚀 QualityFirst Corp - Data Quality Framework Demo")
    print("=" * 55)
    
    # Database connection
    db_url = "postgresql://quality_user:quality_password@localhost:5432/qualityfirst_data"
    
    try:
        # Initialize framework
        framework = QualityFirstDataQualityFramework()
        monitoring = QualityMonitoringSystem()
        
        print("\n📊 Connecting to database...")
        engine = create_engine(db_url)
        
        # Load sample data
        print("📋 Loading sample data...")
        customer_df = pd.read_sql("SELECT * FROM quality_test.customer_data", engine)
        transaction_df = pd.read_sql("SELECT * FROM quality_test.transaction_data", engine)
        
        print(f"✅ Loaded {len(customer_df)} customers and {len(transaction_df)} transactions")
        
        # Initialize Great Expectations context
        print("\n🔧 Initializing Great Expectations...")
        context = gx.get_context()
        
        # Create datasource
        datasource_config = {
            "name": "demo_datasource",
            "class_name": "Datasource",
            "execution_engine": {
                "class_name": "SqlAlchemyExecutionEngine",
                "connection_string": db_url
            },
            "data_connectors": {
                "default_runtime_data_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier_name"]
                }
            }
        }
        
        try:
            datasource = context.add_datasource(**datasource_config)
            print("✅ Datasource created successfully")
        except Exception as e:
            print(f"ℹ️  Using existing datasource: {e}")
            datasource = context.get_datasource("demo_datasource")
        
        # Create expectation suites
        print("\n📝 Creating expectation suites...")
        customer_suite = create_customer_data_expectations()
        transaction_suite = create_transaction_data_expectations()
        
        # Save suites to context
        try:
            context.add_expectation_suite(customer_suite)
            print("✅ Customer expectation suite created")
        except Exception:
            print("ℹ️  Customer suite already exists")
        
        try:
            context.add_expectation_suite(transaction_suite)
            print("✅ Transaction expectation suite created")
        except Exception:
            print("ℹ️  Transaction suite already exists")
        
        # Run validations
        print("\n🧪 Running data quality validations...")
        
        # Customer data validation
        print("\n👥 Validating customer data...")
        customer_batch_request = {
            "datasource_name": "demo_datasource",
            "data_connector_name": "default_runtime_data_connector",
            "data_asset_name": "customer_data",
            "runtime_parameters": {"query": "SELECT * FROM quality_test.customer_data"},
            "batch_identifiers": {"default_identifier_name": "customer_validation"}
        }
        
        customer_validator = context.get_validator(
            batch_request=customer_batch_request,
            expectation_suite_name="customer_data_quality_v1"
        )
        
        customer_results = customer_validator.validate()
        
        # Transaction data validation
        print("💳 Validating transaction data...")
        transaction_batch_request = {
            "datasource_name": "demo_datasource",
            "data_connector_name": "default_runtime_data_connector",
            "data_asset_name": "transaction_data",
            "runtime_parameters": {"query": "SELECT * FROM quality_test.transaction_data"},
            "batch_identifiers": {"default_identifier_name": "transaction_validation"}
        }
        
        transaction_validator = context.get_validator(
            batch_request=transaction_batch_request,
            expectation_suite_name="transaction_data_quality_v1"
        )
        
        transaction_results = transaction_validator.validate()
        
        # Process results with monitoring system
        print("\n📈 Processing validation results...")
        
        customer_metrics = monitoring.monitor_data_quality("customer_data", customer_results)
        transaction_metrics = monitoring.monitor_data_quality("transaction_data", transaction_results)
        
        # Display results
        print("\n📊 VALIDATION RESULTS:")
        print("=" * 40)
        
        print(f"\n👥 Customer Data Quality:")
        print(f"   Overall Success Rate: {customer_metrics['overall_success_rate']:.1%}")
        print(f"   Total Expectations: {customer_metrics['total_expectations']}")
        print(f"   Successful: {customer_metrics['successful_expectations']}")
        print(f"   Failed: {customer_metrics['failed_expectations']}")
        
        print(f"\n💳 Transaction Data Quality:")
        print(f"   Overall Success Rate: {transaction_metrics['overall_success_rate']:.1%}")
        print(f"   Total Expectations: {transaction_metrics['total_expectations']}")
        print(f"   Successful: {transaction_metrics['successful_expectations']}")
        print(f"   Failed: {transaction_metrics['failed_expectations']}")
        
        # Show failed expectations if any
        if customer_results.statistics['unsuccessful_expectations'] > 0:
            print(f"\n❌ Customer Data Issues:")
            for result in customer_results.results:
                if not result.success:
                    print(f"   - {result.expectation_config.expectation_type}")
        
        if transaction_results.statistics['unsuccessful_expectations'] > 0:
            print(f"\n❌ Transaction Data Issues:")
            for result in transaction_results.results:
                if not result.success:
                    print(f"   - {result.expectation_config.expectation_type}")
        
        # Generate dashboard config
        print("\n📊 Generating dashboard configuration...")
        dashboard_config = monitoring.generate_quality_dashboard_config()
        print("✅ Dashboard configuration generated")
        
        print("\n🎉 Data quality validation complete!")
        print("\n📋 Next Steps:")
        print("1. View detailed results in Great Expectations data docs")
        print("2. Set up automated checkpoints for continuous monitoring")
        print("3. Configure alerting for quality threshold violations")
        print("4. Integrate with Airflow for production pipelines")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure PostgreSQL is running and accessible")
        print("Run: docker-compose up -d")

if __name__ == "__main__":
    main()