#!/usr/bin/env python3
"""
Generate sample configuration files and test data for TechCorp pipelines
"""

import json
import yaml
import csv
from datetime import datetime, timedelta
from faker import Faker
import os

fake = Faker()

def generate_sample_customer_data():
    """Generate sample customer data CSV"""
    customers = []
    
    for i in range(1000):
        customer = {
            'customer_id': i + 1,
            'email': fake.email(),
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'phone': fake.phone_number(),
            'address': fake.address().replace('\n', ', '),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip_code': fake.zipcode(),
            'created_at': fake.date_time_between(start_date='-2y', end_date='now').isoformat(),
            'status': fake.random_element(['active', 'inactive', 'suspended'])
        }
        customers.append(customer)
    
    # Write to CSV
    os.makedirs('data/input', exist_ok=True)
    with open('data/input/sample_customers.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=customers[0].keys())
        writer.writeheader()
        writer.writerows(customers)
    
    print(f"✅ Generated {len(customers)} sample customers")

def generate_sample_product_catalog():
    """Generate sample product catalog"""
    products = []
    categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports']
    
    for i in range(500):
        product = {
            'product_id': f'PROD-{i+1:04d}',
            'name': fake.catch_phrase(),
            'description': fake.text(max_nb_chars=200),
            'category': fake.random_element(categories),
            'price': round(fake.random.uniform(9.99, 999.99), 2),
            'cost': round(fake.random.uniform(5.00, 500.00), 2),
            'status': fake.random_element(['active', 'inactive', 'discontinued']),
            'created_at': fake.date_time_between(start_date='-1y', end_date='now').isoformat(),
            'updated_at': fake.date_time_between(start_date='-30d', end_date='now').isoformat()
        }
        products.append(product)
    
    # Write today's catalog
    today = datetime.now().strftime('%Y-%m-%d')
    os.makedirs('data/products', exist_ok=True)
    with open(f'data/products/catalog_{today}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)
    
    print(f"✅ Generated {len(products)} sample products")

def generate_sample_events():
    """Generate sample user behavior events"""
    events = []
    event_types = ['click', 'view', 'purchase', 'signup']
    
    for i in range(5000):
        event = {
            'event_id': f'EVT-{i+1:06d}',
            'user_id': fake.random_int(min=1, max=1000),
            'event_type': fake.random_element(event_types),
            'timestamp': fake.date_time_between(start_date='-7d', end_date='now').isoformat(),
            'page_url': fake.url(),
            'user_agent': fake.user_agent(),
            'ip_address': fake.ipv4(),
            'session_id': fake.uuid4(),
            'properties': json.dumps({
                'product_id': fake.random_int(min=1, max=500) if fake.boolean() else None,
                'category': fake.random_element(['Electronics', 'Clothing', 'Books']),
                'value': round(fake.random.uniform(10, 500), 2) if fake.boolean() else None
            })
        }
        events.append(event)
    
    # Write to JSON
    with open('data/input/sample_events.json', 'w') as f:
        json.dump(events, f, indent=2)
    
    print(f"✅ Generated {len(events)} sample events")

def generate_partner_data():
    """Generate sample partner integration data"""
    partner_data = []
    integration_types = ['payment', 'shipping', 'analytics', 'marketing']
    
    for i in range(200):
        data = {
            'partner_id': f'PARTNER-{fake.random_int(min=1, max=10):02d}',
            'integration_type': fake.random_element(integration_types),
            'transaction_id': fake.uuid4(),
            'timestamp': fake.date_time_between(start_date='-1d', end_date='now').isoformat(),
            'status': fake.random_element(['success', 'pending', 'failed']),
            'metadata': {
                'amount': round(fake.random.uniform(10, 1000), 2),
                'currency': 'USD',
                'customer_id': fake.random_int(min=1, max=1000),
                'reference': fake.bothify('REF-####-????')
            }
        }
        partner_data.append(data)
    
    # Write today's partner data
    today = datetime.now().strftime('%Y-%m-%d')
    with open(f'data/input/partner_data_{today}.json', 'w') as f:
        json.dump(partner_data, f, indent=2)
    
    print(f"✅ Generated {len(partner_data)} partner integration records")

def generate_airflow_variables():
    """Generate Airflow variables configuration"""
    variables = {
        'techcorp_api_base_url': 'https://api.techcorp.com',
        'techcorp_events_api_url': 'https://events.techcorp.com/api/v1',
        'data_quality_threshold': '0.95',
        'max_processing_time_minutes': '120',
        'alert_email': 'data-team@techcorp.com',
        'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
        'partner_sftp_host': 'sftp.partner.com',
        'data_retention_days': '2555'  # 7 years
    }
    
    os.makedirs('config', exist_ok=True)
    with open('config/airflow_variables.json', 'w') as f:
        json.dump(variables, f, indent=2)
    
    print(f"✅ Generated {len(variables)} Airflow variables")

def generate_monitoring_config():
    """Generate monitoring dashboard configuration"""
    monitoring_config = {
        'dashboards': {
            'pipeline_health': {
                'refresh_interval': '30s',
                'panels': [
                    {
                        'title': 'Pipeline Success Rate',
                        'type': 'stat',
                        'query': 'pipeline_success_rate',
                        'thresholds': [0.8, 0.95]
                    },
                    {
                        'title': 'Data Processing Volume',
                        'type': 'graph',
                        'query': 'data_records_processed_total',
                        'time_range': '24h'
                    },
                    {
                        'title': 'Pool Utilization',
                        'type': 'table',
                        'query': 'pool_utilization_percent',
                        'alert_threshold': 80
                    }
                ]
            }
        },
        'alerts': {
            'pipeline_failure': {
                'condition': 'pipeline_success_rate < 0.8',
                'severity': 'critical',
                'channels': ['slack', 'email']
            },
            'high_pool_utilization': {
                'condition': 'pool_utilization_percent > 80',
                'severity': 'warning',
                'channels': ['slack']
            },
            'data_freshness_violation': {
                'condition': 'data_freshness_hours > sla_hours',
                'severity': 'high',
                'channels': ['slack', 'email']
            }
        }
    }
    
    with open('config/monitoring.yml', 'w') as f:
        yaml.dump(monitoring_config, f, default_flow_style=False)
    
    print("✅ Generated monitoring configuration")

def main():
    """Generate all sample configurations and data"""
    print("🚀 Generating sample configurations and test data for TechCorp...")
    print()
    
    # Generate sample data
    generate_sample_customer_data()
    generate_sample_product_catalog()
    generate_sample_events()
    generate_partner_data()
    
    # Generate configurations
    generate_airflow_variables()
    generate_monitoring_config()
    
    print()
    print("🎉 Sample data generation complete!")
    print()
    print("📁 Generated files:")
    print("  • data/input/sample_customers.csv")
    print("  • data/products/catalog_YYYY-MM-DD.csv")
    print("  • data/input/sample_events.json")
    print("  • data/input/partner_data_YYYY-MM-DD.json")
    print("  • config/airflow_variables.json")
    print("  • config/monitoring.yml")
    print()
    print("🔧 Next steps:")
    print("  1. Load Airflow variables: airflow variables import config/airflow_variables.json")
    print("  2. Start Airflow: docker-compose up -d")
    print("  3. Setup pools: ./scripts/setup_pools.sh")
    print("  4. Trigger test pipeline: airflow dags trigger techcorp_pipeline_customer_data")

if __name__ == "__main__":
    main()