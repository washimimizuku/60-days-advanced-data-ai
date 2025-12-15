"""
Day 22: AWS Glue & Data Catalog - Exercise

Build a comprehensive serverless ETL pipeline using AWS Glue, Data Catalog, and Athena
for ServerlessData Corp's customer analytics platform.

Scenario:
You're the Data Engineering Lead at "ServerlessData Corp", a rapidly growing e-commerce 
company. You need to build a scalable, cost-effective data lake solution that can handle 
increasing data volumes while providing business analysts with immediate access to insights.

Business Context:
- Processing 10GB+ of customer transaction data daily from multiple sources
- Data arrives in various formats (CSV, JSON, Parquet) with evolving schemas
- Need automated schema discovery to handle new data sources
- Business analysts require SQL access for ad-hoc analysis
- Cost optimization is critical - serverless-first approach
- Compliance requires data lineage and governance tracking

Your Task:
Implement a production-ready serverless data pipeline using AWS Glue components.

Requirements:
1. Set up Data Catalog with databases and automated schema discovery
2. Create crawlers for multiple data sources with different formats
3. Build Glue ETL jobs for data transformation and enrichment
4. Implement Athena integration for business analyst access
5. Add monitoring, optimization, and cost tracking
6. Ensure data quality and governance throughout the pipeline
"""

import boto3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# =============================================================================
# AWS SERVICE CLIENTS SETUP
# =============================================================================

# Load environment configuration
from dotenv import load_dotenv
load_dotenv()

# Initialize AWS clients with environment configuration
def get_aws_clients():
    """Initialize AWS clients with proper configuration"""
    endpoint_url = os.getenv('AWS_ENDPOINT_URL') if os.getenv('USE_LOCALSTACK', 'false').lower() == 'true' else None
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    return {
        'glue': boto3.client('glue', region_name=region, endpoint_url=endpoint_url),
        'athena': boto3.client('athena', region_name=region, endpoint_url=endpoint_url),
        's3': boto3.client('s3', region_name=region, endpoint_url=endpoint_url),
        'cloudwatch': boto3.client('cloudwatch', region_name=region, endpoint_url=endpoint_url)
    }

# Configuration from environment
DATABASE_NAME = os.getenv('GLUE_DATABASE_NAME', 'serverlessdata_analytics')
S3_DATA_BUCKET = os.getenv('S3_DATA_BUCKET', 'serverlessdata-datalake')
S3_SCRIPTS_BUCKET = os.getenv('S3_SCRIPTS_BUCKET', 'serverlessdata-glue-scripts')
S3_ATHENA_RESULTS = os.getenv('S3_ATHENA_RESULTS', 'serverlessdata-athena-results')
IAM_GLUE_ROLE = os.getenv('GLUE_IAM_ROLE', f"arn:aws:iam::{os.getenv('AWS_ACCOUNT_ID', '123456789012')}:role/GlueServiceRole")

# =============================================================================
# DATA CATALOG MANAGEMENT
# =============================================================================

class DataCatalogManager:
    """Manage AWS Glue Data Catalog operations"""
    
    def __init__(self, database_name: str):
        self.database_name = database_name
        self.clients = get_aws_clients()
        self.glue_client = self.clients['glue']
    
    def setup_data_catalog(self) -> Dict[str, Any]:
        """Set up complete Data Catalog infrastructure"""
        
        catalog_setup = {
            'database_created': False,
            'tables_created': 0,
            'governance_enabled': False
        }
        
        try:
            # Create main analytics database
            database_response = self._create_database(
                self.database_name,
                "Main analytics database for ServerlessData Corp customer data"
            )
            catalog_setup['database_created'] = True
            
            # Create table definitions for known schemas
            table_definitions = self._get_table_definitions()
            
            for table_name, table_config in table_definitions.items():
                table_response = self._create_table_definition(table_name, table_config)
                if table_response:
                    catalog_setup['tables_created'] += 1
            
            # Enable governance and lineage
            governance_response = self._enable_governance_features()
            catalog_setup['governance_enabled'] = governance_response
            
            print(f"✅ Data Catalog setup completed:")
            print(f"   Database: {self.database_name}")
            print(f"   Tables created: {catalog_setup['tables_created']}")
            print(f"   Governance enabled: {catalog_setup['governance_enabled']}")
            
            return catalog_setup
            
        except Exception as e:
            print(f"❌ Error setting up Data Catalog: {str(e)}")
            raise
    
    def _create_database(self, database_name: str, description: str) -> Dict[str, Any]:
        """Create database in Glue Data Catalog"""
        
        # Create database with comprehensive metadata and governance
        
        database_input = {
            'Name': database_name,
            'Description': description,
            'Parameters': {
                'classification': 'customer_analytics',
                'created_by': 'data_engineering_team',
                'environment': 'production',
                'data_owner': 'analytics_team@serverlessdata.com',
                'retention_policy': '7_years',
                'compliance_level': 'high'
            }
        }
        
        try:
            response = self.glue_client.create_database(DatabaseInput=database_input)
            print(f"Database '{database_name}' created successfully")
            return response
            
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Database '{database_name}' already exists")
            return {'status': 'already_exists'}
        except Exception as e:
            print(f"Error creating database: {str(e)}")
            raise
    
    def _get_table_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Define table schemas for known data sources"""
        
        # Define comprehensive table schemas for all data sources
        
        table_definitions = {
            'customer_transactions': {
                'columns': [
                    {'Name': 'transaction_id', 'Type': 'string'},
                    {'Name': 'customer_id', 'Type': 'string'},
                    {'Name': 'product_id', 'Type': 'string'},
                    {'Name': 'transaction_amount', 'Type': 'decimal(10,2)'},
                    {'Name': 'transaction_date', 'Type': 'timestamp'},
                    {'Name': 'payment_method', 'Type': 'string'},
                    {'Name': 'merchant_category', 'Type': 'string'},
                    {'Name': 'currency_code', 'Type': 'string'}
                ],
                's3_location': f's3://{S3_DATA_BUCKET}/raw/transactions/',
                'input_format': 'org.apache.hadoop.mapred.TextInputFormat',
                'output_format': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                'serde_info': {
                    'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                    'Parameters': {'field.delim': ',', 'skip.header.line.count': '1'}
                },
                'partition_keys': [
                    {'Name': 'year', 'Type': 'string'},
                    {'Name': 'month', 'Type': 'string'},
                    {'Name': 'day', 'Type': 'string'}
                ]
            },
            'customer_profiles': {
                'columns': [
                    {'Name': 'customer_id', 'Type': 'string'},
                    {'Name': 'first_name', 'Type': 'string'},
                    {'Name': 'last_name', 'Type': 'string'},
                    {'Name': 'email', 'Type': 'string'},
                    {'Name': 'registration_date', 'Type': 'date'},
                    {'Name': 'customer_segment', 'Type': 'string'},
                    {'Name': 'preferred_category', 'Type': 'string'},
                    {'Name': 'lifetime_value', 'Type': 'decimal(12,2)'}
                ],
                's3_location': f's3://{S3_DATA_BUCKET}/raw/customers/',
                'input_format': 'org.apache.hadoop.mapred.TextInputFormat',
                'output_format': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                'serde_info': {
                    'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                    'Parameters': {'field.delim': ',', 'skip.header.line.count': '1'}
                },
                'partition_keys': [
                    {'Name': 'segment', 'Type': 'string'}
                ]
            }
        }
        
        return table_definitions
    
    def _create_table_definition(self, table_name: str, table_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create table definition in Data Catalog"""
        
        # Create table definition with comprehensive metadata
        
        storage_descriptor = {
            'Columns': table_config['columns'],
            'Location': table_config['s3_location'],
            'InputFormat': table_config['input_format'],
            'OutputFormat': table_config['output_format'],
            'SerdeInfo': table_config['serde_info']
        }
        
        table_input = {
            'Name': table_name,
            'StorageDescriptor': storage_descriptor,
            'PartitionKeys': table_config.get('partition_keys', []),
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'csv',
                'compressionType': 'gzip',
                'typeOfData': 'file',
                'data_owner': 'analytics_team',
                'update_frequency': 'daily',
                'quality_score': 'high'
            }
        }
        
        try:
            response = self.glue_client.create_table(
                DatabaseName=self.database_name,
                TableInput=table_input
            )
            print(f"Table '{table_name}' created successfully")
            return response
            
        except Exception as e:
            print(f"Error creating table '{table_name}': {str(e)}")
            return None
    
    def _enable_governance_features(self) -> bool:
        """Enable governance and lineage features"""
        
        # Enable governance features (simulated for LocalStack environment)
        
        try:
            # In a real implementation, this would configure:
            # - Lake Formation permissions
            # - Data lineage tracking
            # - Data quality rules
            # - Access controls
            
            print("Governance features enabled (simulated)")
            return True
            
        except Exception as e:
            print(f"Error enabling governance: {str(e)}")
            return False

# =============================================================================
# CRAWLER MANAGEMENT
# =============================================================================

class CrawlerManager:
    """Manage AWS Glue Crawlers for automated schema discovery"""
    
    def __init__(self, database_name: str, iam_role: str):
        self.database_name = database_name
        self.iam_role = iam_role
        self.clients = get_aws_clients()
        self.glue_client = self.clients['glue']
    
    def create_comprehensive_crawlers(self) -> Dict[str, Any]:
        """Create crawlers for all data sources"""
        
        # Create multiple crawlers for different data sources and formats
        
        crawler_results = {
            'crawlers_created': 0,
            'crawlers_started': 0,
            'failed_crawlers': []
        }
        
        # Define crawler configurations
        crawler_configs = self._get_crawler_configurations()
        
        for crawler_name, config in crawler_configs.items():
            try:
                # Create crawler
                create_response = self._create_crawler(crawler_name, config)
                if create_response:
                    crawler_results['crawlers_created'] += 1
                    
                    # Start crawler
                    start_response = self._start_crawler(crawler_name)
                    if start_response:
                        crawler_results['crawlers_started'] += 1
                
            except Exception as e:
                print(f"Failed to create/start crawler '{crawler_name}': {str(e)}")
                crawler_results['failed_crawlers'].append({
                    'crawler_name': crawler_name,
                    'error': str(e)
                })
        
        print(f"✅ Crawler setup completed:")
        print(f"   Crawlers created: {crawler_results['crawlers_created']}")
        print(f"   Crawlers started: {crawler_results['crawlers_started']}")
        
        return crawler_results
    
    def _get_crawler_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Define crawler configurations for different data sources"""
        
        # Define comprehensive crawler configurations for all data sources
        
        crawler_configs = {
            'transactions_crawler': {
                'description': 'Crawl customer transaction data in multiple formats',
                'targets': {
                    'S3Targets': [
                        {
                            'Path': f's3://{S3_DATA_BUCKET}/raw/transactions/',
                            'Exclusions': ['*.tmp', '*.log']
                        }
                    ]
                },
                'schedule': 'cron(0 2 * * ? *)',  # Daily at 2 AM
                'schema_change_policy': {
                    'UpdateBehavior': 'UPDATE_IN_DATABASE',
                    'DeleteBehavior': 'LOG'
                },
                'recrawl_policy': {
                    'RecrawlBehavior': 'CRAWL_EVERYTHING'
                }
            },
            'customers_crawler': {
                'description': 'Crawl customer profile data',
                'targets': {
                    'S3Targets': [
                        {
                            'Path': f's3://{S3_DATA_BUCKET}/raw/customers/',
                            'Exclusions': ['*.backup']
                        }
                    ]
                },
                'schedule': 'cron(0 3 * * ? *)',  # Daily at 3 AM
                'schema_change_policy': {
                    'UpdateBehavior': 'UPDATE_IN_DATABASE',
                    'DeleteBehavior': 'DEPRECATE_IN_DATABASE'
                },
                'recrawl_policy': {
                    'RecrawlBehavior': 'CRAWL_NEW_FOLDERS_ONLY'
                }
            },
            'products_crawler': {
                'description': 'Crawl product catalog data (JSON format)',
                'targets': {
                    'S3Targets': [
                        {
                            'Path': f's3://{S3_DATA_BUCKET}/raw/products/',
                            'Exclusions': []
                        }
                    ]
                },
                'schedule': 'cron(0 4 * * ? *)',  # Daily at 4 AM
                'schema_change_policy': {
                    'UpdateBehavior': 'UPDATE_IN_DATABASE',
                    'DeleteBehavior': 'LOG'
                },
                'recrawl_policy': {
                    'RecrawlBehavior': 'CRAWL_EVERYTHING'
                }
            }
        }
        
        return crawler_configs
    
    def _create_crawler(self, crawler_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create individual crawler with advanced configuration"""
        
        # Create crawler with comprehensive configuration including lineage tracking
        
        crawler_input = {
            'Name': crawler_name,
            'Role': self.iam_role,
            'DatabaseName': self.database_name,
            'Description': config['description'],
            'Targets': config['targets'],
            'Schedule': config['schedule'],
            'SchemaChangePolicy': config['schema_change_policy'],
            'RecrawlPolicy': config['recrawl_policy'],
            'LineageConfiguration': {
                'CrawlerLineageSettings': 'ENABLE'
            },
            'Configuration': json.dumps({
                'Version': 1.0,
                'CrawlerOutput': {
                    'Partitions': {'AddOrUpdateBehavior': 'InheritFromTable'},
                    'Tables': {'AddOrUpdateBehavior': 'MergeNewColumns'}
                },
                'Grouping': {
                    'TableGroupingPolicy': 'CombineCompatibleSchemas'
                }
            })
        }
        
        try:
            response = self.glue_client.create_crawler(**crawler_input)
            print(f"Crawler '{crawler_name}' created successfully")
            return response
            
        except self.glue_client.exceptions.AlreadyExistsException:
            print(f"Crawler '{crawler_name}' already exists")
            return {'status': 'already_exists'}
        except Exception as e:
            print(f"Error creating crawler '{crawler_name}': {str(e)}")
            raise
    
    def _start_crawler(self, crawler_name: str) -> bool:
        """Start crawler execution"""
        
        # Start crawler execution with conflict handling
        
        try:
            self.glue_client.start_crawler(Name=crawler_name)
            print(f"Crawler '{crawler_name}' started successfully")
            return True
            
        except self.glue_client.exceptions.CrawlerRunningException:
            print(f"Crawler '{crawler_name}' is already running")
            return True
        except Exception as e:
            print(f"Error starting crawler '{crawler_name}': {str(e)}")
            return False
    
    def monitor_crawler_execution(self, crawler_name: str) -> Dict[str, Any]:
        """Monitor crawler execution and return detailed status"""
        
        # Monitor crawler execution with detailed status tracking
        
        try:
            response = self.glue_client.get_crawler(Name=crawler_name)
            crawler_info = response['Crawler']
            
            status_info = {
                'crawler_name': crawler_name,
                'state': crawler_info['State'],
                'last_crawl': crawler_info.get('LastCrawl', {}),
                'creation_time': crawler_info['CreationTime'],
                'last_updated': crawler_info['LastUpdated']
            }
            
            if crawler_info.get('LastCrawl'):
                last_crawl = crawler_info['LastCrawl']
                status_info.update({
                    'last_crawl_status': last_crawl.get('Status'),
                    'tables_created': last_crawl.get('TablesCreated', 0),
                    'tables_updated': last_crawl.get('TablesUpdated', 0),
                    'tables_deleted': last_crawl.get('TablesDeleted', 0),
                    'start_time': last_crawl.get('StartTime'),
                    'end_time': last_crawl.get('EndTime')
                })
            
            return status_info
            
        except Exception as e:
            print(f"Error monitoring crawler '{crawler_name}': {str(e)}")
            raise

# =============================================================================
# ETL JOB MANAGEMENT
# =============================================================================

class ETLJobManager:
    """Manage AWS Glue ETL Jobs"""
    
    def __init__(self, database_name: str, iam_role: str, scripts_bucket: str):
        self.database_name = database_name
        self.iam_role = iam_role
        self.scripts_bucket = scripts_bucket
        self.clients = get_aws_clients()
        self.glue_client = self.clients['glue']
    
    def create_etl_jobs(self) -> Dict[str, Any]:
        """Create comprehensive ETL jobs for data transformation"""
        
        # Create multiple ETL jobs for different transformation needs
        
        job_results = {
            'jobs_created': 0,
            'jobs_started': 0,
            'failed_jobs': []
        }
        
        # Define ETL job configurations
        job_configs = self._get_etl_job_configurations()
        
        for job_name, config in job_configs.items():
            try:
                # Create ETL job
                create_response = self._create_etl_job(job_name, config)
                if create_response:
                    job_results['jobs_created'] += 1
                    
                    # Optionally start job for testing
                    if config.get('auto_start', False):
                        start_response = self._start_etl_job(job_name, config.get('arguments', {}))
                        if start_response:
                            job_results['jobs_started'] += 1
                
            except Exception as e:
                print(f"Failed to create ETL job '{job_name}': {str(e)}")
                job_results['failed_jobs'].append({
                    'job_name': job_name,
                    'error': str(e)
                })
        
        print(f"✅ ETL jobs setup completed:")
        print(f"   Jobs created: {job_results['jobs_created']}")
        print(f"   Jobs started: {job_results['jobs_started']}")
        
        return job_results
    
    def _get_etl_job_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Define ETL job configurations"""
        
        # Define comprehensive ETL job configurations with optimization
        
        job_configs = {
            'customer_analytics_etl': {
                'description': 'Advanced customer analytics ETL with segmentation',
                'script_location': f's3://{self.scripts_bucket}/customer_analytics_etl.py',
                'max_capacity': 10,
                'timeout': 2880,  # 48 hours
                'max_retries': 1,
                'worker_type': 'G.1X',
                'number_of_workers': 10,
                'arguments': {
                    '--source_database': self.database_name,
                    '--source_table': 'customer_transactions',
                    '--target_s3_path': f's3://{S3_DATA_BUCKET}/processed/customer_analytics/',
                    '--enable_metrics': 'true'
                },
                'auto_start': False
            },
            'data_quality_etl': {
                'description': 'Data quality validation and cleansing',
                'script_location': f's3://{self.scripts_bucket}/data_quality_etl.py',
                'max_capacity': 5,
                'timeout': 1440,  # 24 hours
                'max_retries': 2,
                'worker_type': 'G.1X',
                'number_of_workers': 5,
                'arguments': {
                    '--source_database': self.database_name,
                    '--quality_rules_s3': f's3://{self.scripts_bucket}/quality_rules.json',
                    '--target_s3_path': f's3://{S3_DATA_BUCKET}/processed/quality_validated/',
                    '--enable_metrics': 'true'
                },
                'auto_start': False
            },
            'product_enrichment_etl': {
                'description': 'Product catalog enrichment with external data',
                'script_location': f's3://{self.scripts_bucket}/product_enrichment_etl.py',
                'max_capacity': 3,
                'timeout': 720,  # 12 hours
                'max_retries': 1,
                'worker_type': 'G.1X',
                'number_of_workers': 3,
                'arguments': {
                    '--source_database': self.database_name,
                    '--source_table': 'products',
                    '--enrichment_api_endpoint': 'https://api.productdata.com/v1/enrich',
                    '--target_s3_path': f's3://{S3_DATA_BUCKET}/processed/enriched_products/',
                    '--enable_metrics': 'true'
                },
                'auto_start': False
            }
        }
        
        return job_configs
    
    def _create_etl_job(self, job_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create individual ETL job with optimization"""
        
        # Create ETL job with comprehensive optimization settings
        
        job_input = {
            'Name': job_name,
            'Description': config['description'],
            'Role': self.iam_role,
            'Command': {
                'Name': 'glueetl',
                'ScriptLocation': config['script_location'],
                'PythonVersion': '3'
            },
            'DefaultArguments': {
                '--job-language': 'python',
                '--job-bookmark-option': 'job-bookmark-enable',
                '--enable-metrics': 'true',
                '--enable-continuous-cloudwatch-log': 'true',
                '--enable-spark-ui': 'true',
                '--spark-event-logs-path': f's3://{self.scripts_bucket}/spark-logs/',
                
                # Performance optimizations
                '--conf': 'spark.sql.adaptive.enabled=true',
                '--conf': 'spark.sql.adaptive.coalescePartitions.enabled=true',
                '--conf': 'spark.sql.adaptive.skewJoin.enabled=true',
                '--conf': 'spark.serializer=org.apache.spark.serializer.KryoSerializer',
                
                # Memory optimization
                '--conf': 'spark.sql.execution.arrow.pyspark.enabled=true',
                '--conf': 'spark.sql.execution.arrow.maxRecordsPerBatch=10000',
                
                # S3 optimization
                '--conf': 'spark.hadoop.fs.s3a.multipart.size=104857600',
                '--conf': 'spark.hadoop.fs.s3a.multipart.threshold=104857600',
                '--conf': 'spark.hadoop.fs.s3a.fast.upload=true',
                
                **config['arguments']
            },
            'MaxRetries': config['max_retries'],
            'Timeout': config['timeout'],
            'MaxCapacity': config['max_capacity'],
            'GlueVersion': '3.0',
            'NumberOfWorkers': config['number_of_workers'],
            'WorkerType': config['worker_type']
        }
        
        try:
            response = self.glue_client.create_job(**job_input)
            print(f"ETL job '{job_name}' created successfully")
            return response
            
        except Exception as e:
            print(f"Error creating ETL job '{job_name}': {str(e)}")
            raise
    
    def _start_etl_job(self, job_name: str, arguments: Dict[str, str]) -> Dict[str, Any]:
        """Start ETL job execution"""
        
        # Start ETL job execution with proper argument handling
        
        try:
            response = self.glue_client.start_job_run(
                JobName=job_name,
                Arguments=arguments
            )
            
            job_run_id = response['JobRunId']
            print(f"ETL job '{job_name}' started with run ID: {job_run_id}")
            return response
            
        except Exception as e:
            print(f"Error starting ETL job '{job_name}': {str(e)}")
            raise

# =============================================================================
# ATHENA INTEGRATION
# =============================================================================

class AthenaManager:
    """Manage Athena queries and optimization"""
    
    def __init__(self, database_name: str, results_location: str):
        self.database_name = database_name
        self.results_location = results_location
        self.clients = get_aws_clients()
        self.athena_client = self.clients['athena']
    
    def setup_athena_analytics(self) -> Dict[str, Any]:
        """Set up Athena for business analyst access"""
        
        # Set up Athena workgroups, saved queries, and optimization
        
        setup_results = {
            'workgroup_created': False,
            'saved_queries_created': 0,
            'views_created': 0
        }
        
        try:
            # Create workgroup for cost control and optimization
            workgroup_response = self._create_athena_workgroup()
            setup_results['workgroup_created'] = workgroup_response
            
            # Create commonly used views
            views_response = self._create_analytical_views()
            setup_results['views_created'] = views_response
            
            # Create saved queries for business analysts
            queries_response = self._create_saved_queries()
            setup_results['saved_queries_created'] = queries_response
            
            print(f"✅ Athena analytics setup completed:")
            print(f"   Workgroup created: {setup_results['workgroup_created']}")
            print(f"   Views created: {setup_results['views_created']}")
            print(f"   Saved queries: {setup_results['saved_queries_created']}")
            
            return setup_results
            
        except Exception as e:
            print(f"❌ Error setting up Athena analytics: {str(e)}")
            raise
    
    def _create_athena_workgroup(self) -> bool:
        """Create Athena workgroup with cost controls"""
        
        # Create workgroup with cost controls and optimization (simulated for LocalStack)
        
        workgroup_name = 'serverlessdata-analytics'
        
        try:
            # Note: In a real implementation, you would create the workgroup
            # This is simulated for the exercise
            print(f"Athena workgroup '{workgroup_name}' created (simulated)")
            return True
            
        except Exception as e:
            print(f"Error creating Athena workgroup: {str(e)}")
            return False
    
    def _create_analytical_views(self) -> int:
        """Create analytical views for common business queries"""
        
        # Create analytical views for common business queries
        
        views_created = 0
        
        view_definitions = {
            'customer_summary_view': """
                CREATE OR REPLACE VIEW customer_summary_view AS
                SELECT 
                    c.customer_id,
                    c.first_name,
                    c.last_name,
                    c.customer_segment,
                    COUNT(t.transaction_id) as total_transactions,
                    SUM(t.transaction_amount) as total_spent,
                    AVG(t.transaction_amount) as avg_transaction_amount,
                    MIN(t.transaction_date) as first_transaction_date,
                    MAX(t.transaction_date) as last_transaction_date
                FROM customer_profiles c
                LEFT JOIN customer_transactions t ON c.customer_id = t.customer_id
                GROUP BY c.customer_id, c.first_name, c.last_name, c.customer_segment
            """,
            'monthly_revenue_view': """
                CREATE OR REPLACE VIEW monthly_revenue_view AS
                SELECT 
                    YEAR(transaction_date) as year,
                    MONTH(transaction_date) as month,
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(transaction_id) as total_transactions,
                    SUM(transaction_amount) as total_revenue,
                    AVG(transaction_amount) as avg_transaction_amount
                FROM customer_transactions
                GROUP BY YEAR(transaction_date), MONTH(transaction_date)
                ORDER BY year DESC, month DESC
            """
        }
        
        for view_name, view_sql in view_definitions.items():
            try:
                # In a real implementation, you would execute these CREATE VIEW statements
                print(f"View '{view_name}' created (simulated)")
                views_created += 1
            except Exception as e:
                print(f"Error creating view '{view_name}': {str(e)}")
        
        return views_created
    
    def _create_saved_queries(self) -> int:
        """Create saved queries for business analysts"""
        
        # Create commonly used analytical queries for business analysts
        
        saved_queries = {
            'top_customers_by_revenue': """
                SELECT 
                    customer_id,
                    first_name,
                    last_name,
                    total_spent,
                    total_transactions
                FROM customer_summary_view
                ORDER BY total_spent DESC
                LIMIT 100
            """,
            'monthly_growth_analysis': """
                SELECT 
                    year,
                    month,
                    total_revenue,
                    LAG(total_revenue) OVER (ORDER BY year, month) as prev_month_revenue,
                    (total_revenue - LAG(total_revenue) OVER (ORDER BY year, month)) / 
                    LAG(total_revenue) OVER (ORDER BY year, month) * 100 as growth_rate
                FROM monthly_revenue_view
                ORDER BY year DESC, month DESC
            """,
            'customer_segmentation_analysis': """
                SELECT 
                    customer_segment,
                    COUNT(*) as customer_count,
                    AVG(total_spent) as avg_customer_value,
                    SUM(total_spent) as segment_revenue,
                    AVG(total_transactions) as avg_transactions_per_customer
                FROM customer_summary_view
                GROUP BY customer_segment
                ORDER BY segment_revenue DESC
            """
        }
        
        queries_created = 0
        for query_name, query_sql in saved_queries.items():
            try:
                # In a real implementation, you would save these queries in Athena
                print(f"Saved query '{query_name}' created (simulated)")
                queries_created += 1
            except Exception as e:
                print(f"Error creating saved query '{query_name}': {str(e)}")
        
        return queries_created

# =============================================================================
# MONITORING AND OPTIMIZATION
# =============================================================================

class MonitoringManager:
    """Monitor and optimize Glue jobs and Athena queries"""
    
    def __init__(self):
        self.clients = get_aws_clients()
        self.glue_client = self.clients['glue']
        self.cloudwatch_client = self.clients['cloudwatch']
    
    def setup_monitoring_dashboard(self) -> Dict[str, Any]:
        """Set up comprehensive monitoring for the serverless pipeline"""
        
        # Set up comprehensive monitoring dashboards and alarms
        
        monitoring_setup = {
            'dashboards_created': 0,
            'alarms_created': 0,
            'metrics_enabled': False
        }
        
        try:
            # Create CloudWatch dashboard
            dashboard_response = self._create_cloudwatch_dashboard()
            monitoring_setup['dashboards_created'] = dashboard_response
            
            # Create alarms for job failures and cost overruns
            alarms_response = self._create_monitoring_alarms()
            monitoring_setup['alarms_created'] = alarms_response
            
            # Enable detailed metrics
            metrics_response = self._enable_detailed_metrics()
            monitoring_setup['metrics_enabled'] = metrics_response
            
            print(f"✅ Monitoring setup completed:")
            print(f"   Dashboards created: {monitoring_setup['dashboards_created']}")
            print(f"   Alarms created: {monitoring_setup['alarms_created']}")
            print(f"   Detailed metrics enabled: {monitoring_setup['metrics_enabled']}")
            
            return monitoring_setup
            
        except Exception as e:
            print(f"❌ Error setting up monitoring: {str(e)}")
            raise
    
    def _create_cloudwatch_dashboard(self) -> int:
        """Create CloudWatch dashboard for pipeline monitoring"""
        
        # Create CloudWatch dashboard with key pipeline metrics (simulated)
        
        try:
            # In a real implementation, you would create CloudWatch dashboards
            print("CloudWatch dashboard created (simulated)")
            return 1
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            return 0
    
    def _create_monitoring_alarms(self) -> int:
        """Create CloudWatch alarms for critical events"""
        
        # Create CloudWatch alarms for critical pipeline events
        
        alarms_created = 0
        
        alarm_configs = [
            {
                'name': 'GlueJobFailures',
                'description': 'Alert on Glue job failures',
                'metric': 'glue.driver.aggregate.numFailedTasks'
            },
            {
                'name': 'HighAthenaCosts',
                'description': 'Alert on high Athena query costs',
                'metric': 'athena.DataScannedInBytes'
            },
            {
                'name': 'CrawlerFailures',
                'description': 'Alert on crawler failures',
                'metric': 'glue.driver.aggregate.numFailedTasks'
            }
        ]
        
        for alarm_config in alarm_configs:
            try:
                # In a real implementation, you would create CloudWatch alarms
                print(f"Alarm '{alarm_config['name']}' created (simulated)")
                alarms_created += 1
            except Exception as e:
                print(f"Error creating alarm '{alarm_config['name']}': {str(e)}")
        
        return alarms_created
    
    def _enable_detailed_metrics(self) -> bool:
        """Enable detailed metrics collection"""
        
        # Enable detailed metrics collection for monitoring (simulated)
        
        try:
            # In a real implementation, you would configure detailed metrics
            print("Detailed metrics enabled (simulated)")
            return True
            
        except Exception as e:
            print(f"Error enabling metrics: {str(e)}")
            return False

# =============================================================================
# MAIN EXERCISE EXECUTION
# =============================================================================

def main_exercise():
    """Execute the complete serverless ETL pipeline setup"""
    
    print("🚀 ServerlessData Corp - AWS Glue & Data Catalog Implementation")
    print("=" * 80)
    
    try:
        # Step 1: Set up Data Catalog
        print("\n📊 Step 1: Setting up Data Catalog...")
        catalog_manager = DataCatalogManager(DATABASE_NAME)
        catalog_results = catalog_manager.setup_data_catalog()
        
        # Step 2: Create and configure crawlers
        print("\n🕷️ Step 2: Creating and starting crawlers...")
        crawler_manager = CrawlerManager(DATABASE_NAME, IAM_GLUE_ROLE)
        crawler_results = crawler_manager.create_comprehensive_crawlers()
        
        # Step 3: Create ETL jobs
        print("\n⚙️ Step 3: Creating ETL jobs...")
        etl_manager = ETLJobManager(DATABASE_NAME, IAM_GLUE_ROLE, S3_SCRIPTS_BUCKET)
        etl_results = etl_manager.create_etl_jobs()
        
        # Step 4: Set up Athena analytics
        print("\n📈 Step 4: Setting up Athena analytics...")
        athena_manager = AthenaManager(DATABASE_NAME, S3_ATHENA_RESULTS)
        athena_results = athena_manager.setup_athena_analytics()
        
        # Step 5: Set up monitoring and optimization
        print("\n📊 Step 5: Setting up monitoring...")
        monitoring_manager = MonitoringManager()
        monitoring_results = monitoring_manager.setup_monitoring_dashboard()
        
        # Summary
        print("\n" + "="*80)
        print("✅ SERVERLESS ETL PIPELINE SETUP COMPLETE!")
        print("="*80)
        
        print(f"📊 Data Catalog: {catalog_results['tables_created']} tables created")
        print(f"🕷️ Crawlers: {crawler_results['crawlers_created']} created, {crawler_results['crawlers_started']} started")
        print(f"⚙️ ETL Jobs: {etl_results['jobs_created']} created")
        print(f"📈 Athena: {athena_results['views_created']} views, {athena_results['saved_queries_created']} saved queries")
        print(f"📊 Monitoring: {monitoring_results['dashboards_created']} dashboards, {monitoring_results['alarms_created']} alarms")
        
        print("\n🎯 Next Steps:")
        print("1. Upload sample data to S3 buckets")
        print("2. Monitor crawler execution and schema discovery")
        print("3. Test ETL jobs with sample data")
        print("4. Validate Athena queries and performance")
        print("5. Review cost optimization opportunities")
        
        return {
            'status': 'success',
            'catalog_results': catalog_results,
            'crawler_results': crawler_results,
            'etl_results': etl_results,
            'athena_results': athena_results,
            'monitoring_results': monitoring_results
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline setup failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    # Execute the main exercise
    results = main_exercise()
    
    print("\n" + "="*80)
    print("🎓 Exercise completed! Review the implementation and explore optimization opportunities.")
    print("="*80)
