"""
Day 24: Project - Production Pipeline with Quality & Monitoring - Complete Solution

Comprehensive production data platform integrating all Phase 2 technologies:
- Airflow orchestration with advanced patterns
- dbt transformations with comprehensive testing
- Great Expectations data quality validation
- AWS Glue serverless ETL integration
- Kinesis real-time streaming processing
- Complete observability and monitoring
- Comprehensive testing strategies

This solution demonstrates enterprise-grade data engineering practices.
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import great_expectations as ge
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psycopg2
from sqlalchemy import create_engine
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PRODUCTION DATA PLATFORM ORCHESTRATOR
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for production data pipeline"""
    
    # Database connections
    postgres_conn: str = os.getenv('POSTGRES_CONN', 'postgresql://airflow:airflow@postgres:5432/dataflow_warehouse')
    redis_conn: str = os.getenv('REDIS_CONN', 'redis://redis:6379/0')
    
    # AWS configuration
    aws_region: str = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    aws_endpoint_url: str = os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566')
    s3_bucket: str = os.getenv('S3_BUCKET', 'dataflow-production-data')
    glue_job_name: str = os.getenv('GLUE_JOB_NAME', 'dataflow_advanced_etl')
    kinesis_stream: str = os.getenv('KINESIS_STREAM', 'dataflow_events')
    
    # Processing configuration
    batch_size: int = 10000
    max_workers: int = 5
    quality_threshold: float = 0.95
    
    # Monitoring configuration
    metrics_port: int = 8000
    alert_threshold: float = 0.1

class ProductionDataPlatform:
    """Enterprise-grade data platform orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize connections
        self.db_engine = create_engine(config.postgres_conn)
        self.redis_client = redis.from_url(config.redis_conn)
        
        # Initialize AWS clients
        self.s3_client = boto3.client(
            's3', 
            region_name=config.aws_region,
            endpoint_url=config.aws_endpoint_url,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'test'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'test')
        )
        self.glue_client = boto3.client(
            'glue', 
            region_name=config.aws_region,
            endpoint_url=config.aws_endpoint_url,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'test'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'test')
        )
        self.kinesis_client = boto3.client(
            'kinesis', 
            region_name=config.aws_region,
            endpoint_url=config.aws_endpoint_url,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID', 'test'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', 'test')
        )
        
        # Initialize monitoring
        self._setup_metrics()
        
        # Initialize components
        self.data_ingestion = DataIngestionManager(self)
        self.quality_validator = ProductionQualityValidator(self)
        self.transformation_engine = AdvancedTransformationEngine(self)
        self.aws_integration = AWSIntegrationManager(self)
        self.monitoring = ComprehensiveMonitoring(self)
        
        self.logger.info("Production Data Platform initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for production"""
        
        logger = logging.getLogger('production_data_platform')
        logger.setLevel(logging.INFO)
        
        # Create formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "%(name)s", "message": "%(message)s", '
            '"correlation_id": "%(correlation_id)s"}'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        
        self.metrics = {
            'pipeline_runs': Counter('pipeline_runs_total', 'Total pipeline runs', ['status', 'pipeline']),
            'processing_time': Histogram('processing_time_seconds', 'Time spent processing', ['stage']),
            'data_quality_score': Gauge('data_quality_score', 'Current data quality score', ['dataset']),
            'records_processed': Counter('records_processed_total', 'Total records processed', ['source']),
            'errors_total': Counter('errors_total', 'Total errors encountered', ['component', 'error_type'])
        }
        
        # Start metrics server
        start_http_server(self.config.metrics_port)
        self.logger.info(f"Metrics server started on port {self.config.metrics_port}")
    
    def execute_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete production data pipeline"""
        
        correlation_id = f"pipeline_{int(time.time())}"
        start_time = time.time()
        
        try:
            self.logger.info("Starting complete pipeline execution", extra={'correlation_id': correlation_id})
            
            # Phase 1: Data Ingestion
            with self.metrics['processing_time'].labels(stage='ingestion').time():
                ingestion_results = self.data_ingestion.ingest_all_sources()
            
            # Phase 2: Data Quality Validation
            with self.metrics['processing_time'].labels(stage='quality_validation').time():
                quality_results = self.quality_validator.validate_all_datasets()
            
            # Check quality gate
            if not self._quality_gate_passed(quality_results):
                raise Exception("Data quality validation failed - pipeline stopped")
            
            # Phase 3: Data Transformations
            with self.metrics['processing_time'].labels(stage='transformations').time():
                transformation_results = self.transformation_engine.execute_transformations()
            
            # Phase 4: AWS Integration
            with self.metrics['processing_time'].labels(stage='aws_integration').time():
                aws_results = self.aws_integration.execute_aws_processing()
            
            # Phase 5: Monitoring and Alerting
            monitoring_results = self.monitoring.update_pipeline_metrics()
            
            # Record success
            self.metrics['pipeline_runs'].labels(status='success', pipeline='complete').inc()
            
            execution_time = time.time() - start_time
            
            results = {
                'status': 'success',
                'correlation_id': correlation_id,
                'execution_time': execution_time,
                'ingestion': ingestion_results,
                'quality': quality_results,
                'transformations': transformation_results,
                'aws_integration': aws_results,
                'monitoring': monitoring_results
            }
            
            self.logger.info(
                f"Pipeline execution completed successfully in {execution_time:.2f}s",
                extra={'correlation_id': correlation_id}
            )
            
            return results
            
        except Exception as e:
            # Record failure
            self.metrics['pipeline_runs'].labels(status='failure', pipeline='complete').inc()
            self.metrics['errors_total'].labels(component='pipeline', error_type='execution_failure').inc()
            
            self.logger.error(
                f"Pipeline execution failed: {str(e)}",
                extra={'correlation_id': correlation_id}
            )
            
            # Send alert
            self.monitoring.send_pipeline_alert('pipeline_failure', str(e), correlation_id)
            
            raise
    
    def _quality_gate_passed(self, quality_results: Dict[str, Any]) -> bool:
        """Check if data quality gate criteria are met"""
        
        overall_score = quality_results.get('overall_quality_score', 0)
        return overall_score >= self.config.quality_threshold

# =============================================================================
# DATA INGESTION MANAGER
# =============================================================================

class DataIngestionManager:
    """Advanced data ingestion with multiple source support"""
    
    def __init__(self, platform: ProductionDataPlatform):
        self.platform = platform
        self.logger = platform.logger
        
    def ingest_all_sources(self) -> Dict[str, Any]:
        """Ingest data from all configured sources"""
        
        sources = [
            {'name': 'customers', 'type': 'database', 'table': 'raw_customers'},
            {'name': 'transactions', 'type': 'api', 'endpoint': '/api/transactions'},
            {'name': 'products', 'type': 'file', 'path': '/data/products.csv'},
            {'name': 'events', 'type': 'stream', 'stream': 'customer_events'}
        ]
        
        ingestion_results = {
            'sources_processed': 0,
            'total_records': 0,
            'failed_sources': [],
            'processing_times': {}
        }
        
        # Process sources concurrently
        with ThreadPoolExecutor(max_workers=self.platform.config.max_workers) as executor:
            future_to_source = {
                executor.submit(self._ingest_source, source): source 
                for source in sources
            }
            
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                source_name = source['name']
                
                try:
                    start_time = time.time()
                    result = future.result()
                    processing_time = time.time() - start_time
                    
                    ingestion_results['sources_processed'] += 1
                    ingestion_results['total_records'] += result['record_count']
                    ingestion_results['processing_times'][source_name] = processing_time
                    
                    # Update metrics
                    self.platform.metrics['records_processed'].labels(source=source_name).inc(result['record_count'])
                    
                    self.logger.info(f"Successfully ingested {result['record_count']} records from {source_name}")
                    
                except Exception as e:
                    ingestion_results['failed_sources'].append({
                        'source': source_name,
                        'error': str(e)
                    })
                    
                    self.platform.metrics['errors_total'].labels(
                        component='ingestion', 
                        error_type='source_failure'
                    ).inc()
                    
                    self.logger.error(f"Failed to ingest from {source_name}: {str(e)}")
        
        return ingestion_results
    
    def _ingest_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data from a specific source"""
        
        source_type = source['type']
        source_name = source['name']
        
        if source_type == 'database':
            return self._ingest_from_database(source)
        elif source_type == 'api':
            return self._ingest_from_api(source)
        elif source_type == 'file':
            return self._ingest_from_file(source)
        elif source_type == 'stream':
            return self._ingest_from_stream(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _ingest_from_database(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data from database source"""
        
        table_name = source['table']
        
        # Get incremental processing checkpoint
        checkpoint = self._get_checkpoint(source['name'])
        
        # Build incremental query
        if checkpoint:
            query = f"""
            SELECT * FROM {table_name} 
            WHERE updated_at > '{checkpoint}'
            ORDER BY updated_at
            """
        else:
            query = f"SELECT * FROM {table_name} ORDER BY updated_at"
        
        # Execute query in chunks
        chunk_size = self.platform.config.batch_size
        total_records = 0
        
        for chunk_df in pd.read_sql(query, self.platform.db_engine, chunksize=chunk_size):
            # Process chunk
            processed_chunk = self._process_data_chunk(chunk_df, source['name'])
            
            # Store processed data
            self._store_processed_data(processed_chunk, f"processed_{source['name']}")
            
            total_records += len(processed_chunk)
            
            # Update checkpoint
            if not chunk_df.empty and 'updated_at' in chunk_df.columns:
                latest_timestamp = chunk_df['updated_at'].max()
                self._update_checkpoint(source['name'], latest_timestamp)
        
        return {'record_count': total_records, 'source_type': 'database'}
    
    def _ingest_from_api(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data from API source"""
        
        import requests
        
        endpoint = source['endpoint']
        base_url = "https://api.dataflow.com"  # Configuration
        
        # Get data from API with pagination
        page = 1
        total_records = 0
        
        while True:
            response = requests.get(
                f"{base_url}{endpoint}",
                params={'page': page, 'limit': self.platform.config.batch_size},
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            if not data.get('records'):
                break
            
            # Convert to DataFrame and process
            df = pd.DataFrame(data['records'])
            processed_df = self._process_data_chunk(df, source['name'])
            
            # Store processed data
            self._store_processed_data(processed_df, f"processed_{source['name']}")
            
            total_records += len(processed_df)
            page += 1
            
            # Check if we've reached the end
            if len(data['records']) < self.platform.config.batch_size:
                break
        
        return {'record_count': total_records, 'source_type': 'api'}
    
    def _ingest_from_file(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data from file source"""
        
        file_path = source['path']
        
        # Read file in chunks
        chunk_size = self.platform.config.batch_size
        total_records = 0
        
        for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
            # Process chunk
            processed_chunk = self._process_data_chunk(chunk_df, source['name'])
            
            # Store processed data
            self._store_processed_data(processed_chunk, f"processed_{source['name']}")
            
            total_records += len(processed_chunk)
        
        return {'record_count': total_records, 'source_type': 'file'}
    
    def _ingest_from_stream(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest data from streaming source"""
        
        stream_name = source['stream']
        
        # Get stream records (simplified - in production would use Kinesis consumer)
        try:
            response = self.platform.kinesis_client.describe_stream(StreamName=stream_name)
            shards = response['StreamDescription']['Shards']
            
            total_records = 0
            
            for shard in shards:
                shard_iterator_response = self.platform.kinesis_client.get_shard_iterator(
                    StreamName=stream_name,
                    ShardId=shard['ShardId'],
                    ShardIteratorType='TRIM_HORIZON'
                )
                
                shard_iterator = shard_iterator_response['ShardIterator']
                
                # Get records from shard
                records_response = self.platform.kinesis_client.get_records(
                    ShardIterator=shard_iterator,
                    Limit=1000
                )
                
                if records_response['Records']:
                    # Process stream records
                    stream_data = []
                    for record in records_response['Records']:
                        data = json.loads(record['Data'])
                        stream_data.append(data)
                    
                    if stream_data:
                        df = pd.DataFrame(stream_data)
                        processed_df = self._process_data_chunk(df, source['name'])
                        self._store_processed_data(processed_df, f"processed_{source['name']}")
                        total_records += len(processed_df)
            
            return {'record_count': total_records, 'source_type': 'stream'}
            
        except Exception as e:
            self.logger.warning(f"Stream {stream_name} not available: {str(e)}")
            return {'record_count': 0, 'source_type': 'stream'}
    
    def _process_data_chunk(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Process data chunk with standardization and enrichment"""
        
        # Add processing metadata
        df['ingestion_timestamp'] = datetime.now()
        df['source_name'] = source_name
        df['processing_version'] = '1.0'
        
        # Data standardization
        if 'email' in df.columns:
            df['email'] = df['email'].str.lower().str.strip()
        
        if 'phone' in df.columns:
            df['phone'] = df['phone'].str.replace(r'[^\d]', '', regex=True)
        
        # Data validation
        df['record_quality_score'] = self._calculate_record_quality(df)
        
        return df
    
    def _calculate_record_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate quality score for each record"""
        
        quality_scores = pd.Series(1.0, index=df.index)
        
        # Check for null values in important columns
        important_columns = ['id', 'customer_id', 'transaction_id']
        for col in important_columns:
            if col in df.columns:
                quality_scores -= df[col].isnull() * 0.2
        
        # Check email format if present
        if 'email' in df.columns:
            email_valid = df['email'].str.match(r'^[^@]+@[^@]+\.[^@]+$', na=False)
            quality_scores -= (~email_valid) * 0.1
        
        return quality_scores.clip(0, 1)
    
    def _store_processed_data(self, df: pd.DataFrame, table_name: str):
        """Store processed data to database"""
        
        df.to_sql(
            table_name,
            self.platform.db_engine,
            if_exists='append',
            index=False,
            method='multi'
        )
    
    def _get_checkpoint(self, source_name: str) -> Optional[str]:
        """Get processing checkpoint for incremental loading"""
        
        try:
            checkpoint = self.platform.redis_client.get(f"checkpoint:{source_name}")
            return checkpoint.decode('utf-8') if checkpoint else None
        except:
            return None
    
    def _update_checkpoint(self, source_name: str, timestamp: str):
        """Update processing checkpoint"""
        
        self.platform.redis_client.set(f"checkpoint:{source_name}", str(timestamp))

# =============================================================================
# PRODUCTION QUALITY VALIDATOR
# =============================================================================

class ProductionQualityValidator:
    """Comprehensive data quality validation using Great Expectations"""
    
    def __init__(self, platform: ProductionDataPlatform):
        self.platform = platform
        self.logger = platform.logger
        
        # Initialize Great Expectations context
        self.ge_context = self._setup_great_expectations()
    
    def _setup_great_expectations(self) -> ge.DataContext:
        """Setup Great Expectations context for production"""
        
        # Create context configuration
        context_config = {
            "config_version": 3.0,
            "datasources": {
                "production_datasource": {
                    "class_name": "Datasource",
                    "execution_engine": {
                        "class_name": "SqlAlchemyExecutionEngine",
                        "connection_string": self.platform.config.postgres_conn
                    },
                    "data_connectors": {
                        "default_runtime_data_connector_name": {
                            "class_name": "RuntimeDataConnector",
                            "batch_identifiers": ["default_identifier_name"]
                        }
                    }
                }
            },
            "stores": {
                "expectations_store": {
                    "class_name": "ExpectationsStore",
                    "store_backend": {
                        "class_name": "DatabaseStoreBackend",
                        "connection_string": self.platform.config.postgres_conn
                    }
                },
                "validations_store": {
                    "class_name": "ValidationsStore",
                    "store_backend": {
                        "class_name": "DatabaseStoreBackend",
                        "connection_string": self.platform.config.postgres_conn
                    }
                }
            }
        }
        
        # Create context
        context = ge.get_context(project_config=context_config)
        
        return context
    
    def validate_all_datasets(self) -> Dict[str, Any]:
        """Validate all datasets with comprehensive quality checks"""
        
        datasets = [
            {'name': 'processed_customers', 'suite': 'customers_quality_suite'},
            {'name': 'processed_transactions', 'suite': 'transactions_quality_suite'},
            {'name': 'processed_products', 'suite': 'products_quality_suite'}
        ]
        
        validation_results = {
            'datasets_validated': 0,
            'overall_quality_score': 0.0,
            'failed_validations': [],
            'quality_scores': {}
        }
        
        total_score = 0.0
        
        for dataset in datasets:
            try:
                result = self._validate_dataset(dataset['name'], dataset['suite'])
                
                validation_results['datasets_validated'] += 1
                validation_results['quality_scores'][dataset['name']] = result['quality_score']
                
                total_score += result['quality_score']
                
                # Update metrics
                self.platform.metrics['data_quality_score'].labels(dataset=dataset['name']).set(result['quality_score'])
                
                if not result['success']:
                    validation_results['failed_validations'].append({
                        'dataset': dataset['name'],
                        'failed_expectations': result['failed_expectations']
                    })
                
                self.logger.info(f"Dataset {dataset['name']} quality score: {result['quality_score']:.3f}")
                
            except Exception as e:
                validation_results['failed_validations'].append({
                    'dataset': dataset['name'],
                    'error': str(e)
                })
                
                self.platform.metrics['errors_total'].labels(
                    component='quality_validation',
                    error_type='validation_failure'
                ).inc()
                
                self.logger.error(f"Quality validation failed for {dataset['name']}: {str(e)}")
        
        # Calculate overall quality score
        if validation_results['datasets_validated'] > 0:
            validation_results['overall_quality_score'] = total_score / validation_results['datasets_validated']
        
        return validation_results
    
    def _validate_dataset(self, dataset_name: str, suite_name: str) -> Dict[str, Any]:
        """Validate specific dataset with expectation suite"""
        
        # Create batch request
        batch_request = {
            "datasource_name": "production_datasource",
            "data_connector_name": "default_runtime_data_connector_name",
            "data_asset_name": dataset_name,
            "runtime_parameters": {"query": f"SELECT * FROM {dataset_name}"},
            "batch_identifiers": {"default_identifier_name": "default_identifier"}
        }
        
        # Get or create expectation suite
        try:
            suite = self.ge_context.get_expectation_suite(suite_name)
        except:
            suite = self._create_expectation_suite(suite_name, dataset_name)
        
        # Create validator
        validator = self.ge_context.get_validator(
            batch_request=batch_request,
            expectation_suite=suite
        )
        
        # Run validation
        validation_result = validator.validate()
        
        # Calculate quality score
        total_expectations = len(validation_result.results)
        successful_expectations = sum(1 for result in validation_result.results if result.success)
        
        quality_score = successful_expectations / total_expectations if total_expectations > 0 else 0.0
        
        # Get failed expectations
        failed_expectations = [
            {
                'expectation_type': result.expectation_config.expectation_type,
                'kwargs': result.expectation_config.kwargs,
                'result': result.result
            }
            for result in validation_result.results if not result.success
        ]
        
        return {
            'success': validation_result.success,
            'quality_score': quality_score,
            'total_expectations': total_expectations,
            'successful_expectations': successful_expectations,
            'failed_expectations': failed_expectations
        }
    
    def _create_expectation_suite(self, suite_name: str, dataset_name: str) -> ge.ExpectationSuite:
        """Create comprehensive expectation suite for dataset"""
        
        suite = self.ge_context.create_expectation_suite(suite_name)
        
        # Common expectations for all datasets
        common_expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 100000000}
            }
        ]
        
        # Dataset-specific expectations
        if 'customers' in dataset_name:
            dataset_expectations = [
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "customer_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "customer_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": "customer_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_match_regex",
                    "kwargs": {
                        "column": "email",
                        "regex": r"^[^@]+@[^@]+\.[^@]+$"
                    }
                }
            ]
        elif 'transactions' in dataset_name:
            dataset_expectations = [
                {
                    "expectation_type": "expect_column_to_exist",
                    "kwargs": {"column": "transaction_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": "transaction_id"}
                },
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": "transaction_amount",
                        "min_value": 0,
                        "max_value": 1000000
                    }
                }
            ]
        else:
            dataset_expectations = []
        
        # Add all expectations to suite
        all_expectations = common_expectations + dataset_expectations
        
        for expectation in all_expectations:
            suite.add_expectation(**expectation)
        
        # Save suite
        self.ge_context.save_expectation_suite(suite)
        
        return suite
# =============================================================================
# ADVANCED TRANSFORMATION ENGINE
# =============================================================================

class AdvancedTransformationEngine:
    """Advanced data transformations with dbt integration"""
    
    def __init__(self, platform: ProductionDataPlatform):
        self.platform = platform
        self.logger = platform.logger
    
    def execute_transformations(self) -> Dict[str, Any]:
        """Execute comprehensive data transformations"""
        
        transformation_results = {
            'models_executed': 0,
            'tests_passed': 0,
            'failed_models': [],
            'execution_times': {}
        }
        
        # Define transformation stages
        stages = [
            {'name': 'staging', 'models': ['stg_customers', 'stg_transactions', 'stg_products']},
            {'name': 'intermediate', 'models': ['int_customer_metrics', 'int_transaction_analysis']},
            {'name': 'marts', 'models': ['dim_customers', 'fct_transactions', 'customer_360']}
        ]
        
        for stage in stages:
            stage_start_time = time.time()
            
            try:
                # Execute models in stage
                for model in stage['models']:
                    model_result = self._execute_transformation_model(model)
                    
                    if model_result['success']:
                        transformation_results['models_executed'] += 1
                    else:
                        transformation_results['failed_models'].append({
                            'model': model,
                            'error': model_result['error']
                        })
                
                # Run tests for stage
                test_results = self._run_transformation_tests(stage['name'])
                transformation_results['tests_passed'] += test_results['passed_tests']
                
                stage_time = time.time() - stage_start_time
                transformation_results['execution_times'][stage['name']] = stage_time
                
                self.logger.info(f"Stage '{stage['name']}' completed in {stage_time:.2f}s")
                
            except Exception as e:
                transformation_results['failed_models'].append({
                    'stage': stage['name'],
                    'error': str(e)
                })
                
                self.platform.metrics['errors_total'].labels(
                    component='transformations',
                    error_type='stage_failure'
                ).inc()
                
                self.logger.error(f"Stage '{stage['name']}' failed: {str(e)}")
        
        return transformation_results
    
    def _execute_transformation_model(self, model_name: str) -> Dict[str, Any]:
        """Execute individual transformation model"""
        
        try:
            # Get model SQL
            model_sql = self._get_model_sql(model_name)
            
            # Execute transformation
            start_time = time.time()
            
            with self.platform.db_engine.connect() as conn:
                # Drop and recreate table
                conn.execute(f"DROP TABLE IF EXISTS {model_name}")
                
                # Create table with transformation
                create_sql = f"CREATE TABLE {model_name} AS ({model_sql})"
                result = conn.execute(create_sql)
                
                # Get row count
                count_result = conn.execute(f"SELECT COUNT(*) FROM {model_name}")
                row_count = count_result.fetchone()[0]
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Model '{model_name}' executed successfully: {row_count} rows in {execution_time:.2f}s")
            
            return {
                'success': True,
                'row_count': row_count,
                'execution_time': execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Model '{model_name}' execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_model_sql(self, model_name: str) -> str:
        """Get SQL for transformation model"""
        
        # In production, this would read from dbt model files
        # Here we provide sample SQL for each model
        
        model_sql_map = {
            'stg_customers': """
                SELECT 
                    customer_id,
                    LOWER(TRIM(email)) as email,
                    TRIM(first_name) as first_name,
                    TRIM(last_name) as last_name,
                    registration_date::date as registration_date,
                    CASE 
                        WHEN customer_segment IS NULL OR customer_segment = '' THEN 'unknown'
                        ELSE LOWER(TRIM(customer_segment))
                    END as customer_segment,
                    CASE 
                        WHEN email ~ '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$' THEN true
                        ELSE false
                    END as email_is_valid,
                    ingestion_timestamp,
                    record_quality_score,
                    CURRENT_TIMESTAMP as dbt_updated_at
                FROM processed_customers
                WHERE customer_id IS NOT NULL
            """,
            
            'stg_transactions': """
                SELECT 
                    transaction_id,
                    customer_id,
                    product_id,
                    transaction_amount::decimal(12,2) as transaction_amount,
                    transaction_date::timestamp as transaction_date,
                    EXTRACT(YEAR FROM transaction_date::timestamp) as transaction_year,
                    EXTRACT(MONTH FROM transaction_date::timestamp) as transaction_month,
                    EXTRACT(DOW FROM transaction_date::timestamp) as day_of_week,
                    CASE 
                        WHEN transaction_amount < 50 THEN 'low'
                        WHEN transaction_amount < 200 THEN 'medium'
                        ELSE 'high'
                    END as amount_category,
                    ingestion_timestamp,
                    record_quality_score,
                    CURRENT_TIMESTAMP as dbt_updated_at
                FROM processed_transactions
                WHERE transaction_id IS NOT NULL 
                  AND customer_id IS NOT NULL
                  AND transaction_amount > 0
            """,
            
            'stg_products': """
                SELECT 
                    product_id,
                    TRIM(product_name) as product_name,
                    TRIM(product_category) as product_category,
                    price::decimal(10,2) as price,
                    CASE 
                        WHEN price < 50 THEN 'budget'
                        WHEN price < 200 THEN 'mid_range'
                        ELSE 'premium'
                    END as price_tier,
                    ingestion_timestamp,
                    record_quality_score,
                    CURRENT_TIMESTAMP as dbt_updated_at
                FROM processed_products
                WHERE product_id IS NOT NULL
            """,
            
            'int_customer_metrics': """
                SELECT 
                    c.customer_id,
                    c.email,
                    c.first_name,
                    c.last_name,
                    c.registration_date,
                    c.customer_segment,
                    c.email_is_valid,
                    
                    -- Transaction metrics
                    COALESCE(t.total_transactions, 0) as total_transactions,
                    COALESCE(t.total_spent, 0) as total_spent,
                    COALESCE(t.avg_transaction_amount, 0) as avg_transaction_amount,
                    t.first_transaction_date,
                    t.last_transaction_date,
                    
                    -- Customer lifetime metrics
                    CASE 
                        WHEN t.first_transaction_date IS NOT NULL THEN
                            EXTRACT(DAYS FROM (COALESCE(t.last_transaction_date, CURRENT_DATE) - t.first_transaction_date))
                        ELSE 0
                    END as customer_lifetime_days,
                    
                    -- Recency analysis
                    CASE 
                        WHEN t.last_transaction_date IS NOT NULL THEN
                            EXTRACT(DAYS FROM (CURRENT_DATE - t.last_transaction_date))
                        ELSE 999
                    END as days_since_last_transaction,
                    
                    CURRENT_TIMESTAMP as dbt_updated_at
                    
                FROM stg_customers c
                LEFT JOIN (
                    SELECT 
                        customer_id,
                        COUNT(*) as total_transactions,
                        SUM(transaction_amount) as total_spent,
                        AVG(transaction_amount) as avg_transaction_amount,
                        MIN(transaction_date) as first_transaction_date,
                        MAX(transaction_date) as last_transaction_date
                    FROM stg_transactions
                    GROUP BY customer_id
                ) t ON c.customer_id = t.customer_id
            """,
            
            'customer_360': """
                SELECT 
                    cm.customer_id,
                    cm.email,
                    cm.first_name,
                    cm.last_name,
                    cm.registration_date,
                    cm.customer_segment,
                    cm.total_transactions,
                    cm.total_spent,
                    cm.avg_transaction_amount,
                    cm.customer_lifetime_days,
                    cm.days_since_last_transaction,
                    
                    -- Customer value segmentation
                    CASE 
                        WHEN cm.total_spent > 10000 THEN 'high_value'
                        WHEN cm.total_spent > 1000 THEN 'medium_value'
                        WHEN cm.total_spent > 0 THEN 'low_value'
                        ELSE 'no_purchases'
                    END as value_tier,
                    
                    -- Customer lifecycle stage
                    CASE 
                        WHEN cm.days_since_last_transaction <= 30 THEN 'active'
                        WHEN cm.days_since_last_transaction <= 90 THEN 'at_risk'
                        WHEN cm.days_since_last_transaction <= 365 THEN 'dormant'
                        ELSE 'churned'
                    END as lifecycle_stage,
                    
                    -- RFM Analysis
                    CASE 
                        WHEN cm.days_since_last_transaction <= 30 THEN 5
                        WHEN cm.days_since_last_transaction <= 60 THEN 4
                        WHEN cm.days_since_last_transaction <= 90 THEN 3
                        WHEN cm.days_since_last_transaction <= 180 THEN 2
                        ELSE 1
                    END as recency_score,
                    
                    CASE 
                        WHEN cm.total_transactions >= 50 THEN 5
                        WHEN cm.total_transactions >= 20 THEN 4
                        WHEN cm.total_transactions >= 10 THEN 3
                        WHEN cm.total_transactions >= 5 THEN 2
                        ELSE 1
                    END as frequency_score,
                    
                    CASE 
                        WHEN cm.total_spent >= 10000 THEN 5
                        WHEN cm.total_spent >= 5000 THEN 4
                        WHEN cm.total_spent >= 1000 THEN 3
                        WHEN cm.total_spent >= 500 THEN 2
                        ELSE 1
                    END as monetary_score,
                    
                    CURRENT_TIMESTAMP as dbt_updated_at
                    
                FROM int_customer_metrics cm
                WHERE cm.email_is_valid = true
            """
        }
        
        return model_sql_map.get(model_name, f"SELECT * FROM {model_name}_source")
    
    def _run_transformation_tests(self, stage_name: str) -> Dict[str, Any]:
        """Run data quality tests for transformation stage"""
        
        test_results = {
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Define tests for each stage
        stage_tests = {
            'staging': [
                {'name': 'stg_customers_not_null', 'sql': "SELECT COUNT(*) FROM stg_customers WHERE customer_id IS NULL"},
                {'name': 'stg_customers_unique', 'sql': "SELECT customer_id, COUNT(*) FROM stg_customers GROUP BY customer_id HAVING COUNT(*) > 1"},
                {'name': 'stg_transactions_positive_amounts', 'sql': "SELECT COUNT(*) FROM stg_transactions WHERE transaction_amount <= 0"}
            ],
            'intermediate': [
                {'name': 'int_customer_metrics_completeness', 'sql': "SELECT COUNT(*) FROM int_customer_metrics WHERE customer_id IS NULL"}
            ],
            'marts': [
                {'name': 'customer_360_quality', 'sql': "SELECT COUNT(*) FROM customer_360 WHERE value_tier IS NULL"}
            ]
        }
        
        tests = stage_tests.get(stage_name, [])
        
        for test in tests:
            try:
                with self.platform.db_engine.connect() as conn:
                    result = conn.execute(test['sql'])
                    failure_count = result.fetchone()[0]
                    
                    if failure_count == 0:
                        test_results['passed_tests'] += 1
                        test_status = 'passed'
                    else:
                        test_results['failed_tests'] += 1
                        test_status = 'failed'
                    
                    test_results['test_details'].append({
                        'test_name': test['name'],
                        'status': test_status,
                        'failure_count': failure_count
                    })
                    
            except Exception as e:
                test_results['failed_tests'] += 1
                test_results['test_details'].append({
                    'test_name': test['name'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return test_results

# =============================================================================
# AWS INTEGRATION MANAGER
# =============================================================================

class AWSIntegrationManager:
    """Comprehensive AWS services integration"""
    
    def __init__(self, platform: ProductionDataPlatform):
        self.platform = platform
        self.logger = platform.logger
    
    def execute_aws_processing(self) -> Dict[str, Any]:
        """Execute AWS Glue and Kinesis processing"""
        
        aws_results = {
            'glue_jobs_executed': 0,
            'kinesis_streams_processed': 0,
            'failed_operations': []
        }
        
        try:
            # Execute Glue ETL job
            glue_result = self._execute_glue_etl_job()
            if glue_result['success']:
                aws_results['glue_jobs_executed'] += 1
            else:
                aws_results['failed_operations'].append({
                    'operation': 'glue_etl',
                    'error': glue_result['error']
                })
            
            # Process Kinesis streams
            kinesis_result = self._process_kinesis_streams()
            if kinesis_result['success']:
                aws_results['kinesis_streams_processed'] += kinesis_result['streams_processed']
            else:
                aws_results['failed_operations'].append({
                    'operation': 'kinesis_processing',
                    'error': kinesis_result['error']
                })
            
            # Upload processed data to S3
            s3_result = self._upload_to_s3()
            if not s3_result['success']:
                aws_results['failed_operations'].append({
                    'operation': 's3_upload',
                    'error': s3_result['error']
                })
            
        except Exception as e:
            aws_results['failed_operations'].append({
                'operation': 'aws_integration',
                'error': str(e)
            })
            
            self.platform.metrics['errors_total'].labels(
                component='aws_integration',
                error_type='general_failure'
            ).inc()
        
        return aws_results
    
    def _execute_glue_etl_job(self) -> Dict[str, Any]:
        """Execute AWS Glue ETL job"""
        
        try:
            # Start Glue job
            response = self.platform.glue_client.start_job_run(
                JobName=self.platform.config.glue_job_name,
                Arguments={
                    '--source_database': 'dataflow_warehouse',
                    '--target_s3_path': f's3://{self.platform.config.s3_bucket}/processed/',
                    '--enable_metrics': 'true'
                }
            )
            
            job_run_id = response['JobRunId']
            
            # Monitor job execution (simplified - in production would poll status)
            self.logger.info(f"Glue job started with run ID: {job_run_id}")
            
            # Simulate job completion for demo
            time.sleep(2)
            
            return {
                'success': True,
                'job_run_id': job_run_id,
                'status': 'completed'
            }
            
        except Exception as e:
            self.logger.error(f"Glue job execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _process_kinesis_streams(self) -> Dict[str, Any]:
        """Process Kinesis streams for real-time data"""
        
        try:
            stream_name = self.platform.config.kinesis_stream
            
            # Check if stream exists
            try:
                response = self.platform.kinesis_client.describe_stream(StreamName=stream_name)
                stream_status = response['StreamDescription']['StreamStatus']
                
                if stream_status != 'ACTIVE':
                    return {
                        'success': False,
                        'error': f'Stream {stream_name} is not active (status: {stream_status})'
                    }
                
            except self.platform.kinesis_client.exceptions.ResourceNotFoundException:
                # Create stream if it doesn't exist
                self.platform.kinesis_client.create_stream(
                    StreamName=stream_name,
                    ShardCount=2
                )
                
                # Wait for stream to become active (simplified)
                time.sleep(10)
            
            # Process stream records (simplified)
            processed_records = self._simulate_stream_processing()
            
            return {
                'success': True,
                'streams_processed': 1,
                'records_processed': processed_records
            }
            
        except Exception as e:
            self.logger.error(f"Kinesis stream processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _simulate_stream_processing(self) -> int:
        """Simulate real-time stream processing"""
        
        # Generate sample streaming data
        streaming_records = []
        
        for i in range(100):
            record = {
                'event_id': f'event_{i}',
                'customer_id': f'customer_{i % 50}',
                'event_type': 'page_view',
                'timestamp': datetime.now().isoformat(),
                'properties': {
                    'page': f'/page_{i % 10}',
                    'session_id': f'session_{i % 20}'
                }
            }
            streaming_records.append(record)
        
        # Process records (in production, this would be done by Lambda or Kinesis Analytics)
        processed_count = 0
        
        for record in streaming_records:
            # Simulate processing
            processed_record = {
                **record,
                'processed_at': datetime.now().isoformat(),
                'processing_version': '1.0'
            }
            
            # Store in database (simplified)
            try:
                df = pd.DataFrame([processed_record])
                df.to_sql('real_time_events', self.platform.db_engine, if_exists='append', index=False)
                processed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to store streaming record: {str(e)}")
        
        return processed_count
    
    def _upload_to_s3(self) -> Dict[str, Any]:
        """Upload processed data to S3"""
        
        try:
            # Export processed data
            tables_to_export = ['customer_360', 'stg_transactions', 'int_customer_metrics']
            
            for table_name in tables_to_export:
                # Read data from database
                df = pd.read_sql(f"SELECT * FROM {table_name}", self.platform.db_engine)
                
                # Convert to parquet
                parquet_buffer = df.to_parquet(index=False)
                
                # Upload to S3
                s3_key = f"processed/{table_name}/{datetime.now().strftime('%Y/%m/%d')}/{table_name}.parquet"
                
                self.platform.s3_client.put_object(
                    Bucket=self.platform.config.s3_bucket,
                    Key=s3_key,
                    Body=parquet_buffer
                )
                
                self.logger.info(f"Uploaded {table_name} to S3: s3://{self.platform.config.s3_bucket}/{s3_key}")
            
            return {'success': True, 'tables_uploaded': len(tables_to_export)}
            
        except Exception as e:
            self.logger.error(f"S3 upload failed: {str(e)}")
            return {'success': False, 'error': str(e)}

# =============================================================================
# COMPREHENSIVE MONITORING
# =============================================================================

class ComprehensiveMonitoring:
    """Advanced monitoring and alerting system"""
    
    def __init__(self, platform: ProductionDataPlatform):
        self.platform = platform
        self.logger = platform.logger
    
    def update_pipeline_metrics(self) -> Dict[str, Any]:
        """Update comprehensive pipeline metrics"""
        
        monitoring_results = {
            'metrics_collected': 0,
            'alerts_generated': 0,
            'system_health': 'healthy'
        }
        
        try:
            # Collect system metrics
            system_metrics = self._collect_system_metrics()
            monitoring_results['metrics_collected'] += len(system_metrics)
            
            # Collect data quality metrics
            quality_metrics = self._collect_quality_metrics()
            monitoring_results['metrics_collected'] += len(quality_metrics)
            
            # Collect performance metrics
            performance_metrics = self._collect_performance_metrics()
            monitoring_results['metrics_collected'] += len(performance_metrics)
            
            # Check alert conditions
            alerts = self._check_alert_conditions(system_metrics, quality_metrics, performance_metrics)
            monitoring_results['alerts_generated'] = len(alerts)
            
            # Determine overall system health
            monitoring_results['system_health'] = self._determine_system_health(
                system_metrics, quality_metrics, performance_metrics
            )
            
            # Send alerts if necessary
            for alert in alerts:
                self.send_pipeline_alert(alert['type'], alert['message'], alert.get('correlation_id'))
            
        except Exception as e:
            self.logger.error(f"Monitoring update failed: {str(e)}")
            monitoring_results['system_health'] = 'unhealthy'
        
        return monitoring_results
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        
        metrics = {}
        
        try:
            # Database connection health
            with self.platform.db_engine.connect() as conn:
                result = conn.execute("SELECT 1")
                metrics['database_connection_healthy'] = 1.0 if result.fetchone() else 0.0
        except:
            metrics['database_connection_healthy'] = 0.0
        
        try:
            # Redis connection health
            self.platform.redis_client.ping()
            metrics['redis_connection_healthy'] = 1.0
        except:
            metrics['redis_connection_healthy'] = 0.0
        
        # Memory usage (simplified)
        import psutil
        metrics['memory_usage_percent'] = psutil.virtual_memory().percent
        metrics['cpu_usage_percent'] = psutil.cpu_percent()
        
        return metrics
    
    def _collect_quality_metrics(self) -> Dict[str, float]:
        """Collect data quality metrics"""
        
        metrics = {}
        
        try:
            # Get latest quality scores from database
            with self.platform.db_engine.connect() as conn:
                # Customer data quality
                result = conn.execute("""
                    SELECT AVG(record_quality_score) as avg_quality 
                    FROM processed_customers 
                    WHERE ingestion_timestamp > NOW() - INTERVAL '1 hour'
                """)
                row = result.fetchone()
                metrics['customer_data_quality'] = float(row[0]) if row[0] else 0.0
                
                # Transaction data quality
                result = conn.execute("""
                    SELECT AVG(record_quality_score) as avg_quality 
                    FROM processed_transactions 
                    WHERE ingestion_timestamp > NOW() - INTERVAL '1 hour'
                """)
                row = result.fetchone()
                metrics['transaction_data_quality'] = float(row[0]) if row[0] else 0.0
                
        except Exception as e:
            self.logger.warning(f"Could not collect quality metrics: {str(e)}")
            metrics['customer_data_quality'] = 0.0
            metrics['transaction_data_quality'] = 0.0
        
        return metrics
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect performance metrics"""
        
        metrics = {}
        
        try:
            with self.platform.db_engine.connect() as conn:
                # Data freshness
                result = conn.execute("""
                    SELECT EXTRACT(EPOCH FROM (NOW() - MAX(ingestion_timestamp)))/60 as minutes_since_last_ingestion
                    FROM processed_customers
                """)
                row = result.fetchone()
                metrics['data_freshness_minutes'] = float(row[0]) if row[0] else 999.0
                
                # Record counts
                result = conn.execute("SELECT COUNT(*) FROM customer_360")
                row = result.fetchone()
                metrics['customer_360_record_count'] = float(row[0]) if row[0] else 0.0
                
        except Exception as e:
            self.logger.warning(f"Could not collect performance metrics: {str(e)}")
            metrics['data_freshness_minutes'] = 999.0
            metrics['customer_360_record_count'] = 0.0
        
        return metrics
    
    def _check_alert_conditions(self, system_metrics: Dict[str, float], 
                               quality_metrics: Dict[str, float], 
                               performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check conditions that should trigger alerts"""
        
        alerts = []
        
        # System health alerts
        if system_metrics.get('database_connection_healthy', 1.0) < 1.0:
            alerts.append({
                'type': 'system_failure',
                'message': 'Database connection is unhealthy',
                'severity': 'critical'
            })
        
        if system_metrics.get('memory_usage_percent', 0) > 90:
            alerts.append({
                'type': 'resource_exhaustion',
                'message': f'High memory usage: {system_metrics["memory_usage_percent"]:.1f}%',
                'severity': 'warning'
            })
        
        # Data quality alerts
        avg_quality = (quality_metrics.get('customer_data_quality', 1.0) + 
                      quality_metrics.get('transaction_data_quality', 1.0)) / 2
        
        if avg_quality < self.platform.config.quality_threshold:
            alerts.append({
                'type': 'quality_degradation',
                'message': f'Data quality below threshold: {avg_quality:.3f}',
                'severity': 'warning'
            })
        
        # Performance alerts
        if performance_metrics.get('data_freshness_minutes', 0) > 120:  # 2 hours
            alerts.append({
                'type': 'data_staleness',
                'message': f'Data is stale: {performance_metrics["data_freshness_minutes"]:.1f} minutes old',
                'severity': 'warning'
            })
        
        return alerts
    
    def _determine_system_health(self, system_metrics: Dict[str, float], 
                                quality_metrics: Dict[str, float], 
                                performance_metrics: Dict[str, float]) -> str:
        """Determine overall system health status"""
        
        # Check critical systems
        if (system_metrics.get('database_connection_healthy', 1.0) < 1.0 or
            system_metrics.get('redis_connection_healthy', 1.0) < 1.0):
            return 'critical'
        
        # Check quality and performance
        avg_quality = (quality_metrics.get('customer_data_quality', 1.0) + 
                      quality_metrics.get('transaction_data_quality', 1.0)) / 2
        
        if (avg_quality < 0.8 or 
            performance_metrics.get('data_freshness_minutes', 0) > 240 or  # 4 hours
            system_metrics.get('memory_usage_percent', 0) > 95):
            return 'degraded'
        
        return 'healthy'
    
    def send_pipeline_alert(self, alert_type: str, message: str, correlation_id: str = None):
        """Send pipeline alert through configured channels"""
        
        alert_data = {
            'alert_type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'correlation_id': correlation_id or f"alert_{int(time.time())}",
            'platform': 'dataflow_production'
        }
        
        # Log alert
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # In production, this would integrate with:
        # - PagerDuty for critical alerts
        # - Slack for team notifications
        # - Email for stakeholder updates
        # - SMS for urgent issues
        
        # Simulate alert sending
        print(f"🚨 ALERT SENT: {json.dumps(alert_data, indent=2)}")

# =============================================================================
# MAIN DEMONSTRATION FUNCTION
# =============================================================================

def demonstrate_production_pipeline():
    """Demonstrate the complete production data platform"""
    
    print("🚀 DataFlow Enterprises - Production Data Platform")
    print("=" * 80)
    
    # Initialize configuration
    config = PipelineConfig()
    
    try:
        # Initialize platform
        platform = ProductionDataPlatform(config)
        
        print("\n📊 Platform Components Initialized:")
        print("   ✅ Data Ingestion Manager")
        print("   ✅ Quality Validator (Great Expectations)")
        print("   ✅ Transformation Engine (dbt-style)")
        print("   ✅ AWS Integration Manager")
        print("   ✅ Comprehensive Monitoring")
        print("   ✅ Prometheus Metrics Server")
        
        # Execute complete pipeline
        print(f"\n🔄 Executing Complete Production Pipeline...")
        results = platform.execute_complete_pipeline()
        
        # Display results
        print("\n" + "="*80)
        print("✅ PRODUCTION PIPELINE EXECUTION COMPLETE!")
        print("="*80)
        
        print(f"\n📈 Execution Summary:")
        print(f"   ⏱️  Total Execution Time: {results['execution_time']:.2f} seconds")
        print(f"   📊 Status: {results['status'].upper()}")
        print(f"   🔗 Correlation ID: {results['correlation_id']}")
        
        print(f"\n📥 Data Ingestion Results:")
        ingestion = results['ingestion']
        print(f"   📁 Sources Processed: {ingestion['sources_processed']}")
        print(f"   📊 Total Records: {ingestion['total_records']:,}")
        print(f"   ❌ Failed Sources: {len(ingestion['failed_sources'])}")
        
        print(f"\n🔍 Quality Validation Results:")
        quality = results['quality']
        print(f"   📋 Datasets Validated: {quality['datasets_validated']}")
        print(f"   ⭐ Overall Quality Score: {quality['overall_quality_score']:.3f}")
        print(f"   ❌ Failed Validations: {len(quality['failed_validations'])}")
        
        print(f"\n⚙️ Transformation Results:")
        transformations = results['transformations']
        print(f"   🔄 Models Executed: {transformations['models_executed']}")
        print(f"   ✅ Tests Passed: {transformations['tests_passed']}")
        print(f"   ❌ Failed Models: {len(transformations['failed_models'])}")
        
        print(f"\n☁️ AWS Integration Results:")
        aws = results['aws_integration']
        print(f"   🔧 Glue Jobs Executed: {aws['glue_jobs_executed']}")
        print(f"   🌊 Kinesis Streams Processed: {aws['kinesis_streams_processed']}")
        print(f"   ❌ Failed Operations: {len(aws['failed_operations'])}")
        
        print(f"\n📊 Monitoring Results:")
        monitoring = results['monitoring']
        print(f"   📈 Metrics Collected: {monitoring['metrics_collected']}")
        print(f"   🚨 Alerts Generated: {monitoring['alerts_generated']}")
        print(f"   💚 System Health: {monitoring['system_health'].upper()}")
        
        print(f"\n🎯 Production Capabilities Demonstrated:")
        print("   • Multi-source data ingestion with concurrent processing")
        print("   • Comprehensive data quality validation with Great Expectations")
        print("   • Advanced data transformations with dbt-style SQL models")
        print("   • AWS Glue serverless ETL and Kinesis streaming integration")
        print("   • Real-time monitoring with Prometheus metrics")
        print("   • Intelligent alerting and system health monitoring")
        print("   • Production-grade error handling and recovery")
        print("   • Complete observability and audit trails")
        
        print(f"\n💼 Enterprise Features:")
        print("   • Horizontal scaling with concurrent processing")
        print("   • Incremental data processing with checkpointing")
        print("   • Comprehensive logging with correlation IDs")
        print("   • Quality gates preventing bad data propagation")
        print("   • Multi-stage transformation pipeline")
        print("   • Cloud-native AWS services integration")
        print("   • Production monitoring and alerting")
        print("   • Fault tolerance and graceful degradation")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    # Execute the complete production pipeline demonstration
    results = demonstrate_production_pipeline()
    
    print("\n" + "="*80)
    print("🎓 Production Data Platform demonstration completed!")
    print("This showcases enterprise-grade data engineering with comprehensive")
    print("integration of all Phase 2 technologies and production best practices.")
    print("="*80)