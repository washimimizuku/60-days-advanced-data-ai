# Day 22: AWS Glue & Data Catalog - Serverless ETL

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** AWS Glue for serverless ETL processing and data transformation at scale
- **Implement** Data Catalog crawlers for automated schema discovery and metadata management
- **Build** production-ready Glue ETL jobs using PySpark with advanced transformations
- **Integrate** Athena for serverless SQL analytics on data lake architectures
- **Design** cost-effective serverless data pipelines with monitoring and optimization

---

## Theory

### AWS Glue Serverless ETL Architecture

AWS Glue is a fully managed serverless ETL service that makes it easy to prepare and transform data for analytics. It eliminates the need to provision and manage infrastructure while providing powerful data processing capabilities.

#### 1. AWS Glue Core Components

**Data Catalog - Central Metadata Repository**:
```python
# Example: Programmatic Data Catalog interaction
import boto3

glue_client = boto3.client('glue')

# Create database in Data Catalog
def create_glue_database(database_name, description):
    """Create a database in AWS Glue Data Catalog"""
    
    try:
        response = glue_client.create_database(
            DatabaseInput={
                'Name': database_name,
                'Description': description,
                'Parameters': {
                    'classification': 'data_lake',
                    'created_by': 'data_engineering_team',
                    'environment': 'production'
                }
            }
        )
        
        print(f"Database '{database_name}' created successfully")
        return response
        
    except glue_client.exceptions.AlreadyExistsException:
        print(f"Database '{database_name}' already exists")
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        raise

# Create table definition
def create_glue_table(database_name, table_name, s3_location, columns):
    """Create a table definition in Glue Data Catalog"""
    
    storage_descriptor = {
        'Columns': columns,
        'Location': s3_location,
        'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
        'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
        'SerdeInfo': {
            'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
            'Parameters': {
                'field.delim': ',',
                'skip.header.line.count': '1'
            }
        }
    }
    
    table_input = {
        'Name': table_name,
        'StorageDescriptor': storage_descriptor,
        'PartitionKeys': [
            {
                'Name': 'year',
                'Type': 'string'
            },
            {
                'Name': 'month',
                'Type': 'string'
            }
        ],
        'TableType': 'EXTERNAL_TABLE',
        'Parameters': {
            'classification': 'csv',
            'compressionType': 'gzip',
            'typeOfData': 'file'
        }
    }
    
    try:
        response = glue_client.create_table(
            DatabaseName=database_name,
            TableInput=table_input
        )
        
        print(f"Table '{table_name}' created in database '{database_name}'")
        return response
        
    except Exception as e:
        print(f"Error creating table: {str(e)}")
        raise
```

**Crawlers - Automated Schema Discovery**:
```python
# Example: Comprehensive crawler configuration
def create_glue_crawler(crawler_name, database_name, s3_targets, iam_role):
    """Create a Glue crawler for automated schema discovery"""
    
    crawler_config = {
        'Name': crawler_name,
        'Role': iam_role,
        'DatabaseName': database_name,
        'Description': 'Automated schema discovery for data lake',
        'Targets': {
            'S3Targets': s3_targets
        },
        'Schedule': 'cron(0 2 * * ? *)',  # Daily at 2 AM
        'SchemaChangePolicy': {
            'UpdateBehavior': 'UPDATE_IN_DATABASE',
            'DeleteBehavior': 'LOG'
        },
        'RecrawlPolicy': {
            'RecrawlBehavior': 'CRAWL_EVERYTHING'
        },
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
        response = glue_client.create_crawler(**crawler_config)
        print(f"Crawler '{crawler_name}' created successfully")
        
        # Start the crawler
        glue_client.start_crawler(Name=crawler_name)
        print(f"Crawler '{crawler_name}' started")
        
        return response
        
    except Exception as e:
        print(f"Error creating crawler: {str(e)}")
        raise

# Monitor crawler execution
def monitor_crawler_run(crawler_name):
    """Monitor crawler execution and return status"""
    
    try:
        response = glue_client.get_crawler(Name=crawler_name)
        crawler_state = response['Crawler']['State']
        
        if crawler_state == 'RUNNING':
            print(f"Crawler '{crawler_name}' is currently running...")
            
        elif crawler_state == 'READY':
            # Get last run details
            last_crawl = response['Crawler']['LastCrawl']
            if last_crawl:
                print(f"Last crawl status: {last_crawl['Status']}")
                print(f"Tables created: {last_crawl.get('TablesCreated', 0)}")
                print(f"Tables updated: {last_crawl.get('TablesUpdated', 0)}")
                print(f"Tables deleted: {last_crawl.get('TablesDeleted', 0)}")
        
        return crawler_state
        
    except Exception as e:
        print(f"Error monitoring crawler: {str(e)}")
        raise
```

#### 2. Glue ETL Jobs with PySpark

**Advanced ETL Job Development**:
```python
# Example: Production Glue ETL job script
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'source_database', 'source_table', 'target_s3_path'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

def advanced_customer_analytics_etl():
    """Advanced ETL job for customer analytics with complex transformations"""
    
    # Read from Data Catalog
    source_dyf = glueContext.create_dynamic_frame.from_catalog(
        database=args['source_database'],
        table_name=args['source_table'],
        transformation_ctx="source_dyf"
    )
    
    print(f"Source record count: {source_dyf.count()}")
    
    # Convert to Spark DataFrame for complex operations
    source_df = source_dyf.toDF()
    
    # Data quality checks and cleaning
    cleaned_df = source_df.filter(
        (F.col("customer_id").isNotNull()) &
        (F.col("transaction_amount") > 0) &
        (F.col("transaction_date").isNotNull())
    )
    
    # Advanced transformations
    enriched_df = cleaned_df.withColumn(
        "transaction_year", F.year("transaction_date")
    ).withColumn(
        "transaction_month", F.month("transaction_date")
    ).withColumn(
        "transaction_day_of_week", F.dayofweek("transaction_date")
    ).withColumn(
        "amount_category", 
        F.when(F.col("transaction_amount") < 50, "low")
         .when(F.col("transaction_amount") < 200, "medium")
         .otherwise("high")
    )
    
    # Customer aggregations with window functions
    from pyspark.sql.window import Window
    
    customer_window = Window.partitionBy("customer_id").orderBy("transaction_date")
    
    customer_analytics_df = enriched_df.withColumn(
        "customer_transaction_rank", 
        F.row_number().over(customer_window)
    ).withColumn(
        "customer_running_total",
        F.sum("transaction_amount").over(customer_window.rowsBetween(Window.unboundedPreceding, Window.currentRow))
    ).withColumn(
        "days_since_last_transaction",
        F.datediff(F.col("transaction_date"), F.lag("transaction_date").over(customer_window))
    )
    
    # Calculate customer lifetime metrics
    customer_summary_df = customer_analytics_df.groupBy("customer_id").agg(
        F.sum("transaction_amount").alias("total_spent"),
        F.avg("transaction_amount").alias("avg_transaction_amount"),
        F.count("transaction_id").alias("transaction_count"),
        F.min("transaction_date").alias("first_transaction_date"),
        F.max("transaction_date").alias("last_transaction_date"),
        F.stddev("transaction_amount").alias("spending_volatility")
    ).withColumn(
        "customer_lifetime_days",
        F.datediff("last_transaction_date", "first_transaction_date")
    ).withColumn(
        "avg_days_between_transactions",
        F.col("customer_lifetime_days") / F.col("transaction_count")
    )
    
    # Customer segmentation using RFM analysis
    # Calculate percentiles for segmentation
    percentiles = customer_summary_df.select(
        F.expr("percentile_approx(total_spent, 0.8)").alias("monetary_80th"),
        F.expr("percentile_approx(transaction_count, 0.8)").alias("frequency_80th"),
        F.expr("percentile_approx(customer_lifetime_days, 0.2)").alias("recency_20th")
    ).collect()[0]
    
    segmented_customers_df = customer_summary_df.withColumn(
        "customer_segment",
        F.when(
            (F.col("total_spent") >= percentiles["monetary_80th"]) &
            (F.col("transaction_count") >= percentiles["frequency_80th"]) &
            (F.col("customer_lifetime_days") <= percentiles["recency_20th"]),
            "champion"
        ).when(
            (F.col("total_spent") >= percentiles["monetary_80th"]) &
            (F.col("transaction_count") >= percentiles["frequency_80th"]),
            "loyal_customer"
        ).when(
            F.col("total_spent") >= percentiles["monetary_80th"],
            "big_spender"
        ).when(
            F.col("transaction_count") >= percentiles["frequency_80th"],
            "frequent_buyer"
        ).otherwise("regular_customer")
    )
    
    # Convert back to DynamicFrame for Glue optimizations
    final_dyf = DynamicFrame.fromDF(segmented_customers_df, glueContext, "final_dyf")
    
    # Apply Glue transformations for optimization
    final_dyf = final_dyf.resolveChoice(specs=[('spending_volatility', 'cast:double')])
    
    # Write to S3 with partitioning and compression
    glueContext.write_dynamic_frame.from_options(
        frame=final_dyf,
        connection_type="s3",
        connection_options={
            "path": args['target_s3_path'],
            "partitionKeys": ["customer_segment"]
        },
        format="parquet",
        format_options={
            "compression": "snappy"
        },
        transformation_ctx="final_write"
    )
    
    print(f"ETL job completed. Final record count: {final_dyf.count()}")
    
    # Data quality metrics
    segment_counts = segmented_customers_df.groupBy("customer_segment").count().collect()
    for row in segment_counts:
        print(f"Segment '{row['customer_segment']}': {row['count']} customers")

# Execute ETL job
advanced_customer_analytics_etl()

job.commit()
```

#### 3. Athena Integration and Optimization

**Advanced Athena Query Patterns**:
```python
# Example: Athena integration with optimization
import boto3
import time
import pandas as pd

athena_client = boto3.client('athena')
s3_client = boto3.client('s3')

class AthenaQueryManager:
    """Advanced Athena query management with optimization"""
    
    def __init__(self, database_name, s3_results_location):
        self.database_name = database_name
        self.s3_results_location = s3_results_location
        self.athena_client = boto3.client('athena')
    
    def execute_query(self, query, query_name=None):
        """Execute Athena query with monitoring and optimization"""
        
        query_execution_context = {
            'Database': self.database_name
        }
        
        result_configuration = {
            'OutputLocation': self.s3_results_location,
            'EncryptionConfiguration': {
                'EncryptionOption': 'SSE_S3'
            }
        }
        
        # Add query optimization hints
        optimized_query = self._optimize_query(query)
        
        try:
            response = self.athena_client.start_query_execution(
                QueryString=optimized_query,
                QueryExecutionContext=query_execution_context,
                ResultConfiguration=result_configuration,
                WorkGroup='primary'
            )
            
            query_execution_id = response['QueryExecutionId']
            print(f"Query execution started: {query_execution_id}")
            
            # Monitor query execution
            execution_result = self._monitor_query_execution(query_execution_id)
            
            if execution_result['status'] == 'SUCCEEDED':
                return self._get_query_results(query_execution_id)
            else:
                raise Exception(f"Query failed: {execution_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            raise
    
    def _optimize_query(self, query):
        """Apply query optimization techniques"""
        
        # Add common optimization hints
        optimizations = [
            "-- Query optimized for Athena performance",
            "-- Using columnar format and partitioning"
        ]
        
        # Check for common optimization opportunities
        if "GROUP BY" in query.upper() and "LIMIT" not in query.upper():
            query += " LIMIT 10000"  # Add reasonable limit for aggregations
        
        return "\n".join(optimizations) + "\n" + query
    
    def _monitor_query_execution(self, query_execution_id):
        """Monitor query execution until completion"""
        
        max_wait_time = 300  # 5 minutes
        wait_interval = 2
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            response = self.athena_client.get_query_execution(
                QueryExecutionId=query_execution_id
            )
            
            status = response['QueryExecution']['Status']['State']
            
            if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                execution_stats = response['QueryExecution']['Statistics']
                
                result = {
                    'status': status,
                    'execution_time_ms': execution_stats.get('EngineExecutionTimeInMillis', 0),
                    'data_scanned_bytes': execution_stats.get('DataScannedInBytes', 0),
                    'query_queue_time_ms': execution_stats.get('QueryQueueTimeInMillis', 0)
                }
                
                if status == 'FAILED':
                    result['error'] = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                
                print(f"Query {status.lower()}")
                print(f"Execution time: {result['execution_time_ms']}ms")
                print(f"Data scanned: {result['data_scanned_bytes'] / (1024*1024):.2f} MB")
                
                return result
            
            time.sleep(wait_interval)
            elapsed_time += wait_interval
        
        raise TimeoutError(f"Query execution timed out after {max_wait_time} seconds")
    
    def _get_query_results(self, query_execution_id):
        """Retrieve query results as pandas DataFrame"""
        
        try:
            results = []
            next_token = None
            
            while True:
                if next_token:
                    response = self.athena_client.get_query_results(
                        QueryExecutionId=query_execution_id,
                        NextToken=next_token
                    )
                else:
                    response = self.athena_client.get_query_results(
                        QueryExecutionId=query_execution_id
                    )
                
                # Extract column names from first result set
                if not results:
                    columns = [col['Name'] for col in response['ResultSet']['ResultSetMetadata']['ColumnInfo']]
                
                # Extract data rows (skip header row)
                for row in response['ResultSet']['Rows'][1:]:
                    row_data = [field.get('VarCharValue', '') for field in row['Data']]
                    results.append(row_data)
                
                # Check for more results
                next_token = response.get('NextToken')
                if not next_token:
                    break
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(results, columns=columns)
            return df
            
        except Exception as e:
            print(f"Error retrieving query results: {str(e)}")
            raise

# Example usage of advanced Athena queries
def run_advanced_analytics_queries():
    """Run advanced analytics queries using Athena"""
    
    athena_manager = AthenaQueryManager(
        database_name='customer_analytics_db',
        s3_results_location='s3://my-athena-results-bucket/queries/'
    )
    
    # Complex analytical query with window functions
    customer_cohort_query = """
    WITH customer_cohorts AS (
        SELECT 
            customer_id,
            DATE_TRUNC('month', first_transaction_date) as cohort_month,
            DATE_TRUNC('month', transaction_date) as transaction_month,
            transaction_amount
        FROM customer_transactions ct
        JOIN (
            SELECT 
                customer_id,
                MIN(transaction_date) as first_transaction_date
            FROM customer_transactions
            GROUP BY customer_id
        ) first_tx ON ct.customer_id = first_tx.customer_id
    ),
    cohort_analysis AS (
        SELECT 
            cohort_month,
            transaction_month,
            COUNT(DISTINCT customer_id) as customers,
            SUM(transaction_amount) as revenue,
            DATE_DIFF('month', cohort_month, transaction_month) as period_number
        FROM customer_cohorts
        GROUP BY cohort_month, transaction_month
    )
    SELECT 
        cohort_month,
        period_number,
        customers,
        revenue,
        LAG(customers) OVER (PARTITION BY cohort_month ORDER BY period_number) as prev_period_customers,
        ROUND(
            customers * 100.0 / FIRST_VALUE(customers) OVER (PARTITION BY cohort_month ORDER BY period_number), 
            2
        ) as retention_rate
    FROM cohort_analysis
    ORDER BY cohort_month, period_number
    """
    
    print("Executing customer cohort analysis...")
    cohort_results = athena_manager.execute_query(customer_cohort_query, "customer_cohort_analysis")
    print(f"Cohort analysis completed. Results shape: {cohort_results.shape}")
    
    return cohort_results
```

#### 4. Cost Optimization and Performance Tuning

**Glue Job Optimization Strategies**:
```python
# Example: Glue job optimization configuration
def create_optimized_glue_job(job_name, script_location, iam_role):
    """Create optimized Glue job with performance tuning"""
    
    job_config = {
        'Name': job_name,
        'Role': iam_role,
        'Command': {
            'Name': 'glueetl',
            'ScriptLocation': script_location,
            'PythonVersion': '3'
        },
        'DefaultArguments': {
            '--job-language': 'python',
            '--job-bookmark-option': 'job-bookmark-enable',
            '--enable-metrics': 'true',
            '--enable-continuous-cloudwatch-log': 'true',
            '--enable-spark-ui': 'true',
            '--spark-event-logs-path': 's3://my-glue-logs/spark-logs/',
            
            # Performance optimizations
            '--conf': 'spark.sql.adaptive.enabled=true',
            '--conf': 'spark.sql.adaptive.coalescePartitions.enabled=true',
            '--conf': 'spark.sql.adaptive.skewJoin.enabled=true',
            '--conf': 'spark.serializer=org.apache.spark.serializer.KryoSerializer',
            
            # Memory optimization
            '--conf': 'spark.sql.execution.arrow.pyspark.enabled=true',
            '--conf': 'spark.sql.execution.arrow.maxRecordsPerBatch=10000',
            
            # S3 optimization
            '--conf': 'spark.hadoop.fs.s3a.multipart.size=104857600',  # 100MB
            '--conf': 'spark.hadoop.fs.s3a.multipart.threshold=104857600',
            '--conf': 'spark.hadoop.fs.s3a.fast.upload=true'
        },
        'MaxRetries': 1,
        'Timeout': 2880,  # 48 hours
        'MaxCapacity': 10,  # DPU allocation
        'GlueVersion': '3.0',
        'NumberOfWorkers': 10,
        'WorkerType': 'G.1X'
    }
    
    try:
        response = glue_client.create_job(**job_config)
        print(f"Optimized Glue job '{job_name}' created successfully")
        return response
        
    except Exception as e:
        print(f"Error creating Glue job: {str(e)}")
        raise

# Monitoring and cost tracking
def monitor_glue_job_costs(job_name, start_date, end_date):
    """Monitor Glue job costs and performance metrics"""
    
    cloudwatch = boto3.client('cloudwatch')
    
    # Get job run metrics
    job_runs = glue_client.get_job_runs(JobName=job_name)
    
    total_dpu_hours = 0
    total_cost = 0
    
    for run in job_runs['JobRuns']:
        if start_date <= run['StartedOn'].date() <= end_date:
            execution_time_hours = run.get('ExecutionTime', 0) / 3600
            max_capacity = run.get('MaxCapacity', 2)
            
            dpu_hours = execution_time_hours * max_capacity
            cost = dpu_hours * 0.44  # Current Glue DPU pricing
            
            total_dpu_hours += dpu_hours
            total_cost += cost
            
            print(f"Run {run['Id']}: {execution_time_hours:.2f}h, {dpu_hours:.2f} DPU-hours, ${cost:.2f}")
    
    print(f"Total DPU-hours: {total_dpu_hours:.2f}")
    print(f"Total estimated cost: ${total_cost:.2f}")
    
    return {
        'total_dpu_hours': total_dpu_hours,
        'total_cost': total_cost,
        'avg_cost_per_run': total_cost / len(job_runs['JobRuns']) if job_runs['JobRuns'] else 0
    }
```

## Hands-On Exercise (40 min)

You're the Data Engineering Lead at "ServerlessData Corp", a company that needs to build a scalable, cost-effective data lake solution. Your task is to implement a complete serverless ETL pipeline using AWS Glue, Data Catalog, and Athena.

**Business Context:**
- Processing 10GB+ of customer transaction data daily
- Need for real-time schema discovery and evolution
- Cost optimization is critical (serverless-first approach)
- Business analysts need SQL access to processed data
- Compliance requires data lineage and governance

**Your Mission:**
Build a production-ready serverless data pipeline that automatically discovers schemas, transforms data, and enables analytics.

## Key Takeaways

- **Serverless Architecture**: AWS Glue eliminates infrastructure management while providing powerful ETL capabilities
- **Automated Schema Discovery**: Crawlers automatically detect and catalog data schemas, reducing manual overhead
- **Cost-Effective Processing**: Pay-per-use model with automatic scaling makes Glue ideal for variable workloads
- **Data Catalog Integration**: Centralized metadata management enables consistent data discovery across services
- **Athena Analytics**: Serverless SQL queries on S3 provide immediate analytics capabilities without data movement
- **Performance Optimization**: Proper partitioning, compression, and Spark tuning significantly improve performance and reduce costs
- **Monitoring and Governance**: Built-in CloudWatch integration and Data Catalog lineage support compliance requirements
- **Ecosystem Integration**: Seamless integration with other AWS services creates comprehensive data lake architectures

## What's Next?

Tomorrow we'll explore **AWS Kinesis & Streaming** where you'll learn to build real-time data processing pipelines that complement your batch ETL processes. You'll discover how to handle streaming data ingestion, real-time transformations, and integrate streaming with your data lake architecture for comprehensive analytics solutions.
