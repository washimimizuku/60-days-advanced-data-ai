# Day 4: Data Warehouses - Snowflake Specifics

## 📖 Learning Objectives

**Estimated Time**: 60 minutes

By the end of today, you will:
- Understand Snowflake's unique architecture and advantages
- Master Snowflake SQL and advanced features
- Implement data loading and transformation patterns
- Design efficient data warehouse schemas
- Apply performance optimization techniques for analytics workloads

---

## Theory

### What is Snowflake?

Snowflake is a cloud-native data warehouse that separates compute and storage, enabling independent scaling and pay-per-use pricing. It's designed for modern analytics workloads with built-in features for data sharing, security, and governance.

> **📝 Setup Note**: This lesson uses Snowflake SQL examples. You can follow along with a [free Snowflake trial account](https://signup.snowflake.com/) (30-day trial with $400 credits). No installation required - Snowflake runs entirely in the cloud through a web interface.
>
> **Quick Start Steps**:
> 1. Sign up at [signup.snowflake.com](https://signup.snowflake.com/)
> 2. Choose your cloud provider (AWS/Azure/GCP) and region
> 3. Verify your email and set up your account
> 4. Access the Snowflake web interface (Snowsight)
> 5. Create your first warehouse and database to follow along

**Key advantages**:
- **Separation of compute and storage**: Scale independently
- **Multi-cluster architecture**: Automatic scaling and concurrency
- **Zero-copy cloning**: Instant data copies without storage overhead
- **Time Travel**: Query historical data states
- **Data sharing**: Secure data sharing across organizations
- **Semi-structured data**: Native JSON, XML, Parquet support
- **Cloud-agnostic**: Runs on AWS, Azure, and GCP

### Snowflake Architecture

#### 1. Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVICES LAYER                           │
│  (Cloud Services - Metadata, Security, Optimization)       │
├─────────────────────────────────────────────────────────────┤
│                    COMPUTE LAYER                            │
│  (Virtual Warehouses - Independent compute clusters)       │
├─────────────────────────────────────────────────────────────┤
│                    STORAGE LAYER                            │
│  (Cloud Storage - Compressed, columnar, encrypted)         │
└─────────────────────────────────────────────────────────────┘
```

**Services Layer**:
- Query parsing and optimization
- Metadata management
- Security and access control
- Transaction management

**Compute Layer**:
- Virtual warehouses (compute clusters)
- Independent scaling
- Automatic suspend/resume
- Multi-cluster warehouses for concurrency

**Storage Layer**:
- Cloud object storage (S3, Azure Blob, GCS)
- Automatic compression and encryption
- Columnar storage format
- Micro-partitions for performance

#### 2. Virtual Warehouses

Virtual warehouses are compute clusters that execute queries.

```sql
-- Create virtual warehouse
CREATE WAREHOUSE analytics_wh WITH
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 300          -- Suspend after 5 minutes
  AUTO_RESUME = TRUE          -- Auto-resume on query
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 3       -- Scale out for concurrency
  SCALING_POLICY = 'STANDARD';

-- Different warehouse sizes
-- X-SMALL: 1 credit/hour
-- SMALL: 2 credits/hour  
-- MEDIUM: 4 credits/hour
-- LARGE: 8 credits/hour
-- X-LARGE: 16 credits/hour
-- 2X-LARGE: 32 credits/hour
-- 3X-LARGE: 64 credits/hour
-- 4X-LARGE: 128 credits/hour
```

### Snowflake SQL Features

#### 1. Standard SQL with Extensions

```sql
-- Standard SQL works
SELECT customer_id, SUM(amount) as total_spent
FROM orders 
WHERE order_date >= '2024-01-01'
GROUP BY customer_id
ORDER BY total_spent DESC;

-- Snowflake extensions
SELECT customer_id,
       SUM(amount) as total_spent,
       RANK() OVER (ORDER BY SUM(amount) DESC) as spending_rank,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_order
FROM orders 
WHERE order_date >= CURRENT_DATE - 30
GROUP BY customer_id;
```

#### 2. Semi-Structured Data Support

```sql
-- Create table with VARIANT column for JSON
CREATE TABLE events (
    event_id STRING,
    event_data VARIANT,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Insert JSON data
INSERT INTO events VALUES 
('evt_1', PARSE_JSON('{"user_id": 123, "action": "login", "metadata": {"ip": "192.168.1.1"}}'));

-- Query JSON data
SELECT 
    event_id,
    event_data:user_id::INTEGER as user_id,
    event_data:action::STRING as action,
    event_data:metadata:ip::STRING as ip_address
FROM events;

-- Flatten nested arrays
SELECT 
    event_id,
    f.value:product_id::STRING as product_id,
    f.value:quantity::INTEGER as quantity
FROM events,
LATERAL FLATTEN(input => event_data:items) f;
```

#### 3. Time Travel and Cloning

```sql
-- Query data as it was 1 hour ago
SELECT * FROM orders AT(OFFSET => -3600);

-- Query data as it was at specific timestamp
SELECT * FROM orders AT(TIMESTAMP => '2024-01-15 10:00:00'::TIMESTAMP);

-- Clone table (zero-copy)
CREATE TABLE orders_backup CLONE orders;

-- Clone database
CREATE DATABASE analytics_dev CLONE analytics_prod;

-- Undrop table (within retention period)
DROP TABLE orders;
UNDROP TABLE orders;
```

#### 4. Advanced Window Functions

```sql
-- Running totals and moving averages
SELECT 
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY order_date) as running_total,
    AVG(daily_revenue) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma_7day
FROM (
    SELECT 
        DATE(order_date) as order_date,
        SUM(amount) as daily_revenue
    FROM orders
    GROUP BY DATE(order_date)
) daily_sales
ORDER BY order_date;

-- Lag and lead for period-over-period analysis
SELECT 
    DATE_TRUNC('month', order_date) as month,
    SUM(amount) as monthly_revenue,
    LAG(SUM(amount)) OVER (ORDER BY DATE_TRUNC('month', order_date)) as prev_month_revenue,
    (SUM(amount) - LAG(SUM(amount)) OVER (ORDER BY DATE_TRUNC('month', order_date))) / 
    LAG(SUM(amount)) OVER (ORDER BY DATE_TRUNC('month', order_date)) * 100 as growth_rate
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;
```

### Data Loading Patterns

#### 1. Bulk Loading with COPY INTO

```sql
-- Create file format
CREATE FILE FORMAT csv_format
  TYPE = 'CSV'
  FIELD_DELIMITER = ','
  SKIP_HEADER = 1
  NULL_IF = ('NULL', 'null', '')
  EMPTY_FIELD_AS_NULL = TRUE;

-- Create stage (external location)
CREATE STAGE s3_stage
  URL = 's3://your-company-bucket/data/'
  CREDENTIALS = (AWS_KEY_ID = '<your_aws_key_id>' AWS_SECRET_KEY = '<your_aws_secret_key>')
  FILE_FORMAT = csv_format;

-- Load data
COPY INTO orders
FROM @s3_stage/orders.csv
FILE_FORMAT = csv_format
ON_ERROR = 'CONTINUE';

-- Note: Replace 's3://your-company-bucket' with your actual bucket name
-- Use IAM roles instead of hardcoded credentials in production:
-- CREATE STAGE s3_stage
--   URL = 's3://your-company-bucket/data/'
--   STORAGE_INTEGRATION = my_s3_integration;

-- Check load results
SELECT * FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
  TABLE_NAME => 'ORDERS',
  START_TIME => DATEADD(hours, -1, CURRENT_TIMESTAMP())
));
```

#### 2. Streaming with Snowpipe

```sql
-- Create pipe for automatic loading
CREATE PIPE orders_pipe
  AUTO_INGEST = TRUE
  AS COPY INTO orders
  FROM @s3_stage
  FILE_FORMAT = csv_format;

-- Show pipe status
SELECT SYSTEM$PIPE_STATUS('orders_pipe');

-- Refresh pipe (manual trigger)
ALTER PIPE orders_pipe REFRESH;
```

#### 3. Data Transformation During Load

```sql
-- Transform data during COPY
COPY INTO orders (order_id, customer_id, amount, order_date)
FROM (
  SELECT 
    $1::STRING,
    $2::INTEGER,
    $3::DECIMAL(10,2),
    TO_TIMESTAMP($4, 'YYYY-MM-DD HH24:MI:SS')
  FROM @s3_stage/raw_orders.csv
)
FILE_FORMAT = csv_format;

-- Load JSON data
COPY INTO events
FROM (
  SELECT 
    $1:event_id::STRING,
    $1:event_data::VARIANT,
    $1:timestamp::TIMESTAMP_NTZ
  FROM @s3_stage/events.json
)
FILE_FORMAT = (TYPE = 'JSON');
```

### Schema Design Patterns

#### 1. Star Schema

```sql
-- Fact table
CREATE TABLE fact_sales (
    sale_id STRING PRIMARY KEY,
    customer_key INTEGER,
    product_key INTEGER,
    date_key INTEGER,
    store_key INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_amount DECIMAL(10,2),
    discount_amount DECIMAL(10,2)
);

-- Dimension tables
CREATE TABLE dim_customer (
    customer_key INTEGER PRIMARY KEY,
    customer_id STRING,
    customer_name STRING,
    email STRING,
    segment STRING,
    registration_date DATE
);

CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY,
    product_id STRING,
    product_name STRING,
    category STRING,
    subcategory STRING,
    brand STRING,
    unit_cost DECIMAL(10,2)
);

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    date_value DATE,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN
);
```

#### 2. Slowly Changing Dimensions (SCD)

```sql
-- SCD Type 2 - Track history
CREATE TABLE dim_customer_scd2 (
    customer_key INTEGER AUTOINCREMENT PRIMARY KEY,
    customer_id STRING,
    customer_name STRING,
    email STRING,
    segment STRING,
    effective_date DATE,
    expiration_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Update SCD Type 2
MERGE INTO dim_customer_scd2 AS target
USING (
    SELECT customer_id, customer_name, email, segment
    FROM staging_customers
) AS source
ON target.customer_id = source.customer_id AND target.is_current = TRUE
WHEN MATCHED AND (
    target.customer_name != source.customer_name OR
    target.email != source.email OR
    target.segment != source.segment
) THEN UPDATE SET
    expiration_date = CURRENT_DATE - 1,
    is_current = FALSE
WHEN NOT MATCHED THEN INSERT (
    customer_id, customer_name, email, segment, effective_date
) VALUES (
    source.customer_id, source.customer_name, source.email, source.segment, CURRENT_DATE
);

-- Insert new versions for changed records
INSERT INTO dim_customer_scd2 (customer_id, customer_name, email, segment, effective_date)
SELECT s.customer_id, s.customer_name, s.email, s.segment, CURRENT_DATE
FROM staging_customers s
JOIN dim_customer_scd2 d ON s.customer_id = d.customer_id
WHERE d.is_current = FALSE AND d.expiration_date = CURRENT_DATE - 1;
```

### Performance Optimization

#### 1. Clustering Keys

```sql
-- Create table with clustering key
CREATE TABLE large_orders (
    order_id STRING,
    customer_id INTEGER,
    order_date DATE,
    amount DECIMAL(10,2)
) CLUSTER BY (order_date);

-- Add clustering key to existing table
ALTER TABLE orders CLUSTER BY (order_date, customer_id);

-- Check clustering information
SELECT SYSTEM$CLUSTERING_INFORMATION('orders', '(order_date, customer_id)');

-- Automatic clustering (Enterprise edition)
ALTER TABLE orders SET AUTOMATIC_CLUSTERING = TRUE;
```

#### 2. Search Optimization

```sql
-- Enable search optimization for point lookups
ALTER TABLE customers ADD SEARCH OPTIMIZATION;

-- Check search optimization status
SELECT * FROM TABLE(INFORMATION_SCHEMA.SEARCH_OPTIMIZATION_HISTORY(
  TABLE_NAME => 'CUSTOMERS'
));
```

#### 3. Result Caching

```sql
-- Enable result caching (default)
ALTER SESSION SET USE_CACHED_RESULT = TRUE;

-- Disable for specific query
SELECT /*+ NO_CACHE */ * FROM large_table;

-- Check if result came from cache
SELECT LAST_QUERY_ID();
SELECT * FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
WHERE QUERY_ID = LAST_QUERY_ID();
```

### Advanced Features

#### 1. Stored Procedures

```sql
-- JavaScript stored procedure
CREATE OR REPLACE PROCEDURE process_daily_sales(date_param DATE)
RETURNS STRING
LANGUAGE JAVASCRIPT
AS
$$
  var sql_command = `
    INSERT INTO daily_sales_summary
    SELECT 
      '${DATE_PARAM}' as sale_date,
      COUNT(*) as total_orders,
      SUM(amount) as total_revenue,
      AVG(amount) as avg_order_value
    FROM orders 
    WHERE DATE(order_date) = '${DATE_PARAM}'
  `;
  
  var statement = snowflake.createStatement({sqlText: sql_command});
  var result = statement.execute();
  
  return `Processed ${result.getRowCount()} records for ${DATE_PARAM}`;
$$;

-- Call stored procedure
CALL process_daily_sales('2024-01-15');
```

#### 2. User-Defined Functions (UDFs)

```sql
-- SQL UDF
CREATE OR REPLACE FUNCTION calculate_discount(amount DECIMAL, customer_tier STRING)
RETURNS DECIMAL
AS
$$
  CASE 
    WHEN customer_tier = 'GOLD' THEN amount * 0.15
    WHEN customer_tier = 'SILVER' THEN amount * 0.10
    WHEN customer_tier = 'BRONZE' THEN amount * 0.05
    ELSE 0
  END
$$;

-- Use UDF
SELECT 
    order_id,
    amount,
    customer_tier,
    calculate_discount(amount, customer_tier) as discount_amount
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- JavaScript UDF for complex logic
CREATE OR REPLACE FUNCTION parse_user_agent(user_agent STRING)
RETURNS OBJECT
LANGUAGE JAVASCRIPT
AS
$$
  // Simple user agent parsing
  var result = {
    browser: 'Unknown',
    os: 'Unknown',
    device: 'Desktop'
  };
  
  if (USER_AGENT.includes('Chrome')) result.browser = 'Chrome';
  else if (USER_AGENT.includes('Firefox')) result.browser = 'Firefox';
  else if (USER_AGENT.includes('Safari')) result.browser = 'Safari';
  
  if (USER_AGENT.includes('Windows')) result.os = 'Windows';
  else if (USER_AGENT.includes('Mac')) result.os = 'macOS';
  else if (USER_AGENT.includes('Linux')) result.os = 'Linux';
  
  if (USER_AGENT.includes('Mobile')) result.device = 'Mobile';
  
  return result;
$$;
```

#### 3. Streams and Tasks

```sql
-- Create stream to track changes
CREATE STREAM orders_stream ON TABLE orders;

-- Create task to process stream
CREATE TASK process_orders_task
  WAREHOUSE = analytics_wh
  SCHEDULE = '5 MINUTE'
AS
  INSERT INTO orders_summary
  SELECT 
    DATE(order_date) as order_date,
    COUNT(*) as order_count,
    SUM(amount) as total_amount
  FROM orders_stream
  WHERE METADATA$ACTION = 'INSERT'
  GROUP BY DATE(order_date);

-- Start task
ALTER TASK process_orders_task RESUME;

-- Monitor task execution
SELECT * FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
  TASK_NAME => 'PROCESS_ORDERS_TASK'
));
```

### Data Sharing and Collaboration

#### 1. Secure Data Sharing

```sql
-- Create share
CREATE SHARE sales_data_share;

-- Grant usage on database and schema
GRANT USAGE ON DATABASE analytics TO SHARE sales_data_share;
GRANT USAGE ON SCHEMA analytics.public TO SHARE sales_data_share;

-- Grant select on specific tables
GRANT SELECT ON TABLE analytics.public.orders TO SHARE sales_data_share;
GRANT SELECT ON TABLE analytics.public.customers TO SHARE sales_data_share;

-- Add accounts to share
ALTER SHARE sales_data_share ADD ACCOUNTS = ('partner_account_1', 'partner_account_2');

-- Create secure view for sharing
CREATE SECURE VIEW shared_customer_metrics AS
SELECT 
    customer_segment,
    COUNT(*) as customer_count,
    AVG(total_spent) as avg_spent
FROM customers
GROUP BY customer_segment;

GRANT SELECT ON VIEW shared_customer_metrics TO SHARE sales_data_share;
```

#### 2. Data Marketplace

```sql
-- Access shared data from marketplace
CREATE DATABASE weather_data FROM SHARE SFC_SAMPLES.SAMPLE_DATA;

-- Use shared data in analysis
SELECT 
    o.order_date,
    SUM(o.amount) as daily_sales,
    w.temperature,
    w.precipitation
FROM orders o
JOIN weather_data.weather.daily_weather w 
  ON DATE(o.order_date) = w.date
WHERE o.order_date >= '2024-01-01'
GROUP BY o.order_date, w.temperature, w.precipitation
ORDER BY o.order_date;
```

### Cost Optimization

#### 1. Warehouse Management

```sql
-- Right-size warehouses
-- Monitor query performance
SELECT 
    warehouse_name,
    AVG(execution_time) / 1000 as avg_execution_seconds,
    AVG(queued_overload_time) / 1000 as avg_queue_seconds,
    COUNT(*) as query_count
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
GROUP BY warehouse_name
ORDER BY avg_execution_seconds DESC;

-- Auto-suspend aggressive settings
ALTER WAREHOUSE analytics_wh SET AUTO_SUSPEND = 60; -- 1 minute

-- Use multi-cluster for concurrency, not performance
ALTER WAREHOUSE analytics_wh SET 
  MIN_CLUSTER_COUNT = 1
  MAX_CLUSTER_COUNT = 3
  SCALING_POLICY = 'ECONOMY'; -- Favor queuing over scaling
```

#### 2. Storage Optimization

```sql
-- Monitor storage costs
SELECT 
    table_schema,
    table_name,
    active_bytes / (1024*1024*1024) as active_gb,
    time_travel_bytes / (1024*1024*1024) as time_travel_gb,
    failsafe_bytes / (1024*1024*1024) as failsafe_gb
FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS
WHERE deleted = FALSE
ORDER BY active_bytes DESC;

-- Optimize data retention
ALTER TABLE old_logs SET DATA_RETENTION_TIME_IN_DAYS = 1;

-- Drop unused tables and databases
DROP TABLE IF EXISTS temp_analysis;
DROP DATABASE IF EXISTS old_project;
```

### Security and Governance

#### 1. Role-Based Access Control

```sql
-- Create roles
CREATE ROLE data_analyst;
CREATE ROLE data_scientist;
CREATE ROLE data_engineer;

-- Grant privileges
GRANT USAGE ON WAREHOUSE analytics_wh TO ROLE data_analyst;
GRANT USAGE ON DATABASE analytics TO ROLE data_analyst;
GRANT USAGE ON SCHEMA analytics.public TO ROLE data_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics.public TO ROLE data_analyst;

-- Create custom roles with specific access
CREATE ROLE sales_analyst;
GRANT SELECT ON TABLE customers TO ROLE sales_analyst;
GRANT SELECT ON TABLE orders TO ROLE sales_analyst;
GRANT SELECT ON VIEW customer_metrics TO ROLE sales_analyst;

-- Assign roles to users
GRANT ROLE data_analyst TO USER john_doe;
```

#### 2. Row-Level Security

```sql
-- Create row access policy
CREATE ROW ACCESS POLICY customer_policy AS (customer_region STRING) RETURNS BOOLEAN ->
  CASE 
    WHEN CURRENT_ROLE() = 'ADMIN' THEN TRUE
    WHEN CURRENT_ROLE() = 'US_ANALYST' AND customer_region = 'US' THEN TRUE
    WHEN CURRENT_ROLE() = 'EU_ANALYST' AND customer_region = 'EU' THEN TRUE
    ELSE FALSE
  END;

-- Apply policy to table
ALTER TABLE customers ADD ROW ACCESS POLICY customer_policy ON (region);
```

### Security Best Practices

#### 1. Credential Management

```sql
-- ❌ DON'T: Use hardcoded credentials
CREATE STAGE bad_stage
  URL = 's3://bucket/'
  CREDENTIALS = (AWS_KEY_ID = 'AKIAIOSFODNN7EXAMPLE' AWS_SECRET_KEY = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY');

-- ✅ DO: Use storage integrations with IAM roles
CREATE STORAGE INTEGRATION s3_integration
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = S3
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::<account_id>:role/<role_name>'
  STORAGE_ALLOWED_LOCATIONS = ('s3://<your_company_bucket>/');

CREATE STAGE secure_stage
  URL = 's3://<your_company_bucket>/data/'
  STORAGE_INTEGRATION = s3_integration;
```

#### 2. Bucket Naming Security

```sql
-- ❌ DON'T: Use generic bucket names (vulnerable to sniping)
-- 's3://data-bucket/', 's3://my-bucket/', 's3://temp-files/'

-- ✅ DO: Use organization-specific prefixes
-- 's3://yourcompany-data-prod/', 's3://yourcompany-analytics-dev/'
```

### Performance Best Practices

#### 1. Query Optimization

```sql
-- Use appropriate data types
CREATE TABLE optimized_orders (
    order_id STRING,                    -- Use STRING for IDs
    customer_id INTEGER,                -- Use INTEGER for numeric IDs
    amount NUMBER(10,2),               -- Use NUMBER for currency
    order_date DATE,                   -- Use DATE not TIMESTAMP for dates
    created_at TIMESTAMP_NTZ           -- Use TIMESTAMP_NTZ for timestamps
);

-- Avoid SELECT *
SELECT order_id, customer_id, amount  -- Only select needed columns
FROM orders
WHERE order_date >= '2024-01-01';

-- Use LIMIT for exploration
SELECT * FROM large_table LIMIT 1000;

-- Use appropriate JOINs
SELECT o.order_id, c.customer_name
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id  -- Use INNER JOIN when possible
WHERE o.order_date >= CURRENT_DATE - 30;
```

#### 2. Data Loading Best Practices

```sql
-- Use appropriate file sizes (100-250MB compressed)
-- Partition files by date or logical grouping
-- Use COPY INTO with error handling

COPY INTO orders
FROM @s3_stage/orders/
PATTERN = '.*orders_2024.*\.csv'
FILE_FORMAT = csv_format
ON_ERROR = 'SKIP_FILE'
VALIDATION_MODE = 'RETURN_ERRORS';

-- Monitor and optimize loads
SELECT * FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
  TABLE_NAME => 'ORDERS',
  START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())
))
ORDER BY LAST_LOAD_TIME DESC;
```

---

## 💻 Hands-On Exercise

Build a complete data warehouse solution with Snowflake.

**What you'll create**:
1. Multi-layered data warehouse architecture
2. Efficient data loading pipelines
3. Star schema with fact and dimension tables
4. Advanced analytics queries
5. Performance optimization techniques

**Skills practiced**:
- Snowflake SQL and advanced features
- Data warehouse design patterns
- ETL/ELT processes
- Performance tuning
- Cost optimization

See `exercise.sql` for hands-on practice.

---

## 📚 Resources

- [Snowflake Documentation](https://docs.snowflake.com/)
- [Snowflake SQL Reference](https://docs.snowflake.com/en/sql-reference.html)
- [Snowflake Best Practices](https://docs.snowflake.com/en/user-guide/best-practices.html)
- [Snowflake University](https://learn.snowflake.com/) - Free training
- [Snowflake Community](https://community.snowflake.com/)

---

## 🎯 Key Takeaways

- **Snowflake separates compute and storage** - scale independently for cost efficiency
- **Virtual warehouses** provide isolated compute with auto-suspend/resume
- **Time Travel and cloning** enable data recovery and zero-copy environments
- **Semi-structured data support** handles JSON natively with VARIANT type
- **Clustering keys** optimize query performance on large tables
- **Result caching** automatically speeds up repeated queries
- **Data sharing** enables secure collaboration without data movement
- **Multi-cluster warehouses** handle concurrency automatically
- **Streams and tasks** provide change data capture and scheduling
- **Cost optimization** requires right-sizing warehouses and managing retention

---

## 🚀 What's Next?

Tomorrow (Day 5), you'll learn **Change Data Capture (CDC) with Debezium** - capturing real-time changes from databases and streaming them to data warehouses like Snowflake.

**Preview**: CDC bridges operational databases and analytical systems, enabling real-time data warehousing. You'll learn to capture changes from PostgreSQL and stream them to Snowflake for near real-time analytics.

---

## ✅ Before Moving On

- [ ] Understand Snowflake's three-layer architecture
- [ ] Can create and manage virtual warehouses
- [ ] Know how to load and transform data efficiently
- [ ] Can design star schemas and handle SCDs
- [ ] Understand performance optimization techniques
- [ ] Can implement security and governance controls
- [ ] Complete the exercise in `exercise.sql`
- [ ] Review the solution in `solution.sql`
- [ ] Take the quiz in `quiz.md`

**Time**: ~1 hour | **Difficulty**: ⭐⭐⭐ (Intermediate)

Ready to build modern data warehouses! 🏗️