-- Day 4: Data Warehouses - Snowflake Specifics
-- Exercise: Build a Complete Data Warehouse Solution

-- ============================================================================
-- SETUP: Database and Warehouse Configuration
-- ============================================================================

-- Verify Snowflake connection and basic functionality
SELECT CURRENT_VERSION() as snowflake_version;
SELECT CURRENT_USER() as current_user;
SELECT CURRENT_ROLE() as current_role;
SELECT CURRENT_WAREHOUSE() as current_warehouse;

-- Check available warehouses and databases
SHOW WAREHOUSES;
SHOW DATABASES;

-- TODO: Create database and schema
-- Hint: CREATE DATABASE database_name;
-- Hint: CREATE SCHEMA database_name.schema_name;
-- Create a database for our data warehouse
-- Create schemas for different layers (raw, staging, analytics)

-- TODO: Create and configure virtual warehouse
-- Hint: CREATE WAREHOUSE name WITH WAREHOUSE_SIZE='MEDIUM' AUTO_SUSPEND=300 AUTO_RESUME=TRUE;
-- Create a warehouse with appropriate size
-- Configure auto-suspend and auto-resume
-- Set up multi-cluster if needed

-- ============================================================================
-- EXERCISE 1: Data warehouse architecture setup
-- ============================================================================

-- TODO: Create a multi-layered architecture
-- Layer 1: Raw data (bronze layer)
-- Layer 2: Staging/cleaned data (silver layer)  
-- Layer 3: Analytics/business logic (gold layer)

-- Example structure:
-- CREATE DATABASE ecommerce_dw;
-- CREATE SCHEMA ecommerce_dw.raw;
-- CREATE SCHEMA ecommerce_dw.staging;
-- CREATE SCHEMA ecommerce_dw.analytics;

-- TODO: Create file formats for different data types
-- Hint: CREATE FILE FORMAT csv_format TYPE='CSV' FIELD_DELIMITER=',' SKIP_HEADER=1;
-- Hint: CREATE FILE FORMAT json_format TYPE='JSON';
-- CSV format for structured data
-- JSON format for semi-structured data
-- Parquet format for optimized storage

-- TODO: Create external stages
-- Hint: CREATE STAGE stage_name URL='s3://your-company-bucket/path/' FILE_FORMAT=format_name;
-- SECURITY NOTE: Use storage integrations with IAM roles instead of hardcoded credentials
-- Hint: CREATE STORAGE INTEGRATION ... then reference in stage
-- S3 stage for data files
-- Configure credentials and file formats

-- ============================================================================
-- EXERCISE 2: Raw data tables (Bronze Layer)
-- ============================================================================

-- TODO: Create raw data tables to match source systems
-- These tables should mirror the structure of operational databases

-- Raw customers table
CREATE OR REPLACE TABLE raw.customers (
    -- TODO: Define customer fields
    -- Hint: customer_id STRING, name STRING, email STRING, phone STRING
    -- Hint: address_line1 STRING, city STRING, state STRING, country STRING
    -- Hint: created_at TIMESTAMP_NTZ, updated_at TIMESTAMP_NTZ
    customer_id STRING,
    name STRING,
    email STRING
    -- Add remaining fields
);

-- Raw products table  
CREATE OR REPLACE TABLE raw.products (
    -- TODO: Define product fields
    -- Hint: product_id STRING, name STRING, description STRING
    -- Hint: category STRING, price NUMBER(10,2), cost NUMBER(10,2)
    -- Hint: created_at TIMESTAMP_NTZ, updated_at TIMESTAMP_NTZ
    product_id STRING,
    name STRING
    -- Add remaining fields
);

-- Raw orders table
CREATE OR REPLACE TABLE raw.orders (
    -- TODO: Define order fields
    -- Hint: order_id STRING, customer_id STRING, order_date DATE
    -- Hint: status STRING, total_amount NUMBER(10,2)
    -- Hint: created_at TIMESTAMP_NTZ, updated_at TIMESTAMP_NTZ
    order_id STRING,
    customer_id STRING
    -- Add remaining fields
);

-- Raw order items table
CREATE OR REPLACE TABLE raw.order_items (
    -- TODO: Define order item fields
    -- Hint: order_item_id STRING, order_id STRING, product_id STRING
    -- Hint: quantity INTEGER, unit_price NUMBER(10,2), total_price NUMBER(10,2)
    order_item_id STRING,
    order_id STRING
    -- Add remaining fields
);

-- Raw events table (for semi-structured data)
CREATE OR REPLACE TABLE raw.events (
    -- TODO: Define event fields
    -- Hint: event_id STRING, event_type STRING, event_data VARIANT
    -- Hint: user_id STRING, timestamp TIMESTAMP_NTZ
    event_id STRING,
    event_type STRING,
    event_data VARIANT
    -- Add remaining fields
);

-- ============================================================================
-- EXERCISE 3: Data loading patterns
-- ============================================================================

-- TODO: Load sample data into raw tables
-- Use INSERT statements to create sample data
-- Include various scenarios: different customer types, product categories, order statuses

-- Sample customers data
INSERT INTO raw.customers VALUES
-- TODO: Insert sample customer records
-- Include customers from different regions, segments, registration dates

-- Sample products data  
INSERT INTO raw.products VALUES
-- TODO: Insert sample product records
-- Include different categories, price ranges, brands

-- Sample orders data
INSERT INTO raw.orders VALUES
-- TODO: Insert sample order records
-- Include different statuses, date ranges, amounts

-- Sample order items data
INSERT INTO raw.order_items VALUES
-- TODO: Insert sample order item records
-- Link to orders and products created above

-- Sample events data (JSON)
INSERT INTO raw.events VALUES
-- TODO: Insert sample event records with JSON data
-- Include different event types: page_view, add_to_cart, purchase, etc.

-- TODO: Demonstrate COPY INTO for bulk loading
-- Hint: COPY INTO table_name FROM @stage_name/file.csv FILE_FORMAT=format_name ON_ERROR='CONTINUE';
-- Create sample CSV data and load using COPY INTO
-- Show error handling and validation

-- TODO: Set up Snowpipe for streaming data
-- Hint: CREATE PIPE pipe_name AUTO_INGEST=TRUE AS COPY INTO table FROM @stage;
-- Create pipe for automatic data ingestion
-- Configure for real-time data loading

-- ============================================================================
-- EXERCISE 4: Staging layer transformations (Silver Layer)
-- ============================================================================

-- TODO: Create staging tables with cleaned and standardized data
-- Apply data quality rules, standardize formats, handle nulls

-- Staging customers table
CREATE OR REPLACE TABLE staging.customers AS
SELECT 
    -- TODO: Clean and standardize customer data
    -- Standardize phone numbers, email formats
    -- Handle null values appropriately
    -- Add data quality flags
FROM raw.customers;

-- Staging products table
CREATE OR REPLACE TABLE staging.products AS
SELECT
    -- TODO: Clean and standardize product data
    -- Standardize category names
    -- Calculate profit margins
    -- Add product hierarchies
FROM raw.products;

-- Staging orders table
CREATE OR REPLACE TABLE staging.orders AS
SELECT
    -- TODO: Clean and standardize order data
    -- Standardize status values
    -- Add derived fields (order_year, order_month, etc.)
    -- Calculate business metrics
FROM raw.orders;

-- TODO: Handle semi-structured data transformation
-- Extract and flatten JSON data from events table
CREATE OR REPLACE TABLE staging.events AS
SELECT
    event_id,
    event_type,
    -- TODO: Extract fields from JSON
    -- event_data:user_id::INTEGER as user_id,
    -- event_data:page_url::STRING as page_url,
    -- etc.
FROM raw.events;

-- TODO: Create data quality checks
-- Implement checks for completeness, validity, consistency
-- Create summary tables for data quality monitoring

-- ============================================================================
-- EXERCISE 5: Analytics layer - Star schema design (Gold Layer)
-- ============================================================================

-- TODO: Design and create fact tables
-- Main fact table for sales analysis
CREATE OR REPLACE TABLE analytics.fact_sales (
    -- TODO: Define fact table structure
    -- Surrogate keys to dimensions
    -- Measures (quantities, amounts, counts)
    -- Date keys for time-based analysis
);

-- TODO: Create dimension tables

-- Date dimension
CREATE OR REPLACE TABLE analytics.dim_date (
    -- TODO: Create comprehensive date dimension
    -- date_key, date_value, year, quarter, month, day
    -- is_weekend, is_holiday, fiscal_year, etc.
);

-- Customer dimension (SCD Type 2)
CREATE OR REPLACE TABLE analytics.dim_customer (
    -- TODO: Create customer dimension with history tracking
    -- customer_key (surrogate), customer_id (natural)
    -- customer attributes
    -- effective_date, expiration_date, is_current
);

-- Product dimension
CREATE OR REPLACE TABLE analytics.dim_product (
    -- TODO: Create product dimension
    -- product_key (surrogate), product_id (natural)
    -- product hierarchy (category, subcategory, brand)
    -- product attributes
);

-- TODO: Populate dimension tables
-- Load date dimension with date range
-- Load customer dimension from staging
-- Load product dimension from staging

-- TODO: Populate fact table
-- Join staging tables to get dimension keys
-- Calculate measures and aggregations
-- Handle slowly changing dimensions

-- ============================================================================
-- EXERCISE 6: Advanced Snowflake features
-- ============================================================================

-- TODO: Implement Time Travel queries
-- Hint: SELECT * FROM table_name AT(OFFSET => -3600); -- 1 hour ago
-- Hint: SELECT * FROM table_name AT(TIMESTAMP => '2024-01-01 10:00:00'::TIMESTAMP);
-- Query data as it existed at different points in time
-- Compare current vs historical data
-- Demonstrate data recovery scenarios

-- TODO: Create and use clones
-- Hint: CREATE TABLE table_clone CLONE original_table;
-- Hint: CREATE DATABASE db_clone CLONE original_database;
-- Clone tables for development/testing
-- Clone entire schemas or databases
-- Show zero-copy cloning benefits

-- TODO: Implement clustering keys
-- Hint: ALTER TABLE table_name CLUSTER BY (column1, column2);
-- Hint: SELECT SYSTEM$CLUSTERING_INFORMATION('table_name', '(column1)');
-- Add clustering keys to large tables
-- Monitor clustering effectiveness
-- Demonstrate query performance improvements

-- TODO: Create materialized views
-- Create views for common aggregations
-- Set up automatic refresh
-- Compare performance vs regular views

-- ============================================================================
-- EXERCISE 7: Advanced analytics queries
-- ============================================================================

-- TODO: Customer analytics
-- Hint: Use window functions like SUM() OVER (PARTITION BY customer_id)
-- Hint: Use NTILE() for segmentation, LAG() for cohort analysis
-- Customer lifetime value calculation
-- Customer segmentation (RFM analysis)
-- Customer cohort analysis
-- Churn prediction features

-- TODO: Product analytics  
-- Product performance analysis
-- Market basket analysis
-- Product recommendation features
-- Inventory optimization queries

-- TODO: Time series analysis
-- Sales trends and seasonality
-- Moving averages and growth rates
-- Forecasting data preparation
-- Period-over-period comparisons

-- TODO: Advanced window functions
-- Hint: SUM(amount) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING) -- running total
-- Hint: RANK() OVER (PARTITION BY category ORDER BY sales DESC) -- ranking
-- Hint: LAG(sales, 1) OVER (ORDER BY date) -- previous period
-- Running totals and cumulative metrics
-- Ranking and percentile calculations
-- Lead/lag analysis for trends
-- Complex analytical functions

-- ============================================================================
-- EXERCISE 8: Performance optimization
-- ============================================================================

-- TODO: Query performance analysis
-- Hint: Use EXPLAIN to see query execution plan
-- Hint: Check QUERY_HISTORY() for performance metrics
-- Hint: Use appropriate JOIN types (INNER vs LEFT)
-- Use EXPLAIN to analyze query plans
-- Identify performance bottlenecks
-- Optimize JOIN operations
-- Reduce data scanning with clustering

-- TODO: Warehouse sizing optimization
-- Monitor warehouse utilization
-- Test different warehouse sizes
-- Implement auto-scaling strategies
-- Cost vs performance analysis

-- TODO: Storage optimization
-- Analyze table storage metrics
-- Implement data retention policies
-- Optimize data types and compression
-- Remove unused objects

-- ============================================================================
-- EXERCISE 9: Data governance and security
-- ============================================================================

-- TODO: Implement role-based access control
-- Create roles for different user types
-- Grant appropriate permissions
-- Test access controls
-- Implement principle of least privilege

-- TODO: Create secure views
-- Mask sensitive data (PII)
-- Implement row-level security
-- Create aggregated views for sharing
-- Test data privacy controls

-- TODO: Set up data sharing
-- Create shares for external partners
-- Configure secure data sharing
-- Test shared data access
-- Monitor sharing usage

-- ============================================================================
-- EXERCISE 10: Automation and monitoring
-- ============================================================================

-- TODO: Create stored procedures
-- Hint: CREATE PROCEDURE proc_name() RETURNS STRING LANGUAGE JAVASCRIPT AS $$..$$;
-- Hint: Use try/catch blocks for error handling in JavaScript procedures
-- Automate ETL processes
-- Implement error handling
-- Create reusable data processing logic
-- Schedule regular maintenance tasks

-- TODO: Set up streams and tasks
-- Hint: CREATE STREAM stream_name ON TABLE table_name;
-- Hint: CREATE TASK task_name WAREHOUSE=wh SCHEDULE='5 MINUTE' AS INSERT INTO...;
-- Hint: ALTER TASK task_name RESUME; -- to start the task
-- Create streams to track data changes
-- Build tasks for automated processing
-- Implement change data capture patterns
-- Monitor task execution

-- TODO: Implement monitoring and alerting
-- Monitor query performance
-- Track storage and compute costs
-- Set up alerts for data quality issues
-- Create operational dashboards

-- ============================================================================
-- BONUS EXERCISES
-- ============================================================================

-- TODO: Machine learning integration
-- Prepare data for ML models
-- Create feature engineering pipelines
-- Implement model scoring in SQL
-- Set up ML model deployment patterns

-- TODO: Real-time analytics
-- Implement streaming analytics patterns
-- Create real-time dashboards
-- Handle late-arriving data
-- Optimize for low-latency queries

-- TODO: Multi-cloud deployment
-- Design for cloud portability
-- Implement disaster recovery
-- Test cross-cloud data sharing
-- Optimize for multi-region access

-- ============================================================================
-- QUESTIONS TO CONSIDER
-- ============================================================================

/*
1. When would you choose Snowflake over other data warehouse solutions?
   Answer: _______________

2. How does Snowflake's architecture enable cost optimization?
   Answer: _______________

3. What are the trade-offs between clustering and search optimization?
   Answer: _______________

4. How would you implement a real-time data pipeline with Snowflake?
   Answer: _______________

5. What strategies would you use for handling PII data in Snowflake?
   Answer: _______________

6. How do you optimize Snowflake costs while maintaining performance?
   Answer: _______________

7. When would you use Time Travel vs cloning for data recovery?
   Answer: _______________

8. How do you handle schema evolution in a Snowflake data warehouse?
   Answer: _______________
*/

-- ============================================================================
-- CLEANUP (optional)
-- ============================================================================

-- TODO: Clean up resources to avoid costs
-- Drop tables, views, and other objects
-- Suspend or drop warehouses
-- Remove stages and file formats

-- Uncomment to clean up:
-- DROP DATABASE IF EXISTS ecommerce_dw;
-- DROP WAREHOUSE IF EXISTS analytics_wh;