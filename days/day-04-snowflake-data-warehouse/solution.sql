-- Day 4: Data Warehouses - Snowflake Specifics
-- SOLUTION: Build a Complete Data Warehouse Solution

-- ============================================================================
-- SETUP: Database and Warehouse Configuration
-- ============================================================================

-- Solution: Create database and schema
CREATE DATABASE IF NOT EXISTS ecommerce_dw;
USE DATABASE ecommerce_dw;

-- Create schemas for different layers
CREATE SCHEMA IF NOT EXISTS raw;      -- Bronze layer - raw data
CREATE SCHEMA IF NOT EXISTS staging;  -- Silver layer - cleaned data  
CREATE SCHEMA IF NOT EXISTS analytics; -- Gold layer - business logic

-- Solution: Create and configure virtual warehouse
CREATE WAREHOUSE IF NOT EXISTS analytics_wh WITH
  WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 300          -- Suspend after 5 minutes of inactivity
  AUTO_RESUME = TRUE          -- Auto-resume when queries are submitted
  MIN_CLUSTER_COUNT = 1       -- Minimum clusters for concurrency
  MAX_CLUSTER_COUNT = 3       -- Maximum clusters for scaling
  SCALING_POLICY = 'STANDARD' -- Balance between performance and cost
  COMMENT = 'Warehouse for analytics workloads';

-- Use the warehouse
USE WAREHOUSE analytics_wh;

-- ============================================================================
-- EXERCISE 1: Data warehouse architecture setup
-- ============================================================================

-- Solution: Create file formats for different data types
CREATE OR REPLACE FILE FORMAT csv_format
  TYPE = 'CSV'
  FIELD_DELIMITER = ','
  RECORD_DELIMITER = '\n'
  SKIP_HEADER = 1
  NULL_IF = ('NULL', 'null', '', 'N/A')
  EMPTY_FIELD_AS_NULL = TRUE
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  TRIM_SPACE = TRUE;

CREATE OR REPLACE FILE FORMAT json_format
  TYPE = 'JSON'
  STRIP_OUTER_ARRAY = TRUE
  COMMENT = 'Format for JSON data files';

CREATE OR REPLACE FILE FORMAT parquet_format
  TYPE = 'PARQUET'
  COMMENT = 'Format for Parquet data files';

-- Solution: Create internal stage for demo (in production, use external stages)
CREATE OR REPLACE STAGE internal_stage
  FILE_FORMAT = csv_format
  COMMENT = 'Internal stage for demo data';

-- ============================================================================
-- EXERCISE 2: Raw data tables (Bronze Layer)
-- ============================================================================

-- Solution: Raw customers table
CREATE OR REPLACE TABLE raw.customers (
    customer_id INTEGER,
    first_name STRING,
    last_name STRING,
    email STRING,
    phone STRING,
    address_line1 STRING,
    address_line2 STRING,
    city STRING,
    state STRING,
    postal_code STRING,
    country STRING,
    registration_date DATE,
    customer_segment STRING,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Solution: Raw products table
CREATE OR REPLACE TABLE raw.products (
    product_id INTEGER,
    product_name STRING,
    description STRING,
    category STRING,
    subcategory STRING,
    brand STRING,
    unit_price DECIMAL(10,2),
    unit_cost DECIMAL(10,2),
    weight_kg DECIMAL(8,3),
    dimensions_cm STRING,
    color STRING,
    size STRING,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Solution: Raw orders table
CREATE OR REPLACE TABLE raw.orders (
    order_id INTEGER,
    customer_id INTEGER,
    order_date TIMESTAMP_NTZ,
    order_status STRING,
    total_amount DECIMAL(12,2),
    tax_amount DECIMAL(10,2),
    shipping_amount DECIMAL(8,2),
    discount_amount DECIMAL(8,2),
    payment_method STRING,
    shipping_address STRING,
    created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Solution: Raw order items table
CREATE OR REPLACE TABLE raw.order_items (
    order_item_id INTEGER,
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(12,2),
    discount_amount DECIMAL(8,2)
);

-- Solution: Raw events table (for semi-structured data)
CREATE OR REPLACE TABLE raw.events (
    event_id STRING,
    event_type STRING,
    event_data VARIANT,
    user_id INTEGER,
    session_id STRING,
    timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================================================
-- EXERCISE 3: Data loading patterns
-- ============================================================================

-- Solution: Load sample data into raw tables

-- Sample customers data
INSERT INTO raw.customers (customer_id, first_name, last_name, email, phone, address_line1, city, state, postal_code, country, registration_date, customer_segment) VALUES
(1, 'John', 'Doe', 'john.doe@email.com', '+1-555-0101', '123 Main St', 'New York', 'NY', '10001', 'USA', '2023-01-15', 'Premium'),
(2, 'Jane', 'Smith', 'jane.smith@email.com', '+1-555-0102', '456 Oak Ave', 'Los Angeles', 'CA', '90210', 'USA', '2023-02-20', 'Standard'),
(3, 'Bob', 'Johnson', 'bob.johnson@email.com', '+1-555-0103', '789 Pine St', 'Chicago', 'IL', '60601', 'USA', '2023-03-10', 'Premium'),
(4, 'Alice', 'Brown', 'alice.brown@email.com', '+1-555-0104', '321 Elm Dr', 'Houston', 'TX', '77001', 'USA', '2023-04-05', 'Standard'),
(5, 'Charlie', 'Wilson', 'charlie.wilson@email.com', '+1-555-0105', '654 Maple Ln', 'Phoenix', 'AZ', '85001', 'USA', '2023-05-12', 'Basic');

-- Sample products data
INSERT INTO raw.products (product_id, product_name, description, category, subcategory, brand, unit_price, unit_cost, weight_kg, color, size) VALUES
(101, 'Wireless Headphones', 'Premium noise-canceling wireless headphones', 'Electronics', 'Audio', 'TechBrand', 299.99, 150.00, 0.25, 'Black', 'One Size'),
(102, 'Smartphone Case', 'Protective case for smartphones', 'Electronics', 'Accessories', 'ProtectCo', 24.99, 8.00, 0.05, 'Blue', 'iPhone 14'),
(103, 'Running Shoes', 'Lightweight running shoes for athletes', 'Footwear', 'Athletic', 'SportMax', 129.99, 65.00, 0.80, 'White', '10'),
(104, 'Coffee Maker', 'Automatic drip coffee maker', 'Home & Kitchen', 'Appliances', 'BrewMaster', 89.99, 45.00, 2.50, 'Silver', 'Standard'),
(105, 'Yoga Mat', 'Non-slip yoga and exercise mat', 'Sports', 'Fitness', 'FlexFit', 39.99, 15.00, 1.20, 'Purple', '6mm');

-- Sample orders data
INSERT INTO raw.orders (order_id, customer_id, order_date, order_status, total_amount, tax_amount, shipping_amount, discount_amount, payment_method) VALUES
(1001, 1, '2024-01-15 10:30:00', 'Delivered', 324.98, 24.99, 9.99, 0.00, 'Credit Card'),
(1002, 2, '2024-01-16 14:22:00', 'Shipped', 154.98, 11.99, 7.99, 10.00, 'PayPal'),
(1003, 3, '2024-01-17 09:15:00', 'Processing', 219.97, 16.99, 12.99, 0.00, 'Credit Card'),
(1004, 1, '2024-01-18 16:45:00', 'Delivered', 89.99, 6.99, 0.00, 5.00, 'Credit Card'),
(1005, 4, '2024-01-19 11:30:00', 'Cancelled', 39.99, 3.09, 5.99, 0.00, 'Debit Card');

-- Sample order items data
INSERT INTO raw.order_items (order_item_id, order_id, product_id, quantity, unit_price, total_price, discount_amount) VALUES
(1, 1001, 101, 1, 299.99, 299.99, 0.00),
(2, 1001, 102, 1, 24.99, 24.99, 0.00),
(3, 1002, 103, 1, 129.99, 129.99, 10.00),
(4, 1002, 102, 1, 24.99, 24.99, 0.00),
(5, 1003, 101, 1, 299.99, 299.99, 0.00),
(6, 1004, 104, 1, 89.99, 89.99, 5.00),
(7, 1005, 105, 1, 39.99, 39.99, 0.00);

-- Sample events data (JSON)
INSERT INTO raw.events (event_id, event_type, event_data, user_id, session_id) VALUES
('evt_001', 'page_view', PARSE_JSON('{"page_url": "/products/101", "referrer": "google.com", "user_agent": "Chrome/120.0"}'), 1, 'sess_abc123'),
('evt_002', 'add_to_cart', PARSE_JSON('{"product_id": 101, "quantity": 1, "price": 299.99}'), 1, 'sess_abc123'),
('evt_003', 'purchase', PARSE_JSON('{"order_id": 1001, "total_amount": 324.98, "items": [{"product_id": 101, "quantity": 1}, {"product_id": 102, "quantity": 1}]}'), 1, 'sess_abc123'),
('evt_004', 'page_view', PARSE_JSON('{"page_url": "/products/103", "referrer": "direct", "user_agent": "Safari/17.0"}'), 2, 'sess_def456'),
('evt_005', 'search', PARSE_JSON('{"query": "running shoes", "results_count": 15, "filters": {"category": "Footwear"}}'), 2, 'sess_def456');

-- Solution: Demonstrate COPY INTO for bulk loading
-- First, let's create a simple CSV file content simulation
CREATE OR REPLACE TABLE temp_customers_csv (
    customer_id INTEGER,
    first_name STRING,
    last_name STRING,
    email STRING,
    registration_date DATE
);

-- In a real scenario, you would use COPY INTO like this:
/*
COPY INTO raw.customers
FROM @external_stage/customers.csv
FILE_FORMAT = csv_format
ON_ERROR = 'CONTINUE'
VALIDATION_MODE = 'RETURN_ERRORS';
*/

-- ============================================================================
-- EXERCISE 4: Staging layer transformations (Silver Layer)
-- ============================================================================

-- Solution: Staging customers table with data cleaning
CREATE OR REPLACE TABLE staging.customers AS
SELECT 
    customer_id,
    TRIM(UPPER(first_name)) as first_name,
    TRIM(UPPER(last_name)) as last_name,
    LOWER(TRIM(email)) as email,
    REGEXP_REPLACE(phone, '[^0-9+]', '') as phone_cleaned,
    TRIM(address_line1) as address_line1,
    TRIM(city) as city,
    UPPER(TRIM(state)) as state,
    TRIM(postal_code) as postal_code,
    UPPER(TRIM(country)) as country,
    registration_date,
    UPPER(TRIM(customer_segment)) as customer_segment,
    DATEDIFF('day', registration_date, CURRENT_DATE()) as days_since_registration,
    CASE 
        WHEN customer_segment = 'Premium' THEN 'High Value'
        WHEN customer_segment = 'Standard' THEN 'Medium Value'
        ELSE 'Low Value'
    END as customer_value_tier,
    -- Data quality flags
    CASE WHEN email IS NULL OR email = '' THEN 1 ELSE 0 END as missing_email_flag,
    CASE WHEN phone IS NULL OR phone = '' THEN 1 ELSE 0 END as missing_phone_flag,
    created_at,
    updated_at
FROM raw.customers
WHERE customer_id IS NOT NULL;

-- Solution: Staging products table with enhancements
CREATE OR REPLACE TABLE staging.products AS
SELECT 
    product_id,
    TRIM(product_name) as product_name,
    TRIM(description) as description,
    UPPER(TRIM(category)) as category,
    UPPER(TRIM(subcategory)) as subcategory,
    UPPER(TRIM(brand)) as brand,
    unit_price,
    unit_cost,
    ROUND(unit_price - unit_cost, 2) as profit_amount,
    ROUND((unit_price - unit_cost) / unit_price * 100, 2) as profit_margin_pct,
    weight_kg,
    UPPER(TRIM(color)) as color,
    UPPER(TRIM(size)) as size,
    -- Product hierarchy
    CONCAT(category, ' > ', subcategory) as product_hierarchy,
    -- Price tier
    CASE 
        WHEN unit_price >= 200 THEN 'Premium'
        WHEN unit_price >= 50 THEN 'Mid-Range'
        ELSE 'Budget'
    END as price_tier,
    created_at,
    updated_at
FROM raw.products
WHERE product_id IS NOT NULL;

-- Solution: Staging orders table with derived fields
CREATE OR REPLACE TABLE staging.orders AS
SELECT 
    order_id,
    customer_id,
    order_date,
    DATE(order_date) as order_date_only,
    EXTRACT(YEAR FROM order_date) as order_year,
    EXTRACT(MONTH FROM order_date) as order_month,
    EXTRACT(DAY FROM order_date) as order_day,
    DAYOFWEEK(order_date) as order_day_of_week,
    CASE WHEN DAYOFWEEK(order_date) IN (1, 7) THEN 'Weekend' ELSE 'Weekday' END as order_day_type,
    UPPER(TRIM(order_status)) as order_status,
    total_amount,
    tax_amount,
    shipping_amount,
    discount_amount,
    total_amount - tax_amount - shipping_amount as net_product_amount,
    CASE WHEN discount_amount > 0 THEN 1 ELSE 0 END as has_discount_flag,
    UPPER(TRIM(payment_method)) as payment_method,
    -- Order size categories
    CASE 
        WHEN total_amount >= 500 THEN 'Large'
        WHEN total_amount >= 100 THEN 'Medium'
        ELSE 'Small'
    END as order_size_category,
    created_at,
    updated_at
FROM raw.orders
WHERE order_id IS NOT NULL;

-- Solution: Handle semi-structured data transformation
CREATE OR REPLACE TABLE staging.events AS
SELECT
    event_id,
    UPPER(TRIM(event_type)) as event_type,
    user_id,
    session_id,
    timestamp,
    -- Extract common fields from JSON
    event_data:page_url::STRING as page_url,
    event_data:referrer::STRING as referrer,
    event_data:user_agent::STRING as user_agent,
    event_data:product_id::INTEGER as product_id,
    event_data:quantity::INTEGER as quantity,
    event_data:price::DECIMAL(10,2) as price,
    event_data:order_id::INTEGER as order_id,
    event_data:total_amount::DECIMAL(12,2) as total_amount,
    event_data:query::STRING as search_query,
    event_data:results_count::INTEGER as search_results_count,
    -- Keep original JSON for flexibility
    event_data as original_event_data
FROM raw.events
WHERE event_id IS NOT NULL;

-- Solution: Data quality summary
CREATE OR REPLACE TABLE staging.data_quality_summary AS
SELECT 
    'customers' as table_name,
    COUNT(*) as total_records,
    SUM(missing_email_flag) as missing_email_count,
    SUM(missing_phone_flag) as missing_phone_count,
    CURRENT_TIMESTAMP() as last_updated
FROM staging.customers
UNION ALL
SELECT 
    'products' as table_name,
    COUNT(*) as total_records,
    SUM(CASE WHEN product_name IS NULL THEN 1 ELSE 0 END) as missing_name_count,
    SUM(CASE WHEN unit_price IS NULL THEN 1 ELSE 0 END) as missing_price_count,
    CURRENT_TIMESTAMP() as last_updated
FROM staging.products;

-- ============================================================================
-- EXERCISE 5: Analytics layer - Star schema design (Gold Layer)
-- ============================================================================

-- Solution: Date dimension
CREATE OR REPLACE TABLE analytics.dim_date AS
WITH date_range AS (
    SELECT DATEADD(day, SEQ4(), '2020-01-01') as date_value
    FROM TABLE(GENERATOR(ROWCOUNT => 2000)) -- Generate ~5 years of dates
)
SELECT 
    TO_NUMBER(TO_CHAR(date_value, 'YYYYMMDD')) as date_key,
    date_value,
    EXTRACT(YEAR FROM date_value) as year,
    EXTRACT(QUARTER FROM date_value) as quarter,
    EXTRACT(MONTH FROM date_value) as month,
    EXTRACT(DAY FROM date_value) as day,
    DAYOFWEEK(date_value) as day_of_week,
    DAYNAME(date_value) as day_name,
    MONTHNAME(date_value) as month_name,
    CASE WHEN DAYOFWEEK(date_value) IN (1, 7) THEN TRUE ELSE FALSE END as is_weekend,
    CASE WHEN TO_CHAR(date_value, 'MM-DD') IN ('01-01', '07-04', '12-25') THEN TRUE ELSE FALSE END as is_holiday,
    -- Fiscal year (assuming April start)
    CASE 
        WHEN EXTRACT(MONTH FROM date_value) >= 4 THEN EXTRACT(YEAR FROM date_value)
        ELSE EXTRACT(YEAR FROM date_value) - 1
    END as fiscal_year
FROM date_range
WHERE date_value <= '2025-12-31';

-- Solution: Customer dimension (SCD Type 2)
CREATE OR REPLACE TABLE analytics.dim_customer (
    customer_key INTEGER AUTOINCREMENT PRIMARY KEY,
    customer_id INTEGER,
    first_name STRING,
    last_name STRING,
    full_name STRING,
    email STRING,
    phone_cleaned STRING,
    city STRING,
    state STRING,
    country STRING,
    customer_segment STRING,
    customer_value_tier STRING,
    effective_date DATE,
    expiration_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Populate customer dimension
INSERT INTO analytics.dim_customer (
    customer_id, first_name, last_name, full_name, email, phone_cleaned,
    city, state, country, customer_segment, customer_value_tier, effective_date
)
SELECT 
    customer_id,
    first_name,
    last_name,
    CONCAT(first_name, ' ', last_name) as full_name,
    email,
    phone_cleaned,
    city,
    state,
    country,
    customer_segment,
    customer_value_tier,
    registration_date as effective_date
FROM staging.customers;

-- Solution: Product dimension
CREATE OR REPLACE TABLE analytics.dim_product (
    product_key INTEGER AUTOINCREMENT PRIMARY KEY,
    product_id INTEGER,
    product_name STRING,
    description STRING,
    category STRING,
    subcategory STRING,
    brand STRING,
    product_hierarchy STRING,
    unit_price DECIMAL(10,2),
    unit_cost DECIMAL(10,2),
    profit_margin_pct DECIMAL(5,2),
    price_tier STRING,
    color STRING,
    size STRING,
    weight_kg DECIMAL(8,3)
);

-- Populate product dimension
INSERT INTO analytics.dim_product (
    product_id, product_name, description, category, subcategory, brand,
    product_hierarchy, unit_price, unit_cost, profit_margin_pct, price_tier,
    color, size, weight_kg
)
SELECT 
    product_id, product_name, description, category, subcategory, brand,
    product_hierarchy, unit_price, unit_cost, profit_margin_pct, price_tier,
    color, size, weight_kg
FROM staging.products;

-- Solution: Fact sales table
CREATE OR REPLACE TABLE analytics.fact_sales (
    sale_key INTEGER AUTOINCREMENT PRIMARY KEY,
    order_id INTEGER,
    order_item_id INTEGER,
    customer_key INTEGER,
    product_key INTEGER,
    order_date_key INTEGER,
    quantity INTEGER,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(12,2),
    discount_amount DECIMAL(8,2),
    net_price DECIMAL(12,2),
    unit_cost DECIMAL(10,2),
    total_cost DECIMAL(12,2),
    gross_profit DECIMAL(12,2),
    order_total_amount DECIMAL(12,2),
    tax_amount DECIMAL(10,2),
    shipping_amount DECIMAL(8,2),
    payment_method STRING,
    order_status STRING
);

-- Populate fact table
INSERT INTO analytics.fact_sales (
    order_id, order_item_id, customer_key, product_key, order_date_key,
    quantity, unit_price, total_price, discount_amount, net_price,
    unit_cost, total_cost, gross_profit, order_total_amount,
    tax_amount, shipping_amount, payment_method, order_status
)
SELECT 
    o.order_id,
    oi.order_item_id,
    dc.customer_key,
    dp.product_key,
    TO_NUMBER(TO_CHAR(o.order_date, 'YYYYMMDD')) as order_date_key,
    oi.quantity,
    oi.unit_price,
    oi.total_price,
    oi.discount_amount,
    oi.total_price - oi.discount_amount as net_price,
    dp.unit_cost,
    dp.unit_cost * oi.quantity as total_cost,
    (oi.total_price - oi.discount_amount) - (dp.unit_cost * oi.quantity) as gross_profit,
    o.total_amount as order_total_amount,
    o.tax_amount,
    o.shipping_amount,
    o.payment_method,
    o.order_status
FROM staging.orders o
JOIN raw.order_items oi ON o.order_id = oi.order_id
JOIN analytics.dim_customer dc ON o.customer_id = dc.customer_id AND dc.is_current = TRUE
JOIN analytics.dim_product dp ON oi.product_id = dp.product_id;

-- ============================================================================
-- EXERCISE 6: Advanced Snowflake features
-- ============================================================================

-- Solution: Time Travel queries
-- Query data as it existed 1 hour ago
SELECT COUNT(*) as orders_1_hour_ago
FROM analytics.fact_sales AT(OFFSET => -3600);

-- Compare current vs historical data
SELECT 
    'Current' as time_period,
    COUNT(*) as order_count,
    SUM(net_price) as total_revenue
FROM analytics.fact_sales
UNION ALL
SELECT 
    '1 Hour Ago' as time_period,
    COUNT(*) as order_count,
    SUM(net_price) as total_revenue
FROM analytics.fact_sales AT(OFFSET => -3600);

-- Solution: Create and use clones
-- Clone table for development
CREATE TABLE analytics.fact_sales_dev CLONE analytics.fact_sales;

-- Clone entire schema
CREATE SCHEMA analytics_dev CLONE analytics;

-- Solution: Implement clustering keys
-- Add clustering key to fact table for better query performance
ALTER TABLE analytics.fact_sales CLUSTER BY (order_date_key, customer_key);

-- Check clustering information
SELECT SYSTEM$CLUSTERING_INFORMATION('analytics.fact_sales', '(order_date_key, customer_key)');

-- Solution: Create materialized views
CREATE MATERIALIZED VIEW analytics.mv_daily_sales AS
SELECT 
    dd.date_value,
    dd.year,
    dd.month,
    COUNT(DISTINCT fs.order_id) as order_count,
    SUM(fs.quantity) as total_quantity,
    SUM(fs.net_price) as total_revenue,
    SUM(fs.gross_profit) as total_profit,
    AVG(fs.net_price) as avg_order_value
FROM analytics.fact_sales fs
JOIN analytics.dim_date dd ON fs.order_date_key = dd.date_key
GROUP BY dd.date_value, dd.year, dd.month;

-- ============================================================================
-- EXERCISE 7: Advanced analytics queries
-- ============================================================================

-- Solution: Customer Lifetime Value (CLV)
CREATE OR REPLACE VIEW analytics.customer_lifetime_value AS
SELECT 
    dc.customer_id,
    dc.full_name,
    dc.customer_segment,
    COUNT(DISTINCT fs.order_id) as total_orders,
    SUM(fs.net_price) as total_spent,
    AVG(fs.net_price) as avg_order_value,
    SUM(fs.gross_profit) as total_profit_generated,
    MIN(dd.date_value) as first_order_date,
    MAX(dd.date_value) as last_order_date,
    DATEDIFF('day', MIN(dd.date_value), MAX(dd.date_value)) as customer_lifespan_days,
    CASE 
        WHEN DATEDIFF('day', MIN(dd.date_value), MAX(dd.date_value)) > 0 
        THEN SUM(fs.net_price) / DATEDIFF('day', MIN(dd.date_value), MAX(dd.date_value))
        ELSE SUM(fs.net_price)
    END as daily_value
FROM analytics.fact_sales fs
JOIN analytics.dim_customer dc ON fs.customer_key = dc.customer_key
JOIN analytics.dim_date dd ON fs.order_date_key = dd.date_key
WHERE fs.order_status = 'DELIVERED'
GROUP BY dc.customer_id, dc.full_name, dc.customer_segment;

-- Solution: RFM Analysis (Recency, Frequency, Monetary)
CREATE OR REPLACE VIEW analytics.rfm_analysis AS
WITH customer_metrics AS (
    SELECT 
        dc.customer_id,
        dc.full_name,
        MAX(dd.date_value) as last_order_date,
        DATEDIFF('day', MAX(dd.date_value), CURRENT_DATE()) as recency_days,
        COUNT(DISTINCT fs.order_id) as frequency,
        SUM(fs.net_price) as monetary_value
    FROM analytics.fact_sales fs
    JOIN analytics.dim_customer dc ON fs.customer_key = dc.customer_key
    JOIN analytics.dim_date dd ON fs.order_date_key = dd.date_key
    WHERE fs.order_status = 'DELIVERED'
    GROUP BY dc.customer_id, dc.full_name
),
rfm_scores AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days ASC) as recency_score,
        NTILE(5) OVER (ORDER BY frequency DESC) as frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value DESC) as monetary_score
    FROM customer_metrics
)
SELECT *,
    CONCAT(recency_score, frequency_score, monetary_score) as rfm_score,
    CASE 
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 3 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Big Spenders'
        WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 THEN 'Lost Customers'
        ELSE 'Others'
    END as customer_segment_rfm
FROM rfm_scores;

-- Solution: Product Performance Analysis
CREATE OR REPLACE VIEW analytics.product_performance AS
SELECT 
    dp.product_id,
    dp.product_name,
    dp.category,
    dp.brand,
    COUNT(DISTINCT fs.order_id) as orders_containing_product,
    SUM(fs.quantity) as total_quantity_sold,
    SUM(fs.net_price) as total_revenue,
    SUM(fs.gross_profit) as total_profit,
    AVG(fs.unit_price) as avg_selling_price,
    SUM(fs.gross_profit) / SUM(fs.net_price) * 100 as profit_margin_pct,
    RANK() OVER (ORDER BY SUM(fs.net_price) DESC) as revenue_rank,
    RANK() OVER (ORDER BY SUM(fs.quantity) DESC) as quantity_rank
FROM analytics.fact_sales fs
JOIN analytics.dim_product dp ON fs.product_key = dp.product_key
WHERE fs.order_status = 'DELIVERED'
GROUP BY dp.product_id, dp.product_name, dp.category, dp.brand;

-- Solution: Time Series Analysis with Moving Averages
CREATE OR REPLACE VIEW analytics.sales_trends AS
SELECT 
    dd.date_value,
    dd.year,
    dd.month,
    dd.day_name,
    SUM(fs.net_price) as daily_revenue,
    COUNT(DISTINCT fs.order_id) as daily_orders,
    -- 7-day moving average
    AVG(SUM(fs.net_price)) OVER (
        ORDER BY dd.date_value 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as revenue_7day_ma,
    -- Month-over-month growth
    LAG(SUM(fs.net_price), 30) OVER (ORDER BY dd.date_value) as revenue_30_days_ago,
    (SUM(fs.net_price) - LAG(SUM(fs.net_price), 30) OVER (ORDER BY dd.date_value)) / 
    LAG(SUM(fs.net_price), 30) OVER (ORDER BY dd.date_value) * 100 as mom_growth_pct
FROM analytics.fact_sales fs
JOIN analytics.dim_date dd ON fs.order_date_key = dd.date_key
WHERE fs.order_status = 'DELIVERED'
GROUP BY dd.date_value, dd.year, dd.month, dd.day_name
ORDER BY dd.date_value;

-- ============================================================================
-- EXERCISE 8: Performance optimization
-- ============================================================================

-- Solution: Query performance analysis
-- Check query history and performance
SELECT 
    query_id,
    query_text,
    warehouse_name,
    execution_time / 1000 as execution_seconds,
    queued_overload_time / 1000 as queue_seconds,
    bytes_scanned,
    rows_produced
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE start_time >= DATEADD(hour, -24, CURRENT_TIMESTAMP())
  AND warehouse_name = 'ANALYTICS_WH'
ORDER BY execution_time DESC
LIMIT 10;

-- Solution: Storage optimization analysis
SELECT 
    table_schema,
    table_name,
    active_bytes / (1024*1024*1024) as active_gb,
    time_travel_bytes / (1024*1024*1024) as time_travel_gb,
    failsafe_bytes / (1024*1024*1024) as failsafe_gb,
    (active_bytes + time_travel_bytes + failsafe_bytes) / (1024*1024*1024) as total_gb
FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS
WHERE table_schema IN ('RAW', 'STAGING', 'ANALYTICS')
  AND deleted = FALSE
ORDER BY active_bytes DESC;

-- ============================================================================
-- EXERCISE 9: Data governance and security
-- ============================================================================

-- Solution: Role-based access control
CREATE ROLE IF NOT EXISTS data_analyst;
CREATE ROLE IF NOT EXISTS data_scientist;
CREATE ROLE IF NOT EXISTS business_user;

-- Grant warehouse usage
GRANT USAGE ON WAREHOUSE analytics_wh TO ROLE data_analyst;
GRANT USAGE ON WAREHOUSE analytics_wh TO ROLE data_scientist;
GRANT USAGE ON WAREHOUSE analytics_wh TO ROLE business_user;

-- Grant database and schema access
GRANT USAGE ON DATABASE ecommerce_dw TO ROLE data_analyst;
GRANT USAGE ON SCHEMA ecommerce_dw.analytics TO ROLE data_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA ecommerce_dw.analytics TO ROLE data_analyst;
GRANT SELECT ON ALL VIEWS IN SCHEMA ecommerce_dw.analytics TO ROLE data_analyst;

-- Business users only get access to specific views
GRANT USAGE ON DATABASE ecommerce_dw TO ROLE business_user;
GRANT USAGE ON SCHEMA ecommerce_dw.analytics TO ROLE business_user;
GRANT SELECT ON VIEW ecommerce_dw.analytics.customer_lifetime_value TO ROLE business_user;
GRANT SELECT ON VIEW ecommerce_dw.analytics.product_performance TO ROLE business_user;

-- Solution: Create secure views with data masking
CREATE OR REPLACE SECURE VIEW analytics.customer_summary_masked AS
SELECT 
    customer_id,
    CONCAT(LEFT(first_name, 1), '***') as first_name_masked,
    CONCAT(LEFT(last_name, 1), '***') as last_name_masked,
    CONCAT('***@', SPLIT_PART(email, '@', 2)) as email_masked,
    customer_segment,
    customer_value_tier
FROM analytics.dim_customer
WHERE is_current = TRUE;

-- ============================================================================
-- EXERCISE 10: Automation and monitoring
-- ============================================================================

-- Solution: Stored procedure for daily ETL
CREATE OR REPLACE PROCEDURE analytics.sp_daily_etl()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Refresh staging tables
    DELETE FROM staging.customers;
    INSERT INTO staging.customers 
    SELECT * FROM (
        -- Same logic as staging.customers creation
        SELECT customer_id, TRIM(UPPER(first_name)) as first_name, 
               TRIM(UPPER(last_name)) as last_name, LOWER(TRIM(email)) as email,
               -- ... other fields
               created_at, updated_at
        FROM raw.customers 
        WHERE customer_id IS NOT NULL
    );
    
    -- Refresh fact table
    DELETE FROM analytics.fact_sales WHERE order_date_key = TO_NUMBER(TO_CHAR(CURRENT_DATE(), 'YYYYMMDD'));
    INSERT INTO analytics.fact_sales 
    SELECT * FROM (
        -- Same logic as fact_sales population
        SELECT o.order_id, oi.order_item_id, dc.customer_key, dp.product_key,
               TO_NUMBER(TO_CHAR(o.order_date, 'YYYYMMDD')) as order_date_key,
               -- ... other fields
        FROM staging.orders o
        JOIN raw.order_items oi ON o.order_id = oi.order_id
        JOIN analytics.dim_customer dc ON o.customer_id = dc.customer_id AND dc.is_current = TRUE
        JOIN analytics.dim_product dp ON oi.product_id = dp.product_id
        WHERE DATE(o.order_date) = CURRENT_DATE()
    );
    
    RETURN 'Daily ETL completed successfully';
END;
$$;

-- Solution: Create task for automated execution
CREATE OR REPLACE TASK analytics.daily_etl_task
  WAREHOUSE = analytics_wh
  SCHEDULE = 'USING CRON 0 2 * * * UTC'  -- Run daily at 2 AM UTC
AS
  CALL analytics.sp_daily_etl();

-- Start the task (requires ACCOUNTADMIN role)
-- ALTER TASK analytics.daily_etl_task RESUME;

-- Solution: Monitoring query
CREATE OR REPLACE VIEW analytics.etl_monitoring AS
SELECT 
    'fact_sales' as table_name,
    COUNT(*) as record_count,
    MAX(order_date_key) as latest_date_key,
    MIN(order_date_key) as earliest_date_key,
    CURRENT_TIMESTAMP() as last_checked
FROM analytics.fact_sales
UNION ALL
SELECT 
    'dim_customer' as table_name,
    COUNT(*) as record_count,
    NULL as latest_date_key,
    NULL as earliest_date_key,
    CURRENT_TIMESTAMP() as last_checked
FROM analytics.dim_customer;

-- ============================================================================
-- ANSWERS TO QUESTIONS
-- ============================================================================

/*
1. When would you choose Snowflake over other data warehouse solutions?
   Answer: Choose Snowflake when you need:
   - Separation of compute and storage for cost optimization
   - Automatic scaling and concurrency handling
   - Native semi-structured data support (JSON, XML, Parquet)
   - Zero-copy cloning and Time Travel features
   - Secure data sharing capabilities
   - Cloud-native architecture with minimal maintenance
   - Pay-per-use pricing model

2. How does Snowflake's architecture enable cost optimization?
   Answer: 
   - Compute and storage are billed separately
   - Virtual warehouses auto-suspend when not in use
   - Scale compute up/down based on workload needs
   - Multi-cluster warehouses handle concurrency efficiently
   - Result caching reduces compute usage
   - Compression reduces storage costs
   - Time Travel and cloning don't duplicate storage

3. What are the trade-offs between clustering and search optimization?
   Answer:
   - Clustering: Better for range queries, analytical workloads, requires maintenance
   - Search Optimization: Better for point lookups, equality filters, automatic maintenance
   - Clustering: Works on any columns, more control over performance
   - Search Optimization: Limited to specific query patterns, easier to implement
   - Use clustering for analytical queries, search optimization for operational queries

4. How would you implement a real-time data pipeline with Snowflake?
   Answer:
   - Use Snowpipe for automatic data ingestion from cloud storage
   - Implement change data capture (CDC) from source systems
   - Use Streams to track changes in tables
   - Create Tasks for automated processing
   - Use external functions for real-time transformations
   - Implement micro-batch processing patterns
   - Consider Kafka Connect for streaming integration

5. What strategies would you use for handling PII data in Snowflake?
   Answer:
   - Use Dynamic Data Masking for sensitive fields
   - Implement Row-Level Security policies
   - Create secure views with data masking
   - Use column-level security tags
   - Implement role-based access control
   - Encrypt sensitive data at application level
   - Use Snowflake's data classification features
   - Regular access audits and monitoring

6. How do you optimize Snowflake costs while maintaining performance?
   Answer:
   - Right-size virtual warehouses based on workload
   - Use auto-suspend aggressively (1-5 minutes)
   - Implement query result caching
   - Use clustering keys for large tables
   - Optimize data retention policies
   - Monitor and eliminate unused objects
   - Use resource monitors and alerts
   - Implement workload management with separate warehouses

7. When would you use Time Travel vs cloning for data recovery?
   Answer:
   - Time Travel: For querying historical data states, undropping objects, 
     comparing data at different points in time (up to 90 days)
   - Cloning: For creating development/test environments, backup before major changes,
     creating point-in-time snapshots for analysis
   - Time Travel: No additional storage cost for queries
   - Cloning: Zero-copy until data diverges, better for long-term preservation

8. How do you handle schema evolution in a Snowflake data warehouse?
   Answer:
   - Use ALTER TABLE to add/modify columns
   - Implement versioned schemas for major changes
   - Use VARIANT columns for flexible semi-structured data
   - Create migration scripts with proper testing
   - Use Time Travel for rollback capabilities
   - Implement gradual rollout with cloned environments
   - Document all schema changes
   - Use dbt for version-controlled transformations
*/

-- ============================================================================
-- CLEANUP (optional)
-- ============================================================================

-- Uncomment to clean up resources:
-- DROP DATABASE IF EXISTS ecommerce_dw;
-- DROP WAREHOUSE IF EXISTS analytics_wh;