"""
Day 13: dbt Basics - Complete Solution

Complete dbt project for e-commerce analytics with proper layered architecture,
comprehensive testing, and production-ready patterns.
"""

# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

# dbt_project.yml
DBT_PROJECT_CONFIG = """
name: 'ecommerce_analytics'
version: '1.0.0'
config-version: 2

profile: 'ecommerce_analytics'

model-paths: ["models"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

models:
  ecommerce_analytics:
    # Staging models - clean and standardize
    staging:
      +materialized: view
      +schema: staging
    
    # Intermediate models - business logic
    intermediate:
      +materialized: ephemeral
    
    # Mart models - final analytics tables
    marts:
      +materialized: table
      +schema: marts
      core:
        +schema: core
      finance:
        +schema: finance

# Global variables
vars:
  start_date: '2024-01-01'
  timezone: 'UTC'
  
# On-run hooks
on-run-start:
  - "{{ log('Starting dbt run at ' ~ run_started_at, info=true) }}"
  
on-run-end:
  - "{{ log('Completed dbt run at ' ~ run_started_at, info=true) }}"
"""

# packages.yml
PACKAGES_YML = """
packages:
  - package: dbt-labs/dbt_utils
    version: 1.1.1
  - package: calogica/dbt_expectations
    version: 0.10.1
"""

# =============================================================================
# SOURCES CONFIGURATION
# =============================================================================

# models/sources.yml
SOURCES_YML = """
version: 2

sources:
  - name: raw_data
    description: Raw data from production e-commerce systems
    database: ecommerce_db
    schema: raw
    
    tables:
      - name: customers
        description: Customer account information from user management system
        loaded_at_field: _loaded_at
        freshness:
          warn_after: {count: 6, period: hour}
          error_after: {count: 12, period: hour}
        columns:
          - name: id
            description: Primary key for customers
            tests:
              - unique
              - not_null
          - name: email
            description: Customer email address
            tests:
              - not_null
              - unique
          - name: first_name
            description: Customer first name
            tests:
              - not_null
          - name: last_name
            description: Customer last name
            tests:
              - not_null
          - name: created_at
            description: Account creation timestamp
            tests:
              - not_null
      
      - name: orders
        description: Order transactions from order management system
        loaded_at_field: _loaded_at
        freshness:
          warn_after: {count: 1, period: hour}
          error_after: {count: 3, period: hour}
        columns:
          - name: id
            description: Primary key for orders
            tests:
              - unique
              - not_null
          - name: customer_id
            description: Foreign key to customers table
            tests:
              - not_null
              - relationships:
                  to: source('raw_data', 'customers')
                  field: id
          - name: order_date
            description: Date when order was placed
            tests:
              - not_null
          - name: status
            description: Current order status
            tests:
              - not_null
              - accepted_values:
                  values: ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'returned']
          - name: total_amount
            description: Total order amount in cents
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 0
                  inclusive: true
      
      - name: order_items
        description: Individual line items within orders
        columns:
          - name: id
            description: Primary key for order items
            tests:
              - unique
              - not_null
          - name: order_id
            description: Foreign key to orders table
            tests:
              - not_null
              - relationships:
                  to: source('raw_data', 'orders')
                  field: id
          - name: product_id
            description: Foreign key to products table
            tests:
              - not_null
              - relationships:
                  to: source('raw_data', 'products')
                  field: id
          - name: quantity
            description: Quantity of product ordered
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 1
                  inclusive: true
          - name: unit_price
            description: Price per unit in cents
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 0
                  inclusive: true
      
      - name: products
        description: Product catalog from inventory management system
        columns:
          - name: id
            description: Primary key for products
            tests:
              - unique
              - not_null
          - name: name
            description: Product name
            tests:
              - not_null
          - name: category
            description: Product category
            tests:
              - not_null
          - name: price
            description: Current selling price in cents
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 0
                  inclusive: true
          - name: cost
            description: Product cost in cents
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 0
                  inclusive: true
"""

# =============================================================================
# STAGING MODELS
# =============================================================================

# models/staging/stg_customers.sql
STG_CUSTOMERS_SQL = """
-- Staging model: Clean and standardize customer data
-- This model provides a clean, consistent view of customer data

{{ config(
    materialized='view',
    tags=['staging', 'customers']
) }}

with source as (
    select * from {{ source('raw_data', 'customers') }}
),

cleaned as (
    select
        id as customer_id,
        lower(trim(email)) as email,
        initcap(trim(first_name)) as first_name,
        initcap(trim(last_name)) as last_name,
        created_at,
        updated_at,
        
        -- Derived fields
        concat(initcap(trim(first_name)), ' ', initcap(trim(last_name))) as full_name,
        
        -- Data quality flags
        case 
            when email like '%@%.%' then true 
            else false 
        end as is_valid_email,
        
        -- Audit fields
        current_timestamp as _dbt_loaded_at
        
    from source
    where 
        -- Data quality filters
        id is not null
        and email is not null
        and email != ''
        and first_name is not null
        and first_name != ''
        and last_name is not null
        and last_name != ''
        and created_at is not null
)

select * from cleaned
"""

# models/staging/stg_orders.sql
STG_ORDERS_SQL = """
-- Staging model: Clean and standardize order data
-- This model provides a clean view of order transactions

{{ config(
    materialized='view',
    tags=['staging', 'orders']
) }}

with source as (
    select * from {{ source('raw_data', 'orders') }}
),

cleaned as (
    select
        id as order_id,
        customer_id,
        order_date,
        lower(trim(status)) as order_status,
        total_amount,
        created_at,
        updated_at,
        
        -- Derived fields
        extract(year from order_date) as order_year,
        extract(month from order_date) as order_month,
        extract(day from order_date) as order_day,
        extract(dow from order_date) as order_day_of_week,
        
        -- Convert cents to dollars for easier analysis
        round(total_amount / 100.0, 2) as total_amount_usd,
        
        -- Business logic flags
        case 
            when lower(trim(status)) in ('delivered', 'completed') then true
            else false
        end as is_completed_order,
        
        case
            when lower(trim(status)) in ('cancelled', 'returned') then true
            else false
        end as is_cancelled_order,
        
        -- Audit fields
        current_timestamp as _dbt_loaded_at
        
    from source
    where 
        -- Data quality filters
        id is not null
        and customer_id is not null
        and order_date is not null
        and status is not null
        and total_amount is not null
        and total_amount >= 0
        -- Only include orders from valid date range
        and order_date >= '{{ var("start_date") }}'
        and order_date <= current_date
)

select * from cleaned
"""

# models/staging/stg_order_items.sql
STG_ORDER_ITEMS_SQL = """
-- Staging model: Clean and standardize order items data
-- This model provides a clean view of individual line items

{{ config(
    materialized='view',
    tags=['staging', 'order_items']
) }}

with source as (
    select * from {{ source('raw_data', 'order_items') }}
),

cleaned as (
    select
        id as order_item_id,
        order_id,
        product_id,
        quantity,
        unit_price,
        
        -- Calculated fields
        quantity * unit_price as line_total,
        round((quantity * unit_price) / 100.0, 2) as line_total_usd,
        round(unit_price / 100.0, 2) as unit_price_usd,
        
        -- Audit fields
        current_timestamp as _dbt_loaded_at
        
    from source
    where 
        -- Data quality filters
        id is not null
        and order_id is not null
        and product_id is not null
        and quantity is not null
        and quantity > 0
        and unit_price is not null
        and unit_price >= 0
)

select * from cleaned
"""

# models/staging/stg_products.sql
STG_PRODUCTS_SQL = """
-- Staging model: Clean and standardize product data
-- This model provides a clean view of the product catalog

{{ config(
    materialized='view',
    tags=['staging', 'products']
) }}

with source as (
    select * from {{ source('raw_data', 'products') }}
),

cleaned as (
    select
        id as product_id,
        trim(name) as product_name,
        lower(trim(category)) as product_category,
        price,
        cost,
        coalesce(trim(description), 'No description available') as product_description,
        
        -- Calculated fields
        round(price / 100.0, 2) as price_usd,
        round(cost / 100.0, 2) as cost_usd,
        price - cost as profit_per_unit,
        round((price - cost) / 100.0, 2) as profit_per_unit_usd,
        
        -- Profit margin calculation
        {{ calculate_profit_margin('price', 'cost') }} as profit_margin_percent,
        
        -- Business categorization
        case
            when price - cost <= 0 then 'Loss Leader'
            when (price - cost) / nullif(price, 0) < 0.2 then 'Low Margin'
            when (price - cost) / nullif(price, 0) < 0.5 then 'Medium Margin'
            else 'High Margin'
        end as margin_category,
        
        -- Audit fields
        current_timestamp as _dbt_loaded_at
        
    from source
    where 
        -- Data quality filters
        id is not null
        and name is not null
        and name != ''
        and category is not null
        and category != ''
        and price is not null
        and price >= 0
        and cost is not null
        and cost >= 0
)

select * from cleaned
"""

# =============================================================================
# INTERMEDIATE MODELS
# =============================================================================

# models/intermediate/int_order_metrics.sql
INT_ORDER_METRICS_SQL = """
-- Intermediate model: Calculate comprehensive order-level metrics
-- This model joins orders with order items and calculates key metrics

{{ config(
    materialized='ephemeral',
    tags=['intermediate', 'orders']
) }}

with orders as (
    select * from {{ ref('stg_orders') }}
),

order_items as (
    select * from {{ ref('stg_order_items') }}
),

products as (
    select * from {{ ref('stg_products') }}
),

-- Calculate order-level aggregations from line items
order_item_metrics as (
    select
        oi.order_id,
        count(*) as item_count,
        sum(oi.quantity) as total_quantity,
        sum(oi.line_total) as calculated_order_total,
        avg(oi.unit_price) as avg_item_price,
        min(oi.unit_price) as min_item_price,
        max(oi.unit_price) as max_item_price,
        
        -- Profit calculations
        sum(oi.quantity * p.cost) as total_cost,
        sum(oi.line_total) - sum(oi.quantity * p.cost) as total_profit,
        
        -- Product diversity
        count(distinct p.product_category) as category_count,
        count(distinct oi.product_id) as unique_product_count
        
    from order_items oi
    left join products p on oi.product_id = p.product_id
    group by oi.order_id
),

-- Combine order data with calculated metrics
final as (
    select
        o.order_id,
        o.customer_id,
        o.order_date,
        o.order_status,
        o.total_amount,
        o.total_amount_usd,
        o.order_year,
        o.order_month,
        o.order_day_of_week,
        o.is_completed_order,
        o.is_cancelled_order,
        
        -- Order item metrics
        coalesce(oim.item_count, 0) as item_count,
        coalesce(oim.total_quantity, 0) as total_quantity,
        coalesce(oim.calculated_order_total, 0) as calculated_order_total,
        coalesce(oim.avg_item_price, 0) as avg_item_price,
        coalesce(oim.min_item_price, 0) as min_item_price,
        coalesce(oim.max_item_price, 0) as max_item_price,
        coalesce(oim.category_count, 0) as category_count,
        coalesce(oim.unique_product_count, 0) as unique_product_count,
        
        -- Profit metrics
        coalesce(oim.total_cost, 0) as total_cost,
        coalesce(oim.total_profit, 0) as total_profit,
        round(coalesce(oim.total_cost, 0) / 100.0, 2) as total_cost_usd,
        round(coalesce(oim.total_profit, 0) / 100.0, 2) as total_profit_usd,
        
        -- Profit margin
        case 
            when o.total_amount > 0 then 
                round((coalesce(oim.total_profit, 0) / o.total_amount) * 100, 2)
            else 0
        end as profit_margin_percent,
        
        -- Order size categorization
        case
            when o.total_amount_usd < 50 then 'Small'
            when o.total_amount_usd < 200 then 'Medium'
            when o.total_amount_usd < 500 then 'Large'
            else 'Extra Large'
        end as order_size_category,
        
        -- Data quality flag
        case
            when abs(o.total_amount - coalesce(oim.calculated_order_total, 0)) <= 1 then true
            else false
        end as totals_match
        
    from orders o
    left join order_item_metrics oim on o.order_id = oim.order_id
)

select * from final
"""

# models/intermediate/int_customer_metrics.sql
INT_CUSTOMER_METRICS_SQL = """
-- Intermediate model: Calculate customer lifetime value and behavior metrics
-- This model provides comprehensive customer analytics

{{ config(
    materialized='ephemeral',
    tags=['intermediate', 'customers']
) }}

with customers as (
    select * from {{ ref('stg_customers') }}
),

order_metrics as (
    select * from {{ ref('int_order_metrics') }}
    where is_completed_order = true  -- Only completed orders for LTV
),

-- Calculate customer order statistics
customer_order_stats as (
    select
        customer_id,
        count(*) as total_orders,
        sum(total_amount) as total_spent,
        avg(total_amount) as avg_order_value,
        min(order_date) as first_order_date,
        max(order_date) as last_order_date,
        sum(total_quantity) as total_items_purchased,
        sum(item_count) as total_line_items,
        avg(item_count) as avg_items_per_order,
        sum(total_profit) as total_profit_generated,
        
        -- Frequency metrics
        count(distinct order_year) as years_active,
        count(distinct order_month) as months_active,
        
        -- Product diversity
        sum(unique_product_count) as total_unique_products,
        sum(category_count) as total_categories_purchased
        
    from order_metrics
    group by customer_id
),

-- Calculate recency and frequency
customer_behavior as (
    select
        cos.*,
        
        -- Recency (days since last order)
        current_date - cos.last_order_date as days_since_last_order,
        
        -- Customer lifespan
        cos.last_order_date - cos.first_order_date as customer_lifespan_days,
        
        -- Order frequency (orders per month)
        case 
            when cos.last_order_date - cos.first_order_date > 0 then
                round(cos.total_orders::numeric / 
                     (extract(days from cos.last_order_date - cos.first_order_date) / 30.0), 2)
            else cos.total_orders
        end as orders_per_month,
        
        -- Convert to USD
        round(cos.total_spent / 100.0, 2) as total_spent_usd,
        round(cos.avg_order_value / 100.0, 2) as avg_order_value_usd,
        round(cos.total_profit_generated / 100.0, 2) as total_profit_generated_usd
        
    from customer_order_stats cos
),

-- Add customer segmentation
final as (
    select
        c.customer_id,
        c.email,
        c.first_name,
        c.last_name,
        c.full_name,
        c.created_at as account_created_at,
        
        -- Order metrics
        coalesce(cb.total_orders, 0) as lifetime_orders,
        coalesce(cb.total_spent_usd, 0) as lifetime_value_usd,
        coalesce(cb.avg_order_value_usd, 0) as avg_order_value_usd,
        coalesce(cb.total_items_purchased, 0) as total_items_purchased,
        coalesce(cb.total_profit_generated_usd, 0) as total_profit_generated_usd,
        
        -- Behavioral metrics
        cb.first_order_date,
        cb.last_order_date,
        coalesce(cb.days_since_last_order, 9999) as days_since_last_order,
        coalesce(cb.customer_lifespan_days, 0) as customer_lifespan_days,
        coalesce(cb.orders_per_month, 0) as orders_per_month,
        coalesce(cb.years_active, 0) as years_active,
        
        -- Product engagement
        coalesce(cb.total_unique_products, 0) as total_unique_products,
        coalesce(cb.total_categories_purchased, 0) as total_categories_purchased,
        
        -- Customer segmentation (RFM-inspired)
        case
            when cb.total_spent_usd >= 1000 and cb.days_since_last_order <= 90 then 'Champions'
            when cb.total_spent_usd >= 500 and cb.days_since_last_order <= 180 then 'Loyal Customers'
            when cb.total_orders >= 5 and cb.days_since_last_order <= 365 then 'Potential Loyalists'
            when cb.total_orders = 1 and cb.days_since_last_order <= 90 then 'New Customers'
            when cb.total_spent_usd >= 500 and cb.days_since_last_order > 365 then 'At Risk'
            when cb.days_since_last_order > 365 then 'Hibernating'
            when cb.total_orders = 1 and cb.days_since_last_order > 90 then 'Lost'
            else 'Others'
        end as customer_segment,
        
        -- Value tier
        case
            when cb.total_spent_usd >= 2000 then 'High Value'
            when cb.total_spent_usd >= 500 then 'Medium Value'
            when cb.total_spent_usd > 0 then 'Low Value'
            else 'No Purchases'
        end as value_tier,
        
        -- Activity status
        case
            when cb.days_since_last_order <= 30 then 'Very Active'
            when cb.days_since_last_order <= 90 then 'Active'
            when cb.days_since_last_order <= 180 then 'Moderately Active'
            when cb.days_since_last_order <= 365 then 'Inactive'
            else 'Very Inactive'
        end as activity_status
        
    from customers c
    left join customer_behavior cb on c.customer_id = cb.customer_id
)

select * from final
"""

# models/intermediate/int_product_performance.sql
INT_PRODUCT_PERFORMANCE_SQL = """
-- Intermediate model: Calculate product performance metrics
-- This model provides comprehensive product analytics

{{ config(
    materialized='ephemeral',
    tags=['intermediate', 'products']
) }}

with products as (
    select * from {{ ref('stg_products') }}
),

order_items as (
    select * from {{ ref('stg_order_items') }}
),

orders as (
    select * from {{ ref('stg_orders') }}
    where is_completed_order = true  -- Only completed orders
),

-- Calculate product sales metrics
product_sales as (
    select
        oi.product_id,
        count(distinct oi.order_id) as orders_containing_product,
        count(distinct o.customer_id) as unique_customers,
        sum(oi.quantity) as total_quantity_sold,
        sum(oi.line_total) as total_revenue,
        avg(oi.quantity) as avg_quantity_per_order,
        min(o.order_date) as first_sale_date,
        max(o.order_date) as last_sale_date,
        
        -- Time-based metrics
        count(distinct o.order_year) as years_sold,
        count(distinct o.order_month) as months_sold
        
    from order_items oi
    inner join orders o on oi.order_id = o.order_id
    group by oi.product_id
),

-- Calculate product rankings
product_rankings as (
    select
        product_id,
        row_number() over (order by total_revenue desc) as revenue_rank,
        row_number() over (order by total_quantity_sold desc) as quantity_rank,
        row_number() over (order by orders_containing_product desc) as popularity_rank
    from product_sales
),

-- Combine all metrics
final as (
    select
        p.product_id,
        p.product_name,
        p.product_category,
        p.price_usd,
        p.cost_usd,
        p.profit_per_unit_usd,
        p.profit_margin_percent,
        p.margin_category,
        
        -- Sales metrics
        coalesce(ps.orders_containing_product, 0) as orders_containing_product,
        coalesce(ps.unique_customers, 0) as unique_customers,
        coalesce(ps.total_quantity_sold, 0) as total_quantity_sold,
        round(coalesce(ps.total_revenue, 0) / 100.0, 2) as total_revenue_usd,
        coalesce(ps.avg_quantity_per_order, 0) as avg_quantity_per_order,
        ps.first_sale_date,
        ps.last_sale_date,
        coalesce(ps.years_sold, 0) as years_sold,
        coalesce(ps.months_sold, 0) as months_sold,
        
        -- Calculated metrics
        round((coalesce(ps.total_quantity_sold, 0) * p.profit_per_unit_usd), 2) as total_profit_usd,
        
        -- Performance indicators
        case
            when ps.total_quantity_sold > 0 then
                round(coalesce(ps.total_revenue, 0) / ps.total_quantity_sold / 100.0, 2)
            else p.price_usd
        end as avg_selling_price_usd,
        
        -- Velocity metrics (sales per month)
        case
            when ps.months_sold > 0 then
                round(coalesce(ps.total_quantity_sold, 0)::numeric / ps.months_sold, 2)
            else 0
        end as avg_monthly_sales,
        
        -- Customer penetration
        case
            when ps.unique_customers > 0 and ps.orders_containing_product > 0 then
                round(ps.orders_containing_product::numeric / ps.unique_customers, 2)
            else 0
        end as repeat_purchase_rate,
        
        -- Rankings
        coalesce(pr.revenue_rank, 999999) as revenue_rank,
        coalesce(pr.quantity_rank, 999999) as quantity_rank,
        coalesce(pr.popularity_rank, 999999) as popularity_rank,
        
        -- Performance categories
        case
            when ps.total_quantity_sold = 0 then 'No Sales'
            when ps.total_quantity_sold < 10 then 'Low Performer'
            when ps.total_quantity_sold < 100 then 'Medium Performer'
            when ps.total_quantity_sold < 500 then 'High Performer'
            else 'Top Performer'
        end as performance_category,
        
        -- Lifecycle stage
        case
            when ps.first_sale_date is null then 'Never Sold'
            when current_date - ps.last_sale_date <= 30 then 'Active'
            when current_date - ps.last_sale_date <= 90 then 'Declining'
            when current_date - ps.last_sale_date <= 180 then 'Inactive'
            else 'Discontinued'
        end as lifecycle_stage
        
    from products p
    left join product_sales ps on p.product_id = ps.product_id
    left join product_rankings pr on p.product_id = pr.product_id
)

select * from final
"""

# =============================================================================
# MART MODELS (DIMENSIONAL MODEL)
# =============================================================================

# models/marts/core/dim_customers.sql
DIM_CUSTOMERS_SQL = """
-- Customer dimension table with comprehensive customer attributes and metrics
-- This is the primary customer dimension for analytics and reporting

{{ config(
    materialized='table',
    tags=['marts', 'dimensions', 'customers'],
    indexes=[
      {'columns': ['customer_id'], 'unique': True},
      {'columns': ['email'], 'unique': True},
      {'columns': ['customer_segment']},
      {'columns': ['value_tier']}
    ]
) }}

with customer_metrics as (
    select * from {{ ref('int_customer_metrics') }}
),

final as (
    select
        -- Primary key
        customer_id,
        
        -- Customer attributes
        email,
        first_name,
        last_name,
        full_name,
        account_created_at,
        
        -- Lifetime metrics
        lifetime_orders,
        lifetime_value_usd,
        avg_order_value_usd,
        total_items_purchased,
        total_profit_generated_usd,
        
        -- Behavioral metrics
        first_order_date,
        last_order_date,
        days_since_last_order,
        customer_lifespan_days,
        orders_per_month,
        years_active,
        
        -- Product engagement
        total_unique_products,
        total_categories_purchased,
        
        -- Segmentation
        customer_segment,
        value_tier,
        activity_status,
        
        -- Derived flags
        case when lifetime_orders > 0 then true else false end as has_purchased,
        case when lifetime_orders > 1 then true else false end as is_repeat_customer,
        case when days_since_last_order <= 90 then true else false end as is_active_customer,
        case when lifetime_value_usd >= 500 then true else false end as is_high_value,
        
        -- Audit fields
        current_timestamp as _dbt_updated_at
        
    from customer_metrics
)

select * from final
"""

# models/marts/core/dim_products.sql
DIM_PRODUCTS_SQL = """
-- Product dimension table with comprehensive product attributes and performance metrics
-- This is the primary product dimension for analytics and reporting

{{ config(
    materialized='table',
    tags=['marts', 'dimensions', 'products'],
    indexes=[
      {'columns': ['product_id'], 'unique': True},
      {'columns': ['product_category']},
      {'columns': ['performance_category']},
      {'columns': ['lifecycle_stage']}
    ]
) }}

with product_performance as (
    select * from {{ ref('int_product_performance') }}
),

final as (
    select
        -- Primary key
        product_id,
        
        -- Product attributes
        product_name,
        product_category,
        price_usd,
        cost_usd,
        profit_per_unit_usd,
        profit_margin_percent,
        margin_category,
        
        -- Performance metrics
        orders_containing_product,
        unique_customers,
        total_quantity_sold,
        total_revenue_usd,
        total_profit_usd,
        avg_quantity_per_order,
        avg_selling_price_usd,
        avg_monthly_sales,
        repeat_purchase_rate,
        
        -- Time metrics
        first_sale_date,
        last_sale_date,
        years_sold,
        months_sold,
        
        -- Rankings
        revenue_rank,
        quantity_rank,
        popularity_rank,
        
        -- Categories
        performance_category,
        lifecycle_stage,
        
        -- Derived flags
        case when total_quantity_sold > 0 then true else false end as has_sales,
        case when profit_margin_percent > 20 then true else false end as is_high_margin,
        case when revenue_rank <= 100 then true else false end as is_top_revenue_product,
        case when lifecycle_stage = 'Active' then true else false end as is_active_product,
        
        -- Audit fields
        current_timestamp as _dbt_updated_at
        
    from product_performance
)

select * from final
"""

# models/marts/core/fct_orders.sql
FCT_ORDERS_SQL = """
-- Order fact table containing all order transactions with comprehensive metrics
-- This is the primary fact table for order analysis

{{ config(
    materialized='incremental',
    unique_key='order_id',
    on_schema_change='fail',
    tags=['marts', 'facts', 'orders'],
    indexes=[
      {'columns': ['order_id'], 'unique': True},
      {'columns': ['customer_id']},
      {'columns': ['order_date']},
      {'columns': ['order_status']}
    ]
) }}

with order_metrics as (
    select * from {{ ref('int_order_metrics') }}
    
    {% if is_incremental() %}
        -- Only process new or updated orders
        where order_date > (select max(order_date) from {{ this }})
    {% endif %}
),

final as (
    select
        -- Primary key
        order_id,
        
        -- Foreign keys
        customer_id,
        
        -- Order attributes
        order_date,
        order_status,
        order_year,
        order_month,
        order_day_of_week,
        
        -- Order metrics (additive measures)
        total_amount,
        total_amount_usd,
        item_count,
        total_quantity,
        total_cost,
        total_profit,
        total_cost_usd,
        total_profit_usd,
        
        -- Calculated measures
        avg_item_price,
        min_item_price,
        max_item_price,
        profit_margin_percent,
        
        -- Product diversity
        category_count,
        unique_product_count,
        
        -- Order categorization
        order_size_category,
        
        -- Status flags
        is_completed_order,
        is_cancelled_order,
        
        -- Data quality
        totals_match,
        
        -- Audit fields
        current_timestamp as _dbt_updated_at
        
    from order_metrics
)

select * from final
"""

# models/marts/finance/revenue_summary.sql
REVENUE_SUMMARY_SQL = """
-- Daily revenue summary for finance team with comprehensive metrics
-- This table provides aggregated revenue data for financial reporting

{{ config(
    materialized='table',
    tags=['marts', 'finance', 'revenue'],
    indexes=[
      {'columns': ['revenue_date'], 'unique': True},
      {'columns': ['revenue_year', 'revenue_month']},
      {'columns': ['revenue_week']}
    ]
) }}

with orders as (
    select * from {{ ref('fct_orders') }}
    where is_completed_order = true  -- Only completed orders for revenue
),

products as (
    select * from {{ ref('dim_products') }}
),

-- Daily revenue aggregation
daily_revenue as (
    select
        order_date as revenue_date,
        extract(year from order_date) as revenue_year,
        extract(month from order_date) as revenue_month,
        extract(week from order_date) as revenue_week,
        extract(dow from order_date) as revenue_day_of_week,
        
        -- Order metrics
        count(*) as total_orders,
        count(distinct customer_id) as unique_customers,
        sum(item_count) as total_items_sold,
        sum(total_quantity) as total_quantity_sold,
        
        -- Revenue metrics
        sum(total_amount_usd) as total_revenue_usd,
        avg(total_amount_usd) as avg_order_value_usd,
        min(total_amount_usd) as min_order_value_usd,
        max(total_amount_usd) as max_order_value_usd,
        
        -- Cost and profit metrics
        sum(total_cost_usd) as total_cost_usd,
        sum(total_profit_usd) as total_profit_usd,
        avg(profit_margin_percent) as avg_profit_margin_percent,
        
        -- Order size distribution
        sum(case when order_size_category = 'Small' then 1 else 0 end) as small_orders,
        sum(case when order_size_category = 'Medium' then 1 else 0 end) as medium_orders,
        sum(case when order_size_category = 'Large' then 1 else 0 end) as large_orders,
        sum(case when order_size_category = 'Extra Large' then 1 else 0 end) as extra_large_orders
        
    from orders
    group by order_date
),

-- Add period-over-period comparisons
with_comparisons as (
    select
        *,
        
        -- Previous day comparison
        lag(total_revenue_usd, 1) over (order by revenue_date) as prev_day_revenue_usd,
        total_revenue_usd - lag(total_revenue_usd, 1) over (order by revenue_date) as revenue_change_usd,
        
        -- Week-over-week comparison
        lag(total_revenue_usd, 7) over (order by revenue_date) as prev_week_revenue_usd,
        
        -- Month-to-date calculations
        sum(total_revenue_usd) over (
            partition by revenue_year, revenue_month 
            order by revenue_date 
            rows unbounded preceding
        ) as mtd_revenue_usd,
        
        -- Year-to-date calculations
        sum(total_revenue_usd) over (
            partition by revenue_year 
            order by revenue_date 
            rows unbounded preceding
        ) as ytd_revenue_usd,
        
        -- Rolling averages
        avg(total_revenue_usd) over (
            order by revenue_date 
            rows between 6 preceding and current row
        ) as revenue_7day_avg_usd,
        
        avg(total_revenue_usd) over (
            order by revenue_date 
            rows between 29 preceding and current row
        ) as revenue_30day_avg_usd
        
    from daily_revenue
),

final as (
    select
        -- Date dimensions
        revenue_date,
        revenue_year,
        revenue_month,
        revenue_week,
        revenue_day_of_week,
        
        -- Order metrics
        total_orders,
        unique_customers,
        total_items_sold,
        total_quantity_sold,
        
        -- Revenue metrics
        total_revenue_usd,
        avg_order_value_usd,
        min_order_value_usd,
        max_order_value_usd,
        
        -- Profitability
        total_cost_usd,
        total_profit_usd,
        round(avg_profit_margin_percent, 2) as avg_profit_margin_percent,
        round((total_profit_usd / nullif(total_revenue_usd, 0)) * 100, 2) as daily_profit_margin_percent,
        
        -- Order distribution
        small_orders,
        medium_orders,
        large_orders,
        extra_large_orders,
        
        -- Comparisons
        prev_day_revenue_usd,
        revenue_change_usd,
        round(((revenue_change_usd / nullif(prev_day_revenue_usd, 0)) * 100), 2) as revenue_change_percent,
        
        prev_week_revenue_usd,
        round(((total_revenue_usd - prev_week_revenue_usd) / nullif(prev_week_revenue_usd, 0)) * 100, 2) as wow_revenue_change_percent,
        
        -- Cumulative metrics
        mtd_revenue_usd,
        ytd_revenue_usd,
        
        -- Moving averages
        round(revenue_7day_avg_usd, 2) as revenue_7day_avg_usd,
        round(revenue_30day_avg_usd, 2) as revenue_30day_avg_usd,
        
        -- Performance indicators
        case
            when total_revenue_usd > revenue_7day_avg_usd * 1.2 then 'Above Average'
            when total_revenue_usd < revenue_7day_avg_usd * 0.8 then 'Below Average'
            else 'Average'
        end as daily_performance,
        
        -- Audit fields
        current_timestamp as _dbt_updated_at
        
    from with_comparisons
)

select * from final
"""

# =============================================================================
# TESTING CONFIGURATION
# =============================================================================

# models/schema.yml
SCHEMA_YML = """
version: 2

models:
  # Staging models
  - name: stg_customers
    description: Cleaned and standardized customer data from raw source
    columns:
      - name: customer_id
        description: Unique customer identifier
        tests:
          - unique
          - not_null
      - name: email
        description: Customer email address (cleaned and lowercased)
        tests:
          - unique
          - not_null
      - name: full_name
        description: Customer full name (first + last)
        tests:
          - not_null
      - name: is_valid_email
        description: Flag indicating if email format is valid
        tests:
          - not_null
          - accepted_values:
              values: [true, false]

  - name: stg_orders
    description: Cleaned and standardized order data from raw source
    columns:
      - name: order_id
        description: Unique order identifier
        tests:
          - unique
          - not_null
      - name: customer_id
        description: Foreign key to customers
        tests:
          - not_null
          - relationships:
              to: ref('stg_customers')
              field: customer_id
      - name: order_status
        description: Current order status (standardized)
        tests:
          - not_null
          - accepted_values:
              values: ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'returned']
      - name: total_amount_usd
        description: Total order amount in USD
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              inclusive: true

  - name: stg_order_items
    description: Cleaned order line items with calculated totals
    columns:
      - name: order_item_id
        description: Unique order item identifier
        tests:
          - unique
          - not_null
      - name: order_id
        description: Foreign key to orders
        tests:
          - not_null
          - relationships:
              to: ref('stg_orders')
              field: order_id
      - name: product_id
        description: Foreign key to products
        tests:
          - not_null
          - relationships:
              to: ref('stg_products')
              field: product_id
      - name: line_total_usd
        description: Line total in USD (quantity * unit_price)
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              inclusive: true

  - name: stg_products
    description: Cleaned product catalog with calculated metrics
    columns:
      - name: product_id
        description: Unique product identifier
        tests:
          - unique
          - not_null
      - name: product_name
        description: Product name (trimmed)
        tests:
          - not_null
      - name: product_category
        description: Product category (standardized)
        tests:
          - not_null
      - name: profit_margin_percent
        description: Profit margin as percentage
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: -100
              max_value: 1000

  # Mart models
  - name: dim_customers
    description: Customer dimension with lifetime metrics and segmentation
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - customer_id
            - email
    columns:
      - name: customer_id
        description: Unique customer identifier (primary key)
        tests:
          - unique
          - not_null
      - name: email
        description: Customer email address
        tests:
          - unique
          - not_null
      - name: lifetime_value_usd
        description: Total customer lifetime value in USD
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              inclusive: true
      - name: customer_segment
        description: Customer segment based on RFM analysis
        tests:
          - not_null
          - accepted_values:
              values: ['Champions', 'Loyal Customers', 'Potential Loyalists', 'New Customers', 'At Risk', 'Hibernating', 'Lost', 'Others']
      - name: value_tier
        description: Customer value tier
        tests:
          - not_null
          - accepted_values:
              values: ['High Value', 'Medium Value', 'Low Value', 'No Purchases']

  - name: dim_products
    description: Product dimension with performance metrics and categorization
    columns:
      - name: product_id
        description: Unique product identifier (primary key)
        tests:
          - unique
          - not_null
      - name: product_name
        description: Product name
        tests:
          - not_null
      - name: total_revenue_usd
        description: Total revenue generated by product
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              inclusive: true
      - name: performance_category
        description: Product performance category
        tests:
          - not_null
          - accepted_values:
              values: ['No Sales', 'Low Performer', 'Medium Performer', 'High Performer', 'Top Performer']
      - name: lifecycle_stage
        description: Product lifecycle stage
        tests:
          - not_null
          - accepted_values:
              values: ['Never Sold', 'Active', 'Declining', 'Inactive', 'Discontinued']

  - name: fct_orders
    description: Order fact table with comprehensive order metrics
    columns:
      - name: order_id
        description: Unique order identifier (primary key)
        tests:
          - unique
          - not_null
      - name: customer_id
        description: Foreign key to dim_customers
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
      - name: total_amount_usd
        description: Total order amount in USD
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              inclusive: true
      - name: total_profit_usd
        description: Total profit from order in USD
        tests:
          - not_null
      - name: totals_match
        description: Flag indicating if order total matches calculated total
        tests:
          - not_null
          - accepted_values:
              values: [true, false]

  - name: revenue_summary
    description: Daily revenue summary for financial reporting
    columns:
      - name: revenue_date
        description: Revenue date (primary key)
        tests:
          - unique
          - not_null
      - name: total_revenue_usd
        description: Total daily revenue in USD
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              inclusive: true
      - name: total_profit_usd
        description: Total daily profit in USD
        tests:
          - not_null
      - name: daily_profit_margin_percent
        description: Daily profit margin percentage
        tests:
          - dbt_utils.accepted_range:
              min_value: -100
              max_value: 100
"""

# =============================================================================
# CUSTOM TESTS
# =============================================================================

# tests/assert_order_totals_match.sql
CUSTOM_TEST_ORDER_TOTALS = """
-- Test: Ensure order totals match sum of line items
-- This test validates data integrity between orders and order_items

with order_totals as (
    select
        o.order_id,
        o.total_amount as order_total,
        sum(oi.line_total) as calculated_total,
        abs(o.total_amount - sum(oi.line_total)) as difference
    from {{ ref('stg_orders') }} o
    left join {{ ref('stg_order_items') }} oi on o.order_id = oi.order_id
    group by o.order_id, o.total_amount
)

select
    order_id,
    order_total,
    calculated_total,
    difference
from order_totals
where difference > 1  -- Allow for rounding differences
"""

# tests/assert_customer_email_format.sql
CUSTOM_TEST_EMAIL_FORMAT = """
-- Test: Validate customer email format
-- This test ensures all customer emails have valid format

select
    customer_id,
    email
from {{ ref('stg_customers') }}
where not regexp_like(email, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$')
"""

# tests/assert_positive_profit_margins.sql
CUSTOM_TEST_PROFIT_MARGINS = """
-- Test: Ensure profit margins are reasonable
-- This test identifies products with suspicious profit margins

select
    product_id,
    product_name,
    profit_margin_percent
from {{ ref('dim_products') }}
where profit_margin_percent < -50  -- Flag products with extreme losses
   or profit_margin_percent > 200  -- Flag products with extreme margins
"""

# tests/assert_customer_segment_distribution.sql
CUSTOM_TEST_SEGMENT_DISTRIBUTION = """
-- Test: Validate customer segment distribution
-- This test ensures customer segmentation is working correctly

with segment_counts as (
    select
        customer_segment,
        count(*) as customer_count,
        count(*) * 100.0 / sum(count(*)) over () as percentage
    from {{ ref('dim_customers') }}
    where has_purchased = true
    group by customer_segment
)

select
    customer_segment,
    customer_count,
    percentage
from segment_counts
where percentage > 80  -- Flag if any segment has more than 80% of customers
"""

# =============================================================================
# MACROS
# =============================================================================

# macros/get_order_statuses.sql
MACRO_ORDER_STATUSES = """
-- Macro: Get valid order statuses
-- Returns a list of valid order statuses for validation

{% macro get_order_statuses() %}
    {{ return(['pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'returned']) }}
{% endmacro %}
"""

# macros/calculate_profit_margin.sql
MACRO_PROFIT_MARGIN = """
-- Macro: Calculate profit margin percentage
-- Calculates profit margin with proper null handling

{% macro calculate_profit_margin(revenue_column, cost_column) %}
    case 
        when {{ revenue_column }} > 0 then 
            round((({{ revenue_column }} - {{ cost_column }}) / {{ revenue_column }}) * 100, 2)
        else 0 
    end
{% endmacro %}
"""

# macros/generate_schema_name.sql
MACRO_GENERATE_SCHEMA_NAME = """
-- Macro: Custom schema naming
-- Generates schema names based on environment and model path

{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ default_schema }}_{{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
"""

# =============================================================================
# COMMANDS AND WORKFLOW
# =============================================================================

def print_dbt_workflow_commands():
    """Print essential dbt workflow commands for the e-commerce analytics project"""
    
    print("\n🚀 dbt Commands for E-commerce Analytics Project")
    print("=" * 55)
    
    print("\n📦 Setup Commands:")
    print("dbt deps                    # Install packages")
    print("dbt compile                 # Compile SQL without running")
    print("dbt parse                   # Parse project files")
    
    print("\n🏃 Execution Commands:")
    print("dbt run                     # Run all models")
    print("dbt test                    # Run all tests")
    print("dbt build                   # Run + test in dependency order")
    print("dbt snapshot                # Run snapshots")
    
    print("\n🎯 Selective Execution:")
    print("dbt run --models staging            # Run staging models only")
    print("dbt run --models +dim_customers     # Run model and upstream")
    print("dbt run --models fct_orders+        # Run model and downstream")
    print("dbt test --models marts             # Test mart models only")
    print("dbt run --models tag:staging        # Run by tag")
    
    print("\n📊 Incremental & Refresh:")
    print("dbt run --full-refresh              # Full refresh all incremental")
    print("dbt run --models fct_orders --full-refresh  # Refresh specific model")
    
    print("\n📚 Documentation:")
    print("dbt docs generate           # Generate documentation")
    print("dbt docs serve              # Serve docs locally")
    
    print("\n🔍 Debugging:")
    print("dbt compile --models dim_customers  # Check compiled SQL")
    print("dbt run --models dim_customers --debug  # Debug mode")
    print("dbt test --store-failures           # Store test failures")
    
    print("\n🎯 Production Workflow:")
    print("1. dbt deps                 # Install dependencies")
    print("2. dbt compile              # Validate SQL compilation")
    print("3. dbt run --models staging # Run staging layer")
    print("4. dbt test --models staging # Test staging layer")
    print("5. dbt run --models marts   # Run mart layer")
    print("6. dbt test --models marts  # Test mart layer")
    print("7. dbt docs generate        # Update documentation")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎯 dbt Basics - Complete Solution")
    print("=" * 50)
    
    print("\n✅ SOLUTION INCLUDES:")
    print("• Complete dbt project configuration")
    print("• Comprehensive source definitions with tests")
    print("• Layered architecture (staging → intermediate → marts)")
    print("• Production-ready SQL models")
    print("• Comprehensive testing strategy")
    print("• Custom business rule tests")
    print("• Reusable macros")
    print("• Proper documentation")
    print("• Incremental models for large tables")
    print("• Performance optimizations")
    
    print("\n🏗️ ARCHITECTURE HIGHLIGHTS:")
    print("• Staging: 1:1 with sources, data cleaning")
    print("• Intermediate: Business logic, ephemeral CTEs")
    print("• Marts: Final tables for analytics")
    print("• Dimensional modeling with facts and dimensions")
    print("• Comprehensive customer and product analytics")
    print("• Financial reporting with period comparisons")
    
    print("\n🧪 TESTING STRATEGY:")
    print("• Source freshness monitoring")
    print("• Primary key uniqueness and not-null")
    print("• Referential integrity between tables")
    print("• Business rule validation")
    print("• Data quality assertions")
    print("• Custom tests for complex business logic")
    
    print("\n📊 KEY MODELS:")
    print("• dim_customers: Customer dimension with LTV and segmentation")
    print("• dim_products: Product dimension with performance metrics")
    print("• fct_orders: Order fact table (incremental)")
    print("• revenue_summary: Daily revenue aggregations")
    
    print("\n🎯 PRODUCTION FEATURES:")
    print("• Incremental models for performance")
    print("• Proper indexing strategies")
    print("• Environment-specific configurations")
    print("• Comprehensive audit trails")
    print("• Data lineage through refs")
    print("• Automated documentation generation")
    
    print_dbt_workflow_commands()
    
    print("\n" + "="*50)
    print("🚀 Solution demonstrates production-ready dbt practices!")
    print("This project showcases enterprise-grade analytics engineering.")
    print("="*50)