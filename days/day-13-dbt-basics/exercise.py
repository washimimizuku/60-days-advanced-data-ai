"""
Day 13: dbt Basics - Exercise

Build a comprehensive dbt project for e-commerce analytics.

Scenario:
You're the analytics engineer at "DataMart", an e-commerce company. You need to build
a dbt project that transforms raw transactional data into analytics-ready datasets
for the business intelligence team.

Raw Data Sources:
- raw.customers: Customer account information
- raw.orders: Order transactions  
- raw.order_items: Individual items within orders
- raw.products: Product catalog information

Your Task:
Create a complete dbt project with proper layered architecture, testing, and documentation.

Requirements:
1. Define sources with appropriate tests and freshness checks
2. Create staging models that clean and standardize raw data
3. Build intermediate models with business logic
4. Create mart models (dimensions and facts) for analytics
5. Implement comprehensive testing strategy
6. Add proper documentation for all models
7. Use different materialization strategies appropriately
8. Include custom tests for business rules

Note: This is a Python file for consistency, but in practice you would create
.sql files in the models/ directory and .yml files for configuration.
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
"""

# =============================================================================
# SOURCES CONFIGURATION
# =============================================================================

# TODO: Create models/sources.yml
# Define your raw data sources with tests and freshness checks
SOURCES_YML = """
# Define sources here following this structure:
# version: 2
# sources:
#   - name: raw_data
#     description: Raw data from production systems
#     database: ecommerce_db
#     schema: raw
#     tables:
#       - name: customers
#         description: Customer account information
#         # Add freshness checks
#         # Add column tests
#       - name: orders
#         # Add configuration
#       - name: order_items
#         # Add configuration  
#       - name: products
#         # Add configuration
"""

# =============================================================================
# STAGING MODELS
# =============================================================================

# TODO: Create models/staging/stg_customers.sql
# Clean and standardize customer data
STG_CUSTOMERS_SQL = """
-- Staging model: Clean customer data
-- TODO: Write SQL to:
-- 1. Select from raw customers source
-- 2. Clean and standardize fields (email, names)
-- 3. Add data quality filters
-- 4. Rename columns to follow naming conventions

-- Example structure:
-- with source as (
--     select * from {{ source('raw_data', 'customers') }}
-- ),
-- 
-- cleaned as (
--     select
--         id as customer_id,
--         -- Add your transformations here
--     from source
--     where -- Add filters
-- )
-- 
-- select * from cleaned
"""

# TODO: Create models/staging/stg_orders.sql
STG_ORDERS_SQL = """
-- Staging model: Clean order data
-- TODO: Write SQL to clean and standardize order data
-- Include order status standardization and date formatting
"""

# TODO: Create models/staging/stg_order_items.sql
STG_ORDER_ITEMS_SQL = """
-- Staging model: Clean order items data
-- TODO: Write SQL to clean order items
-- Calculate line totals and handle quantity/price edge cases
"""

# TODO: Create models/staging/stg_products.sql
STG_PRODUCTS_SQL = """
-- Staging model: Clean product data
-- TODO: Write SQL to clean product catalog
-- Standardize categories and handle missing descriptions
"""

# =============================================================================
# INTERMEDIATE MODELS
# =============================================================================

# TODO: Create models/intermediate/int_order_metrics.sql
INT_ORDER_METRICS_SQL = """
-- Intermediate model: Calculate order-level metrics
-- TODO: Write SQL to:
-- 1. Join orders with order_items
-- 2. Calculate order totals, item counts
-- 3. Add profit calculations
-- 4. Include customer information

-- Should include fields like:
-- - order_id, customer_id, order_date
-- - total_amount, item_count, avg_item_price
-- - profit_margin, order_status
"""

# TODO: Create models/intermediate/int_customer_metrics.sql
INT_CUSTOMER_METRICS_SQL = """
-- Intermediate model: Calculate customer lifetime metrics
-- TODO: Write SQL to calculate:
-- 1. Customer lifetime value
-- 2. Order frequency and recency
-- 3. Average order value
-- 4. Customer segments based on behavior
"""

# TODO: Create models/intermediate/int_product_performance.sql
INT_PRODUCT_PERFORMANCE_SQL = """
-- Intermediate model: Product performance metrics
-- TODO: Write SQL to calculate:
-- 1. Product sales volumes and revenue
-- 2. Profit margins by product
-- 3. Return rates and customer satisfaction
-- 4. Inventory turnover metrics
"""

# =============================================================================
# MART MODELS (DIMENSIONAL MODEL)
# =============================================================================

# TODO: Create models/marts/core/dim_customers.sql
DIM_CUSTOMERS_SQL = """
-- Dimension: Customer dimension with SCD Type 1
-- TODO: Create customer dimension table with:
-- 1. Customer attributes (name, email, registration date)
-- 2. Calculated lifetime metrics
-- 3. Customer segmentation
-- 4. Geographic information

-- Use materialized='table' for this dimension
"""

# TODO: Create models/marts/core/dim_products.sql
DIM_PRODUCTS_SQL = """
-- Dimension: Product dimension
-- TODO: Create product dimension with:
-- 1. Product attributes (name, category, price)
-- 2. Product performance metrics
-- 3. Profitability indicators
-- 4. Inventory status
"""

# TODO: Create models/marts/core/fct_orders.sql
FCT_ORDERS_SQL = """
-- Fact: Order transactions fact table
-- TODO: Create fact table with:
-- 1. Order grain (one row per order)
-- 2. Foreign keys to dimensions
-- 3. Additive measures (amounts, quantities)
-- 4. Semi-additive measures (balances)

-- Use materialized='incremental' with proper unique_key
-- {{ config(
--     materialized='incremental',
--     unique_key='order_id'
-- ) }}
"""

# TODO: Create models/marts/finance/revenue_summary.sql
REVENUE_SUMMARY_SQL = """
-- Mart: Daily revenue summary for finance team
-- TODO: Create aggregated revenue table with:
-- 1. Daily, weekly, monthly aggregations
-- 2. Revenue by product category
-- 3. Profit margins and costs
-- 4. Year-over-year comparisons
"""

# =============================================================================
# TESTING CONFIGURATION
# =============================================================================

# TODO: Create models/schema.yml
# Add comprehensive tests for all models
SCHEMA_YML = """
# Add tests for all your models following this structure:
# version: 2
# 
# models:
#   - name: stg_customers
#     description: Cleaned customer data
#     columns:
#       - name: customer_id
#         tests:
#           - unique
#           - not_null
#       - name: email
#         tests:
#           - unique
#           - not_null
#   
#   - name: dim_customers
#     description: Customer dimension table
#     tests:
#       - dbt_utils.unique_combination_of_columns:
#           combination_of_columns:
#             - customer_id
#     columns:
#       - name: customer_id
#         tests:
#           - unique
#           - not_null
#       - name: lifetime_value
#         tests:
#           - not_null
#           - dbt_utils.accepted_range:
#               min_value: 0
#               inclusive: true
"""

# =============================================================================
# CUSTOM TESTS
# =============================================================================

# TODO: Create tests/assert_order_totals_match.sql
CUSTOM_TEST_ORDER_TOTALS = """
-- Custom test: Ensure order totals match sum of line items
-- TODO: Write SQL that returns rows where totals don't match
-- This should return 0 rows if all totals are correct

-- select
--     order_id,
--     order_total,
--     calculated_total,
--     abs(order_total - calculated_total) as difference
-- from (
--     -- Your calculation logic here
-- )
-- where abs(order_total - calculated_total) > 0.01
"""

# TODO: Create tests/assert_customer_email_format.sql
CUSTOM_TEST_EMAIL_FORMAT = """
-- Custom test: Validate email format
-- TODO: Write SQL to find invalid email addresses
-- Should return 0 rows if all emails are valid
"""

# =============================================================================
# MACROS
# =============================================================================

# TODO: Create macros/get_order_statuses.sql
MACRO_ORDER_STATUSES = """
-- Macro: Get valid order statuses
-- TODO: Create macro that returns list of valid order statuses
-- {% macro get_order_statuses() %}
--     {{ return(['pending', 'confirmed', 'shipped', 'delivered', 'cancelled']) }}
-- {% endmacro %}
"""

# TODO: Create macros/calculate_profit_margin.sql
MACRO_PROFIT_MARGIN = """
-- Macro: Calculate profit margin
-- TODO: Create reusable macro for profit margin calculation
-- {% macro calculate_profit_margin(revenue_column, cost_column) %}
--     case 
--         when {{ revenue_column }} > 0 
--         then round(({{ revenue_column }} - {{ cost_column }}) / {{ revenue_column }} * 100, 2)
--         else 0 
--     end
-- {% endmacro %}
"""

# =============================================================================
# PACKAGES CONFIGURATION
# =============================================================================

# TODO: Create packages.yml
PACKAGES_YML = """
# Add useful dbt packages
# packages:
#   - package: dbt-labs/dbt_utils
#     version: 1.1.1
#   - package: calogica/dbt_expectations
#     version: 0.10.1
"""

# =============================================================================
# EXERCISE INSTRUCTIONS
# =============================================================================

def print_exercise_instructions():
    """Print detailed exercise instructions"""
    
    print("🎯 dbt Basics Exercise - E-commerce Analytics Project")
    print("=" * 60)
    
    print("\n📋 REQUIREMENTS:")
    print("1. Create a complete dbt project structure")
    print("2. Define sources with tests and freshness checks")
    print("3. Build staging models (1:1 with sources)")
    print("4. Create intermediate models with business logic")
    print("5. Build mart models (dimensions and facts)")
    print("6. Implement comprehensive testing")
    print("7. Add proper documentation")
    print("8. Use appropriate materializations")
    
    print("\n🏗️ PROJECT STRUCTURE:")
    print("""
    ecommerce_analytics/
    ├── dbt_project.yml
    ├── packages.yml
    ├── models/
    │   ├── sources.yml
    │   ├── schema.yml
    │   ├── staging/
    │   │   ├── stg_customers.sql
    │   │   ├── stg_orders.sql
    │   │   ├── stg_order_items.sql
    │   │   └── stg_products.sql
    │   ├── intermediate/
    │   │   ├── int_order_metrics.sql
    │   │   ├── int_customer_metrics.sql
    │   │   └── int_product_performance.sql
    │   └── marts/
    │       ├── core/
    │       │   ├── dim_customers.sql
    │       │   ├── dim_products.sql
    │       │   └── fct_orders.sql
    │       └── finance/
    │           └── revenue_summary.sql
    ├── tests/
    │   ├── assert_order_totals_match.sql
    │   └── assert_customer_email_format.sql
    └── macros/
        ├── get_order_statuses.sql
        └── calculate_profit_margin.sql
    """)
    
    print("\n📊 DATA SOURCES:")
    print("- raw.customers: id, email, first_name, last_name, created_at")
    print("- raw.orders: id, customer_id, order_date, status, total_amount")
    print("- raw.order_items: id, order_id, product_id, quantity, unit_price")
    print("- raw.products: id, name, category, price, cost, description")
    
    print("\n🧪 TESTING REQUIREMENTS:")
    print("- Source freshness tests")
    print("- Unique and not_null tests on primary keys")
    print("- Referential integrity tests")
    print("- Business rule validation tests")
    print("- Custom data quality tests")
    
    print("\n📚 DOCUMENTATION REQUIREMENTS:")
    print("- Model descriptions explaining business purpose")
    print("- Column descriptions with business context")
    print("- Business rules and assumptions")
    print("- Data lineage through proper refs")
    
    print("\n⚙️ MATERIALIZATION STRATEGY:")
    print("- Staging: Views (fast, no storage)")
    print("- Intermediate: Ephemeral (CTEs only)")
    print("- Dimensions: Tables (fast queries)")
    print("- Facts: Incremental (large datasets)")
    print("- Aggregates: Tables (pre-computed)")
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("✅ All models compile without errors")
    print("✅ All tests pass")
    print("✅ Proper layered architecture")
    print("✅ Comprehensive documentation")
    print("✅ Appropriate materializations")
    print("✅ Business logic is correct")
    
    print("\n🚀 GETTING STARTED:")
    print("1. Create dbt_project.yml with proper configuration")
    print("2. Define sources in models/sources.yml")
    print("3. Build staging models (start with stg_customers)")
    print("4. Create intermediate models with joins")
    print("5. Build mart models (dimensions first, then facts)")
    print("6. Add tests in models/schema.yml")
    print("7. Create custom tests for business rules")
    print("8. Test everything with 'dbt build'")
    
    print("\n💡 TIPS:")
    print("- Start simple and add complexity gradually")
    print("- Use {{ ref() }} for model dependencies")
    print("- Use {{ source() }} for raw data references")
    print("- Test early and often")
    print("- Document as you build")
    print("- Follow naming conventions consistently")

# =============================================================================
# SAMPLE DATA STRUCTURES
# =============================================================================

def show_sample_data():
    """Show sample data structures for reference"""
    
    print("\n📋 SAMPLE DATA STRUCTURES:")
    print("=" * 40)
    
    print("\n🔹 raw.customers:")
    print("""
    id | email              | first_name | last_name | created_at
    ---|--------------------|-----------|-----------|-----------
    1  | john@email.com     | John      | Doe       | 2024-01-15
    2  | jane@email.com     | Jane      | Smith     | 2024-01-16
    3  | bob@email.com      | Bob       | Johnson   | 2024-01-17
    """)
    
    print("\n🔹 raw.orders:")
    print("""
    id | customer_id | order_date | status    | total_amount
    ---|-------------|------------|-----------|-------------
    1  | 1          | 2024-01-20 | completed | 150.00
    2  | 2          | 2024-01-21 | shipped   | 75.50
    3  | 1          | 2024-01-22 | pending   | 200.00
    """)
    
    print("\n🔹 raw.order_items:")
    print("""
    id | order_id | product_id | quantity | unit_price
    ---|----------|------------|----------|----------
    1  | 1        | 101        | 2        | 50.00
    2  | 1        | 102        | 1        | 50.00
    3  | 2        | 101        | 1        | 50.00
    4  | 2        | 103        | 1        | 25.50
    """)
    
    print("\n🔹 raw.products:")
    print("""
    id  | name        | category    | price | cost  | description
    ----|-------------|-------------|-------|-------|------------
    101 | Widget A    | Electronics | 50.00 | 30.00 | Great widget
    102 | Gadget B    | Electronics | 50.00 | 35.00 | Useful gadget
    103 | Tool C      | Tools       | 25.50 | 15.00 | Handy tool
    """)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_exercise_instructions()
    show_sample_data()
    
    print("\n" + "="*60)
    print("🎯 Ready to build your dbt project!")
    print("Start by creating the files shown above with the TODO items completed.")
    print("Remember: This exercise should take about 40 minutes to complete.")
    print("Focus on getting the core functionality working first, then add tests and documentation.")
    print("="*60)