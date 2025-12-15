# Day 13: dbt Basics - Models, Sources, Tests

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- Master dbt (data build tool) fundamentals and architecture
- Create and configure models, sources, and comprehensive tests
- Implement analytics engineering patterns with layered architecture
- Build production-ready transformation pipelines with proper documentation
- Apply software engineering practices to data transformation workflows

---

## Theory

### What is dbt?

dbt (data build tool) is the industry standard for analytics engineering, transforming data in your warehouse using SQL while bringing software engineering best practices to data transformation. It's used by companies like Spotify, GitLab, Shopify, and thousands of data teams worldwide.

**Why dbt Matters**:
- **SQL-First**: Use familiar SQL for transformations
- **Version Control**: All transformations in git
- **Testing**: Built-in data quality testing
- **Documentation**: Auto-generated, always up-to-date
- **Lineage**: Visual dependency graphs
- **Modularity**: Reusable, composable transformations

### Core dbt Concepts

#### 1. Models

Models are the heart of dbt - they're SQL SELECT statements that create tables or views in your warehouse.

```sql
-- models/staging/stg_customers.sql
-- This model cleans and standardizes raw customer data

{{ config(materialized='view') }}

with source as (
    select * from {{ source('raw_data', 'customers') }}
),

cleaned as (
    select
        id as customer_id,
        lower(trim(email)) as email,
        initcap(first_name) as first_name,
        initcap(last_name) as last_name,
        created_at,
        updated_at
    from source
    where email is not null
      and email like '%@%'
)

select * from cleaned
```

**Model Materializations**:

```sql
-- View (default) - Virtual table, no storage
{{ config(materialized='view') }}

-- Table - Physical table, full refresh each run
{{ config(materialized='table') }}

-- Incremental - Only process new/changed data
{{ config(
    materialized='incremental',
    unique_key='id',
    on_schema_change='fail'
) }}

-- Ephemeral - CTE only, not materialized
{{ config(materialized='ephemeral') }}
```

#### 2. Sources

Sources define raw data tables in your warehouse, providing a single point of truth for upstream data.

```yaml
# models/sources.yml
version: 2

sources:
  - name: raw_data
    description: Raw data from production systems
    database: analytics_db
    schema: raw
    tables:
      - name: customers
        description: Customer account information
        columns:
          - name: id
            description: Primary key
            tests:
              - unique
              - not_null
          - name: email
            description: Customer email address
            tests:
              - not_null
              - unique
        
      - name: orders
        description: Order transactions
        loaded_at_field: _loaded_at
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
        columns:
          - name: id
            tests:
              - unique
              - not_null
          - name: customer_id
            tests:
              - not_null
              - relationships:
                  to: source('raw_data', 'customers')
                  field: id
```

#### 3. Tests

dbt provides comprehensive testing capabilities to ensure data quality.

**Schema Tests** (built-in):
```yaml
# models/schema.yml
version: 2

models:
  - name: stg_customers
    description: Cleaned customer data
    columns:
      - name: customer_id
        description: Unique customer identifier
        tests:
          - unique
          - not_null
      - name: email
        description: Customer email address
        tests:
          - not_null
          - unique
      - name: first_name
        tests:
          - not_null
      - name: created_at
        tests:
          - not_null
```

**Data Tests** (custom SQL):
```sql
-- tests/assert_customer_email_format.sql
-- Test that all customer emails have valid format

select
    customer_id,
    email
from {{ ref('stg_customers') }}
where not regexp_like(email, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
```

**Singular Tests** (one-off tests):
```sql
-- tests/assert_positive_order_amounts.sql
select
    order_id,
    amount
from {{ ref('fct_orders') }}
where amount <= 0
```

#### 4. Project Structure

dbt follows a layered architecture that promotes maintainability and understanding:

```
dbt_project/
├── dbt_project.yml           # Project configuration
├── profiles.yml              # Connection settings
├── packages.yml              # dbt packages
│
├── models/
│   ├── staging/              # 1:1 with source tables, cleaning
│   │   ├── _sources.yml      # Source definitions
│   │   ├── stg_customers.sql
│   │   └── stg_orders.sql
│   │
│   ├── intermediate/         # Business logic, joins
│   │   ├── int_customer_orders.sql
│   │   └── int_order_metrics.sql
│   │
│   ├── marts/                # Final business tables
│   │   ├── core/             # Key business entities
│   │   │   ├── dim_customers.sql
│   │   │   └── fct_orders.sql
│   │   └── finance/          # Department-specific
│   │       └── revenue_summary.sql
│   │
│   └── schema.yml            # Model documentation and tests
│
├── tests/                    # Custom data tests
├── macros/                   # Reusable SQL functions
├── seeds/                    # CSV files to load
└── snapshots/                # Slowly changing dimensions
```

### Advanced dbt Features

#### 1. Incremental Models

For large tables, incremental models only process new or changed data:

```sql
-- models/marts/fct_orders.sql
{{ config(
    materialized='incremental',
    unique_key='order_id',
    on_schema_change='fail'
) }}

with orders as (
    select * from {{ ref('stg_orders') }}
    
    {% if is_incremental() %}
        -- Only process new orders since last run
        where created_at > (select max(created_at) from {{ this }})
    {% endif %}
),

customers as (
    select * from {{ ref('dim_customers') }}
),

final as (
    select
        o.order_id,
        o.customer_id,
        c.customer_name,
        o.order_date,
        o.amount,
        o.status,
        o.created_at
    from orders o
    left join customers c on o.customer_id = c.customer_id
)

select * from final
```

#### 2. Macros

Macros are reusable SQL functions that promote DRY principles:

```sql
-- macros/get_payment_methods.sql
{% macro get_payment_methods() %}
    {{ return(['credit_card', 'debit_card', 'paypal', 'bank_transfer']) }}
{% endmacro %}

-- macros/cents_to_dollars.sql
{% macro cents_to_dollars(column_name, precision=2) %}
    round({{ column_name }} / 100.0, {{ precision }})
{% endmacro %}

-- Usage in models
select
    order_id,
    {{ cents_to_dollars('amount_cents') }} as amount_dollars,
    payment_method
from {{ ref('stg_orders') }}
where payment_method in ({{ get_payment_methods() | join("','") }})
```

#### 3. Snapshots

Snapshots capture slowly changing dimensions:

```sql
-- snapshots/customers_snapshot.sql
{% snapshot customers_snapshot %}
    {{
        config(
            target_database='analytics',
            target_schema='snapshots',
            unique_key='id',
            strategy='timestamp',
            updated_at='updated_at',
        )
    }}
    
    select * from {{ source('raw_data', 'customers') }}
    
{% endsnapshot %}
```

### Production Best Practices

#### 1. Layered Architecture

**Staging Layer** (1:1 with sources):
```sql
-- Purpose: Clean, standardize, and document raw data
-- Naming: stg_{source_table}
-- Materialization: Usually views
-- Tests: Basic data quality (not_null, unique)

-- models/staging/stg_customers.sql
with source as (
    select * from {{ source('raw_data', 'customers') }}
),

cleaned as (
    select
        id as customer_id,
        lower(trim(email)) as email,
        coalesce(first_name, 'Unknown') as first_name,
        coalesce(last_name, 'Unknown') as last_name,
        created_at,
        updated_at
    from source
    where id is not null
)

select * from cleaned
```

**Intermediate Layer** (business logic):
```sql
-- Purpose: Complex joins, business logic, calculations
-- Naming: int_{business_concept}
-- Materialization: Usually views or ephemeral
-- Tests: Business rule validation

-- models/intermediate/int_customer_lifetime_value.sql
with customer_orders as (
    select
        customer_id,
        count(*) as total_orders,
        sum(amount) as total_spent,
        min(order_date) as first_order_date,
        max(order_date) as last_order_date
    from {{ ref('stg_orders') }}
    group by customer_id
),

customer_metrics as (
    select
        customer_id,
        total_orders,
        total_spent,
        first_order_date,
        last_order_date,
        total_spent / total_orders as avg_order_value,
        datediff('day', first_order_date, last_order_date) as customer_lifespan_days
    from customer_orders
)

select * from customer_metrics
```

**Marts Layer** (final business tables):
```sql
-- Purpose: Final tables for analysis and reporting
-- Naming: dim_{entity} or fct_{process}
-- Materialization: Usually tables
-- Tests: Comprehensive business validation

-- models/marts/core/dim_customers.sql
{{ config(materialized='table') }}

with customers as (
    select * from {{ ref('stg_customers') }}
),

customer_metrics as (
    select * from {{ ref('int_customer_lifetime_value') }}
),

final as (
    select
        c.customer_id,
        c.email,
        c.first_name,
        c.last_name,
        c.created_at,
        coalesce(m.total_orders, 0) as lifetime_orders,
        coalesce(m.total_spent, 0) as lifetime_value,
        coalesce(m.avg_order_value, 0) as avg_order_value,
        case
            when m.total_spent >= 1000 then 'High Value'
            when m.total_spent >= 500 then 'Medium Value'
            else 'Low Value'
        end as customer_segment
    from customers c
    left join customer_metrics m on c.customer_id = m.customer_id
)

select * from final
```

#### 2. Testing Strategy

**Comprehensive Test Coverage**:
```yaml
# models/schema.yml
version: 2

models:
  - name: dim_customers
    description: Customer dimension table with lifetime metrics
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - customer_id
            - email
    columns:
      - name: customer_id
        description: Unique customer identifier
        tests:
          - unique
          - not_null
      - name: email
        description: Customer email address
        tests:
          - not_null
          - unique
      - name: lifetime_value
        description: Total customer spend
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0
              inclusive: true
      - name: customer_segment
        description: Customer value segment
        tests:
          - accepted_values:
              values: ['High Value', 'Medium Value', 'Low Value']
```

#### 3. Documentation

**Model Documentation**:
```yaml
# models/schema.yml
version: 2

models:
  - name: fct_orders
    description: |
      Order fact table containing all order transactions with customer information.
      
      This table is updated incrementally and contains one row per order.
      It includes calculated fields like profit margin and customer segment.
      
      **Business Rules:**
      - Only includes orders with status 'completed' or 'shipped'
      - Amounts are in USD cents
      - Profit margin calculated as (revenue - cost) / revenue
      
    columns:
      - name: order_id
        description: |
          Unique identifier for each order. This is the primary key.
          Format: ORD-{timestamp}-{random}
        tests:
          - unique
          - not_null
      
      - name: customer_id
        description: |
          Foreign key to dim_customers table.
          Links to the customer who placed this order.
        tests:
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
```

#### 4. Environment Management

**dbt_project.yml Configuration**:
```yaml
name: 'analytics'
version: '1.0.0'
config-version: 2

profile: 'analytics'

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
  analytics:
    # Staging models
    staging:
      +materialized: view
      +schema: staging
    
    # Intermediate models  
    intermediate:
      +materialized: ephemeral
    
    # Mart models
    marts:
      +materialized: table
      +schema: marts
      core:
        +schema: core
      finance:
        +schema: finance

# Environment-specific configurations
vars:
  # Development
  start_date: '2024-01-01'
  
  # Production (override in profiles.yml)
  # start_date: '2023-01-01'
```

### dbt Commands and Workflow

#### Essential Commands

```bash
# Project setup
dbt init my_project
dbt deps  # Install packages

# Development workflow
dbt compile  # Compile SQL without running
dbt run      # Execute all models
dbt test     # Run all tests
dbt build    # Run + test in dependency order

# Selective execution
dbt run --models staging        # Run staging models only
dbt run --models +dim_customers # Run model and all upstream
dbt run --models dim_customers+ # Run model and all downstream
dbt test --models fct_orders    # Test specific model

# Documentation
dbt docs generate  # Generate documentation
dbt docs serve     # Serve docs locally

# Production commands
dbt run --target prod           # Run against production
dbt run --full-refresh         # Full refresh incremental models
dbt snapshot                   # Run snapshots
```

#### CI/CD Integration

```yaml
# .github/workflows/dbt.yml
name: dbt CI/CD

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dbt
        run: |
          pip install dbt-core dbt-postgres
      
      - name: Run dbt tests
        run: |
          dbt deps
          dbt compile
          dbt run --target ci
          dbt test --target ci
        env:
          DBT_PROFILES_DIR: .
  
  deploy:
    if: github.ref == 'refs/heads/main'
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          dbt run --target prod
          dbt test --target prod
```

---

## 💻 Hands-On Exercise (40 minutes)

Build a complete dbt project for an e-commerce analytics platform.

**Scenario**: You're building analytics for an e-commerce company with raw data about customers, orders, and products. You need to create a layered dbt project with proper testing and documentation.

**Requirements**:
1. **Sources**: Define raw data sources with freshness tests
2. **Staging**: Clean and standardize raw data
3. **Intermediate**: Create business logic models
4. **Marts**: Build final dimensional and fact tables
5. **Tests**: Implement comprehensive testing strategy
6. **Documentation**: Add descriptions and business context

**Data Sources**:
- `raw.customers` - Customer account information
- `raw.orders` - Order transactions
- `raw.order_items` - Individual items within orders
- `raw.products` - Product catalog

See `exercise.py` for starter code and detailed requirements.

---

## 📚 Resources

- **Official Documentation**: [docs.getdbt.com](https://docs.getdbt.com/)
- **dbt Learn**: [courses.getdbt.com](https://courses.getdbt.com/) - Free comprehensive courses
- **Best Practices Guide**: [docs.getdbt.com/guides/best-practices](https://docs.getdbt.com/guides/best-practices)
- **dbt Utils Package**: [github.com/dbt-labs/dbt-utils](https://github.com/dbt-labs/dbt-utils)
- **Community**: [getdbt.com/community](https://www.getdbt.com/community/)
- **dbt Slack**: [getdbt.slack.com](https://getdbt.slack.com/)

---

## 🎯 Key Takeaways

- **dbt transforms data using SQL** with software engineering best practices
- **Layered architecture** (staging → intermediate → marts) promotes maintainability
- **Models are SELECT statements** with different materialization strategies
- **Sources define raw data** with freshness and quality tests
- **Comprehensive testing** ensures data quality and business rules
- **Documentation is generated** from code and stays up-to-date
- **Version control and CI/CD** enable collaborative development
- **Incremental models** handle large datasets efficiently

---

## 🚀 What's Next?

Tomorrow (Day 14), you'll build a **Governed Data Platform Project** that combines everything from Days 8-13:
- Data catalogs and lineage tracking
- Privacy and access control
- Airflow orchestration
- dbt transformations

**Preview**: You'll create a complete data platform that ingests raw data, applies governance policies, orchestrates transformations with Airflow, and produces analytics-ready datasets with dbt - all with proper monitoring, testing, and documentation!

---

## ✅ Before Moving On

- [ ] Understand dbt's layered architecture and why it matters
- [ ] Can create models with different materializations
- [ ] Know how to define sources and implement tests
- [ ] Understand the difference between schema and data tests
- [ ] Can use refs and sources for dependencies
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐ (Intermediate)

Ready to build production analytics pipelines! 🚀