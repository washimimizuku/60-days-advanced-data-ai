# Day 17: dbt Deep Dive - Advanced Patterns, Incremental Models, Snapshots

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** advanced dbt materialization strategies for enterprise-scale data
- **Implement** sophisticated incremental models with multiple update strategies
- **Create** slowly changing dimensions using dbt snapshots
- **Optimize** dbt performance for large datasets and complex transformations
- **Apply** advanced analytics engineering patterns for production environments

---

## Theory

### Advanced dbt Materializations Deep Dive

Building on Day 13's foundation, we'll explore sophisticated materialization strategies that handle enterprise-scale data processing requirements.

#### 1. Incremental Models - Production Patterns

Incremental models are essential for processing large datasets efficiently. They only process new or changed data, dramatically reducing compute costs and processing time.

**Basic Incremental Configuration**:
```sql
-- models/marts/fct_user_events.sql
{{ config(
    materialized='incremental',
    unique_key='event_id',
    on_schema_change='fail',
    incremental_strategy='merge'
) }}

with events as (
    select * from {{ ref('stg_user_events') }}
    
    {% if is_incremental() %}
        -- Only process events since last run
        where event_timestamp > (select max(event_timestamp) from {{ this }})
    {% endif %}
)

select * from events
```

**Advanced Incremental Strategies**:

**1. Merge Strategy** (Default for most warehouses):
```sql
{{ config(
    materialized='incremental',
    unique_key='user_id',
    incremental_strategy='merge',
    merge_exclude_columns=['created_at']  -- Don't update these columns
) }}

-- Handles updates to existing records and inserts new ones
-- Perfect for slowly changing dimensions
```

**2. Append Strategy** (Insert-only):
```sql
{{ config(
    materialized='incremental',
    incremental_strategy='append'
) }}

-- Only adds new rows, never updates existing ones
-- Ideal for immutable event data
```

**3. Delete+Insert Strategy**:
```sql
{{ config(
    materialized='incremental',
    unique_key='date_key',
    incremental_strategy='delete+insert'
) }}

-- Deletes all records matching unique_key, then inserts new ones
-- Useful for daily aggregations that need complete refresh
```

**4. Insert Overwrite Strategy** (BigQuery):
```sql
{{ config(
    materialized='incremental',
    partition_by={'field': 'event_date', 'data_type': 'date'},
    incremental_strategy='insert_overwrite'
) }}

-- Overwrites specific partitions
-- Extremely efficient for partitioned tables
```

#### 2. Complex Incremental Logic

**Multi-Condition Incremental Processing**:
```sql
-- models/marts/fct_customer_metrics.sql
{{ config(
    materialized='incremental',
    unique_key='customer_id',
    on_schema_change='sync_all_columns'
) }}

with base_data as (
    select * from {{ ref('stg_customers') }}
    
    {% if is_incremental() %}
        -- Process customers that have been updated OR have new orders
        where updated_at > (select max(last_updated) from {{ this }})
           or customer_id in (
               select distinct customer_id 
               from {{ ref('stg_orders') }}
               where created_at > (select max(last_updated) from {{ this }})
           )
    {% endif %}
),

customer_metrics as (
    select
        c.customer_id,
        c.email,
        c.first_name,
        c.last_name,
        c.created_at,
        c.updated_at,
        
        -- Calculate metrics from orders
        coalesce(o.total_orders, 0) as total_orders,
        coalesce(o.total_spent, 0) as total_spent,
        coalesce(o.avg_order_value, 0) as avg_order_value,
        o.first_order_date,
        o.last_order_date,
        
        -- Calculate customer lifetime value
        case
            when o.total_spent >= 10000 then 'VIP'
            when o.total_spent >= 5000 then 'High Value'
            when o.total_spent >= 1000 then 'Medium Value'
            else 'Low Value'
        end as customer_segment,
        
        -- Recency, Frequency, Monetary (RFM) scoring
        case
            when datediff('day', o.last_order_date, current_date()) <= 30 then 5
            when datediff('day', o.last_order_date, current_date()) <= 90 then 4
            when datediff('day', o.last_order_date, current_date()) <= 180 then 3
            when datediff('day', o.last_order_date, current_date()) <= 365 then 2
            else 1
        end as recency_score,
        
        case
            when o.total_orders >= 20 then 5
            when o.total_orders >= 10 then 4
            when o.total_orders >= 5 then 3
            when o.total_orders >= 2 then 2
            else 1
        end as frequency_score,
        
        case
            when o.total_spent >= 5000 then 5
            when o.total_spent >= 2000 then 4
            when o.total_spent >= 1000 then 3
            when o.total_spent >= 500 then 2
            else 1
        end as monetary_score,
        
        current_timestamp() as last_updated
        
    from base_data c
    left join (
        select
            customer_id,
            count(*) as total_orders,
            sum(amount) as total_spent,
            avg(amount) as avg_order_value,
            min(order_date) as first_order_date,
            max(order_date) as last_order_date
        from {{ ref('stg_orders') }}
        group by customer_id
    ) o on c.customer_id = o.customer_id
)

select * from customer_metrics
```

#### 3. dbt Snapshots - Slowly Changing Dimensions

Snapshots capture how data changes over time, essential for tracking historical states of mutable data.

**Type 2 SCD with Snapshots**:
```sql
-- snapshots/customers_snapshot.sql
{% snapshot customers_snapshot %}
    {{
        config(
            target_database='analytics',
            target_schema='snapshots',
            unique_key='customer_id',
            strategy='timestamp',
            updated_at='updated_at',
            invalidate_hard_deletes=true
        )
    }}
    
    select
        customer_id,
        email,
        first_name,
        last_name,
        customer_status,
        subscription_tier,
        created_at,
        updated_at
    from {{ source('raw_data', 'customers') }}
    
{% endsnapshot %}
```

**Check Strategy Snapshot** (for data without timestamps):
```sql
-- snapshots/product_prices_snapshot.sql
{% snapshot product_prices_snapshot %}
    {{
        config(
            target_database='analytics',
            target_schema='snapshots',
            unique_key='product_id',
            strategy='check',
            check_cols=['price', 'discount_percentage', 'status']
        )
    }}
    
    select
        product_id,
        product_name,
        price,
        discount_percentage,
        status,
        category_id
    from {{ source('raw_data', 'products') }}
    
{% endsnapshot %}
```

#### 4. Advanced Model Configurations

**Performance Optimization**:
```sql
-- models/marts/fct_large_dataset.sql
{{ config(
    materialized='incremental',
    unique_key='composite_key',
    
    -- Partitioning for performance
    partition_by={
        'field': 'event_date',
        'data_type': 'date',
        'granularity': 'day'
    },
    
    -- Clustering for query performance
    cluster_by=['customer_id', 'product_category'],
    
    -- Schema evolution handling
    on_schema_change='sync_all_columns',
    
    -- Custom incremental predicates
    incremental_predicates=[
        "event_date >= date_sub(current_date(), interval 7 day)"
    ],
    
    -- Pre and post hooks
    pre_hook="delete from {{ this }} where event_date < date_sub(current_date(), interval 2 year)",
    post_hook=[
        "create or replace view {{ this.schema }}.latest_events as select * from {{ this }} where event_date = current_date()",
        "grant select on {{ this }} to role analyst"
    ]
) }}
```

#### 5. Advanced Testing Patterns

**Custom Data Tests for Complex Business Logic**:
```sql
-- tests/assert_customer_rfm_scores_valid.sql
-- Test that RFM scores are within expected ranges and combinations make sense

with rfm_validation as (
    select
        customer_id,
        recency_score,
        frequency_score,
        monetary_score,
        total_orders,
        total_spent,
        last_order_date
    from {{ ref('fct_customer_metrics') }}
    where
        -- Invalid score ranges
        recency_score not between 1 and 5
        or frequency_score not between 1 and 5
        or monetary_score not between 1 and 5
        
        -- Logical inconsistencies
        or (frequency_score = 5 and total_orders < 10)  -- High frequency but low order count
        or (monetary_score = 5 and total_spent < 2000)  -- High monetary but low spend
        or (recency_score = 5 and datediff('day', last_order_date, current_date()) > 30)  -- High recency but old order
)

select * from rfm_validation
```

**Incremental Test Strategy**:
```yaml
# models/schema.yml
version: 2

models:
  - name: fct_customer_metrics
    description: Customer metrics with RFM scoring
    tests:
      # Test incremental logic
      - dbt_utils.expression_is_true:
          expression: "last_updated >= current_date() - interval '1 day'"
          config:
            where: "last_updated is not null"
      
      # Test business logic
      - dbt_utils.expression_is_true:
          expression: "total_spent >= 0"
      
      # Test referential integrity
      - dbt_utils.equal_rowcount:
          compare_model: ref('stg_customers')
    
    columns:
      - name: customer_id
        tests:
          - unique
          - not_null
          - relationships:
              to: ref('dim_customers')
              field: customer_id
      
      - name: customer_segment
        tests:
          - accepted_values:
              values: ['VIP', 'High Value', 'Medium Value', 'Low Value']
      
      - name: recency_score
        tests:
          - dbt_utils.accepted_range:
              min_value: 1
              max_value: 5
              inclusive: true
```

#### 6. Performance Optimization Strategies

**Query Optimization Patterns**:
```sql
-- models/intermediate/int_order_aggregations.sql
{{ config(
    materialized='incremental',
    unique_key='date_customer_key',
    indexes=[
        {'columns': ['order_date', 'customer_id'], 'type': 'btree'},
        {'columns': ['customer_segment'], 'type': 'hash'}
    ]
) }}

-- Use window functions efficiently
with daily_customer_metrics as (
    select
        order_date,
        customer_id,
        
        -- Efficient aggregations
        sum(order_amount) as daily_revenue,
        count(*) as daily_orders,
        avg(order_amount) as avg_order_value,
        
        -- Window functions for running totals
        sum(sum(order_amount)) over (
            partition by customer_id 
            order by order_date 
            rows unbounded preceding
        ) as running_total_revenue,
        
        -- Efficient ranking
        row_number() over (
            partition by customer_id 
            order by order_date desc
        ) as recency_rank,
        
        -- Concatenate key for uniqueness
        concat(order_date, '-', customer_id) as date_customer_key
        
    from {{ ref('stg_orders') }}
    
    {% if is_incremental() %}
        where order_date > (select max(order_date) from {{ this }})
    {% endif %}
    
    group by order_date, customer_id
)

select * from daily_customer_metrics
```

#### 7. Advanced Jinja and Macros

**Dynamic Model Generation**:
```sql
-- macros/generate_metric_models.sql
{% macro generate_daily_metrics(metric_name, source_table, date_column, metric_columns) %}

{{ config(
    materialized='incremental',
    unique_key='date_key'
) }}

with daily_{{ metric_name }} as (
    select
        {{ date_column }}::date as date_key,
        
        {% for column in metric_columns %}
        sum({{ column }}) as daily_{{ column }},
        avg({{ column }}) as avg_{{ column }},
        {% endfor %}
        
        count(*) as daily_record_count
        
    from {{ ref(source_table) }}
    
    {% if is_incremental() %}
        where {{ date_column }}::date > (select max(date_key) from {{ this }})
    {% endif %}
    
    group by {{ date_column }}::date
)

select * from daily_{{ metric_name }}

{% endmacro %}

-- Usage in model
-- models/marts/daily_revenue_metrics.sql
{{ generate_daily_metrics(
    'revenue', 
    'stg_orders', 
    'order_date', 
    ['amount', 'tax_amount', 'shipping_amount']
) }}
```

#### 8. Error Handling and Data Quality

**Robust Error Handling**:
```sql
-- models/staging/stg_orders_with_quality_checks.sql
{{ config(
    materialized='view',
    on_schema_change='fail'
) }}

with source as (
    select * from {{ source('raw_data', 'orders') }}
),

quality_checks as (
    select
        *,
        
        -- Data quality flags
        case
            when order_id is null then 'missing_order_id'
            when customer_id is null then 'missing_customer_id'
            when amount <= 0 then 'invalid_amount'
            when order_date > current_date() then 'future_order_date'
            when order_date < '2020-01-01' then 'too_old_order_date'
            else 'valid'
        end as data_quality_flag,
        
        -- Standardize and clean data
        coalesce(trim(upper(status)), 'UNKNOWN') as cleaned_status,
        
        -- Handle currency conversion
        case
            when currency = 'USD' then amount
            when currency = 'EUR' then amount * 1.1  -- Simplified conversion
            when currency = 'GBP' then amount * 1.3
            else amount  -- Default to original amount
        end as amount_usd,
        
        current_timestamp() as processed_at
        
    from source
),

final as (
    select
        order_id,
        customer_id,
        cleaned_status as status,
        amount_usd as amount,
        order_date,
        created_at,
        updated_at,
        data_quality_flag,
        processed_at
    from quality_checks
    
    -- Only include valid records in downstream models
    where data_quality_flag = 'valid'
)

select * from final
```

---

## 💻 Hands-On Exercise (40 minutes)

Build an advanced dbt project with incremental models, snapshots, and complex business logic.

**Scenario**: You're the Senior Analytics Engineer at "DataCorp", a fast-growing SaaS company. You need to build sophisticated analytics models that can handle millions of events daily while tracking customer behavior changes over time.

**Requirements**:
1. **Advanced Incremental Models**: Build customer metrics with complex incremental logic
2. **Snapshots**: Track slowly changing customer and product dimensions
3. **Performance Optimization**: Implement partitioning and clustering strategies
4. **Data Quality**: Add comprehensive testing and error handling
5. **Business Logic**: Implement RFM analysis and customer segmentation
6. **Documentation**: Create detailed model documentation

**Data Sources**:
- `raw.users` - User account information (changes over time)
- `raw.subscriptions` - Subscription data (changes over time)
- `raw.events` - User behavior events (append-only, millions daily)
- `raw.products` - Product catalog (changes over time)

See `exercise.py` for starter code and detailed requirements.

---

## 📚 Resources

- **dbt Incremental Models**: [docs.getdbt.com/docs/build/incremental-models](https://docs.getdbt.com/docs/build/incremental-models)
- **dbt Snapshots**: [docs.getdbt.com/docs/build/snapshots](https://docs.getdbt.com/docs/build/snapshots)
- **Performance Optimization**: [docs.getdbt.com/guides/best-practices/how-we-structure/4-marts](https://docs.getdbt.com/guides/best-practices/how-we-structure/4-marts)
- **Advanced Testing**: [docs.getdbt.com/docs/build/tests](https://docs.getdbt.com/docs/build/tests)
- **dbt Utils Package**: [hub.getdbt.com/dbt-labs/dbt_utils/latest/](https://hub.getdbt.com/dbt-labs/dbt_utils/latest/)
- **Jinja and Macros**: [docs.getdbt.com/docs/build/jinja-macros](https://docs.getdbt.com/docs/build/jinja-macros)

---

## 🎯 Key Takeaways

- **Incremental models are essential** for processing large datasets efficiently in production
- **Multiple incremental strategies** (merge, append, delete+insert) serve different use cases
- **Snapshots capture historical changes** in mutable data for slowly changing dimensions
- **Performance optimization** through partitioning, clustering, and indexing is crucial at scale
- **Complex business logic** can be implemented efficiently with advanced dbt patterns
- **Data quality checks** should be built into every model for production reliability
- **Testing strategy** must cover both technical and business logic validation
- **Documentation and monitoring** are essential for maintaining complex analytics systems

---

## 🚀 What's Next?

Tomorrow (Day 18), you'll learn **dbt Advanced** - custom materializations, macros, packages, and advanced analytics engineering patterns.

**Preview**: You'll explore custom materializations, build reusable macro libraries, integrate external packages, and implement advanced analytics patterns like cohort analysis and attribution modeling!

---

## ✅ Before Moving On

- [ ] Understand different incremental strategies and when to use each
- [ ] Can implement complex incremental logic with multiple conditions
- [ ] Know how to create and configure snapshots for SCDs
- [ ] Understand performance optimization techniques
- [ ] Can implement comprehensive data quality checks
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced Analytics Engineering)

Ready to master enterprise-scale dbt! 🚀
