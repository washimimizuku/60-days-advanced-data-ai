"""
Day 17: dbt Deep Dive - Exercise

Build an advanced dbt project with incremental models, snapshots, and complex business logic.

Scenario:
You're the Senior Analytics Engineer at "DataCorp", a fast-growing SaaS company. 
You need to build sophisticated analytics models that can handle millions of events 
daily while tracking customer behavior changes over time.

Business Context:
- 10M+ daily user events across web and mobile platforms
- 500K+ active users with changing subscription tiers
- Product catalog that changes frequently (pricing, features)
- Need for real-time customer segmentation and behavior analysis
- Requirement to track historical changes for compliance and analysis

Your Task:
Build advanced dbt models with incremental processing, snapshots, and complex analytics.

Requirements:
1. Advanced incremental models with complex business logic
2. Snapshots for slowly changing dimensions
3. Performance optimization with partitioning/clustering
4. Comprehensive data quality checks and error handling
5. RFM analysis and customer segmentation
6. Detailed documentation and testing
"""

import os
from typing import Dict, List, Any
from datetime import datetime, timedelta

# =============================================================================
# PROJECT STRUCTURE SETUP
# =============================================================================

def create_dbt_project_structure():
    """Create the complete dbt project structure for DataCorp analytics"""
    
    project_structure = {
        "dbt_project.yml": generate_dbt_project_config(),
        "profiles.yml": generate_profiles_config(),
        "packages.yml": generate_packages_config(),
        
        # Source definitions
        "models/sources.yml": generate_sources_config(),
        
        # Staging models
        "models/staging/stg_users.sql": generate_staging_users_model(),
        "models/staging/stg_subscriptions.sql": generate_staging_subscriptions_model(),
        "models/staging/stg_events.sql": generate_staging_events_model(),
        "models/staging/stg_products.sql": generate_staging_products_model(),
        
        # Intermediate models
        "models/intermediate/int_user_event_aggregates.sql": generate_user_event_aggregates(),
        "models/intermediate/int_subscription_history.sql": generate_subscription_history(),
        "models/intermediate/int_user_rfm_scores.sql": generate_rfm_scores(),
        
        # Mart models (incremental)
        "models/marts/fct_user_behavior.sql": generate_user_behavior_fact(),
        "models/marts/dim_users_scd.sql": generate_users_dimension(),
        "models/marts/fct_subscription_metrics.sql": generate_subscription_metrics(),
        
        # Snapshots
        "snapshots/users_snapshot.sql": generate_users_snapshot(),
        "snapshots/products_snapshot.sql": generate_products_snapshot(),
        
        # Tests
        "tests/assert_user_behavior_quality.sql": generate_behavior_quality_test(),
        "tests/assert_rfm_scores_valid.sql": generate_rfm_validation_test(),
        
        # Macros
        "macros/calculate_rfm_scores.sql": generate_rfm_macro(),
        "macros/get_customer_segments.sql": generate_segmentation_macro(),
        
        # Documentation
        "models/schema.yml": generate_schema_documentation()
    }
    
    return project_structure

# =============================================================================
# CONFIGURATION FILES
# =============================================================================

def generate_dbt_project_config():
    """Generate dbt_project.yml with advanced configurations"""
    
    # TODO: Create comprehensive dbt project configuration
    # Include model-specific configs, vars, and optimization settings
    
    config = """
name: 'datacorp_analytics'
version: '1.0.0'
config-version: 2

profile: 'datacorp'

model-paths: ["models"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]
analysis-paths: ["analysis"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

# Model configurations
models:
  datacorp_analytics:
    # Staging models - lightweight views
    staging:
      +materialized: view
      +schema: staging
    
    # Intermediate models - ephemeral for performance
    intermediate:
      +materialized: ephemeral
    
    # Mart models - tables with optimization
    marts:
      +materialized: table
      +schema: marts
      
      # Fact tables - incremental with partitioning
      +tags: ["fact"]
      +partition_by:
        field: "event_date"
        data_type: "date"
        granularity: "day"
      
      # Dimension tables - SCD Type 2
      +tags: ["dimension"]

# Snapshot configurations
snapshots:
  datacorp_analytics:
    +target_database: analytics
    +target_schema: snapshots
    +strategy: timestamp
    +updated_at: updated_at

# Global variables
vars:
  # Date ranges for development vs production
  start_date: '2024-01-01'
  lookback_days: 90
  
  # Business logic parameters
  high_value_threshold: 1000
  medium_value_threshold: 500
  active_user_days: 30
  
  # RFM scoring parameters
  rfm_recency_days: [30, 90, 180, 365]
  rfm_frequency_orders: [1, 3, 7, 15]
  rfm_monetary_amounts: [100, 500, 1500, 5000]

# Hooks for data quality
on-run-start:
  - "{{ log('Starting DataCorp analytics run at ' ~ run_started_at, info=true) }}"

on-run-end:
  - "{{ log('Completed DataCorp analytics run at ' ~ run_started_at, info=true) }}"
"""
    
    return config.strip()

def generate_profiles_config():
    """Generate profiles.yml for different environments"""
    
    # TODO: Create profiles configuration for dev/staging/prod environments
    
    config = """
datacorp:
  outputs:
    dev:
      type: postgres
      host: localhost
      user: "{{ env_var('DBT_USER') }}"
      password: "{{ env_var('DBT_PASSWORD') }}"
      port: 5432
      dbname: datacorp_dev
      schema: analytics_dev
      threads: 4
      keepalives_idle: 0
      search_path: "analytics_dev,public"
    
    staging:
      type: postgres
      host: staging-db.datacorp.com
      user: "{{ env_var('DBT_USER') }}"
      password: "{{ env_var('DBT_PASSWORD') }}"
      port: 5432
      dbname: datacorp_staging
      schema: analytics_staging
      threads: 8
      keepalives_idle: 0
    
    prod:
      type: postgres
      host: prod-db.datacorp.com
      user: "{{ env_var('DBT_USER') }}"
      password: "{{ env_var('DBT_PASSWORD') }}"
      port: 5432
      dbname: datacorp_prod
      schema: analytics
      threads: 16
      keepalives_idle: 0
      
  target: dev
"""
    
    return config.strip()

def generate_packages_config():
    """Generate packages.yml with useful dbt packages"""
    
    # TODO: Include essential dbt packages for advanced analytics
    
    config = """
packages:
  - package: dbt-labs/dbt_utils
    version: 1.1.1
  
  - package: calogica/dbt_expectations
    version: 0.10.1
  
  - package: dbt-labs/audit_helper
    version: 0.9.0
  
  - package: tnightengale/dbt_meta_testing
    version: 0.3.6
"""
    
    return config.strip()

# =============================================================================
# SOURCE DEFINITIONS
# =============================================================================

def generate_sources_config():
    """Generate comprehensive source definitions"""
    
    # TODO: Define all raw data sources with freshness tests and descriptions
    
    config = """
version: 2

sources:
  - name: raw_data
    description: Raw data from DataCorp production systems
    database: datacorp_prod
    schema: raw
    
    tables:
      - name: users
        description: |
          User account information from the main application database.
          This table contains user profile data that changes over time.
        
        columns:
          - name: user_id
            description: Primary key - unique user identifier
            tests:
              - unique
              - not_null
          
          - name: email
            description: User email address (unique)
            tests:
              - unique
              - not_null
          
          - name: subscription_tier
            description: Current subscription level
            tests:
              - accepted_values:
                  values: ['free', 'basic', 'premium', 'enterprise']
          
          - name: created_at
            description: Account creation timestamp
            tests:
              - not_null
          
          - name: updated_at
            description: Last profile update timestamp
            tests:
              - not_null
        
        freshness:
          warn_after: {count: 6, period: hour}
          error_after: {count: 12, period: hour}
        
        loaded_at_field: _loaded_at
      
      - name: subscriptions
        description: |
          Subscription data tracking user plan changes over time.
          Contains historical subscription events and current status.
        
        columns:
          - name: subscription_id
            description: Primary key for subscription records
            tests:
              - unique
              - not_null
          
          - name: user_id
            description: Foreign key to users table
            tests:
              - not_null
              - relationships:
                  to: source('raw_data', 'users')
                  field: user_id
          
          - name: plan_name
            description: Subscription plan name
            tests:
              - not_null
              - accepted_values:
                  values: ['free', 'basic_monthly', 'basic_yearly', 'premium_monthly', 'premium_yearly', 'enterprise']
          
          - name: status
            description: Current subscription status
            tests:
              - accepted_values:
                  values: ['active', 'cancelled', 'expired', 'trial']
          
          - name: started_at
            description: Subscription start date
            tests:
              - not_null
          
          - name: amount_cents
            description: Subscription amount in cents
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 0
                  inclusive: true
        
        freshness:
          warn_after: {count: 2, period: hour}
          error_after: {count: 6, period: hour}
      
      - name: events
        description: |
          User behavior events from web and mobile applications.
          High-volume append-only table with millions of daily events.
        
        columns:
          - name: event_id
            description: Unique event identifier
            tests:
              - unique
              - not_null
          
          - name: user_id
            description: User who performed the event
            tests:
              - not_null
              - relationships:
                  to: source('raw_data', 'users')
                  field: user_id
          
          - name: event_type
            description: Type of event performed
            tests:
              - not_null
              - accepted_values:
                  values: ['page_view', 'click', 'form_submit', 'purchase', 'login', 'logout', 'feature_use']
          
          - name: event_timestamp
            description: When the event occurred
            tests:
              - not_null
          
          - name: properties
            description: JSON properties specific to event type
        
        freshness:
          warn_after: {count: 30, period: minute}
          error_after: {count: 2, period: hour}
        
        loaded_at_field: _loaded_at
      
      - name: products
        description: |
          Product catalog with features and pricing.
          Changes frequently as new features are added or pricing updated.
        
        columns:
          - name: product_id
            description: Unique product identifier
            tests:
              - unique
              - not_null
          
          - name: product_name
            description: Product display name
            tests:
              - not_null
          
          - name: price_cents
            description: Current price in cents
            tests:
              - not_null
              - dbt_utils.accepted_range:
                  min_value: 0
                  inclusive: true
          
          - name: category
            description: Product category
            tests:
              - not_null
          
          - name: is_active
            description: Whether product is currently available
            tests:
              - not_null
              - accepted_values:
                  values: [true, false]
          
          - name: updated_at
            description: Last product update timestamp
            tests:
              - not_null
        
        freshness:
          warn_after: {count: 24, period: hour}
          error_after: {count: 48, period: hour}
"""
    
    return config.strip()

# =============================================================================
# STAGING MODELS
# =============================================================================

def generate_staging_users_model():
    """Generate staging model for users with data quality checks"""
    
    # TODO: Create staging model with comprehensive data cleaning and validation
    
    model = """
-- models/staging/stg_users.sql
-- Staging model for user data with data quality checks and standardization

{{ config(
    materialized='view',
    tags=['staging', 'users']
) }}

with source as (
    select * from {{ source('raw_data', 'users') }}
),

data_quality_checks as (
    select
        *,
        
        -- Data quality flags
        case
            when user_id is null then 'missing_user_id'
            when email is null or email = '' then 'missing_email'
            when not regexp_like(email, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$') then 'invalid_email_format'
            when subscription_tier not in ('free', 'basic', 'premium', 'enterprise') then 'invalid_subscription_tier'
            when created_at is null then 'missing_created_at'
            when created_at > current_timestamp() then 'future_created_at'
            when updated_at is null then 'missing_updated_at'
            when updated_at < created_at then 'invalid_update_timestamp'
            else 'valid'
        end as data_quality_flag
        
    from source
),

cleaned_data as (
    select
        user_id,
        
        -- Standardize email
        lower(trim(email)) as email,
        
        -- Standardize names
        upper(left(trim(coalesce(first_name, 'Unknown')), 1)) || lower(substring(trim(coalesce(first_name, 'Unknown')), 2)) as first_name,
        upper(left(trim(coalesce(last_name, 'Unknown')), 1)) || lower(substring(trim(coalesce(last_name, 'Unknown')), 2)) as last_name,
        
        -- Standardize subscription tier
        lower(trim(subscription_tier)) as subscription_tier,
        
        -- Handle missing values
        coalesce(country, 'Unknown') as country,
        coalesce(city, 'Unknown') as city,
        
        -- Timestamps
        created_at,
        updated_at,
        
        -- Calculated fields
        date_trunc('day', created_at) as created_date,
        date_trunc('day', updated_at) as updated_date,
        
        -- User tenure
        extract(day from current_timestamp - created_at) as days_since_signup,
        
        -- Data quality
        data_quality_flag,
        
        -- Processing metadata
        current_timestamp() as processed_at
        
    from data_quality_checks
    
    -- Only include valid records
    where data_quality_flag = 'valid'
)

select * from cleaned_data
"""
    
    return model.strip()

def generate_staging_events_model():
    """Generate staging model for events with partitioning"""
    
    # TODO: Create high-performance staging model for large event data
    
    model = """
-- models/staging/stg_events.sql
-- Staging model for user events with performance optimizations

{{ config(
    materialized='view',
    tags=['staging', 'events', 'high_volume']
) }}

with source as (
    select * from {{ source('raw_data', 'events') }}
    
    -- Limit data for development
    {% if target.name == 'dev' %}
        where event_timestamp >= current_date - interval '7 days'
    {% endif %}
),

parsed_events as (
    select
        event_id,
        user_id,
        event_type,
        event_timestamp,
        
        -- Parse JSON properties safely
        case
            when properties is not null then
                case when properties::text ~ '^\\{.*\\}$' then properties::json else null end
            else null
        end as event_properties,
        
        -- Extract common properties
        case
            when properties is not null then
                properties->>'page_url'
            else null
        end as page_url,
        
        case
            when properties is not null then
                properties->>'referrer'
            else null
        end as referrer,
        
        case
            when properties is not null then
                properties->>'device_type'
            else null
        end as device_type,
        
        case
            when properties is not null then
                (properties->>'session_duration')::integer
            else null
        end as session_duration_seconds,
        
        -- Date partitioning
        event_timestamp::date as event_date,
        date_trunc('hour', event_timestamp) as event_hour,
        
        -- Time-based groupings
        extract(hour from event_timestamp) as hour_of_day,
        extract(dayofweek from event_timestamp) as day_of_week,
        
        -- Data quality checks
        case
            when event_id is null then 'missing_event_id'
            when user_id is null then 'missing_user_id'
            when event_type is null then 'missing_event_type'
            when event_timestamp is null then 'missing_timestamp'
            when event_timestamp > current_timestamp then 'future_timestamp'
            when event_timestamp < '2020-01-01'::timestamp then 'too_old_timestamp'
            else 'valid'
        end as data_quality_flag
        
    from source
),

final as (
    select
        event_id,
        user_id,
        event_type,
        event_timestamp,
        event_date,
        event_hour,
        hour_of_day,
        day_of_week,
        
        -- Cleaned properties
        coalesce(page_url, 'unknown') as page_url,
        coalesce(referrer, 'direct') as referrer,
        coalesce(device_type, 'unknown') as device_type,
        coalesce(session_duration_seconds, 0) as session_duration_seconds,
        
        -- Full properties for advanced analysis
        event_properties,
        
        -- Metadata
        data_quality_flag,
        current_timestamp() as processed_at
        
    from parsed_events
    
    -- Only include valid events
    where data_quality_flag = 'valid'
)

select * from final
"""
    
    return model.strip()

# =============================================================================
# INCREMENTAL MODELS
# =============================================================================

def generate_user_behavior_fact():
    """Generate advanced incremental fact table for user behavior"""
    
    # TODO: Create sophisticated incremental model with complex business logic
    
    model = """
-- models/marts/fct_user_behavior.sql
-- Incremental fact table for user behavior analytics with advanced patterns

{{ config(
    materialized='incremental',
    unique_key='behavior_key',
    on_schema_change='sync_all_columns',
    incremental_strategy='merge',
    
    -- Performance optimizations
    partition_by={
        'field': 'behavior_date',
        'data_type': 'date',
        'granularity': 'day'
    },
    cluster_by=['user_id', 'event_type'],
    
    -- Incremental predicates for performance
    incremental_predicates=[
        "behavior_date >= date_sub(current_date(), interval 7 day)"
    ],
    
    tags=['fact', 'incremental', 'user_behavior']
) }}

with events_base as (
    select * from {{ ref('stg_events') }}
    
    {% if is_incremental() %}
        -- Complex incremental logic: process events from last 2 days to handle late-arriving data
        where event_date >= (
            select coalesce(max(behavior_date), '1900-01-01') - interval '2 days'
            from {{ this }}
        )
    {% endif %}
),

user_context as (
    select * from {{ ref('stg_users') }}
),

subscription_context as (
    select * from {{ ref('stg_subscriptions') }}
    where status = 'active'
),

-- Aggregate events by user and day for behavior analysis
daily_user_behavior as (
    select
        e.user_id,
        e.event_date as behavior_date,
        
        -- Event counts by type
        count(*) as total_events,
        count(case when e.event_type = 'page_view' then 1 end) as page_views,
        count(case when e.event_type = 'click' then 1 end) as clicks,
        count(case when e.event_type = 'form_submit' then 1 end) as form_submits,
        count(case when e.event_type = 'purchase' then 1 end) as purchases,
        count(case when e.event_type = 'feature_use' then 1 end) as feature_uses,
        
        -- Session metrics
        count(distinct e.event_hour) as active_hours,
        sum(e.session_duration_seconds) as total_session_duration,
        avg(e.session_duration_seconds) as avg_session_duration,
        
        -- Engagement metrics
        count(distinct e.page_url) as unique_pages_visited,
        count(distinct e.device_type) as device_types_used,
        
        -- Time-based patterns
        min(e.event_timestamp) as first_event_time,
        max(e.event_timestamp) as last_event_time,
        
        -- Calculate engagement score (custom business logic)
        (
            count(case when e.event_type = 'page_view' then 1 end) * 1 +
            count(case when e.event_type = 'click' then 1 end) * 2 +
            count(case when e.event_type = 'form_submit' then 1 end) * 5 +
            count(case when e.event_type = 'purchase' then 1 end) * 10 +
            count(case when e.event_type = 'feature_use' then 1 end) * 3
        ) as engagement_score
        
    from events_base e
    group by e.user_id, e.event_date
),

-- Enrich with user and subscription context
enriched_behavior as (
    select
        b.*,
        
        -- User context
        u.email,
        u.first_name,
        u.last_name,
        u.subscription_tier,
        u.country,
        u.days_since_signup,
        
        -- Subscription context
        s.plan_name,
        s.amount_cents as subscription_amount_cents,
        
        -- Calculate user lifecycle stage
        case
            when u.days_since_signup <= 7 then 'new_user'
            when u.days_since_signup <= 30 then 'onboarding'
            when u.days_since_signup <= 90 then 'growing'
            when u.days_since_signup <= 365 then 'established'
            else 'mature'
        end as user_lifecycle_stage,
        
        -- Calculate engagement level
        case
            when b.engagement_score >= 100 then 'highly_engaged'
            when b.engagement_score >= 50 then 'moderately_engaged'
            when b.engagement_score >= 20 then 'lightly_engaged'
            else 'minimally_engaged'
        end as engagement_level,
        
        -- Calculate behavior flags
        case when b.purchases > 0 then true else false end as is_purchaser,
        case when b.feature_uses > 0 then true else false end as is_feature_user,
        case when b.active_hours >= 8 then true else false end as is_power_user,
        
        -- Create composite key for incremental updates
        concat(b.user_id, '-', b.behavior_date) as behavior_key,
        
        -- Metadata
        current_timestamp() as processed_at
        
    from daily_user_behavior b
    left join user_context u on b.user_id = u.user_id
    left join subscription_context s on b.user_id = s.user_id
),

-- Add rolling metrics for trend analysis
final_with_trends as (
    select
        *,
        
        -- 7-day rolling averages
        avg(engagement_score) over (
            partition by user_id 
            order by behavior_date 
            rows between 6 preceding and current row
        ) as engagement_score_7d_avg,
        
        avg(total_events) over (
            partition by user_id 
            order by behavior_date 
            rows between 6 preceding and current row
        ) as total_events_7d_avg,
        
        -- Trend indicators
        case
            when engagement_score > lag(engagement_score, 1) over (
                partition by user_id order by behavior_date
            ) then 'increasing'
            when engagement_score < lag(engagement_score, 1) over (
                partition by user_id order by behavior_date
            ) then 'decreasing'
            else 'stable'
        end as engagement_trend,
        
        -- Days since last activity (for churn prediction)
        datediff('day', behavior_date, current_date()) as days_since_activity
        
    from enriched_behavior
)

select * from final_with_trends
"""
    
    return model.strip()

# =============================================================================
# SNAPSHOTS
# =============================================================================

def generate_users_snapshot():
    """Generate snapshot for tracking user changes over time"""
    
    # TODO: Create comprehensive snapshot for slowly changing user dimensions
    
    snapshot = """
-- snapshots/users_snapshot.sql
-- Snapshot to track user profile changes over time (SCD Type 2)

{% snapshot users_snapshot %}
    {{
        config(
            target_database='analytics',
            target_schema='snapshots',
            unique_key='user_id',
            strategy='timestamp',
            updated_at='updated_at',
            invalidate_hard_deletes=true,
            tags=['snapshot', 'scd_type_2']
        )
    }}
    
    select
        user_id,
        email,
        first_name,
        last_name,
        subscription_tier,
        country,
        city,
        created_at,
        updated_at,
        
        -- Additional fields for analysis
        case
            when subscription_tier in ('premium', 'enterprise') then 'paid'
            else 'free'
        end as user_type,
        
        -- Calculate user value segment at time of snapshot
        case
            when subscription_tier = 'enterprise' then 'high_value'
            when subscription_tier = 'premium' then 'medium_value'
            when subscription_tier = 'basic' then 'low_value'
            else 'no_value'
        end as value_segment
        
    from {{ source('raw_data', 'users') }}
    
{% endsnapshot %}
"""
    
    return snapshot.strip()

# =============================================================================
# ADVANCED MACROS
# =============================================================================

def generate_rfm_macro():
    """Generate macro for RFM score calculation"""
    
    # TODO: Create reusable macro for RFM analysis
    
    macro = """
-- macros/calculate_rfm_scores.sql
-- Macro to calculate RFM (Recency, Frequency, Monetary) scores

{% macro calculate_rfm_scores(
    table_name, 
    user_id_column='user_id', 
    date_column='order_date', 
    amount_column='amount',
    as_of_date='current_date()'
) %}

    -- Calculate recency (days since last purchase)
    datediff('day', max({{ date_column }}), {{ as_of_date }}) as recency_days,
    
    -- Calculate frequency (number of purchases)
    count(*) as frequency_count,
    
    -- Calculate monetary (total amount spent)
    sum({{ amount_column }}) as monetary_total
    
    -- Calculate RFM scores (1-5 scale)
    case
        when datediff('day', max({{ date_column }}), {{ as_of_date }}) <= {{ var('rfm_recency_days')[0] }} then 5
        when datediff('day', max({{ date_column }}), {{ as_of_date }}) <= {{ var('rfm_recency_days')[1] }} then 4
        when datediff('day', max({{ date_column }}), {{ as_of_date }}) <= {{ var('rfm_recency_days')[2] }} then 3
        when datediff('day', max({{ date_column }}), {{ as_of_date }}) <= {{ var('rfm_recency_days')[3] }} then 2
        else 1
    end as recency_score,
    
    case
        when count(*) >= {{ var('rfm_frequency_orders')[3] }} then 5
        when count(*) >= {{ var('rfm_frequency_orders')[2] }} then 4
        when count(*) >= {{ var('rfm_frequency_orders')[1] }} then 3
        when count(*) >= {{ var('rfm_frequency_orders')[0] }} then 2
        else 1
    end as frequency_score,
    
    case
        when sum({{ amount_column }}) >= {{ var('rfm_monetary_amounts')[3] }} then 5
        when sum({{ amount_column }}) >= {{ var('rfm_monetary_amounts')[2] }} then 4
        when sum({{ amount_column }}) >= {{ var('rfm_monetary_amounts')[1] }} then 3
        when sum({{ amount_column }}) >= {{ var('rfm_monetary_amounts')[0] }} then 2
        else 1
    end as monetary_score

    
    group by {{ user_id_column }}

{% endmacro %}
"""
    
    return macro.strip()

# =============================================================================
# TESTING
# =============================================================================

def generate_rfm_validation_test():
    """Generate custom test for RFM score validation"""
    
    # TODO: Create comprehensive test for RFM business logic
    
    test = """
-- tests/assert_rfm_scores_valid.sql
-- Test that RFM scores are within expected ranges and logically consistent

with rfm_data as (
    select
        user_id,
        recency_score,
        frequency_score,
        monetary_score,
        recency_days,
        frequency_count,
        monetary_total
    from {{ ref('int_user_rfm_scores') }}
),

validation_failures as (
    select
        user_id,
        'invalid_score_range' as failure_type,
        'RFM scores must be between 1 and 5' as failure_reason
    from rfm_data
    where recency_score not between 1 and 5
       or frequency_score not between 1 and 5
       or monetary_score not between 1 and 5
    
    union all
    
    select
        user_id,
        'logical_inconsistency' as failure_type,
        'High frequency score but low order count' as failure_reason
    from rfm_data
    where frequency_score = 5 and frequency_count < {{ var('rfm_frequency_orders')[2] }}
    
    union all
    
    select
        user_id,
        'logical_inconsistency' as failure_type,
        'High monetary score but low total spend' as failure_reason
    from rfm_data
    where monetary_score = 5 and monetary_total < {{ var('rfm_monetary_amounts')[2] }}
    
    union all
    
    select
        user_id,
        'logical_inconsistency' as failure_type,
        'High recency score but old last purchase' as failure_reason
    from rfm_data
    where recency_score = 5 and recency_days > {{ var('rfm_recency_days')[1] }}
)

select * from validation_failures
"""
    
    return test.strip()

# =============================================================================
# EXERCISE INSTRUCTIONS
# =============================================================================

def print_exercise_instructions():
    """Print detailed exercise instructions"""
    
    print("🎯 dbt Deep Dive Exercise - DataCorp Advanced Analytics")
    print("=" * 65)
    
    print("\n📋 REQUIREMENTS:")
    print("1. Build advanced incremental models with complex business logic")
    print("2. Create snapshots for slowly changing dimensions")
    print("3. Implement performance optimizations (partitioning, clustering)")
    print("4. Add comprehensive data quality checks and error handling")
    print("5. Build RFM analysis and customer segmentation")
    print("6. Create detailed documentation and testing")
    
    print("\n🏗️ PROJECT ARCHITECTURE:")
    print("""
    DataCorp dbt Project Structure:
    
    ├── dbt_project.yml          # Project configuration with optimizations
    ├── profiles.yml             # Multi-environment setup
    ├── packages.yml             # Advanced dbt packages
    │
    ├── models/
    │   ├── sources.yml          # Raw data source definitions
    │   │
    │   ├── staging/             # Data cleaning and standardization
    │   │   ├── stg_users.sql           # User data with quality checks
    │   │   ├── stg_events.sql          # High-volume event data
    │   │   ├── stg_subscriptions.sql   # Subscription data
    │   │   └── stg_products.sql        # Product catalog
    │   │
    │   ├── intermediate/        # Business logic and aggregations
    │   │   ├── int_user_event_aggregates.sql    # Daily user behavior
    │   │   ├── int_subscription_history.sql     # Subscription changes
    │   │   └── int_user_rfm_scores.sql          # RFM analysis
    │   │
    │   ├── marts/               # Final business tables
    │   │   ├── fct_user_behavior.sql     # Incremental fact table
    │   │   ├── dim_users_scd.sql         # User dimension with SCD
    │   │   └── fct_subscription_metrics.sql  # Subscription analytics
    │   │
    │   └── schema.yml           # Model documentation and tests
    │
    ├── snapshots/               # Slowly changing dimensions
    │   ├── users_snapshot.sql          # Track user profile changes
    │   └── products_snapshot.sql       # Track product changes
    │
    ├── tests/                   # Custom data tests
    │   ├── assert_user_behavior_quality.sql    # Behavior data validation
    │   └── assert_rfm_scores_valid.sql         # RFM logic validation
    │
    └── macros/                  # Reusable SQL functions
        ├── calculate_rfm_scores.sql     # RFM calculation macro
        └── get_customer_segments.sql    # Segmentation logic
    """)
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("• Incremental models process only new/changed data efficiently")
    print("• Snapshots capture historical changes in user and product data")
    print("• Performance optimizations handle millions of daily events")
    print("• Data quality checks prevent bad data from propagating")
    print("• RFM analysis provides actionable customer insights")
    print("• Comprehensive testing validates both technical and business logic")
    print("• Documentation explains business context and technical decisions")
    
    print("\n🚀 GETTING STARTED:")
    print("1. Review the project structure and configuration files")
    print("2. Implement the staging models with data quality checks")
    print("3. Build intermediate models with business logic")
    print("4. Create incremental mart models with performance optimizations")
    print("5. Set up snapshots for slowly changing dimensions")
    print("6. Add comprehensive testing and validation")
    print("7. Document models with business context")
    print("8. Test the complete pipeline with sample data")

if __name__ == "__main__":
    print_exercise_instructions()
    
    print("\n" + "="*65)
    print("🎯 Ready to build enterprise-scale dbt analytics!")
    print("Complete the TODOs above to create a production-ready dbt project.")
    print("="*65)
