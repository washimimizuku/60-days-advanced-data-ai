# Day 18: dbt Advanced - Custom Materializations, Packages, Advanced Analytics

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** advanced dbt macros and Jinja templating for complex analytics
- **Develop** custom materializations and reusable package components
- **Implement** sophisticated analytics patterns (cohort analysis, attribution modeling)
- **Optimize** dbt performance with advanced techniques and monitoring
- **Build** enterprise-grade analytics engineering workflows with best practices

---

## Theory

### Advanced Macro Development

Building on Day 17's foundation, we'll explore sophisticated macro patterns that enable complex analytics and code reusability at enterprise scale.

#### 1. Advanced Jinja Patterns

**Dynamic SQL Generation**:
```sql
-- macros/generate_pivot_table.sql
{% macro generate_pivot_table(table_name, group_by_col, pivot_col, value_col, agg_func='sum') %}

  {% set pivot_values_query %}
    select distinct {{ pivot_col }}
    from {{ table_name }}
    where {{ pivot_col }} is not null
    order by {{ pivot_col }}
  {% endset %}

  {% set results = run_query(pivot_values_query) %}
  {% if execute %}
    {% set pivot_values = results.columns[0].values() %}
  {% else %}
    {% set pivot_values = [] %}
  {% endif %}

  select
    {{ group_by_col }},
    {% for value in pivot_values %}
    {{ agg_func }}(case when {{ pivot_col }} = '{{ value }}' then {{ value_col }} end) as {{ value | replace(' ', '_') | lower }}
    {%- if not loop.last -%},{%- endif %}
    {% endfor %}
  from {{ table_name }}
  group by {{ group_by_col }}

{% endmacro %}
```

**Conditional Logic and Loops**:
```sql
-- macros/generate_cohort_analysis.sql
{% macro generate_cohort_analysis(
    table_name, 
    user_id_col, 
    date_col, 
    cohort_periods=['week', 'month', 'quarter']
) %}

  {% for period in cohort_periods %}
  
  with {{ period }}_cohorts as (
    select
      {{ user_id_col }},
      date_trunc('{{ period }}', {{ date_col }}) as cohort_{{ period }},
      date_trunc('{{ period }}', {{ date_col }}) as period_{{ period }},
      
      -- Calculate period number
      {% if period == 'week' %}
        floor(datediff('day', date_trunc('{{ period }}', {{ date_col }}), {{ date_col }}) / 7) as period_number
      {% elif period == 'month' %}
        datediff('month', date_trunc('{{ period }}', {{ date_col }}), {{ date_col }}) as period_number
      {% elif period == 'quarter' %}
        datediff('quarter', date_trunc('{{ period }}', {{ date_col }}), {{ date_col }}) as period_number
      {% endif %}
      
    from {{ table_name }}
  ),
  
  {{ period }}_cohort_sizes as (
    select
      cohort_{{ period }},
      count(distinct {{ user_id_col }}) as cohort_size
    from {{ period }}_cohorts
    where period_number = 0
    group by cohort_{{ period }}
  ),
  
  {{ period }}_cohort_retention as (
    select
      c.cohort_{{ period }},
      c.period_number,
      count(distinct c.{{ user_id_col }}) as users_active,
      s.cohort_size,
      round(count(distinct c.{{ user_id_col }}) * 100.0 / s.cohort_size, 2) as retention_rate
    from {{ period }}_cohorts c
    join {{ period }}_cohort_sizes s on c.cohort_{{ period }} = s.cohort_{{ period }}
    group by c.cohort_{{ period }}, c.period_number, s.cohort_size
  )
  
  select 
    '{{ period }}' as cohort_type,
    *
  from {{ period }}_cohort_retention
  
  {% if not loop.last %}
  union all
  {% endif %}
  
  {% endfor %}

{% endmacro %}
```

#### 2. Advanced Testing Macros

**Custom Generic Tests**:
```sql
-- macros/test_referential_integrity.sql
{% test referential_integrity(model, column_name, to, field) %}

  with parent as (
    select {{ field }} as id
    from {{ to }}
  ),
  
  child as (
    select {{ column_name }} as id
    from {{ model }}
    where {{ column_name }} is not null
  ),
  
  validation_errors as (
    select c.id
    from child c
    left join parent p on c.id = p.id
    where p.id is null
  )
  
  select * from validation_errors

{% endtest %}
```

**Business Logic Tests**:
```sql
-- macros/test_business_rules.sql
{% test customer_lifetime_value_logic(model) %}

  with validation_errors as (
    select
      customer_id,
      total_orders,
      total_spent,
      avg_order_value,
      customer_segment
    from {{ model }}
    where
      -- CLV should equal total spent
      abs(total_spent - (total_orders * avg_order_value)) > 0.01
      
      -- Segment logic validation
      or (customer_segment = 'high_value' and total_spent < 1000)
      or (customer_segment = 'medium_value' and (total_spent < 500 or total_spent >= 1000))
      or (customer_segment = 'low_value' and total_spent >= 500)
      
      -- Business rule: customers with orders should have positive spend
      or (total_orders > 0 and total_spent <= 0)
  )
  
  select * from validation_errors

{% endtest %}
```

#### 3. Performance Optimization Macros

**Dynamic Partitioning**:
```sql
-- macros/optimize_table_performance.sql
{% macro optimize_table_performance(table_name, partition_column, cluster_columns=[]) %}

  {% set optimization_sql %}
    -- Add partitioning
    alter table {{ table_name }}
    add partition ({{ partition_column }});
    
    {% if cluster_columns %}
    -- Add clustering
    alter table {{ table_name }}
    cluster by ({{ cluster_columns | join(', ') }});
    {% endif %}
    
    -- Update table statistics
    analyze table {{ table_name }} compute statistics;
    
    {% for column in cluster_columns %}
    analyze table {{ table_name }} compute statistics for columns {{ column }};
    {% endfor %}
  {% endset %}
  
  {{ return(optimization_sql) }}

{% endmacro %}
```

### Advanced Package Development

#### 1. Creating Custom Packages

**Package Structure**:
```
my_analytics_package/
├── dbt_project.yml
├── README.md
├── macros/
│   ├── analytics/
│   │   ├── cohort_analysis.sql
│   │   ├── attribution_modeling.sql
│   │   └── customer_segmentation.sql
│   ├── utils/
│   │   ├── data_quality.sql
│   │   └── performance_optimization.sql
│   └── materializations/
│       └── custom_incremental.sql
├── models/
│   └── example/
│       ├── example_cohort_analysis.sql
│       └── example_attribution_model.sql
└── tests/
    └── generic/
        ├── test_cohort_logic.sql
        └── test_attribution_logic.sql
```

**Package Configuration**:
```yaml
# dbt_project.yml for custom package
name: 'analytics_engineering_toolkit'
version: '1.0.0'
config-version: 2

require-dbt-version: ">=1.0.0"

macro-paths: ["macros"]
model-paths: ["models"]

models:
  analytics_engineering_toolkit:
    example:
      +materialized: view
      +tags: ["example"]

vars:
  # Default configuration
  cohort_periods: ['week', 'month', 'quarter']
  attribution_window_days: 30
  customer_segments:
    high_value: 1000
    medium_value: 500
    low_value: 100
```

#### 2. Advanced Package Integration

**Multi-Package Coordination**:
```yaml
# packages.yml
packages:
  - package: dbt-labs/dbt_utils
    version: 1.1.1
  
  - package: calogica/dbt_expectations
    version: 0.10.1
  
  - package: dbt-labs/audit_helper
    version: 0.9.0
  
  - package: elementary-data/elementary
    version: 0.13.1
  
  # Custom internal package
  - git: "https://github.com/company/analytics-toolkit.git"
    revision: v2.1.0
  
  # Local development package
  - local: ../shared_analytics_macros
```

### Custom Materializations

#### 1. Advanced Incremental Strategies

**Custom Incremental Materialization**:
```sql
-- macros/materializations/custom_incremental.sql
{% materialization custom_incremental, default %}
  
  {%- set unique_key = config.get('unique_key') -%}
  {%- set full_refresh_mode = (should_full_refresh()) -%}
  {%- set on_schema_change = incremental_validate_on_schema_change(config.get('on_schema_change'), default='ignore') -%}
  
  {% set target_relation = this %}
  {% set existing_relation = load_relation(this) %}
  {% set tmp_relation = make_temp_relation(this) %}
  
  {{ run_hooks(pre_hooks, inside_transaction=False) }}
  
  -- Setup
  {{ run_hooks(pre_hooks, inside_transaction=True) }}
  
  {% if existing_relation is none %}
    -- First run: create table
    {% set build_sql = create_table_as(False, target_relation, sql) %}
  
  {% elif full_refresh_mode %}
    -- Full refresh: drop and recreate
    {% do adapter.drop_relation(existing_relation) %}
    {% set build_sql = create_table_as(False, target_relation, sql) %}
  
  {% else %}
    -- Incremental run: custom merge logic
    {% set build_sql = custom_incremental_merge(target_relation, sql, unique_key) %}
  
  {% endif %}
  
  {% call statement('main') -%}
    {{ build_sql }}
  {%- endcall %}
  
  {{ run_hooks(post_hooks, inside_transaction=True) }}
  
  {% do persist_docs(target_relation, model) %}
  
  {{ run_hooks(post_hooks, inside_transaction=False) }}
  
  {{ return({'relations': [target_relation]}) }}

{% endmaterialization %}

{% macro custom_incremental_merge(target_relation, sql, unique_key) %}
  
  {% set tmp_relation = make_temp_relation(target_relation) %}
  
  -- Create temp table with new data
  {{ create_table_as(True, tmp_relation, sql) }}
  
  -- Custom merge logic with conflict resolution
  merge into {{ target_relation }} as target
  using {{ tmp_relation }} as source
  on target.{{ unique_key }} = source.{{ unique_key }}
  
  when matched and source._updated_at > target._updated_at then
    update set *
  
  when not matched then
    insert *
  
  when not matched by source and target._is_deleted = false then
    update set _is_deleted = true, _deleted_at = current_timestamp()

{% endmacro %}
```

### Advanced Analytics Patterns

#### 1. Attribution Modeling

**Multi-Touch Attribution**:
```sql
-- macros/attribution_modeling.sql
{% macro calculate_attribution(
    events_table,
    conversions_table,
    attribution_window_days=30,
    attribution_model='linear'
) %}

  with attribution_events as (
    select
      e.user_id,
      e.event_timestamp,
      e.channel,
      e.campaign,
      c.conversion_timestamp,
      c.conversion_value,
      
      -- Calculate time to conversion
      datediff('hour', e.event_timestamp, c.conversion_timestamp) as hours_to_conversion,
      
      -- Rank touchpoints
      row_number() over (
        partition by c.user_id, c.conversion_timestamp 
        order by e.event_timestamp
      ) as touchpoint_sequence,
      
      count(*) over (
        partition by c.user_id, c.conversion_timestamp
      ) as total_touchpoints
      
    from {{ events_table }} e
    join {{ conversions_table }} c 
      on e.user_id = c.user_id
      and e.event_timestamp <= c.conversion_timestamp
      and e.event_timestamp >= dateadd('day', -{{ attribution_window_days }}, c.conversion_timestamp)
  ),
  
  attribution_weights as (
    select
      *,
      case 
        when '{{ attribution_model }}' = 'first_touch' then
          case when touchpoint_sequence = 1 then 1.0 else 0.0 end
        
        when '{{ attribution_model }}' = 'last_touch' then
          case when touchpoint_sequence = total_touchpoints then 1.0 else 0.0 end
        
        when '{{ attribution_model }}' = 'linear' then
          1.0 / total_touchpoints
        
        when '{{ attribution_model }}' = 'time_decay' then
          power(2, -(total_touchpoints - touchpoint_sequence))
        
        when '{{ attribution_model }}' = 'u_shaped' then
          case 
            when touchpoint_sequence = 1 then 0.4
            when touchpoint_sequence = total_touchpoints then 0.4
            else 0.2 / (total_touchpoints - 2)
          end
        
        else 1.0 / total_touchpoints  -- Default to linear
      end as attribution_weight
      
    from attribution_events
  )
  
  select
    user_id,
    channel,
    campaign,
    conversion_timestamp,
    sum(conversion_value * attribution_weight) as attributed_value,
    sum(attribution_weight) as attribution_weight_total,
    count(*) as attributed_conversions
  from attribution_weights
  group by user_id, channel, campaign, conversion_timestamp

{% endmacro %}
```

#### 2. Advanced Cohort Analysis

**Behavioral Cohorts**:
```sql
-- macros/behavioral_cohorts.sql
{% macro generate_behavioral_cohorts(
    events_table,
    cohort_definition_sql,
    analysis_periods=['week', 'month']
) %}

  with cohort_definitions as (
    {{ cohort_definition_sql }}
  ),
  
  user_cohorts as (
    select
      e.user_id,
      e.event_date,
      cd.cohort_name,
      cd.cohort_date,
      
      {% for period in analysis_periods %}
      datediff('{{ period }}', cd.cohort_date, e.event_date) as periods_since_cohort_{{ period }}
      {%- if not loop.last -%},{%- endif %}
      {% endfor %}
      
    from {{ events_table }} e
    join cohort_definitions cd on e.user_id = cd.user_id
  ),
  
  {% for period in analysis_periods %}
  cohort_{{ period }}_analysis as (
    select
      cohort_name,
      cohort_date,
      periods_since_cohort_{{ period }} as period_number,
      count(distinct user_id) as active_users,
      
      -- Calculate retention rate
      count(distinct user_id) * 100.0 / first_value(count(distinct user_id)) over (
        partition by cohort_name, cohort_date 
        order by periods_since_cohort_{{ period }}
      ) as retention_rate_{{ period }}
      
    from user_cohorts
    where periods_since_cohort_{{ period }} >= 0
    group by cohort_name, cohort_date, periods_since_cohort_{{ period }}
  )
  
  select 
    '{{ period }}' as analysis_period,
    *
  from cohort_{{ period }}_analysis
  
  {% if not loop.last %}
  union all
  {% endif %}
  {% endfor %}

{% endmacro %}
```

#### 3. Customer Lifetime Value Modeling

**Predictive CLV**:
```sql
-- macros/clv_modeling.sql
{% macro calculate_predictive_clv(
    customer_metrics_table,
    prediction_horizon_months=12,
    discount_rate=0.1
) %}

  with customer_features as (
    select
      customer_id,
      
      -- Historical metrics
      total_orders,
      total_spent,
      avg_order_value,
      days_since_first_order,
      days_since_last_order,
      
      -- Calculate frequency (orders per month)
      case 
        when days_since_first_order > 0 then
          total_orders * 30.0 / days_since_first_order
        else 0
      end as monthly_frequency,
      
      -- Calculate recency score (inverse of days since last order)
      case
        when days_since_last_order <= 30 then 1.0
        when days_since_last_order <= 90 then 0.8
        when days_since_last_order <= 180 then 0.6
        when days_since_last_order <= 365 then 0.4
        else 0.2
      end as recency_multiplier,
      
      -- Calculate churn probability (simplified model)
      case
        when days_since_last_order > 365 then 0.9
        when days_since_last_order > 180 then 0.7
        when days_since_last_order > 90 then 0.4
        when days_since_last_order > 30 then 0.2
        else 0.1
      end as churn_probability
      
    from {{ customer_metrics_table }}
  ),
  
  clv_predictions as (
    select
      customer_id,
      
      -- Predicted monthly value
      monthly_frequency * avg_order_value * recency_multiplier as predicted_monthly_value,
      
      -- Predicted lifetime (months until churn)
      case
        when churn_probability > 0 then
          -ln(churn_probability) / ln(1 + {{ discount_rate }}/12)
        else {{ prediction_horizon_months }}
      end as predicted_lifetime_months,
      
      -- Calculate CLV components
      monthly_frequency,
      avg_order_value,
      recency_multiplier,
      churn_probability
      
    from customer_features
  )
  
  select
    customer_id,
    predicted_monthly_value,
    predicted_lifetime_months,
    
    -- Calculate discounted CLV
    case
      when predicted_lifetime_months > 0 then
        predicted_monthly_value * 
        (1 - power(1 + {{ discount_rate }}/12, -predicted_lifetime_months)) / 
        ({{ discount_rate }}/12)
      else 0
    end as predicted_clv,
    
    -- Confidence intervals (simplified)
    case
      when churn_probability < 0.3 then 'high'
      when churn_probability < 0.6 then 'medium'
      else 'low'
    end as prediction_confidence,
    
    monthly_frequency,
    avg_order_value,
    recency_multiplier,
    churn_probability
    
  from clv_predictions

{% endmacro %}
```

### Performance Monitoring and Optimization

#### 1. dbt Performance Monitoring

**Model Performance Tracking**:
```sql
-- macros/performance_monitoring.sql
{% macro track_model_performance() %}

  {% if execute %}
    {% set performance_data = {
      'model_name': this.name,
      'schema': this.schema,
      'run_started_at': run_started_at,
      'invocation_id': invocation_id,
      'dbt_version': dbt_version
    } %}
    
    {% do log("Model performance data: " ~ performance_data, info=true) %}
  {% endif %}

{% endmacro %}
```

**Query Optimization Analysis**:
```sql
-- analysis/query_performance_analysis.sql
-- Analysis to identify slow-running models and optimization opportunities

with model_run_stats as (
  select
    model_name,
    avg(execution_time_seconds) as avg_execution_time,
    max(execution_time_seconds) as max_execution_time,
    count(*) as total_runs,
    sum(case when execution_time_seconds > 300 then 1 else 0 end) as slow_runs
  from {{ ref('dbt_run_results') }}
  where run_date >= current_date() - interval '30 days'
  group by model_name
),

optimization_recommendations as (
  select
    model_name,
    avg_execution_time,
    max_execution_time,
    total_runs,
    slow_runs,
    
    case
      when avg_execution_time > 600 then 'Consider partitioning or incremental materialization'
      when avg_execution_time > 300 then 'Review query complexity and indexing'
      when slow_runs > total_runs * 0.2 then 'Investigate performance variability'
      else 'Performance acceptable'
    end as recommendation
    
  from model_run_stats
)

select * from optimization_recommendations
order by avg_execution_time desc
```

---

## 💻 Hands-On Exercise (40 minutes)

Build an advanced dbt analytics platform with custom macros, packages, and sophisticated analytics patterns.

**Scenario**: You're the Lead Analytics Engineer at "InnovateCorp", a high-growth technology company. You need to build a sophisticated analytics platform that includes custom attribution modeling, cohort analysis, and predictive customer lifetime value calculations.

**Requirements**:
1. **Custom Macro Library**: Build reusable macros for complex analytics
2. **Package Development**: Create a custom analytics package
3. **Attribution Modeling**: Implement multi-touch attribution analysis
4. **Cohort Analysis**: Build behavioral and temporal cohort analysis
5. **CLV Modeling**: Create predictive customer lifetime value models
6. **Performance Optimization**: Implement monitoring and optimization patterns

**Data Sources**:
- Customer events and touchpoints
- Conversion and revenue data
- Product usage metrics
- Customer profile information

See `exercise.py` for starter code and detailed requirements.

### 🐳 Development Environment Setup

**Quick Start with Docker**:
```bash
# Start the complete development environment
./setup.sh

# Access the dbt container
docker-compose exec dbt bash

# Run dbt models
dbt run

# Run tests
dbt test

# Generate and serve documentation
dbt docs generate && dbt docs serve
```

**Manual Setup**:
1. Install dbt-core and dbt-postgres
2. Configure PostgreSQL database
3. Set up dbt profiles
4. Install package dependencies

**Infrastructure Included**:
- 🐳 **Docker Compose**: Complete development environment
- 🗄️ **PostgreSQL**: Sample database with test data
- 📦 **dbt Project**: Pre-configured with macros and models
- 🧪 **Sample Data**: Customer events, conversions, and metrics
- 🔧 **Setup Script**: Automated environment initialization

---

## 📚 Resources

- **dbt Jinja Functions**: [docs.getdbt.com/reference/dbt-jinja-functions](https://docs.getdbt.com/reference/dbt-jinja-functions)
- **Custom Materializations**: [docs.getdbt.com/guides/create-new-materializations](https://docs.getdbt.com/guides/create-new-materializations)
- **Package Development**: [docs.getdbt.com/docs/build/packages](https://docs.getdbt.com/docs/build/packages)
- **Advanced Macros**: [docs.getdbt.com/docs/build/jinja-macros](https://docs.getdbt.com/docs/build/jinja-macros)
- **dbt Hub**: [hub.getdbt.com](https://hub.getdbt.com/) - Community packages
- **Performance Optimization**: [docs.getdbt.com/guides/best-practices/how-we-structure/5-the-rest-of-the-project](https://docs.getdbt.com/guides/best-practices/how-we-structure/5-the-rest-of-the-project)

---

## 🎯 Key Takeaways

- **Advanced macros enable sophisticated analytics** with reusable, parameterized SQL generation
- **Custom packages promote code reuse** across teams and projects
- **Attribution modeling provides insights** into marketing effectiveness and customer journeys
- **Cohort analysis reveals user behavior patterns** and retention trends over time
- **Predictive CLV modeling** enables data-driven customer investment decisions
- **Performance monitoring is essential** for maintaining analytics platform health
- **Custom materializations** can optimize specific use cases beyond standard options
- **Jinja templating is powerful** for dynamic SQL generation and complex logic

---

## 🚀 What's Next?

Tomorrow (Day 19), you'll learn **Data Quality in Production** - comprehensive data quality frameworks, monitoring, and automated validation.

**Preview**: You'll explore advanced data quality patterns, implement comprehensive monitoring systems, and build automated data validation pipelines that ensure data reliability at enterprise scale!

---

## ✅ Before Moving On

- [ ] Understand advanced Jinja patterns and macro development
- [ ] Can create custom materializations for specific use cases
- [ ] Know how to develop and distribute dbt packages
- [ ] Can implement sophisticated analytics patterns (attribution, cohorts, CLV)
- [ ] Understand performance monitoring and optimization techniques
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐⭐ (Expert-Level Analytics Engineering)

Ready to master enterprise analytics engineering! 🚀
