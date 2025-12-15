-- macros/analytics/cohort_analysis.sql
-- Advanced cohort analysis with behavioral and temporal dimensions

{% macro generate_cohort_analysis(
    table_name, 
    user_id_col='user_id', 
    date_col='event_date',
    cohort_period='month'
) %}

  with cohort_definitions as (
    select
      {{ user_id_col }},
      date_trunc('{{ cohort_period }}', min({{ date_col }})) as cohort_date
    from {{ table_name }}
    group by {{ user_id_col }}
  ),
  
  user_activity as (
    select
      t.{{ user_id_col }},
      t.{{ date_col }},
      cd.cohort_date,
      
      -- Calculate periods since cohort
      {% if cohort_period == 'day' %}
        (t.{{ date_col }} - cd.cohort_date) as periods_since_cohort
      {% elif cohort_period == 'week' %}
        floor((t.{{ date_col }} - cd.cohort_date) / 7) as periods_since_cohort
      {% elif cohort_period == 'month' %}
        extract(year from t.{{ date_col }}) * 12 + extract(month from t.{{ date_col }}) - 
        (extract(year from cd.cohort_date) * 12 + extract(month from cd.cohort_date)) as periods_since_cohort
      {% endif %}
      
    from {{ table_name }} t
    join cohort_definitions cd on t.{{ user_id_col }} = cd.{{ user_id_col }}
  ),
  
  cohort_sizes as (
    select
      cohort_date,
      count(distinct {{ user_id_col }}) as cohort_size
    from cohort_definitions
    group by cohort_date
  ),
  
  cohort_analysis as (
    select
      ua.cohort_date,
      ua.periods_since_cohort as period_number,
      
      -- Core retention metrics
      count(distinct ua.{{ user_id_col }}) as active_users,
      cs.cohort_size,
      
      round(
        count(distinct ua.{{ user_id_col }}) * 100.0 / nullif(cs.cohort_size, 0), 
        2
      ) as retention_rate
      
    from user_activity ua
    join cohort_sizes cs on ua.cohort_date = cs.cohort_date
    
    where ua.periods_since_cohort >= 0
    group by 
      ua.cohort_date, 
      ua.periods_since_cohort,
      cs.cohort_size
  )
  
  select 
    '{{ cohort_period }}' as analysis_period,
    cohort_date,
    period_number,
    active_users,
    cohort_size,
    retention_rate
  from cohort_analysis
  order by cohort_date, period_number

{% endmacro %}