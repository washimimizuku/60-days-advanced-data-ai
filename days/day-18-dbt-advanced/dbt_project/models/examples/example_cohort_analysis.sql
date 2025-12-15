-- models/examples/example_cohort_analysis.sql
-- Example cohort retention analysis

{{ config(materialized='view') }}

select
  cohort_date,
  period_number,
  active_users,
  cohort_size,
  retention_rate,
  
  -- Additional insights
  case 
    when period_number = 0 then 100.0
    else retention_rate
  end as adjusted_retention_rate,
  
  case
    when retention_rate >= 80 then 'excellent'
    when retention_rate >= 60 then 'good'
    when retention_rate >= 40 then 'average'
    when retention_rate >= 20 then 'poor'
    else 'very_poor'
  end as retention_quality
  
from (
  {{ innovatecorp_analytics_toolkit.generate_cohort_analysis(
      table_name=ref('user_events'),
      user_id_col='user_id',
      date_col='event_date',
      cohort_period='month'
  ) }}
) cohort_results
where period_number <= 12  -- Focus on first year
order by cohort_date, period_number