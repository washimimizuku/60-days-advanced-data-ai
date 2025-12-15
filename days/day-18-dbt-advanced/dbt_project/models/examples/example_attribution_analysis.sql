-- models/examples/example_attribution_analysis.sql
-- Example multi-touch attribution analysis

{{ config(materialized='view') }}

select
  channel,
  campaign,
  sum(attributed_revenue) as total_attributed_revenue,
  count(distinct conversion_id) as conversions,
  avg(avg_days_to_conversion) as avg_customer_journey_days,
  sum(touchpoints_in_journey) as total_touchpoints
from (
  {{ innovatecorp_analytics_toolkit.calculate_attribution(
      events_table=ref('user_events'),
      conversions_table=ref('conversions'),
      attribution_model='linear',
      attribution_window_days=30
  ) }}
) attribution_results
group by channel, campaign
order by total_attributed_revenue desc