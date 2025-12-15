-- models/examples/example_clv_analysis.sql
-- Example customer lifetime value analysis

{{ config(materialized='view') }}

select
  customer_id,
  customer_segment,
  acquisition_channel,
  
  -- Historical metrics
  total_orders,
  total_spent,
  avg_order_value,
  
  -- Predictions
  predicted_clv,
  churn_probability,
  clv_category,
  churn_risk_category,
  prediction_confidence,
  
  -- Business insights
  case
    when clv_category in ('very_high_value', 'high_value') and churn_risk_category in ('high_risk', 'medium_risk') then 'retention_priority'
    when clv_category in ('very_high_value', 'high_value') and churn_risk_category in ('low_risk', 'very_low_risk') then 'upsell_opportunity'
    when clv_category in ('medium_value') and churn_risk_category in ('high_risk', 'medium_risk') then 'win_back_campaign'
    when clv_category in ('low_value', 'very_low_value') and churn_risk_category in ('high_risk') then 'let_churn'
    else 'maintain_engagement'
  end as recommended_action,
  
  -- Investment priority scoring
  case
    when predicted_clv >= 5000 and churn_probability <= 0.3 then 5  -- Highest priority
    when predicted_clv >= 2000 and churn_probability <= 0.5 then 4
    when predicted_clv >= 1000 and churn_probability <= 0.7 then 3
    when predicted_clv >= 500 then 2
    else 1  -- Lowest priority
  end as investment_priority_score
  
from (
  {{ innovatecorp_analytics_toolkit.calculate_predictive_clv(
      customer_metrics_table=ref('customer_metrics'),
      prediction_horizon_months=12,
      discount_rate=0.1
  ) }}
) clv_results
order by predicted_clv desc