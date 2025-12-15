-- macros/analytics/clv_modeling.sql
-- Customer lifetime value modeling with predictive capabilities

{% macro calculate_predictive_clv(
    customer_metrics_table,
    prediction_horizon_months=12,
    discount_rate=0.1
) %}

  with customer_features as (
    select
      customer_id,
      
      -- Historical transaction metrics
      total_orders,
      total_spent,
      avg_order_value,
      days_since_first_order,
      days_since_last_order,
      
      -- Customer profile features
      coalesce(customer_segment, 'unknown') as customer_segment,
      coalesce(acquisition_channel, 'unknown') as acquisition_channel,
      coalesce(geographic_region, 'unknown') as geographic_region,
      
      -- Calculate behavioral metrics
      case 
        when days_since_first_order > 0 then
          total_orders * 30.0 / nullif(days_since_first_order, 0)
        else 0
      end as monthly_purchase_frequency,
      
      -- Recency scoring (0-1 scale)
      case
        when days_since_last_order <= 30 then 1.0
        when days_since_last_order <= 60 then 0.8
        when days_since_last_order <= 90 then 0.6
        when days_since_last_order <= 180 then 0.4
        when days_since_last_order <= 365 then 0.2
        else 0.1
      end as recency_score,
      
      -- Frequency scoring (0-1 scale)
      case
        when total_orders >= 20 then 1.0
        when total_orders >= 10 then 0.8
        when total_orders >= 5 then 0.6
        when total_orders >= 2 then 0.4
        when total_orders >= 1 then 0.2
        else 0.0
      end as frequency_score,
      
      -- Monetary scoring (0-1 scale)
      case
        when total_spent >= 5000 then 1.0
        when total_spent >= 2000 then 0.8
        when total_spent >= 1000 then 0.6
        when total_spent >= 500 then 0.4
        when total_spent >= 100 then 0.2
        else 0.1
      end as monetary_score
      
    from {{ customer_metrics_table }}
  ),
  
  churn_probability_calculation as (
    select
      *,
      
      -- Simple churn probability model
      case
        when days_since_last_order > 365 then 0.80
        when days_since_last_order > 180 then 0.50
        when days_since_last_order > 90 then 0.30
        when days_since_last_order > 60 then 0.20
        when days_since_last_order > 30 then 0.10
        else 0.05
      end as churn_probability
      
    from customer_features
  ),
  
  clv_calculations as (
    select
      *,
      
      -- Expected monthly value
      monthly_purchase_frequency * avg_order_value * recency_score as predicted_monthly_value,
      
      -- Expected customer lifetime in months (simplified)
      case
        when churn_probability > 0 and churn_probability < 1 then
          least({{ prediction_horizon_months }}, 1.0 / nullif(churn_probability, 0))
        else {{ prediction_horizon_months }}
      end as predicted_lifetime_months,
      
      -- Confidence scoring
      case
        when total_orders >= 10 and days_since_first_order >= 365 then 'high'
        when total_orders >= 5 and days_since_first_order >= 180 then 'medium'
        when total_orders >= 2 and days_since_first_order >= 90 then 'low'
        else 'very_low'
      end as prediction_confidence
      
    from churn_probability_calculation
  )
  
  select
    customer_id,
    customer_segment,
    acquisition_channel,
    geographic_region,
    
    -- Historical metrics
    total_orders,
    total_spent,
    avg_order_value,
    days_since_first_order,
    days_since_last_order,
    
    -- Behavioral scores
    recency_score,
    frequency_score,
    monetary_score,
    
    -- Predictions
    predicted_monthly_value,
    predicted_lifetime_months,
    churn_probability,
    
    -- Calculate discounted CLV (simplified)
    case
      when predicted_lifetime_months > 0 and predicted_monthly_value > 0 then
        predicted_monthly_value * predicted_lifetime_months * 
        (1 - power(1 + {{ discount_rate }}/12, -predicted_lifetime_months)) / 
        nullif({{ discount_rate }}/12, 0)
      else 0
    end as predicted_clv,
    
    prediction_confidence,
    
    -- Risk categorization
    case
      when churn_probability >= 0.7 then 'high_risk'
      when churn_probability >= 0.4 then 'medium_risk'
      when churn_probability >= 0.2 then 'low_risk'
      else 'very_low_risk'
    end as churn_risk_category,
    
    -- Value categorization
    case
      when predicted_monthly_value * predicted_lifetime_months >= 5000 then 'very_high_value'
      when predicted_monthly_value * predicted_lifetime_months >= 2000 then 'high_value'
      when predicted_monthly_value * predicted_lifetime_months >= 1000 then 'medium_value'
      when predicted_monthly_value * predicted_lifetime_months >= 500 then 'low_value'
      else 'very_low_value'
    end as clv_category,
    
    -- Model metadata
    {{ prediction_horizon_months }} as prediction_horizon_months,
    {{ discount_rate }} as discount_rate_used,
    current_timestamp as calculated_at
    
  from clv_calculations

{% endmacro %}