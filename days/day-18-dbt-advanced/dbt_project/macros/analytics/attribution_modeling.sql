-- macros/analytics/attribution_modeling.sql
-- Advanced multi-touch attribution modeling

{% macro calculate_attribution(
    events_table,
    conversions_table,
    attribution_window_days=30,
    attribution_model='linear',
    channel_column='channel',
    campaign_column='campaign'
) %}

  with attribution_events as (
    select
      e.user_id,
      e.event_id,
      e.event_timestamp,
      e.{{ channel_column }} as channel,
      e.{{ campaign_column }} as campaign,
      
      c.conversion_id,
      c.conversion_timestamp,
      c.conversion_value,
      c.conversion_type,
      
      -- Time-based calculations
      extract(epoch from (c.conversion_timestamp - e.event_timestamp))/3600 as hours_to_conversion,
      extract(epoch from (c.conversion_timestamp - e.event_timestamp))/86400 as days_to_conversion,
      
      -- Position in customer journey
      row_number() over (
        partition by e.user_id, c.conversion_id 
        order by e.event_timestamp asc
      ) as touchpoint_sequence,
      
      count(*) over (
        partition by e.user_id, c.conversion_id
      ) as total_touchpoints
      
    from {{ events_table }} e
    join {{ conversions_table }} c 
      on e.user_id = c.user_id
      and e.event_timestamp <= c.conversion_timestamp
      and e.event_timestamp >= c.conversion_timestamp - interval '{{ attribution_window_days }} days'
    
    where e.{{ channel_column }} is not null
      and e.{{ campaign_column }} is not null
  ),
  
  attribution_weights as (
    select
      *,
      
      -- Calculate attribution weights based on model
      case 
        when '{{ attribution_model }}' = 'first_touch' then
          case when touchpoint_sequence = 1 then 1.0 else 0.0 end
        
        when '{{ attribution_model }}' = 'last_touch' then
          case when touchpoint_sequence = total_touchpoints then 1.0 else 0.0 end
        
        when '{{ attribution_model }}' = 'linear' then
          1.0 / nullif(total_touchpoints, 0)
        
        when '{{ attribution_model }}' = 'time_decay' then
          -- Exponential decay: more recent touchpoints get higher weight
          exp(-0.1 * days_to_conversion) / 
          sum(exp(-0.1 * days_to_conversion)) over (
            partition by user_id, conversion_id
          )
        
        else 1.0 / nullif(total_touchpoints, 0)  -- Default to linear
      end as attribution_weight
      
    from attribution_events
  )
  
  select
    user_id,
    channel,
    campaign,
    conversion_id,
    conversion_timestamp,
    conversion_type,
    
    -- Core attribution metrics
    sum(conversion_value * attribution_weight) as attributed_revenue,
    sum(attribution_weight) as attribution_weight_total,
    
    -- Journey insights
    count(*) as touchpoints_in_journey,
    min(touchpoint_sequence) as first_touchpoint_position,
    max(touchpoint_sequence) as last_touchpoint_position,
    avg(days_to_conversion) as avg_days_to_conversion,
    
    -- Model metadata
    '{{ attribution_model }}' as attribution_model,
    {{ attribution_window_days }} as attribution_window_days,
    current_timestamp as calculated_at
    
  from attribution_weights
  group by 
    user_id, channel, campaign, conversion_id, 
    conversion_timestamp, conversion_type

{% endmacro %}