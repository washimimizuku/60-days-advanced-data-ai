-- Macro: Calculate profit margin percentage with proper null handling
-- Calculates profit margin with safe division and null checks

{% macro calculate_profit_margin(revenue_column, cost_column) %}
    case 
        when {{ revenue_column }} > 0 and {{ revenue_column }} is not null and {{ cost_column }} is not null then 
            round((({{ revenue_column }} - {{ cost_column }}) / {{ revenue_column }}) * 100, 2)
        else 0 
    end
{% endmacro %}