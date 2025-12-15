{{ config(materialized='table') }}

with customer_base as (
    select * from {{ ref('stg_customers') }}
    where email_is_valid = true
),

transaction_metrics as (
    select
        customer_id,
        count(*) as total_transactions,
        sum(amount) as total_spent,
        avg(amount) as avg_transaction_amount,
        max(transaction_date) as last_transaction_date
    from {{ ref('stg_transactions') }}
    group by customer_id
)

select
    cb.customer_id,
    cb.email,
    cb.first_name,
    cb.customer_segment,
    coalesce(tm.total_transactions, 0) as total_transactions,
    coalesce(tm.total_spent, 0) as total_spent,
    case
        when tm.total_spent > 10000 then 'high_value'
        when tm.total_spent > 1000 then 'medium_value'
        else 'low_value'
    end as value_tier,
    current_timestamp as updated_at
from customer_base cb
left join transaction_metrics tm on cb.customer_id = tm.customer_id