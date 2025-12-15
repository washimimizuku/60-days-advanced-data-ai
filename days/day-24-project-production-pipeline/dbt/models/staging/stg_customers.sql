{{ config(materialized='view') }}

select
    customer_id,
    lower(trim(email)) as email,
    trim(first_name) as first_name,
    case 
        when customer_segment is null then 'unknown'
        else lower(trim(customer_segment))
    end as customer_segment,
    case 
        when email ~ '^[^@]+@[^@]+\.[^@]+$' then true
        else false
    end as email_is_valid,
    current_timestamp as dbt_updated_at
from {{ source('raw', 'customers') }}
where customer_id is not null