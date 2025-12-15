-- Staging model: Clean and standardize customer data
-- This model provides a clean, consistent view of customer data

{{ config(
    materialized='view',
    tags=['staging', 'customers']
) }}

with source as (
    select * from {{ source('raw_data', 'customers') }}
),

cleaned as (
    select
        id as customer_id,
        lower(trim(email)) as email,
        initcap(trim(first_name)) as first_name,
        initcap(trim(last_name)) as last_name,
        created_at,
        
        -- Derived fields
        concat(initcap(trim(first_name)), ' ', initcap(trim(last_name))) as full_name,
        
        -- Data quality flags
        case 
            when email like '%@%.%' then true 
            else false 
        end as is_valid_email,
        
        -- Audit fields
        current_timestamp as _dbt_loaded_at
        
    from source
    where 
        -- Data quality filters
        id is not null
        and email is not null
        and email != ''
        and first_name is not null
        and first_name != ''
        and last_name is not null
        and last_name != ''
        and created_at is not null
)

select * from cleaned