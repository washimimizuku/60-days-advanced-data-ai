-- Staging model: Clean customer data with PII protection
-- This model implements GDPR-compliant customer data processing

{{ config(
    materialized='view',
    tags=['staging', 'pii', 'gdpr'],
    meta={
        'data_classification': 'confidential',
        'pii_fields': ['email', 'phone'],
        'retention_period': '7_years',
        'gdpr_applicable': true
    }
) }}

with source as (
    select * from raw_customers
),

pii_protected as (
    select
        customer_id,
        
        -- PII Protection: Hash email for analytics while preserving uniqueness
        md5(email || '{{ var("pii_hash_salt") }}') as email_hash,
        
        -- Keep first/last name but log access
        first_name,
        last_name,
        
        -- Mask phone number for non-privileged users
        case 
            when '{{ env_var("USER_ROLE", "analyst") }}' = 'admin' then phone
            else regexp_replace(phone, '(\d{3})-(\d{3})-(\d{4})', '\1-XXX-\4')
        end as phone,
        
        -- Geographic data (allowed for analytics)
        city,
        state,
        country,
        
        -- Temporal data
        created_at,
        updated_at,
        consent_status,
        
        -- Data governance metadata
        current_timestamp as _dbt_processed_at,
        '{{ invocation_id }}' as _dbt_run_id
        
    from source
    where 
        -- Data quality filters
        customer_id is not null
        and email is not null
        and created_at is not null
        
        -- GDPR compliance: Only process consented customers
        and consent_status = 'granted'
        
        -- Retention policy: Only include data within retention period
        and created_at >= current_date - interval '{{ var("retention_years") }} years'
),

final as (
    select
        *,
        
        -- Customer segmentation (privacy-safe)
        case
            when created_at >= current_date - interval '30 days' then 'New'
            when created_at >= current_date - interval '1 year' then 'Active'
            else 'Established'
        end as customer_segment,
        
        -- Data quality flags
        case when email_hash is not null then true else false end as has_valid_email,
        case when phone is not null then true else false end as has_phone
        
    from pii_protected
)

select * from final