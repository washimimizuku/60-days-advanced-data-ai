-- GDPR Compliance Reporting Model
-- Generates automated compliance metrics and violation reports

{{ config(
    materialized='table',
    tags=['compliance', 'gdpr', 'audit'],
    schema='compliance'
) }}

with data_inventory as (
    select
        'customers' as table_name,
        count(*) as total_records,
        count(case when consent_status = 'granted' then 1 end) as consented_records,
        count(case when created_at < current_date - interval '{{ var("retention_years") }} years' then 1 end) as retention_violations,
        count(case when email_hash is null then 1 end) as pii_protection_violations
    from {{ ref('stg_customers') }}
),

compliance_summary as (
    select
        current_date as report_date,
        sum(total_records) as total_data_subjects,
        sum(consented_records) as consented_data_subjects,
        sum(retention_violations) as retention_violations,
        sum(pii_protection_violations) as pii_violations,
        
        -- Calculate compliance scores
        round(
            (sum(consented_records)::numeric / nullif(sum(total_records), 0)) * 100, 2
        ) as consent_compliance_percent,
        
        round(
            ((sum(total_records) - sum(retention_violations))::numeric / nullif(sum(total_records), 0)) * 100, 2
        ) as retention_compliance_percent,
        
        round(
            ((sum(total_records) - sum(pii_violations))::numeric / nullif(sum(total_records), 0)) * 100, 2
        ) as pii_protection_percent
        
    from data_inventory
),

final as (
    select
        *,
        
        -- Overall GDPR compliance score
        round(
            (consent_compliance_percent + retention_compliance_percent + pii_protection_percent) / 3, 2
        ) as overall_gdpr_score,
        
        -- Compliance status
        case
            when (consent_compliance_percent + retention_compliance_percent + pii_protection_percent) / 3 >= 95 then 'Compliant'
            when (consent_compliance_percent + retention_compliance_percent + pii_protection_percent) / 3 >= 85 then 'Minor Issues'
            else 'Non-Compliant'
        end as compliance_status,
        
        -- Next review date
        current_date + interval '30 days' as next_review_date,
        
        -- Audit metadata
        current_timestamp as report_generated_at,
        '{{ invocation_id }}' as dbt_run_id
        
    from compliance_summary
)

select * from final