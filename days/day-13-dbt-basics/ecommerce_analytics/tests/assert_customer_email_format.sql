-- Test: Validate customer email format with proper SQL escaping
-- This test ensures all customer emails have valid format

select
    customer_id,
    email
from {{ ref('stg_customers') }}
where not regexp_like(email, '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')