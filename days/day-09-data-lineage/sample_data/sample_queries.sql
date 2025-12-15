-- Day 9: Data Lineage Tracking - Sample SQL Queries
-- These queries demonstrate various lineage extraction scenarios

-- 1. Simple INSERT with JOIN
INSERT INTO analytics.user_metrics
SELECT 
    u.user_id,
    u.email,
    u.first_name || ' ' || u.last_name as full_name,
    COUNT(o.order_id) as order_count,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.created_at) as last_order_date
FROM ecommerce.users u
LEFT JOIN ecommerce.orders o ON u.user_id = o.user_id
WHERE u.status = 'active'
  AND u.created_at >= '2023-01-01'
GROUP BY u.user_id, u.email, u.first_name, u.last_name;

-- 2. Complex CTE with Window Functions
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', created_at) as month,
        user_id,
        SUM(total_amount) as monthly_revenue,
        COUNT(*) as monthly_orders,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY DATE_TRUNC('month', created_at)) as month_rank
    FROM orders
    WHERE status = 'completed'
      AND created_at >= '2023-01-01'
    GROUP BY DATE_TRUNC('month', created_at), user_id
),
user_growth AS (
    SELECT 
        user_id,
        month,
        monthly_revenue,
        LAG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month) as prev_month_revenue,
        monthly_revenue - LAG(monthly_revenue) OVER (PARTITION BY user_id ORDER BY month) as revenue_growth
    FROM monthly_sales
)
INSERT INTO analytics.user_growth_metrics
SELECT 
    ug.user_id,
    ug.month,
    ug.monthly_revenue,
    ug.revenue_growth,
    CASE 
        WHEN ug.revenue_growth > 0 THEN 'growing'
        WHEN ug.revenue_growth < 0 THEN 'declining'
        ELSE 'stable'
    END as growth_status,
    u.email,
    u.segment
FROM user_growth ug
JOIN ecommerce.users u ON ug.user_id = u.user_id
WHERE ug.month_rank > 1;