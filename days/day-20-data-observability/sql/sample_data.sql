-- Generate sample data for observability testing

-- Insert sample customer transactions
INSERT INTO data_sources.customer_transactions (customer_id, email, amount, quantity, price, status, category, region, created_at) VALUES
('CUST_001', 'customer1@example.com', 150.50, 2, 75.25, 'active', 'electronics', 'north', CURRENT_TIMESTAMP - INTERVAL '1 hour'),
('CUST_002', 'customer2@example.com', 89.99, 1, 89.99, 'active', 'clothing', 'south', CURRENT_TIMESTAMP - INTERVAL '2 hours'),
('CUST_003', 'customer3@example.com', 245.00, 3, 81.67, 'inactive', 'books', 'east', CURRENT_TIMESTAMP - INTERVAL '3 hours'),
('CUST_004', 'customer4@example.com', 67.50, 1, 67.50, 'trial', 'home', 'west', CURRENT_TIMESTAMP - INTERVAL '4 hours'),
('CUST_005', 'customer5@example.com', 199.99, 2, 99.99, 'active', 'sports', 'central', CURRENT_TIMESTAMP - INTERVAL '5 hours');

-- Generate more sample data using a loop simulation
DO $$
DECLARE
    i INTEGER;
    statuses TEXT[] := ARRAY['active', 'inactive', 'suspended', 'cancelled', 'trial'];
    categories TEXT[] := ARRAY['electronics', 'clothing', 'books', 'home', 'sports'];
    regions TEXT[] := ARRAY['north', 'south', 'east', 'west', 'central'];
BEGIN
    FOR i IN 6..1000 LOOP
        INSERT INTO data_sources.customer_transactions (
            customer_id, email, amount, quantity, price, status, category, region, created_at
        ) VALUES (
            'CUST_' || LPAD(i::TEXT, 6, '0'),
            'customer' || i || '@example.com',
            ROUND((RANDOM() * 1000 + 10)::NUMERIC, 2),
            FLOOR(RANDOM() * 10 + 1)::INTEGER,
            ROUND((RANDOM() * 100 + 5)::NUMERIC, 2),
            statuses[FLOOR(RANDOM() * array_length(statuses, 1) + 1)],
            categories[FLOOR(RANDOM() * array_length(categories, 1) + 1)],
            regions[FLOOR(RANDOM() * array_length(regions, 1) + 1)],
            CURRENT_TIMESTAMP - (RANDOM() * INTERVAL '30 days')
        );
    END LOOP;
END $$;

-- Insert sample monitoring metrics
INSERT INTO monitoring.data_quality_metrics (table_name, metric_type, metric_value, threshold_value, is_anomaly, severity) VALUES
('customer_transactions', 'freshness_score', 0.95, 0.90, FALSE, 'low'),
('customer_transactions', 'volume_score', 0.88, 0.85, FALSE, 'low'),
('customer_transactions', 'schema_score', 1.00, 0.95, FALSE, 'low'),
('customer_transactions', 'distribution_score', 0.92, 0.90, FALSE, 'low'),
('customer_transactions', 'lineage_score', 0.85, 0.80, FALSE, 'low');

-- Insert sample alert history
INSERT INTO alerts.alert_history (metric_name, table_name, current_value, threshold_value, severity, status, message) VALUES
('data_freshness', 'customer_transactions', 0.75, 0.90, 'medium', 'resolved', 'Data freshness below threshold'),
('data_volume', 'customer_transactions', 0.60, 0.85, 'high', 'open', 'Significant volume drop detected'),
('overall_health_score', 'customer_transactions', 0.82, 0.90, 'medium', 'open', 'Overall health score degraded');