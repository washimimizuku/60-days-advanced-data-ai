-- Sample data for InnovateCorp Analytics Toolkit testing

-- Create schema
CREATE SCHEMA IF NOT EXISTS analytics_dev;
SET search_path TO analytics_dev;

-- User events table
CREATE TABLE IF NOT EXISTS user_events (
    event_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    event_timestamp TIMESTAMP NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    campaign VARCHAR(100),
    event_date DATE NOT NULL
);

-- Sample user events data
INSERT INTO user_events (user_id, event_timestamp, event_type, channel, campaign, event_date) VALUES
('user_001', '2024-01-01 10:00:00', 'click', 'paid_search', 'google_brand_campaign', '2024-01-01'),
('user_001', '2024-01-02 14:30:00', 'view', 'email', 'newsletter_jan', '2024-01-02'),
('user_001', '2024-01-03 16:45:00', 'click', 'social', 'facebook_retargeting', '2024-01-03'),
('user_002', '2024-01-01 09:15:00', 'click', 'paid_search', 'google_product_campaign', '2024-01-01'),
('user_002', '2024-01-15 11:20:00', 'click', 'email', 'promotional_email', '2024-01-15'),
('user_003', '2024-02-01 13:00:00', 'view', 'social', 'instagram_ad', '2024-02-01'),
('user_003', '2024-02-05 15:30:00', 'click', 'display', 'banner_ad', '2024-02-05'),
('user_003', '2024-03-01 12:00:00', 'click', 'email', 'newsletter_mar', '2024-03-01');

-- Conversions table
CREATE TABLE IF NOT EXISTS conversions (
    conversion_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    conversion_timestamp TIMESTAMP NOT NULL,
    conversion_type VARCHAR(50) NOT NULL,
    conversion_value DECIMAL(10,2) NOT NULL,
    conversion_date DATE NOT NULL
);

-- Sample conversions data
INSERT INTO conversions (user_id, conversion_timestamp, conversion_type, conversion_value, conversion_date) VALUES
('user_001', '2024-01-04 18:00:00', 'purchase', 1250.00, '2024-01-04'),
('user_002', '2024-01-16 20:15:00', 'purchase', 850.00, '2024-01-16'),
('user_003', '2024-03-02 14:45:00', 'purchase', 2100.00, '2024-03-02');

-- Customer metrics table
CREATE TABLE IF NOT EXISTS customer_metrics (
    customer_id VARCHAR(50) PRIMARY KEY,
    total_orders INTEGER NOT NULL,
    total_spent DECIMAL(10,2) NOT NULL,
    avg_order_value DECIMAL(10,2) NOT NULL,
    days_since_first_order INTEGER NOT NULL,
    days_since_last_order INTEGER NOT NULL,
    customer_segment VARCHAR(50),
    acquisition_channel VARCHAR(50),
    geographic_region VARCHAR(50)
);

-- Sample customer metrics data
INSERT INTO customer_metrics VALUES
('customer_001', 8, 3200.00, 400.00, 365, 15, 'high_value', 'paid_search', 'north_america'),
('customer_002', 12, 4800.00, 400.00, 450, 7, 'high_value', 'email', 'europe'),
('customer_003', 3, 750.00, 250.00, 180, 45, 'medium_value', 'social', 'asia_pacific'),
('customer_004', 15, 7500.00, 500.00, 600, 3, 'vip', 'paid_search', 'north_america'),
('customer_005', 2, 300.00, 150.00, 90, 60, 'low_value', 'display', 'europe'),
('customer_006', 6, 1800.00, 300.00, 270, 30, 'medium_value', 'email', 'asia_pacific'),
('customer_007', 20, 10000.00, 500.00, 720, 5, 'vip', 'paid_search', 'north_america'),
('customer_008', 1, 150.00, 150.00, 30, 90, 'low_value', 'social', 'europe');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON user_events(user_id);
CREATE INDEX IF NOT EXISTS idx_user_events_timestamp ON user_events(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_user_events_channel ON user_events(channel);
CREATE INDEX IF NOT EXISTS idx_conversions_user_id ON conversions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversions_timestamp ON conversions(conversion_timestamp);