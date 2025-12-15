-- Insert sample data for dbt development and testing

-- Insert sample users
INSERT INTO raw.users (email, first_name, last_name, subscription_tier, country, city, created_at, updated_at) VALUES
('john.doe@example.com', 'John', 'Doe', 'premium', 'US', 'New York', '2024-01-15', '2024-01-15'),
('jane.smith@example.com', 'Jane', 'Smith', 'enterprise', 'US', 'San Francisco', '2024-01-10', '2024-01-20'),
('bob.wilson@example.com', 'Bob', 'Wilson', 'basic', 'CA', 'Toronto', '2024-01-20', '2024-01-20'),
('alice.brown@example.com', 'Alice', 'Brown', 'free', 'GB', 'London', '2024-01-25', '2024-01-25'),
('charlie.davis@example.com', 'Charlie', 'Davis', 'premium', 'AU', 'Sydney', '2024-01-12', '2024-01-30');

-- Insert sample subscriptions
INSERT INTO raw.subscriptions (user_id, plan_name, status, amount_cents, started_at) VALUES
(1, 'premium_monthly', 'active', 2999, '2024-01-15'),
(2, 'enterprise', 'active', 9999, '2024-01-10'),
(3, 'basic_monthly', 'active', 999, '2024-01-20'),
(5, 'premium_yearly', 'active', 29999, '2024-01-12');

-- Insert sample events
INSERT INTO raw.events (user_id, event_type, event_timestamp, properties) VALUES
(1, 'login', '2024-01-15 09:00:00', '{"device_type": "desktop", "session_duration": "1800"}'),
(1, 'page_view', '2024-01-15 09:05:00', '{"page_url": "/dashboard", "referrer": "direct"}'),
(1, 'feature_use', '2024-01-15 09:10:00', '{"feature": "analytics", "duration": "300"}'),
(2, 'login', '2024-01-16 10:00:00', '{"device_type": "mobile", "session_duration": "900"}'),
(2, 'purchase', '2024-01-16 10:30:00', '{"amount": "99.99", "product_id": "1"}'),
(3, 'page_view', '2024-01-17 14:00:00', '{"page_url": "/pricing", "referrer": "google"}'),
(4, 'form_submit', '2024-01-18 16:00:00', '{"form_type": "contact", "success": "true"}'),
(5, 'feature_use', '2024-01-19 11:00:00', '{"feature": "reports", "duration": "600"}');

-- Insert sample products
INSERT INTO raw.products (product_name, price_cents, category, is_active) VALUES
('Basic Plan', 999, 'subscription', true),
('Premium Plan', 2999, 'subscription', true),
('Enterprise Plan', 9999, 'subscription', true),
('Add-on Feature A', 499, 'addon', true),
('Add-on Feature B', 799, 'addon', false);