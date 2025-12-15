-- Day 32: ML Platform Database Initialization

-- Create database schema for ML platform
CREATE SCHEMA IF NOT EXISTS ml_platform;

-- User features table
CREATE TABLE IF NOT EXISTS ml_platform.user_features (
    user_id INTEGER PRIMARY KEY,
    age REAL,
    income REAL,
    credit_score REAL,
    avg_transaction_amount REAL,
    transaction_frequency REAL,
    risk_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transaction features table
CREATE TABLE IF NOT EXISTS ml_platform.transaction_features (
    transaction_id VARCHAR(50) PRIMARY KEY,
    user_id INTEGER,
    amount REAL,
    timestamp TIMESTAMP,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    merchant_category VARCHAR(50),
    location_type VARCHAR(20),
    amount_zscore REAL,
    is_fraud BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Credit applications table
CREATE TABLE IF NOT EXISTS ml_platform.credit_applications (
    application_id VARCHAR(50) PRIMARY KEY,
    user_id INTEGER,
    age REAL,
    income REAL,
    credit_score REAL,
    credit_history_length REAL,
    loan_amount REAL,
    loan_purpose VARCHAR(50),
    employment_length REAL,
    debt_to_income_ratio REAL,
    default_risk REAL,
    approved BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market data table
CREATE TABLE IF NOT EXISTS ml_platform.market_data (
    date DATE PRIMARY KEY,
    stock_index REAL,
    interest_rate REAL,
    volatility_index REAL,
    exchange_rate REAL,
    stock_return_1d REAL,
    stock_return_7d REAL,
    volatility_ma_7d REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model predictions log
CREATE TABLE IF NOT EXISTS ml_platform.predictions_log (
    prediction_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(50),
    input_features JSONB,
    prediction REAL,
    probability REAL,
    explanation JSONB,
    processing_time_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS ml_platform.model_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature store metadata
CREATE TABLE IF NOT EXISTS ml_platform.feature_metadata (
    feature_name VARCHAR(100) PRIMARY KEY,
    feature_group VARCHAR(50),
    data_type VARCHAR(20),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_features_user_id ON ml_platform.user_features(user_id);
CREATE INDEX IF NOT EXISTS idx_transaction_features_user_id ON ml_platform.transaction_features(user_id);
CREATE INDEX IF NOT EXISTS idx_transaction_features_timestamp ON ml_platform.transaction_features(timestamp);
CREATE INDEX IF NOT EXISTS idx_credit_applications_user_id ON ml_platform.credit_applications(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_log_model_name ON ml_platform.predictions_log(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_log_created_at ON ml_platform.predictions_log(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON ml_platform.model_metrics(model_name);

-- Insert sample feature metadata
INSERT INTO ml_platform.feature_metadata (feature_name, feature_group, data_type, description) VALUES
('user_age', 'user_features', 'REAL', 'Age of the user in years'),
('user_income', 'user_features', 'REAL', 'Annual income of the user'),
('credit_score', 'user_features', 'REAL', 'Credit score (300-850)'),
('avg_transaction_amount', 'user_features', 'REAL', 'Average transaction amount'),
('transaction_frequency', 'user_features', 'REAL', 'Number of transactions per month'),
('transaction_amount', 'transaction_features', 'REAL', 'Individual transaction amount'),
('hour_of_day', 'transaction_features', 'INTEGER', 'Hour when transaction occurred (0-23)'),
('day_of_week', 'transaction_features', 'INTEGER', 'Day of week (0=Monday, 6=Sunday)'),
('merchant_category', 'transaction_features', 'VARCHAR', 'Category of merchant'),
('location_type', 'transaction_features', 'VARCHAR', 'Domestic or international transaction'),
('stock_index', 'market_features', 'REAL', 'Stock market index value'),
('interest_rate', 'market_features', 'REAL', 'Current interest rate'),
('volatility_index', 'market_features', 'REAL', 'Market volatility index')
ON CONFLICT (feature_name) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA ml_platform TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_platform TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_platform TO ml_user;

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION ml_platform.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_user_features_updated_at 
    BEFORE UPDATE ON ml_platform.user_features 
    FOR EACH ROW EXECUTE FUNCTION ml_platform.update_updated_at_column();

CREATE TRIGGER update_feature_metadata_updated_at 
    BEFORE UPDATE ON ml_platform.feature_metadata 
    FOR EACH ROW EXECUTE FUNCTION ml_platform.update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW ml_platform.user_summary AS
SELECT 
    u.user_id,
    u.age,
    u.income,
    u.credit_score,
    u.avg_transaction_amount,
    u.transaction_frequency,
    COUNT(t.transaction_id) as total_transactions,
    COUNT(CASE WHEN t.is_fraud THEN 1 END) as fraud_transactions,
    MAX(t.timestamp) as last_transaction_date
FROM ml_platform.user_features u
LEFT JOIN ml_platform.transaction_features t ON u.user_id = t.user_id
GROUP BY u.user_id, u.age, u.income, u.credit_score, u.avg_transaction_amount, u.transaction_frequency;

CREATE OR REPLACE VIEW ml_platform.daily_metrics AS
SELECT 
    DATE(created_at) as date,
    model_name,
    COUNT(*) as prediction_count,
    AVG(processing_time_ms) as avg_processing_time,
    AVG(probability) as avg_probability
FROM ml_platform.predictions_log
GROUP BY DATE(created_at), model_name
ORDER BY date DESC, model_name;

-- Insert sample data for testing
INSERT INTO ml_platform.user_features (user_id, age, income, credit_score, avg_transaction_amount, transaction_frequency, risk_score) VALUES
(1, 35, 75000, 720, 150.50, 25, 0.15),
(2, 28, 45000, 650, 89.25, 18, 0.35),
(3, 42, 95000, 780, 220.75, 32, 0.08),
(4, 25, 35000, 580, 65.00, 12, 0.55),
(5, 55, 120000, 800, 350.25, 28, 0.05)
ON CONFLICT (user_id) DO NOTHING;

-- Log initialization
INSERT INTO ml_platform.model_metrics (model_name, metric_name, metric_value) VALUES
('credit_risk', 'initialization', 1.0),
('fraud_detection', 'initialization', 1.0),
('market_forecasting', 'initialization', 1.0),
('recommendations', 'initialization', 1.0);

COMMIT;