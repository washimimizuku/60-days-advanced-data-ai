-- Day 30: Ensemble Methods - Database Initialization

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS ensemble_db;

-- Use the database
\c ensemble_db;

-- Create models table
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    hyperparameters JSONB,
    performance_metrics JSONB,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_path VARCHAR(500),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    description TEXT,
    dataset_info JSONB,
    configuration JSONB,
    results JSONB,
    best_model_id INTEGER REFERENCES models(id),
    status VARCHAR(50) DEFAULT 'running',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    input_features JSONB,
    prediction FLOAT,
    probability FLOAT,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latency_ms FLOAT,
    request_id VARCHAR(255)
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    dataset_type VARCHAR(50) NOT NULL, -- 'train', 'validation', 'test'
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create ensemble_compositions table
CREATE TABLE IF NOT EXISTS ensemble_compositions (
    id SERIAL PRIMARY KEY,
    ensemble_model_id INTEGER REFERENCES models(id),
    base_model_id INTEGER REFERENCES models(id),
    weight FLOAT DEFAULT 1.0,
    role VARCHAR(50), -- 'base', 'meta'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create monitoring_metrics table
CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags JSONB
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    message TEXT NOT NULL,
    threshold_value FLOAT,
    actual_value FLOAT,
    is_resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
CREATE INDEX IF NOT EXISTS idx_predictions_model_id ON predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time);
CREATE INDEX IF NOT EXISTS idx_performance_model_id ON model_performance(model_id);
CREATE INDEX IF NOT EXISTS idx_monitoring_model_id ON monitoring_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_monitoring_timestamp ON monitoring_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_model_id ON alerts(model_id);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(is_resolved);

-- Insert sample data
INSERT INTO models (name, model_type, algorithm, hyperparameters, performance_metrics) VALUES
('random_forest_baseline', 'ensemble', 'RandomForest', 
 '{"n_estimators": 100, "max_depth": 10, "random_state": 42}',
 '{"accuracy": 0.85, "auc": 0.88, "f1": 0.82}'),
('xgboost_baseline', 'ensemble', 'XGBoost',
 '{"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6}',
 '{"accuracy": 0.87, "auc": 0.90, "f1": 0.84}'),
('stacking_ensemble', 'ensemble', 'Stacking',
 '{"base_models": ["RandomForest", "XGBoost", "GradientBoosting"], "meta_model": "LogisticRegression"}',
 '{"accuracy": 0.89, "auc": 0.92, "f1": 0.86}');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ensemble_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ensemble_user;