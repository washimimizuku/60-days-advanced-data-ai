-- Day 31: Model Explainability - Database Initialization

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS explainability_db;

-- Use the database
\c explainability_db;

-- Create models table
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    model_type VARCHAR(100) NOT NULL,
    algorithm VARCHAR(100) NOT NULL,
    performance_metrics JSONB,
    explainability_config JSONB,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_path VARCHAR(500),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create explanations table
CREATE TABLE IF NOT EXISTS explanations (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    explanation_type VARCHAR(50) NOT NULL, -- 'shap', 'lime', 'permutation'
    instance_data JSONB,
    explanation_result JSONB,
    computation_time_ms FLOAT,
    request_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create explanation_requests table
CREATE TABLE IF NOT EXISTS explanation_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    model_id INTEGER REFERENCES models(id),
    explanation_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    input_data JSONB,
    result JSONB,
    error_message TEXT,
    computation_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Create explanation_quality_metrics table
CREATE TABLE IF NOT EXISTS explanation_quality_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id),
    explanation_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL, -- 'faithfulness', 'stability', 'comprehensiveness'
    metric_value FLOAT NOT NULL,
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluation_config JSONB
);

-- Create clinical_explanations table
CREATE TABLE IF NOT EXISTS clinical_explanations (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(255),
    model_id INTEGER REFERENCES models(id),
    risk_probability FLOAT,
    risk_level VARCHAR(50),
    feature_contributions JSONB,
    recommendations JSONB,
    clinician_feedback JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create explanation_cache table
CREATE TABLE IF NOT EXISTS explanation_cache (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    model_id INTEGER REFERENCES models(id),
    explanation_type VARCHAR(50) NOT NULL,
    input_hash VARCHAR(255) NOT NULL,
    cached_result JSONB,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    user_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_explanations_model_id ON explanations(model_id);
CREATE INDEX IF NOT EXISTS idx_explanations_type ON explanations(explanation_type);
CREATE INDEX IF NOT EXISTS idx_explanations_created_at ON explanations(created_at);
CREATE INDEX IF NOT EXISTS idx_explanation_requests_request_id ON explanation_requests(request_id);
CREATE INDEX IF NOT EXISTS idx_explanation_requests_status ON explanation_requests(status);
CREATE INDEX IF NOT EXISTS idx_explanation_requests_created_at ON explanation_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_model_id ON explanation_quality_metrics(model_id);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_type ON explanation_quality_metrics(explanation_type);
CREATE INDEX IF NOT EXISTS idx_clinical_explanations_patient_id ON clinical_explanations(patient_id);
CREATE INDEX IF NOT EXISTS idx_clinical_explanations_created_at ON clinical_explanations(created_at);
CREATE INDEX IF NOT EXISTS idx_explanation_cache_key ON explanation_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_explanation_cache_expires ON explanation_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Insert sample models
INSERT INTO models (name, model_type, algorithm, performance_metrics, explainability_config) VALUES
('patient_readmission_rf', 'classification', 'RandomForest', 
 '{"accuracy": 0.87, "auc": 0.91, "precision": 0.84, "recall": 0.79}',
 '{"shap_enabled": true, "lime_enabled": true, "max_features": 15}'),
('patient_readmission_gb', 'classification', 'GradientBoosting',
 '{"accuracy": 0.89, "auc": 0.93, "precision": 0.86, "recall": 0.81}',
 '{"shap_enabled": true, "lime_enabled": true, "max_features": 15}'),
('patient_readmission_lr', 'classification', 'LogisticRegression',
 '{"accuracy": 0.83, "auc": 0.88, "precision": 0.80, "recall": 0.75}',
 '{"shap_enabled": true, "lime_enabled": true, "max_features": 15}');

-- Insert sample explanation quality metrics
INSERT INTO explanation_quality_metrics (model_id, explanation_type, metric_name, metric_value) VALUES
(1, 'shap', 'faithfulness', 0.85),
(1, 'shap', 'stability', 0.78),
(1, 'lime', 'faithfulness', 0.82),
(1, 'lime', 'stability', 0.74),
(2, 'shap', 'faithfulness', 0.88),
(2, 'shap', 'stability', 0.81),
(3, 'shap', 'faithfulness', 0.91),
(3, 'shap', 'stability', 0.89);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO explain_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO explain_user;