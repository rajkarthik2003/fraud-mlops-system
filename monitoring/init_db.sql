-- filepath: monitoring/init_db.sql
-- Prediction logging database schema
-- Run this against the PostgreSQL container:
-- docker exec -i fraud_postgres psql -U fraud_user -d fraud_db < monitoring/init_db.sql

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_version VARCHAR(20) NOT NULL,
    threshold FLOAT NOT NULL,
    features JSONB NOT NULL,
    fraud_probability FLOAT NOT NULL,
    prediction INTEGER NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    latency_ms FLOAT,
    client_ip VARCHAR(45),
    endpoint VARCHAR(50) NOT NULL,
    drift_detected BOOLEAN DEFAULT FALSE,
    drift_features JSONB
);

-- Create indexes for common queries
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_request_id ON predictions(request_id);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);
CREATE INDEX idx_predictions_is_fraud ON predictions(is_fraud);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    model_version VARCHAR(20) NOT NULL,
    roc_auc FLOAT,
    pr_auc FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    f1_score FLOAT,
    threshold FLOAT,
    total_predictions INTEGER,
    fraud_predictions INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER
);

CREATE INDEX idx_model_metrics_timestamp ON model_metrics(timestamp);
CREATE INDEX idx_model_metrics_version ON model_metrics(model_version);

-- Drift detection log
CREATE TABLE IF NOT EXISTS drift_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    feature_name VARCHAR(20) NOT NULL,
    feature_index INTEGER NOT NULL,
    z_score FLOAT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    reference_mean FLOAT,
    reference_std FLOAT,
    current_value FLOAT
);

CREATE INDEX idx_drift_log_timestamp ON drift_log(timestamp);
CREATE INDEX idx_drift_log_feature ON drift_log(feature_name);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    endpoint VARCHAR(50) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms FLOAT NOT NULL,
    client_ip VARCHAR(45),
    api_key_hash VARCHAR(64)
);

CREATE INDEX idx_api_usage_timestamp ON api_usage(timestamp);
CREATE INDEX idx_api_usage_endpoint ON api_usage(endpoint);