-- Sample data setup for Day 17: dbt Deep Dive
-- Creates tables and inserts sample data for DataCorp analytics

-- Create raw schema
CREATE SCHEMA IF NOT EXISTS raw;

-- Users table
CREATE TABLE raw.users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    subscription_tier VARCHAR(50) DEFAULT 'free',
    country VARCHAR(50),
    city VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subscriptions table
CREATE TABLE raw.subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES raw.users(user_id),
    plan_name VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    amount_cents INTEGER NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL
);

-- Events table
CREATE TABLE raw.events (
    event_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES raw.users(user_id),
    event_type VARCHAR(50) NOT NULL,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    properties JSONB,
    _loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE raw.products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    price_cents INTEGER NOT NULL,
    category VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);