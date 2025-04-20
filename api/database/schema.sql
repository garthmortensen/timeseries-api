-- Create pipeline_runs table
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    source_type VARCHAR(50),
    start_date VARCHAR(50),
    end_date VARCHAR(50),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP
);

-- Create pipeline_results table
CREATE TABLE IF NOT EXISTS pipeline_results (
    id SERIAL PRIMARY KEY,
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id),
    symbol VARCHAR(50) NOT NULL,
    result_type VARCHAR(50) NOT NULL,
    is_stationary BOOLEAN,
    adf_statistic FLOAT,
    p_value FLOAT,
    model_summary TEXT,
    forecast TEXT,
    interpretation TEXT
);
