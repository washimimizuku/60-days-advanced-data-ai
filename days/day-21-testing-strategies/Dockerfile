# Day 21: Testing Strategies - Docker Environment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for test results and baselines
RUN mkdir -p /app/test_results /app/test_baselines /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV TESTING_MODE=true

# Expose ports
EXPOSE 8888 8000

# Default command
CMD ["bash"]