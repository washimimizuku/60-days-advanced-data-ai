FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dbt and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/dbt_project /app/profiles /app/data

# Copy project files
COPY dbt_project/ /app/dbt_project/
COPY profiles/ /app/profiles/

# Set environment variables
ENV DBT_PROFILES_DIR=/app/profiles
ENV DBT_PROJECT_DIR=/app/dbt_project

# Default command
CMD ["tail", "-f", "/dev/null"]