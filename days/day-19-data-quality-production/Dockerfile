FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/great_expectations /app/data /app/monitoring

# Copy project files
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["tail", "-f", "/dev/null"]