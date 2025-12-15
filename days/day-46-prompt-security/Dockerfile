# Day 46: Prompt Security - Docker Configuration

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/backups

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from solution import SecureLLMGateway; gateway = SecureLLMGateway('test'); print('Security system healthy')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "solution.py"]

# Alternative commands for different use cases:
# Development: docker run -it security-app python exercise.py
# Testing: docker run security-app python -m pytest test_security.py
# API Server: docker run -p 8000:8000 security-app python -m uvicorn api:app --host 0.0.0.0 --port 8000