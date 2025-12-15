# Day 45: Prompt Engineering with DSPy - Docker Configuration

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
RUN mkdir -p /app/cache /app/logs /app/traces

# Set permissions
RUN chmod +x *.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import dspy; print('DSPy is working')" || exit 1

# Default command
CMD ["python", "solution.py"]

# Alternative commands for different use cases:
# Development: docker run -it dspy-app python exercise.py
# Testing: docker run dspy-app python -m pytest test_dspy.py
# Production API: docker run -p 8000:8000 dspy-app uvicorn api:app --host 0.0.0.0 --port 8000