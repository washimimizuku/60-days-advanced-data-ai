FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model (optional)
RUN python -m spacy download en_core_web_sm || echo "spaCy model download failed (optional)"

# Create non-root user for security
RUN useradd -m -u 1000 privacy && \
    mkdir -p /app/logs /app/data && \
    chown -R privacy:privacy /app

# Copy application code
COPY --chown=privacy:privacy . .

# Switch to non-root user
USER privacy

# Create necessary directories
RUN mkdir -p logs data sample_data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "solution.py"]