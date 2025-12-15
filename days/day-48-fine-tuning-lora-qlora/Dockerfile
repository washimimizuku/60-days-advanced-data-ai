# Day 48: Fine-tuning Techniques - LoRA & QLoRA - Production Docker Image

FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional optimization libraries
RUN pip install \
    flash-attn==2.3.3 \
    xformers==0.0.22 \
    triton==2.1.0 \
    deepspeed==0.12.3

# Create necessary directories
RUN mkdir -p /app/models \
    /app/data \
    /app/logs \
    /app/adapters \
    /app/checkpoints \
    /app/configs

# Copy application code
COPY . .

# Set up model cache directory
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_DATASETS_CACHE=/app/data

# Create non-root user for security
RUN useradd -m -u 1000 lora-user && \
    chown -R lora-user:lora-user /app
USER lora-user

# Expose ports for monitoring and API
EXPOSE 8000 6006 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python", "solution.py"]

# =============================================================================
# Multi-stage build for production optimization
# =============================================================================

# Production stage
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as production

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=0 /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=0 /usr/local/bin /usr/local/bin

# Copy application
COPY --from=0 /app /app

# Create non-root user
RUN useradd -m -u 1000 lora-user && \
    chown -R lora-user:lora-user /app
USER lora-user

# Set environment variables
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

EXPOSE 8000

CMD ["python", "solution.py"]

# =============================================================================
# Development stage with Jupyter and debugging tools
# =============================================================================

FROM nvidia/cuda:12.1-devel-ubuntu22.04 as development

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install development dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install \
        jupyter \
        jupyterlab \
        ipywidgets \
        matplotlib \
        seaborn \
        plotly \
        tensorboard \
        wandb

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/notebooks /app/experiments

# Expose additional ports for development
EXPOSE 8000 6006 8888 8889

# Start Jupyter Lab by default in development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Build instructions and usage examples
# =============================================================================

# Build commands:
# docker build -t lora-qlora:latest .
# docker build --target production -t lora-qlora:prod .
# docker build --target development -t lora-qlora:dev .

# Run commands:
# docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data lora-qlora:latest
# docker run --gpus all -p 8888:8888 -v $(pwd):/app lora-qlora:dev

# Docker Compose usage:
# docker-compose up -d

# Environment variables for runtime:
# -e CUDA_VISIBLE_DEVICES=0
# -e LORA_RANK=32
# -e BATCH_SIZE=16
# -e ENABLE_WANDB=true
# -e WANDB_API_KEY=your_key_here