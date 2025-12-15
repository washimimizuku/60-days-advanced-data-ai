# Day 50: Quantization - Model Compression & Optimization - Docker Container
# Multi-stage build for optimized quantization environment

# Stage 1: Base environment with PyTorch and quantization libraries
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development environment with additional tools
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipywidgets \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Create directories for outputs
RUN mkdir -p models/quantized models/gguf logs profiles

# Set up Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Expose ports
EXPOSE 8888 8000

# Default command for development
CMD ["jupyter", "notebook", "--allow-root"]

# Stage 3: Production environment (optimized)
FROM base as production

# Copy only necessary files
COPY exercise.py solution.py ./
COPY .env.example .env

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash quantization && \
    chown -R quantization:quantization /app

USER quantization

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('PyTorch available:', torch.__version__)" || exit 1

# Default command for production
CMD ["python", "solution.py"]

# Stage 4: CPU-optimized environment
FROM base as cpu-optimized

# Install Intel optimizations
RUN pip install --no-cache-dir \
    intel-extension-for-pytorch \
    mkl \
    mkl-include

# Set Intel optimizations
ENV KMP_AFFINITY=granularity=fine,verbose,compact,1,0
ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=1
ENV OMP_NUM_THREADS=4

# Copy source code
COPY . .

# Default command
CMD ["python", "solution.py"]

# Stage 5: GPU-optimized environment
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-optimized

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-specific packages
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy source code
COPY . .

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default command
CMD ["python", "solution.py"]

# Build instructions:
# 
# Development environment:
# docker build --target development -t quantization:dev .
# docker run -p 8888:8888 -v $(pwd):/app quantization:dev
#
# Production environment:
# docker build --target production -t quantization:prod .
# docker run quantization:prod
#
# CPU-optimized environment:
# docker build --target cpu-optimized -t quantization:cpu .
# docker run quantization:cpu
#
# GPU-optimized environment:
# docker build --target gpu-optimized -t quantization:gpu .
# docker run --gpus all quantization:gpu