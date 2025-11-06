# RunPod Serverless WhisperX Multi-Chunk Dockerfile
# Optimized for CUDA 12.1 and serverless deployment

# Force AMD64 architecture for RunPod compatibility
FROM --platform=linux/amd64 nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first
# Use compatible versions: torch 2.1.0 is stable with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining Python dependencies
# Use --no-deps for torch packages to prevent version conflicts
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models/vad /root/.cache/torch

# Download VAD model
RUN python3 get_vad_model_url.py || echo "VAD model download handled at runtime"

# Pre-download WhisperX models (optional, can be done at runtime)
# This reduces cold start time but increases image size
RUN python3 -c "import whisperx; print('WhisperX imported successfully')" || echo "WhisperX will load models at runtime"

# Set permissions
RUN chmod +x build.sh || true

# Expose port for health checks
EXPOSE 8000

# Health check for RunPod
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Default command for RunPod serverless
CMD ["python3", "-u", "predict.py"]

