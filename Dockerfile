# RunPod Serverless WhisperX Multi-Chunk Dockerfile
# Version: 1.4
# Using official WhisperX with Silero VAD (compatible with modern torch/pyannote)
# Fixed: Using RunPod's PyTorch 2.8.0 image with CUDA 12.8.1 + cuDNN 9.8 (matches PyTorch compilation)
# Fixed: Use chunk_size_seconds parameter instead of unreliable FLAC metadata
# Fixed: Correct DiarizationPipeline import from whisperx.diarize
# Fixed: Event loop handling for RunPod serverless environment

# Force AMD64 architecture for RunPod compatibility
# Using RunPod's official PyTorch 2.8.0 image with CUDA 12.8.1 + cuDNN 9.8 (optimized for serverless)
FROM --platform=linux/amd64 runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Install system dependencies (Python 3.11 and PyTorch already included in base image)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install base dependencies
RUN pip install --no-cache-dir ffmpeg-python==0.2.0 requests>=2.31.0 aiohttp>=3.9.0 aiofiles>=23.2.1

# Install RunPod SDK
RUN pip install --no-cache-dir runpod>=1.6.0

# Install cog
RUN pip install --no-cache-dir cog>=0.9.0

# Install WhisperX directly from main repo (handles all dependencies itself)
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# Verify installations
RUN python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); import torchaudio; print(f'✓ TorchAudio: {torchaudio.__version__}'); import pyannote.audio; print(f'✓ pyannote.audio: {pyannote.audio.__version__}'); import whisperx; print('✓ WhisperX: OK')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /root/.cache/torch

# Expose port for health checks
EXPOSE 8000

# Health check for RunPod
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Default command for RunPod serverless
CMD ["python3", "-u", "predict.py"]

