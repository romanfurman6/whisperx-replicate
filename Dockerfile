# RunPod Serverless WhisperX Multi-Chunk Dockerfile
# Version: 1.4
# Using official WhisperX with Silero VAD (compatible with modern torch/pyannote)
# Fixed: Using CUDA 12.6.2 + cuDNN 9 (compatible with PyTorch 2.8.0)
# Fixed: Use chunk_size_seconds parameter instead of unreliable FLAC metadata
# Fixed: Correct DiarizationPipeline import from whisperx.diarize
# Fixed: Event loop handling for RunPod serverless environment
# Strategy: Let WhisperX manage PyTorch 2.8.0+cu128 installation to avoid dependency conflicts

# Force AMD64 architecture for RunPod compatibility
# Using NVIDIA CUDA 12.6.2 base with cuDNN 9 - WhisperX will install PyTorch 2.8.0
FROM --platform=linux/amd64 nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Install system dependencies including Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

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

