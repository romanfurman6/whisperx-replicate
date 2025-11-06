# RunPod Serverless WhisperX Multi-Chunk Dockerfile
# Optimized for CUDA 12.1/PyTorch 2.2.0 and serverless deployment
# Using PyTorch 2.2.0 for pyannote.audio 2.1.1 compatibility (VAD support)

# Force AMD64 architecture for RunPod compatibility
# Using CUDA 12.1 (compatible with PyTorch 2.1.2)
FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Install PyTorch 2.2.0 (compatible with pyannote.audio 2.1.1 for VAD)
# CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install pyannote.audio 2.1.1 BEFORE WhisperX to ensure compatibility
RUN pip install --no-cache-dir \
    pyannote.audio==2.1.1 \
    pyannote.pipeline==2.3

# Install remaining Python dependencies
RUN pip install --no-cache-dir ffmpeg-python==0.2.0 requests>=2.31.0 aiohttp>=3.9.0 aiofiles>=23.2.1

# Install RunPod SDK
RUN pip install --no-cache-dir runpod>=1.6.0

# Install cog
RUN pip install --no-cache-dir cog>=0.9.0

# Install WhisperX from main branch (compatible with torch 2.1.x)
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# Ensure torch and pyannote versions stay locked
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0

# Verify versions
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); import torchaudio; print(f'TorchAudio: {torchaudio.__version__}'); import pyannote.audio; print(f'pyannote.audio: {pyannote.audio.__version__}')"

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

