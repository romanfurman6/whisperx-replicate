# RunPod Serverless WhisperX Multi-Chunk Dockerfile
# Version: 2.0
# Removed Cog dependency - using pure RunPod handler pattern
# Pre-downloading WhisperX large-v3 model to reduce cold start time
# Using RunPod official base image with CUDA 12.6.2
# Fixed: Use chunk_size_seconds parameter instead of unreliable FLAC metadata
# Fixed: Correct DiarizationPipeline import from whisperx.diarize
# Fixed: Event loop handling for RunPod serverless environment

# Using RunPod official base image with CUDA 12.6.2
FROM runpod/base:0.6.2-cuda12.6.2

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set working directory
WORKDIR /app

# Install system dependencies (RunPod base already has Python)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install base dependencies
RUN python3 -m pip install --no-cache-dir ffmpeg-python==0.2.0 requests>=2.31.0 aiohttp>=3.9.0 aiofiles>=23.2.1

# Install RunPod SDK
RUN python3 -m pip install --no-cache-dir runpod>=1.6.0

# Install WhisperX directly from main repo (handles all dependencies itself)
RUN python3 -m pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# Note: LD_LIBRARY_PATH for cuDNN is set dynamically in predict.py at runtime
# This ensures compatibility across different Python versions and base images

# Verify installations and check cuDNN version
RUN python3 -c "import sys; print(f'Python: {sys.version}'); import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ cuDNN version: {torch.backends.cudnn.version()}'); import whisperx; print('✓ WhisperX: OK')"

# Note: Model pre-download is disabled to prevent Docker build failures
# Models will download on first serverless run and cache to network volume (/runpod-volume)
# This adds ~30-60s to first cold start but enables reliable builds
# To pre-populate: Deploy temporary instance, attach volume, download models manually

# Copy application code
COPY . .

# Create cache directories (used if no network volume attached)
RUN mkdir -p /root/.cache/torch /root/.cache/huggingface

# Expose port for health checks
EXPOSE 8000

# Health check for RunPod
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Default command for RunPod serverless
CMD ["python3", "-u", "predict.py"]

