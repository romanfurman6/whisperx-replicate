#!/bin/bash
set -e

echo "=========================================="
echo "WhisperX RunPod Test Setup"
echo "PyTorch 2.1.2 + pyannote.audio 2.1.1"
echo "VAD-compatible configuration"
echo "=========================================="

# Update system
echo "Updating system..."
apt-get update
apt-get install -y ffmpeg git wget curl build-essential

# Check CUDA
echo ""
echo "Checking CUDA..."
nvidia-smi
echo ""
nvcc --version 2>/dev/null || echo "nvcc not in PATH (this is OK)"
echo ""

# Setup Python virtual environment
echo "Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel

# Check if PyTorch is already installed
echo ""
echo "Checking existing PyTorch installation..."
python3 -c "import torch; print(f'Existing PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch not installed yet"

# Uninstall conflicting packages first
echo ""
echo "Removing conflicting packages..."
pip uninstall -y torch torchvision torchaudio pyannote-audio pyannote-pipeline pyannote-core whisperx 2>/dev/null || true

# Install PyTorch 2.1.2 (compatible with pyannote.audio 2.1.1 for VAD)
echo ""
echo "Installing PyTorch 2.1.2 (VAD-compatible)..."
pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install pyannote.audio 2.1.1 BEFORE WhisperX
echo ""
echo "Installing pyannote.audio 2.1.1 (VAD-compatible)..."
pip install --no-cache-dir \
    pyannote.audio==2.1.1 \
    pyannote.pipeline==2.1.1 \
    pyannote.core==5.0.0

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --no-cache-dir ffmpeg-python==0.2.0 requests aiohttp aiofiles
pip install --no-cache-dir runpod
pip install --no-cache-dir cog

# Install WhisperX from main branch (compatible with torch 2.1.x)
echo ""
echo "Installing WhisperX (main branch)..."
pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# Re-lock PyTorch and pyannote versions
echo ""
echo "Locking PyTorch and pyannote versions..."
pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'✓ CUDA version: {torch.version.cuda}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -c "import torch; print(f'✓ GPU: {torch.cuda.get_device_name(0)}')"
    python3 -c "import torch; print(f'✓ GPU count: {torch.cuda.device_count()}')"
fi
python3 -c "import torchaudio; print(f'✓ TorchAudio: {torchaudio.__version__}')"
python3 -c "import torchvision; print(f'✓ TorchVision: {torchvision.__version__}')"
python3 -c "import pyannote.audio; print(f'✓ pyannote.audio: {pyannote.audio.__version__}')"
python3 -c "import whisperx; print('✓ WhisperX: OK')"
python3 -c "import cog; print(f'✓ Cog: {cog.__version__}')"
python3 -c "import runpod; print(f'✓ RunPod: {runpod.__version__}')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - PyTorch: 2.1.2"
echo "  - pyannote.audio: 2.1.1 (VAD-compatible)"
echo "  - WhisperX: main branch"
echo "  - CUDA: 12.1 compatible"
echo ""
echo "VAD (Voice Activity Detection) is now enabled!"
echo ""
echo "To test, run:"
echo "  python3 predict.py"
echo ""
