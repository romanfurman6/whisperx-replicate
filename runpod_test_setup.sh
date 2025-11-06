#!/bin/bash
set -e

echo "=========================================="
echo "WhisperX RunPod Test Setup"
echo "Using official WhisperX repo"
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

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --no-cache-dir ffmpeg-python==0.2.0 requests aiohttp aiofiles
pip install --no-cache-dir runpod
pip install --no-cache-dir cog

# Install WhisperX directly from main repo (handles all dependencies itself)
echo ""
echo "Installing WhisperX from main repo..."
pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

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
echo "  - WhisperX: main branch (latest)"
echo "  - All dependencies auto-resolved by WhisperX"
echo ""
echo "To test, run:"
echo "  python3 predict.py"
echo ""
