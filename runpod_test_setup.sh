#!/bin/bash
set -e

echo "=========================================="
echo "WhisperX RunPod Test Setup"
echo "PyTorch 2.1.0 (whisperx-worker verified config)"
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

# Install PyTorch 2.1.0 (from working whisperx-worker repo)
echo ""
echo "Installing PyTorch 2.1.0+cu121..."
pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --no-cache-dir ffmpeg-python==0.2.0 requests aiohttp aiofiles
pip install --no-cache-dir runpod
pip install --no-cache-dir cog

# Install WhisperX and pyannote (exact versions from working whisperx-worker repo)
echo ""
echo "Installing WhisperX (commit 8f00339) and pyannote.audio 3.1.1..."
pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperX.git@8f00339af7dcc9705ef40d97a1f40764b7cf555f \
    pyannote.audio==3.1.1 \
    speechbrain==0.5.16

# Re-lock PyTorch versions
echo ""
echo "Locking PyTorch versions..."
pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

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
echo "  - PyTorch: 2.1.0+cu121"
echo "  - pyannote.audio: 3.1.1"
echo "  - WhisperX: commit 8f00339 (verified working)"
echo "  - CUDA: 12.1 compatible"
echo ""
echo "To test, run:"
echo "  python3 predict.py"
echo ""
