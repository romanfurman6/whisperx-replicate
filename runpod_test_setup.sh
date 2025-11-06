#!/bin/bash
set -e

echo "=========================================="
echo "WhisperX RunPod Test Setup"
echo "=========================================="

# Update system
echo "Updating system..."
apt-get update
apt-get install -y ffmpeg git wget curl build-essential

# Check CUDA
echo "Checking CUDA..."
nvidia-smi
echo "CUDA Version: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"

# Setup Python virtual environment
echo "Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch 2.5.1 with CUDA 12.1
echo "Installing PyTorch 2.5.1+cu121..."
pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir ffmpeg-python==0.2.0 requests aiohttp aiofiles
pip install --no-cache-dir runpod
pip install --no-cache-dir cog

# Install WhisperX v3.7.3
echo "Installing WhisperX v3.7.3..."
pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git@v3.7.3

# Re-lock PyTorch versions (WhisperX may have upgraded them)
echo "Locking PyTorch versions..."
pip install --no-cache-dir --force-reinstall --no-deps \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python3 -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python3 -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
python3 -c "import whisperx; print('WhisperX: OK')"
python3 -c "import cog; print(f'Cog: {cog.__version__}')"
python3 -c "import runpod; print(f'RunPod: {runpod.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To test, run:"
echo "  python3 predict.py"
echo ""

