#!/bin/bash

# WhisperX RunPod Build Script
# Quick build without deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "=========================================="
echo "  WhisperX RunPod - Quick Build"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

print_status "Docker is installed ✓"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi

print_status "Docker daemon is running ✓"

# Default image name
IMAGE_NAME="whisperx-runpod-serverless:latest"

print_status "Building Docker image for linux/amd64: ${IMAGE_NAME}"
echo ""

# Build the image for AMD64 (RunPod compatibility)
if docker build --platform linux/amd64 -t "${IMAGE_NAME}" .; then
    echo ""
    print_status "✓ Build completed successfully!"
    echo ""
    
    # Get image size
    IMAGE_SIZE=$(docker images "${IMAGE_NAME}" --format "{{.Size}}")
    print_status "Image size: ${IMAGE_SIZE}"
    echo ""
    
    echo "=========================================="
    echo "  Next Steps"
    echo "=========================================="
    echo ""
    echo "1. To deploy to Docker Hub:"
    echo "   ./deploy.sh"
    echo ""
    echo "2. To test locally (requires GPU):"
    echo "   docker run --rm --gpus all ${IMAGE_NAME} \\"
    echo "     python3 -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
    echo ""
    echo "3. See DEPLOYMENT.md for full instructions"
    echo ""
else
    print_error "Build failed!"
    exit 1
fi
