#!/bin/bash

# WhisperX RunPod Serverless - Build and Deploy Script
# This script builds the Docker image and pushes it to Docker Hub

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  WhisperX Multi-Chunk RunPod Serverless Deployment       ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
}

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
IMAGE_NAME="${IMAGE_NAME:-whisperx-runpod-serverless}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

print_header

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi
print_status "Docker is installed ✓"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi
print_status "Docker daemon is running ✓"

# Get Docker username if not set
if [ -z "$DOCKER_USERNAME" ]; then
    echo ""
    read -p "Enter your Docker Hub username: " DOCKER_USERNAME
    if [ -z "$DOCKER_USERNAME" ]; then
        print_error "Docker Hub username is required."
        exit 1
    fi
fi

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

print_status "Building image: ${FULL_IMAGE_NAME}"
echo ""

# Check if user is logged in to Docker Hub
if ! docker info | grep -q "Username: ${DOCKER_USERNAME}"; then
    print_warning "You are not logged in to Docker Hub."
    echo ""
    read -p "Do you want to login now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker login
    else
        print_error "You must be logged in to push to Docker Hub."
        exit 1
    fi
fi
print_status "Logged in to Docker Hub ✓"

# Build the Docker image
print_status "Starting Docker build for linux/amd64 platform..."
echo ""

if docker build --platform linux/amd64 -t "${FULL_IMAGE_NAME}" .; then
    print_status "Docker build completed successfully ✓"
else
    print_error "Docker build failed!"
    exit 1
fi

# Get image size
IMAGE_SIZE=$(docker images "${FULL_IMAGE_NAME}" --format "{{.Size}}")
print_status "Image size: ${IMAGE_SIZE}"

# Ask user if they want to push
echo ""
read -p "Do you want to push the image to Docker Hub? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Pushing image to Docker Hub..."
    echo ""
    
    if docker push "${FULL_IMAGE_NAME}"; then
        print_status "Image pushed successfully ✓"
        echo ""
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║              Deployment Information                        ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "  Image: ${BLUE}${FULL_IMAGE_NAME}${NC}"
        echo -e "  Size:  ${BLUE}${IMAGE_SIZE}${NC}"
        echo ""
        echo -e "${YELLOW}Next Steps:${NC}"
        echo "  1. Go to RunPod Serverless: https://www.runpod.io/console/serverless"
        echo "  2. Click 'New Endpoint'"
        echo "  3. Select 'Custom Image'"
        echo "  4. Enter image name: ${FULL_IMAGE_NAME}"
        echo "  5. Configure:"
        echo "     - Container Disk: 10GB (minimum)"
        echo "     - GPU Type: RTX 4090, A40, or A100 recommended"
        echo "     - Max Workers: Based on your needs"
        echo "     - Idle Timeout: 5 seconds"
        echo "  6. Add environment variables (if needed):"
        echo "     - HF_TOKEN: Your HuggingFace token (for diarization)"
        echo "  7. Click 'Deploy'"
        echo ""
        echo -e "${GREEN}Done! Your serverless endpoint will be ready in a few minutes.${NC}"
        echo ""
    else
        print_error "Failed to push image to Docker Hub!"
        exit 1
    fi
else
    print_warning "Image not pushed to Docker Hub."
    echo ""
    echo "To push manually later, run:"
    echo "  docker push ${FULL_IMAGE_NAME}"
fi

# Optional: Test the image locally
echo ""
read -p "Do you want to test the image locally? (requires GPU) (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting local test..."
    echo ""
    print_warning "Note: This requires a CUDA-capable GPU on your system."
    echo ""
    
    docker run --rm --gpus all \
        -e CUDA_VISIBLE_DEVICES=0 \
        "${FULL_IMAGE_NAME}" \
        python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    
    print_status "Local test completed ✓"
fi

echo ""
print_status "Script completed successfully!"

