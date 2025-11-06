#!/bin/bash
set -e

# WhisperX RunPod Docker Build and Push Script
# Version: 1.2

VERSION="v1.2"
IMAGE_NAME="romanfurman/whisperx-runpod-serverless"

echo "=========================================="
echo "Building WhisperX RunPod Docker Image"
echo "Version: ${VERSION}"
echo "=========================================="

# Build for AMD64 (RunPod requires this architecture)
echo ""
echo "Building Docker image..."
docker build --platform linux/amd64 \
    -t ${IMAGE_NAME}:${VERSION} \
    -t ${IMAGE_NAME}:latest \
    .

echo ""
echo "✓ Build complete!"
echo ""
echo "Image tags:"
echo "  - ${IMAGE_NAME}:${VERSION}"
echo "  - ${IMAGE_NAME}:latest"
echo ""
echo "=========================================="
echo "Pushing to Docker Hub..."
echo "=========================================="

# Push both tags
docker push ${IMAGE_NAME}:${VERSION}
docker push ${IMAGE_NAME}:latest

echo ""
echo "=========================================="
echo "✓ Successfully built and pushed v${VERSION}"
echo "=========================================="
echo ""
echo "Image available at:"
echo "  docker pull ${IMAGE_NAME}:${VERSION}"
echo "  docker pull ${IMAGE_NAME}:latest"
echo ""

