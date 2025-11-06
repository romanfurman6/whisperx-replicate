# Quick Start Guide

## Deploy to RunPod in 5 Minutes

### Step 1: Build and Push Docker Image

```bash
# Set your Docker Hub username
export DOCKER_USERNAME="your-username"

# Run the deploy script
./deploy.sh
```

### Step 2: Create RunPod Endpoint

1. Go to https://www.runpod.io/console/serverless
2. Click **"New Endpoint"** → **"Custom Image"**
3. Enter image: `docker.io/YOUR_USERNAME/whisperx-runpod-serverless:latest`
4. Configure:
   - Container Disk: **10 GB**
   - GPU: **RTX 4090** (recommended)
   - Min Workers: **0**
   - Idle Timeout: **5 seconds**
5. Click **"Deploy"**

### Step 3: Test Your Endpoint

```python
import runpod

runpod.api_key = "YOUR_API_KEY"
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

job = endpoint.run({
    "input": {
        "audio_urls": ["https://example.com/audio.mp3"],
        "total_duration_seconds": 30.0,
        "chunk_size_seconds": 30.0,
        "language": "en",
        "align_output": True,
        "debug": True
    }
})

print(job.output())
```

## Common Issues

### "CUDA failed with error unknown error"
**Fixed!** The updated code includes automatic CUDA error recovery.

### Out of Memory
- Reduce `batch_size` to 16 or 8
- Use a larger GPU (A40 or A100)

### Slow Cold Starts
- Set Min Workers to 1
- Use network volumes for model caching

## Need Help?

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

