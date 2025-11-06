# RunPod Serverless Deployment Guide

This guide will help you deploy WhisperX Multi-Chunk to RunPod Serverless.

## Prerequisites

- Docker installed and running
- Docker Hub account
- RunPod account with serverless access
- (Optional) HuggingFace token for speaker diarization

## Quick Start

### 1. Build and Deploy

Run the deployment script:

```bash
./deploy.sh
```

The script will:
- Build the Docker image
- Push to Docker Hub
- Provide deployment instructions

### 2. Manual Build (Alternative)

If you prefer to build manually:

```bash
# Set your Docker Hub username
export DOCKER_USERNAME="your-dockerhub-username"

# Build the image
docker build -t ${DOCKER_USERNAME}/whisperx-runpod-serverless:latest .

# Login to Docker Hub
docker login

# Push the image
docker push ${DOCKER_USERNAME}/whisperx-runpod-serverless:latest
```

## RunPod Configuration

### 1. Create Serverless Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **"New Endpoint"**
3. Select **"Custom Image"**

### 2. Container Configuration

**Image Name:**
```
docker.io/YOUR_USERNAME/whisperx-runpod-serverless:latest
```

**Container Settings:**
- **Container Disk:** 10-20 GB (minimum 10GB for models)
- **Volume Size:** Optional, for persistent model caching

### 3. GPU Selection

Recommended GPUs (in order of preference):
1. **RTX 4090** - Best price/performance
2. **A40** - Good for medium workloads
3. **A100** - Best for large batches

### 4. Scaling Configuration

```yaml
Workers:
  Min Workers: 0 (for cost savings)
  Max Workers: 3-10 (based on load)
  
Timeouts:
  Idle Timeout: 5 seconds
  Execution Timeout: 300 seconds (5 minutes)
  
Throttle:
  Requests per second: 10
```

### 5. Environment Variables (Optional)

Add if using speaker diarization:

```bash
HF_TOKEN=your_huggingface_token_here
```

Get HuggingFace token at: https://huggingface.co/settings/tokens

## Testing Your Endpoint

### Using RunPod SDK

```python
import runpod

# Initialize client
runpod.api_key = "your-runpod-api-key"

# Create endpoint client
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

# Run inference
job = endpoint.run({
    "input": {
        "audio_urls": [
            "https://example.com/audio_chunk_1.mp3",
            "https://example.com/audio_chunk_2.mp3"
        ],
        "total_duration_seconds": 60.0,
        "chunk_size_seconds": 30.0,
        "language": "en",
        "align_output": True,
        "debug": True
    }
})

# Wait for result
result = job.output()
print(result)
```

### Using cURL

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio_urls": ["https://example.com/audio.mp3"],
      "total_duration_seconds": 30.0,
      "chunk_size_seconds": 30.0,
      "language": "en",
      "align_output": true,
      "debug": true
    }
  }'
```

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `audio_urls` | List[str] | Yes | - | URLs of audio chunks |
| `total_duration_seconds` | float | Yes | - | Total audio duration |
| `chunk_size_seconds` | float | Yes | - | Duration per chunk |
| `language` | str | No | auto-detect | ISO language code |
| `align_output` | bool | No | False | Word-level timestamps |
| `diarization` | bool | No | False | Speaker identification |
| `batch_size` | int | No | 32 | Transcription batch size |
| `temperature` | float | No | 0.2 | Sampling temperature |
| `debug` | bool | No | True | Debug logging |

## Output Format

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Transcribed text here",
      "chunk_index": 0,
      "words": [
        {"start": 0.0, "end": 0.5, "word": "Transcribed"}
      ]
    }
  ],
  "detected_language": "en",
  "total_chunks": 2,
  "processing_time": 12.5
}
```

## Troubleshooting

### CUDA Errors

The updated `predict.py` includes:
- Automatic CUDA cache clearing
- Model reload on CUDA errors
- Retry logic with exponential backoff
- Proper memory management

### Cold Start Issues

To reduce cold starts:
1. Set `Min Workers: 1` (costs more but faster)
2. Use network volumes for model caching
3. Pre-warm endpoints with test requests

### Memory Issues

If you encounter OOM errors:
- Reduce `batch_size` parameter (try 16 or 8)
- Use larger GPU (A40 or A100)
- Reduce `chunk_size_seconds`

### Network Timeouts

For large files:
- Increase `Execution Timeout` in RunPod settings
- Split into smaller chunks
- Use faster storage (S3, GCS with CDN)

## Cost Optimization

### Tips to Reduce Costs

1. **Use Spot Instances:**
   - Enable "Spot" workers in RunPod
   - Can save 50-70% on GPU costs

2. **Set Idle Timeout:**
   - Keep at 5 seconds to spin down quickly
   - Workers auto-scale to zero when idle

3. **Choose Right GPU:**
   - RTX 4090: Best value for most workloads
   - A40: For medium workloads
   - A100: Only for very large batches

4. **Batch Processing:**
   - Process multiple files in one request
   - Reuse loaded models across chunks

## Monitoring

### View Logs

In RunPod Console:
1. Go to your endpoint
2. Click "Logs" tab
3. View real-time logs

### Metrics to Watch

- **Request Count:** Total requests processed
- **Execution Time:** Average processing time
- **Error Rate:** Failed requests percentage
- **GPU Utilization:** GPU usage during processing

## Updates and Maintenance

### Updating Your Deployment

1. Make changes to code
2. Run deployment script again:
   ```bash
   ./deploy.sh
   ```
3. RunPod will pull new image automatically
4. No downtime during update

### Version Tagging

Use semantic versioning:

```bash
export IMAGE_TAG="v1.0.0"
./deploy.sh
```

## Support

- **RunPod Docs:** https://docs.runpod.io/
- **WhisperX Issues:** https://github.com/m-bain/whisperX/issues
- **This Project:** [Your GitHub Issues Link]

## Best Practices

1. **Always test locally first** with `docker run`
2. **Use environment variables** for sensitive data
3. **Monitor costs** in RunPod dashboard
4. **Set up alerts** for high usage
5. **Version your images** with tags
6. **Keep models cached** in network volumes
7. **Use debug mode** initially, disable in production

---

**Last Updated:** November 2024

