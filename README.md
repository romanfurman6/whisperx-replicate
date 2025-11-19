# WhisperX Multi-Chunk for RunPod Serverless

High-performance automatic speech recognition (ASR) optimized for RunPod serverless infrastructure. Processes large audio/video files by splitting them into manageable chunks with automatic timestamp alignment.

## 🚀 Quick Deploy

> **⚠️ Important for M1/M2/M3 Mac users:** The image is automatically built for AMD64 (RunPod compatible). No action needed!

```bash
# 1. Build and deploy (automatically builds for AMD64)
./deploy.sh

# 2. Configure on RunPod
# See QUICKSTART.md for step-by-step instructions
```

## ✨ Key Improvements

- ✅ **Fixed CUDA Errors**: Automatic recovery from "CUDA failed with error unknown error"
- ✅ **Memory Management**: Intelligent cache clearing and model reloading
- ✅ **Retry Logic**: Automatic retries with exponential backoff
- ✅ **Better Logging**: Comprehensive debug information
- ✅ **RunPod Optimized**: Built specifically for serverless deployment

# Model Information

WhisperX provides fast automatic speech recognition (70x realtime with large-v3-turbo) with word-level timestamps and speaker diarization.

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching, but WhisperX does.

Model used is for transcription is large-v3-turbo (CTranslate2 version from deepdml) from faster-whisper.

For more information about WhisperX, including implementation details, see the [WhisperX github repo](https://github.com/m-bain/whisperX).

## Multi-Chunk Processing

This implementation supports processing multiple audio chunks sequentially, which is useful for transcribing large videos that have been split into smaller segments. 

### Key Features:

- **Sequential Processing**: Audio chunks are processed one at a time to ensure consistent language detection and memory management
- **Automatic Timestamp Alignment**: Timestamps are automatically adjusted to maintain temporal continuity across chunks
- **Smart Merging**: Results are merged while preserving the correct temporal order of segments
- **Language Detection**: Language is detected from the first chunk and applied consistently across all chunks
- **URL Download Support**: Works with public audio URLs with automatic download and cleanup

### Input Parameters:

- `audio_urls`: Array of public audio URLs (chunks of one video in temporal order)
- `total_duration_seconds`: Total duration of the complete audio in seconds
- `chunk_size_seconds`: Duration of each chunk in seconds (used for timestamp calculation). Latest chunk can be shorter, it will be calculated based on the total duration and the number of chunks.
- All other parameters remain the same as the original WhisperX model

### Usage Example:

```python
# Process multiple chunks of a video
audio_urls = [
    "https://example.com/video_chunk_001.mp3",
    "https://example.com/video_chunk_002.mp3", 
    "https://example.com/video_chunk_003.mp3"
]

result = predictor.predict(
    audio_urls=audio_urls,
    total_duration_seconds=90.0,  # Total duration of complete audio
    chunk_size_seconds=30.0,      # Duration of each chunk
    language="en",                # Optional: will be auto-detected if not provided
    align_output=True,
    debug=True
)
```

The output includes merged segments with correct timestamps, total processing time, and chunk count information.

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Deploy in 5 minutes
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment guide
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## 🔧 Development

```bash
# Build locally
./build.sh

# Test locally (requires GPU)
python3 test_local.py

# Deploy to Docker Hub
./deploy.sh
```

## 🐛 Troubleshooting

### CUDA Errors (Fixed!)

The code now automatically handles:
- CUDA initialization failures
- Out of memory errors
- Model loading issues
- Cache management

### Still Having Issues?

See [DEPLOYMENT.md](DEPLOYMENT.md#troubleshooting) for detailed solutions.

## 📄 License

This project is open source. See LICENSE file for details.

# Citation

```bibtex
@misc{bain2023whisperx,
      title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio}, 
      author={Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
      year={2023},
      eprint={2303.00747},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## 🙏 Acknowledgments

- **WhisperX**: [m-bain/whisperX](https://github.com/m-bain/whisperX)
- **RunPod**: Serverless GPU infrastructure
- **OpenAI**: Original Whisper model