# whisperX on Replicate

This repo is the codebase behind the following Replicate models, which we use at [Upmeet](https://upmeet.ai):

- [victor-upmeet/whisperx](https://replicate.com/victor-upmeet/whisperx) : if you don't know which model to use, use this one. It uses a low-cost hardware, which suits most cases
- [victor-upmeet/whisperx-a40-large](https://replicate.com/victor-upmeet/whisperx-a40-large) : if you encounter some memory issues with previous models, consider this one. It can happen when dealing with long audio files and performing alignment and/or diarization
- [victor-upmeet/whisperx-a100-80gb](https://replicate.com/victor-upmeet/whisperx-a100-80gb) : if you encounter some memory issues with previous models, consider this one. It can happen when dealing with long audio files and performing alignment and/or diarization

# Model Information

WhisperX provides fast automatic speech recognition (70x realtime with large-v3) with word-level timestamps and speaker diarization.

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI's whisper does not natively support batching, but WhisperX does.

Model used is for transcription is large-v3 from faster-whisper.

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

# Citation

```
@misc{bain2023whisperx,
      title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio}, 
      author={Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
      year={2023},
      eprint={2303.00747},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```