"""
WhisperX RunPod Serverless Worker
Multi-chunk audio transcription with speaker diarization support.
"""

import os
import site

# Configure cache directories to use RunPod network volume (if available)
# This enables persistent model caching across serverless invocations
VOLUME_PATH = "/runpod-volume"
if os.path.exists(VOLUME_PATH):
    os.environ['HF_HOME'] = f"{VOLUME_PATH}/huggingface"
    os.environ['TORCH_HOME'] = f"{VOLUME_PATH}/torch"
    os.environ['XDG_CACHE_HOME'] = f"{VOLUME_PATH}/cache"
    print(f"✓ Using network volume for model caching: {VOLUME_PATH}")
else:
    print("⚠ No network volume found, models will download on each cold start")

# Fix cuDNN version mismatch - use WhisperX's installed cuDNN
# This must be set before importing torch/whisperx
# See: https://github.com/m-bain/whisperX/blob/main/docs/troubleshooting.md

# Dynamically find the cuDNN paths (works across Python 3.10, 3.11, 3.12, etc.)
site_packages = site.getsitepackages()[0]
cudnn_base = os.path.join(site_packages, "nvidia", "cudnn")
candidate_dirs = [
    os.path.join(cudnn_base, "lib"),
    os.path.join(cudnn_base, "lib64"),
    os.path.join(cudnn_base, "lib", "stubs"),
]

existing_dirs = [path for path in candidate_dirs if os.path.exists(path)]

if existing_dirs:
    original_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths = existing_dirs + ([original_path] if original_path else [])
    os.environ['LD_LIBRARY_PATH'] = ":".join([p for p in paths if p])
    print(f"✓ Using WhisperX cuDNN from: {', '.join(existing_dirs)}")
else:
    print(f"⚠ WhisperX cuDNN not found under: {cudnn_base}, using system cuDNN")

from typing import Any, List, Dict, Optional
from dataclasses import dataclass, field
from whisperx.audio import N_SAMPLES, log_mel_spectrogram
from whisperx.diarize import DiarizationPipeline

import asyncio
import aiohttp
import aiofiles
import gc
import math
import shutil
import whisperx
import tempfile
import time
import torch
import ffmpeg
from pathlib import Path as PathlibPath
import urllib.parse
import sys

compute_type = "float16"
device = "cuda"
whisper_arch = "deepdml/faster-whisper-large-v3-turbo-ct2"


def ensure_cuda_initialized():
    """Ensure CUDA is properly initialized with error handling."""
    try:
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.empty_cache()
            # Force CUDA context creation
            torch.zeros(1).cuda()
            return True
        return False
    except Exception as e:
        print(f"CUDA initialization warning: {e}")
        return False


@dataclass
class ChunkResult:
    """Result from processing a single audio chunk."""
    chunk_index: int
    segments: Any
    detected_language: str
    start_time_offset: float


def sanitize_for_json(value: Any) -> Any:
    """Convert common model outputs into JSON-serializable primitives."""
    if value is None:
        return None

    if isinstance(value, (str, bool)):
        return value

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0

    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(v) for v in value]

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")

    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            pass

    try:
        coerced = float(value)
        return coerced if math.isfinite(coerced) else 0.0
    except (TypeError, ValueError):
        return str(value)


@dataclass
class SupabaseRealtimePublisher:
    """Best-effort publisher for Supabase Realtime broadcast RPC."""
    project_url: Optional[str]
    service_role_key: Optional[str]
    channel_prefix: str = "task"
    topic_suffix: str = "updates"
    event_update: str = "partial"
    event_complete: str = "final"
    event_failed: str = "error"
    private: bool = False
    timeout_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> "SupabaseRealtimePublisher":
        """Initialize publisher configuration from environment variables."""
        project_url = os.getenv("SUPABASE_PROJECT_URL") or os.getenv("SUPABASE_URL")
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

        channel_prefix = os.getenv("SUPABASE_REALTIME_CHANNEL_PREFIX", "task")
        topic_suffix = os.getenv("SUPABASE_REALTIME_TOPIC_SUFFIX", "updates")
        event_update = os.getenv("SUPABASE_REALTIME_EVENT_UPDATE", "partial")
        event_complete = os.getenv("SUPABASE_REALTIME_EVENT_COMPLETE", "final")
        event_failed = os.getenv("SUPABASE_REALTIME_EVENT_FAILED", "error")

        private_raw = os.getenv("SUPABASE_REALTIME_PRIVATE", "false").lower()
        private = False

        timeout_seconds = float(os.getenv("SUPABASE_REALTIME_TIMEOUT_SECONDS", "2.0"))

        return cls(
            project_url=project_url,
            service_role_key=service_role_key,
            channel_prefix=channel_prefix,
            topic_suffix=topic_suffix,
            event_update=event_update,
            event_complete=event_complete,
            event_failed=event_failed,
            private=private,
            timeout_seconds=timeout_seconds,
        )

    @property
    def is_enabled(self) -> bool:
        return bool(self.project_url and self.service_role_key)

    def _broadcast_endpoint(self) -> Optional[str]:
        if not self.project_url:
            return None
        base = self.project_url.rstrip("/")
        return f"{base}/rest/v1/rpc/broadcast"

    def _build_topic(self, task_id: Optional[str]) -> Optional[str]:
        if not task_id:
            print("⚠ Supabase realtime broadcast skipped: task_id not provided")
            return None
        prefix = self.channel_prefix.strip(":")
        suffix = self.topic_suffix.strip(":")
        return f"{prefix}:{task_id}:{suffix}"

    async def broadcast(self, task_id: Optional[str], event: str, payload: Dict[str, Any]) -> None:
        """Send a broadcast payload to Supabase Realtime."""
        if not self.is_enabled:
            return

        endpoint = self._broadcast_endpoint()
        topic = self._build_topic(task_id)
        if not endpoint or not topic:
            return

        headers = {
            "Content-Type": "application/json",
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
        }

        body = {
            "topic": topic,
            "event": event,
            "payload": payload,
            "private": self.private,
        }

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=body, headers=headers) as response:
                    if response.status >= 400:
                        snippet = (await response.text())[:256]
                        print(f"⚠ Supabase Realtime HTTP {response.status}: {snippet}")
        except Exception as exc:
            print(f"⚠ Supabase Realtime error: {exc}")


@dataclass
class RealtimeUpdateClient:
    """Best-effort client for Supabase realtime transcription updates."""
    supabase_publisher: Optional[SupabaseRealtimePublisher] = None
    task_id: Optional[str] = None
    min_interval_seconds: float = 0.5
    _last_emit_monotonic: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def is_enabled(self) -> bool:
        return bool(self.supabase_publisher and self.supabase_publisher.is_enabled)

    async def emit_snapshot(
        self,
        segments: List[Dict[str, Any]]
    ) -> None:
        """Send a cumulative snapshot to the Supabase realtime channel."""
        if not self.is_enabled:
            return

        now = time.monotonic()
        if (
            self.min_interval_seconds > 0
            and self._last_emit_monotonic is not None
            and (now - self._last_emit_monotonic) < self.min_interval_seconds
        ):
            return
        self._last_emit_monotonic = now

        sanitized_segments = sanitize_for_json(segments)
        supabase_payload: Dict[str, Any] = {
            "task_id": self.task_id,
            "segments": sanitized_segments
        }

        if sanitized_segments:
            first_start = sanitized_segments[0].get("start")
            last_end = sanitized_segments[-1].get("end")
        else:
            first_start = None
            last_end = None

        await self.supabase_publisher.broadcast(
            task_id=self.task_id,
            event=self.supabase_publisher.event_update,
            payload=sanitize_for_json(supabase_payload)
        )

        print(
            f"→ Supabase partial broadcast | task_id={self.task_id} "
            f"| segments={len(sanitized_segments)} "
            f"| window=({first_start}, {last_end})"
        )

    async def emit_completion(self, payload: Dict[str, Any], *, success: bool = True) -> None:
        """Send completion/failure event to Supabase Realtime."""
        if not self.is_enabled:
            return

        payload = {"task_id": self.task_id, **(payload or {})}
        event = self.supabase_publisher.event_complete if success else self.supabase_publisher.event_failed
        await self.supabase_publisher.broadcast(
            task_id=self.task_id,
            event=event,
            payload=sanitize_for_json(payload)
        )

        print(
            f"→ Supabase {'final' if success else 'error'} broadcast | "
            f"task_id={self.task_id} | payload={payload}"
        )


class Predictor:
    """WhisperX Predictor for RunPod Serverless."""
    
    def __init__(self):
        self._asr_model_cached = None
        self._cached_language = None
        self._setup_complete = False
        self._diarize_model_cached = None  # Cache diarization model for reuse
        self._supabase_publisher = SupabaseRealtimePublisher.from_env()
        try:
            interval_value = float(os.getenv("SUPABASE_REALTIME_MIN_INTERVAL_SECONDS", "0.5"))
            self._realtime_min_interval = max(0.0, interval_value)
        except ValueError:
            self._realtime_min_interval = 0.5

    def setup(self):
        """Initialize the predictor with proper CUDA error handling."""
        print("=" * 50)
        print("Setting up WhisperX Predictor...")
        print("=" * 50)
        
        # Initialize CUDA
        cuda_available = ensure_cuda_initialized()
        if cuda_available:
            print(f"✓ CUDA initialized successfully")
            print(f"  - Device count: {torch.cuda.device_count()}")
            print(f"  - Current device: {torch.cuda.current_device()}")
            print(f"  - Device name: {torch.cuda.get_device_name(0)}")
            print(f"  - Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        else:
            print("✗ WARNING: CUDA not available! Model will run on CPU (very slow)")

        # Setup VAD model directory
        source_folder = './models/vad'
        destination_folder = os.path.expanduser('~/.cache/torch')
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)
            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)
                print(f"✓ Copied VAD model to {destination_folder}")
        else:
            print(f"⚠ VAD model not found at {source_file_path}")
        
        self._setup_complete = True
        print("✓ Setup completed successfully")
        print("=" * 50)

    def _run_coro(self, coro):
        """Run async coroutine in sync context with proper event loop handling."""
        import concurrent.futures
        import threading
        
        # Always run in a new thread with its own event loop to avoid conflicts
        def run_in_thread():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()

    def predict(
            self,
            audio_urls: List[str],
            total_duration_seconds: float,
            chunk_size_seconds: float,
            language: Optional[str] = None,
            language_detection_min_prob: float = 0.7,
            language_detection_max_tries: int = 5,
            initial_prompt: Optional[str] = None,
            batch_size: int = 32,
            temperature: float = 0.2,
            vad_onset: float = 0.500,
            vad_offset: float = 0.363,
            align_output: bool = False,
            diarization: bool = False,
            huggingface_access_token: Optional[str] = None,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None,
            debug: bool = True,
            realtime: bool = False,
            task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio chunks with optional alignment and diarization.
        
        Args:
            audio_urls: List of public audio URLs to process
            total_duration_seconds: Total duration of the complete audio
            chunk_size_seconds: Duration of each chunk in seconds
            language: ISO code of the language (None for auto-detection)
            language_detection_min_prob: Minimum probability for language detection
            language_detection_max_tries: Maximum retries for language detection
            initial_prompt: Optional text prompt for the first window
            batch_size: Parallelization of input audio transcription
            temperature: Temperature to use for sampling
            vad_onset: VAD onset threshold
            vad_offset: VAD offset threshold
            align_output: Whether to align output for word-level timestamps
            diarization: Whether to perform speaker diarization
            huggingface_access_token: HuggingFace token for diarization models
            min_speakers: Minimum number of speakers if diarization is activated
            max_speakers: Maximum number of speakers if diarization is activated
            debug: Print debug information
            realtime: Whether to emit realtime updates
            task_id: Optional transcription task ID (used for Supabase broadcast topics)
            
        Returns:
            Dictionary containing transcription results
        """
        start_processing_time = time.time()
        
        if not audio_urls:
            raise ValueError("audio_urls cannot be empty")
        
        num_chunks = len(audio_urls)
        
        if debug:
            print(f"\n{'='*50}")
            print(f"Processing {num_chunks} audio chunks")
            print(f"Total duration: {total_duration_seconds:.2f}s")
            print(f"Chunk size: {chunk_size_seconds:.2f}s")
            print(f"{'='*50}\n")
        
        # Language detection if not provided
        if language is None and debug:
            print("Language not specified, will detect from first chunk...")
        
        # Reset cached ASR model when language needs to be detected.
        # The previous invocation might have pinned the model to a specific language,
        # which would bias the detector. Clearing cache ensures fresh detection.
        if language is None:
            self._cached_language = None
            self._asr_model_cached = None
        
        # Process chunks
        task_identifier = task_id or None
        if realtime and not task_identifier:
            print("⚠ Realtime enabled but no task_id provided in job input; supabase broadcasts will be skipped.")

        realtime_client = RealtimeUpdateClient(
            supabase_publisher=self._supabase_publisher if (realtime and self._supabase_publisher and self._supabase_publisher.is_enabled) else None,
            task_id=task_identifier,
            min_interval_seconds=self._realtime_min_interval
        )

        try:
            chunk_results = self._run_coro(self.download_and_process_pipeline(
                audio_urls, language, chunk_size_seconds, batch_size,
                temperature, initial_prompt, vad_onset, vad_offset,
                align_output, diarization, huggingface_access_token,
                min_speakers, max_speakers, debug, language_detection_min_prob,
                language_detection_max_tries, realtime_client, total_duration_seconds
            ))
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if realtime_client.is_enabled:
                failure_payload = {
                    "status": "failed",
                    "error": str(e),
                }
                self._run_coro(
                    realtime_client.emit_completion(failure_payload, success=False)
                )
            raise

        # Merge results
        merged = self.merge_chunk_results(chunk_results, debug)
        processing_time = time.time() - start_processing_time
        
        if debug:
            print(f"\n{'='*50}")
            print(f"✓ Processing completed in {processing_time:.2f}s")
            print(f"{'='*50}\n")
        
        if realtime_client.is_enabled:
            completion_payload = {"status": "completed"}
            self._run_coro(
                realtime_client.emit_completion(completion_payload, success=True)
            )

        return {
            "segments": merged["segments"],
            "detected_language": merged["language"],
            "total_chunks": len(audio_urls),
            "processing_time": processing_time
        }

    async def download_audio_files(self, urls: List[str]) -> List[PathlibPath]:
        """Download audio files asynchronously."""
        downloaded_files = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, url in enumerate(urls):
                task = self.download_single_file(session, url, i)
                tasks.append(task)
            
            downloaded_files = await asyncio.gather(*tasks)
        
        return downloaded_files
    
    async def download_single_file(self, session: aiohttp.ClientSession, url: str, index: int) -> PathlibPath:
        """Download a single audio file."""
        try:
            print(f"  Downloading chunk {index + 1}: {url}")
            
            parsed_url = urllib.parse.urlparse(url)
            file_extension = PathlibPath(parsed_url.path).suffix or '.mp3'
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_path = PathlibPath(temp_file.name)
            temp_file.close()
            
            async with session.get(url) as response:
                response.raise_for_status()
                
                async with aiofiles.open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
            
            print(f"  ✓ Downloaded chunk {index + 1}")
            return temp_path
            
        except Exception as e:
            print(f"  ✗ Error downloading chunk {index + 1}: {e}")
            raise
    
    async def download_and_process_pipeline(self, urls: List[str], language: str, chunk_duration: float,
                                           batch_size: int, temperature: float, initial_prompt: str,
                                           vad_onset: float, vad_offset: float,
                                           align_output: bool, diarization: bool,
                                           huggingface_access_token: str, min_speakers: int, max_speakers: int,
                                           debug: bool, language_detection_min_prob: float,
                                           language_detection_max_tries: int,
                                           realtime_client: Optional[RealtimeUpdateClient],
                                           total_duration_seconds: Optional[float]) -> List[ChunkResult]:
        """Download and process chunks with pipelined execution."""
        
        # If no language specified, detect from first NON-SILENT chunk
        if language is None:
            print("Detecting language from first audio chunk with speech...")
            async with aiohttp.ClientSession() as detection_session:
                max_attempts = min(5, len(urls))  # Try first 5 chunks max
                
                for attempt_idx in range(max_attempts):
                    if debug:
                        print(f"  Trying chunk {attempt_idx + 1} for language detection...")
                    
                    chunk_file = await self.download_single_file(detection_session, urls[attempt_idx], attempt_idx)
                    
                    try:
                        # Quick transcribe to check for speech and detect language
                        audio = whisperx.load_audio(str(chunk_file))
                        
                        # Load model if not cached
                        if self._asr_model_cached is None:
                            if debug:
                                print("  Loading WhisperX model...")
                            self._asr_model_cached = self._load_model_with_retry(None, temperature, initial_prompt, vad_onset, vad_offset, debug)
                        
                        # Transcribe with small batch for speed
                        result = self._asr_model_cached.transcribe(audio, batch_size=min(4, batch_size))
                        
                        # Check if we found speech
                        if result.get("segments") and len(result["segments"]) > 0:
                            language = result.get("language", "en")
                            print(f"✓ Detected language: {language} from chunk {attempt_idx + 1}")
                            chunk_file.unlink()
                            break
                        else:
                            if debug:
                                print(f"  ⚠ Chunk {attempt_idx + 1} is silent, trying next...")
                            chunk_file.unlink()
                            
                    except Exception as e:
                        print(f"  ⚠ Error detecting language from chunk {attempt_idx + 1}: {e}")
                        chunk_file.unlink()
                        continue
                
                # Fallback to English if no speech found
                if language is None:
                    print("  ⚠ No speech found in first 5 chunks, defaulting to English")
                    language = "en"
        
        async with aiohttp.ClientSession() as session:
            download_task = asyncio.create_task(self.download_single_file(session, urls[0], 0))
            results: List[ChunkResult] = []
            start_time_offset = 0.0

            for i, url in enumerate(urls):
                file_path = await download_task
                
                if i + 1 < len(urls):
                    download_task = asyncio.create_task(
                        self.download_single_file(session, urls[i+1], i+1)
                    )

                # Use provided chunk_duration parameter instead of probing file metadata
                # (FLAC metadata can be unreliable for split files)
                actual_duration = chunk_duration

                result = await asyncio.to_thread(
                    self.process_single_chunk,
                    i, file_path, start_time_offset, language, batch_size,
                    temperature, initial_prompt, vad_onset, vad_offset,
                    align_output, diarization, huggingface_access_token,
                    min_speakers, max_speakers, debug
                )
                results.append(result)

                try:
                    file_path.unlink()
                except:
                    pass

                if realtime_client and realtime_client.is_enabled:
                    processed_duration = start_time_offset + actual_duration
                    progress = None
                    if total_duration_seconds and total_duration_seconds > 0:
                        progress = min(processed_duration / total_duration_seconds, 1.0)
                    elif len(urls) > 0:
                        progress = min((i + 1) / len(urls), 1.0)

                    merged_state = self.merge_chunk_results(results, debug=False)
                    await realtime_client.emit_snapshot(
                        segments=merged_state["segments"]
                    )

                start_time_offset += actual_duration
            
            return results

    def detect_language_from_file(self, audio_file: PathlibPath, min_prob: float, max_tries: int,
                                  temperature: float, initial_prompt: str, vad_onset: float, vad_offset: float,
                                  batch_size: int, debug: bool) -> str:
        """Detect language from audio file with CUDA error handling."""
        if min_prob <= 0:
            return None
        
        audio_duration = get_audio_duration(audio_file)
        
        if debug:
            print(f"  Detecting language from {audio_duration/1000:.1f}s audio...")
        
        try:
            with torch.inference_mode():
                audio = whisperx.load_audio(str(audio_file))
                
                # Load model with error handling
                if self._asr_model_cached is None:
                    if debug:
                        print("  Loading WhisperX model...")
                    self._asr_model_cached = self._load_model_with_retry(None, temperature, initial_prompt, vad_onset, vad_offset, debug)
                
                # VAD is already disabled via vad_options=None in model loading
                result = self._asr_model_cached.transcribe(audio, batch_size=min(4, batch_size))
                detected_language = result.get("language", "en")
                
                if debug:
                    print(f"  ✓ Detected language: {detected_language}")
                
                return detected_language
                
        except Exception as e:
            print(f"  ✗ Language detection failed: {e}")
            print("  Falling back to English")
            return "en"

    def _load_model_with_retry(self, language: Optional[str], temperature: float, initial_prompt: str,
                               vad_onset: float, vad_offset: float, debug: bool, max_retries: int = 3):
        """Load WhisperX model with retry logic for CUDA errors."""
        asr_options = {
            "temperatures": [temperature],
            "initial_prompt": initial_prompt
        }
        
        vad_options = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset
        }
        
        # Use Silero VAD instead of Pyannote VAD (Silero is compatible with modern torch/pyannote)
        # Pyannote's old VAD checkpoint requires pyannote 0.0.1/torch 1.10 which causes segfaults
        if debug:
            print("  ℹ Using Silero VAD (Pyannote VAD incompatible with torch 2.8+)")
        
        # Ensure CUDA is initialized in this thread when using GPU
        if device == "cuda":
            cuda_ok = ensure_cuda_initialized()
            if not cuda_ok:
                raise RuntimeError(
                    "CUDA initialization failed in worker thread. "
                    "This worker requires CUDA; CPU fallback is not supported."
                )
        
        for attempt in range(max_retries):
            try:
                # Clear CUDA cache before loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                if debug and attempt > 0:
                    print(f"  Retry attempt {attempt + 1}/{max_retries} to load model...")
                
                # Use Silero VAD instead of Pyannote (Silero is compatible with modern torch/pyannote)
                model = whisperx.load_model(
                    whisper_arch,
                    device,
                    compute_type=compute_type,
                    language=language,
                    asr_options=asr_options,
                    vad_method="silero",  # Use Silero VAD instead of Pyannote
                    vad_options=vad_options
                )
                
                if debug and attempt > 0:
                    print(f"  ✓ Model loaded successfully on retry {attempt + 1}")
                
                return model
                
            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "out of memory" in error_msg:
                    print(f"  ⚠ CUDA error on attempt {attempt + 1}: {error_msg}")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(1)
                    
                    if attempt == max_retries - 1:
                        print("  ✗ Failed to load model after all retries")
                        raise
                else:
                    raise
        
        raise RuntimeError("Failed to load model after all retries")

    def process_single_chunk(self, chunk_index: int, audio_file: PathlibPath, start_time_offset: float,
                            language: str, batch_size: int, temperature: float, initial_prompt: str,
                            vad_onset: float, vad_offset: float, align_output: bool, diarization: bool,
                            huggingface_access_token: str, min_speakers: int, max_speakers: int, debug: bool) -> ChunkResult:
        """Process a single audio chunk with CUDA error handling."""
        
        if debug:
            print(f"\nProcessing chunk {chunk_index + 1} (offset: {start_time_offset:.2f}s)...")
        
        # Initialize CUDA in this thread only once (CUDA contexts are thread-local)
        # Keep this lightweight and idempotent to avoid overhead across chunks
        import threading
        thread_local = threading.local()
        if device == "cuda" and not getattr(thread_local, "cuda_initialized", False):
            ensure_cuda_initialized()
            thread_local.cuda_initialized = True
        
        try:
            with torch.inference_mode():
                audio = whisperx.load_audio(str(audio_file))
                
                # Load or reuse cached model
                if self._asr_model_cached is None or self._cached_language != language:
                    if debug:
                        print(f"  Loading model for language: {language}")
                    self._asr_model_cached = self._load_model_with_retry(
                        language, temperature, initial_prompt, vad_onset, vad_offset, debug
                    )
                    self._cached_language = language
                
                # Transcribe with retry
                effective_batch_size = max(1, batch_size)
                result = self._transcribe_with_retry(audio, effective_batch_size, debug)
                detected_language = result.get("language", language or "en")
                
                # Alignment
                if align_output:
                    if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or \
                       detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                        result = align(audio, result, debug)
                    elif debug:
                        print(f"  ⚠ Alignment not available for language: {detected_language}")
                
                # Diarization (with model caching for performance)
                # Skip if no segments found (no speech detected)
                if diarization and result.get("segments") and len(result["segments"]) > 0:
                    result = diarize(self, audio, result, debug, huggingface_access_token, min_speakers, max_speakers)
                elif diarization and debug:
                    print(f"  ⚠ Skipping diarization - no speech segments found")
                
                # Adjust timestamps
                for segment in result["segments"]:
                    segment["start"] += start_time_offset
                    segment["end"] += start_time_offset
                    segment["chunk_index"] = chunk_index
                    
                    if "words" in segment:
                        for word in segment["words"]:
                            if "start" in word:
                                word["start"] += start_time_offset
                            if "end" in word:
                                word["end"] += start_time_offset
                
                if debug:
                    print(f"  ✓ Chunk {chunk_index + 1} completed ({len(result['segments'])} segments)")
                
                return ChunkResult(
                    chunk_index=chunk_index,
                    segments=result["segments"],
                    detected_language=detected_language,
                    start_time_offset=start_time_offset
                )
                
        except Exception as e:
            print(f"  ✗ Error processing chunk {chunk_index + 1}: {e}")
            # Clear cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._asr_model_cached = None
            self._cached_language = None
            raise

    def _transcribe_with_retry(self, audio, batch_size: int, debug: bool, max_retries: int = 2):
        """Transcribe audio with retry logic."""
        for attempt in range(max_retries):
            try:
                # VAD is already disabled via vad_options=None in model loading
                return self._asr_model_cached.transcribe(audio, batch_size=batch_size)
            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e):
                    if debug:
                        print(f"  ⚠ CUDA error during transcription, retry {attempt + 1}/{max_retries}")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(0.5)
                    
                    # Reduce batch size
                    batch_size = max(1, batch_size // 2)
                    
                    if attempt == max_retries - 1:
                        # Reset model cache and reload
                        self._asr_model_cached = None
                        raise
                else:
                    raise
    
    def merge_chunk_results(self, chunk_results: List[ChunkResult], debug: bool) -> Dict[str, Any]:
        """Merge results from multiple chunks."""
        
        if not chunk_results:
            raise ValueError("No chunk results to merge")
        
        merged_language = chunk_results[0].detected_language
        all_segments = []
        text_parts: List[str] = []
        
        for chunk_result in chunk_results:
            all_segments.extend(chunk_result.segments)
            for segment in chunk_result.segments:
                text = segment.get("text")
                if isinstance(text, str):
                    stripped = text.strip()
                    if stripped:
                        text_parts.append(stripped)
        
        all_segments.sort(key=lambda x: (x["start"], x.get("chunk_index", 0)))
        transcription_text = " ".join(text_parts) if text_parts else ""
        
        if debug:
            print(f"\n✓ Merged {len(all_segments)} segments from {len(chunk_results)} chunks")
            if all_segments:
                print(f"  Duration: {all_segments[0]['start']:.2f}s - {all_segments[-1]['end']:.2f}s")
        
        return {
            "segments": all_segments,
            "language": merged_language,
            "text": transcription_text
        }


def get_audio_duration(file_path):
    """Get audio duration in milliseconds."""
    probe = ffmpeg.probe(str(file_path))
    stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    return float(stream['duration']) * 1000


def align(audio, result, debug):
    """Align audio with word-level timestamps."""
    start_time = time.time()
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)
    
    if debug:
        print(f"  Alignment completed in {time.time() - start_time:.2f}s")
    
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    
    return result


def diarize(predictor_instance, audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    """Perform speaker diarization with model caching for better performance."""
    start_time = time.time()
    
    # Load diarization model only once and cache it (HUGE performance improvement!)
    if predictor_instance._diarize_model_cached is None:
        if debug:
            print(f"  Loading diarization model (first time only)...")
        predictor_instance._diarize_model_cached = DiarizationPipeline(
            use_auth_token=huggingface_access_token, 
            device=device
        )
    
    diarize_segments = predictor_instance._diarize_model_cached(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    if debug:
        print(f"  Diarization completed in {time.time() - start_time:.2f}s")
    
    # Clean up memory but keep the model cached
    gc.collect()
    torch.cuda.empty_cache()
    
    return result


if __name__ == "__main__":
    import runpod
    
    # Initialize predictor once at module level for reuse
    predictor = Predictor()
    predictor.setup()
    
    def handler(job):
        """RunPod serverless handler."""
        try:
            job_input = dict(job.get("input", {}) or {})
            job_input.pop("realtime_updates_url", None)
            result = predictor.predict(**job_input)
            return result
        except Exception as e:
            print(f"Handler error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    runpod.serverless.start({"handler": handler})
