from cog import BasePredictor, Input, BaseModel
from typing import Any, List, Dict, Optional
from whisperx.audio import N_SAMPLES, log_mel_spectrogram

import asyncio
import aiohttp
import aiofiles
import gc
import math
import os
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
whisper_arch = "large-v3"


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


class ChunkResult(BaseModel):
    chunk_index: int
    segments: Any
    detected_language: str
    start_time_offset: float


class Output(BaseModel):
    segments: Any
    detected_language: str
    total_chunks: int
    processing_time: float


class Predictor(BasePredictor):
    def __init__(self):
        super().__init__()
        self._asr_model_cached = None
        self._cached_language = None
        self._setup_complete = False

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
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, use a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(coro)

    def predict(
            self,
            audio_urls: List[str] = Input(
                description="Array of public audio urls to process"
            ),
            total_duration_seconds: float = Input(
                description="Total duration of the complete audio in seconds"
            ),
            chunk_size_seconds: float = Input(
                description="Duration of each chunk in seconds"
            ),
            language: Optional[str] = Input(
                description="ISO code of the language spoken in the audio",
                default=None
            ),
            language_detection_min_prob: float = Input(
                description="Minimum probability for language detection",
                default=0.7
            ),
            language_detection_max_tries: int = Input(
                description="Maximum retries for language detection",
                default=5
            ),
            initial_prompt: Optional[str] = Input(
                description="Optional text prompt for the first window",
                default=None
            ),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=32
            ),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0.2
            ),
            vad_onset: float = Input(
                description="VAD onset threshold",
                default=0.500
            ),
            vad_offset: float = Input(
                description="VAD offset threshold",
                default=0.363
            ),
            align_output: bool = Input(
                description="Whether to align output for word-level timestamps",
                default=False
            ),
            diarization: bool = Input(
                description="Whether to perform diarization",
                default=False
            ),
            huggingface_access_token: Optional[str] = Input(
                description="HuggingFace token for diarization",
                default=None
            ),
            min_speakers: Optional[int] = Input(
                description="Minimum number of speakers if diarization is activated",
                default=None
            ),
            max_speakers: Optional[int] = Input(
                description="Maximum number of speakers if diarization is activated",
                default=None
            ),
            debug: bool = Input(
                description="Print debug information",
                default=True
            )
    ) -> Output:
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
        
        # Process chunks
        try:
            chunk_results = self._run_coro(self.download_and_process_pipeline(
                audio_urls, language, chunk_size_seconds, batch_size,
                temperature, initial_prompt, vad_onset, vad_offset,
                align_output, diarization, huggingface_access_token,
                min_speakers, max_speakers, debug, language_detection_min_prob,
                language_detection_max_tries
            ))
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        # Merge results
        merged = self.merge_chunk_results(chunk_results, debug)
        processing_time = time.time() - start_processing_time
        
        if debug:
            print(f"\n{'='*50}")
            print(f"✓ Processing completed in {processing_time:.2f}s")
            print(f"{'='*50}\n")
        
        return Output(
            segments=merged["segments"],
            detected_language=merged["language"],
            total_chunks=len(audio_urls),
            processing_time=processing_time
        )

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
                                           language_detection_max_tries: int) -> List[ChunkResult]:
        """Download and process chunks with pipelined execution."""
        
        # If no language specified, detect from first chunk
        if language is None:
            print("Downloading first chunk for language detection...")
            first_file = await self.download_single_file(
                aiohttp.ClientSession(), urls[0], 0
            )
            language = self.detect_language_from_file(
                first_file, language_detection_min_prob, language_detection_max_tries,
                temperature, initial_prompt, vad_onset, vad_offset, batch_size, debug
            )
            try:
                first_file.unlink()
            except:
                pass
            print(f"✓ Detected language: {language}")
        
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

                actual_duration = get_audio_duration(file_path) / 1000.0

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
        
        for attempt in range(max_retries):
            try:
                # Clear CUDA cache before loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                if debug and attempt > 0:
                    print(f"  Retry attempt {attempt + 1}/{max_retries} to load model...")
                
                model = whisperx.load_model(
                    whisper_arch,
                    device,
                    compute_type=compute_type,
                    language=language,
                    asr_options=asr_options,
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
                
                # Diarization
                if diarization:
                    result = diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)
                
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
        
        for chunk_result in chunk_results:
            all_segments.extend(chunk_result.segments)
        
        all_segments.sort(key=lambda x: (x["start"], x.get("chunk_index", 0)))
        
        if debug:
            print(f"\n✓ Merged {len(all_segments)} segments from {len(chunk_results)} chunks")
            if all_segments:
                print(f"  Duration: {all_segments[0]['start']:.2f}s - {all_segments[-1]['end']:.2f}s")
        
        return {
            "segments": all_segments,
            "language": merged_language
        }


def get_audio_duration(file_path):
    """Get audio duration in milliseconds."""
    probe = ffmpeg.probe(str(file_path))
    stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
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


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    """Perform speaker diarization."""
    start_time = time.time()
    
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=huggingface_access_token, device=device)
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    if debug:
        print(f"  Diarization completed in {time.time() - start_time:.2f}s")
    
    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model
    
    return result


if __name__ == "__main__":
    import runpod
    
    def handler(job):
        """RunPod serverless handler."""
        try:
            predictor = Predictor()
            predictor.setup()
            result = predictor.predict(**job["input"])
            return result.dict()
        except Exception as e:
            print(f"Handler error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    runpod.serverless.start({"handler": handler})
