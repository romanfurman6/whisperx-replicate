from cog import BasePredictor, Input, Path, BaseModel
from typing import Any, List, Dict, Optional, Tuple
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
import requests
import concurrent.futures
from pathlib import Path as PathlibPath
import urllib.parse

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"
whisper_arch = "large-v3"


class ChunkResult(BaseModel):
    chunk_index: int
    segments: Any
    detected_language: str
    start_time_offset: float  # in seconds


class Output(BaseModel):
    segments: Any
    detected_language: str
    total_chunks: int
    processing_time: float


class Predictor(BasePredictor):
    def setup(self):
        source_folder = './models/vad'
        destination_folder = '../root/.cache/torch'
        file_name = 'whisperx-vad-segmentation.bin'

        os.makedirs(destination_folder, exist_ok=True)

        source_file_path = os.path.join(source_folder, file_name)
        if os.path.exists(source_file_path):
            destination_file_path = os.path.join(destination_folder, file_name)

            if not os.path.exists(destination_file_path):
                shutil.copy(source_file_path, destination_folder)

    def predict(
            self,
            audio_urls: List[str] = Input(description="Array of audio file URLs (chunks of one video in temporal order)"),
            chunk_duration_seconds: Optional[float] = Input(
                description="Duration of each chunk in seconds (used for timestamp calculation, auto-detected if None)",
                default=None),
            language: str = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None),
            language_detection_min_prob: float = Input(
                description="If language is not specified, then the language will be detected recursively on different "
                            "parts of the file until it reaches the given probability",
                default=0.7
            ),
            language_detection_max_tries: int = Input(
                description="If language is not specified, then the language will be detected following the logic of "
                            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
                            "retries is reached, the most probable language is kept.",
                default=5
            ),
            initial_prompt: str = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=32),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0.2),
            vad_onset: float = Input(
                description="VAD onset",
                default=0.500),
            vad_offset: float = Input(
                description="VAD offset",
                default=0.363),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=False),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=False),
            huggingface_access_token: str = Input(
                description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
                            "the user agreement for the models specified in the README.",
                default=None),
            min_speakers: int = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: int = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            debug: bool = Input(
                description="Print out compute/inference times and memory usage information",
                default=True)
    ) -> Output:
        start_processing_time = time.time()
        
        if not audio_urls:
            raise ValueError("audio_urls cannot be empty")
        
        print(f"Processing {len(audio_urls)} audio chunks...")
        
        # Download all audio files first
        downloaded_files = asyncio.run(self.download_audio_files(audio_urls))
        
        try:
            # Detect language from the first chunk if not provided
            if language is None:
                print("Detecting language from first chunk...")
                first_file = downloaded_files[0]
                language = self.detect_language_from_file(
                    first_file, language_detection_min_prob, language_detection_max_tries,
                    temperature, initial_prompt, vad_onset, vad_offset
                )
                print(f"Detected language: {language}")
            
            # Process chunks in parallel
            chunk_results = self.process_chunks_parallel(
                downloaded_files, language, chunk_duration_seconds,
                batch_size, temperature, initial_prompt, vad_onset, vad_offset,
                align_output, diarization, huggingface_access_token,
                min_speakers, max_speakers, debug
            )
            
            # Merge results
            merged_result = self.merge_chunk_results(chunk_results, debug)
            
            processing_time = time.time() - start_processing_time
            
            if debug:
                print(f"Total processing time: {processing_time:.2f} seconds")
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
            
            return Output(
                segments=merged_result["segments"],
                detected_language=merged_result["language"],
                total_chunks=len(audio_urls),
                processing_time=processing_time
            )
        
        finally:
            # Clean up downloaded files
            for file_path in downloaded_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {file_path}: {e}")

    async def download_audio_files(self, urls: List[str]) -> List[PathlibPath]:
        """Download audio files from URLs asynchronously."""
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
            print(f"Downloading chunk {index + 1}: {url}")
            
            # Parse URL to get file extension
            parsed_url = urllib.parse.urlparse(url)
            file_extension = PathlibPath(parsed_url.path).suffix or '.mp3'
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_path = PathlibPath(temp_file.name)
            temp_file.close()
            
            # Download file
            async with session.get(url) as response:
                response.raise_for_status()
                
                async with aiofiles.open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
            
            print(f"Downloaded chunk {index + 1} to {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"Error downloading chunk {index + 1} from {url}: {e}")
            raise
    
    def detect_language_from_file(self, audio_file: PathlibPath, min_prob: float, max_tries: int,
                                  temperature: float, initial_prompt: str, vad_onset: float, vad_offset: float) -> str:
        """Detect language from a single audio file."""
        if min_prob <= 0:
            return None
        
        asr_options = {
            "temperatures": [temperature],
            "initial_prompt": initial_prompt
        }
        
        vad_options = {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset
        }
        
        audio_duration = get_audio_duration(audio_file)
        
        if audio_duration > 30000:  # 30 seconds in ms
            segments_duration_ms = 30000
            max_tries = min(max_tries, math.floor(audio_duration / segments_duration_ms))
            segments_starts = distribute_segments_equally(audio_duration, segments_duration_ms, max_tries)
            
            detected_language_details = detect_language(audio_file, segments_starts, min_prob,
                                                        max_tries, asr_options, vad_options)
            return detected_language_details["language"]
        else:
            # For short files, use simple detection
            with torch.inference_mode():
                model = whisperx.load_model(whisper_arch, device, compute_type=compute_type,
                                            asr_options=asr_options, vad_options=vad_options)
                audio = whisperx.load_audio(audio_file)
                result = model.transcribe(audio, batch_size=16)
                detected_language = result["language"]
                
                gc.collect()
                torch.cuda.empty_cache()
                del model
                
                return detected_language
    
    def process_chunks_parallel(self, audio_files: List[PathlibPath], language: str, chunk_duration: Optional[float],
                               batch_size: int, temperature: float, initial_prompt: str, vad_onset: float, vad_offset: float,
                               align_output: bool, diarization: bool, huggingface_access_token: str,
                               min_speakers: int, max_speakers: int, debug: bool) -> List[ChunkResult]:
        """Process audio chunks in parallel using ThreadPoolExecutor."""
        
        # Calculate chunk durations if not provided
        if chunk_duration is None:
            chunk_durations = []
            for audio_file in audio_files:
                duration_ms = get_audio_duration(audio_file)
                chunk_durations.append(duration_ms / 1000.0)  # Convert to seconds
        else:
            chunk_durations = [chunk_duration] * len(audio_files)
        
        # Calculate start time offsets
        start_time_offsets = [0.0]
        for i in range(1, len(chunk_durations)):
            start_time_offsets.append(start_time_offsets[-1] + chunk_durations[i-1])
        
        # Process chunks in parallel using threads (since WhisperX uses CUDA)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(audio_files))) as executor:
            futures = []
            
            for i, (audio_file, start_offset) in enumerate(zip(audio_files, start_time_offsets)):
                future = executor.submit(
                    self.process_single_chunk,
                    i, audio_file, start_offset, language, batch_size, temperature, initial_prompt,
                    vad_onset, vad_offset, align_output, diarization, huggingface_access_token,
                    min_speakers, max_speakers, debug
                )
                futures.append(future)
            
            # Collect results
            chunk_results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                chunk_results.append(result)
        
        # Sort results by chunk index to maintain order
        chunk_results.sort(key=lambda x: x.chunk_index)
        
        return chunk_results
    
    def process_single_chunk(self, chunk_index: int, audio_file: PathlibPath, start_time_offset: float,
                            language: str, batch_size: int, temperature: float, initial_prompt: str,
                            vad_onset: float, vad_offset: float, align_output: bool, diarization: bool,
                            huggingface_access_token: str, min_speakers: int, max_speakers: int, debug: bool) -> ChunkResult:
        """Process a single audio chunk."""
        
        if debug:
            print(f"Processing chunk {chunk_index + 1} with offset {start_time_offset:.2f}s")
        
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt
            }
            
            vad_options = {
                "vad_onset": vad_onset,
                "vad_offset": vad_offset
            }
            
            # Load model
            model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, language=language,
                                        asr_options=asr_options, vad_options=vad_options)
            
            # Load and transcribe audio
            audio = whisperx.load_audio(audio_file)
            result = model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]
            
            gc.collect()
            torch.cuda.empty_cache()
            del model
            
            # Apply alignment if requested
            if align_output:
                if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                    result = align(audio, result, debug)
                else:
                    if debug:
                        print(f"Cannot align output for chunk {chunk_index + 1} as language {detected_language} is not supported for alignment")
            
            # Apply diarization if requested
            if diarization:
                result = diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers)
            
            # Adjust timestamps by adding the start time offset
            for segment in result["segments"]:
                segment["start"] += start_time_offset
                segment["end"] += start_time_offset
                
                # Adjust word-level timestamps if they exist
                if "words" in segment:
                    for word in segment["words"]:
                        if "start" in word:
                            word["start"] += start_time_offset
                        if "end" in word:
                            word["end"] += start_time_offset
            
            if debug:
                print(f"Completed chunk {chunk_index + 1}")
            
            return ChunkResult(
                chunk_index=chunk_index,
                segments=result["segments"],
                detected_language=detected_language,
                start_time_offset=start_time_offset
            )
    
    def merge_chunk_results(self, chunk_results: List[ChunkResult], debug: bool) -> Dict[str, Any]:
        """Merge results from multiple chunks into a single result."""
        
        if not chunk_results:
            raise ValueError("No chunk results to merge")
        
        # Use the language from the first chunk (they should all be the same)
        merged_language = chunk_results[0].detected_language
        
        # Collect all segments
        all_segments = []
        for chunk_result in chunk_results:
            all_segments.extend(chunk_result.segments)
        
        # Sort segments by start time to ensure proper order
        all_segments.sort(key=lambda x: x["start"])
        
        if debug:
            print(f"Merged {len(all_segments)} segments from {len(chunk_results)} chunks")
            total_duration = all_segments[-1]["end"] if all_segments else 0
            print(f"Total transcription duration: {total_duration:.2f} seconds")
        
        return {
            "segments": all_segments,
            "language": merged_language
        }


def get_audio_duration(file_path):
    probe = ffmpeg.probe(file_path)
    stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    return float(stream['duration']) * 1000


def detect_language(full_audio_file_path, segments_starts, language_detection_min_prob,
                    language_detection_max_tries, asr_options, vad_options, iteration=1):
    model = whisperx.load_model(whisper_arch, device, compute_type=compute_type, asr_options=asr_options,
                                vad_options=vad_options)

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(full_audio_file_path, start_ms, 30000)

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                  n_mels=model_n_mels if model_n_mels is not None else 80,
                                  padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    print(f"Iteration {iteration} - Detected language: {language} ({language_probability:.2f})")

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration
    }

    if language_probability >= language_detection_min_prob or iteration >= language_detection_max_tries:
        return detected_language

    next_iteration_detected_language = detect_language(full_audio_file_path, segments_starts,
                                                       language_detection_min_prob, language_detection_max_tries,
                                                       asr_options, vad_options, iteration + 1)

    if next_iteration_detected_language["probability"] > detected_language["probability"]:
        return next_iteration_detected_language

    return detected_language


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    input_file_path = Path(input_file_path) if not isinstance(input_file_path, Path) else input_file_path
    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)

        print(f"Extracting from {input_file_path.name} to {temp_file.name}")

        try:
            (
                ffmpeg
                .input(input_file_path, ss=start_time_ms/1000)
                .output(temp_file.name, t=duration_ms/1000)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print("ffmpeg error occurred: ", e.stderr.decode('utf-8'))
            raise e

    return temp_file_path


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to align output: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def diarize(audio, result, debug, huggingface_access_token, min_speakers, max_speakers):
    start_time = time.time_ns() / 1e6

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=huggingface_access_token, device=device)
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result
