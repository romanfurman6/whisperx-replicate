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
            audio_urls: List[str] = Input(
                description="Array of public audio urls to process"
            ),
            total_duration_seconds: float = Input(
                description="Total duration of the complete audio in seconds"
            ),
            chunk_size_seconds: float = Input(
                description="Duration of each chunk in seconds (used for timestamp calculation). Latest chunk can be shorter, it will be calculated based on the total duration and the number of chunks."
            ),
            language: Optional[str] = Input(
                description="ISO code of the language spoken in the audio, specify None to perform language detection",
                default=None
            ),
            language_detection_min_prob: float = Input(
                description="Minimum probability for recursive language detection",
                default=0.7
            ),
            language_detection_max_tries: int = Input(
                description="Maximum retries for recursive language detection",
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
        
        # Calculate chunk metadata based on total duration and chunk size
        num_chunks = len(audio_urls)
        expected_chunk_duration = total_duration_seconds / num_chunks
        
        if debug:
            print(f"Processing {num_chunks} audio URLs")
            print(f"Total duration: {total_duration_seconds:.2f} seconds")
            print(f"Expected chunk duration: {expected_chunk_duration:.2f} seconds")
            print(f"Chunk size parameter: {chunk_size_seconds:.2f} seconds")
        
        # If language not provided, download and detect from first file
        if language is None:
            print("Downloading first audio file for language detection...")
            first_file = asyncio.run(self.download_audio_files([audio_urls[0]]))[0]
            language = self.detect_language_from_file(
                first_file, language_detection_min_prob, language_detection_max_tries,
                temperature, initial_prompt, vad_onset, vad_offset
            )
            if language:
                print(f"Detected language: {language}")
            else:
                print("Could not confidently detect language from first file – falling back to per-chunk detection.")
            try:
                first_file.unlink()
            except Exception:
                pass
        
        # Process chunks sequentially with download pipeline
        chunk_results = asyncio.run(self.download_and_process_pipeline(
            audio_urls, language, chunk_size_seconds, batch_size,
            temperature, initial_prompt, vad_onset, vad_offset,
            align_output, diarization, huggingface_access_token,
            min_speakers, max_speakers, debug
        ))

        # Merge and return
        merged = self.merge_chunk_results(chunk_results, debug)
        processing_time = time.time() - start_processing_time
        if debug:
            print(f"Total processing time: {processing_time:.2f} seconds")
        return Output(
            segments=merged["segments"],
            detected_language=merged["language"],
            total_chunks=len(audio_urls),
            processing_time=processing_time
        )

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
    
    async def download_and_process_pipeline(self, urls: List[str], language: str, chunk_duration: float,
                                           batch_size: int, temperature: float, initial_prompt: str,
                                           vad_onset: float, vad_offset: float,
                                           align_output: bool, diarization: bool,
                                           huggingface_access_token: str, min_speakers: int, max_speakers: int,
                                           debug: bool) -> List[ChunkResult]:
        """
        Download next chunk while processing current one sequentially, keeping downloads active during GPU processing.
        """
        async with aiohttp.ClientSession() as session:
            # start fetching first chunk
            download_task = asyncio.create_task(self.download_single_file(session, urls[0], 0))
            results: List[ChunkResult] = []
            start_time_offset = 0.0

            for i, url in enumerate(urls):
                # wait for the current chunk to finish downloading
                file_path = await download_task
                # schedule next download for the following chunk
                if i + 1 < len(urls):
                    download_task = asyncio.create_task(
                        self.download_single_file(session, urls[i+1], i+1)
                    )

                # compute actual duration of the downloaded chunk (seconds)
                actual_duration = get_audio_duration(file_path) / 1000.0

                # process the chunk in a separate thread so downloads continue concurrently
                result = await asyncio.to_thread(
                    self.process_single_chunk,
                    i, file_path, start_time_offset, language, batch_size,
                    temperature, initial_prompt, vad_onset, vad_offset,
                    align_output, diarization, huggingface_access_token,
                    min_speakers, max_speakers, debug
                )
                results.append(result)

                # cleanup temp file
                try:
                    file_path.unlink()
                except Exception:
                    pass

                # update offset using the actual duration of this chunk
                start_time_offset += actual_duration
            return results
    

    
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
                # Add chunk index for debugging and sorting
                segment["chunk_index"] = chunk_index
                
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
        
        # Sort segments by start time to ensure proper chronological order
        # Use both start time and chunk index as tiebreaker for segments with same start time
        all_segments.sort(key=lambda x: (x["start"], x.get("chunk_index", 0)))
        
        # Validate that segments are in chronological order
        if len(all_segments) > 1:
            for i in range(1, len(all_segments)):
                if all_segments[i]["start"] < all_segments[i-1]["start"]:
                    if debug:
                        print(f"Warning: Segment {i} starts at {all_segments[i]['start']:.2f}s but previous segment ends at {all_segments[i-1]['end']:.2f}s")
        
        if debug:
            print(f"Merged {len(all_segments)} segments from {len(chunk_results)} chunks")
            if all_segments:
                total_duration = all_segments[-1]["end"]
                print(f"Total transcription duration: {total_duration:.2f} seconds")
                print(f"First segment: {all_segments[0]['start']:.2f}s - {all_segments[0]['end']:.2f}s")
                print(f"Last segment: {all_segments[-1]['start']:.2f}s - {all_segments[-1]['end']:.2f}s")
        
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
