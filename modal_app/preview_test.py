"""Ephemeral test of the worker (not deployed): `modal run modal_app/preview_test.py`.

Runs the current predict.py on a GPU in a temporary app, so the deployed `whisperx` app and
its in-flight jobs are untouched. Checks the default path and the max_seconds preview path.
"""
import time

import modal

from app import WhisperX, app  # noqa: F401  (same image/GPU config as the deployed app)

JFK = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
BASE = {
    "language_detection_min_prob": 0.7, "language_detection_max_tries": 5, "batch_size": 32,
    "temperature": 0.2, "vad_onset": 0.5, "vad_offset": 0.363, "align_output": False,
    "diarization": False, "debug": True,
}


@app.local_entrypoint()
def main(long_url: str = "", seconds: float = 30):
    w = WhisperX()

    def run(name, inp):
        t0 = time.time()
        try:
            out = w.transcribe.remote("test-" + name, {**BASE, **inp}, time.time())
            segs = out.get("segments") or []
            end = max((s.get("end", 0) for s in segs), default=0)
            text = " ".join(s.get("text", "").strip() for s in segs)
            print(f"[{name}] OK {time.time() - t0:.1f}s lang={out.get('detected_language')} "
                  f"segments={len(segs)} last_end={end:.1f}s text={text[:120]!r}")
        except Exception as e:
            print(f"[{name}] ERROR {time.time() - t0:.1f}s {e}")

    # 1. Default path, exactly what async/backfill send: must be unchanged.
    run("default-jfk", {"audio_urls": [JFK], "total_duration_seconds": 11, "chunk_size_seconds": 30})
    # 2. Preview on a short file (max_seconds longer than the file).
    run("preview-jfk-60", {"audio_urls": [JFK], "total_duration_seconds": 0, "chunk_size_seconds": 30, "max_seconds": 60})
    # 3. Preview cut shorter than the file.
    run("preview-jfk-5", {"audio_urls": [JFK], "total_duration_seconds": 0, "chunk_size_seconds": 30, "max_seconds": 5})
    # 4. Preview of a long file: should only read the start.
    if long_url:
        run(f"preview-long-{seconds:.0f}", {"audio_urls": [long_url], "total_duration_seconds": 0,
                                             "chunk_size_seconds": 30, "max_seconds": seconds})
    # 5. Errors: two URLs, non-http URL.
    run("err-two-urls", {"audio_urls": [JFK, JFK], "total_duration_seconds": 0, "chunk_size_seconds": 30, "max_seconds": 10})
    run("err-file-url", {"audio_urls": ["file:///etc/passwd"], "total_duration_seconds": 0, "chunk_size_seconds": 30, "max_seconds": 10})
