"""Modal deployment of the WhisperX worker, with a RunPod-style queue API.

Runs the same Predictor from predict.py as the RunPod image, so inputs and outputs are identical.

Deploy from the repo root (deploys, then pre-builds memory snapshots on a few containers):
    modal_app/deploy.sh

HTTP API (Modal proxy auth: send Modal-Key / Modal-Secret headers from a proxy auth token):
    POST /run            {"input": {...}, "webhook"?: "https://..."}  -> {"id": "fc-...", "status": "IN_QUEUE"}
    POST /runsync        same body; waits up to 90 s, else returns {"id", "status": "IN_PROGRESS"}
    GET  /status/{id}                      -> {"id", "status", "output" | "error"}
    POST /cancel/{id}                      -> {"id", "status": "CANCELLED"}
status is IN_QUEUE, IN_PROGRESS (a worker picked it up), COMPLETED, FAILED, CANCELLED or TIMED_OUT.
With "webhook", the final /status body is POSTed there for every terminal status, signed with
WEBHOOK_SECRET from the Modal secret "audio-webhook" (see _send_webhook). Same contract as the
demucs and resemble-enhance apps.

Nothing deployment-specific lives in this public repo. Modal secret "whisperx" holds
SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY (realtime partial broadcasts; optional) and HF_TOKEN
(pyannote diarization; optional, falls back to huggingface_access_token from the job input).
"""

import time

import modal

WHISPER_ARCH = "deepdml/faster-whisper-large-v3-turbo-ct2"  # keep in sync with predict.py
# Fallback list, tried in order when the previous type has no free capacity, instead of queueing:
# L4 ~$0.80/h, A10 ~$1.10/h (both 24 GB), L40S ~$1.95/h (48 GB). Turbo fp16 needs ~4-6 GB.
GPU = ["L4", "A10", "L40S"]
SITE = "/usr/local/lib/python3.11/site-packages"

app = modal.App("whisperx")

# call id -> start time, written when a worker picks the job up, so /status can tell
# IN_QUEUE from IN_PROGRESS (Modal itself only knows "not finished yet").
started = modal.Dict.from_name("whisperx-started", create_if_missing=True)


def bake_models():
    """Download every model a default job needs into the image, so cold starts never download."""
    import nltk
    import torch
    import torchaudio
    from faster_whisper.utils import download_model

    download_model(WHISPER_ARCH)  # ASR (~1.6 GB) -> HF_HOME
    torch.hub.load("snakers4/silero-vad", model="silero_vad", onnx=False, trust_repo=True)  # VAD -> TORCH_HOME
    torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()  # English alignment -> TORCH_HOME
    nltk.download("punkt_tab", download_dir="/models/nltk")  # alignment sentence splitter
    # Other alignment languages and pyannote diarization download on first use per container.


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("torch==2.8.0", "torchaudio==2.8.0")
    .pip_install(
        "whisperx==3.8.6",
        "ffmpeg-python==0.2.0",
        "aiohttp>=3.9.0",
        "aiofiles>=23.2.1",
        "requests>=2.31.0",
        "hf_transfer>=0.1.3",
    )
    .env(
        {
            "HF_HOME": "/models/hf",
            "TORCH_HOME": "/models/torch",
            "NLTK_DATA": "/models/nltk",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # ctranslate2 needs the pip-installed cuBLAS/cuDNN that torch ships with.
            "LD_LIBRARY_PATH": f"{SITE}/nvidia/cublas/lib:{SITE}/nvidia/cudnn/lib",
            "PYTHONPATH": "/opt/worker",
        }
    )
    .run_function(bake_models)
    .add_local_file("predict.py", "/opt/worker/predict.py")
)


@app.cls(
    gpu=GPU,
    cpu=2.0,  # ffmpeg decode of each chunk + CPU-side VAD
    memory=8192,
    image=image,
    secrets=[
        modal.Secret.from_name("whisperx"),  # SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, HF_TOKEN
        modal.Secret.from_name("audio-webhook"),  # WEBHOOK_SECRET
    ],
    timeout=3600,  # long files: turbo does ~1 h of audio in a few minutes on L4
    scaledown_window=120,  # requests come in bursts; ~$0.017/min per idle L4 container
    # The workspace GPU limit is shared with demucs and resemble-enhance; above it, jobs queue.
    max_containers=5,
    enable_memory_snapshot=True,
)
class WhisperX:
    # Memory snapshot: torch/whisperx imports (~10-15 s) are captured once. No GPU in this phase.
    @modal.enter(snap=True)
    def load_cpu(self):
        t0 = time.perf_counter()
        import predict

        self.predict = predict
        self.predictor = predict.Predictor()
        print(f"imports done in {time.perf_counter() - t0:.2f}s")

    @modal.enter(snap=False)
    def load_gpu(self):
        t0 = time.perf_counter()
        self.predictor.setup()
        # Preload with the options any2text sends, so the first job doesn't pay the model load.
        self.predictor._get_model(temperature=0.2, initial_prompt=None, vad_onset=0.5, vad_offset=0.363, debug=True)
        print(f"model loaded on GPU in {time.perf_counter() - t0:.2f}s")

    @modal.method()
    def warmup(self, hold_seconds: float = 0) -> dict:
        """No-op used by warmup.py after a deploy to build memory snapshots before users do."""
        import os

        import torch

        time.sleep(hold_seconds)  # keeps this container busy so parallel calls land on others
        return {"task": os.environ.get("MODAL_TASK_ID"), "device": torch.cuda.get_device_name(0)}

    @modal.method()
    def transcribe(self, job_id: str, inp: dict, submitted_at: float) -> dict:
        import resource

        import torch

        started_at = time.time()
        started[modal.current_function_call_id()] = started_at
        torch.cuda.reset_peak_memory_stats()
        inp = dict(inp)
        inp.pop("realtime_updates_url", None)  # same as the RunPod handler
        print(f"job {job_id}: {len(inp.get('audio_urls') or [])} chunks, task_id={inp.get('task_id')}")
        out = self.predictor.predict(**inp)
        out["queue_seconds"] = round(started_at - submitted_at, 3)
        out["total_seconds"] = round(time.time() - started_at, 3)
        out["peak_ram_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() // 2**20
        return out


def _status_body(call_id: str, out) -> dict:
    """The /status response for a finished call; webhooks send the same body."""
    if isinstance(out, dict) and "error" in out:
        return {"id": call_id, "status": "FAILED", "error": out["error"]}
    return {"id": call_id, "status": "COMPLETED", "output": out}


def _error_status(call_id: str, e: Exception) -> dict:
    """Map an exception from FunctionCall.get to a RunPod-style terminal status."""
    if isinstance(e, modal.exception.FunctionTimeoutError):
        status = "TIMED_OUT"
    elif "cancelled" in str(e).lower():  # a cancelled call surfaces as RemoteError("Function call was cancelled ...")
        status = "CANCELLED"
    else:
        status = "FAILED"
    return {"id": call_id, "status": status, "error": f"{type(e).__name__}: {e}"}


def _send_webhook(url: str, body: dict) -> None:
    """POST body to url, signed with WEBHOOK_SECRET; 5 attempts over ~12 min, never raises.

    Headers: X-Webhook-Timestamp: <unix seconds>
             X-Webhook-Signature: v1=<hex HMAC-SHA256(secret, f"{timestamp}.{raw body}")>
    """
    import hashlib
    import hmac
    import json
    import os

    import requests

    raw = json.dumps(body, separators=(",", ":")).encode()
    for attempt, delay in enumerate((0, 5, 30, 120, 600)):
        time.sleep(delay)
        ts = str(int(time.time()))
        sig = hmac.new(os.environ["WEBHOOK_SECRET"].encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
        headers = {"Content-Type": "application/json", "X-Webhook-Timestamp": ts, "X-Webhook-Signature": f"v1={sig}"}
        try:
            r = requests.post(url, data=raw, headers=headers, timeout=30)
            if r.status_code < 300:
                return
            print(f"webhook attempt {attempt + 1}: HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"webhook attempt {attempt + 1}: {e}")
    print(f"webhook to {url} failed; the result is still available via /status")


web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]", "requests")


@app.function(image=web_image, secrets=[modal.Secret.from_name("audio-webhook")], timeout=6 * 3600)
@modal.concurrent(max_inputs=1000)
async def watch(call_id: str, webhook: str) -> None:
    """Wait for a job to reach ANY terminal state and deliver the webhook.

    Runs outside the GPU worker, so timeouts, crashes and cancellations still produce a
    webhook (TIMED_OUT / FAILED / CANCELLED), like RunPod's platform-side webhooks.
    """
    import asyncio

    try:
        body = _status_body(call_id, await modal.FunctionCall.from_id(call_id).get.aio())
    except Exception as e:
        body = _error_status(call_id, e)
    await asyncio.to_thread(_send_webhook, webhook, body)


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)  # requests only enqueue/poll/wait, so one container serves many
@modal.asgi_app(requires_proxy_auth=True)
def api():
    import uuid

    from fastapi import FastAPI, HTTPException

    web = FastAPI()

    async def spawn(body: dict):
        inp = body.get("input")
        if not isinstance(inp, dict):
            raise HTTPException(400, "body must be {\"input\": {...}, \"webhook\"?: \"https://...\"}")
        if not isinstance(inp.get("audio_urls"), list) or not inp["audio_urls"]:
            raise HTTPException(400, "input.audio_urls must be a non-empty list")
        webhook = body.get("webhook")
        if webhook is not None and not (isinstance(webhook, str) and webhook.startswith("https://")):
            raise HTTPException(400, "webhook must be an https:// URL")
        call = await WhisperX().transcribe.spawn.aio(uuid.uuid4().hex, inp, time.time())
        if webhook:
            await watch.spawn.aio(call.object_id, webhook)
        return call

    @web.post("/run")
    async def run(body: dict):
        call = await spawn(body)
        return {"id": call.object_id, "status": "IN_QUEUE"}

    @web.post("/runsync")
    async def runsync(body: dict):
        """Waits up to 90 s; if the job isn't done by then, returns IN_PROGRESS and the id to poll."""
        call = await spawn(body)
        try:
            out = await call.get.aio(timeout=90)
        except TimeoutError:  # builtin: not finished within 90 s (a job timeout is FunctionTimeoutError)
            running = await started.contains.aio(call.object_id)
            return {"id": call.object_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except Exception as e:
            return _error_status(call.object_id, e)
        return _status_body(call.object_id, out)

    def check_id(call_id: str) -> None:
        if not call_id.startswith("fc-"):  # Modal raises ExecutionError, not NotFound, for these
            raise HTTPException(404, "job not found")

    @web.get("/status/{call_id}")
    async def status(call_id: str):
        check_id(call_id)
        try:
            out = await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
        except TimeoutError:  # builtin: not finished yet
            running = await started.contains.aio(call_id)
            return {"id": call_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        except Exception as e:  # the call itself raised (crash, timeout, cancel)
            return _error_status(call_id, e)
        return _status_body(call_id, out)

    @web.post("/cancel/{call_id}")
    async def cancel(call_id: str):
        check_id(call_id)
        try:
            await modal.FunctionCall.from_id(call_id).cancel.aio()
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        return {"id": call_id, "status": "CANCELLED"}

    return web
