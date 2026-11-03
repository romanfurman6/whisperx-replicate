# Modal deployment

Runs the same WhisperX worker as RunPod (`Predictor` in `predict.py`, same input/output) on Modal,
with the same RunPod-style queue API as the demucs and resemble-enhance Modal apps.

```bash
modal_app/deploy.sh     # modal deploy + pre-build memory snapshots (needs modal CLI and uv)
```

Keep this directory out of a folder named `modal` (it would shadow the `modal` package).

## Secrets (one-time)

This repo is public: nothing deployment-specific is committed. Everything comes from Modal secrets.

```bash
# Realtime partial transcripts (Supabase broadcast RPC) + pyannote token for diarization.
# Any of the keys may be left out: no SUPABASE_* = no realtime, no HF_TOKEN = use the job's token.
modal secret create whisperx SUPABASE_URL=https://<project>.supabase.co SUPABASE_SERVICE_ROLE_KEY=<key> HF_TOKEN=<hf token>
# Webhook signing; already exists if demucs is deployed in the same workspace.
modal secret create audio-webhook WEBHOOK_SECRET=<random>
```

## API

`modal deploy` prints the URL (`https://<workspace>--whisperx-api.modal.run`). It is protected by Modal
proxy auth: send `Modal-Key` / `Modal-Secret` headers from a proxy auth token (Modal dashboard →
Settings → Proxy Auth Tokens).

| request | response |
|---|---|
| `POST /run` `{"input": {...}, "webhook"?: "https://..."}` | `{"id": "fc-...", "status": "IN_QUEUE"}` |
| `POST /runsync` (same body, waits ≤ 90 s) | finished body, or `{"id", "status": "IN_PROGRESS"}` |
| `GET /status/{id}` | `{"id", "status", "output" \| "error"}` |
| `POST /cancel/{id}` | `{"id", "status": "CANCELLED"}` |

`status`: `IN_QUEUE`, `IN_PROGRESS`, `COMPLETED`, `FAILED`, `CANCELLED`, `TIMED_OUT`.
`input` is exactly the RunPod input (`audio_urls`, `total_duration_seconds`, `chunk_size_seconds`,
`language`, `align_output`, `diarization`, `task_id`, `realtime`, ... see `runpod_request.json`).
`output` is the RunPod output (`segments`, `detected_language`, `total_chunks`, `processing_time`) plus
`queue_seconds`, `total_seconds`, `peak_ram_mb`, `peak_gpu_mb`.

With `webhook`, the final `/status` body is POSTed there for every terminal status (also timeouts and
cancels), retried for ~12 min, with headers `X-Webhook-Timestamp` and
`X-Webhook-Signature: v1=<hex HMAC-SHA256(WEBHOOK_SECRET, "<timestamp>.<raw body>")>`.

Quick test:

```bash
curl -s -X POST "$URL/run" -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
  -H 'Content-Type: application/json' -d @runpod_request.json
curl -s "$URL/status/fc-..." -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET"
```

## Setup

- GPU L4, falling back to A10 and then L40S when none is free; 2 CPU, 8 GB RAM, timeout 60 min,
  scale to zero after 2 min idle, max 5 containers (the workspace GPU limit is shared with demucs and
  resemble-enhance; above it jobs wait in the queue).
- whisperx pinned to 3.8.6 (torch 2.8). Baked into the image: the ASR model
  (`deepdml/faster-whisper-large-v3-turbo-ct2`), Silero VAD, the English alignment model and NLTK
  punkt. Other alignment languages and pyannote diarization download on first use per container.
- Memory snapshot covers the torch/whisperx imports; the GPU phase loads the ASR model once per
  container. The model is no longer reloaded per language: the language is passed per `transcribe` call.
