# Capten Apex Worker

[![Runpod](https://api.runpod.io/badge/jaseemuddinn/capten-apex-worker)](https://console.runpod.io/hub/jaseemuddinn/capten-apex-worker)

RunPod Serverless worker for [Oriserve/Whisper-Hindi2Hinglish-Apex](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex) + MMS forced alignment.

## If Hub tests hang on “Waiting for container startup”

Hub’s **GPU test pods** have been unreliable with this ~5–6 GB CUDA image (build succeeds; test never runs the handler). `cu128-v15` changes Hub tests to a **CPU** health check (`FORCE_CPU=true`, no torch init) and starts via `python -u /handler.py`.

**Recommended: skip Hub for Capten production — deploy a manual Serverless endpoint** from the registry image Hub already built:

1. RunPod Console → **Serverless** → **New Endpoint** → **Import from Docker registry**
2. Image (from a successful Hub build log), e.g.  
   `registry.runpod.net/jaseemuddinn-capten-apex-worker-main-runpod-dockerfile:<tag>`
3. GPU: 16 GB+ · Container disk: **50 GB** · `workersMin` 0–1
4. Env:
   - `MODEL_ID=Oriserve/Whisper-Hindi2Hinglish-Apex`
   - `ALIGN_MODEL=MahmoudAshraf/mms-300m-1130-forced-aligner`
   - `ALIGN_LANGUAGE=hin`
   - `ENABLE_ALIGNMENT=true`
   - `RUNPOD_INIT_TIMEOUT=900`
5. Put the endpoint id in Capten `.env` as `RUNPOD_ENDPOINT_ID=...`
6. Smoke test: `{ "input": { "health_check": true } }` → expect `"build": "cu128-v15"`

## Hub release (optional)

1. Push `main`, create GitHub **Release** `v1.0.xx`
2. Rebuild on Hub — tests should use CPU flavor `cpu3c`, not RTX 4090
3. Logs should show `[startup] capten apex worker cu128-v15` within ~1–2 minutes of pod deploy

## Input

```json
{ "input": { "audio_url": "https://example.com/audio.wav" } }
```

```json
{ "input": { "align_text": "hinglish transcript...", "audio_url": "..." } }
```

## Capten `.env`

```env
GPU_MODE=runpod
RUNPOD_API_KEY=rpa_...
RUNPOD_ENDPOINT_ID=your_endpoint_id
```
