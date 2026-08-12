# Capten Apex Worker

[![Runpod](https://api.runpod.io/badge/jaseemuddinn/capten-apex-worker)](https://console.runpod.io/hub/jaseemuddinn/capten-apex-worker)

RunPod Serverless worker for [Oriserve/Whisper-Hindi2Hinglish-Apex](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex) + MMS forced alignment.

## If Hub tests hang on “Waiting for container startup”

Hub **always uses a GPU test pod** for GPU listings (`runsOn: GPU`). `cpuFlavor` in `tests.json` is ignored — your Aug 8 log still says RTX 4090.

`cu128-v16` replaces the NVIDIA entrypoint with `/start.sh` (prints immediately, then `exec python -u /handler.py`) and prints before `import runpod`. Empty `ENTRYPOINT []` was breaking Hub’s start command, which is why tests never logged a single worker line.

You don’t need Docker on your laptop. Use **Deploy from a GitHub repository** (not Hub, not `registry.runpod.net`).

1. Push this repo to GitHub (`main`).
2. RunPod → **Serverless** → **New Endpoint** → **Deploy from a GitHub repository**.
3. Repo: `jaseemuddinn/capten-apex-worker` · Branch: `main` · **Dockerfile path:** `Dockerfile` (repo root).
4. Endpoint type: **Queue**. GPU 16 GB+. Container disk **50 GB**.
5. Env: `MODEL_ID`, `ALIGN_MODEL`, `ALIGN_LANGUAGE=hin`, `ENABLE_ALIGNMENT=true`, `RUNPOD_INIT_TIMEOUT=900`.
6. Deploy — RunPod builds the image. No Hub tests.

Smoke test `{ "input": { "health_check": true } }`, then set `RUNPOD_ENDPOINT_ID` in Capten.

**Do not** use Deploy from the Hub (2h tests) or paste `registry.runpod.net/...` (Hub registry auth error).

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
