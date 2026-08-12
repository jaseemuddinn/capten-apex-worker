# Capten Apex Worker

RunPod Serverless worker: Apex Hinglish STT + MMS word alignment.

## Why Hub always “tests”

This repo is already a **Hub listing**. Hub has **no skip-tests button**. If `.runpod/tests.json` exists, every GitHub **Release** (and Hub rebuild) does:

1. Build image  
2. Spin a **GPU test pod** (`Deploying test pod on RTX 4090… Waiting for container startup…`)  
3. Hang for up to 2 hours if that pod never starts  

That is why it looks the same every time. Changing the handler does not skip step 2.

**Fix in this repo:** `.runpod/tests.json` is renamed to `.runpod/tests_.json` (same trick as official `worker-comfyui`). Hub then **skips the test pod** and only builds.

## What to do now

1. Commit + push `main`.
2. Create a **new GitHub Release** (Hub only indexes releases).
3. In Hub, wait for **build** only — you should **not** see “Deploying test pod”.
4. After build succeeds, **Create an endpoint → Deploy from the Hub** (Hub can pull its own registry).

### Capten production (no Hub at all)

**Serverless → New Endpoint → Deploy from a GitHub repository** (the GitHub card).

- Repo `jaseemuddinn/capten-apex-worker`, branch `main`, Dockerfile `Dockerfile`
- Queue, 16 GB+ GPU, **50 GB** disk
- Env: `ENABLE_ALIGNMENT=true`, `RUNPOD_INIT_TIMEOUT=900`

This path does **not** use Hub tests. If you still see “test pod”, you clicked **Hub**, not GitHub.

Do **not** paste `registry.runpod.net/...` into Docker deploy (Hub auth error).  
Do **not** install Docker on your Mac.

Smoke: `{ "input": { "health_check": true } }` → set `RUNPOD_ENDPOINT_ID` in Capten.
