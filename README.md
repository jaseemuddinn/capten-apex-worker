# Capten Apex Worker

[![Runpod](https://api.runpod.io/badge/jaseemuddinn/capten-apex-worker)](https://console.runpod.io/hub/jaseemuddinn/capten-apex-worker)

RunPod Serverless worker for [Oriserve/Whisper-Hindi2Hinglish-Apex](https://huggingface.co/Oriserve/Whisper-Hindi2Hinglish-Apex).

Transcribes Hindi audio into Hinglish (Latin script). Built for the [Capten](https://github.com) caption editor.

## Input

Send a job with one of:

```json
{ "input": { "audio_url": "https://example.com/audio.wav" } }
```

```json
{ "input": { "audio_base64": "<base64-encoded wav>" } }
```

Capten also sends `url`, `audio`, and `data:audio/wav;base64,...` formats.

## Output

```json
{
  "text": "transcribed hinglish text",
  "words": [{ "word": "hello", "start": 0.0, "end": 0.3, "confidence": 0.9 }],
  "segments": [{ "text": "...", "start": 0.0, "end": 1.2 }],
  "language": "HINGLISH",
  "model": "Oriserve/Whisper-Hindi2Hinglish-Apex"
}
```

## Deploy on RunPod Hub

1. Push this repo to GitHub.
2. Ensure `.runpod/Dockerfile`, `.runpod/hub.json`, and `.runpod/tests.json` are present (handler lives at repo-root `handler.py`).
3. Create a **GitHub Release** (Hub indexes releases — rebuilds from the release tag).
4. RunPod Console → Hub → rebuild the endpoint.

If Hub logs stall on **Waiting for container startup**, you need `cu128-v13+` (`ENTRYPOINT []` + 50 GB disk). Confirm health returns `"build": "cu128-v13"`.

## Deploy manually

1. New Serverless Endpoint → import from GitHub or Docker registry.
2. Set **Model** to `Oriserve/Whisper-Hindi2Hinglish-Apex` for cached cold starts.
3. GPU: 16 GB+, container disk: **50 GB** (Apex + MMS baked into the image).

## Capten `.env`


```env
GPU_MODE=runpod
RUNPOD_API_KEY=rpa_...
RUNPOD_ENDPOINT_ID=your_endpoint_id
```
