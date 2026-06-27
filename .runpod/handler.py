import base64
import io
import os
import re
from pathlib import Path

import requests
import runpod
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("MODEL_ID", "Oriserve/Whisper-Hindi2Hinglish-Apex")
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

_pipe = None


def resolve_cached_model_path(model_id: str) -> str | None:
    """Resolve RunPod HF cache path, e.g. models--Oriserve--Whisper-Hindi2Hinglish-Apex/snapshots/<hash>"""
    folder = "models--" + model_id.replace("/", "--")
    snapshots = CACHE_ROOT / folder / "snapshots"
    if not snapshots.exists():
        return None
    for snap in sorted(snapshots.iterdir()):
        if snap.is_dir():
            return str(snap)
    return None


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    local_path = resolve_cached_model_path(MODEL_ID)
    model_source = local_path or MODEL_ID

    if local_path:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=bool(local_path),
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_source,
        local_files_only=bool(local_path),
    )

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        generate_kwargs={
            "task": "transcribe",
            "language": "en",  # Apex outputs Hinglish in Latin script
        },
    )
    return _pipe


def load_audio_bytes(job_input: dict) -> bytes:
    for key in ("audio_url", "url", "audio"):
        val = job_input.get(key)
        if isinstance(val, str) and val.startswith("http"):
            r = requests.get(val, timeout=120)
            r.raise_for_status()
            return r.content

    b64 = job_input.get("audio_base64")
    if isinstance(b64, str):
        return base64.b64decode(b64)

    audio = job_input.get("audio")
    if isinstance(audio, str) and "base64," in audio:
        return base64.b64decode(audio.split("base64,", 1)[1])

    raise ValueError("No audio found. Pass audio_url, url, audio (http), or audio_base64.")


def bytes_to_float_array(data: bytes):
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
  # Resample to 16 kHz if needed (Whisper expects 16k)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return {"array": audio, "sampling_rate": sr}


def chunks_to_words(chunks):
    words = []
    for chunk in chunks or []:
        text = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp")
        if not text or not ts or ts[0] is None:
            continue
        start, end = float(ts[0]), float(ts[1] if ts[1] is not None else ts[0] + 0.3)
        for w in text.split():
            words.append({"word": w, "start": start, "end": end, "confidence": 0.9})
    return words


def handler(job):
    job_input = job["input"]
    pipe = load_pipeline()
    audio_bytes = load_audio_bytes(job_input)
    sample = bytes_to_float_array(audio_bytes)

    result = pipe(
        sample,
        return_timestamps="word",
        chunk_length_s=30,
        batch_size=8,
    )

    text = (result.get("text") or "").strip()
    chunks = result.get("chunks") or []
    words = chunks_to_words(chunks)

    if not words and text:
        words = [
            {"word": w, "start": i * 0.4, "end": (i + 1) * 0.4, "confidence": 0.85}
            for i, w in enumerate(text.split())
        ]

    return {
        "text": text,
        "words": words,
        "segments": [{"text": text, "start": 0, "end": words[-1]["end"] if words else 0}],
        "language": "HINGLISH",
        "model": MODEL_ID,
    }


runpod.serverless.start({"handler": handler})