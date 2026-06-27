import base64
import io
import os
from pathlib import Path

import requests
import runpod
import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("MODEL_ID", "Oriserve/Whisper-Hindi2Hinglish-Apex")
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

_pipe = None


def resolve_snapshot_path(model_id: str) -> str | None:
    """Resolve RunPod HF cache: models--Org--Name/snapshots/<hash>/"""
    folder = "models--" + model_id.replace("/", "--")
    snapshots = CACHE_ROOT / folder / "snapshots"
    if not snapshots.exists():
        return None

    refs_main = CACHE_ROOT / folder / "refs" / "main"
    if refs_main.exists():
        commit = refs_main.read_text().strip()
        snap = snapshots / commit
        if snap.is_dir():
            return str(snap)

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

    local_path = resolve_snapshot_path(MODEL_ID)
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
        attn_implementation="eager",
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
            "language": "en",
        },
    )
    return _pipe


def load_audio_bytes(job_input: dict) -> bytes:
    for key in ("audio_url", "url", "audio"):
        val = job_input.get(key)
        if isinstance(val, str) and val.startswith("http"):
            res = requests.get(val, timeout=120)
            res.raise_for_status()
            return res.content

    b64 = job_input.get("audio_base64")
    if isinstance(b64, str):
        return base64.b64decode(b64)

    audio = job_input.get("audio")
    if isinstance(audio, str) and "base64," in audio:
        return base64.b64decode(audio.split("base64,", 1)[1])

    raise ValueError(
        "No audio input. Pass audio_url, url, audio (http/https), or audio_base64."
    )


def bytes_to_sample(data: bytes) -> dict:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return {"array": audio, "sampling_rate": sr}


def text_to_words(text: str, duration: float | None = None) -> list[dict]:
    tokens = text.split()
    if not tokens:
        return []
    if duration and duration > 0:
        step = duration / len(tokens)
        return [
            {
                "word": w,
                "start": i * step,
                "end": (i + 1) * step,
                "confidence": 0.85,
            }
            for i, w in enumerate(tokens)
        ]
    return [
        {
            "word": w,
            "start": i * 0.4,
            "end": (i + 1) * 0.4,
            "confidence": 0.85,
        }
        for i, w in enumerate(tokens)
    ]


def run_transcription(pipe, sample: dict) -> dict:
    """Apex fine-tune lacks alignment_heads — word timestamps crash transformers."""
    try:
        result = pipe(
            sample,
            return_timestamps=True,
            chunk_length_s=30,
            batch_size=8,
        )
        if (result.get("text") or "").strip():
            return result
    except Exception:
        pass

    return pipe(sample, chunk_length_s=30, batch_size=8)


def chunks_to_words(chunks: list) -> list[dict]:
    words: list[dict] = []
    for chunk in chunks or []:
        text = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp")
        if not text or not ts or ts[0] is None:
            continue
        start = float(ts[0])
        end = float(ts[1] if ts[1] is not None else ts[0] + 0.3)
        for token in text.split():
            words.append(
                {"word": token, "start": start, "end": end, "confidence": 0.9}
            )
    return words


def handler(job):
    job_input = job["input"]
    pipe = load_pipeline()
    audio_bytes = load_audio_bytes(job_input)
    sample = bytes_to_sample(audio_bytes)
    duration = len(sample["array"]) / sample["sampling_rate"]

    result = run_transcription(pipe, sample)

    text = (result.get("text") or "").strip()
    words = chunks_to_words(result.get("chunks") or [])

    if not words and text:
        words = text_to_words(text, duration)

    return {
        "text": text,
        "words": words,
        "segments": [
            {
                "text": text,
                "start": 0.0,
                "end": words[-1]["end"] if words else duration,
            }
        ],
        "language": "HINGLISH",
        "model": MODEL_ID,
    }


runpod.serverless.start({"handler": handler})
