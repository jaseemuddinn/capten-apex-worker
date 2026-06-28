"""Capten Apex RunPod worker — keep top-level imports minimal for fast Hub health checks."""
from __future__ import annotations

import os
import re
from pathlib import Path

import runpod

print("[startup] capten apex worker", flush=True)

MODEL_ID = os.getenv("MODEL_ID", "Oriserve/Whisper-Hindi2Hinglish-Apex")
ALIGN_LANGUAGE = os.getenv("ALIGN_LANGUAGE", "hi")
ENABLE_ALIGNMENT = os.getenv("ENABLE_ALIGNMENT", "true").lower() not in (
    "0",
    "false",
    "no",
)

_pipe = None
_align_model = None
_align_metadata = None


def hub_cache_roots() -> list[Path]:
    """HF cache locations: RunPod volume first, then image-baked default cache."""
    roots: list[Path] = [Path("/runpod-volume/huggingface-cache/hub")]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        roots.append(Path(hub_cache))
    return roots


def resolve_snapshot_path(model_id: str) -> str | None:
    """Resolve HF cache: models--Org--Name/snapshots/<hash>/"""
    folder = "models--" + model_id.replace("/", "--")
    for root in hub_cache_roots():
        snapshots = root / folder / "snapshots"
        if not snapshots.exists():
            continue

        refs_main = root / folder / "refs" / "main"
        if refs_main.exists():
            commit = refs_main.read_text().strip()
            snap = snapshots / commit
            if snap.is_dir():
                return str(snap)

        for snap in sorted(snapshots.iterdir()):
            if snap.is_dir():
                return str(snap)
    return None


def device_name() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def verify_cuda() -> None:
    """Fail fast with a clear message when PyTorch lacks kernels for this GPU."""
    import torch

    if not torch.cuda.is_available():
        print("[cuda] GPU not visible — running on CPU")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch_list = []
    if hasattr(torch.cuda, "get_arch_list"):
        try:
            arch_list = torch.cuda.get_arch_list()
        except Exception:
            pass
    print(
        f"[cuda] device={name} sm_{cap[0]}{cap[1]} "
        f"torch={torch.__version__} arch_list={arch_list}"
    )
    try:
        # float32 smoke test
        torch.zeros(1, device="cuda")
        # Apex runs fp16 on GPU — catch arch mismatches that only show up in half kernels
        probe = torch.zeros(8, 8, device="cuda", dtype=torch.float16)
        torch.matmul(probe, probe)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(
            f"PyTorch {torch.__version__} (cuda {torch.version.cuda}) cannot run on "
            f"{name} (sm_{cap[0]}{cap[1]}). Rebuild with cu128 torch>=2.7 "
            "(see Dockerfile). Original: {exc}"
        ) from exc


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    print("[load] apex model", flush=True)
    device = device_name()
    dtype = torch.float16 if device == "cuda" else torch.float32

    local_path = resolve_snapshot_path(MODEL_ID)
    model_source = local_path or MODEL_ID

    if local_path:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"[load] using cached snapshot {local_path}", flush=True)

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
    print("[load] apex ready", flush=True)
    return _pipe


def load_align_model():
    """WhisperX wav2vec2 forced-alignment model (lazy, cached)."""
    global _align_model, _align_metadata
    if _align_model is not None:
        return _align_model, _align_metadata

    import whisperx

    device = device_name()
    _align_model, _align_metadata = whisperx.load_align_model(
        language_code=ALIGN_LANGUAGE,
        device=device,
    )
    return _align_model, _align_metadata


def load_audio_bytes(job_input: dict) -> bytes:
    import base64

    import requests

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
    import io

    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return {"array": audio, "sampling_rate": sr}


def run_transcription(pipe, sample: dict) -> dict:
    """Apex fine-tune has no alignment_heads — text only, chunk-level times."""
    return pipe(sample, chunk_length_s=30, batch_size=4)


def apex_chunks_to_segments(chunks: list, text: str, duration: float) -> list[dict]:
    """Build coarse segments from Apex chunks for WhisperX forced alignment."""
    segments: list[dict] = []

    for chunk in chunks or []:
        chunk_text = (chunk.get("text") or "").strip()
        ts = chunk.get("timestamp")
        if not chunk_text:
            continue
        if ts and ts[0] is not None:
            start = float(ts[0])
            end = float(ts[1] if ts[1] is not None else ts[0] + 0.3)
        else:
            continue
        segments.append({"text": chunk_text, "start": start, "end": max(end, start + 0.05)})

    if segments:
        return segments

    return text_to_segments(text, duration)


def text_to_segments(text: str, duration: float) -> list[dict]:
    """Fallback segment split when Apex returns no chunk timestamps."""
    raw = text.strip()
    if not raw or duration <= 0:
        return []

    parts = re.split(r"(?<=[.!?।])\s+", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == 1 and len(raw) > 100:
        parts = [p.strip() for p in re.split(r",\s+", raw) if p.strip()]
    if not parts:
        parts = [raw]

    total = sum(max(len(p), 1) for p in parts)
    segments: list[dict] = []
    t = 0.0
    for part in parts:
        share = max(len(part), 1) / total
        end = min(t + duration * share, duration)
        if end <= t:
            end = min(t + 0.25, duration)
        segments.append({"text": part, "start": t, "end": end})
        t = end

    if segments:
        segments[-1]["end"] = duration
    return segments


def align_segments_to_words(audio, segments: list[dict], device: str) -> list[dict]:
    """Pass 2: forced phoneme alignment — per-word times that respect silence."""
    import whisperx

    if not segments:
        return []

    model_a, metadata = load_align_model()
    aligned = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return aligned_segments_to_words(aligned)


def aligned_segments_to_words(aligned: dict) -> list[dict]:
    words: list[dict] = []
    for seg in aligned.get("segments") or []:
        for w in seg.get("words") or []:
            token = (w.get("word") or "").strip()
            if not token:
                continue
            start = w.get("start")
            if start is None:
                continue
            start = float(start)
            end = w.get("end")
            end = float(end) if end is not None else start + 0.15
            if end <= start:
                end = start + 0.1
            score = w.get("score")
            words.append(
                {
                    "word": token,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "confidence": round(float(score), 3) if score is not None else 0.92,
                }
            )

    words.sort(key=lambda x: x["start"])
    return words


def chunks_to_words(chunks: list) -> list[dict]:
    """Legacy chunk-level timestamps (all words in a chunk share the same span)."""
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


def text_to_words(text: str, duration: float | None = None) -> list[dict]:
    tokens = text.split()
    if not tokens:
        return []
    if duration and duration > 0:
        step = duration / len(tokens)
        return [
            {
                "word": w,
                "start": round(i * step, 3),
                "end": round((i + 1) * step, 3),
                "confidence": 0.85,
            }
            for i, w in enumerate(tokens)
        ]
    return [
        {
            "word": w,
            "start": round(i * 0.4, 3),
            "end": round((i + 1) * 0.4, 3),
            "confidence": 0.85,
        }
        for i, w in enumerate(tokens)
    ]


def build_output_segments(words: list[dict], text: str, duration: float) -> list[dict]:
    if not words:
        return [{"text": text, "start": 0.0, "end": duration}]
    return [
        {
            "text": text,
            "start": words[0]["start"],
            "end": words[-1]["end"],
        }
    ]


def alignment_enabled(job_input: dict) -> bool:
    if job_input.get("skip_alignment") or job_input.get("health_check"):
        return False
    return ENABLE_ALIGNMENT


def handler(job):
    job_input = job["input"]

    if job_input.get("health_check"):
        import torch

        verify_cuda()
        info: dict = {
            "status": "ok",
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "model": MODEL_ID,
            "alignment_default": ENABLE_ALIGNMENT,
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
        return info

    import numpy as np
    import torch

    verify_cuda()
    pipe = load_pipeline()
    audio_bytes = load_audio_bytes(job_input)
    sample = bytes_to_sample(audio_bytes)
    duration = len(sample["array"]) / sample["sampling_rate"]
    device = device_name()
    audio = np.asarray(sample["array"], dtype=np.float32)

    result = run_transcription(pipe, sample)
    text = (result.get("text") or "").strip()
    chunks = result.get("chunks") or []
    segments = apex_chunks_to_segments(chunks, text, duration)

    words: list[dict] = []
    alignment = "disabled"
    do_align = alignment_enabled(job_input)

    if do_align and segments:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            words = align_segments_to_words(audio, segments, device)
            alignment = "whisperx" if words else "whisperx_empty"
        except Exception as exc:
            alignment = f"whisperx_failed:{type(exc).__name__}"
            print(f"[align] WhisperX alignment failed: {exc}")

    if not words:
        words = chunks_to_words(chunks)
        if words:
            alignment = "chunk_fallback"

    if not words and text:
        words = text_to_words(text, duration)
        alignment = "even_fallback"

    return {
        "text": text,
        "words": words,
        "segments": build_output_segments(words, text, duration),
        "language": "HINGLISH",
        "model": MODEL_ID,
        "alignment": alignment,
        "align_language": ALIGN_LANGUAGE,
    }


print("[startup] registering handler", flush=True)
runpod.serverless.start({"handler": handler})
