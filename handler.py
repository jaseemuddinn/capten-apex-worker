"""Capten Apex RunPod worker — keep top-level imports minimal for fast Hub health checks."""
from __future__ import annotations

import os
from pathlib import Path

import runpod

WORKER_BUILD_ID = "cu128-v8"
print(f"[startup] capten apex worker {WORKER_BUILD_ID}", flush=True)

MODEL_ID = os.getenv("MODEL_ID", "Oriserve/Whisper-Hindi2Hinglish-Apex")
ALIGN_MODEL = os.getenv(
    "ALIGN_MODEL", "MahmoudAshraf/mms-300m-1130-forced-aligner"
)
# ISO 639-3 — MMS forced aligner vocabulary is Latin a–z; works with Apex Hinglish text.
ALIGN_LANGUAGE = os.getenv("ALIGN_LANGUAGE", "hin")
ENABLE_ALIGNMENT = os.getenv("ENABLE_ALIGNMENT", "true").lower() not in (
    "0",
    "false",
    "no",
)
MMS_BATCH_SIZE = int(os.getenv("MMS_BATCH_SIZE", "4"))

_pipe = None
_mms_model = None
_mms_tokenizer = None
_resolved_device: str | None = None


def force_cpu() -> bool:
    return os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes")


def cuda_kernels_ok() -> bool:
    """True when this PyTorch build can run fp16 kernels on the visible GPU."""
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        arch_list = []
        if hasattr(torch.cuda, "get_arch_list"):
            try:
                arch_list = torch.cuda.get_arch_list()
            except Exception:
                pass
        torch.zeros(1, device="cuda")
        probe = torch.zeros(8, 8, device="cuda", dtype=torch.float16)
        torch.matmul(probe, probe)
        torch.cuda.synchronize()
        print(
            f"[cuda] ok device={name} sm_{cap[0]}{cap[1]} "
            f"torch={torch.__version__} cuda={torch.version.cuda} arch_list={arch_list}",
            flush=True,
        )
        return True
    except RuntimeError as exc:
        print(f"[cuda] kernels unavailable: {exc}", flush=True)
        return False


def resolve_device() -> str:
    """Pick cuda when kernels work; otherwise CPU (slow but always works)."""
    global _resolved_device
    if _resolved_device is not None:
        return _resolved_device
    if force_cpu():
        _resolved_device = "cpu"
    elif cuda_kernels_ok():
        _resolved_device = "cuda"
    else:
        _resolved_device = "cpu"
    print(f"[device] using {_resolved_device}", flush=True)
    return _resolved_device


def device_name() -> str:
    return resolve_device()


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


def load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    print("[load] apex model", flush=True)
    device = resolve_device()
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


def load_mms_align_model():
    """MMS CTC forced-alignment model (lazy, cached)."""
    global _mms_model, _mms_tokenizer
    if _mms_model is not None:
        return _mms_model, _mms_tokenizer

    import torch
    from ctc_forced_aligner import load_alignment_model

    device = device_name()
    dtype = torch.float16 if device == "cuda" else torch.float32
    local_path = resolve_snapshot_path(ALIGN_MODEL)
    model_source = local_path or ALIGN_MODEL

    if local_path:
        print(f"[align] using cached MMS snapshot {local_path}", flush=True)

    _mms_model, _mms_tokenizer = load_alignment_model(
        device,
        model_path=model_source,
        dtype=dtype,
    )
    print("[align] MMS model ready", flush=True)
    return _mms_model, _mms_tokenizer


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


def align_text_to_words_mms(audio, text: str, device: str) -> list[dict]:
    """Pass 2: MMS forced alignment of full Apex text → per-word times with silence gaps."""
    import torch
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        postprocess_results,
        preprocess_text,
    )

    cleaned = text.strip()
    if not cleaned:
        return []

    model, tokenizer = load_mms_align_model()
    dtype = torch.float16 if device == "cuda" else torch.float32

    waveform = torch.from_numpy(audio).float()
    if waveform.dim() > 1:
        waveform = waveform.squeeze()
    waveform = waveform.to(device=device, dtype=dtype)

    emissions, stride = generate_emissions(
        model,
        waveform,
        batch_size=MMS_BATCH_SIZE,
    )

    tokens_starred, text_starred = preprocess_text(
        cleaned,
        romanize=True,
        language=ALIGN_LANGUAGE,
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        tokenizer,
    )
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    words: list[dict] = []
    for wt in word_timestamps:
        token = (wt.get("text") or "").strip()
        if not token:
            continue
        start = float(wt["start"])
        end = float(wt["end"])
        if end <= start:
            end = start + 0.1
        score = wt.get("score")
        if score is not None:
            span_frames = max((end - start) * 50, 1.0)
            confidence = min(max(float(score) / span_frames, 0.5), 0.99)
        else:
            confidence = 0.92
        words.append(
            {
                "word": token,
                "start": round(start, 3),
                "end": round(end, 3),
                "confidence": round(confidence, 3),
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

        cuda_ok = cuda_kernels_ok()
        device = resolve_device()
        info: dict = {
            "status": "ok",
            "build": WORKER_BUILD_ID,
            "device": device,
            "cuda_kernels_ok": cuda_ok,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "model": MODEL_ID,
            "align_model": ALIGN_MODEL,
            "alignment_default": ENABLE_ALIGNMENT,
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["capability"] = f"sm_{cap[0]}{cap[1]}"
            if hasattr(torch.cuda, "get_arch_list"):
                try:
                    info["arch_list"] = torch.cuda.get_arch_list()
                except Exception:
                    pass
        return info

    import numpy as np
    import torch

    device = resolve_device()
    pipe = load_pipeline()
    audio_bytes = load_audio_bytes(job_input)
    sample = bytes_to_sample(audio_bytes)
    duration = len(sample["array"]) / sample["sampling_rate"]
    audio = np.asarray(sample["array"], dtype=np.float32)

    result = run_transcription(pipe, sample)
    text = (result.get("text") or "").strip()
    chunks = result.get("chunks") or []

    words: list[dict] = []
    alignment = "disabled"
    do_align = alignment_enabled(job_input)

    if do_align and text:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            words = align_text_to_words_mms(audio, text, device)
            alignment = "mms" if words else "mms_empty"
            if words:
                print(f"[align] MMS ok words={len(words)}", flush=True)
        except Exception as exc:
            alignment = f"mms_failed:{type(exc).__name__}"
            print(f"[align] MMS alignment failed: {exc}", flush=True)

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
        "build": WORKER_BUILD_ID,
        "device": device,
        "alignment": alignment,
        "align_language": ALIGN_LANGUAGE,
        "align_model": ALIGN_MODEL,
    }


print("[startup] registering handler", flush=True)
runpod.serverless.start({"handler": handler})
