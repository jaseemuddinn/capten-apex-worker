"""Capten Apex RunPod worker — keep top-level imports minimal for fast Hub health checks."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import runpod

WORKER_BUILD_ID = "cu128-v14"
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return max(int(raw), 1)
    except ValueError:
        print(f"[config] invalid {name}={raw!r}, using {default}", flush=True)
        return default


MMS_BATCH_SIZE = _env_int("MMS_BATCH_SIZE", 4)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        print(f"[config] invalid {name}={raw!r}, using {default}", flush=True)
        return default


# "segment" puts a wildcard before every word so pauses, breaths and music park
# on a star instead of stretching a real word. "edges" only stars the ends,
# which forces mid-file silence onto actual words and drifts the whole timeline.
ALIGN_STAR_FREQUENCY = os.getenv("ALIGN_STAR_FREQUENCY", "segment")
# Audio padding around each ASR segment before aligning it.
ALIGN_WINDOW_PAD = _env_float("ALIGN_WINDOW_PAD", 0.35)
# ASR segments are merged up to this length so alignment stays local (bounded
# drift) without paying per-segment overhead on every short sentence.
ALIGN_WINDOW_MAX_SEC = _env_float("ALIGN_WINDOW_MAX_SEC", 24.0)
MIN_ALIGN_WINDOW_SEC = 0.2
# Digit-only words are dropped by MMS text normalization and come back with a
# zero-width span; give them a readable slice of the following gap instead.
MIN_WORD_SEC = 0.04
DEGENERATE_WORD_SEC = 0.24

_pipe = None
_mms_model = None
_mms_tokenizer = None
_resolved_device: str | None = None


def force_cpu() -> bool:
    return os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes")


def cuda_kernels_ok(timeout_sec: float = 15.0) -> bool:
    """True when this PyTorch build can run fp16 kernels on the visible GPU.

    Hub test pods have hung forever on a stuck CUDA matmul — always bound the
    probe with a thread timeout so health_check cannot block the 2h Hub budget.
    """
    import threading

    import torch

    if not torch.cuda.is_available():
        return False

    result: dict = {"ok": False, "err": None}

    def probe() -> None:
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
            probe_t = torch.zeros(8, 8, device="cuda", dtype=torch.float16)
            torch.matmul(probe_t, probe_t)
            torch.cuda.synchronize()
            print(
                f"[cuda] ok device={name} sm_{cap[0]}{cap[1]} "
                f"torch={torch.__version__} cuda={torch.version.cuda} arch_list={arch_list}",
                flush=True,
            )
            result["ok"] = True
        except Exception as exc:
            result["err"] = exc
            print(f"[cuda] kernels unavailable: {exc}", flush=True)

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        print(
            f"[cuda] probe timed out after {timeout_sec:.0f}s — treating as unavailable",
            flush=True,
        )
        return False
    return bool(result["ok"])


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
    from transformers import AutoModelForCTC, AutoTokenizer

    device = device_name()
    dtype = torch.float16 if device == "cuda" else torch.float32
    local_path = resolve_snapshot_path(ALIGN_MODEL)
    model_source = local_path or ALIGN_MODEL
    offline = bool(local_path)

    if local_path:
        print(f"[align] using cached MMS snapshot {local_path}", flush=True)
    elif not offline:
        print(f"[align] downloading MMS model {ALIGN_MODEL}", flush=True)

    # Load directly — ctc_forced_aligner's wrapper uses `dtype=` which breaks on transformers 4.46+.
    _mms_model = AutoModelForCTC.from_pretrained(
        model_source,
        torch_dtype=dtype,
        local_files_only=offline,
    ).to(device).eval()
    _mms_tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=offline,
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
    """Apex fine-tune has no alignment_heads, so word timings come from MMS.

    Segment-level timestamps are still requested: they give the aligner local
    audio windows to work in, which is what keeps long videos from drifting.
    """
    try:
        return pipe(
            sample,
            chunk_length_s=30,
            batch_size=4,
            return_timestamps=True,
        )
    except Exception as exc:  # tokenizer without timestamp tokens
        print(
            f"[asr] segment timestamps unavailable ({type(exc).__name__}: {exc}) "
            "— decoding text only",
            flush=True,
        )
        return pipe(sample, chunk_length_s=30, batch_size=4)


def needs_romanize(text: str) -> bool:
    """MMS targets a Latin a–z vocab; non-ASCII text must go through uroman.

    Unknown characters are silently dropped from the CTC target, which then
    trips an assertion in get_spans and loses the whole window.
    """
    return any(ord(ch) > 127 for ch in text)


def align_window(
    audio_slice, text: str, device: str, offset: float = 0.0
) -> list[dict]:
    """Forced-align `text` inside one audio window; returns absolute-time words."""
    import numpy as np
    import torch
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        postprocess_results,
        preprocess_text,
    )

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    model, tokenizer = load_mms_align_model()
    dtype = torch.float16 if device == "cuda" else torch.float32

    waveform = torch.from_numpy(np.ascontiguousarray(audio_slice)).float()
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
        romanize=needs_romanize(cleaned),
        language=ALIGN_LANGUAGE,
        star_frequency=ALIGN_STAR_FREQUENCY,
    )

    content_tokens = [t for t in tokens_starred if t != "<star>"]
    if not content_tokens:
        print("[align] MMS preprocess produced no alignable tokens", flush=True)
        return []

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        tokenizer,
    )
    spans = get_spans(tokens_starred, segments, blank_token)

    if len(spans) != len(text_starred):
        raise ValueError(
            f"MMS span count {len(spans)} != text token count {len(text_starred)}"
        )

    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    words: list[dict] = []
    for wt in word_timestamps:
        token = (wt.get("text") or "").strip()
        if not token or token == "<star>":
            continue
        start = float(wt["start"]) + offset
        end = float(wt["end"]) + offset
        score = wt.get("score")
        if score is not None:
            span_frames = max((end - start) * 50, 1.0)
            confidence = min(max(float(score) / span_frames, 0.5), 0.99)
        else:
            confidence = 0.92
        words.append(
            {
                "word": token,
                "start": start,
                "end": end,
                "confidence": round(confidence, 3),
            }
        )

    return words


def weighted_words_in_span(text: str, start: float, end: float) -> list[dict]:
    """Last-resort split of a span, weighted by letters so short words stay short."""
    tokens = text.split()
    if not tokens:
        return []
    span = max(end - start, MIN_WORD_SEC * len(tokens))
    weights = [max(len(t.strip()), 1) for t in tokens]
    total = float(sum(weights))

    words: list[dict] = []
    cursor = start
    for token, weight in zip(tokens, weights):
        width = span * (weight / total)
        words.append(
            {
                "word": token,
                "start": cursor,
                "end": cursor + width,
                "confidence": 0.6,
            }
        )
        cursor += width
    return words


def plan_align_windows(chunks: list, total: float) -> list[dict]:
    """Turn ASR segment timestamps into merged, ordered alignment windows."""
    raw: list[dict] = []
    cursor = 0.0

    for chunk in chunks or []:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        ts = chunk.get("timestamp") or (None, None)
        start = ts[0] if ts and ts[0] is not None else cursor
        end = ts[1] if len(ts) > 1 and ts[1] is not None else None

        start = max(0.0, min(float(start), total))
        end = total if end is None else max(0.0, min(float(end), total))
        if end <= start:
            end = min(total, start + 0.3)
        raw.append({"text": text, "start": start, "end": end})
        cursor = end

    if not raw:
        return []

    raw.sort(key=lambda s: s["start"])
    merged: list[dict] = [dict(raw[0])]
    for seg in raw[1:]:
        last = merged[-1]
        if seg["end"] - last["start"] <= ALIGN_WINDOW_MAX_SEC:
            last["text"] = f"{last['text']} {seg['text']}".strip()
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged


def assign_caller_text_to_chunks(text: str, chunks: list) -> list:
    """Keep ASR window times, replace each chunk's text with a slice of caller text.

    Used by Hybrid (Sarvam wording + MMS). Whole-file MMS on long audio drifts;
    windowing from Apex segments keeps alignment local like Kalakar/WhisperX.
    """
    tokens = [t for t in (text or "").split() if t]
    if not tokens or not chunks:
        return chunks

    usable: list[dict] = []
    weights: list[float] = []
    for chunk in chunks:
        ts = chunk.get("timestamp") or (None, None)
        if ts[0] is None:
            continue
        start = float(ts[0])
        end = float(ts[1]) if ts[1] is not None else start + 0.3
        weights.append(max(0.05, end - start))
        usable.append(chunk)

    if not usable:
        return chunks

    total_w = sum(weights) or 1.0
    raw = [(w / total_w) * len(tokens) for w in weights]
    counts = [int(v) for v in raw]
    assigned = sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda i: raw[i] - counts[i],
        reverse=True,
    )
    k = 0
    while assigned < len(tokens) and order:
        counts[order[k % len(order)]] += 1
        assigned += 1
        k += 1

    # Ensure every non-empty window that got 0 still can receive leftovers later.
    out: list[dict] = []
    idx = 0
    for i, chunk in enumerate(usable):
        n = counts[i]
        if n <= 0:
            continue
        slice_tokens = tokens[idx : idx + n]
        idx += n
        if not slice_tokens:
            continue
        out.append({**chunk, "text": " ".join(slice_tokens)})

    if idx < len(tokens):
        leftover = " ".join(tokens[idx:])
        if out:
            out[-1]["text"] = f"{out[-1]['text']} {leftover}".strip()
        else:
            # No windows produced — fabricate one covering full span.
            first = usable[0]
            last = usable[-1]
            ts0 = first.get("timestamp") or (0.0, 0.0)
            ts1 = last.get("timestamp") or (0.0, 0.0)
            out.append(
                {
                    "text": leftover,
                    "timestamp": (ts0[0] or 0.0, ts1[1] if ts1[1] is not None else ts0[0]),
                }
            )

    print(
        f"[align] assigned caller text tokens={len(tokens)} windows={len(out)}",
        flush=True,
    )
    return out


def finalize_words(words: list[dict], total: float) -> list[dict]:
    """Sort, de-overlap and give zero-width tokens a visible duration."""
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w["start"], w["end"]))

    for i, w in enumerate(ordered):
        w["start"] = max(0.0, min(float(w["start"]), total))
        w["end"] = max(float(w["end"]), w["start"])
        if w["end"] - w["start"] < MIN_WORD_SEC:
            next_start = ordered[i + 1]["start"] if i + 1 < len(ordered) else total
            room = max(0.0, next_start - w["start"])
            w["end"] = w["start"] + min(DEGENERATE_WORD_SEC, room or DEGENERATE_WORD_SEC)
        w["end"] = min(w["end"], total)

    for i in range(1, len(ordered)):
        prev, cur = ordered[i - 1], ordered[i]
        if cur["start"] < prev["end"]:
            cur["start"] = prev["end"]
            if cur["end"] < cur["start"]:
                cur["end"] = cur["start"]

    out: list[dict] = []
    for w in ordered:
        if w["end"] - w["start"] <= 0:
            continue
        out.append(
            {
                "word": w["word"],
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
                "confidence": w.get("confidence", 0.9),
            }
        )
    return out


def align_transcript(audio, sr: int, text: str, chunks: list, device: str):
    """Align per ASR segment when possible, else one whole-file pass.

    Windowing keeps a bad stretch of audio from shifting every later word and
    isolates alignment failures to the segment that caused them.
    """
    total = len(audio) / float(sr)
    windows = plan_align_windows(chunks, total)

    if windows:
        words: list[dict] = []
        failed = 0
        for win in windows:
            w0 = max(0.0, win["start"] - ALIGN_WINDOW_PAD)
            w1 = min(total, win["end"] + ALIGN_WINDOW_PAD)
            if w1 - w0 < MIN_ALIGN_WINDOW_SEC:
                continue
            audio_slice = audio[int(w0 * sr) : int(w1 * sr)]
            try:
                aligned = align_window(audio_slice, win["text"], device, w0)
            except Exception as exc:
                failed += 1
                print(
                    f"[align] window {w0:.2f}-{w1:.2f}s failed "
                    f"({type(exc).__name__}: {exc}) — weighted split",
                    flush=True,
                )
                aligned = weighted_words_in_span(
                    win["text"], win["start"], win["end"]
                )
            words.extend(aligned)

        if words:
            source = "mms_segments" if not failed else f"mms_segments_partial:{failed}"
            print(
                f"[align] windows={len(windows)} failed={failed} words={len(words)}",
                flush=True,
            )
            return finalize_words(words, total), source

    words = align_window(audio, text, device, 0.0)
    return finalize_words(words, total), "mms" if words else "mms_empty"


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
        # Ultra-fast path first so Hub marks the worker alive even if torch/CUDA
        # init is slow or wedged on the test GPU.
        info: dict = {
            "status": "ok",
            "build": WORKER_BUILD_ID,
            "model": MODEL_ID,
            "align_model": ALIGN_MODEL,
            "align_language": ALIGN_LANGUAGE,
            "alignment_default": ENABLE_ALIGNMENT,
        }
        try:
            import torch

            try:
                cuda_ok = cuda_kernels_ok(timeout_sec=8.0)
                device = resolve_device()
            except Exception as exc:
                print(f"[health] probe failed: {exc}", flush=True)
                cuda_ok = False
                device = "cpu"

            info.update(
                {
                    "device": device,
                    "cuda_kernels_ok": cuda_ok,
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                }
            )
            if torch.cuda.is_available():
                try:
                    info["gpu"] = torch.cuda.get_device_name(0)
                    cap = torch.cuda.get_device_capability(0)
                    info["capability"] = f"sm_{cap[0]}{cap[1]}"
                    if hasattr(torch.cuda, "get_arch_list"):
                        try:
                            info["arch_list"] = torch.cuda.get_arch_list()
                        except Exception:
                            pass
                except Exception as exc:
                    info["gpu_error"] = str(exc)
        except Exception as exc:
            print(f"[health] torch import failed: {exc}", flush=True)
            info["device"] = "unknown"
            info["torch_error"] = str(exc)
        return info

    import numpy as np
    import torch

    device = resolve_device()
    # Caller-supplied text (e.g. Sarvam) is MMS-aligned; Apex still provides windows.
    align_text = job_input.get("align_text")
    align_text = align_text.strip() if isinstance(align_text, str) else ""

    pipe = load_pipeline()
    audio_bytes = load_audio_bytes(job_input)
    sample = bytes_to_sample(audio_bytes)
    duration = len(sample["array"]) / sample["sampling_rate"]
    audio = np.asarray(sample["array"], dtype=np.float32)

    if duration <= 0:
        return {
            "text": "",
            "words": [],
            "segments": [{"text": "", "start": 0.0, "end": 0.0}],
            "language": "HINGLISH",
            "model": MODEL_ID,
            "build": WORKER_BUILD_ID,
            "device": device,
            "alignment": "empty_audio",
            "align_language": ALIGN_LANGUAGE,
            "align_model": ALIGN_MODEL,
        }

    if align_text:
        text = align_text
        # Still run ASR for segment *windows* only — caller text is what we align.
        # Without windows, MMS on long files drifts (the old Hybrid failure mode).
        asr = run_transcription(pipe, sample)
        chunks = assign_caller_text_to_chunks(text, asr.get("chunks") or [])
        print(
            f"[asr] windows from Apex, text from caller ({len(text)} chars)",
            flush=True,
        )
    else:
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
            words, alignment = align_transcript(
                audio, sample["sampling_rate"], text, chunks, device
            )
            if words:
                print(f"[align] {alignment} words={len(words)}", flush=True)
        except Exception as exc:
            alignment = f"mms_failed:{type(exc).__name__}"
            print(f"[align] MMS alignment failed: {exc}", flush=True)
            traceback.print_exc()

    if align_text and words:
        alignment = f"{alignment}+provided_text"

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
try:
    runpod.serverless.start({"handler": handler})
except Exception:
    print("[startup] FATAL: runpod.serverless.start failed", flush=True)
    traceback.print_exc()
    raise
