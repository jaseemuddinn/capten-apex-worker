"""
Production word-level forced alignment via torchaudio MMS_FA.

Maps a known transcript (e.g. Sarvam STT) onto 16 kHz mono audio using
CTC forced alignment (torchaudio.functional.forced_align + merge_tokens).

CLI:
  python mms_fa_align.py --audio clip.wav --text "yeh perfect hai" \\
      --lang hin --json out.json --srt out.srt

Import:
  from mms_fa_align import align_words
  words = align_words(audio_path_or_array, "yeh perfect hai", language="hin")
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

# MMS CTC frame hop: 320 samples @ 16 kHz → 20 ms / frame
SAMPLE_RATE = 16_000
FRAME_SHIFT_SAMPLES = 320
SECONDS_PER_FRAME = FRAME_SHIFT_SAMPLES / SAMPLE_RATE  # 0.02

# Capten Language → MMS / ISO-ish codes used for romanization hints
LANG_ALIASES = {
    "hinglish": "hin",
    "hindi": "hin",
    "hi": "hin",
    "en": "eng",
    "english": "eng",
    "tamil": "tam",
    "ta": "tam",
    "telugu": "tel",
    "te": "tel",
    "kannada": "kan",
    "kn": "kan",
    "bengali": "ben",
    "bn": "ben",
    "marathi": "mar",
    "mr": "mar",
    "punjabi": "pan",
    "pa": "pan",
    "malayalam": "mal",
    "ml": "mal",
    "gujarati": "guj",
    "gu": "guj",
    "odia": "ori",
    "or": "ori",
    "assamese": "asm",
    "as": "asm",
    "urdu": "urd",
    "ur": "urd",
    "nepali": "nep",
    "ne": "nep",
    "auto": "hin",
}

_model = None
_device = None
_dictionary = None
_labels = None


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    confidence: float = 0.9


class AlignmentError(Exception):
    """Raised when audio/text cannot be aligned safely."""


def normalize_lang(language: Optional[str]) -> str:
    if not language:
        return "hin"
    key = str(language).strip().lower().replace("_", "-")
    if "-" in key:
        key = key.split("-", 1)[0]
    return LANG_ALIASES.get(key, key if len(key) == 3 else "hin")


def tokenize_transcript(text: str) -> List[str]:
    """
    Split Sarvam (or any) transcript into display words.

    Keeps alphanumeric tokens; strips most punctuation so CTC char mapping
    stays stable, without dropping code-mixed Latin/Indic tokens.
    """
    if not text or not str(text).strip():
        raise AlignmentError("Empty transcript text")

    cleaned = str(text).replace("\u00a0", " ").strip()
    # Keep letters/digits across scripts; drop standalone punctuation.
    raw = re.findall(r"[\w']+", cleaned, flags=re.UNICODE)
    words = [w for w in raw if w and not re.fullmatch(r"_+", w)]
    if not words:
        raise AlignmentError("Transcript produced no alignable words after cleaning")
    return words


def _needs_romanize(words: Sequence[str]) -> bool:
    return any(ord(ch) > 127 for w in words for ch in w)


def _romanize_word(word: str, language: str) -> str:
    """Map a display word to Latin a–z for the MMS_FA dictionary."""
    if not _needs_romanize([word]):
        return word.lower()

    try:
        import uroman as ur

        romanizer = getattr(ur, "Uroman", None)
        if romanizer is not None:
            r = romanizer()
            out = r.romanize_string(word, lcode=language)
        else:
            # Older uroman API
            out = ur.romanize_string(word)  # type: ignore[attr-defined]
        return re.sub(r"[^a-z0-9']+", "", out.lower())
    except Exception:
        # Last resort: drop non-ASCII (better than silent CTC assert).
        return re.sub(r"[^a-z0-9']+", "", word.lower())


def prepare_align_tokens(
    words: Sequence[str], language: str, dictionary: dict
) -> Tuple[List[str], List[int], List[int]]:
    """
    Returns:
      roman_words: per-display-word Latin strings used for CTC
      flat_token_ids: character ids (no word separators in target)
      char_counts: chars per word (for unflattening TokenSpans)
    """
    roman_words: List[str] = []
    flat: List[int] = []
    counts: List[int] = []

    for w in words:
        roman = _romanize_word(w, language)
        chars = [c for c in roman if c in dictionary]
        if not chars:
            # Keep a placeholder so word count stays aligned with display words.
            # Use 'a' if present; otherwise skip and mark count 0.
            if "a" in dictionary:
                chars = ["a"]
            else:
                roman_words.append("")
                counts.append(0)
                continue
        roman_words.append("".join(chars))
        counts.append(len(chars))
        flat.extend(dictionary[c] for c in chars)

    if not flat:
        raise AlignmentError(
            "No characters mapped into the MMS_FA dictionary "
            "(check script / romanization)"
        )
    return roman_words, flat, counts


def load_audio(
    source: Union[str, Path, "torch.Tensor"],
    sample_rate: Optional[int] = None,
) -> "torch.Tensor":
    """Load / convert to mono float waveform at 16 kHz. Shape: (1, T)."""
    import torch
    import torchaudio

    if isinstance(source, torch.Tensor):
        waveform = source.detach().float().cpu()
        sr = sample_rate or SAMPLE_RATE
    else:
        path = Path(source)
        if not path.is_file():
            raise AlignmentError(f"Audio file not found: {path}")
        waveform, sr = torchaudio.load(str(path))

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    if waveform.numel() < SAMPLE_RATE * 0.05:
        raise AlignmentError("Audio too short to align (<50ms)")

    return waveform


def _get_bundle_state(device: Optional[str] = None):
    global _model, _device, _dictionary, _labels
    import torch
    from torchaudio.pipelines import MMS_FA

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _model is None or _device != device:
        print(f"[mms_fa] loading MMS_FA on {device}", flush=True)
        _model = MMS_FA.get_model(with_star=False).to(device).eval()
        _dictionary = MMS_FA.get_dict(star=None)
        _labels = list(MMS_FA.get_labels(star=None))
        _device = device
        print(f"[mms_fa] ready labels={len(_labels)}", flush=True)

    return _model, _dictionary, _labels, _device


def _unflatten_spans(spans, char_counts: Sequence[int]):
    """Group character TokenSpans into word-level span lists."""
    words = []
    idx = 0
    for n in char_counts:
        if n <= 0:
            words.append([])
            continue
        chunk = spans[idx : idx + n]
        if len(chunk) != n:
            raise AlignmentError(
                f"TokenSpan count mismatch: need {n} chars, got {len(chunk)} "
                f"(audio/text length mismatch)"
            )
        words.append(chunk)
        idx += n
    if idx != len(spans):
        raise AlignmentError(
            f"Leftover TokenSpans after word grouping "
            f"(used={idx}, total={len(spans)})"
        )
    return words


def _spans_to_words(
    display_words: Sequence[str],
    word_span_groups,
    audio_duration: float,
) -> List[AlignedWord]:
    out: List[AlignedWord] = []
    for word, spans in zip(display_words, word_span_groups):
        if not spans:
            # No CTC chars — borrow previous end or 0, tiny width.
            start = out[-1].end if out else 0.0
            end = min(audio_duration, start + 0.05)
            out.append(AlignedWord(word=word, start=start, end=end, confidence=0.5))
            continue
        start_f = spans[0].start
        end_f = spans[-1].end
        start = max(0.0, start_f * SECONDS_PER_FRAME)
        end = min(audio_duration, max(start + 0.02, end_f * SECONDS_PER_FRAME))
        score = float(sum(s.score for s in spans) / max(len(spans), 1))
        conf = min(max(score, 0.5), 0.99)
        out.append(
            AlignedWord(
                word=word,
                start=round(start, 3),
                end=round(end, 3),
                confidence=round(conf, 3),
            )
        )

    # Soft de-overlap
    for i in range(1, len(out)):
        if out[i].start < out[i - 1].end:
            out[i].start = out[i - 1].end
            if out[i].end < out[i].start:
                out[i].end = round(out[i].start + 0.02, 3)
    return out


def align_waveform(
    waveform: "torch.Tensor",
    text: str,
    language: str = "hin",
    device: Optional[str] = None,
) -> List[AlignedWord]:
    """Align transcript to a (1, T) 16 kHz waveform."""
    import torch
    import torchaudio.functional as F

    model, dictionary, _labels, dev = _get_bundle_state(device)
    lang = normalize_lang(language)
    display_words = tokenize_transcript(text)
    _roman, token_ids, char_counts = prepare_align_tokens(
        display_words, lang, dictionary
    )

    duration = waveform.shape[-1] / SAMPLE_RATE
    # Rough guard: more CTC chars than frames → almost certainly mismatched.
    n_frames_est = max(1, int(waveform.shape[-1] / FRAME_SHIFT_SAMPLES))
    if len(token_ids) > n_frames_est:
        raise AlignmentError(
            f"Text longer than audio capacity "
            f"(chars={len(token_ids)} frames≈{n_frames_est} duration={duration:.2f}s)"
        )

    wav = waveform.to(dev)
    with torch.inference_mode():
        emission, _ = model(wav)
        # emission: (batch, time, class) or (time, class)
        if emission.dim() == 2:
            emission = emission.unsqueeze(0)
        targets = torch.tensor([token_ids], dtype=torch.int32, device=dev)
        input_lengths = torch.tensor([emission.size(1)], device=dev)
        target_lengths = torch.tensor([len(token_ids)], device=dev)

        aligned_tokens, scores = F.forced_align(
            emission, targets, input_lengths, target_lengths, blank=0
        )
        aligned_tokens, scores = aligned_tokens[0], scores[0]
        scores = scores.exp()
        token_spans = F.merge_tokens(aligned_tokens, scores)

    word_groups = _unflatten_spans(token_spans, char_counts)
    return _spans_to_words(display_words, word_groups, duration)


def align_words(
    audio: Union[str, Path, "torch.Tensor"],
    text: str,
    language: str = "hin",
    sample_rate: Optional[int] = None,
    device: Optional[str] = None,
    max_window_sec: float = 28.0,
) -> List[dict]:
    """
    High-level API used by Capten RunPod worker.

    For long audio, splits the transcript into time-proportional windows so
    CTC does not drift across the whole file.
    """
    waveform = load_audio(audio, sample_rate=sample_rate)
    duration = waveform.shape[-1] / SAMPLE_RATE
    display_words = tokenize_transcript(text)

    if duration <= max_window_sec or len(display_words) <= 48:
        aligned = align_waveform(waveform, text, language=language, device=device)
        return [asdict(w) for w in aligned]

    # Chunk by word count with proportional time windows + pad.
    chunk_size = max(12, min(40, int(len(display_words) * max_window_sec / duration)))
    total_chars = sum(max(len(w), 1) for w in display_words) or 1
    results: List[AlignedWord] = []
    idx = 0
    t_cursor = 0.0

    while idx < len(display_words):
        chunk_words = display_words[idx : idx + chunk_size]
        weight = sum(max(len(w), 1) for w in chunk_words) / total_chars
        win_dur = max(0.4, duration * weight)
        t0 = max(0.0, t_cursor - 0.25)
        t1 = min(duration, t_cursor + win_dur + 0.35)
        if t1 <= t0:
            t1 = min(duration, t0 + 0.5)

        s0 = int(t0 * SAMPLE_RATE)
        s1 = int(t1 * SAMPLE_RATE)
        slice_wav = waveform[:, s0:s1]
        chunk_text = " ".join(chunk_words)
        try:
            part = align_waveform(
                slice_wav, chunk_text, language=language, device=device
            )
            for w in part:
                results.append(
                    AlignedWord(
                        word=w.word,
                        start=round(w.start + t0, 3),
                        end=round(w.end + t0, 3),
                        confidence=w.confidence,
                    )
                )
        except AlignmentError as exc:
            print(f"[mms_fa] window {t0:.2f}-{t1:.2f} failed: {exc}", flush=True)
            # Weighted fallback inside the window
            span = max(t1 - t0, 0.05 * len(chunk_words))
            weights = [max(len(w), 1) for w in chunk_words]
            tw = float(sum(weights))
            c = t0
            for w, wt in zip(chunk_words, weights):
                width = span * (wt / tw)
                results.append(
                    AlignedWord(
                        word=w,
                        start=round(c, 3),
                        end=round(c + width, 3),
                        confidence=0.55,
                    )
                )
                c += width

        idx += len(chunk_words)
        t_cursor = min(duration, t_cursor + win_dur)

    # Soft de-overlap across windows
    for i in range(1, len(results)):
        if results[i].start < results[i - 1].end:
            results[i].start = results[i - 1].end
            if results[i].end < results[i].start:
                results[i].end = round(results[i].start + 0.02, 3)

    return [asdict(w) for w in results]


def words_to_srt(words: Sequence[dict], max_words_per_cue: int = 6) -> str:
    """Build a simple SRT from word-level timestamps."""

    def fmt(t: float) -> str:
        if t < 0:
            t = 0.0
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        if ms == 1000:
            s += 1
            ms = 0
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    cue = 1
    i = 0
    items = list(words)
    while i < len(items):
        chunk = items[i : i + max_words_per_cue]
        start = float(chunk[0]["start"])
        end = float(chunk[-1]["end"])
        text = " ".join(str(w["word"]) for w in chunk)
        lines.append(str(cue))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(text)
        lines.append("")
        cue += 1
        i += max_words_per_cue
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Word-level CTC forced alignment (torchaudio MMS_FA)"
    )
    parser.add_argument("--audio", required=True, help="Path to .wav / .mp3 / .flac")
    parser.add_argument("--text", required=True, help="Transcript string (e.g. Sarvam)")
    parser.add_argument(
        "--lang",
        default="hin",
        help="Language hint for romanization (hin, eng, tam, ...)",
    )
    parser.add_argument("--json", dest="json_out", help="Write words JSON array")
    parser.add_argument("--srt", dest="srt_out", help="Write SRT subtitle file")
    parser.add_argument(
        "--device",
        default=None,
        help="cuda | cpu (default: auto)",
    )
    args = parser.parse_args(argv)

    try:
        words = align_words(
            args.audio, args.text, language=args.lang, device=args.device
        )
    except AlignmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(words, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.json_out} ({len(words)} words)", flush=True)
    else:
        print(payload)

    if args.srt_out:
        Path(args.srt_out).write_text(words_to_srt(words), encoding="utf-8")
        print(f"wrote {args.srt_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
