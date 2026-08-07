"""Download HF model files at image build time (no GPU / model instantiation)."""
import os
import sys

from huggingface_hub import snapshot_download


def main() -> None:
    models = [
        os.environ.get("MODEL_ID", "Oriserve/Whisper-Hindi2Hinglish-Apex"),
        os.environ.get(
            "ALIGN_MODEL", "MahmoudAshraf/mms-300m-1130-forced-aligner"
        ),
    ]
    # De-dupe while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for mid in models:
        if mid and mid not in seen:
            seen.add(mid)
            ordered.append(mid)

    for model_id in ordered:
        print(f"[cache] downloading {model_id}", flush=True)
        path = snapshot_download(repo_id=model_id)
        print(f"[cache] done: {path}", flush=True)

    print(f"[cache] finished {len(ordered)} model(s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[cache] FAILED: {exc}", file=sys.stderr, flush=True)
        raise
