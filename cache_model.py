"""Download HF model files at image build time (no GPU / model instantiation)."""
import os
import sys

from huggingface_hub import snapshot_download


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "Oriserve/Whisper-Hindi2Hinglish-Apex")
    print(f"[cache] downloading {model_id}", flush=True)
    path = snapshot_download(repo_id=model_id)
    print(f"[cache] done: {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[cache] FAILED: {exc}", file=sys.stderr, flush=True)
        raise
