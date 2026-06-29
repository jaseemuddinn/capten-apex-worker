FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir whisperx==3.3.1 --no-deps && \
    pip install --no-cache-dir \
    "faster-whisper>=1.1.0" \
    "ctranslate2>=4.4.0" \
    "pandas" \
    "nltk"

# Default whisperx __init__ imports transcribe/VAD → pyannote (not needed for align-only).
RUN python - <<'PY'
import site
import os

patch = "from .alignment import load_align_model, align\n"
for base in site.getsitepackages():
    init = os.path.join(base, "whisperx", "__init__.py")
    if os.path.isfile(init):
        with open(init, "w", encoding="utf-8") as f:
            f.write(patch)
        print("patched", init)
        break
else:
    raise SystemExit("whisperx __init__.py not found")
PY

RUN python -c "\
import torch; \
from whisperx.alignment import load_align_model, align; \
print('torch', torch.__version__, 'alignment import ok')"

ENV MODEL_ID=Oriserve/Whisper-Hindi2Hinglish-Apex
ENV ALIGN_LANGUAGE=hi
ENV ENABLE_ALIGNMENT=true
ENV HF_HOME=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
ENV RUNPOD_INIT_TIMEOUT=900

COPY cache_model.py .
RUN python cache_model.py

COPY handler.py .

CMD ["python", "-u", "handler.py"]
