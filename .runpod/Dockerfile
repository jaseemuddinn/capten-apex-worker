FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "torch>=2.7.0" "torchaudio>=2.7.0" \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir whisperx==3.3.1 --no-deps && \
    pip install --no-cache-dir \
    "faster-whisper>=1.1.0" \
    "ctranslate2>=4.4.0" \
    pandas \
    nltk

ENV MODEL_ID=Oriserve/Whisper-Hindi2Hinglish-Apex
ENV ALIGN_LANGUAGE=hi
ENV ENABLE_ALIGNMENT=true

RUN python -c "\
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; \
import os; \
mid = os.environ['MODEL_ID']; \
print('Caching', mid); \
AutoProcessor.from_pretrained(mid); \
AutoModelForSpeechSeq2Seq.from_pretrained(mid, use_safetensors=True); \
print('Apex model cached')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
