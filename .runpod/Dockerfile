FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch cu128: supports Ampere/Ada AND Blackwell (sm_120).
# Do NOT pin torch in requirements.txt — pip would install cu124 wheels
# that fail with "no kernel image is available for execution on the device".
RUN pip install --no-cache-dir \
    "torch>=2.7.0" "torchaudio>=2.7.0" \
    --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# WhisperX without replacing our cu128 torch build
RUN pip install --no-cache-dir whisperx==3.3.1 --no-deps && \
    pip install --no-cache-dir \
    "faster-whisper>=1.1.0" \
    "ctranslate2>=4.4.0" \
    pandas \
    nltk

COPY handler.py .

ENV MODEL_ID=Oriserve/Whisper-Hindi2Hinglish-Apex
ENV ALIGN_LANGUAGE=hi
ENV ENABLE_ALIGNMENT=true

CMD ["python", "-u", "handler.py"]
