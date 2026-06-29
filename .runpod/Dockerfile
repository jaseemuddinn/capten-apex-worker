FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# WhisperX 3.3 + pyannote 3.3 need torchaudio.AudioMetaData (removed in torchaudio 2.9+).
# pyannote.audio may upgrade torch — re-pin cu128 2.7.1 stack after all ML deps.
RUN pip install --no-cache-dir whisperx==3.3.1 --no-deps && \
    pip install --no-cache-dir \
    "faster-whisper>=1.1.0" \
    "ctranslate2>=4.4.0" \
    "pandas" \
    "nltk" \
    "omegaconf>=2.3.0" \
    "pyannote.audio==3.3.2"

RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -c "\
import torch, torchaudio, pyannote.audio, whisperx; \
assert hasattr(torchaudio, 'AudioMetaData'), torchaudio.__version__; \
v = torch.__version__.split('+')[0]; \
major, minor = (int(x) for x in v.split('.')[:2]); \
assert (major, minor) == (2, 7), torch.__version__; \
print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'pyannote ok')"

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
