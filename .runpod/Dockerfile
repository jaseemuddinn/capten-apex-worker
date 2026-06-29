FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install cu128 stack LAST — requirements must not overwrite this with base-image torch 2.4.
RUN pip uninstall -y torch torchvision torchaudio 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -c "\
import torch; \
v = torch.__version__.split('+')[0]; \
major, minor = (int(x) for x in v.split('.')[:2]); \
assert (major, minor) >= (2, 7), f'Expected torch>=2.7, got {torch.__version__}'; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
arch = torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []; \
print('arch_list', arch)"

RUN pip install --no-cache-dir whisperx==3.3.1 --no-deps && \
    pip install --no-cache-dir \
    "faster-whisper>=1.1.0" \
    "ctranslate2>=4.4.0" \
    "pandas" \
    "nltk" \
    "omegaconf>=2.3.0" \
    "pyannote.audio==3.3.2"

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
