FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential git python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MMS forced aligner (Latin vocabulary — matches Apex Hinglish output).
# Needs git + python3-dev to compile the pybind11 C++ extension.
# --no-deps keeps the base image torch stack intact.
RUN pip install --no-cache-dir pybind11 && \
    pip install --no-cache-dir --no-deps \
    "https://github.com/MahmoudAshraf97/ctc-forced-aligner/archive/264e7a1f81bff9ff5e787a5537020c2ad0b0df02.tar.gz" && \
    pip install --no-cache-dir uroman Unidecode nltk

RUN python -c "\
import torch; \
from ctc_forced_aligner import load_alignment_model, generate_emissions, preprocess_text; \
print('torch', torch.__version__, 'mms align import ok')"

ENV MODEL_ID=Oriserve/Whisper-Hindi2Hinglish-Apex
ENV ALIGN_MODEL=MahmoudAshraf/mms-300m-1130-forced-aligner
ENV ALIGN_LANGUAGE=hin
ENV ENABLE_ALIGNMENT=true
ENV HF_HOME=/root/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
ENV RUNPOD_INIT_TIMEOUT=900

COPY cache_model.py .
RUN python cache_model.py

COPY handler.py .

CMD ["python", "-u", "handler.py"]
