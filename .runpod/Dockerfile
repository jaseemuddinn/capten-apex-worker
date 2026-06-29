FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# Conda python in the base image includes dev headers — do not install apt python3-dev (wrong ABI).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# MMS forced aligner — compiles a pybind11 C++ extension at install time.
RUN pip install --no-cache-dir pybind11 setuptools wheel && \
    pip install --no-cache-dir --no-deps \
    "https://github.com/MahmoudAshraf97/ctc-forced-aligner/archive/264e7a1f81bff9ff5e787a5537020c2ad0b0df02.tar.gz" && \
    pip install --no-cache-dir uroman Unidecode nltk

# Verify compiled extension + Python API (no GPU model load at build time).
RUN python -c "\
import torch; \
from ctc_forced_aligner.alignment_utils import forced_align; \
from ctc_forced_aligner import generate_emissions, preprocess_text, get_alignments, get_spans, postprocess_results; \
t, s = preprocess_text('hello world', romanize=False, language='hin', star_frequency='edges'); \
assert any(x != '<star>' for x in t), t; \
print('torch', torch.__version__, 'mms cpp ext ok')"

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
