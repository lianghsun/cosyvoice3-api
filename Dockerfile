# ─────────────────────────────────────────────────────────────────────────────
# cosyvoice3-api  Dockerfile
#
# Base: PyTorch 2.3.1 + CUDA 12.1 (matches CosyVoice3 requirements)
# Build clones CosyVoice source and installs all deps.
# Model weights (~9.75 GB) are expected at runtime via a mounted volume.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL maintainer="lianghsun"
LABEL description="FastAPI wrapper for Fun-CosyVoice3-0.5B zero-shot TTS"

# ── system packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      git-lfs \
      ffmpeg \
      libsndfile1 \
      sox \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── clone CosyVoice source (depth=1 to keep image lean) ─────────────────────
RUN git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /app/CosyVoice \
    && cd /app/CosyVoice \
    && git submodule update --init --recursive

# ── install CosyVoice Python dependencies ────────────────────────────────────
# torchaudio must match the torch version already in the base image (2.3.1)
RUN pip install --no-cache-dir torchaudio==2.3.1 \
      --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r /app/CosyVoice/requirements.txt || true
# CosyVoice's requirements.txt may include extras that fail; continue anyway.

# Install CosyVoice as an editable package if it has a setup file;
# otherwise PYTHONPATH handles it.
RUN pip install --no-cache-dir -e /app/CosyVoice 2>/dev/null || true

# ── install wrapper requirements ─────────────────────────────────────────────
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── copy application code ─────────────────────────────────────────────────────
COPY server.py /app/server.py

# ── runtime directories (will be overridden by volume mounts) ─────────────────
RUN mkdir -p /app/models /app/voices /app/outputs

# ── PYTHONPATH: make CosyVoice and its third-party deps importable ───────────
ENV PYTHONPATH="/app/CosyVoice:/app/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH:-}"

# ── environment defaults (overridable via .env / docker-compose env_file) ────
ENV MODEL_DIR=/app/models/Fun-CosyVoice3-0.5B \
    VOICES_DIR=/app/voices \
    OUTPUT_DIR=/app/outputs \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# ── healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
      || exit 1

# ── entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "server.py"]
