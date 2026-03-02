#!/usr/bin/env bash
# setup.sh - Set up cozyvoice3-api in an isolated virtual environment
# Usage: bash setup.sh [--skip-model-download]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
COSYVOICE_DIR="$SCRIPT_DIR/CosyVoice"
PYTHON_BIN="python3"

# ─── helpers ────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

# ─── detect platform ────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Platform: $OS / $ARCH"

# ─── Python version check ───────────────────────────────────────────────────
PY_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version: $PY_VERSION"
[[ "$PY_VERSION" == "3.10" || "$PY_VERSION" == "3.11" || "$PY_VERSION" == "3.12" ]] \
  || warn "Recommended Python 3.10–3.12. Got $PY_VERSION — may have issues."

# ─── virtual environment ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at $VENV_DIR ..."
  $PYTHON_BIN -m venv "$VENV_DIR"
else
  info "Virtual environment already exists at $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

info "Upgrading pip / setuptools / wheel ..."
"$PIP" install --upgrade pip setuptools wheel --quiet

# ─── clone CosyVoice repo ────────────────────────────────────────────────────
if [[ ! -d "$COSYVOICE_DIR" ]]; then
  info "Cloning FunAudioLLM/CosyVoice ..."
  git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_DIR"
else
  info "CosyVoice repo already exists at $COSYVOICE_DIR"
fi

# initialise / update third-party submodules (WeTextProcessing, etc.)
info "Initialising submodules ..."
cd "$COSYVOICE_DIR"
git submodule update --init --recursive
cd "$SCRIPT_DIR"

# ─── install CosyVoice dependencies (platform-aware) ─────────────────────────
info "Installing PyTorch (CPU / MPS compatible) ..."
if [[ "$OS" == "Darwin" ]]; then
  # macOS: use default PyTorch (CPU + MPS), not the CUDA build
  "$PIP" install torch torchaudio --quiet
else
  # Linux: install CUDA-enabled PyTorch 2.3.x as CosyVoice requires
  "$PIP" install torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121 --quiet
fi

info "Installing CosyVoice Python requirements ..."
COSYVOICE_REQ="$COSYVOICE_DIR/requirements.txt"
if [[ -f "$COSYVOICE_REQ" ]]; then
  # filter out platform-incompatible packages on macOS
  if [[ "$OS" == "Darwin" ]]; then
    grep -v -E 'onnxruntime-gpu|deepspeed|tensorrt|nvidia|cuda' "$COSYVOICE_REQ" \
      > /tmp/cosyvoice_req_filtered.txt
    "$PIP" install -r /tmp/cosyvoice_req_filtered.txt --quiet || true
    info "Skipped GPU-only packages (onnxruntime-gpu, deepspeed, tensorrt) on macOS"
  else
    "$PIP" install -r "$COSYVOICE_REQ" --quiet || true
  fi
else
  warn "No requirements.txt found in CosyVoice — installing core deps manually"
  "$PIP" install transformers omegaconf hydra-core conformer \
    librosa soundfile grpcio fastapi uvicorn diffusers --quiet
fi

# install onnxruntime (CPU version for macOS, GPU version for Linux handled above)
if [[ "$OS" == "Darwin" ]]; then
  "$PIP" install onnxruntime --quiet
fi

# ─── install CosyVoice as editable package ───────────────────────────────────
info "Installing CosyVoice as editable package ..."
"$PIP" install -e "$COSYVOICE_DIR" --quiet 2>/dev/null || \
  info "No setup.py/pyproject.toml found — CosyVoice will be imported via PYTHONPATH"

# ─── install our API wrapper requirements ────────────────────────────────────
info "Installing cozyvoice3-api requirements ..."
"$PIP" install \
  fastapi \
  uvicorn[standard] \
  python-multipart \
  huggingface_hub \
  python-dotenv \
  aiofiles \
  --quiet

# ─── optional: download model weights ────────────────────────────────────────
if [[ "${1:-}" != "--skip-model-download" ]]; then
  MODEL_DIR="$SCRIPT_DIR/models/Fun-CosyVoice3-0.5B"
  if [[ ! -d "$MODEL_DIR" ]]; then
    info "Downloading Fun-CosyVoice3-0.5B model weights (~9.7 GB) ..."
    info "  This will take a while. Re-run with --skip-model-download to skip."
    "$PYTHON" - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
model_dir = os.path.join(os.path.dirname(os.path.abspath(".")), "models", "Fun-CosyVoice3-0.5B")
os.makedirs(model_dir, exist_ok=True)
snapshot_download(
    repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    local_dir=model_dir,
    ignore_patterns=["*.git*", "*.h5"],
)
print(f"Model downloaded to: {model_dir}")
PYEOF
  else
    info "Model weights already exist at $MODEL_DIR"
  fi
else
  info "Skipping model download (--skip-model-download)"
fi

# ─── env file ────────────────────────────────────────────────────────────────
if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  info "Created .env from .env.example — please fill in HF_TOKEN and other settings."
fi

# ─── done ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Activate venv:  source .venv/bin/activate"
echo "  Start server:   python server.py"
echo "  (or)            .venv/bin/python server.py"
echo "============================================================"
