"""
cozyvoice3-api  —  FastAPI server for Fun-CosyVoice3-0.5B
==========================================================
Endpoints:
  POST   /voices                  upload a voice seed (wav, 16 kHz mono recommended)
  GET    /voices                  list available voice seeds
  DELETE /voices/{name}           remove a voice seed
  POST   /tts/zero-shot           synthesize speech from a voice seed + text
  GET    /health                  health / model status

Generated audio can optionally be uploaded to HuggingFace Hub (controlled by
the `upload_to_hf` form field or the HF_TOKEN env var).
"""

import io
import os
import sys
import uuid
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import aiofiles
import torch
import torchaudio
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

# ── load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
COSYVOICE_DIR = BASE_DIR / "CosyVoice"

MODEL_DIR  = Path(os.getenv("MODEL_DIR", BASE_DIR / "models" / "Fun-CosyVoice3-0.5B"))
VOICES_DIR = Path(os.getenv("VOICES_DIR", BASE_DIR / "voices"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))

VOICES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── HuggingFace config ───────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN", "")
HF_REPO_ID   = os.getenv("HF_REPO_ID", "")
HF_REPO_TYPE = os.getenv("HF_REPO_TYPE", "dataset")

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("cozyvoice3-api")

# ── add CosyVoice to sys.path ─────────────────────────────────────────────────
if COSYVOICE_DIR.exists():
    sys.path.insert(0, str(COSYVOICE_DIR))
    # also add third_party dirs that CosyVoice expects
    for subdir in ["third_party/Matcha-TTS"]:
        p = COSYVOICE_DIR / subdir
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
else:
    logger.warning(
        "CosyVoice directory not found at %s. "
        "Run setup.sh first.",
        COSYVOICE_DIR,
    )

# ── lazy-load the model (done on first request or at startup) ─────────────────
_model = None
_model_lock = asyncio.Lock()


async def get_model():
    global _model
    if _model is not None:
        return _model
    async with _model_lock:
        if _model is not None:
            return _model
        logger.info("Loading CosyVoice3 model from %s …", MODEL_DIR)
        try:
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore
            _model = AutoModel(
                model_dir=str(MODEL_DIR),
                load_jit=False,    # JIT optional — skip for portability
                load_trt=False,    # TRT: Linux only
                load_vllm=False,   # vLLM internal accelerator — optional
                fp16=False,        # fp16 requires CUDA
            )
            logger.info("Model loaded successfully. Sample rate: %d Hz", _model.sample_rate)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise RuntimeError(f"Could not load CosyVoice3 model: {exc}") from exc
        return _model


# ── helpers ───────────────────────────────────────────────────────────────────

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _validate_audio_filename(filename: str) -> str:
    """Return the safe file stem or raise HTTPException."""
    p = Path(filename)
    if p.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{p.suffix}'. "
                   f"Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
        )
    # sanitise: keep only alphanumeric, dash, underscore, dot
    safe = "".join(c for c in p.stem if c.isalnum() or c in "-_.")
    if not safe:
        safe = uuid.uuid4().hex[:8]
    return safe


def _load_audio_16k(path: Path) -> torch.Tensor:
    """Load audio, resample to 16 kHz, convert to mono, return (1, T) tensor."""
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform


def _audio_tensor_to_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """Encode waveform tensor as WAV bytes."""
    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sample_rate, format="wav")
    buf.seek(0)
    return buf.read()


async def _upload_to_hf(local_path: Path, repo_path: str) -> str:
    """Upload a file to HuggingFace Hub and return the URL."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set — cannot upload to HuggingFace.")
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID is not set — cannot upload to HuggingFace.")
    from huggingface_hub import HfApi  # type: ignore
    api = HfApi(token=HF_TOKEN)
    # ensure the repo exists (creates it if missing, no-op if it already exists)
    api.create_repo(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        exist_ok=True,
        private=False,
    )
    url = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    )
    logger.info("Uploaded %s → %s", local_path.name, url)
    return url


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CosyVoice3 TTS API",
    description=(
        "Synthesize speech using Fun-CosyVoice3-0.5B. "
        "Upload a voice seed (reference audio) and provide text to generate audio. "
        "Optionally uploads results to HuggingFace Hub."
    ),
    version="0.1.0",
)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health():
    model_loaded = _model is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_dir": str(MODEL_DIR),
        "hf_upload_enabled": bool(HF_TOKEN and HF_REPO_ID),
    }


# ── /voices ───────────────────────────────────────────────────────────────────
@app.post("/voices", tags=["voices"], summary="Upload a voice seed")
async def upload_voice(
    file: UploadFile = File(..., description="Reference audio file (WAV preferred, 16 kHz mono)"),
    name: Optional[str] = Form(None, description="Friendly name (defaults to filename stem)"),
    transcript: Optional[str] = Form(
        None,
        description=(
            "Transcript of the reference audio. "
            "For CosyVoice3, this is used as the 'prompt_text'. "
            "Leave empty to use a generic placeholder."
        ),
    ),
):
    """Store a voice seed on disk for later use in /tts/zero-shot."""
    if file.filename is None or file.filename == "":
        raise HTTPException(status_code=422, detail="File must have a filename.")

    safe_stem = _validate_audio_filename(file.filename)
    voice_name = name.strip() if name else safe_stem
    # sanitise voice name
    voice_name = "".join(c for c in voice_name if c.isalnum() or c in "-_")
    if not voice_name:
        voice_name = uuid.uuid4().hex[:8]

    voice_dir = VOICES_DIR / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)

    # save audio
    audio_path = voice_dir / f"seed{Path(file.filename).suffix.lower()}"
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")
    async with aiofiles.open(audio_path, "wb") as f:
        await f.write(raw)

    # save metadata
    meta = {
        "name": voice_name,
        "original_filename": file.filename,
        "transcript": transcript or "",
        "audio_file": audio_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    async with aiofiles.open(voice_dir / "meta.json", "w") as f:
        await f.write(json.dumps(meta, ensure_ascii=False, indent=2))

    logger.info("Voice seed saved: %s → %s", voice_name, audio_path)
    return {"voice_name": voice_name, "audio_file": str(audio_path), "transcript": transcript or ""}


@app.get("/voices", tags=["voices"], summary="List available voice seeds")
async def list_voices():
    voices = []
    for voice_dir in sorted(VOICES_DIR.iterdir()):
        meta_path = voice_dir / "meta.json"
        if meta_path.exists():
            async with aiofiles.open(meta_path) as f:
                meta = json.loads(await f.read())
            voices.append(meta)
    return {"voices": voices}


@app.delete("/voices/{voice_name}", tags=["voices"], summary="Delete a voice seed")
async def delete_voice(voice_name: str):
    voice_dir = VOICES_DIR / voice_name
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")
    import shutil
    shutil.rmtree(voice_dir)
    logger.info("Voice seed deleted: %s", voice_name)
    return {"deleted": voice_name}


# ── /tts/zero-shot ────────────────────────────────────────────────────────────
@app.post("/tts/zero-shot", tags=["tts"], summary="Synthesize speech (zero-shot voice cloning)")
async def tts_zero_shot(
    text: str = Form(..., description="Text to synthesize."),
    voice_name: str = Form(..., description="Name of a previously uploaded voice seed."),
    upload_to_hf: bool = Form(False, description="Upload the generated WAV to HuggingFace Hub."),
    stream: bool = Form(False, description="Stream audio chunks as they are generated."),
):
    """
    Generate speech using zero-shot voice cloning.

    1. Loads the reference audio for `voice_name`.
    2. Runs `cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav)`.
    3. Returns the generated WAV (or streams it chunk by chunk).
    4. Optionally uploads the result to HuggingFace Hub.
    """
    # ── validate voice seed ──────────────────────────────────────────────────
    voice_dir  = VOICES_DIR / voice_name
    meta_path  = voice_dir / "meta.json"
    if not voice_dir.exists() or not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")

    async with aiofiles.open(meta_path) as f:
        meta = json.loads(await f.read())

    audio_candidates = list(voice_dir.glob("seed.*"))
    if not audio_candidates:
        raise HTTPException(status_code=500, detail=f"No audio file found for voice '{voice_name}'.")
    seed_audio_path = audio_candidates[0]

    prompt_text = meta.get("transcript") or "You are a helpful assistant.<|endofprompt|>"

    # ── load model ───────────────────────────────────────────────────────────
    try:
        model = await get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # ── run inference in a thread (blocking) ──────────────────────────────────
    prompt_speech_16k = _load_audio_16k(seed_audio_path)

    def _run_inference():
        chunks = []
        for chunk in model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_16k,
            stream=stream,
        ):
            chunks.append(chunk["tts_speech"])
        if not chunks:
            return None
        return torch.cat(chunks, dim=-1)

    loop = asyncio.get_event_loop()
    try:
        waveform = await loop.run_in_executor(None, _run_inference)
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    if waveform is None:
        raise HTTPException(status_code=500, detail="Model returned no audio.")

    sample_rate = model.sample_rate
    wav_bytes   = _audio_tensor_to_bytes(waveform, sample_rate)

    # ── save locally ─────────────────────────────────────────────────────────
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_name = f"{voice_name}_{ts}_{uuid.uuid4().hex[:6]}.wav"
    out_path = OUTPUT_DIR / out_name
    async with aiofiles.open(out_path, "wb") as f:
        await f.write(wav_bytes)
    logger.info("Generated audio saved: %s", out_path)

    # ── optionally upload to HF ──────────────────────────────────────────────
    hf_url: Optional[str] = None
    if upload_to_hf:
        if not HF_TOKEN or not HF_REPO_ID:
            raise HTTPException(
                status_code=503,
                detail="HF_TOKEN and HF_REPO_ID must be set in .env to upload to HuggingFace.",
            )
        try:
            repo_path = f"audio/{out_name}"
            hf_url = await _upload_to_hf(out_path, repo_path)
        except Exception as exc:
            logger.error("HuggingFace upload failed: %s", exc)
            # don't fail the request — just return without hf_url
            hf_url = None

    # ── return audio ─────────────────────────────────────────────────────────
    headers = {
        "X-Generated-File": out_name,
        "X-Sample-Rate": str(sample_rate),
    }
    if hf_url:
        headers["X-HF-URL"] = hf_url

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers=headers,
    )


# ── startup: pre-load model ───────────────────────────────────────────────────
@app.on_event("startup")
async def _preload_model():
    if MODEL_DIR.exists():
        logger.info("Pre-loading model at startup …")
        try:
            await get_model()
        except Exception as exc:
            logger.warning("Model pre-load failed: %s — will retry on first request.", exc)
    else:
        logger.warning(
            "MODEL_DIR %s does not exist. "
            "Run setup.sh to download model weights.",
            MODEL_DIR,
        )


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )
