"""
cozyvoice3-api  —  FastAPI server for Fun-CosyVoice3-0.5B
==========================================================
Endpoints:
  GET    /health                  server / model status + queue depth
  POST   /voices                  upload a voice seed
  GET    /voices                  list available voice seeds
  DELETE /voices/{name}           remove a voice seed
  POST   /tts/zero-shot           enqueue a TTS job → returns job_id immediately
  GET    /jobs/{job_id}           poll job status / result
  GET    /jobs                    list all jobs (recent first)
  GET    /audio/{filename}        download a generated WAV

Inference is processed by a single background worker (sequential, GPU-safe).
Multiple requests are queued and assigned job numbers so callers can poll.
"""

import io
import os
import sys
import uuid
import json
import logging
import asyncio
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import aiofiles
import torch
import torchaudio
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
COSYVOICE_DIR = BASE_DIR / "CosyVoice"
MODEL_DIR     = Path(os.getenv("MODEL_DIR", BASE_DIR / "models" / "Fun-CosyVoice3-0.5B"))
VOICES_DIR    = Path(os.getenv("VOICES_DIR", BASE_DIR / "voices"))
OUTPUT_DIR    = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "outputs"))

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
    for subdir in ["third_party/Matcha-TTS"]:
        p = COSYVOICE_DIR / subdir
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
else:
    logger.warning("CosyVoice directory not found at %s. Run setup.sh first.", COSYVOICE_DIR)

# ── model (loaded once at startup) ───────────────────────────────────────────
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
                load_jit=False,
                load_trt=False,
                load_vllm=False,
                fp16=False,
            )
            logger.info("Model loaded. Sample rate: %d Hz", _model.sample_rate)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            raise RuntimeError(f"Could not load CosyVoice3 model: {exc}") from exc
        return _model


# ── job queue ─────────────────────────────────────────────────────────────────
#
# _job_queue  — asyncio.Queue of job_id strings, consumed by _worker()
# _jobs       — dict[job_id, JobRecord]; in-memory; survives the request lifecycle
#
# JobRecord fields:
#   job_id, status, position, text, voice_name, upload_to_hf,
#   created_at, started_at, finished_at,
#   audio_filename, hf_url, error

_job_queue: asyncio.Queue = asyncio.Queue()
_jobs: dict[str, dict] = {}
_queue_counter = 0          # monotonically increasing position number


def _new_job(text: str, voice_name: str, upload_to_hf: bool) -> dict:
    global _queue_counter
    _queue_counter += 1
    return {
        "job_id":        uuid.uuid4().hex,
        "status":        "queued",          # queued | processing | done | failed
        "position":      _queue_counter,
        "queue_depth":   None,              # filled in by /tts/zero-shot
        "text":          text,
        "voice_name":    voice_name,
        "upload_to_hf":  upload_to_hf,
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "started_at":    None,
        "finished_at":   None,
        "audio_filename": None,
        "hf_url":        None,
        "error":         None,
    }


async def _worker():
    """Single background coroutine — processes one TTS job at a time."""
    logger.info("Job worker started.")
    while True:
        job_id = await _job_queue.get()
        job = _jobs.get(job_id)
        if job is None:
            _job_queue.task_done()
            continue

        job["status"]     = "processing"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("[job %s] started — voice=%s text='%.60s'",
                    job_id, job["voice_name"], job["text"])

        try:
            # ── resolve voice seed ────────────────────────────────────────────
            voice_dir = VOICES_DIR / job["voice_name"]
            meta_path = voice_dir / "meta.json"
            if not voice_dir.exists() or not meta_path.exists():
                raise ValueError(f"Voice '{job['voice_name']}' not found.")

            async with aiofiles.open(meta_path) as f:
                meta = json.loads(await f.read())

            audio_candidates = list(voice_dir.glob("seed.*"))
            if not audio_candidates:
                raise ValueError(f"No audio file for voice '{job['voice_name']}'.")
            seed_audio_path = audio_candidates[0]
            prompt_text = meta.get("transcript") or "You are a helpful assistant.<|endofprompt|>"

            # ── load model ────────────────────────────────────────────────────
            model = await get_model()

            # ── run inference (blocking → thread pool) ────────────────────────
            prompt_speech_16k = _load_audio_16k(seed_audio_path)

            def _run():
                chunks = []
                for chunk in model.inference_zero_shot(
                    tts_text=job["text"],
                    prompt_text=prompt_text,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=False,
                ):
                    chunks.append(chunk["tts_speech"])
                return torch.cat(chunks, dim=-1) if chunks else None

            loop = asyncio.get_event_loop()
            waveform = await loop.run_in_executor(None, _run)
            if waveform is None:
                raise RuntimeError("Model returned empty audio.")

            # ── save to disk ──────────────────────────────────────────────────
            ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"{job['voice_name']}_{ts}_{job_id[:6]}.wav"
            out_path = OUTPUT_DIR / filename
            wav_bytes = _audio_tensor_to_bytes(waveform, model.sample_rate)
            async with aiofiles.open(out_path, "wb") as f:
                await f.write(wav_bytes)
            job["audio_filename"] = filename
            logger.info("[job %s] audio saved: %s", job_id, filename)

            # ── optional HF upload ────────────────────────────────────────────
            if job["upload_to_hf"]:
                try:
                    hf_url = await _upload_to_hf(out_path, f"audio/{filename}")
                    job["hf_url"] = hf_url
                    logger.info("[job %s] uploaded to HF: %s", job_id, hf_url)
                except Exception as hf_exc:
                    logger.warning("[job %s] HF upload failed (audio still saved): %s", job_id, hf_exc)

            job["status"] = "done"

        except Exception as exc:
            logger.error("[job %s] failed: %s", job_id, exc, exc_info=True)
            job["status"] = "failed"
            job["error"]  = str(exc)

        finally:
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            _job_queue.task_done()


# ── helpers ───────────────────────────────────────────────────────────────────

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _validate_audio_filename(filename: str) -> str:
    p = Path(filename)
    if p.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format '{p.suffix}'. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}",
        )
    safe = "".join(c for c in p.stem if c.isalnum() or c in "-_.")
    return safe or uuid.uuid4().hex[:8]


def _load_audio_16k(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform


def _audio_tensor_to_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    torchaudio.save(buf, waveform, sample_rate, format="wav")
    buf.seek(0)
    return buf.read()


async def _upload_to_hf(local_path: Path, repo_path: str) -> str:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is not set.")
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID is not set.")
    from huggingface_hub import HfApi  # type: ignore
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, exist_ok=True, private=False)
    url = api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    )
    return url


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CosyVoice3 TTS API",
    description=(
        "Synthesize speech using Fun-CosyVoice3-0.5B. "
        "Requests are queued and processed sequentially by a single GPU worker. "
        "POST /tts/zero-shot returns a job_id immediately; poll GET /jobs/{job_id} for status."
    ),
    version="0.2.0",
)


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
async def health():
    queued = sum(1 for j in _jobs.values() if j["status"] == "queued")
    return {
        "status":           "ok",
        "model_loaded":     _model is not None,
        "model_dir":        str(MODEL_DIR),
        "hf_upload_enabled": bool(HF_TOKEN and HF_REPO_ID),
        "queue_depth":      queued,
        "total_jobs":       len(_jobs),
    }


# ── /voices ───────────────────────────────────────────────────────────────────
@app.post("/voices", tags=["voices"], summary="Upload a voice seed")
async def upload_voice(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    transcript: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=422, detail="File must have a filename.")

    safe_stem  = _validate_audio_filename(file.filename)
    voice_name = "".join(c for c in (name.strip() if name else safe_stem) if c.isalnum() or c in "-_")
    if not voice_name:
        voice_name = uuid.uuid4().hex[:8]

    voice_dir  = VOICES_DIR / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50 MB).")

    audio_path = voice_dir / f"seed{Path(file.filename).suffix.lower()}"
    async with aiofiles.open(audio_path, "wb") as f:
        await f.write(raw)

    meta = {
        "name":              voice_name,
        "original_filename": file.filename,
        "transcript":        transcript or "",
        "audio_file":        audio_path.name,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    async with aiofiles.open(voice_dir / "meta.json", "w") as f:
        await f.write(json.dumps(meta, ensure_ascii=False, indent=2))

    logger.info("Voice seed saved: %s", voice_name)
    return {"voice_name": voice_name, "transcript": transcript or ""}


@app.get("/voices", tags=["voices"], summary="List voice seeds")
async def list_voices():
    voices = []
    if VOICES_DIR.exists():
        for voice_dir in sorted(VOICES_DIR.iterdir()):
            meta_path = voice_dir / "meta.json"
            if meta_path.exists():
                async with aiofiles.open(meta_path) as f:
                    voices.append(json.loads(await f.read()))
    return {"voices": voices}


@app.delete("/voices/{voice_name}", tags=["voices"], summary="Delete a voice seed")
async def delete_voice(voice_name: str):
    voice_dir = VOICES_DIR / voice_name
    if not voice_dir.exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")
    shutil.rmtree(voice_dir)
    logger.info("Voice seed deleted: %s", voice_name)
    return {"deleted": voice_name}


# ── /tts/zero-shot ────────────────────────────────────────────────────────────
@app.post("/tts/zero-shot", tags=["tts"], summary="Enqueue a TTS synthesis job")
async def tts_zero_shot(
    text: str = Form(..., description="Text to synthesize."),
    voice_name: str = Form(..., description="Name of an uploaded voice seed."),
    upload_to_hf: bool = Form(False, description="Upload generated WAV to HuggingFace Hub."),
):
    """
    Enqueues a TTS job. Returns `job_id` and current `queue_depth` immediately.
    Poll `GET /jobs/{job_id}` to get status and, when done, the audio download URL.
    """
    # validate voice exists before queuing
    voice_dir = VOICES_DIR / voice_name
    if not voice_dir.exists() or not (voice_dir / "meta.json").exists():
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found.")

    job = _new_job(text=text, voice_name=voice_name, upload_to_hf=upload_to_hf)
    job["queue_depth"] = _job_queue.qsize() + 1   # position in queue at submission time
    _jobs[job["job_id"]] = job
    await _job_queue.put(job["job_id"])

    logger.info("[job %s] queued (depth=%d)", job["job_id"], job["queue_depth"])
    return {
        "job_id":      job["job_id"],
        "status":      "queued",
        "queue_depth": job["queue_depth"],
    }


# ── /jobs ─────────────────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}", tags=["jobs"], summary="Get job status / result")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    result = dict(job)
    # attach download URL when done
    if job["status"] == "done" and job["audio_filename"]:
        result["audio_url"] = f"/audio/{job['audio_filename']}"
    return result


@app.get("/jobs", tags=["jobs"], summary="List all jobs (most recent first)")
async def list_jobs(limit: int = 50):
    sorted_jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    jobs = []
    for job in sorted_jobs[:limit]:
        entry = dict(job)
        if job["status"] == "done" and job["audio_filename"]:
            entry["audio_url"] = f"/audio/{job['audio_filename']}"
        jobs.append(entry)
    return {"jobs": jobs, "total": len(_jobs)}


# ── /audio ────────────────────────────────────────────────────────────────────
@app.get("/audio/{filename}", tags=["audio"], summary="Download a generated WAV file")
async def get_audio(filename: str):
    # prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(path, media_type="audio/wav", filename=filename)


# ── startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    # launch the single background worker
    asyncio.create_task(_worker())

    # pre-load model
    if MODEL_DIR.exists():
        logger.info("Pre-loading model at startup …")
        try:
            await get_model()
        except Exception as exc:
            logger.warning("Model pre-load failed: %s — will retry on first job.", exc)
    else:
        logger.warning("MODEL_DIR %s does not exist. Run setup.sh to download weights.", MODEL_DIR)


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
