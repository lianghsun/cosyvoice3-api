# CLAUDE.md — Project Context & Architecture Notes

## Project Overview

This repo wraps **Fun-CosyVoice3-0.5B** (by FunAudioLLM) into a production-ready FastAPI
service that allows clients to:

1. Upload voice seeds (reference audio + optional transcript)
2. Request zero-shot TTS synthesis for arbitrary text
3. Automatically upload the generated WAV to a HuggingFace Hub dataset/model repo

The intended deployment target is a **Linux server with an NVIDIA GPU**.
macOS/CPU inference is possible but slow and not accelerated.

---

## Key Architectural Decision: vLLM Cannot Be Used as a Server

**Investigation result:** CosyVoice3 is NOT compatible with `vllm serve` /
`python -m vllm.entrypoints.openai.api_server`.

Reasons:
- CosyVoice3 is a **3-component pipeline** (LLM → Flow Matching DiT → HiFi-GAN vocoder),
  not a plain language model.
- The LLM sub-component (`CosyVoice2ForCausalLM`) generates *speech tokens* (vocab ~6561),
  not text tokens. This is incompatible with vLLM's OpenAI-compatible chat/completion API.
- vLLM is used *internally* as an inference accelerator for the LLM sub-component via
  `ModelRegistry.register_model("CosyVoice2ForCausalLM", ...)` and `load_vllm=True` in
  `AutoModel(...)`. It is never exposed as a network endpoint.
- The only supported acceleration modes are:
  | Flag            | Backend        | Platform     |
  |-----------------|----------------|--------------|
  | `load_vllm=True`| vLLM 0.11+     | Linux/GPU    |
  | `load_trt=True` | TensorRT 10.x  | Linux/GPU    |
  | `load_jit=True` | TorchScript    | Any          |
  | (default)       | PyTorch eager  | Any          |

**Conclusion:** Inference must be driven by the CosyVoice Python API inside our own
FastAPI server. This is what `server.py` does.

---

## CosyVoice3 Pipeline Components

| Component            | File(s)                              | Size  | Role                          |
|----------------------|--------------------------------------|-------|-------------------------------|
| LLM (Qwen2-based)    | `llm.pt`, `llm.rl.pt`               | 2 GB  | Text → discrete speech tokens |
| Flow Matching DiT    | `flow.pt`, `flow.decoder.*.onnx`    | 1.3 GB| Speech tokens → mel spectrogram|
| HiFi-GAN Vocoder     | `hift.pt`                            | 83 MB | Mel spectrogram → 24 kHz WAV  |
| Speech Tokenizer     | `speech_tokenizer_v3.batch.onnx`    | 969 MB| Encode reference audio         |
| Speaker Embedder     | `campplus.onnx`                      | 28 MB | Speaker embedding (CAMPPlus)  |

Total download: **~9.75 GB**
Model repo: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` on HuggingFace

---

## Project File Structure

```
cosyvoice3-api/
├── CLAUDE.md                ← this file
├── README.md                ← user-facing documentation
├── setup.sh                 ← bootstrap script (venv + CosyVoice clone + deps + model download)
├── server.py                ← FastAPI application (with async job queue)
├── gradio_app.py            ← Gradio web UI (polls FastAPI, plays audio)
├── requirements.txt         ← wrapper-layer Python deps
├── .env.example             ← environment variable template
├── .gitignore
├── Dockerfile               ← PyTorch 2.3.1 + CUDA 12.1; CosyVoice cloned at build time
├── Dockerfile.gradio        ← python:3.11-slim + gradio + requests (no GPU)
├── docker-compose.yml       ← api (GPU) + gradio + model-downloader (profile: download)
├── docker-compose.cpu.yml   ← CPU-only override
├── .dockerignore
│
├── .venv/               ← created by setup.sh (gitignored)
├── CosyVoice/           ← cloned by setup.sh or in Docker build (gitignored locally)
├── models/              ← model weights; bind-mounted into Docker container (gitignored)
├── voices/              ← uploaded voice seeds at runtime; named Docker volume (gitignored)
└── outputs/             ← generated WAV files at runtime; named Docker volume (gitignored)
```

---

## API Surface

| Method | Path                  | Description                                                      |
|--------|-----------------------|------------------------------------------------------------------|
| GET    | `/health`             | Server status, model load state, HF upload config, queue depth   |
| POST   | `/voices`             | Upload voice seed (audio + optional transcript)                  |
| GET    | `/voices`             | List all stored voice seeds                                      |
| DELETE | `/voices/{name}`      | Delete a voice seed                                              |
| POST   | `/tts/zero-shot`      | Enqueue TTS job → returns `job_id` + `queue_depth` immediately   |
| GET    | `/jobs/{job_id}`      | Poll job status (`queued/processing/done/failed`) + result URL   |
| GET    | `/jobs`               | List recent jobs (most recent first, limit param)                |
| GET    | `/audio/{filename}`   | Download a generated WAV file                                    |

## Job Queue Design

- `POST /tts/zero-shot` validates input and pushes to an `asyncio.Queue`, returns immediately
- A single background coroutine (`_worker`) processes one job at a time (GPU-safe)
- In-memory `_jobs` dict holds all job state (survives request lifecycle, lost on restart)
- Clients poll `GET /jobs/{job_id}` every ~2 s; Gradio handles this automatically

---

## Environment Variables (`.env`)

| Variable       | Required | Default                        | Description                        |
|----------------|----------|--------------------------------|------------------------------------|
| `MODEL_DIR`    | No       | `./models/Fun-CosyVoice3-0.5B` | Path to downloaded model weights   |
| `HF_TOKEN`     | For upload | —                            | HuggingFace write token            |
| `HF_REPO_ID`   | For upload | —                            | e.g. `username/tts-outputs`        |
| `HF_REPO_TYPE` | No       | `dataset`                      | `dataset` or `model`               |
| `HOST`         | No       | `0.0.0.0`                      | Server bind address                |
| `PORT`         | No       | `8000`                         | Server port                        |
| `VOICES_DIR`   | No       | `./voices`                     | Where to store voice seed files    |
| `OUTPUT_DIR`   | No       | `./outputs`                    | Where to store generated WAVs      |

---

## Deployment Notes (Linux + GPU)

```bash
# One-time setup
bash setup.sh                    # creates .venv, clones CosyVoice, downloads model

# Optional: enable vLLM acceleration (requires vLLM 0.11+ installed separately)
# In server.py → get_model(), set:
#   load_vllm=True
#   fp16=True

# Optional: enable TensorRT (requires tensorrt-cu12 ≥ 10.x)
# In server.py → get_model(), set:
#   load_trt=True

# Start server
source .venv/bin/activate
python server.py
# or with gunicorn for production:
# gunicorn server:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

> **Note on worker count:** Keep `-w 1` (single worker). The model is loaded once into
> GPU memory at startup; multiple workers would each load a full copy, exhausting VRAM.

---

## Next Steps / TODO

- [ ] Add GPU acceleration flags (`load_vllm`, `load_trt`, `fp16`) as env vars so they
      can be toggled without editing `server.py`
- [ ] Add `POST /tts/instruct` endpoint for emotion/style-controlled synthesis
      (`inference_instruct2`)
- [ ] Add `POST /tts/cross-lingual` endpoint
- [ ] Add request queue / concurrency limiter (model is not thread-safe under concurrent load)
- [ ] Add auth (API key header) for production use
- [ ] GitHub Actions CI for linting
