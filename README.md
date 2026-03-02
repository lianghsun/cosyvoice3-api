# cosyvoice3-api

A **FastAPI** service that wraps [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) — a state-of-the-art, streaming-capable, zero-shot voice cloning TTS model by FunAudioLLM.

Upload a voice seed (any reference audio clip), send text, and get back a synthesized WAV.
Generated audio can be automatically uploaded to a **HuggingFace Hub** dataset or model repo.

> **Target platform:** Linux with NVIDIA GPU.
> macOS / CPU inference works but is significantly slower.

---

## Features

- Zero-shot voice cloning via REST API
- Voice seed management (upload, list, delete)
- Optional automatic upload of generated audio to HuggingFace Hub
- Fully isolated Python virtual environment (no system Python pollution)
- Platform-aware setup (skips Linux-only GPU packages on macOS)

---

## Quick Start

### 1. Clone this repo

```bash
git clone https://github.com/lianghsun/cosyvoice3-api.git
cd cosyvoice3-api
```

### 2. Run the setup script

```bash
bash setup.sh
```

This will:
1. Create a Python virtual environment at `.venv/`
2. Clone the [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) source repo
3. Install all dependencies (GPU-only packages are skipped on macOS)
4. Download the **Fun-CosyVoice3-0.5B-2512** model weights (~9.7 GB)

To skip the model download (e.g. you already have the weights):

```bash
bash setup.sh --skip-model-download
```

### 3. Configure environment

```bash
cp .env.example .env
# edit .env and fill in your values
```

| Variable | Required | Description |
|---|---|---|
| `MODEL_DIR` | No | Path to model weights (default: `./models/Fun-CosyVoice3-0.5B`) |
| `HF_TOKEN` | For HF upload | Your HuggingFace write token — [get one here](https://huggingface.co/settings/tokens) |
| `HF_REPO_ID` | For HF upload | Destination repo, e.g. `your-username/tts-outputs` |
| `HF_REPO_TYPE` | No | `dataset` (default) or `model` |
| `HOST` | No | Bind address (default: `0.0.0.0`) |
| `PORT` | No | Port (default: `8000`) |

### 4. Start the server

```bash
.venv/bin/python server.py
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### `GET /health`
Returns server status, model load state, and HF upload configuration.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_dir": "./models/Fun-CosyVoice3-0.5B",
  "hf_upload_enabled": true
}
```

---

### `POST /voices`
Upload a voice seed (reference audio) for later use.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | ✅ | Reference audio (WAV recommended, 16 kHz mono; max 50 MB) |
| `name` | string | No | Friendly name (defaults to filename stem) |
| `transcript` | string | No | Text content of the reference audio (improves voice quality) |

```bash
curl -X POST http://localhost:8000/voices \
  -F "file=@my_voice.wav" \
  -F "name=alice" \
  -F "transcript=Hi, I'm Alice. This is my voice sample."
```

---

### `GET /voices`
List all uploaded voice seeds.

```bash
curl http://localhost:8000/voices
```

---

### `DELETE /voices/{name}`
Delete a voice seed by name.

```bash
curl -X DELETE http://localhost:8000/voices/alice
```

---

### `POST /tts/zero-shot`
Synthesize speech using zero-shot voice cloning.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ | Text to synthesize |
| `voice_name` | string | ✅ | Name of a previously uploaded voice seed |
| `upload_to_hf` | bool | No | Upload generated WAV to HuggingFace Hub (default: `false`) |
| `stream` | bool | No | Use streaming inference internally (default: `false`) |

Returns the generated audio as `audio/wav`.
Response headers include `X-Generated-File`, `X-Sample-Rate`, and (if uploaded) `X-HF-URL`.

```bash
# Synthesize and save locally
curl -X POST http://localhost:8000/tts/zero-shot \
  -F "text=Hello, this is a test of zero-shot voice cloning." \
  -F "voice_name=alice" \
  --output result.wav

# Synthesize and upload to HuggingFace
curl -X POST http://localhost:8000/tts/zero-shot \
  -F "text=Hello world!" \
  -F "voice_name=alice" \
  -F "upload_to_hf=true" \
  --output result.wav \
  --dump-header -  # shows X-HF-URL in response headers
```

---

## Architecture

CosyVoice3 is a **3-stage TTS pipeline**, not a plain language model:

```
Text ──► LLM (Qwen2) ──► Speech Tokens ──► Flow Matching DiT ──► Mel Spectrogram ──► HiFi-GAN ──► WAV
                                              ▲
                              Reference Audio (voice seed)
```

### Why vLLM server mode is not possible

CosyVoice3 registers a custom `CosyVoice2ForCausalLM` class with vLLM's `ModelRegistry`
to accelerate the *internal* speech token generation step. This class generates speech tokens
(vocabulary size ~6561), not text tokens — it is fundamentally incompatible with vLLM's
OpenAI-compatible serving endpoint (`/v1/chat/completions`, `/v1/audio/speech`, etc.).

vLLM can only be used as an **in-process accelerator** via `AutoModel(load_vllm=True, ...)`,
which is what `server.py` does (disabled by default, can be enabled for Linux + GPU deployments).

---

## Enabling GPU Acceleration (Linux + NVIDIA GPU)

Edit `server.py`, find the `get_model()` function, and change the flags:

```python
_model = AutoModel(
    model_dir=str(MODEL_DIR),
    load_jit=False,
    load_trt=True,    # ← TensorRT (~4x faster, requires tensorrt-cu12 ≥ 10.x)
    load_vllm=True,   # ← vLLM LLM accelerator (requires vLLM 0.11+)
    fp16=True,        # ← FP16 inference (requires CUDA)
)
```

Install vLLM (in the venv) if using `load_vllm=True`:

```bash
.venv/bin/pip install vllm==0.11.0
```

---

## Production Deployment

Use a single Uvicorn worker (to avoid loading the model multiple times into VRAM):

```bash
.venv/bin/pip install gunicorn
.venv/bin/gunicorn server:app \
  -w 1 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

---

## Model Information

| Item | Detail |
|---|---|
| Model | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
| Source repo | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| Sample rate | 24 kHz |
| Total size | ~9.75 GB |
| License | Apache 2.0 |

---

## License

This wrapper is released under the **MIT License**.
The underlying CosyVoice3 model and source code are licensed under **Apache 2.0** by FunAudioLLM.
