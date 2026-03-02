"""
cosyvoice3-api  —  Gradio Web UI
=================================
Runs as a separate service; calls the FastAPI backend via HTTP.

Set the backend URL via env var:
  API_URL=http://api:8000   (docker-compose service name)
  API_URL=http://localhost:8000  (local dev)
"""

import os
import time
import tempfile
from pathlib import Path

import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = 2   # seconds between status polls
MAX_WAIT = 300      # give up after 5 minutes


# ── API helpers ───────────────────────────────────────────────────────────────

def _api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.ConnectionError:
        raise gr.Error(f"Cannot reach API at {API_URL}. Is the backend running?")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        raise gr.Error(f"API error: {detail}")


def fetch_voices() -> list[str]:
    try:
        data = _api("GET", "/voices").json()
        return [v["name"] for v in data.get("voices", [])]
    except Exception:
        return []


def upload_voice(audio_file, name: str, transcript: str):
    if audio_file is None:
        raise gr.Error("Please select an audio file.")
    name = name.strip()
    if not name:
        raise gr.Error("Please provide a name for this voice seed.")

    with open(audio_file, "rb") as f:
        resp = _api("POST", "/voices",
                    data={"name": name, "transcript": transcript},
                    files={"file": (Path(audio_file).name, f, "audio/wav")})

    result = resp.json()
    voices = fetch_voices()
    return (
        gr.update(choices=voices, value=voices[0] if voices else None),  # seed selector
        gr.update(choices=voices, value=voices[0] if voices else None),  # tts selector
        f"Voice **{result['voice_name']}** saved.",
    )


def delete_voice(voice_name: str):
    if not voice_name:
        raise gr.Error("Select a voice to delete.")
    _api("DELETE", f"/voices/{voice_name}")
    voices = fetch_voices()
    return (
        gr.update(choices=voices, value=voices[0] if voices else None),
        gr.update(choices=voices, value=voices[0] if voices else None),
        f"Voice **{voice_name}** deleted.",
    )


def refresh_voices():
    voices = fetch_voices()
    upd = gr.update(choices=voices, value=voices[0] if voices else None)
    return upd, upd


def synthesize(voice_name: str, text: str, upload_to_hf: bool):
    if not voice_name:
        raise gr.Error("Select a voice seed first.")
    if not text.strip():
        raise gr.Error("Enter some text to synthesize.")

    # ── enqueue ───────────────────────────────────────────────────────────────
    resp = _api("POST", "/tts/zero-shot",
                data={"voice_name": voice_name,
                      "text": text,
                      "upload_to_hf": str(upload_to_hf).lower()})
    job = resp.json()
    job_id = job["job_id"]
    queue_depth = job.get("queue_depth", "?")

    yield None, f"Queued (position {queue_depth}). Job ID: `{job_id}`", ""

    # ── poll ──────────────────────────────────────────────────────────────────
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            status_resp = _api("GET", f"/jobs/{job_id}")
        except Exception as e:
            yield None, f"Polling error: {e}", ""
            continue

        status = status_resp.json()
        state  = status["status"]

        if state == "queued":
            yield None, f"Queued... Job ID: `{job_id}`", ""

        elif state == "processing":
            yield None, f"Processing... Job ID: `{job_id}`", ""

        elif state == "done":
            audio_url = status.get("audio_url", "")
            hf_url    = status.get("hf_url") or ""

            # download the WAV to a temp file so Gradio can play it
            audio_resp = requests.get(f"{API_URL}{audio_url}", timeout=60, stream=True)
            audio_resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            for chunk in audio_resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()

            hf_msg = f"\nUploaded to HF: {hf_url}" if hf_url else ""
            yield tmp.name, f"Done! Job ID: `{job_id}`{hf_msg}", hf_url
            return

        elif state == "failed":
            yield None, f"Failed: {status.get('error', 'unknown error')}", ""
            return

    yield None, f"Timed out after {MAX_WAIT}s waiting for job `{job_id}`.", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    initial_voices = fetch_voices()

    with gr.Blocks(title="CosyVoice3 TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# CosyVoice3 TTS\nZero-shot voice cloning powered by **Fun-CosyVoice3-0.5B**.")

        # ── shared state: voice dropdown (synced across tabs) ─────────────────
        # We create the dropdowns per-tab but keep them in sync via events.

        with gr.Tabs():

            # ─────────────────────────────────────────────────────────────────
            # Tab 1 — Voice Seeds
            # ─────────────────────────────────────────────────────────────────
            with gr.Tab("Voice Seeds"):
                gr.Markdown("Upload a short audio clip (5–30 s recommended) as a voice seed.")

                with gr.Row():
                    with gr.Column():
                        seed_audio  = gr.Audio(label="Reference Audio", type="filepath", sources=["upload"])
                        seed_name   = gr.Textbox(label="Voice Name", placeholder="e.g. alice")
                        seed_transcript = gr.Textbox(
                            label="Transcript (recommended)",
                            placeholder="What is being said in the reference audio?",
                            lines=3,
                        )
                        with gr.Row():
                            upload_btn = gr.Button("Upload Voice Seed", variant="primary")
                            refresh_btn = gr.Button("Refresh List")

                    with gr.Column():
                        seed_selector = gr.Dropdown(
                            label="Existing Voice Seeds",
                            choices=initial_voices,
                            value=initial_voices[0] if initial_voices else None,
                            interactive=True,
                        )
                        delete_btn = gr.Button("Delete Selected Voice", variant="stop")
                        voice_msg  = gr.Markdown("")

                # internal: tts_selector is defined in Tab 2; sync happens via events below
                _tts_selector_placeholder = gr.Dropdown(visible=False)  # will be replaced

            # ─────────────────────────────────────────────────────────────────
            # Tab 2 — Synthesize
            # ─────────────────────────────────────────────────────────────────
            with gr.Tab("Synthesize"):
                with gr.Row():
                    with gr.Column(scale=1):
                        tts_selector = gr.Dropdown(
                            label="Voice Seed",
                            choices=initial_voices,
                            value=initial_voices[0] if initial_voices else None,
                            interactive=True,
                        )
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Type what you want the voice to say…",
                            lines=5,
                        )
                        hf_upload = gr.Checkbox(label="Upload result to HuggingFace Hub", value=False)
                        synth_btn = gr.Button("Synthesize", variant="primary")

                    with gr.Column(scale=1):
                        audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        status_md = gr.Markdown("Status: idle")
                        hf_url_md = gr.Markdown("")

        # ── events ────────────────────────────────────────────────────────────
        upload_btn.click(
            fn=upload_voice,
            inputs=[seed_audio, seed_name, seed_transcript],
            outputs=[seed_selector, tts_selector, voice_msg],
        )

        delete_btn.click(
            fn=delete_voice,
            inputs=[seed_selector],
            outputs=[seed_selector, tts_selector, voice_msg],
        )

        refresh_btn.click(
            fn=refresh_voices,
            inputs=[],
            outputs=[seed_selector, tts_selector],
        )

        synth_btn.click(
            fn=synthesize,
            inputs=[tts_selector, tts_text, hf_upload],
            outputs=[audio_out, status_md, hf_url_md],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        show_api=False,
    )
