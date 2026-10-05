"""
app.py — Hugging Face Spaces entry point.

This file wraps our FastAPI app so it runs on Hugging Face's
free Gradio SDK (port 7860). The full custom UI is preserved.
Gradio is mounted at /gradio-info as a sub-path only.
"""

import os
import sys
import subprocess

# Auto-set demo mode for all public deployments
os.environ.setdefault("APP_MODE", "demo")

sys.path.insert(0, os.path.dirname(__file__))

# ── Install face_recognition WITHOUT dlib dependency ─────────────────────────
# dlib-bin (pre-compiled) is already installed from requirements.txt.
# Using --no-deps skips face_recognition's dlib source compilation entirely,
# which avoids OOM and build timeouts on the free HF tier.
print("[INFO] Installing face_recognition (no-deps, dlib-bin already present)...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet", "face_recognition"],
    check=False
)
print("[INFO] face_recognition ready.")

import gradio as gr
import uvicorn

# Import the main FastAPI app (with all routes, WebSocket, etc.)
from main import app as fastapi_app


# ── Minimal Gradio block (satisfies HF Gradio SDK requirement) ──────────────
# The real UI is our custom FastAPI HTML frontend, served at /
with gr.Blocks(title="FaceRecog Enterprise Vision", theme=gr.themes.Soft()) as gradio_ui:
    gr.Markdown("""
    # 🎯 FaceRecog Enterprise Vision
    **Real-time facial recognition and tracking engine.**

    > **[→ Click here to open the full application dashboard](/)** 

    Built with FastAPI · MediaPipe FaceMesh · dlib · WebSockets
    """)

# Mount Gradio into FastAPI at /gradio-info (does NOT replace the root UI)
fastapi_app = gr.mount_gradio_app(fastapi_app, gradio_ui, path="/gradio-info")

if __name__ == "__main__":
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=7860,   # HF Spaces requires port 7860
        log_level="info"
    )
