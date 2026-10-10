"""
app.py — Hugging Face Spaces entry point.

Runs the FastAPI app directly on port 7860 (HF Spaces required port).
No Gradio import — HF only requires the app to listen on port 7860.
"""

import os
import sys
import subprocess

# ── Demo mode for all public deployments ─────────────────────────────────────
os.environ.setdefault("APP_MODE", "demo")

sys.path.insert(0, os.path.dirname(__file__))

# ── Background Setup ─────────────────────────────────────────────────────────
# We run the installation in a background thread so the server starts instantly
# and passes Replit's strict 3-second health check.
def setup_dependencies():
    print("[INFO] Background: Installing face_recognition (no-deps)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
         "--root-user-action=ignore", "face_recognition"],
        check=False, capture_output=True
    )
    print("[INFO] Background: face_recognition ready.")

import threading
threading.Thread(target=setup_dependencies, daemon=True).start()

# ── Import the main FastAPI app ───────────────────────────────────────────────
import uvicorn
from main import app as fastapi_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[INFO] Starting FaceRecog Enterprise Vision on port {port}...")
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
