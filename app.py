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

# ── Install face_recognition without dlib compilation ────────────────────────
# dlib-bin (pre-compiled wheel) is already installed from requirements.txt.
# --no-deps skips face_recognition's own dlib source download entirely.
print("[INFO] Installing face_recognition (no-deps, dlib-bin already present)...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
     "--root-user-action=ignore", "face_recognition"],
    check=False, capture_output=True, text=True
)
if result.returncode == 0:
    print("[INFO] face_recognition installed successfully.")
else:
    print(f"[WARN] face_recognition install issue: {result.stderr[:200]}")

# ── Import the main FastAPI app ───────────────────────────────────────────────
import uvicorn
from main import app as fastapi_app

if __name__ == "__main__":
    print("[INFO] Starting FaceRecog Enterprise Vision on port 7860...")
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
