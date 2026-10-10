"""
main.py — FastAPI server for the Face Recognition Web Application.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open:  http://localhost:8000
"""

import os
import sys
import io
import csv
import json
import base64
import asyncio
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import builtins
import sys
import time
import os
from collections import defaultdict

# ── Application Mode ────────────────────────────────────────────────────────
# Set APP_MODE=demo in your deployment environment (e.g. Hugging Face)
APP_MODE = os.environ.get("APP_MODE", "admin").lower()

# ── Demo Rate Limits ────────────────────────────────────────────────────────
MAX_FACES_PER_IP = 3
MAX_STREAM_SECONDS_PER_IP = 3 * 60 * 60  # 3 hours

_ip_enrollments = defaultdict(int)
_ip_stream_time = defaultdict(float)
_ip_last_seen = {}

# --- Python 3.11+ Compatibility Fix for face_recognition_models ---
# setuptools >= 70 removed pkg_resources, which breaks face_recognition_models.
# We inject a mock here so it can find its .dat files using standard os.path
if 'pkg_resources' not in sys.modules:
    class MockPkgResources:
        @staticmethod
        def resource_filename(package_name, resource_name):
            import importlib
            mod = importlib.import_module(package_name)
            return os.path.join(os.path.dirname(mod.__file__), resource_name)
    sys.modules['pkg_resources'] = MockPkgResources()

# Prevent broken face_recognition from killing the Uvicorn server with quit()
old_quit = builtins.quit
builtins.quit = lambda *args: print("[WARN] face_recognition tried to quit(), but we intercepted it.")
try:
    import face_recognition as fr
except Exception as e:
    print(f"[WARN] Failed to import face_recognition: {e}")
    fr = None
builtins.quit = old_quit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
MODELS_DIR.mkdir(exist_ok=True)
KNOWN_FACES_DIR.mkdir(exist_ok=True)

# ── Auto-download DNN model if missing (avoids storing 10MB in git) ──────────
_CAFFEMODEL_PATH = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
_PROTOTXT_PATH   = MODELS_DIR / "deploy.prototxt"
_CAFFEMODEL_URL  = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
_PROTOTXT_URL    = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

def _download_if_missing(path: Path, url: str):
    if not path.exists():
        print(f"[INFO] Downloading {path.name} ...")
        import urllib.request
        try:
            urllib.request.urlretrieve(url, str(path))
            print(f"[INFO] Downloaded {path.name} ({path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"[WARN] Could not download {path.name}: {e}")

# ── Import processing helpers from Face_rec ──────────────────────────────────
sys.path.insert(0, str(BASE_DIR / "Face_rec"))
try:
    import processing
    print("[INFO] Loaded processing module from Face_rec/")
except ImportError:
    print("[WARN] Face_rec/processing.py not found. Detection may be limited.")
    processing = None

# ── Database ─────────────────────────────────────────────────────────────────
from face_db import FaceDB
db = FaceDB()

# ── DNN Detector — loaded in background ────────────────────────────────────
_net = None

def _background_setup():
    _download_if_missing(_CAFFEMODEL_PATH, _CAFFEMODEL_URL)
    _download_if_missing(_PROTOTXT_PATH,   _PROTOTXT_URL)
    global _net
    if processing:
        _net = processing.load_dnn_detector_if_available()

import threading
threading.Thread(target=_background_setup, daemon=True).start()

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="FaceRecog API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # In Demo Mode, only protect DELETE routes. In Admin Mode, protect everything except /api/login.
    if request.url.path.startswith("/api/") and request.url.path not in ["/api/login", "/api/config"]:
        requires_auth = True
        if APP_MODE == "demo" and request.method != "DELETE":
            requires_auth = False
            
        if requires_auth and request.cookies.get("auth_token") != "demo_logged_in":
            return JSONResponse({"detail": "Unauthorized: Admin access required"}, status_code=401)
            
    return await call_next(request)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Serve registered face images
app.mount("/faces", StaticFiles(directory=str(KNOWN_FACES_DIR)), name="faces")


# ── Utility Functions ─────────────────────────────────────────────────────────

def decode_frame(b64_string: str):
    """Decode a base64 JPEG (from browser canvas) to a numpy BGR image."""
    try:
        header, encoded = b64_string.split(",", 1) if "," in b64_string else ("", b64_string)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERROR] decode_frame: {e}")
        return None


# ── Recognition Pipeline (non-blocking) ──────────────────────────────────────
_recog_lock   = threading.Lock()
_recog_busy   = False
MIN_FACE_AREA = 3500   # Reject boxes smaller than this (prevents background faces)

def _filter_boxes(boxes, frame_w, frame_h):
    """Remove tiny boxes that are probably background faces, not the user."""
    out = []
    for (x1, y1, x2, y2) in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area >= MIN_FACE_AREA:
            out.append((x1, y1, x2, y2))
    return out

def _detect_only(frame):
    """Fast detection pass (20+ FPS)"""
    if processing:
        proc = processing.preprocess_frame(frame)
        boxes = processing.detect_faces(_net, proc)
    elif fr:
        rgb_s = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        locs = fr.face_locations(rgb_s, model="hog")
        boxes = [(l, t, r, b) for (t, r, b, l) in locs]
    else:
        boxes = []
    return _filter_boxes(boxes, frame.shape[1], frame.shape[0])

def _recognize_worker(frame, boxes, registry, local_result):
    """Slow recognition pass (runs in background)"""
    global _recog_busy
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = []
        for (x1, y1, x2, y2) in boxes:
            name, confidence = "Unknown", 0.0
            h_b = y2 - y1; w_b = x2 - x1
            py, px = int(h_b * 0.25), int(w_b * 0.25)
            cy1 = max(0, y1-py); cy2 = min(rgb.shape[0], y2+py)
            cx1 = max(0, x1-px); cx2 = min(rgb.shape[1], x2+px)
            chip = rgb[cy1:cy2, cx1:cx2]
            
            if chip.size > 0:
                chip_s = cv2.resize(chip, (150, 150))
                encs = fr.face_encodings(chip_s, [(0, 150, 150, 0)])
                if encs:
                    dists = fr.face_distance(registry["encs"], encs[0])
                    if len(dists):
                        bi = int(np.argmin(dists))
                        bd = float(dists[bi])
                        if bd < 0.55:  # Stricter tolerance (prevents false matches on clothes)
                            name = registry["names"][bi]
                            confidence = round((1.0 - bd) * 100, 1)
            results.append({"name": name, "confidence": confidence})
            
        with _recog_lock:
            local_result.clear(); local_result.extend(results)
    finally:
        _recog_busy = False



# ── REST Endpoints ────────────────────────────────────────────────────────────


@app.get("/")
def serve_ui():
    """Serve the main web application."""
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/people")
def get_people():
    """Return all registered people."""
    return db.get_all_people()


from typing import Optional

@app.post("/api/register")
async def register_face(
    name: str = Form(...),
    mode: str = Form("upload"),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    path: Optional[str] = Form(None)
):
    """
    Register a new face from an uploaded image, live camera snapshot, URL, or local path.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    client_ip = request.client.host if request.client else "unknown"
    if APP_MODE == "demo" and client_ip != "unknown":
        if _ip_enrollments[client_ip] >= MAX_FACES_PER_IP:
            raise HTTPException(status_code=429, detail="Demo limit reached: You can only enroll up to 3 faces per day.")

    img = None

    if mode in ["upload", "camera"]:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided.")
        contents = await file.read()
        arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
    elif mode == "url":
        if not url:
            raise HTTPException(status_code=400, detail="No URL provided.")
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
            
    elif mode == "path":
        if not path:
            raise HTTPException(status_code=400, detail="No path provided.")
        # Strip quotes if user pasted path with quotes
        clean_path = path.strip('"\'')
        if not os.path.exists(clean_path):
            raise HTTPException(status_code=400, detail="Local file not found.")
        img = cv2.imread(clean_path)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not load or decode the image.")

    enc = None
    if processing:
        try:
            enc = processing.encode_single_face(img)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
    else:
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        try:
            if fr is None:
                raise RuntimeError("face_recognition is broken")
            locs = fr.face_locations(rgb, model="hog")
            if locs:
                encs = fr.face_encodings(rgb, locs)
                enc = encs[0] if encs else None
        except Exception as e:
            raise HTTPException(status_code=503, detail="Recognition engine offline: " + str(e))

    if enc is None:
        raise HTTPException(status_code=422, detail="No face detected in image. Use a clearer photo.")

    # Save thumbnail
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).strip()
    img_path = str(KNOWN_FACES_DIR / f"{safe_name}.jpg")
    cv2.imwrite(img_path, img)

    success = db.add_person(name, enc, f"/faces/{safe_name}.jpg")
    if not success:
        raise HTTPException(status_code=500, detail="Database error. Please try again.")

    client_ip = request.client.host if request.client else "unknown"
    if client_ip != "unknown":
        _ip_enrollments[client_ip] += 1

    return {"success": True, "name": name, "image": f"/faces/{safe_name}.jpg"}

@app.post("/api/login")
def login(password: str = Form(...)):
    if password == "admin123":
        response = JSONResponse({"success": True})
        response.set_cookie("auth_token", "demo_logged_in", httponly=True)
        return response
    raise HTTPException(status_code=401, detail="Invalid authorization code.")

@app.get("/api/config")
def get_config():
    """Return the current application mode to the frontend."""
    return {"mode": APP_MODE}

@app.post("/api/logout")
def logout():
    response = JSONResponse({"success": True})
    response.delete_cookie("auth_token")
    return response

@app.delete("/api/people/{name}")
def delete_person(name: str):
    """Remove a registered person."""
    db.delete_person(name)
    return {"success": True, "deleted": name}


@app.get("/api/attendance")
def get_attendance():
    """Get today's attendance records."""
    return db.get_today_attendance()

@app.delete("/api/attendance/clear")
def clear_attendance():
    """Clear today's attendance records."""
    db.clear_today_attendance()
    return {"success": True}


@app.get("/api/attendance/export")
def export_attendance():
    """Download today's attendance as a CSV file."""
    rows = db.get_today_attendance()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["name", "seen_at", "confidence"])
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/stats")
def get_stats():
    """Get dashboard summary statistics."""
    return db.get_attendance_stats()


# ── WebSocket Stream ──────────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    High-performance real-time face detection + recognition.

    Architecture:
    - DNN detection (fast) runs every frame via asyncio.to_thread (~20+ FPS)
    - dlib recognition (slow) runs in a daemon background thread, never blocks UI
    - Client always gets fresh box coordinates; name label updates asynchronously
    """
    await ws.accept()
    
    # In admin mode, strictly require websocket auth
    if APP_MODE == "admin" and ws.cookies.get("auth_token") != "demo_logged_in":
        print("[WS] Rejected unauthorized connection.")
        await ws.close(code=1008)
        return

    client_ip = ws.client.host if ws.client else "unknown"
    if APP_MODE == "demo" and client_ip != "unknown":
        _ip_last_seen[client_ip] = time.time()

    print(f"[WS] Client connected: {client_ip}")

    registry = db.get_registry()
    frame_count = 0
    local_result = []  # shared buffer: [{name, confidence}, ...]
    global _recog_busy

    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            
            # Enforce streaming limits only in demo mode
            if APP_MODE == "demo" and client_ip != "unknown":
                now = time.time()
                _ip_stream_time[client_ip] += (now - _ip_last_seen[client_ip])
                _ip_last_seen[client_ip] = now
                
                if _ip_stream_time[client_ip] > MAX_STREAM_SECONDS_PER_IP:
                    await ws.send_text(json.dumps({"error": "LIMIT_REACHED"}))
                    await ws.close(code=1008)
                    return

            mode = payload.get("mode", "recognize")
            frame_b64 = payload.get("frame", "")

            if not frame_b64:
                await ws.send_text(json.dumps({"faces": [], "count": 0}))
                continue

            # ── Step 1: Decode + resize in thread (non-blocking) ──────────────
            frame = await asyncio.to_thread(decode_frame, frame_b64)
            if frame is None:
                await ws.send_text(json.dumps({"faces": [], "count": 0}))
                continue
            frame = await asyncio.to_thread(cv2.resize, frame, (480, 360))

            # ── Step 2: Fast DNN detection every frame ────────────────────────
            boxes = await asyncio.to_thread(_detect_only, frame)

            # ── Step 3: Kick off background recognition (if idle) ─────────────
            if mode == "recognize" and boxes and fr and registry["encs"] and not _recog_busy:
                _recog_busy = True
                threading.Thread(
                    target=_recognize_worker,
                    args=(frame.copy(), boxes, registry, local_result),
                    daemon=True
                ).start()

            # ── Step 4: Build response using LIVE boxes + CACHED names ─────────
            if mode == "recognize":
                with _recog_lock:
                    cached = list(local_result)
                faces_out = []
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    name = cached[i]["name"] if i < len(cached) else "Scanning…"
                    conf = cached[i]["confidence"] if i < len(cached) else 0.0
                    faces_out.append({
                        "x": int(x1), "y": int(y1),
                        "w": int(x2-x1), "h": int(y2-y1),
                        "name": name, "confidence": conf
                    })
                # Log confirmed faces
                for face in faces_out:
                    if face["name"] not in ("Unknown", "Scanning…"):
                        db.log_attendance(face["name"], face["confidence"])
            else:
                faces_out = [
                    {"x": int(x1), "y": int(y1), "w": int(x2-x1), "h": int(y2-y1),
                     "name": "Person", "confidence": 0.0}
                    for (x1, y1, x2, y2) in boxes
                ]

            frame_count += 1
            if frame_count % 60 == 0:
                registry = db.get_registry()  # refresh registry periodically

            await ws.send_text(json.dumps({"faces": faces_out, "count": len(faces_out)}))

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {ws.client}")
    except Exception as e:
        print(f"[WS] Error: {e}")



# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  FaceRecog API  |  http://localhost:8000")
    print("=" * 55 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
