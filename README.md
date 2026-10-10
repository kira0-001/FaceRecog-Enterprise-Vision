---
title: FaceRecog Enterprise Vision
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
short_description: Real-time face recognition with MediaPipe FaceMesh & dlib
---

# FaceRecog — Enterprise Vision Engine 🎯

A high-performance, real-time facial recognition and tracking engine built with **FastAPI**, **WebSockets**, **MediaPipe FaceMesh**, and **Python dlib**. Features a premium dark-mode UI and a dual Admin / Public Demo mode.

> 🚀 **[Live Demo →](https://face-recog-enterprise-vision--karthimrkl.replit.app/)**

---

## ✨ Features

- **Real-Time Face Tracking** — MediaPipe FaceMesh runs at 60 FPS directly in the browser GPU with zero lag.
- **Server-Side Recognition** — `dlib` + `face_recognition` for highly accurate identity matching.
- **3D Scan Enrollment** — Animated FaceID-style scan UI with Red→Blue→Green color progression.
- **Oval Face Overlay** — HUD-style bounding ellipse with corner accents.
- **Welcome Toasts** — "👋 Welcome, Name!" triggers the first time a face is recognized.
- **2-Hour Session Limiter** — Per-user AI processing time limit to prevent server overload.
- **3-Face Enrollment Limit** — Public demo users can enroll up to 3 faces per session.
- **Admin Mode** — Full control: delete faces, clear activity logs, and reset the demo.
- **Activity Log** — Auto-updating attendance log with timestamps and confidence scores.
- **Object Detection Mode** — Powered by COCO-SSD for real-time object detection.
- **Dark / Light Theme** — Persists between sessions via localStorage.
- **Fullscreen Mode** — One-click fullscreen for presentations.
- **CSV Export** — Download the full attendance log.

---

## 🖥️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Real-Time Stream | WebSockets |
| Face Recognition | Python `face_recognition` (dlib) |
| Local Face Tracking | MediaPipe FaceMesh (WebGL) |
| Object Detection | TensorFlow.js + COCO-SSD |
| Database | SQLite (via custom `face_db.py`) |
| Frontend | Vanilla HTML/CSS/JavaScript |
| Deployment | Docker + Hugging Face Spaces |

---

## ⚙️ Application Modes

This app supports two modes, controlled by the `APP_MODE` environment variable:

| Feature | Admin Mode (`APP_MODE=admin`) | Demo Mode (`APP_MODE=demo`) |
|---|---|---|
| Login Required | ✅ Yes (`admin123`) | ❌ No |
| Enroll Faces | ✅ Unlimited | ✅ Up to 3 faces |
| Delete Faces | ✅ Yes | ❌ No |
| Clear Activity Log | ✅ Yes | ❌ No |
| Streaming Time | ✅ Unlimited | ✅ Up to 3 hours/day |
| Reset Demo Button | ✅ Visible | ❌ Hidden |

---

## 🚀 Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/FaceRecog-Enterprise-Vision.git
cd FaceRecog-Enterprise-Vision
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
> **Note:** `face_recognition` requires CMake and C++ build tools. On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) first.
```bash
pip install -r requirements.txt
```

### 4. Run the Server

**Admin Mode (default):**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Public Demo Mode:**
```bash
APP_MODE=demo uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at **http://localhost:8000**

---

## 🐳 Deployment (Docker / Hugging Face Spaces)

A `Dockerfile` is included. The `APP_MODE` environment variable controls the mode.

1. Push this repository to GitHub.
2. Go to [Hugging Face Spaces](https://huggingface.co/spaces) → Create New Space.
3. Select **Docker** as the Space SDK.
4. Link your GitHub repository.
5. In the **Space Settings → Repository Secrets**, add:
   - `APP_MODE` = `demo`
6. Hugging Face will build and deploy automatically!

---

## 📁 Project Structure

```
├── main.py             # FastAPI server, WebSocket stream, rate limiting
├── face_db.py          # SQLite database layer (register, attend, stats)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Production Docker image
├── .gitignore          # Excludes database, known_faces/, __pycache__
├── Face_rec/
│   └── processing.py   # DNN face detection + encoding helpers
├── static/
│   ├── index.html      # Single-page application HTML
│   ├── app.js          # Frontend logic (WebSocket, MediaPipe, UI)
│   └── style.css       # Premium dark/light theme CSS
└── known_faces/        # (Auto-created) Stores enrolled face images
```

---

## 🔐 Admin Access

Default admin password: `admin123`

To change it, edit line in `main.py`:
```python
if password == "your_new_password":
```

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

*Built with ❤️ as part of an AI internship project.*
