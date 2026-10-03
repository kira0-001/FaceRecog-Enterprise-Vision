"""
recognition_menu.py

- Asks how many known faces to register (name + local path or URL).
- Loads encodings (reports skips).
- Shows menu: 1) Detect  2) Recognize  0) Quit
- Only opens camera after you pick an option.
- Uses OpenCV DNN (if models in ./models/) else HOG fallback.
- Auto-enhances dark frames to help detection.
- Recognition runs every N frames for smoothness.
"""

import os
import cv2
import numpy as np
import face_recognition
import requests
from urllib.parse import urlparse
import time

# ---------------- CONFIG ----------------
RECOG_EVERY_N_FRAMES = 6        # recognition frequency
DNN_CONF_THRESHOLD = 0.45       # DNN detection sensitivity (lower finds smaller faces)
HOG_SCALE = 0.25                # scale for HOG detection (speed)
ENHANCE_BRIGHTNESS = True       # toggle auto-enhancement in low light
GAMMA_LOWLIGHT = 1.6
CLAHE_CLIP = 2.0
MATCH_TOLERANCE = 0.5
VIDEO_SOURCE = 0
# ----------------------------------------

def is_url(s):
    try:
        u = urlparse(s.strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def load_image_from_source(src):
    src = src.strip().strip("\"'")
    if is_url(src):
        try:
            r = requests.get(src, timeout=12)
            r.raise_for_status()
            data = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[ERROR] Failed to load URL '{src}': {e}")
            return None
    else:
        if not os.path.exists(src):
            print(f"[ERROR] File not found: {src}")
            return None
        img = cv2.imread(src)
        if img is None:
            print(f"[ERROR] Could not read file: {src}")
        return img

def encode_single_face(img_bgr):
    if img_bgr is None:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None
    # choose largest face (most likely target)
    areas = [(b[2]-b[0])*(b[1]-b[3]) for b in boxes]
    best = boxes[int(np.argmax(areas))]
    enc = face_recognition.face_encodings(rgb, [best])
    return enc[0] if enc else None

def load_dnn_detector_if_available():
    base = os.path.dirname(os.path.abspath(__file__))
    prototxt = os.path.join(base, "models", "deploy.prototxt")
    caffemodel = os.path.join(base, "models", "res10_300x300_ssd_iter_140000.caffemodel")
    if os.path.exists(prototxt) and os.path.exists(caffemodel):
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            print("[INFO] Using OpenCV DNN face detector.")
            return net
        except Exception as e:
            print(f"[WARN] Could not load DNN detector: {e}")
    print("[INFO] Using HOG face detector (fallback).")
    return None

def enhance_frame(frame, gamma=1.3, clahe_clip=2.0):
    """Improve low-light frames: CLAHE on Y channel + gamma."""
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8,8))
        y2 = clahe.apply(y)
        merged = cv2.merge([y2, cr, cb])
        img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        img = cv2.LUT(img, table)
        return img
    except Exception:
        return frame

def dnn_detect(net, frame, conf_thresh=DNN_CONF_THRESHOLD):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if conf >= conf_thresh:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1,y1,x2,y2))
    return boxes

def hog_detect(frame, scale=HOG_SCALE):
    small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb_small, model="hog")
    boxes = []
    s = 1.0/scale
    for (t,r,b,l) in locs:
        boxes.append((int(l*s), int(t*s), int(r*s), int(b*s)))
    return boxes

# -------- interactive loading of known faces ----------
known_encs = []
known_names = []

try:
    n = int(input("How many known faces do you want to register? ").strip())
except:
    n = 0

for i in range(n):
    name = input(f"Enter name for person {i+1}: ").strip()
    src = input(f"Enter local PATH or URL for {name}: ").strip()
    img = load_image_from_source(src)
    enc = encode_single_face(img)
    if enc is not None:
        known_encs.append(enc)
        known_names.append(name)
        print(f"[OK] Loaded {name}")
    else:
        print(f"[SKIP] Could not find a usable face for {name}")

if not known_encs:
    print("[INFO] No known faces loaded. Recognition will label everyone 'Unknown'.")

# -------- prepare detector (but DO NOT open camera yet) ----------
net = load_dnn_detector_if_available()

# -------- menu loop (camera opens only after you choose) ----------
while True:
    print("""
MENU
1) Detect Faces (fast)
2) Recognize Faces
0) Quit
""")
    ch = input(">>> ").strip()
    if ch == "0":
        print("Over")
        break
    if ch not in ("1","2"):
        print("Invalid. Try 0,1,2")
        continue

    # Open camera now (only when user chooses)
    cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check device index:", VIDEO_SOURCE)
        continue

    print("[INFO] Camera opened. Press 'q' in window to return to menu.")
    frame_idx = 0
    last_labels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera frame not received.")
            break

        # optionally auto-enhance if dim scene
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b = np.mean(gray)
        proc = frame
        if ENHANCE_BRIGHTNESS and mean_b < 95:
            # choose stronger enhancement for darker scenes
            gamma = GAMMA_LOWLIGHT if mean_b < 60 else 1.3
            proc = enhance_frame(frame, gamma=gamma, clahe_clip=CLAHE_CLIP)

        # choose detector
        if net is not None:
            boxes = dnn_detect(net, proc, conf_thresh=DNN_CONF_THRESHOLD)
        else:
            boxes = hog_detect(proc, scale=HOG_SCALE)

        # If no boxes found, and scene dim, try slightly more sensitive pass (optional)
        if not boxes and ENHANCE_BRIGHTNESS and mean_b < 50:
            # second pass with slightly different params
            if net is not None:
                boxes = dnn_detect(net, proc, conf_thresh=max(0.35, DNN_CONF_THRESHOLD-0.1))
            else:
                boxes = hog_detect(proc, scale=HOG_SCALE * 0.9)

        if ch == "1":
            # Detection-only: draw 'Person'
            for (L,T,R,B) in boxes:
                cv2.rectangle(frame, (L,T), (R,B), (255,0,0), 2)
                cv2.putText(frame, "Person", (L, T-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        else:
            # Recognition mode: only run encodings every N frames for speed
            labels = []
            if frame_idx % RECOG_EVERY_N_FRAMES == 0 and boxes:
                # convert boxes to face_recognition format (top,right,bottom,left)
                trbl = [(T, R, B, L) for (L, T, R, B) in boxes]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb, trbl)
                for enc in encs:
                    name = "Unknown"
                    if known_encs:
                        matches = face_recognition.compare_faces(known_encs, enc, tolerance=MATCH_TOLERANCE)
                        dists = face_recognition.face_distance(known_encs, enc)
                        if len(dists) > 0:
                            best = int(np.argmin(dists))
                            if matches[best]:
                                name = known_names[best]
                    labels.append(name)
                last_labels = labels[:]    # cache results
            else:
                labels = last_labels[:]    # reuse previous labels so bounding boxes don't flicker

            # draw boxes + labels
            for idx, (L,T,R,B) in enumerate(boxes):
                label = labels[idx] if idx < len(labels) else "Unknown"
                cv2.rectangle(frame, (L,T), (R,B), (0,255,0), 2)
                cv2.rectangle(frame, (L, B-24), (R, B), (0,255,0), -1)
                cv2.putText(frame, label, (L+6, B-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            # if no boxes and known faces exist, we can show hint
            if not boxes:
                cv2.putText(frame, "No faces detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        mode = "Detection" if ch == "1" else "Recognition"
        cv2.imshow(f"[{mode}] Press 'q' to menu", frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
