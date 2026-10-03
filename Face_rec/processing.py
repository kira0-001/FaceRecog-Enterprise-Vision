import os
import cv2
import numpy as np
import builtins
import sys
import config

if 'pkg_resources' not in sys.modules:
    class MockPkgResources:
        @staticmethod
        def resource_filename(package_name, resource_name):
            import importlib
            mod = importlib.import_module(package_name)
            return os.path.join(os.path.dirname(mod.__file__), resource_name)
    sys.modules['pkg_resources'] = MockPkgResources()

old_quit = builtins.quit
builtins.quit = lambda *args: print("[WARN] processing.py intercepted face_recognition quit()")
try:
    import face_recognition
except:
    face_recognition = None
builtins.quit = old_quit

# ==============================================================================
# -------------------------- CORE PIPELINE FUNCTIONS ---------------------------
# ==============================================================================
def preprocess_frame(frame):
    """
    Checks the average brightness of the frame. If low-light, applies local
    histogram equalization (CLAHE) on the Y (luminance) channel followed by
    gamma adjustment to boost overall visibility for detection.
    """
    if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return frame
    if not config.ENHANCE_BRIGHTNESS:
        return frame
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b = np.mean(gray)
        if mean_b < 95:
            # Pick higher gamma correction for extremely dark conditions
            gamma = config.GAMMA_LOWLIGHT if mean_b < 60 else 1.3
            
            # Convert to YCrCb to isolate brightness channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(lab)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP, tileGridSize=(8, 8))
            y2 = clahe.apply(y)
            
            # Merge back and convert to BGR
            merged = cv2.merge([y2, cr, cb])
            img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
            
            # Apply Gamma Correction using Lookup Table
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
            return cv2.LUT(img, table)
    except Exception as e:
        print(f"[WARN] Error during frame enhancement: {e}")
    return frame

def load_dnn_detector_if_available():
    """Loads the OpenCV Caffe DNN face detector if prototype and model files exist."""
    base = os.path.dirname(os.path.abspath(__file__))
    # The models are stored inside face_detection_and_recognition/models (which is parent of Face_rec)
    # Wait, let's locate the models folder correctly.
    # In face_recognize.py: os.path.join(base, "models", ...) which means it expected it inside Face_rec/models.
    # Wait, let's verify if the models folder is in face_detection_and_recognition or face_recognize folder.
    # Earlier we listed Face_rec contents:
    # {"name":"face_recognize.py","sizeBytes":"19560"}
    # {"name":"known_faces","isDir":true}
    # Wait, the models folder is in the parent directory!
    # Let's verify directory structure from list_dir:
    # Workspace contains: Face_rec (isDir), models (isDir), recognition.py
    # So `models` is at: `workspace/models` which is the parent folder of `Face_rec`.
    # Let's adjust the path resolution to look in the parent folder or look inside Face_rec too.
    # We can check both locations to be robust!
    # Base is Face_rec directory. Parent is workspace directory.
    # Let's search in both!
    parent = os.path.dirname(base)
    
    # Path candidate 1: Face_rec/models
    # Path candidate 2: workspace/models (parent of Face_rec)
    prototxt = os.path.join(base, "models", "deploy.prototxt")
    caffemodel = os.path.join(base, "models", "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not (os.path.exists(prototxt) and os.path.exists(caffemodel)):
        # Fallback to parent directory models
        prototxt = os.path.join(parent, "models", "deploy.prototxt")
        caffemodel = os.path.join(parent, "models", "res10_300x300_ssd_iter_140000.caffemodel")
        
    if os.path.exists(prototxt) and os.path.exists(caffemodel):
        try:
            net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            print("[INFO] Using OpenCV DNN face detector.")
            return net
        except Exception as e:
            print(f"[WARN] Failed to load DNN: {e}. Falling back to HOG.")
    else:
        print(f"[WARN] Models not found at: {prototxt} or {caffemodel}")
    print("[INFO] Using HOG face detector (fallback).")
    return None

def dnn_detect(net, frame, conf_thresh=config.DNN_CONF_THRESHOLD):
    """
    Performs face detection using the ResNet SSD Caffe model.
    Returns a list of boxes in (x1, y1, x2, y2) coordinate format.
    """
    h, w = frame.shape[:2]
    # Resize to 300x300, subtract mean colors of dataset
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes

def detect_faces(net, frame):
    """
    Detects face boundaries in the frame.
    Uses Caffe DNN if available, otherwise falls back to HOG-based dlib detector.
    """
    if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return []
    if net is not None:
        return dnn_detect(net, frame)
    else:
        # Scale image down for faster HOG scanning
        small_frame = cv2.resize(frame, (0, 0), fx=config.HOG_SCALE, fy=config.HOG_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb_small, model="hog")
        boxes = []
        s = 1.0 / config.HOG_SCALE
        for (t, r, b, l) in locs:
            boxes.append((int(l * s), int(t * s), int(r * s), int(b * s)))
        return boxes

def encode_single_face(img_bgr):
    """
    Extracts a face encoding vector from a static image during face registration.
    Uses 5 jitters for higher database matching precision.
    """
    if face_recognition is None:
        raise RuntimeError("face_recognition engine is not installed. Face enrollment is disabled.")

    if img_bgr is None or img_bgr.size == 0 or img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
        return None
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    
    try:
        boxes = face_recognition.face_locations(rgb, model="hog")
    except Exception as e:
        raise RuntimeError(f"Engine failure during face detection: {e}")

    if not boxes:
        return None
    
    # Target the largest face in the image
    areas = [(b[2] - b[0]) * (b[1] - b[3]) for b in boxes]
    best = boxes[int(np.argmax(areas))]
    
    top, right, bottom, left = best
    h, w = bottom - top, right - left
    pad_y, pad_x = int(h * 0.2), int(w * 0.2)
    cy1 = max(0, top - pad_y)
    cy2 = min(rgb.shape[0], bottom + pad_y)
    cx1 = max(0, left - pad_x)
    cx2 = min(rgb.shape[1], right + pad_x)
    
    chip = rgb[cy1:cy2, cx1:cx2]
    if chip.size == 0:
        return None
        
    chip_small = cv2.resize(chip, (150, 150))
    chip_box = [(0, 150, 150, 0)]
    
    enc = face_recognition.face_encodings(chip_small, chip_box, num_jitters=5)
    return enc[0] if enc else None

def recognize_faces(registry, frame, boxes):
    """
    Crops and scales down detected regions, calculates face encodings, and compares
    them against the loaded registry. Returns a list of matched names or 'Unknown'.
    """
    labels = []
    if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0 or not boxes:
        return labels
        
    # Resize frame to 0.25 size to greatly speed up face encoding calculations
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    scaled_boxes = [(int(T * 0.25), int(R * 0.25), int(B * 0.25), int(L * 0.25)) for (L, T, R, B) in boxes]
    
    try:
        encs = face_recognition.face_encodings(rgb_small, scaled_boxes)
        for enc in encs:
            name = "Unknown"
            if registry["encs"]:
                matches = face_recognition.compare_faces(registry["encs"], enc, tolerance=config.MATCH_TOLERANCE)
                dists = face_recognition.face_distance(registry["encs"], enc)
                if len(dists) > 0:
                    best = int(np.argmin(dists))
                    if matches[best]:
                        name = registry["names"][best]
            labels.append(name)
    except Exception as e:
        print(f"[WARN] Failed to compute face encodings: {e}")
        labels = ["Unknown"] * len(boxes)
    return labels


# ==============================================================================
# ------------------------ DRAWING & ANNOTATION --------------------------------
# ==============================================================================
def draw_annotations(frame, boxes, labels, mode):
    """Draws bounding boxes and labels around detected faces on the frame."""
    for idx, (L, T, R, B) in enumerate(boxes):
        label = labels[idx] if idx < len(labels) else "Unknown"
        
        # Color schemes based on modes and names
        if mode == "1":
            color = (255, 0, 0) # Solid Blue for detection mode
        else:
            # Green for recognized people, Red for Unknowns
            color = (0, 255, 0) if label not in ["Unknown", "Person"] else (0, 0, 255)
            
        # Draw bounding rectangle
        cv2.rectangle(frame, (L, T), (R, B), color, 2)
        # Draw filled rectangle for background tag label
        cv2.rectangle(frame, (L, B - 24), (R, B), color, -1)
        # Draw label text
        cv2.putText(frame, label, (L + 6, B - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
