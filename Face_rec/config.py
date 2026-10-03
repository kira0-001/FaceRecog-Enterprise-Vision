import os

# ==============================================================================
# ----------------------------- CONFIGURATION ----------------------------------
# ==============================================================================
DNN_CONF_THRESHOLD = 0.85       # Confidence threshold for DNN face detection (0.0 to 1.0)
HOG_SCALE = 0.25                # Scale factor for HOG face detection (smaller = faster)
ENHANCE_BRIGHTNESS = True       # Automatically enhance brightness in dim scenes
GAMMA_LOWLIGHT = 1.6            # Gamma correction value for low light conditions
CLAHE_CLIP = 2.0                # CLAHE clip limit for local contrast enhancement
MATCH_TOLERANCE = 0.45          # Strict tolerance for face matching (lower = stricter)
VIDEO_SOURCE = 0                # Camera index (0 is typically the default webcam)

DATA_DIR = "known_faces"        # Directory where registered face images and DB are stored
DB_FILE = os.path.join(DATA_DIR, "registry.pkl") # Pickle database file path

# Ensure the registry data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
