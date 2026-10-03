import os
import pickle
import requests
import cv2
import numpy as np
from urllib.parse import urlparse
import config

# ==============================================================================
# ------------------------- REGISTRY DATABASE HELPERS --------------------------
# ==============================================================================
def load_registry():
    """Loads saved faces database from the pickle file."""
    if os.path.exists(config.DB_FILE):
        try:
            with open(config.DB_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load registry: {e}")
    return {"names": [], "encs": [], "paths": []}

def save_registry(data):
    """Saves faces database to the pickle file."""
    try:
        with open(config.DB_FILE, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"[ERROR] Failed to save registry: {e}")

def get_unique_name(name, existing_names):
    """Checks for name collisions in the database and appends a numeric suffix if needed."""
    if name not in existing_names:
        return name
    idx = 1
    while f"{name}_{idx}" in existing_names:
        idx += 1
    return f"{name}_{idx}"


# ==============================================================================
# --------------------------- NETWORKING & IMAGES ------------------------------
# ==============================================================================
def is_url(s):
    """Checks if a string is a valid HTTP/HTTPS URL."""
    try:
        u = urlparse(s.strip())
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def load_image_from_source(src):
    """Loads an image from either a local file path or a remote URL."""
    src = src.strip().strip("\"'")
    if is_url(src):
        try:
            r = requests.get(src, timeout=10)
            r.raise_for_status()
            data = np.frombuffer(r.content, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERROR] Failed to download image from URL: {e}")
            return None
    else:
        if not os.path.exists(src):
            print(f"[ERROR] File path does not exist: {src}")
            return None
        return cv2.imread(src)
