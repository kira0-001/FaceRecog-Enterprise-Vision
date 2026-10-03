import cv2
import threading
import time
import config

# ==============================================================================
# ------------------------- THREADED CAMERA CLASS ------------------------------
# ==============================================================================
class ThreadedCamera:
    """
    Spawns a background thread to continuously capture frames from the camera.
    This prevents the main GUI thread from waiting on OpenCV's camera I/O,
    ensuring maximum display framerate.
    """
    def __init__(self, src=config.VIDEO_SOURCE):
        # Open camera using DirectShow (CAP_DSHOW) on Windows for faster startup
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.ret, self.frame = self.cap.read()
        self.running = True
        
        # Start a daemon background thread to grab frames
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Continuously reads frames from the capture object."""
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            time.sleep(0.01)  # Small sleep to prevent CPU starvation

    def read(self):
        """Returns the most recently grabbed frame and success status."""
        return self.ret, self.frame

    def isOpened(self):
        """Checks if the underlying OpenCV capture object is successfully opened."""
        return self.cap.isOpened()

    def release(self):
        """Stops the capture thread and releases camera resources."""
        self.running = False
        self.thread.join()
        self.cap.release()
