import threading
import processing

# ==============================================================================
# ----------------------- BACKGROUND FACE WORKER THREAD ------------------------
# ==============================================================================
class FaceProcessingWorker:
    """
    An asynchronous background worker thread that processes face detection
    and face recognition on frames submitted from the main rendering loop.
    Decoupling the heavy dlib/DNN computation prevents video stuttering.
    """
    def __init__(self, registry, net):
        self.registry = registry
        self.net = net
        self.frame = None
        self.mode = "1"  # "1" = Detection, "2" = Recognition
        self.last_boxes = []
        self.last_labels = []
        self.running = True
        self.busy = False
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        
        # Start the background execution loop
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit_task(self, frame, mode):
        """
        Submits a frame for face processing.
        Returns True if the task was successfully scheduled,
        or False if the worker thread is currently busy processing.
        """
        with self.lock:
            if self.busy:
                return False  # Skip scheduling this frame to prevent queue lag
            self.frame = frame.copy()
            self.mode = mode
            self.busy = True
            with self.condition:
                self.condition.notify()
            return True

    def get_latest_results(self):
        """Safely returns the most recently computed face boxes and labels."""
        with self.lock:
            return self.last_boxes.copy(), self.last_labels.copy()

    def stop(self):
        """Instructs the background thread to exit and joins it."""
        self.running = False
        with self.condition:
            self.condition.notify()
        self.thread.join()

    def _run(self):
        """Background processing execution loop."""
        while self.running:
            with self.condition:
                # Wait until there is a task to execute or thread is stopped
                while self.running and not self.busy:
                    self.condition.wait(timeout=0.1)
                if not self.running:
                    break
            
            # Extract variables local to this run
            frame_to_process = self.frame
            mode_to_process = self.mode
            
            try:
                # 1. Preprocess & Brightness Enhance
                proc = processing.preprocess_frame(frame_to_process)
                
                # 2. Face Detection
                boxes = processing.detect_faces(self.net, proc)
                
                # 3. Face Recognition (if mode == "2")
                if mode_to_process == "2":
                    labels = processing.recognize_faces(self.registry, proc, boxes)
                else:
                    labels = ["Person"] * len(boxes)
                
                # Update shared variables thread-safely
                with self.lock:
                    self.last_boxes = boxes
                    self.last_labels = labels
            except Exception as e:
                print(f"[ERROR] Error in background worker thread: {e}")
            
            # Mark task completion
            with self.lock:
                self.busy = False
