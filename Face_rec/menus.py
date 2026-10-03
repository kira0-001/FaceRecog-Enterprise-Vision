import os
import cv2
import numpy as np
import face_recognition
import tkinter as tk
from tkinter import filedialog
import time

import config
import database
import camera
import processing
import worker

# ==============================================================================
# -------------------------- LIVE CAMERA STREAM LOOP ---------------------------
# ==============================================================================
def run_live_session(mode, registry, net):
    """
    Manages the camera frame grab, asynchronous processing delegation, 
    and UI loop. Decouples processing from frame rendering to reduce frame drops.
    """
    print("[INFO] Starting threaded camera. Press 'q' in window to return to menu.")
    cap = camera.ThreadedCamera(config.VIDEO_SOURCE)
    time.sleep(1.0) # Warmup period
    
    if not cap.ret:
        print("[ERROR] Cannot read frame from camera.")
        cap.release()
        return
    if not cap.isOpened():
        print("[ERROR] Camera is not opened. Check VIDEO_SOURCE index:", config.VIDEO_SOURCE)
        cap.release()
        return

    print("[INFO] Camera stream opened. Press 'q' inside video window to stop.")
    
    # Initialize background face processing worker thread
    proc_worker = worker.FaceProcessingWorker(registry, net)
    
    mode_title = "Detection" if mode == "1" else "Recognition"
    window_name = f"[{mode_title}] Press 'q' to return to menu"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Asynchronously submit the frame copy to the worker thread.
        # This returns instantly, and the worker works in the background.
        proc_worker.submit_task(frame, mode)
        
        # Non-blocking fetch of the latest computed results
        boxes, labels = proc_worker.get_latest_results()
        
        # Draw annotations on a copy of the camera frame
        annotated_frame = frame.copy()
        processing.draw_annotations(annotated_frame, boxes, labels, mode)
        
        # Display the smooth frame
        cv2.imshow(window_name, annotated_frame)
        
        # Yield to OpenCV GUI loop for window refreshes (1ms check)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean shut down of background worker and camera
    proc_worker.stop()
    cap.release()
    cv2.destroyAllWindows()


# ==============================================================================
# ------------------------- REGISTRATION & MANAGEMENT --------------------------
# ==============================================================================
def register_new_faces(registry):
    """Console UI sub-menu to register new faces via Webcam, File, Path, or URL."""
    while True:
        print("\n" + "-"*30)
        print(" ADD A NEW PERSON ")
        print("-"*30)
        
        base_name = input("Enter the person's name (or 'cancel' to go back): ").strip()
        if base_name.lower() == 'cancel':
            break
        
        name = database.get_unique_name(base_name, registry["names"])
        if name != base_name:
            print(f"[NOTE] The name '{base_name}' is already registered. Using '{name}' instead.")

        print(f"\nHow would you like to add the face for {name}?")
        print("1) Live Face Scan (Webcam - Highest Accuracy)")
        print("2) Browse for File (Opens File Explorer)")
        print("3) Enter Local File Path Manually")
        print("4) Enter Image URL")
        print("0) Cancel")
        
        choice = input(">>> ").strip()
        if choice == "0":
            continue
            
        # Choice 1: Live Webcam Scan
        if choice == "1":
            print("[INFO] Opening camera... Look at the lens and slowly move your head.")
            cap = cv2.VideoCapture(config.VIDEO_SOURCE, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("[ERROR] Cannot open camera.")
                continue

            # Warmup camera and clear system capture buffers
            time.sleep(1.0)
            for _ in range(5):
                cap.read()

            collected_encodings = []
            required_samples = 20
            best_frame = None

            while len(collected_encodings) < required_samples:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    time.sleep(0.05)
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb_frame, model="hog")
                
                if boxes:
                    # Save clean frame before drawing box
                    best_frame = frame.copy()
                    
                    areas = [(b[2]-b[0])*(b[1]-b[3]) for b in boxes]
                    best = boxes[int(np.argmax(areas))]
                    enc = face_recognition.face_encodings(rgb_frame, [best])[0]
                    collected_encodings.append(enc)
                    
                    L, T, R, B = best[3], best[0], best[1], best[2]
                    progress = int((len(collected_encodings) / required_samples) * 100)
                    cv2.rectangle(frame, (L, T), (R, B), (0, 255, 255), 2)
                    cv2.putText(frame, f"Scanning... {progress}%", (L, T-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "No face detected. Look at camera.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Live Scanner", frame)
                cv2.waitKey(100)

            cap.release()
            cv2.destroyWindow("Live Scanner")

            if collected_encodings:
                master_encoding = np.mean(collected_encodings, axis=0)
                
                # OPTIMIZATION: Save the visual scan frame to a file and registry
                local_img_path = os.path.join(config.DATA_DIR, f"{name}.jpg")
                if best_frame is not None:
                    cv2.imwrite(local_img_path, best_frame)
                    saved_path = local_img_path
                    print(f"[INFO] Saved face snapshot to: {local_img_path}")
                else:
                    saved_path = "LIVE_SCAN"

                registry["names"].append(name)
                registry["encs"].append(master_encoding)
                registry["paths"].append(saved_path)
                database.save_registry(registry)
                print(f"[SUCCESS] Live scan complete for {name}!")
            else:
                print("[ERROR] Scan failed. Could not collect enough face data.")
            continue

        # Static Images (Choice 2, 3, or 4)
        src = None
        if choice == "2":
            print("[INFO] Opening File Explorer... Please select an image window.")
            root = tk.Tk()
            root.withdraw() # Hide empty tkinter frame
            root.attributes('-topmost', True) # Bring file selection to front
            src = filedialog.askopenfilename(
                title=f"Select Face Image for {name}", 
                filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
            root.destroy()
            if not src:
                print("[INFO] File selection cancelled.")
                continue
                
        elif choice == "3":
            src = input(f"Enter exact local PATH for {name}: ").strip()
        elif choice == "4":
            src = input(f"Enter URL for {name}: ").strip()
        else:
            print("Invalid choice.")
            continue

        # Load and process the selected static image
        print(f"[INFO] Processing {name} from {src}...")
        img = database.load_image_from_source(src)
        if img is None:
            print("[ERROR] Could not load image. Check path/URL.")
            continue
            
        enc = processing.encode_single_face(img)
        if enc is not None:
            local_img_path = os.path.join(config.DATA_DIR, f"{name}.jpg")
            cv2.imwrite(local_img_path, img)
            
            registry["names"].append(name)
            registry["encs"].append(enc)
            registry["paths"].append(src)
            database.save_registry(registry)
            print(f"[SUCCESS] {name} successfully registered!")
        else:
            print(f"[ERROR] Could not find a clear face in that image. Try a different one.")

            
def manage_faces(registry):
    """Console UI sub-menu to rename or delete registered faces safely."""
    while True:
        if not registry["names"]:
            print("\n[INFO] The database is currently empty.")
            return

        print("\n" + "-"*30)
        print(" MANAGE KNOWN FACES ")
        print("-" *30)
        
        for i, name in enumerate(registry["names"]):
            print(f"{i + 1}) {name}")
            
        print("0) Go Back to Main Menu")
        
        choice = input("\nEnter the number of the person to manage (or 0 to exit) >>> ").strip()
        if choice == "0":
            break
            
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(registry["names"]):
                print("Invalid selection.")
                continue
        except ValueError:
            print("Please enter a valid number.")
            continue

        selected_name = registry["names"][idx]
        print(f"\nSelected: {selected_name}")
        print("1) Delete this person")
        print("2) Rename this person")
        print("0) Cancel")
        
        action = input(">>> ").strip()
        
        if action == "1":
            confirm = input(f"Are you sure you want to permanently delete {selected_name}? (y/n): ").strip().lower()
            if confirm == 'y':
                registry["names"].pop(idx)
                registry["encs"].pop(idx)
                registry["paths"].pop(idx)
                database.save_registry(registry)
                
                # Delete backup local image file
                img_path = os.path.join(config.DATA_DIR, f"{selected_name}.jpg")
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        print(f"[WARN] Could not delete image file: {e}")
                print(f"[SUCCESS] {selected_name} has been deleted.")
                
        elif action == "2":
            new_name = input(f"Enter the new name for {selected_name}: ").strip()
            if not new_name:
                continue
                
            new_name = database.get_unique_name(new_name, registry["names"])
            registry["names"][idx] = new_name
            database.save_registry(registry)
            
            # Rename backup local image file
            old_img_path = os.path.join(config.DATA_DIR, f"{selected_name}.jpg")
            new_img_path = os.path.join(config.DATA_DIR, f"{new_name}.jpg")
            if os.path.exists(old_img_path):
                try:
                    os.rename(old_img_path, new_img_path)
                except Exception as e:
                    print(f"[WARN] Database updated, but could not rename image file: {e}")
            print(f"[SUCCESS] Renamed {selected_name} to {new_name}.")
