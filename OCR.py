import cv2
import numpy as np
import threading
import time
import os
import pytesseract

# --- NUCLEAR FIX FOR MEDIAPIPE ---
try:
    import mediapipe as mp
    # Manually trigger the loading of sub-modules
    mp_pose = mp.solutions.pose
    mp_selfie = mp.solutions.selfie_segmentation
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    print("✅ MediaPipe Solutions loaded successfully!")
except AttributeError:
    print("❌ ERROR: MediaPipe is still having attribute issues.")
    print("print(Try running: pip install mediapipe>=0.10.5)")
    exit()

# --- CONFIGURATION ---
# Ensure this matches your folder name 'Tessaract' vs 'Tesseract'
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print(f"⚠️ WARNING: Tesseract not found at {tesseract_path}")

class SharedState:
    def __init__(self):
        self.current_topic = "Waiting for slide..."
        self.lock = threading.Lock()
    def update_topic(self, text):
        with self.lock:
            if len(text) > 5:
                self.current_topic = text.replace("\n", " ").strip()
    def get_topic(self):
        with self.lock: return self.current_topic

state = SharedState()

def slide_reader_thread(frame, seg_instance):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = seg_instance.process(rgb_frame)
        if res.segmentation_mask is not None:
            # Create a mask where the human is removed
            mask = np.stack((res.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            clean_slide = np.where(mask, bg_image, frame)
            
            # OCR with --psm 3 (Automatic page segmentation)
            text = pytesseract.image_to_string(clean_slide, config='--psm 3')
            state.update_topic(text)
    except Exception as e:
        print(f"OCR Thread Error: {e}")

def main():
    cap = cv2.VideoCapture(0)
    
    # Initialize trackers ONCE outside the loop
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    sel_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    
    last_ocr_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Pose Tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 2. OCR (Every 4 seconds to save CPU)
        if time.time() - last_ocr_time > 4.0:
            threading.Thread(target=slide_reader_thread, args=(frame.copy(), sel_seg), daemon=True).start()
            last_ocr_time = time.time()

        # 3. UI
        topic = state.get_topic()
        cv2.rectangle(frame, (0,0), (640, 70), (0,0,0), -1)
        cv2.putText(frame, f"Topic: {topic[:45]}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.imshow('SynthSpeak AI', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()