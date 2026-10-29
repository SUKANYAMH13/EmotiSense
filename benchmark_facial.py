import os
import time
import numpy as np
import cv2
import dlib
from imutils import face_utils
from tensorflow.keras.models import load_model
from scipy.ndimage import zoom

# Paths
NEW_FACIAL_DIR = r"C:\Users\ADC\Downloads\MultimodalProject\MultimodalProject\video\Models"
TEST_IMAGE = r"C:\Users\ADC\Downloads\MultimodalProject\MultimodalProject\video\03-Video\video_app.png"

# Setup
print("Loading dlib detector...")
detector = dlib.get_frontal_face_detector()
print("Loading facial model...")
model = load_model(os.path.join(NEW_FACIAL_DIR, "video.h5"))
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

frame = cv2.imread(TEST_IMAGE)
if frame is None:
    # Use random noise if image fails
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

start_time = time.time()
rects = detector(gray, 0)
detect_time = time.time() - start_time
print(f"Detection took {detect_time:.4f}s. Found {len(rects)} faces.")

for i, rect in enumerate(rects):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    roi_gray = gray[y:min(y+h, gray.shape[0]), x:min(x+w, gray.shape[1])]
    
    if roi_gray.size == 0: continue
    
    start_p = time.time()
    roi_gray = zoom(roi_gray, (48 / roi_gray.shape[0], 48 / roi_gray.shape[1]))
    roi_gray = roi_gray.astype(np.float32)
    roi_max = float(roi_gray.max())
    if roi_max > 0: roi_gray /= roi_max
    roi = np.reshape(roi_gray.flatten(), (1, 48, 48, 1))
    preds = model.predict(roi, verbose=0)[0]
    predict_time = time.time() - start_p
    print(f"Face {i} prediction took {predict_time:.4f}s.")

total_time = time.time() - start_time
print(f"Total processing time: {total_time:.4f}s.")
