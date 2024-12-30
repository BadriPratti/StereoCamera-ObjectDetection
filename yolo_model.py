from ultralytics import YOLO
from config import YOLO_MODEL_PATH, DEVICE

def load_yolo_model():
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH).to(DEVICE)
    return model

def detect_objects(model, frame):
    results = model(frame, verbose=False)
    return results[0].boxes
