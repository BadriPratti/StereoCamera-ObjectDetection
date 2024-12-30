import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
YOLO_MODEL_PATH = '/home/oshkosh/stereo_detect/doorv1weights.pt'
CONFIDENCE_THRESHOLD = 0.04

# Camera Parameters
BASELINE = 0.2413  # meters
FOCAL_LENGTH = 645.0  # pixels
