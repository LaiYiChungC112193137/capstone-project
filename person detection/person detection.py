# Colab-ready person detector using YOLOv5 (no command-line args)
# Set IMAGE_PATH to the image you want to analyze
IMAGE_PATH = "/content/test.jpg"  # <- change this to your image path

# Install dependencies (Colab usually has torch; this ensures yolov5 helper code is available)
!pip install -q --upgrade pip
!pip install -q opencv-python pillow ultralytics

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image

# Load YOLOv5 model from ultralytics via torch.hub (works in Colab)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
model.conf = 0.25  # confidence threshold
model.iou = 0.45   # NMS IoU threshold

# Verify image exists
img_path = Path(IMAGE_PATH)
if not img_path.exists():
    raise FileNotFoundError(f"Image not found: {img_path}")

# Run inference
results = model(str(img_path))

# results.xyxy[0] -> tensor of detections: [x1, y1, x2, y2, conf, cls]
detections = results.xyxy[0].cpu().numpy()

# Filter for person class (COCO id 0)
person_dets = [d for d in detections if int(d[5]) == 0]

# Load image with OpenCV (BGR) and draw boxes
img = cv2.imread(str(img_path))
if img is None:
    raise RuntimeError("Failed to load image with OpenCV")

def draw_boxes_bgr(img_bgr, detections, conf_threshold=0.25):
    for d in detections:
        x1, y1, x2, y2, conf, cls = d
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"person {conf:.2f}"
        color = (10, 255, 0)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        t_w, t_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(img_bgr, (x1, y1 - t_h - 6), (x1 + t_w + 6, y1), color, -1)
        cv2.putText(img_bgr, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

draw_boxes_bgr(img, person_dets, conf_threshold=model.conf)

# Save and display result
out_path = Path("output_persons.jpg")
cv2.imwrite(str(out_path), img)

# Convert BGR->RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,8))
plt.axis('off')
plt.imshow(img_rgb)
plt.title(f"Persons detected: {len(person_dets)}")
plt.show()

print(f"Saved annotated image to: {out_path}")
