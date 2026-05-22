# Detection

> **Extra required:** `pip install "ai-vision-tool[detection]"` for YOLO and MediaPipe backends.

---

## Object Detection

`ObjectDetector` supports YOLO (via `ultralytics`) and ONNX-based detectors with built-in greedy NMS.

```python
from ai_vision_tool.detection import ObjectDetector
from ai_vision_tool.visualization import BBoxRenderer

# YOLO model
detector = ObjectDetector(model="yolov8n.pt", confidence=0.5, iou=0.45)

import cv2
image = cv2.imread("street.jpg")
payload = detector.run(image)
# payload["bboxes"]  → list of [x1, y1, x2, y2, score, class_id]

rendered = BBoxRenderer().run(payload)
cv2.imwrite("detected.jpg", rendered["frame"])
```

### ONNX Object Detector

```python
detector = ObjectDetector(
    model="yolov8n.onnx",
    input_size=(640, 640),
    confidence=0.4,
)
payload = detector.run(image)
```

### Custom Class Labels

```python
detector = ObjectDetector(
    model="best.pt",
    class_names=["cat", "dog", "person"],
    confidence=0.5,
)
```

---

## Face Detection

`FaceDetector` supports OpenCV Haar cascade and MediaPipe Face Detection.

```python
from ai_vision_tool.detection import FaceDetector

# OpenCV Haar (no extra deps)
detector = FaceDetector(backend="haar", scale_factor=1.1, min_neighbors=5)
payload = detector.run(image)
# payload["bboxes"] → face bounding boxes

# MediaPipe (requires [detection] extra)
detector = FaceDetector(backend="mediapipe", min_detection_confidence=0.7)
payload = detector.run(image)
```

---

## Keypoint / Pose Detection

`KeypointDetector` wraps MediaPipe Pose and YOLO-Pose.

```python
from ai_vision_tool.detection import KeypointDetector

# MediaPipe Pose
detector = KeypointDetector(backend="mediapipe", min_detection_confidence=0.7)
payload = detector.run(image)
# payload["keypoints"] → list of Keypoint(x, y, z, visibility, name)

# YOLO-Pose
detector = KeypointDetector(backend="yolo", model="yolov8n-pose.pt")
payload = detector.run(image)
```

Draw pose keypoints:

```python
from ai_vision_tool.utils import DrawUtils

annotated = DrawUtils.draw_keypoints(image, payload["keypoints"])
```

---

## Text Detection / OCR

`TextDetector` supports EasyOCR, PaddleOCR, and EAST.

> **Extra required:** install EasyOCR or PaddleOCR separately.

```python
from ai_vision_tool.detection import TextDetector

# EasyOCR
detector = TextDetector(backend="easyocr", languages=["en"])
payload = detector.run(image)
# payload["text_regions"] → list of {"bbox": ..., "text": ..., "confidence": ...}

# PaddleOCR
detector = TextDetector(backend="paddleocr")

# EAST (detection only, no recognition — pure OpenCV)
detector = TextDetector(backend="east", confidence=0.5)
```

---

## Anomaly Detection

`AnomalyDetector` identifies anomalous regions using statistical, PatchCore, or PCA methods.

```python
from ai_vision_tool.detection import AnomalyDetector

# Fit on normal images
detector = AnomalyDetector(method="patchcore")
detector.fit(normal_images)

# Score a test image
payload = detector.run(test_image)
# payload["anomaly_map"] → heatmap (higher = more anomalous)
# payload["anomaly_score"] → scalar score
```

Visualize with `HeatmapRenderer`:

```python
from ai_vision_tool.visualization import HeatmapRenderer

rendered = HeatmapRenderer(mode="anomaly").run(payload)
```

---

## Using Detectors in a Pipeline

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize, CLAHE
from ai_vision_tool.detection import ObjectDetector
from ai_vision_tool.visualization import BBoxRenderer

pipeline = AIVisionPipeline([
    Resize(640, 640),
    CLAHE(),
    ObjectDetector(model="yolov8n.pt", confidence=0.45),
    BBoxRenderer(show_labels=True, show_confidence=True),
])

result = pipeline.execute(image)
```
