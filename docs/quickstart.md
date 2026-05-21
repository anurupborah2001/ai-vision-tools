# Quick Start

Three patterns cover 90% of use cases: single-image processing, pipeline composition, and live camera streaming.

---

## 1. Single Image Processing

```python
import cv2
from ai_vision_tool.preprocessing import Resize, CLAHE, Denoise

image = cv2.imread("photo.jpg")

# Components are stateless — call .run() directly
image = Resize(640, 480).run(image)
image = CLAHE(clip_limit=3.0).run(image)
image = Denoise(method="bilateral").run(image)

cv2.imwrite("output.jpg", image)
```

---

## 2. Pipeline Composition

Chain components with `AIVisionPipeline`. Each component receives the output of the previous one.

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize, CLAHE, WhiteBalance
from ai_vision_tool.augmentation import Flip, Brightness
from ai_vision_tool.visualization import FrameAnnotator

pipeline = AIVisionPipeline([
    Resize(width=640, height=480),
    WhiteBalance(),
    CLAHE(clip_limit=2.5),
    Flip(direction="horizontal"),
    Brightness(factor=1.1),
    FrameAnnotator(label="demo"),
])

import cv2
image = cv2.imread("photo.jpg")
result = pipeline.execute(image)

# result is a payload dict; "frame" holds the processed image
cv2.imwrite("result.jpg", result["frame"])
```

### Payload dict convention

When a component returns metadata alongside the image, it uses a dict:

```python
payload = {
    "frame": numpy_array,     # always present
    "bboxes": [...],           # optional: detected boxes
    "masks": [...],            # optional: segmentation masks
    "tracks": [...],           # optional: tracked objects
}
```

Downstream components receive the full dict and extract what they need.

---

## 3. Live Camera / Webcam

```python
import cv2
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize
from ai_vision_tool.visualization import FrameViewer

pipeline = AIVisionPipeline([
    Resize(640, 480),
    FrameViewer(window_title="Live Feed"),
])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pipeline.execute(frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Or use the built-in CLI:

```bash
ai-vision-tool --webcam
```

---

## 4. Batch Directory Processing

```python
from ai_vision_tool.core.batch_processor import BatchProcessor
from ai_vision_tool.preprocessing import Resize, CLAHE

processor = BatchProcessor(
    components=[Resize(640, 480), CLAHE()],
    max_workers=4,
)
results = processor.process_directory("./input_images", extensions=[".jpg", ".png"])
print(f"Processed {len(results)} images")
```

---

## 5. Object Detection

```python
# Requires: pip install "ai-vision-tool[detection]"
import cv2
from ai_vision_tool.detection import ObjectDetector
from ai_vision_tool.visualization import BBoxRenderer

detector = ObjectDetector(model="yolov8n.pt", confidence=0.5)
renderer = BBoxRenderer()

image = cv2.imread("street.jpg")
payload = detector.run(image)
output = renderer.run(payload)
cv2.imwrite("detected.jpg", output["frame"])
```

---

## 6. Save and Load Pipelines

```python
from ai_vision_tool.pipelines.serializer import PipelineSerializer

# Save
PipelineSerializer.save(pipeline, "my_pipeline.yaml")

# Load
pipeline = PipelineSerializer.load("my_pipeline.yaml")
result = pipeline.execute(image)
```

---

## 7. Use a Prebuilt Pipeline

```python
from ai_vision_tool.pipelines.prebuilt import PrebuiltPipelines

# Ready-made augmentation pipeline for dataset preparation
pipeline = PrebuiltPipelines.augmentation(target_size=(640, 640))
result = pipeline.execute(image)
```

Available prebuilts: `detection`, `augmentation`, `tracking`, `segmentation`, `preprocessing`.

---

## Next Steps

- [Preprocessing guide](user-guide/preprocessing.md) — geometry, intensity, quality filters
- [Augmentation guide](user-guide/augmentation.md) — 50+ augmentation transforms
- [Pipeline guide](user-guide/pipelines.md) — async, parallel, serialization
- [Detection guide](user-guide/detection.md) — YOLO, MediaPipe, EAST OCR
- [CLI reference](cli.md) — all command-line options
