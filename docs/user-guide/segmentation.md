# Segmentation

> **Extra required:** `pip install "ai-vision-tool[segmentation]"` for YOLO-seg, SAM, and torch backends.

Segmentation components populate `payload["masks"]` — a list of binary NumPy arrays (one per detected region).

---

## Semantic Segmentation

`SemanticSegmenter` assigns a class label to each pixel. Supports ONNX, OpenCV DNN, and PyTorch backends with VOC 21-class labels by default.

```python
from ai_vision_tool.segmentation import SemanticSegmenter

segmenter = SemanticSegmenter(
    model_path="deeplabv3.onnx",
    backend="onnx",
    input_size=(513, 513),
)

import cv2
image = cv2.imread("cityscape.jpg")
payload = segmenter.run(image)
# payload["semantic_mask"] → H×W label map (int)
# payload["colored_mask"]  → H×W×3 colorized visualization
```

---

## Instance Segmentation

`InstanceSegmenter` wraps YOLO-seg to produce per-object binary masks alongside bounding boxes.

```python
from ai_vision_tool.segmentation import InstanceSegmenter

segmenter = InstanceSegmenter(model="yolov8n-seg.pt", confidence=0.5)
payload = segmenter.run(image)
# payload["bboxes"] → [[x1, y1, x2, y2, score, class_id], ...]
# payload["masks"]  → list of binary H×W masks
```

Render masks:

```python
from ai_vision_tool.utils import DrawUtils

annotated = DrawUtils.draw_masks(image, payload["masks"], alpha=0.5)
```

---

## Panoptic Segmentation

`PanopticSegmenter` separates "things" (countable objects) from "stuff" (uncountable background regions).

```python
from ai_vision_tool.segmentation import PanopticSegmenter

segmenter = PanopticSegmenter(model_path="panoptic.onnx")
payload = segmenter.run(image)
# payload["panoptic_map"]  → H×W map encoding both stuff and things
# payload["segments_info"] → list of {"id", "category_id", "is_thing", "area"}
```

---

## Segment Anything (SAM)

`SAMSegmenter` wraps Meta's Segment Anything Model. Supports point prompts, box prompts, and automatic everything mode.

```python
from ai_vision_tool.segmentation import SAMSegmenter

sam = SAMSegmenter(
    model_type="vit_b",             # vit_b | vit_l | vit_h
    checkpoint="sam_vit_b.pth",
    device="cuda",
)

# Point prompt
payload = sam.run({
    "frame": image,
    "points": [(320, 240)],         # click coordinates
    "point_labels": [1],            # 1=foreground, 0=background
})

# Box prompt
payload = sam.run({
    "frame": image,
    "boxes": [[100, 50, 600, 400]], # [x1, y1, x2, y2]
})

# Auto everything (no prompt — segment all objects)
payload = sam.run(image)
```

---

## Mask Post-Processing

`MaskPostProcessor` refines raw segmentation masks.

```python
from ai_vision_tool.segmentation import MaskPostProcessor

processor = MaskPostProcessor(
    operations=["erode", "dilate", "fill_holes", "largest_only"],
    kernel_size=5,
    iterations=2,
)

payload = processor.run({"frame": image, "masks": masks})
```

**Available operations:**

| Operation | Effect |
|-----------|--------|
| `erode` | Shrink mask edges |
| `dilate` | Expand mask edges |
| `fill_holes` | Fill enclosed holes in mask |
| `largest_only` | Keep only the largest connected component |
| `smooth` | Morphological open/close for smoother boundary |

---

## Pipeline Integration

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize
from ai_vision_tool.segmentation import InstanceSegmenter, MaskPostProcessor
from ai_vision_tool.visualization import BBoxRenderer

pipeline = AIVisionPipeline([
    Resize(640, 640),
    InstanceSegmenter(model="yolov8n-seg.pt", confidence=0.5),
    MaskPostProcessor(operations=["fill_holes"]),
    BBoxRenderer(draw_masks=True, mask_alpha=0.4),
])

result = pipeline.execute(image)
```
