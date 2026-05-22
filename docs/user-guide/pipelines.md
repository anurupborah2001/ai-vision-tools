# Pipelines

`AIVisionPipeline` implements a Chain-of-Responsibility pattern. Each component receives the output of the previous one as its input.

---

## Basic Pipeline

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize, CLAHE
from ai_vision_tool.augmentation import Flip, Noise

pipeline = AIVisionPipeline([
    Resize(640, 480),
    CLAHE(clip_limit=3.0),
    Flip(direction="horizontal"),
    Noise(intensity=0.02),
])

import cv2
image = cv2.imread("input.jpg")
result = pipeline.execute(image)
cv2.imwrite("output.jpg", result["frame"])
```

---

## Global Config

Pass a `global_config` dict to `execute()` — it is forwarded unchanged to every component's `run(data, config)` call.

```python
result = pipeline.execute(image, global_config={"probability": 0.5, "seed": 42})
```

---

## Prebuilt Pipelines

Ready-made pipelines for common workflows:

```python
from ai_vision_tool.pipelines.prebuilt import PrebuiltPipelines

# Augmentation pipeline for training data generation
pipeline = PrebuiltPipelines.augmentation(target_size=(640, 640))

# Object detection pipeline
pipeline = PrebuiltPipelines.detection(
    model="yolov8n.pt", confidence=0.5, target_size=(640, 640)
)

# Multi-object tracking pipeline
pipeline = PrebuiltPipelines.tracking(model="yolov8n.pt")

# Instance segmentation pipeline
pipeline = PrebuiltPipelines.segmentation(model="yolov8n-seg.pt")

# Standard preprocessing pipeline
pipeline = PrebuiltPipelines.preprocessing(target_size=(640, 640))
```

---

## Serialization

Save and load pipelines as YAML or JSON:

```python
from ai_vision_tool.pipelines.serializer import PipelineSerializer

# Save to YAML
PipelineSerializer.save(pipeline, "pipeline.yaml")

# Load from YAML
pipeline = PipelineSerializer.load("pipeline.yaml")

# Save as JSON
PipelineSerializer.save(pipeline, "pipeline.json")

# Dict round-trip
config_dict = PipelineSerializer.to_dict(pipeline)
pipeline = PipelineSerializer.from_dict(config_dict)
```

**Example pipeline.yaml:**

```yaml
components:
  - class: Resize
    module: ai_vision_tool.preprocessing.geometry
    params:
      width: 640
      height: 480
      interpolation: linear
  - class: CLAHE
    module: ai_vision_tool.preprocessing.intensity
    params:
      clip_limit: 3.0
      tile_grid_size: [8, 8]
  - class: Flip
    module: ai_vision_tool.augmentation.flip
    params:
      direction: horizontal
      probability: 0.5
```

---

## Async Pipeline

`AsyncPipeline` runs component execution in a thread-pool executor without blocking the event loop.

```python
import asyncio
from ai_vision_tool.pipelines.async_pipeline import AsyncPipeline
from ai_vision_tool.preprocessing import Resize, CLAHE

async def main():
    pipeline = AsyncPipeline([Resize(640, 480), CLAHE()])
    result = await pipeline.execute(image)
    return result

result = asyncio.run(main())
```

---

## Parallel Pipeline

`ParallelPipeline` executes multiple independent branches concurrently using `ThreadPoolExecutor`.

```python
from ai_vision_tool.pipelines.parallel_pipeline import ParallelPipeline, FanOutPipeline
from ai_vision_tool.preprocessing import Resize, CLAHE
from ai_vision_tool.augmentation import Flip, Rotation

# Fan-out: run branches on the same input, collect all results
fan_out = FanOutPipeline(
    branches=[
        [Flip(direction="horizontal")],
        [Rotation(angle=15)],
        [CLAHE()],
    ]
)

results = fan_out.execute(image)
# results is a list, one entry per branch
```

---

## Custom Components

Create a component by subclassing `AIVisionComponent` and implementing `_execute`:

```python
from ai_vision_tool.core.base import AIVisionComponent
import cv2
import numpy as np

class Vignette(AIVisionComponent):
    def __init__(self, strength=0.5):
        super().__init__()
        self.strength = strength

    def _execute(self, data, config):
        from ai_vision_tool.utils.image_utils import extract_frame, replace_frame
        frame = extract_frame(data)
        h, w = frame.shape[:2]
        # Build radial mask
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        mask = 1 - self.strength * ((X - cx)**2 + (Y - cy)**2) / (cx**2 + cy**2)
        mask = np.clip(mask, 0, 1)[..., np.newaxis]
        vignetted = (frame * mask).astype(np.uint8)
        return replace_frame(data, vignetted)
```

Drop it into any pipeline:

```python
pipeline = AIVisionPipeline([Resize(640, 480), Vignette(strength=0.6)])
```
