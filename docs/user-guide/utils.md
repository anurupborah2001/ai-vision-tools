# Utilities

General-purpose helpers used across the library and available for end-user code.

```python
from ai_vision_tool.utils import <ClassName>
```

---

## ColorPalette

Generate visually distinct colors for class labels, bounding boxes, or heatmaps using golden-ratio hue distribution.

```python
from ai_vision_tool.utils import ColorPalette

palette = ColorPalette(num_colors=80)

# Get color for a class index (BGR tuple)
color = palette.get(class_id=3)       # e.g. (120, 200, 80)

# Get all colors as a dict
colors_dict = palette.as_dict()       # {0: (r,g,b), 1: (r,g,b), ...}

# Use with BBoxRenderer
from ai_vision_tool.visualization import BBoxRenderer
renderer = BBoxRenderer(color_palette=palette)
```

---

## MetricsLogger

Log and aggregate component-level metrics (latency, throughput, counts) with pipeline integration.

```python
from ai_vision_tool.utils import MetricsLogger, MetricsLoggerComponent

# Standalone logger
logger = MetricsLogger(name="detector", log_interval=100)
logger.log("latency_ms", 12.3)
logger.log("detections", 5)
logger.report()                  # prints summary every log_interval calls

# As a pipeline component
pipeline = AIVisionPipeline([
    Resize(640, 480),
    ObjectDetector(model="yolov8n.pt"),
    MetricsLoggerComponent(name="pipeline", log_interval=50),
])
```

---

## FrameSampler

Sub-sample video frames by count, rate, or randomly.

```python
from ai_vision_tool.utils import FrameSampler

# Sample every Nth frame (by count)
sampler = FrameSampler(mode="count", n=5)

# Sample at a target FPS from a source FPS
sampler = FrameSampler(mode="fps", source_fps=30, target_fps=5)

# Random sampling (probability per frame)
sampler = FrameSampler(mode="random", probability=0.1)

for i, frame in enumerate(all_frames):
    if sampler.should_sample(i):
        process(frame)
```

---

## ImageHash

Perceptual hashing for duplicate detection and near-duplicate search.

```python
from ai_vision_tool.utils import ImageHash

# Compute hash
h = ImageHash(method="phash")             # phash | ahash | dhash
hash_val = h.compute(image)

# Compare two images
distance = h.distance(hash_val, other_hash)
is_duplicate = distance < 8               # threshold 0–64

# Batch duplicate detection
hasher = ImageHash(method="phash")
images = [img1, img2, img3, ...]
duplicates = hasher.find_duplicates(images, threshold=8)
# duplicates → list of (idx_a, idx_b, distance)
```

---

## DrawUtils

Low-level drawing primitives for custom renderers.

```python
from ai_vision_tool.utils import DrawUtils

# Draw bounding boxes
annotated = DrawUtils.draw_bboxes(
    image,
    bboxes=[[100, 50, 300, 250, 0.9, 0]],
    labels=["cat"],
    colors=[(0, 255, 0)],
    thickness=2,
    font_scale=0.6,
)

# Draw segmentation masks with transparency
annotated = DrawUtils.draw_masks(image, masks=mask_list, alpha=0.4)

# Draw pose skeleton
annotated = DrawUtils.draw_keypoints(
    image,
    keypoints=kp_list,
    draw_skeleton=True,
    joint_color=(0, 255, 0),
    limb_color=(255, 0, 0),
)
```
