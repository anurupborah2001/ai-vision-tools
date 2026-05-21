# Visualization

The `visualization` module renders bounding boxes, keypoints, masks, heatmaps, and text overlays onto frames. All components return a payload dict with `"frame"` holding the annotated image.

```python
from ai_vision_tool.visualization import <ClassName>
```

---

## FrameViewer

Displays frames in an OpenCV window with optional FPS overlay. Headless-safe — skips display if no display server is available.

```python
from ai_vision_tool.visualization import FrameViewer

viewer = FrameViewer(window_title="Live Feed", show_fps=True)

for frame in frames:
    payload = viewer.run(frame)     # shows window, returns payload
    if payload.get("quit"):
        break                        # user pressed 'q'
```

---

## FrameAnnotator

Adds text, lines, and rectangles to frames.

```python
from ai_vision_tool.visualization import FrameAnnotator

annotator = FrameAnnotator(
    label="My Pipeline",
    font_scale=0.7,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
)

output = annotator.run(image)
```

---

## BBoxRenderer

Renders bounding boxes with labels, confidence scores, and optional track IDs. Uses `ColorPalette` to assign unique colors per class.

```python
from ai_vision_tool.visualization import BBoxRenderer

renderer = BBoxRenderer(
    show_labels=True,
    show_confidence=True,
    show_track_id=True,
    line_thickness=2,
    font_scale=0.6,
    alpha=0.3,              # fill transparency (0=no fill, 1=solid)
    draw_masks=True,        # draw instance segmentation masks
    mask_alpha=0.4,
)

payload = {"frame": image, "bboxes": [[100, 50, 300, 250, 0.92, 0]]}
output = renderer.run(payload)
cv2.imwrite("annotated.jpg", output["frame"])
```

---

## HeatmapRenderer

Overlays Gaussian-blob heatmaps, motion maps, or anomaly score maps onto frames.

```python
from ai_vision_tool.visualization import HeatmapRenderer

# Gaussian blob at keypoint locations
renderer = HeatmapRenderer(mode="keypoint", sigma=15)
output = renderer.run({"frame": image, "keypoints": [(320, 240), (100, 150)]})

# Motion heatmap (accumulates motion over frames)
renderer = HeatmapRenderer(mode="motion", decay=0.95)
output = renderer.run({"frame": frame, "motion_mask": motion})

# Anomaly score map
renderer = HeatmapRenderer(mode="anomaly", colormap="jet")
output = renderer.run({"frame": image, "anomaly_map": score_map})
```

---

## DashboardSink

Serves annotated frames as a live Gradio app or MJPEG HTTP stream.

```python
from ai_vision_tool.visualization import DashboardSink

# Gradio dashboard (requires gradio)
sink = DashboardSink(backend="gradio", port=7860, title="AI Vision Tool Demo")
sink.setup({})

# MJPEG HTTP server (no extra required)
sink = DashboardSink(backend="mjpeg", host="0.0.0.0", port=8080)
sink.setup({})

for frame in camera_stream:
    payload = pipeline.execute(frame)
    sink.run(payload)          # pushes frame to dashboard
```

Access at `http://localhost:8080` in a browser.

---

## VideoAnnotationExporter

Burns annotations into a video file and writes a JSON sidecar with per-frame metadata.

```python
from ai_vision_tool.visualization import VideoAnnotationExporter

exporter = VideoAnnotationExporter(
    output_video="annotated.mp4",
    output_json="annotations.json",
    fps=30,
)

exporter.setup({})
for frame in frames:
    payload = detection_pipeline.execute(frame)
    exporter.run(payload)
exporter.cleanup()
```

**JSON sidecar format:**

```json
[
  {
    "frame_index": 0,
    "timestamp_ms": 0,
    "bboxes": [[100, 50, 300, 250, 0.92, 0]],
    "tracks": [{"track_id": 1, "bbox": [100, 50, 300, 250]}]
  },
  ...
]
```

---

## DrawUtils

Low-level drawing utilities — useful when building custom renderers.

```python
from ai_vision_tool.utils import DrawUtils

# Draw bounding boxes
annotated = DrawUtils.draw_bboxes(image, bboxes, labels=class_names, colors=colors)

# Draw segmentation masks
annotated = DrawUtils.draw_masks(image, masks, alpha=0.4, colors=colors)

# Draw pose keypoints and skeleton
annotated = DrawUtils.draw_keypoints(image, keypoints, draw_skeleton=True)
```
