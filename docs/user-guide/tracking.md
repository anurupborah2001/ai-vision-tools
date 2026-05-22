# Tracking

Multi-object tracking connects detections across frames by maintaining consistent IDs. All trackers accept a `payload` dict with `bboxes` from a detector.

---

## Architecture

```
Detection → TrackManager (IoU + Kalman) → output payload["tracks"]
         ↘ ByteTracker (two-stage)       ↗
         ↘ DeepSORT (appearance cues)   ↗
```

Each `Track` has: `track_id`, `bbox`, `state` (tentative/active/lost), and `age`.

---

## TrackManager (SORT-style)

Pure IoU-based association with a Kalman filter for motion prediction.

```python
from ai_vision_tool.tracking import TrackManager

tracker = TrackManager(
    max_age=30,          # frames before a lost track is deleted
    min_hits=3,          # frames before a track becomes active
    iou_threshold=0.3,
)

for frame in video_frames:
    payload = detector.run(frame)
    payload = tracker.run(payload)
    for track in payload["tracks"]:
        print(track.track_id, track.bbox)
```

---

## ByteTracker

Two-stage association: high-confidence detections first, then low-confidence ones matched against lost tracks. Better recall in crowded scenes.

```python
from ai_vision_tool.tracking import ByteTracker

tracker = ByteTracker(
    track_thresh=0.5,        # high-confidence threshold
    track_buffer=30,         # frames to keep lost tracks
    match_thresh=0.8,        # IoU threshold for association
)

payload = tracker.run({"frame": frame, "bboxes": detections})
```

---

## DeepSORT

Combines motion prediction (Kalman) with appearance embedding (HOG or ONNX ReID model) for robust re-identification after occlusion.

> **Extra:** `pip install "ai-vision-tool[tracking]"` for ONNX ReID.

```python
from ai_vision_tool.tracking import DeepSORTTracker

tracker = DeepSORTTracker(
    max_age=50,
    n_init=3,
    max_cosine_distance=0.3,
    reid_model=None,            # use HOG embedding (default)
)

payload = tracker.run({"frame": frame, "bboxes": detections})
```

### With ONNX ReID Model

```python
from ai_vision_tool.tracking import ReIDExtractor, DeepSORTTracker

reid = ReIDExtractor(model_path="osnet.onnx")
tracker = DeepSORTTracker(reid_model=reid, max_cosine_distance=0.25)
```

---

## ReIDExtractor

Extract appearance embeddings for gallery-based re-identification.

```python
from ai_vision_tool.tracking import ReIDExtractor

extractor = ReIDExtractor(model_path="osnet.onnx")

# Build a gallery from reference images
gallery = extractor.build_gallery({"person_1": [img1, img2], "person_2": [img3]})

# Query
embedding = extractor.run({"frame": crop_image})
```

---

## Kalman Filter (low-level)

Direct access to the 7-state SORT Kalman filter:

```python
from ai_vision_tool.tracking import KalmanFilter

kf = KalmanFilter()
kf.initiate(bbox_to_xyah(detection))   # [cx, cy, aspect_ratio, height]
mean, cov = kf.predict(mean, cov)
mean, cov = kf.update(mean, cov, measurement)
```

---

## Tracking in a Pipeline

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize
from ai_vision_tool.detection import ObjectDetector
from ai_vision_tool.tracking import ByteTracker
from ai_vision_tool.visualization import BBoxRenderer

pipeline = AIVisionPipeline([
    Resize(1280, 720),
    ObjectDetector(model="yolov8n.pt", confidence=0.4),
    ByteTracker(track_thresh=0.5),
    BBoxRenderer(show_track_id=True),
])

cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = pipeline.execute(frame)
    cv2.imshow("Tracking", result["frame"])
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```
