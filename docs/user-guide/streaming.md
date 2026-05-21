# Streaming

The `streaming` module provides iterator-based frame sources with context-manager support, RTSP client with background reading and reconnect, and buffered sliding-window streams.

```python
from ai_vision_tool.streaming import <ClassName>
```

---

## FrameStream

Iterate over a video file or camera source as a context manager.

```python
from ai_vision_tool.streaming import FrameStream

# From video file
with FrameStream("video.mp4") as stream:
    for frame in stream:
        process(frame)

# From webcam
with FrameStream(0) as stream:
    for i, frame in enumerate(stream):
        if i >= 100:
            break
        process(frame)
```

---

## DirectoryStream

Iterate over all images in a directory in sorted order.

```python
from ai_vision_tool.streaming import DirectoryStream

with DirectoryStream("./images", extensions=[".jpg", ".png"]) as stream:
    for path, frame in stream:
        print(f"Processing {path}")
        process(frame)
```

---

## RTSPClient

Background reader that decodes an RTSP stream in a separate thread and provides the latest frame on demand. Auto-reconnects on network drop.

```python
from ai_vision_tool.streaming import RTSPClient

client = RTSPClient(
    url="rtsp://192.168.1.100:554/h264/ch1/main/av_stream",
    reconnect=True,
    reconnect_delay=2.0,
    buffer_size=5,
)

client.start()

try:
    while True:
        frame = client.read()
        if frame is None:
            continue
        result = pipeline.execute(frame)
        cv2.imshow("RTSP", result["frame"])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    client.stop()
```

---

## BufferedStream

Wraps any frame source with a queue and drop policy, so slow consumers don't block the producer.

```python
from ai_vision_tool.streaming import BufferedStream

raw_stream = FrameStream("video.mp4")
buffered = BufferedStream(
    source=raw_stream,
    max_buffer=10,
    drop_policy="oldest",    # "oldest" | "newest" | "none"
)

buffered.start()
try:
    while True:
        frame = buffered.get()
        if frame is None:
            break
        process(frame)
finally:
    buffered.stop()
```

---

## SlidingWindowBuffer

Accumulates a rolling window of N recent frames for temporal analysis (optical flow, background subtraction, etc.).

```python
from ai_vision_tool.streaming import SlidingWindowBuffer

window = SlidingWindowBuffer(window_size=10)

for frame in stream:
    window.push(frame)
    if window.is_full():
        frames = window.get()   # list of 10 most recent frames
        analyze_temporal(frames)
```

---

## Combining with Pipelines

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize
from ai_vision_tool.detection import ObjectDetector
from ai_vision_tool.visualization import BBoxRenderer
from ai_vision_tool.streaming import RTSPClient

pipeline = AIVisionPipeline([
    Resize(1280, 720),
    ObjectDetector(model="yolov8n.pt", confidence=0.4),
    BBoxRenderer(show_labels=True),
])

client = RTSPClient("rtsp://camera.local/stream", reconnect=True)
client.start()

try:
    while True:
        frame = client.read()
        if frame is not None:
            result = pipeline.execute(frame)
            cv2.imshow("Stream", result["frame"])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    client.stop()
```
