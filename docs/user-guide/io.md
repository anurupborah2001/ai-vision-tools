# I/O

The `io` module handles reading and writing images and video, camera source management with auto-reconnect, and exporting datasets in standard formats.

```python
from ai_vision_tool.io import <ClassName>
```

---

## ImageReader / ImageWriter

```python
from ai_vision_tool.io import ImageReader, ImageWriter

# Read
reader = ImageReader()
image = reader.read("photo.jpg")

# Read with auto-orient (EXIF)
image = reader.read("photo.jpg", auto_orient=True)

# Write
writer = ImageWriter(output_dir="./output")
writer.write(image, filename="result.jpg")

# Pattern filenames (e.g. frame_0001.jpg, frame_0002.jpg)
writer = ImageWriter(output_dir="./frames", pattern="frame_{:04d}.jpg")
for i, frame in enumerate(frames):
    writer.write(frame, index=i)
```

---

## VideoReader

```python
from ai_vision_tool.io import VideoReader

reader = VideoReader("video.mp4")

# Iterate all frames
for frame in reader:
    process(frame)

# Read all at once
frames = reader.read_all()

# Seek to a specific frame
reader.seek(frame_number=100)
frame = reader.read_frame()

# Video metadata
print(reader.fps, reader.width, reader.height, reader.frame_count)

# Context manager
with VideoReader("video.mp4") as reader:
    for frame in reader:
        process(frame)
```

---

## VideoWriter

```python
from ai_vision_tool.io import VideoWriter

writer = VideoWriter(
    output_path="output.mp4",
    fps=30,
    width=1280,
    height=720,
    codec="mp4v",
)

for frame in frames:
    writer.write(frame)

writer.release()
```

---

## CameraSource

`CameraSource` wraps webcam, RTSP, and HTTP stream sources with auto-reconnect.

```python
from ai_vision_tool.io import CameraSource

# Webcam
cam = CameraSource(source=0)

# RTSP stream
cam = CameraSource(source="rtsp://192.168.1.100:554/stream", reconnect=True)

# HTTP MJPEG stream
cam = CameraSource(source="http://192.168.1.100/video.mjpg")

cam.setup({})
while True:
    frame = cam.grab()
    if frame is None:
        break
    process(frame)

cam.release()
```

---

## DatasetCollector

Collects images and labels while recording from a camera.

```python
from ai_vision_tool.io import DatasetCollector

collector = DatasetCollector(
    output_dir="./dataset",
    label_mode="manual",          # or "auto" with a detector
    image_format="jpg",
    max_images=1000,
)

collector.setup({})
collector.collect_from_camera(camera_index=0)
```

---

## DatasetExporter

Export annotated datasets in YOLO, COCO, or VOC formats.

```python
from ai_vision_tool.io import DatasetExporter

exporter = DatasetExporter(
    input_dir="./raw_dataset",
    output_dir="./yolo_dataset",
    format="yolo",                # "yolo" | "coco" | "voc"
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    class_names=["cat", "dog"],
)

exporter.export()
```

**YOLO output structure:**

```
yolo_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

---

## ImageExporter

Simple bulk export with optional format conversion.

```python
from ai_vision_tool.io import ImageExporter

exporter = ImageExporter(
    output_dir="./exported",
    format="png",
    quality=95,
)

for image, name in zip(images, names):
    exporter.export(image, name)
```
