# Capture

The `capture` module provides components for acquiring images and video from cameras, files, and screen — with motion detection, ROI selection, burst capture, and timelapse.

```python
from ai_vision_tool.capture import <ClassName>
```

---

## Single Image Capture

```python
from ai_vision_tool.capture import PictureTaker

taker = PictureTaker(camera_index=0, output_dir="./captures")
taker.setup({})

# Take one picture
filepath = taker.capture(filename="snapshot.jpg")
print(f"Saved to: {filepath}")
```

---

## Burst Image Capture

```python
from ai_vision_tool.capture import BurstPictureTaker

burst = BurstPictureTaker(
    camera_index=0,
    output_dir="./burst",
    num_frames=10,
    delay_ms=100,    # 100ms between frames
)

burst.setup({})
filepaths = burst.capture_burst()
```

---

## Video Recording

```python
from ai_vision_tool.capture import VideoTaker, VideoRecorder

# Simple video recorder
recorder = VideoRecorder(
    camera_index=0,
    output_path="output.mp4",
    fps=30,
    codec="mp4v",
    duration_seconds=10,
)
recorder.setup({})
recorder.record()

# Streaming-style (manual control)
taker = VideoTaker(camera_index=0, output_path="out.avi", fps=30)
taker.setup({})
taker.start_recording()
# ... do work ...
taker.stop_recording()
```

---

## Frame Grabber

Grab individual frames from a camera or video file on demand.

```python
from ai_vision_tool.capture import FrameGrabber

grabber = FrameGrabber(source=0)         # webcam
grabber = FrameGrabber(source="vid.mp4") # video file
grabber.setup({})

frame = grabber.grab()                   # single frame
for frame in grabber.grab_n(frames=30): # N frames
    process(frame)
```

---

## ROI Capture

Let the user draw a region of interest interactively, then capture only that region.

```python
from ai_vision_tool.capture import ROICapture

roi = ROICapture(camera_index=0, window_title="Select ROI")
roi.setup({})

# Opens a window — user draws rectangle, then presses ENTER
roi_payload = roi.capture()
# roi_payload["frame"] is the cropped ROI image
```

---

## Motion Detection

Capture frames only when significant motion is detected.

```python
from ai_vision_tool.capture import MotionDetector

detector = MotionDetector(
    threshold=25,           # pixel difference threshold
    min_area=500,           # minimum contour area (px²)
    output_dir="./motion",
    save_on_motion=True,
)
detector.setup({})

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    payload = detector.run(frame)
    if payload.get("motion_detected"):
        print(f"Motion! Saved to {payload.get('saved_path')}")
```

---

## Screen Capture

Capture the desktop or a specific window region.

```python
from ai_vision_tool.capture import ScreenCapture

screen = ScreenCapture(
    region=None,           # None = full screen; or (x, y, w, h)
    fps=10,
)
screen.setup({})

frame = screen.capture()                    # single screenshot
for frame in screen.capture_stream(n=100): # 100 frames
    process(frame)
```

---

## Timelapse

```python
from ai_vision_tool.capture import TimeLapseCapture, TimeLapse

# Capture + assemble timelapse
capture = TimeLapseCapture(
    camera_index=0,
    interval_seconds=5,
    total_duration_seconds=3600,   # 1 hour
    output_dir="./timelapse",
)
capture.setup({})
capture.start()      # blocks for total_duration_seconds

# Assemble captured frames into video
assembler = TimeLapse(frames_dir="./timelapse", output_path="timelapse.mp4", fps=24)
assembler.assemble()
```

---

## Template Helpers

Quick functional helpers for common patterns:

```python
from ai_vision_tool.capture import image_template, video_capture_template, save_screenshot

# One-shot image capture
image_template(camera_index=0, output_path="photo.jpg")

# Interactive video capture with display
video_capture_template(camera_index=0, output_path="recording.mp4", show=True)

# Screenshot
save_screenshot(output_path="screen.png")
```
