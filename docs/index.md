# AI Vision Tool

**Composable, dependency-light computer-vision pipeline components** for image preprocessing, augmentation, detection, tracking, segmentation, enhancement, streaming, and dataset collection.

---

## Why AI Vision Tool?

| Feature | Detail |
|---------|--------|
| **Zero heavy deps by default** | Core installs with only `numpy`, `opencv-python`, and `pyyaml`. Heavy backends (ONNX, PyTorch, MediaPipe) are opt-in extras. |
| **Uniform component interface** | Every component accepts a raw NumPy array or a payload dict. Swap components without changing downstream code. |
| **Pipeline composition** | Chain preprocessing → augmentation → detection → export in one `AIVisionPipeline`. |
| **Device-aware** | Auto-selects CUDA → MPS → CPU. No manual device management. |
| **Production-ready extras** | RTSP streaming, Kafka/WebSocket sinks, S3/GCS sources, async and parallel pipelines. |

---

## Quick Example

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.preprocessing import Resize, CLAHE
from ai_vision_tool.augmentation import Flip, Noise
from ai_vision_tool.visualization import FrameAnnotator

pipeline = AIVisionPipeline([
    Resize(width=640, height=480),
    CLAHE(clip_limit=3.0),
    Flip(direction="horizontal"),
    Noise(intensity=0.02),
    FrameAnnotator(label="processed"),
])

import cv2
image = cv2.imread("photo.jpg")
result = pipeline.execute(image)
cv2.imwrite("output.jpg", result["frame"])
```

---

## Install

=== "Base (cv2 + numpy only)"

    ```bash
    pip install ai-vision-tool
    ```

=== "With detection (YOLO + MediaPipe)"

    ```bash
    pip install "ai-vision-tool[detection]"
    ```

=== "With ONNX runtime"

    ```bash
    pip install "ai-vision-tool[onnx]"
    ```

=== "Everything"

    ```bash
    pip install "ai-vision-tool[all]"
    ```

See [Installation](installation.md) for all optional extras.

---

## Module Overview

```
ai_vision_tool/
├── core/           ← AIVisionComponent base, data types, device, batch, memory
├── preprocessing/  ← geometry, intensity, quality checks
├── augmentation/   ← blur, brightness, noise, composite, weather, etc.
├── detection/      ← object, face, keypoint, text, anomaly detectors
├── tracking/       ← Kalman, ByteTracker, DeepSORT, ReID
├── segmentation/   ← semantic, instance, panoptic, SAM
├── enhancement/    ← denoising, super-resolution, colorization, deblurring
├── capture/        ← image/video/burst/timelapse/ROI/motion capture
├── io/             ← image/video readers-writers, camera, dataset exporters
├── pipelines/      ← AIVisionPipeline, async, parallel, prebuilt, serializer
├── streaming/      ← FrameStream, RTSP, buffered sliding-window
├── visualization/  ← viewer, annotator, bbox renderer, heatmap, dashboard
├── models/         ← registry, downloader, benchmark, ONNX/Torch/TFLite backends
├── config/         ← YAML/JSON config, component registry, env vars
├── integrations/   ← cloud (S3/GCS), labeling, streaming (Kafka/WebSocket)
└── utils/          ← color palette, metrics logger, frame sampler, image hash
```

---

## Navigation

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install the package and run your first pipeline in minutes.

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   **User Guide**

    ---

    In-depth how-to guides for every domain — preprocessing, augmentation, detection, and more.

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   **API Reference**

    ---

    Auto-generated reference for every class and function.

    [:octicons-arrow-right-24: API Reference](api/index.md)

-   **CLI**

    ---

    Run the webcam app and batch image processing from the terminal.

    [:octicons-arrow-right-24: CLI Reference](cli.md)

</div>
