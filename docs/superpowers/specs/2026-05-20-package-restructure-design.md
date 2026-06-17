# Package Restructure Design — ai-vision-tool v0.2.0

**Date:** 2026-05-20
**Status:** Approved
**Strategy:** Big Bang (single branch, one pass)
**Breaking change:** Yes — all `ai_vision_tool.components.*` import paths removed

---

## Problem

Current layout has two issues:

1. `components/` is a flat grab-bag holding preprocessing, augmentation, and loose capture/visualization wrappers with no clear boundary
2. Heavy deps (ONNX, PyTorch, MediaPipe, ultralytics, Kafka, boto3) are co-mingled with cv2-only code — `pip install ai-vision-tool` is heavier than it should be

---

## Target Directory Layout

```
ai_vision_tool/
├── core/                        # base class, data types, device, batch, scheduler, memory
│   ├── base.py                  # AIVisionComponent ← components/base.py
│   ├── data_types.py
│   ├── device.py
│   ├── batch_processor.py
│   ├── scheduler.py
│   └── memory_manager.py
│
├── io/                          # local media IO (cv2 + numpy only)
│   ├── image_io.py
│   ├── video_io.py
│   ├── camera_source.py
│   ├── dataset_exporter.py
│   ├── dataset_collector.py     # ← components/dataset_collector.py
│   └── image_exporter.py        # ← components/image_exporter.py
│
├── preprocessing/               # ← components/preprocessing/*
│   ├── geometry.py
│   ├── intensity.py
│   ├── quality.py
│   ├── classical_segmentation.py  # renamed from segmentation.py — avoids clash with DL segmentation/
│   ├── auto_orient.py
│   ├── auto_adjust_contrast.py
│   └── frame_resizer.py         # ← components/frame_resizer.py
│
├── augmentation/                # ← components/augmentations/*
│   ├── blur.py
│   ├── blur_artifact.py
│   ├── brightness.py
│   ├── camera_gain.py
│   ├── common.py                # parse_component_profile
│   ├── composite.py
│   ├── crop.py
│   ├── cutout.py
│   ├── exposure.py
│   ├── flip.py
│   ├── geometric_random.py
│   ├── grayscale.py             # renamed from greyscale.py
│   ├── hue.py
│   ├── mosaic.py
│   ├── motion_blur.py
│   ├── noise.py
│   ├── noise_dropout.py
│   ├── rotate90.py
│   ├── rotation.py
│   ├── saturation.py
│   ├── shear.py
│   └── weather_light.py
│
├── enhancement/                 # cv2-only algorithmic enhancement
│   ├── denoiser.py              # NLM, bilateral, gaussian ← enhancement/denoiser.py
│   ├── low_light.py             # CLAHE, gamma ← enhancement/low_light_enhancer.py
│   ├── sharpen.py
│   ├── contrast.py
│   ├── frame_enhancer.py        # ← components/frame_enhancer.py
│   └── models/                  # ONNX/DL-backed enhancement ([onnx] extra)
│       ├── super_resolution.py  # ← enhancement/super_resolution.py
│       ├── deblurring.py        # ← enhancement/deblurrer.py
│       ├── colorization.py      # ← enhancement/colorizer.py
│       └── dncnn.py
│
├── capture/                     # webcam, screen, frame grabbing, time-lapse
│   ├── image_template.py
│   ├── video_template.py
│   ├── screen_capture.py
│   ├── video_recorder.py
│   ├── frame_grabber.py         # ← components/frame_grabber.py
│   ├── motion_detector.py       # ← components/motion_detector.py (cv2 bg-subtraction)
│   ├── image_capture.py         # renamed from picture_taker.py
│   ├── burst_image_capture.py  # renamed from burst_picture_taker.py
│   ├── roi_capture.py
│   ├── video_capture.py         # renamed from video_taker.py
│   ├── time_lapse_capture.py
│   └── time_lapse.py
│
├── visualization/               # draw, render, display
│   ├── frame_viewer.py
│   ├── bbox_renderer.py
│   ├── heatmap_renderer.py
│   ├── dashboard_view.py        # renamed from dashboard_sink.py
│   ├── video_annotation_exporter.py
│   └── frame_annotator.py       # ← components/frame_annotator.py
│
├── pipelines/                   # merged pipeline/ (singular) + pipelines/
│   ├── vision_pipeline.py       # ← pipeline/vision_pipeline.py (AIVisionPipeline)
│   ├── prebuilt.py
│   ├── serializer.py
│   ├── async_pipeline.py
│   └── parallel_pipeline.py
│
├── models/                      # lightweight model metadata + lifecycle
│   ├── registry.py
│   ├── downloader.py
│   ├── benchmark.py
│   └── backends/                # heavy runtime adapters (guarded by extras)
│       ├── base.py
│       ├── onnx_model.py        # [onnx] extra
│       ├── torch_model.py       # [torch] extra
│       └── tflite_model.py      # [tflite] extra
│
├── streaming/                   # local frame streaming (cv2 + pure Python)
│   ├── frame_stream.py
│   ├── buffered_stream.py
│   └── rtsp_client.py
│
├── integrations/                # all external connectors (optional extras)
│   ├── streaming/
│   │   ├── websocket_sink.py    # [websocket] extra
│   │   └── kafka_io.py          # [kafka] extra
│   ├── cloud/
│   │   ├── s3_source.py         # [cloud] extra — ← cloud_source.py S3Source
│   │   └── gcs_source.py        # [cloud] extra — ← cloud_source.py GCSSource
│   └── labeling/
│       ├── auto_labeller.py     # ← components/auto_labeller.py
│       ├── darknet_auto_labeler.py
│       └── tensorflow_auto_labeler.py
│
├── config/                      # unchanged
├── utils/                       # generic helpers only
│   ├── color_palette.py
│   ├── metrics_logger.py
│   ├── frame_sampler.py
│   ├── image_hash.py
│   ├── draw_utils.py
│   └── image_utils.py           # ← components/_image_utils.py
│
├── cli/
│   └── main.py                  # ← cli.py
│
# Heavy optional extension modules (inside ai_vision_tool/, guarded by extras):
├── detection/                   # [detection] extra — unchanged
├── segmentation/                # [segmentation] extra — unchanged
└── tracking/                    # [tracking] extra — unchanged
```

**Deleted:** `components/` (entirely), `pipeline/` (singular)

---

## Base Install Boundary

`pip install ai-vision-tool` pulls only `numpy>=1.26`, `opencv-python>=4.8`, and `pyyaml>=6.0`.

| Module | Runtime deps |
|--------|-------------|
| `core/` | pure Python |
| `io/` | cv2 |
| `preprocessing/` | cv2, numpy |
| `augmentation/` | cv2, numpy |
| `enhancement/` (top-level only) | cv2 |
| `capture/` | cv2 |
| `visualization/` | cv2 |
| `pipelines/` | pure Python |
| `models/` (top-level only) | pure Python (urllib, hashlib) |
| `streaming/` | cv2 |
| `config/` | PyYAML |
| `utils/` | pure Python |
| `cli/` | pure Python |

`fastapi` and `uvicorn` removed from base dependencies (API moved to `backup/`).

---

## Optional Extras (pyproject.toml)

```toml
[project.optional-dependencies]
onnx         = ["onnxruntime>=1.18"]
torch        = ["torch>=2.3", "torchvision>=0.18"]
tflite       = ["tflite-runtime>=2.14"]
websocket    = ["websockets>=12.0"]
kafka        = ["confluent-kafka>=2.3.0"]
streaming    = ["websockets>=12.0", "confluent-kafka>=2.3.0"]
cloud        = ["boto3>=1.34", "google-cloud-storage>=2.16"]
detection    = ["ultralytics>=8.0", "mediapipe>=0.10"]
segmentation = ["ultralytics>=8.0", "segment-anything>=1.0", "torch>=2.3", "torchvision>=0.18"]
tracking     = ["onnxruntime>=1.18"]
api          = ["fastapi>=0.115", "uvicorn>=0.30"]
all          = [
    "onnxruntime>=1.18",
    "torch>=2.3", "torchvision>=0.18",
    "websockets>=12.0", "confluent-kafka>=2.3.0",
    "boto3>=1.34", "google-cloud-storage>=2.16",
    "ultralytics>=8.0", "mediapipe>=0.10",
    "segment-anything>=1.0",
    "fastapi>=0.115", "uvicorn>=0.30",
]
```

---

## Import Guard Pattern

Module-level guard in backend-specific modules only. `__init__.py` never imports optional modules.

```python
# models/backends/onnx_model.py
try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "ONNX support requires: pip install ai-vision-tool[onnx]"
    ) from exc
```

---

## `__init__.py` Strategy

Small `_EXPORTS` dict — only stable core public API. Cached on first access via `globals()`.

```python
import importlib

__version__ = "0.2.0"

_EXPORTS = {
    # core
    "AIVisionComponent":  ("ai_vision_tool.core.base",                 "AIVisionComponent"),
    "AIVisionPipeline":   ("ai_vision_tool.pipelines.vision_pipeline",  "AIVisionPipeline"),
    # preprocessing
    "Resize":             ("ai_vision_tool.preprocessing.geometry",     "Resize"),
    "CenterCrop":         ("ai_vision_tool.preprocessing.geometry",     "CenterCrop"),
    "CLAHE":              ("ai_vision_tool.preprocessing.intensity",    "CLAHE"),
    # augmentation
    "Blur":               ("ai_vision_tool.augmentation.blur",          "Blur"),
    "Flip":               ("ai_vision_tool.augmentation.flip",          "Flip"),
    # enhancement (cv2 only)
    "Denoiser":           ("ai_vision_tool.enhancement.denoiser",       "Denoiser"),
    "LowLightEnhancer":   ("ai_vision_tool.enhancement.low_light",      "LowLightEnhancer"),
    # ... remaining stable core classes
}

__all__ = sorted(_EXPORTS.keys()) + ["__version__"]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ai_vision_tool' has no attribute {name!r}")
    module_path, attr = _EXPORTS[name]
    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value  # cache — import once, reuse for process lifetime
    return value
```

**Rule:** No optional module paths in `_EXPORTS`. Users import those directly:

```python
from ai_vision_tool.detection.object_detector import ObjectDetector
from ai_vision_tool.enhancement.models.super_resolution import SuperResolution
```

---

## Testing Strategy

All 8 existing test files import from `ai_vision_tool.components.*` — every import path must be updated to new paths. Test logic is unchanged.

Add `tests/test_imports.py`:

```python
def test_import_ai_vision_tool_base_is_lightweight():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-I", "-c", "import ai_vision_tool; print('ok')"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert "onnxruntime" not in result.stderr.lower()
    assert "torch" not in result.stderr.lower()
```

---

## Migration Execution Order (Big Bang)

1. Create new dirs: `augmentation/`, `integrations/`, `models/backends/`, `enhancement/models/`, `cli/`
2. Move files with `git mv` — preserves history
3. Update internal imports in all moved files
4. Update `__init__.py` — new `_EXPORTS`, remove all `components.*` paths
5. Update `pyproject.toml` — remove `fastapi`/`uvicorn` from base, add all extras, update scripts entrypoint to `ai_vision_tool.cli.main:main`
6. Update all 8 test files to new import paths
7. Add `tests/test_imports.py`
8. Delete `components/`, `pipeline/` (singular)
9. Run `pytest` — fix failures
10. Commit: `feat!: restructure package layout — breaking change v0.2.0`

---

## Breaking Change Notice

```
Breaking change in v0.2.0-alpha:
ai_vision_tool.components.* has been removed.

Migrate to:
  ai_vision_tool.preprocessing.*
  ai_vision_tool.augmentation.*
  ai_vision_tool.enhancement.*
  ai_vision_tool.io.*
  ai_vision_tool.streaming.*
  ai_vision_tool.capture.*
  ai_vision_tool.visualization.*
  ai_vision_tool.pipelines.*
  ai_vision_tool.core.*
```
