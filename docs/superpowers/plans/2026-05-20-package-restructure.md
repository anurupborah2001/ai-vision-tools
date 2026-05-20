# Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `ai_vision_tool` from a flat `components/` grab-bag into a clean module hierarchy where `pip install ai-vision-tool` pulls only `numpy + opencv-python + pyyaml`.

**Architecture:** Move `components/preprocessing/*` → `preprocessing/`, `components/augmentations/*` → `augmentation/`, split `enhancement/` into cv2-only vs `enhancement/models/` (ONNX-backed), relocate external connectors to `integrations/`, and move `models/onnx_model`, `torch_model`, `tflite_model` to `models/backends/`. All 30+ files that imported `from ai_vision_tool.components.base` are updated in one sed pass before any other moves.

**Tech Stack:** Python 3.10+, git mv (history-preserving moves), pytest, ruff, BSD sed (macOS `-i ''`)

**Spec:** `docs/superpowers/specs/2026-05-20-package-restructure-design.md`

---

## File Map

### New packages (empty `__init__.py` needed)
| New path | Purpose |
|----------|---------|
| `ai_vision_tool/augmentation/` | Top-level augmentation package (was `components/augmentations/`) |
| `ai_vision_tool/integrations/` | External connectors root |
| `ai_vision_tool/integrations/streaming/` | Kafka + WebSocket sinks |
| `ai_vision_tool/integrations/cloud/` | S3 + GCS sources |
| `ai_vision_tool/integrations/labeling/` | Auto-labelers |
| `ai_vision_tool/models/backends/` | ONNX / Torch / TFLite adapters |
| `ai_vision_tool/enhancement/models/` | DL-backed enhancement |
| `ai_vision_tool/cli/` | CLI entrypoint |

### Key file moves
| From | To | Notes |
|------|----|-------|
| `components/base.py` | `core/base.py` | AIVisionComponent base class |
| `components/_image_utils.py` | `utils/image_utils.py` | Shared helpers |
| `components/preprocessing/*.py` | `preprocessing/*.py` | 6 files |
| `components/preprocessing/segmentation.py` | `preprocessing/classical_segmentation.py` | Renamed |
| `components/augmentations/*.py` | `augmentation/*.py` | 21 files |
| `components/augmentations/greyscale.py` | `augmentation/grayscale.py` | Renamed |
| `components/frame_enhancer.py` | `enhancement/frame_enhancer.py` | |
| `components/frame_resizer.py` | `preprocessing/frame_resizer.py` | |
| `components/frame_annotator.py` | `visualization/frame_annotator.py` | |
| `components/frame_grabber.py` | `capture/frame_grabber.py` | |
| `components/motion_detector.py` | `capture/motion_detector.py` | |
| `components/picture_taker.py` | `capture/image_capture.py` | Renamed |
| `components/burst_picture_taker.py` | `capture/burst_image_capture.py` | Renamed |
| `components/roi_capture.py` | `capture/roi_capture.py` | |
| `components/video_taker.py` | `capture/video_capture.py` | Renamed |
| `components/time_lapse_capture.py` | `capture/time_lapse_capture.py` | |
| `components/time_lapse.py` | `capture/time_lapse.py` | |
| `components/dataset_collector.py` | `io/dataset_collector.py` | |
| `components/image_exporter.py` | `io/image_exporter.py` | |
| `components/auto_labeller.py` | `integrations/labeling/auto_labeller.py` | |
| `components/darknet_auto_labeler.py` | `integrations/labeling/darknet_auto_labeler.py` | |
| `components/tensorflow_auto_labeler.py` | `integrations/labeling/tensorflow_auto_labeler.py` | |
| `enhancement/super_resolution.py` | `enhancement/models/super_resolution.py` | |
| `enhancement/deblurrer.py` | `enhancement/models/deblurring.py` | Renamed |
| `enhancement/colorizer.py` | `enhancement/models/colorization.py` | Renamed |
| `enhancement/low_light_enhancer.py` | `enhancement/low_light.py` | Renamed |
| `models/onnx_model.py` | `models/backends/onnx_model.py` | |
| `models/torch_model.py` | `models/backends/torch_model.py` | |
| `models/tflite_model.py` | `models/backends/tflite_model.py` | |
| `streaming/websocket_sink.py` | `integrations/streaming/websocket_sink.py` | |
| `streaming/kafka_io.py` | `integrations/streaming/kafka_io.py` | |
| `io/cloud_source.py` | `integrations/cloud/s3_source.py` + `integrations/cloud/gcs_source.py` | Split |
| `pipeline/vision_pipeline.py` | `pipelines/vision_pipeline.py` | Merges `pipeline/` into `pipelines/` |
| `cli.py` | `cli/main.py` | |
| `visualization/dashboard_sink.py` | `visualization/dashboard_view.py` | Renamed |

### Deleted after all moves
- `ai_vision_tool/components/` (entire directory)
- `ai_vision_tool/pipeline/` (singular — merged into `pipelines/`)

---

## Task 1: Create new package scaffolding

**Files:**
- Create: `ai_vision_tool/augmentation/__init__.py`
- Create: `ai_vision_tool/integrations/__init__.py`
- Create: `ai_vision_tool/integrations/streaming/__init__.py`
- Create: `ai_vision_tool/integrations/cloud/__init__.py`
- Create: `ai_vision_tool/integrations/labeling/__init__.py`
- Create: `ai_vision_tool/models/backends/__init__.py`
- Create: `ai_vision_tool/enhancement/models/__init__.py`
- Create: `ai_vision_tool/cli/__init__.py`

- [ ] **Step 1: Create empty `__init__.py` files for all new packages**

```bash
touch ai_vision_tool/augmentation/__init__.py \
      ai_vision_tool/integrations/__init__.py \
      ai_vision_tool/integrations/streaming/__init__.py \
      ai_vision_tool/integrations/cloud/__init__.py \
      ai_vision_tool/integrations/labeling/__init__.py \
      ai_vision_tool/models/backends/__init__.py \
      ai_vision_tool/enhancement/models/__init__.py \
      ai_vision_tool/cli/__init__.py
```

- [ ] **Step 2: Verify directories created**

```bash
find ai_vision_tool/augmentation ai_vision_tool/integrations \
     ai_vision_tool/models/backends ai_vision_tool/enhancement/models \
     ai_vision_tool/cli -name "__init__.py"
```
Expected: 8 lines, one per new package.

- [ ] **Step 3: Commit**

```bash
git add ai_vision_tool/augmentation/__init__.py \
        ai_vision_tool/integrations/__init__.py \
        ai_vision_tool/integrations/streaming/__init__.py \
        ai_vision_tool/integrations/cloud/__init__.py \
        ai_vision_tool/integrations/labeling/__init__.py \
        ai_vision_tool/models/backends/__init__.py \
        ai_vision_tool/enhancement/models/__init__.py \
        ai_vision_tool/cli/__init__.py
git commit -m "chore: scaffold new package directories for restructure"
```

---

## Task 2: Move core base class and image utils

**Files:**
- Move: `ai_vision_tool/components/base.py` → `ai_vision_tool/core/base.py`
- Move: `ai_vision_tool/components/_image_utils.py` → `ai_vision_tool/utils/image_utils.py`

These two files are imported by every other module. Move them first; update all callers in Task 3.

- [ ] **Step 1: Move base class**

```bash
git mv ai_vision_tool/components/base.py ai_vision_tool/core/base.py
```

- [ ] **Step 2: Move image utils**

```bash
git mv ai_vision_tool/components/_image_utils.py ai_vision_tool/utils/image_utils.py
```

- [ ] **Step 3: Verify files moved**

```bash
ls ai_vision_tool/core/base.py ai_vision_tool/utils/image_utils.py
```
Expected: both paths print without error.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: move AIVisionComponent base and image_utils to core/ and utils/"
```

---

## Task 3: Global import sweep — update all callers of base and image_utils

Every file in the codebase that imports from `components.base` or `components._image_utils` must be updated before moving any other files. There are ~35 such files.

**Files:** Every `.py` file under `ai_vision_tool/` and `tests/`

- [ ] **Step 1: Replace all absolute imports of `components.base`**

```bash
find ai_vision_tool tests -name "*.py" | xargs sed -i '' \
  's|from ai_vision_tool\.components\.base import|from ai_vision_tool.core.base import|g'
```

- [ ] **Step 2: Replace all absolute imports of `components._image_utils`**

```bash
find ai_vision_tool tests -name "*.py" | xargs sed -i '' \
  's|from ai_vision_tool\.components\._image_utils import|from ai_vision_tool.utils.image_utils import|g'
```

- [ ] **Step 3: Fix relative imports inside `components/preprocessing/` (still in old location)**

Files in `components/preprocessing/` use `from ..base import` and `from .._image_utils import`. They will be moved in Task 4, but fix relative paths now so they work after the move to `preprocessing/` (one level up from `components/`):

```bash
find ai_vision_tool/components/preprocessing -name "*.py" | xargs sed -i '' \
  's|from \.\.base import|from ..core.base import|g'

find ai_vision_tool/components/preprocessing -name "*.py" | xargs sed -i '' \
  's|from \.\._image_utils import|from ..utils.image_utils import|g'
```

- [ ] **Step 4: Fix relative imports inside `components/augmentations/` (still in old location)**

```bash
find ai_vision_tool/components/augmentations -name "*.py" | xargs sed -i '' \
  's|from \.\.base import|from ..core.base import|g'

find ai_vision_tool/components/augmentations -name "*.py" | xargs sed -i '' \
  's|from \.\._image_utils import|from ..utils.image_utils import|g'
```

- [ ] **Step 5: Fix relative import in `pipeline/vision_pipeline.py`**

```bash
sed -i '' 's|from \.\.components\.base import|from ..core.base import|g' \
  ai_vision_tool/pipeline/vision_pipeline.py
```

- [ ] **Step 6: Verify no `components.base` or `components._image_utils` references remain**

```bash
grep -rn "components\.base\|components\._image_utils" ai_vision_tool tests \
  --include="*.py" | grep -v __pycache__
```
Expected: no output (zero matches).

- [ ] **Step 7: Quick smoke test — import core base class**

```bash
python -c "from ai_vision_tool.core.base import AIVisionComponent; print('ok')"
```
Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: update all imports from components.base and components._image_utils to new paths"
```

---

## Task 4: Move preprocessing module

**Files:**
- Move: `ai_vision_tool/components/preprocessing/*.py` → `ai_vision_tool/preprocessing/`
- Rename: `segmentation.py` → `classical_segmentation.py`
- Move: `ai_vision_tool/components/frame_resizer.py` → `ai_vision_tool/preprocessing/frame_resizer.py`
- Modify: `ai_vision_tool/preprocessing/__init__.py`

- [ ] **Step 1: Move all preprocessing source files**

```bash
git mv ai_vision_tool/components/preprocessing/geometry.py          ai_vision_tool/preprocessing/geometry.py
git mv ai_vision_tool/components/preprocessing/intensity.py         ai_vision_tool/preprocessing/intensity.py
git mv ai_vision_tool/components/preprocessing/quality.py           ai_vision_tool/preprocessing/quality.py
git mv ai_vision_tool/components/preprocessing/segmentation.py      ai_vision_tool/preprocessing/classical_segmentation.py
git mv ai_vision_tool/components/preprocessing/auto_orient.py       ai_vision_tool/preprocessing/auto_orient.py
git mv ai_vision_tool/components/preprocessing/auto_adjust_contrast.py ai_vision_tool/preprocessing/auto_adjust_contrast.py
git mv ai_vision_tool/components/frame_resizer.py                   ai_vision_tool/preprocessing/frame_resizer.py
```

- [ ] **Step 2: Update `preprocessing/__init__.py` to import from new paths**

The existing `preprocessing/__init__.py` already exists; replace its content:

```python
"""Preprocessing component exports."""

from .auto_adjust_contrast import AutoAdjustContrast
from .auto_orient import AutoOrient
from .classical_segmentation import RemoveBackground
from .frame_resizer import FrameResizer
from .geometry import (
    AutoCrop,
    BoundingBoxClamp,
    BoundingBoxNormalize,
    CenterCrop,
    Deskew,
    FaceAlign,
    LetterboxResize,
    MaskResize,
    ObjectCrop,
    PadToSquare,
    PerspectiveCorrection,
    Resize,
)
from .intensity import (
    AdaptiveThreshold,
    BGRToRGB,
    CLAHE,
    ConvertColorSpace,
    ContourExtraction,
    Deblur,
    Denoise,
    EdgeDetection,
    GammaCorrection,
    HistogramEqualization,
    Normalize,
    RescalePixels,
    RGBToBGR,
    Sharpen,
    Standardize,
    Threshold,
    WhiteBalance,
)
from .quality import (
    AspectRatioFilter,
    BlurDetection,
    BrightnessCheck,
    CorruptImageCheck,
    DuplicateImageCheck,
    ImageQualityCheck,
    MaxSizeFilter,
    MinSizeFilter,
)

__all__ = [
    "AdaptiveThreshold",
    "AspectRatioFilter",
    "AutoAdjustContrast",
    "AutoCrop",
    "AutoOrient",
    "BGRToRGB",
    "BlurDetection",
    "BoundingBoxClamp",
    "BoundingBoxNormalize",
    "BrightnessCheck",
    "CenterCrop",
    "CLAHE",
    "ClassicalSegmentation",
    "ContourExtraction",
    "ConvertColorSpace",
    "CorruptImageCheck",
    "Deblur",
    "Denoise",
    "Deskew",
    "DuplicateImageCheck",
    "EdgeDetection",
    "FaceAlign",
    "FrameResizer",
    "GammaCorrection",
    "HistogramEqualization",
    "ImageQualityCheck",
    "LetterboxResize",
    "MaskResize",
    "MaxSizeFilter",
    "MinSizeFilter",
    "Normalize",
    "ObjectCrop",
    "PadToSquare",
    "PerspectiveCorrection",
    "RemoveBackground",
    "RescalePixels",
    "Resize",
    "RGBToBGR",
    "Sharpen",
    "Standardize",
    "Threshold",
    "WhiteBalance",
]
```

- [ ] **Step 3: Verify preprocessing imports work**

```bash
python -c "
from ai_vision_tool.preprocessing import Resize, CLAHE, RemoveBackground, FrameResizer
print('preprocessing ok')
"
```
Expected: `preprocessing ok`

- [ ] **Step 4: Run preprocessing tests to confirm no regression**

```bash
pytest tests/test_preprocessing_components.py -v 2>&1 | tail -20
```
Expected: tests that previously passed still pass (import errors are expected — they'll be fixed in Task 14).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: move preprocessing module from components/ to top-level preprocessing/"
```

---

## Task 5: Move augmentation module

**Files:**
- Move: `ai_vision_tool/components/augmentations/*.py` → `ai_vision_tool/augmentation/`
- Rename: `greyscale.py` → `grayscale.py`
- Modify: `ai_vision_tool/augmentation/__init__.py`

- [ ] **Step 1: Move all augmentation source files**

```bash
git mv ai_vision_tool/components/augmentations/blur.py              ai_vision_tool/augmentation/blur.py
git mv ai_vision_tool/components/augmentations/blur_artifact.py     ai_vision_tool/augmentation/blur_artifact.py
git mv ai_vision_tool/components/augmentations/brightness.py        ai_vision_tool/augmentation/brightness.py
git mv ai_vision_tool/components/augmentations/camera_gain.py       ai_vision_tool/augmentation/camera_gain.py
git mv ai_vision_tool/components/augmentations/common.py            ai_vision_tool/augmentation/common.py
git mv ai_vision_tool/components/augmentations/composite.py         ai_vision_tool/augmentation/composite.py
git mv ai_vision_tool/components/augmentations/crop.py              ai_vision_tool/augmentation/crop.py
git mv ai_vision_tool/components/augmentations/cutout.py            ai_vision_tool/augmentation/cutout.py
git mv ai_vision_tool/components/augmentations/exposure.py          ai_vision_tool/augmentation/exposure.py
git mv ai_vision_tool/components/augmentations/flip.py              ai_vision_tool/augmentation/flip.py
git mv ai_vision_tool/components/augmentations/geometric_random.py  ai_vision_tool/augmentation/geometric_random.py
git mv ai_vision_tool/components/augmentations/greyscale.py         ai_vision_tool/augmentation/grayscale.py
git mv ai_vision_tool/components/augmentations/hue.py               ai_vision_tool/augmentation/hue.py
git mv ai_vision_tool/components/augmentations/mosaic.py            ai_vision_tool/augmentation/mosaic.py
git mv ai_vision_tool/components/augmentations/motion_blur.py       ai_vision_tool/augmentation/motion_blur.py
git mv ai_vision_tool/components/augmentations/noise.py             ai_vision_tool/augmentation/noise.py
git mv ai_vision_tool/components/augmentations/noise_dropout.py     ai_vision_tool/augmentation/noise_dropout.py
git mv ai_vision_tool/components/augmentations/rotate90.py          ai_vision_tool/augmentation/rotate90.py
git mv ai_vision_tool/components/augmentations/rotation.py          ai_vision_tool/augmentation/rotation.py
git mv ai_vision_tool/components/augmentations/saturation.py        ai_vision_tool/augmentation/saturation.py
git mv ai_vision_tool/components/augmentations/shear.py             ai_vision_tool/augmentation/shear.py
git mv ai_vision_tool/components/augmentations/weather_light.py     ai_vision_tool/augmentation/weather_light.py
```

- [ ] **Step 2: Update `augmentation/__init__.py`**

```python
"""Augmentation component exports."""

from .blur import Blur
from .blur_artifact import (
    CompressionArtifacts,
    DefocusBlur,
    Downscale,
    Emboss,
    Equalize,
    GaussianBlur,
    GlassBlur,
    JPEGCompression,
    MedianBlur,
    Posterize,
    Sharpen,
    Solarize,
    Superpixel,
    ZoomBlur,
)
from .brightness import Brightness
from .camera_gain import CameraGain
from .composite import (
    BoundingBoxJitter,
    CopyPaste,
    CutMix,
    MixUp,
    Mosaic9,
    ObjectPaste,
    RandomOcclusion,
)
from .crop import Crop
from .cutout import Cutout
from .exposure import Exposure
from .flip import Flip
from .geometric_random import (
    AffineTransform,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    PerspectiveTransform,
    RandomCrop,
    RandomPadding,
    RandomResize,
    RandomResizedCrop,
    RandomScale,
    Translate,
)
from .grayscale import Greyscale
from .hue import Hue
from .mosaic import Mosaic
from .motion_blur import MotionBlur
from .noise import Noise
from .noise_dropout import (
    CoarseDropout,
    GridDropout,
    ISONoise,
    MaskDropout,
    MultiplicativeNoise,
    PixelDropout,
    RandomErasing,
    SaltPepperNoise,
)
from .rotate90 import Rotate90
from .rotation import Rotation
from .saturation import Saturation
from .shear import Shear
from .weather_light import (
    ChannelShuffle,
    ColorJitter,
    HSVShift,
    InvertImage,
    RandomBrightnessContrast,
    RandomFog,
    RandomGamma,
    RandomRain,
    RandomShadow,
    RandomSnow,
    RandomSunFlare,
    RGBShift,
    ToSepia,
)

__all__ = [
    "AffineTransform", "Blur", "BoundingBoxJitter", "Brightness",
    "CameraGain", "ChannelShuffle", "CoarseDropout", "ColorJitter",
    "CompressionArtifacts", "CopyPaste", "Crop", "CutMix", "Cutout",
    "DefocusBlur", "Downscale", "ElasticTransform", "Emboss", "Equalize",
    "Exposure", "Flip", "GaussianBlur", "GlassBlur", "Greyscale",
    "GridDistortion", "GridDropout", "HSVShift", "Hue", "ISONoise",
    "InvertImage", "JPEGCompression", "MaskDropout", "MedianBlur",
    "MixUp", "Mosaic", "Mosaic9", "MotionBlur", "MultiplicativeNoise",
    "Noise", "ObjectPaste", "OpticalDistortion", "PerspectiveTransform",
    "PixelDropout", "Posterize", "RandomBrightnessContrast", "RandomCrop",
    "RandomErasing", "RandomFog", "RandomGamma", "RandomOcclusion",
    "RandomPadding", "RandomRain", "RandomResize", "RandomResizedCrop",
    "RandomScale", "RandomShadow", "RandomSnow", "RandomSunFlare",
    "RGBShift", "Rotate90", "Rotation", "Saturation", "SaltPepperNoise",
    "Sharpen", "Shear", "Solarize", "Superpixel", "ToSepia",
    "Translate", "ZoomBlur",
]
```

- [ ] **Step 3: Verify augmentation imports work**

```bash
python -c "
from ai_vision_tool.augmentation import Blur, Flip, Rotation, Greyscale
print('augmentation ok')
"
```
Expected: `augmentation ok`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: move augmentation module from components/augmentations/ to top-level augmentation/"
```

---

## Task 6: Move loose capture and visualization files from components

**Files:**
- Move 8 capture-related files from `components/` → `capture/`
- Move 1 visualization file from `components/` → `visualization/`
- Move 2 io files from `components/` → `io/`
- Rename `visualization/dashboard_sink.py` → `visualization/dashboard_view.py`

- [ ] **Step 1: Move capture files**

```bash
git mv ai_vision_tool/components/frame_grabber.py       ai_vision_tool/capture/frame_grabber.py
git mv ai_vision_tool/components/motion_detector.py     ai_vision_tool/capture/motion_detector.py
git mv ai_vision_tool/components/picture_taker.py       ai_vision_tool/capture/image_capture.py
git mv ai_vision_tool/components/burst_picture_taker.py ai_vision_tool/capture/burst_image_capture.py
git mv ai_vision_tool/components/roi_capture.py         ai_vision_tool/capture/roi_capture.py
git mv ai_vision_tool/components/video_taker.py         ai_vision_tool/capture/video_capture.py
git mv ai_vision_tool/components/time_lapse_capture.py  ai_vision_tool/capture/time_lapse_capture.py
git mv ai_vision_tool/components/time_lapse.py          ai_vision_tool/capture/time_lapse.py
```

- [ ] **Step 2: Fix class names in renamed files — `image_capture.py`**

`picture_taker.py` exported `PictureTaker`. The file is renamed but the class stays the same; just check it has no broken internal import:

```bash
grep "^from\|^import" ai_vision_tool/capture/image_capture.py
```
If it imports from `ai_vision_tool.components.*` fix with:
```bash
sed -i '' 's|from ai_vision_tool\.components\.|from ai_vision_tool.|g' \
  ai_vision_tool/capture/image_capture.py \
  ai_vision_tool/capture/burst_image_capture.py \
  ai_vision_tool/capture/video_capture.py
```

- [ ] **Step 3: Move frame_annotator to visualization**

```bash
git mv ai_vision_tool/components/frame_annotator.py ai_vision_tool/visualization/frame_annotator.py
```

- [ ] **Step 4: Rename dashboard_sink to dashboard_view**

```bash
git mv ai_vision_tool/visualization/dashboard_sink.py ai_vision_tool/visualization/dashboard_view.py
```

- [ ] **Step 5: Update `dashboard_view.py` class name reference in any imports**

```bash
grep -rn "dashboard_sink\|DashboardSink" ai_vision_tool --include="*.py" | grep -v __pycache__
```
Update any hits to `dashboard_view` / `DashboardView` (or keep class name `DashboardSink` — rename only the file, not the class, to minimize churn).

- [ ] **Step 6: Move io files**

```bash
git mv ai_vision_tool/components/dataset_collector.py ai_vision_tool/io/dataset_collector.py
git mv ai_vision_tool/components/image_exporter.py    ai_vision_tool/io/image_exporter.py
```

- [ ] **Step 7: Verify moved files have no broken imports**

```bash
python -c "
from ai_vision_tool.capture.frame_grabber import FrameGrabber
from ai_vision_tool.capture.motion_detector import MotionDetector
from ai_vision_tool.visualization.frame_annotator import FrameAnnotator
from ai_vision_tool.io.dataset_collector import DatasetCollector
print('capture/viz/io ok')
"
```
Expected: `capture/viz/io ok`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: move loose capture, visualization, and io files out of components/"
```

---

## Task 7: Split enhancement module (cv2-only vs model-backed)

**Files:**
- Move: `enhancement/super_resolution.py` → `enhancement/models/super_resolution.py`
- Move: `enhancement/deblurrer.py` → `enhancement/models/deblurring.py`
- Move: `enhancement/colorizer.py` → `enhancement/models/colorization.py`
- Rename: `enhancement/low_light_enhancer.py` → `enhancement/low_light.py`
- Move: `components/frame_enhancer.py` → `enhancement/frame_enhancer.py`

- [ ] **Step 1: Move model-backed enhancers to `enhancement/models/`**

```bash
git mv ai_vision_tool/enhancement/super_resolution.py ai_vision_tool/enhancement/models/super_resolution.py
git mv ai_vision_tool/enhancement/deblurrer.py        ai_vision_tool/enhancement/models/deblurring.py
git mv ai_vision_tool/enhancement/colorizer.py        ai_vision_tool/enhancement/models/colorization.py
```

- [ ] **Step 2: Rename `low_light_enhancer.py` → `low_light.py`**

```bash
git mv ai_vision_tool/enhancement/low_light_enhancer.py ai_vision_tool/enhancement/low_light.py
```

- [ ] **Step 3: Move `frame_enhancer.py` from components**

```bash
git mv ai_vision_tool/components/frame_enhancer.py ai_vision_tool/enhancement/frame_enhancer.py
```

- [ ] **Step 4: Update any references to old enhancement paths**

```bash
grep -rn "enhancement\.low_light_enhancer\|enhancement\.deblurrer\|enhancement\.colorizer\|enhancement\.super_resolution" \
  ai_vision_tool --include="*.py" | grep -v __pycache__
```

Update each hit:
- `enhancement.low_light_enhancer` → `enhancement.low_light`
- `enhancement.deblurrer` → `enhancement.models.deblurring`
- `enhancement.colorizer` → `enhancement.models.colorization`
- `enhancement.super_resolution` → `enhancement.models.super_resolution`

The key file to update is `pipelines/prebuilt.py` lines 88-89:
```python
from ai_vision_tool.enhancement.low_light import LowLightEnhancer
from ai_vision_tool.enhancement.denoiser import Denoiser
```

- [ ] **Step 5: Verify enhancement imports**

```bash
python -c "
from ai_vision_tool.enhancement.denoiser import Denoiser
from ai_vision_tool.enhancement.low_light import LowLightEnhancer
from ai_vision_tool.enhancement.models.super_resolution import SuperResolution
print('enhancement ok')
"
```
Expected: `enhancement ok`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: split enhancement/ into cv2-only and enhancement/models/ (DL-backed)"
```

---

## Task 8: Merge `pipeline/` (singular) into `pipelines/`

**Files:**
- Move: `ai_vision_tool/pipeline/vision_pipeline.py` → `ai_vision_tool/pipelines/vision_pipeline.py`
- Delete: `ai_vision_tool/pipeline/` directory
- Modify: `ai_vision_tool/pipelines/prebuilt.py`, `pipelines/serializer.py`

- [ ] **Step 1: Move vision_pipeline.py**

```bash
git mv ai_vision_tool/pipeline/vision_pipeline.py ai_vision_tool/pipelines/vision_pipeline.py
```

- [ ] **Step 2: Update `pipelines/prebuilt.py` — it imports from old path**

Find line: `from ai_vision_tool.pipeline.vision_pipeline import AIVisionPipeline`
Replace with: `from ai_vision_tool.pipelines.vision_pipeline import AIVisionPipeline`

Also the lambda inside `_make_pipeline`:
```bash
sed -i '' 's|from ai_vision_tool\.pipeline\.vision_pipeline import|from ai_vision_tool.pipelines.vision_pipeline import|g' \
  ai_vision_tool/pipelines/prebuilt.py \
  ai_vision_tool/pipelines/serializer.py
```

- [ ] **Step 3: Update `pipelines/__init__.py` to export `AIVisionPipeline`**

```python
"""Pipeline exports."""

from .vision_pipeline import AIVisionPipeline

__all__ = ["AIVisionPipeline"]
```

- [ ] **Step 4: Sweep for any remaining `ai_vision_tool.pipeline` references**

```bash
grep -rn "ai_vision_tool\.pipeline[^s]" ai_vision_tool tests --include="*.py" | grep -v __pycache__
```
Update any hits to `ai_vision_tool.pipelines`.

- [ ] **Step 5: Also handle the `cli.py` import `from ai_vision_tool.pipeline import AIVisionPipeline`**

```bash
grep -n "from ai_vision_tool.pipeline" ai_vision_tool/cli.py
```
Update to: `from ai_vision_tool.pipelines import AIVisionPipeline`

- [ ] **Step 6: Verify pipeline works**

```bash
python -c "
from ai_vision_tool.pipelines.vision_pipeline import AIVisionPipeline
from ai_vision_tool.pipelines import AIVisionPipeline
print('pipelines ok')
"
```
Expected: `pipelines ok`

- [ ] **Step 7: Remove now-empty `pipeline/` (singular) directory**

```bash
git rm -r ai_vision_tool/pipeline/
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: merge pipeline/ (singular) into pipelines/"
```

---

## Task 9: Move model backends

**Files:**
- Move: `models/onnx_model.py` → `models/backends/onnx_model.py`
- Move: `models/torch_model.py` → `models/backends/torch_model.py`
- Move: `models/tflite_model.py` → `models/backends/tflite_model.py`
- Modify: each backend file to add module-level import guard

- [ ] **Step 1: Move backend model files**

```bash
git mv ai_vision_tool/models/onnx_model.py   ai_vision_tool/models/backends/onnx_model.py
git mv ai_vision_tool/models/torch_model.py  ai_vision_tool/models/backends/torch_model.py
git mv ai_vision_tool/models/tflite_model.py ai_vision_tool/models/backends/tflite_model.py
```

- [ ] **Step 2: Add module-level guard to `models/backends/onnx_model.py`**

Open `ai_vision_tool/models/backends/onnx_model.py` and ensure its first import block looks like:

```python
from __future__ import annotations

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError(
        "ONNX backend requires: pip install ai-vision-tool[onnx]"
    ) from exc
```

- [ ] **Step 3: Add module-level guard to `models/backends/torch_model.py`**

```python
from __future__ import annotations

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch backend requires: pip install ai-vision-tool[torch]"
    ) from exc
```

- [ ] **Step 4: Add module-level guard to `models/backends/tflite_model.py`**

```python
from __future__ import annotations

try:
    import tflite_runtime.interpreter as tflite  # noqa: F401
except ImportError:
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TFLite backend requires: pip install ai-vision-tool[tflite]"
        ) from exc
```

- [ ] **Step 5: Update `models/__init__.py` — remove backend re-exports**

```python
"""Model registry and lifecycle utilities (lightweight core)."""

from .benchmark import ModelBenchmark
from .downloader import ModelDownloader
from .registry import ModelRegistry

__all__ = ["ModelBenchmark", "ModelDownloader", "ModelRegistry"]
```

- [ ] **Step 6: Sweep for any old `models.onnx_model` / `models.torch_model` / `models.tflite_model` references**

```bash
grep -rn "from ai_vision_tool\.models\.onnx_model\|from ai_vision_tool\.models\.torch_model\|from ai_vision_tool\.models\.tflite_model" \
  ai_vision_tool tests --include="*.py" | grep -v __pycache__
```
Update any hits to `ai_vision_tool.models.backends.*`.

- [ ] **Step 7: Verify lightweight models import with no heavy deps**

```bash
python -c "
from ai_vision_tool.models.registry import ModelRegistry
from ai_vision_tool.models.downloader import ModelDownloader
print('models core ok')
"
```
Expected: `models core ok` with no onnxruntime/torch import errors.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: move model runtime adapters to models/backends/ with import guards"
```

---

## Task 10: Move streaming integrations

**Files:**
- Move: `streaming/websocket_sink.py` → `integrations/streaming/websocket_sink.py`
- Move: `streaming/kafka_io.py` → `integrations/streaming/kafka_io.py`
- Leave in `streaming/`: `frame_stream.py`, `buffered_stream.py`, `rtsp_client.py`

- [ ] **Step 1: Move heavy streaming files to integrations**

```bash
git mv ai_vision_tool/streaming/websocket_sink.py ai_vision_tool/integrations/streaming/websocket_sink.py
git mv ai_vision_tool/streaming/kafka_io.py       ai_vision_tool/integrations/streaming/kafka_io.py
```

- [ ] **Step 2: Update `integrations/streaming/__init__.py`**

```python
"""External streaming transport integrations (websocket, kafka)."""
```

- [ ] **Step 3: Sweep for old import references**

```bash
grep -rn "from ai_vision_tool\.streaming\.websocket_sink\|from ai_vision_tool\.streaming\.kafka_io" \
  ai_vision_tool tests --include="*.py" | grep -v __pycache__
```
Update any hits to `ai_vision_tool.integrations.streaming.*`.

- [ ] **Step 4: Verify core streaming still works**

```bash
python -c "
from ai_vision_tool.streaming.frame_stream import FrameStream
from ai_vision_tool.streaming.buffered_stream import BufferedStream
print('streaming core ok')
"
```
Expected: `streaming core ok`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: move websocket and kafka sinks to integrations/streaming/"
```

---

## Task 11: Split cloud_source into integrations/cloud

**Files:**
- Split: `io/cloud_source.py` → `integrations/cloud/s3_source.py` + `integrations/cloud/gcs_source.py`

`cloud_source.py` contains two classes: `S3Source` and `GCSSource`. They need to be separated into their own files.

- [ ] **Step 1: Read current `cloud_source.py` to identify class boundaries**

```bash
grep -n "^class " ai_vision_tool/io/cloud_source.py
```
Expected output: two class definitions — `S3Source` and `GCSSource` with their line numbers.

- [ ] **Step 2: Create `integrations/cloud/s3_source.py`**

Copy everything from the top of `cloud_source.py` up to (and including) the `S3Source` class. Add boto3 import guard at the top:

```python
from __future__ import annotations

import io

import cv2
import numpy as np

try:
    import boto3
except ImportError as exc:
    raise ImportError(
        "S3 source requires: pip install ai-vision-tool[cloud]"
    ) from exc

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame

# Paste S3Source class here (copy from cloud_source.py)
```

- [ ] **Step 3: Create `integrations/cloud/gcs_source.py`**

```python
from __future__ import annotations

import io

import cv2
import numpy as np

try:
    from google.cloud import storage
except ImportError as exc:
    raise ImportError(
        "GCS source requires: pip install ai-vision-tool[cloud]"
    ) from exc

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame

# Paste GCSSource class here (copy from cloud_source.py)
```

- [ ] **Step 4: Update `integrations/cloud/__init__.py`**

```python
"""Cloud storage source integrations (S3, GCS)."""
```

- [ ] **Step 5: Remove old `io/cloud_source.py`**

```bash
git rm ai_vision_tool/io/cloud_source.py
git add ai_vision_tool/integrations/cloud/s3_source.py \
        ai_vision_tool/integrations/cloud/gcs_source.py
```

- [ ] **Step 6: Sweep for old import references**

```bash
grep -rn "from ai_vision_tool\.io\.cloud_source" ai_vision_tool tests --include="*.py" | grep -v __pycache__
```
Update any hits to the appropriate `integrations.cloud.*` path.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: split cloud_source into integrations/cloud/ with import guards"
```

---

## Task 12: Move labeling integrations

**Files:**
- Move: `components/auto_labeller.py` → `integrations/labeling/auto_labeller.py`
- Move: `components/darknet_auto_labeler.py` → `integrations/labeling/darknet_auto_labeler.py`
- Move: `components/tensorflow_auto_labeler.py` → `integrations/labeling/tensorflow_auto_labeler.py`

- [ ] **Step 1: Move labeling files**

```bash
git mv ai_vision_tool/components/auto_labeller.py          ai_vision_tool/integrations/labeling/auto_labeller.py
git mv ai_vision_tool/components/darknet_auto_labeler.py    ai_vision_tool/integrations/labeling/darknet_auto_labeler.py
git mv ai_vision_tool/components/tensorflow_auto_labeler.py ai_vision_tool/integrations/labeling/tensorflow_auto_labeler.py
```

- [ ] **Step 2: Update `integrations/labeling/__init__.py`**

```python
"""Auto-labeling integrations (Darknet, TensorFlow)."""
```

- [ ] **Step 3: Check and fix imports in moved labeler files**

```bash
grep "^from\|^import" ai_vision_tool/integrations/labeling/auto_labeller.py \
                       ai_vision_tool/integrations/labeling/darknet_auto_labeler.py \
                       ai_vision_tool/integrations/labeling/tensorflow_auto_labeler.py
```
Fix any remaining `ai_vision_tool.components.*` references.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: move auto-labelers to integrations/labeling/"
```

---

## Task 13: Move CLI

**Files:**
- Move: `ai_vision_tool/cli.py` → `ai_vision_tool/cli/main.py`

- [ ] **Step 1: Move cli.py**

```bash
git mv ai_vision_tool/cli.py ai_vision_tool/cli/main.py
```

- [ ] **Step 2: Update imports inside `cli/main.py`**

The cli currently imports:
- `from ai_vision_tool.components import (...)` — update to import from new paths
- `from ai_vision_tool.components.augmentations.common import parse_component_profile` — update to `from ai_vision_tool.augmentation.common import parse_component_profile`
- `from ai_vision_tool.pipeline import AIVisionPipeline` — update to `from ai_vision_tool.pipelines import AIVisionPipeline`
- `import uvicorn` and `from ai_vision_tool.api_service import ...` — remove these (API is in backup/)
- The string literal `f"from ai_vision_tool.components import {name}\n{python_call}"` — update to `f"from ai_vision_tool import {name}\n{python_call}"`

Update all component imports to use the new top-level namespace:
```python
from ai_vision_tool import (
    AutoAdjustContrast,
    AutoOrient,
    Blur,
    Brightness,
    CameraGain,
    # ... same list, now sourced from top-level __init__.py
)
```

- [ ] **Step 3: Verify CLI module is importable**

```bash
python -c "import ai_vision_tool.cli.main; print('cli ok')"
```
Expected: `cli ok`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: move cli.py to cli/main.py and update internal imports"
```

---

## Task 14: Rewrite `__init__.py`

**Files:**
- Modify: `ai_vision_tool/__init__.py` — replace entire `_EXPORTS` with new paths, add globals() caching

- [ ] **Step 1: Write the new `__init__.py`**

```python
"""Top-level package for AI Vision Tool."""

import importlib

__version__ = "0.2.0"

_EXPORTS = {
    # core
    "AIVisionComponent": ("ai_vision_tool.core.base", "AIVisionComponent"),
    "AIVisionPipeline":  ("ai_vision_tool.pipelines.vision_pipeline", "AIVisionPipeline"),

    # preprocessing
    "AdaptiveThreshold":        ("ai_vision_tool.preprocessing.intensity",             "AdaptiveThreshold"),
    "AspectRatioFilter":        ("ai_vision_tool.preprocessing.quality",               "AspectRatioFilter"),
    "AutoAdjustContrast":       ("ai_vision_tool.preprocessing.auto_adjust_contrast",  "AutoAdjustContrast"),
    "AutoCrop":                 ("ai_vision_tool.preprocessing.geometry",              "AutoCrop"),
    "AutoOrient":               ("ai_vision_tool.preprocessing.auto_orient",           "AutoOrient"),
    "BGRToRGB":                 ("ai_vision_tool.preprocessing.intensity",             "BGRToRGB"),
    "BlurDetection":            ("ai_vision_tool.preprocessing.quality",               "BlurDetection"),
    "BoundingBoxClamp":         ("ai_vision_tool.preprocessing.geometry",              "BoundingBoxClamp"),
    "BoundingBoxNormalize":     ("ai_vision_tool.preprocessing.geometry",              "BoundingBoxNormalize"),
    "BrightnessCheck":          ("ai_vision_tool.preprocessing.quality",               "BrightnessCheck"),
    "CenterCrop":               ("ai_vision_tool.preprocessing.geometry",              "CenterCrop"),
    "CLAHE":                    ("ai_vision_tool.preprocessing.intensity",             "CLAHE"),
    "ContourExtraction":        ("ai_vision_tool.preprocessing.intensity",             "ContourExtraction"),
    "ConvertColorSpace":        ("ai_vision_tool.preprocessing.intensity",             "ConvertColorSpace"),
    "CorruptImageCheck":        ("ai_vision_tool.preprocessing.quality",               "CorruptImageCheck"),
    "Deblur":                   ("ai_vision_tool.preprocessing.intensity",             "Deblur"),
    "Denoise":                  ("ai_vision_tool.preprocessing.intensity",             "Denoise"),
    "Deskew":                   ("ai_vision_tool.preprocessing.geometry",              "Deskew"),
    "DuplicateImageCheck":      ("ai_vision_tool.preprocessing.quality",               "DuplicateImageCheck"),
    "EdgeDetection":            ("ai_vision_tool.preprocessing.intensity",             "EdgeDetection"),
    "FaceAlign":                ("ai_vision_tool.preprocessing.geometry",              "FaceAlign"),
    "FrameResizer":             ("ai_vision_tool.preprocessing.frame_resizer",         "FrameResizer"),
    "GammaCorrection":          ("ai_vision_tool.preprocessing.intensity",             "GammaCorrection"),
    "HistogramEqualization":    ("ai_vision_tool.preprocessing.intensity",             "HistogramEqualization"),
    "ImageQualityCheck":        ("ai_vision_tool.preprocessing.quality",               "ImageQualityCheck"),
    "LetterboxResize":          ("ai_vision_tool.preprocessing.geometry",              "LetterboxResize"),
    "MaskResize":               ("ai_vision_tool.preprocessing.geometry",              "MaskResize"),
    "MaxSizeFilter":            ("ai_vision_tool.preprocessing.quality",               "MaxSizeFilter"),
    "MinSizeFilter":            ("ai_vision_tool.preprocessing.quality",               "MinSizeFilter"),
    "Normalize":                ("ai_vision_tool.preprocessing.intensity",             "Normalize"),
    "ObjectCrop":               ("ai_vision_tool.preprocessing.geometry",              "ObjectCrop"),
    "PadToSquare":              ("ai_vision_tool.preprocessing.geometry",              "PadToSquare"),
    "PerspectiveCorrection":    ("ai_vision_tool.preprocessing.geometry",              "PerspectiveCorrection"),
    "RemoveBackground":         ("ai_vision_tool.preprocessing.classical_segmentation","RemoveBackground"),
    "RescalePixels":            ("ai_vision_tool.preprocessing.intensity",             "RescalePixels"),
    "Resize":                   ("ai_vision_tool.preprocessing.geometry",              "Resize"),
    "RGBToBGR":                 ("ai_vision_tool.preprocessing.intensity",             "RGBToBGR"),
    "Sharpen":                  ("ai_vision_tool.preprocessing.intensity",             "Sharpen"),
    "Standardize":              ("ai_vision_tool.preprocessing.intensity",             "Standardize"),
    "Threshold":                ("ai_vision_tool.preprocessing.intensity",             "Threshold"),
    "WhiteBalance":             ("ai_vision_tool.preprocessing.intensity",             "WhiteBalance"),

    # augmentation
    "AffineTransform":          ("ai_vision_tool.augmentation.geometric_random",  "AffineTransform"),
    "AugmentationSharpen":      ("ai_vision_tool.augmentation.blur_artifact",     "Sharpen"),
    "Blur":                     ("ai_vision_tool.augmentation.blur",              "Blur"),
    "BoundingBoxJitter":        ("ai_vision_tool.augmentation.composite",         "BoundingBoxJitter"),
    "Brightness":               ("ai_vision_tool.augmentation.brightness",        "Brightness"),
    "CameraGain":               ("ai_vision_tool.augmentation.camera_gain",       "CameraGain"),
    "ChannelShuffle":           ("ai_vision_tool.augmentation.weather_light",     "ChannelShuffle"),
    "CoarseDropout":            ("ai_vision_tool.augmentation.noise_dropout",     "CoarseDropout"),
    "ColorJitter":              ("ai_vision_tool.augmentation.weather_light",     "ColorJitter"),
    "CompressionArtifacts":     ("ai_vision_tool.augmentation.blur_artifact",     "CompressionArtifacts"),
    "CopyPaste":                ("ai_vision_tool.augmentation.composite",         "CopyPaste"),
    "Crop":                     ("ai_vision_tool.augmentation.crop",              "Crop"),
    "CutMix":                   ("ai_vision_tool.augmentation.composite",         "CutMix"),
    "Cutout":                   ("ai_vision_tool.augmentation.cutout",            "Cutout"),
    "DefocusBlur":              ("ai_vision_tool.augmentation.blur_artifact",     "DefocusBlur"),
    "Downscale":                ("ai_vision_tool.augmentation.blur_artifact",     "Downscale"),
    "ElasticTransform":         ("ai_vision_tool.augmentation.geometric_random",  "ElasticTransform"),
    "Emboss":                   ("ai_vision_tool.augmentation.blur_artifact",     "Emboss"),
    "Equalize":                 ("ai_vision_tool.augmentation.blur_artifact",     "Equalize"),
    "Exposure":                 ("ai_vision_tool.augmentation.exposure",          "Exposure"),
    "Flip":                     ("ai_vision_tool.augmentation.flip",              "Flip"),
    "GaussianBlur":             ("ai_vision_tool.augmentation.blur_artifact",     "GaussianBlur"),
    "GlassBlur":                ("ai_vision_tool.augmentation.blur_artifact",     "GlassBlur"),
    "Greyscale":                ("ai_vision_tool.augmentation.grayscale",         "Greyscale"),
    "GridDistortion":           ("ai_vision_tool.augmentation.geometric_random",  "GridDistortion"),
    "GridDropout":              ("ai_vision_tool.augmentation.noise_dropout",     "GridDropout"),
    "HSVShift":                 ("ai_vision_tool.augmentation.weather_light",     "HSVShift"),
    "Hue":                      ("ai_vision_tool.augmentation.hue",               "Hue"),
    "ISONoise":                 ("ai_vision_tool.augmentation.noise_dropout",     "ISONoise"),
    "InvertImage":              ("ai_vision_tool.augmentation.weather_light",     "InvertImage"),
    "JPEGCompression":          ("ai_vision_tool.augmentation.blur_artifact",     "JPEGCompression"),
    "MaskDropout":              ("ai_vision_tool.augmentation.noise_dropout",     "MaskDropout"),
    "MedianBlur":               ("ai_vision_tool.augmentation.blur_artifact",     "MedianBlur"),
    "MixUp":                    ("ai_vision_tool.augmentation.composite",         "MixUp"),
    "Mosaic":                   ("ai_vision_tool.augmentation.mosaic",            "Mosaic"),
    "Mosaic9":                  ("ai_vision_tool.augmentation.composite",         "Mosaic9"),
    "MotionBlur":               ("ai_vision_tool.augmentation.motion_blur",       "MotionBlur"),
    "MultiplicativeNoise":      ("ai_vision_tool.augmentation.noise_dropout",     "MultiplicativeNoise"),
    "Noise":                    ("ai_vision_tool.augmentation.noise",             "Noise"),
    "ObjectPaste":              ("ai_vision_tool.augmentation.composite",         "ObjectPaste"),
    "OpticalDistortion":        ("ai_vision_tool.augmentation.geometric_random",  "OpticalDistortion"),
    "PerspectiveTransform":     ("ai_vision_tool.augmentation.geometric_random",  "PerspectiveTransform"),
    "PixelDropout":             ("ai_vision_tool.augmentation.noise_dropout",     "PixelDropout"),
    "Posterize":                ("ai_vision_tool.augmentation.blur_artifact",     "Posterize"),
    "RandomBrightnessContrast": ("ai_vision_tool.augmentation.weather_light",     "RandomBrightnessContrast"),
    "RandomCrop":               ("ai_vision_tool.augmentation.geometric_random",  "RandomCrop"),
    "RandomErasing":            ("ai_vision_tool.augmentation.noise_dropout",     "RandomErasing"),
    "RandomFog":                ("ai_vision_tool.augmentation.weather_light",     "RandomFog"),
    "RandomGamma":              ("ai_vision_tool.augmentation.weather_light",     "RandomGamma"),
    "RandomOcclusion":          ("ai_vision_tool.augmentation.composite",         "RandomOcclusion"),
    "RandomPadding":            ("ai_vision_tool.augmentation.geometric_random",  "RandomPadding"),
    "RandomRain":               ("ai_vision_tool.augmentation.weather_light",     "RandomRain"),
    "RandomResize":             ("ai_vision_tool.augmentation.geometric_random",  "RandomResize"),
    "RandomResizedCrop":        ("ai_vision_tool.augmentation.geometric_random",  "RandomResizedCrop"),
    "RandomScale":              ("ai_vision_tool.augmentation.geometric_random",  "RandomScale"),
    "RandomShadow":             ("ai_vision_tool.augmentation.weather_light",     "RandomShadow"),
    "RandomSnow":               ("ai_vision_tool.augmentation.weather_light",     "RandomSnow"),
    "RandomSunFlare":           ("ai_vision_tool.augmentation.weather_light",     "RandomSunFlare"),
    "RGBShift":                 ("ai_vision_tool.augmentation.weather_light",     "RGBShift"),
    "Rotate90":                 ("ai_vision_tool.augmentation.rotate90",          "Rotate90"),
    "Rotation":                 ("ai_vision_tool.augmentation.rotation",          "Rotation"),
    "Saturation":               ("ai_vision_tool.augmentation.saturation",        "Saturation"),
    "SaltPepperNoise":          ("ai_vision_tool.augmentation.noise_dropout",     "SaltPepperNoise"),
    "Shear":                    ("ai_vision_tool.augmentation.shear",             "Shear"),
    "Solarize":                 ("ai_vision_tool.augmentation.blur_artifact",     "Solarize"),
    "Superpixel":               ("ai_vision_tool.augmentation.blur_artifact",     "Superpixel"),
    "ToSepia":                  ("ai_vision_tool.augmentation.weather_light",     "ToSepia"),
    "Translate":                ("ai_vision_tool.augmentation.geometric_random",  "Translate"),
    "ZoomBlur":                 ("ai_vision_tool.augmentation.blur_artifact",     "ZoomBlur"),

    # enhancement (cv2-only)
    "Denoiser":           ("ai_vision_tool.enhancement.denoiser",       "Denoiser"),
    "FrameEnhancer":      ("ai_vision_tool.enhancement.frame_enhancer", "FrameEnhancer"),
    "LowLightEnhancer":   ("ai_vision_tool.enhancement.low_light",      "LowLightEnhancer"),

    # capture
    "FrameGrabber":       ("ai_vision_tool.capture.frame_grabber",      "FrameGrabber"),
    "MotionDetector":     ("ai_vision_tool.capture.motion_detector",    "MotionDetector"),
    "PictureTaker":       ("ai_vision_tool.capture.image_capture",      "PictureTaker"),
    "BurstPictureTaker":  ("ai_vision_tool.capture.burst_image_capture","BurstPictureTaker"),
    "ROICapture":         ("ai_vision_tool.capture.roi_capture",        "ROICapture"),
    "VideoTaker":         ("ai_vision_tool.capture.video_capture",      "VideoTaker"),
    "TimeLapseCapture":   ("ai_vision_tool.capture.time_lapse_capture", "TimeLapseCapture"),

    # visualization
    "FrameAnnotator":     ("ai_vision_tool.visualization.frame_annotator", "FrameAnnotator"),

    # io
    "DatasetCollector":   ("ai_vision_tool.io.dataset_collector", "DatasetCollector"),
    "ImageExporter":      ("ai_vision_tool.io.image_exporter",    "ImageExporter"),
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

- [ ] **Step 2: Verify top-level import resolves a few key names**

```bash
python -c "
import ai_vision_tool as avt
c = avt.AIVisionComponent
p = avt.AIVisionPipeline
r = avt.Resize
b = avt.Blur
print('__init__ ok')
"
```
Expected: `__init__ ok`

- [ ] **Step 3: Commit**

```bash
git add ai_vision_tool/__init__.py
git commit -m "chore: rewrite __init__.py _EXPORTS to point to restructured module paths"
```

---

## Task 15: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update `[project].dependencies` — remove fastapi/uvicorn, add pyyaml**

```toml
[project]
dependencies = [
  "numpy>=1.26",
  "opencv-python>=4.8",
  "pyyaml>=6.0",
]
```

- [ ] **Step 2: Replace `[project.optional-dependencies]`**

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

- [ ] **Step 3: Update scripts entrypoint**

```toml
[project.scripts]
ai-vision-tool = "ai_vision_tool.cli.main:main"
```

- [ ] **Step 4: Mirror changes in `[tool.poetry.dependencies]` and `[tool.poetry.scripts]`**

```toml
[tool.poetry.dependencies]
python = ">=3.10,<4.0"
numpy = ">=1.26"
opencv-python = ">=4.8"
pyyaml = ">=6.0"

[tool.poetry.scripts]
ai-vision-tool = "ai_vision_tool.cli.main:main"
```

Remove `fastapi` and `uvicorn` from `[tool.poetry.dependencies]`.

- [ ] **Step 5: Update `[tool.poetry.extras]` to match new extras**

```toml
[tool.poetry.extras]
onnx      = ["onnxruntime"]
torch     = ["torch", "torchvision"]
tflite    = ["tflite-runtime"]
websocket = ["websockets"]
kafka     = ["confluent-kafka"]
streaming = ["websockets", "confluent-kafka"]
cloud     = ["boto3", "google-cloud-storage"]
detection = ["ultralytics", "mediapipe"]
segmentation = ["ultralytics", "segment-anything", "torch", "torchvision"]
tracking  = ["onnxruntime"]
api       = ["fastapi", "uvicorn"]
all       = ["onnxruntime", "torch", "torchvision", "websockets", "confluent-kafka",
             "boto3", "google-cloud-storage", "ultralytics", "mediapipe",
             "segment-anything", "fastapi", "uvicorn"]
```

- [ ] **Step 6: Verify pyproject.toml is valid**

```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('toml ok')"
```
Expected: `toml ok`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update pyproject.toml — remove fastapi from base, add all optional extras"
```

---

## Task 16: Update test import paths

**Files:**
- Modify: `tests/test_core_components.py`
- Modify: `tests/test_basic_augmentations.py`
- Modify: `tests/test_advanced_augmentations.py`
- Modify: `tests/test_preprocessing_components.py`
- Modify: `tests/test_capture_components.py`
- Modify: `tests/test_labeler_components.py`
- Modify: `tests/test_api.py`
- Modify: `tests/test_cli_file_processing.py`

- [ ] **Step 1: Update `tests/test_core_components.py`**

Replace old imports:
```python
# OLD
from ai_vision_tool.components.auto_labeller import AutoLabeller
from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components.dataset_collector import DatasetCollector
from ai_vision_tool.components.frame_annotator import FrameAnnotator
from ai_vision_tool.components.frame_enhancer import FrameEnhancer
from ai_vision_tool.components.frame_resizer import FrameResizer
from ai_vision_tool.components.motion_detector import MotionDetector
from ai_vision_tool.components.time_lapse import TimeLapseCapture

# NEW
from ai_vision_tool.integrations.labeling.auto_labeller import AutoLabeller
from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.io.dataset_collector import DatasetCollector
from ai_vision_tool.visualization.frame_annotator import FrameAnnotator
from ai_vision_tool.enhancement.frame_enhancer import FrameEnhancer
from ai_vision_tool.preprocessing.frame_resizer import FrameResizer
from ai_vision_tool.capture.motion_detector import MotionDetector
from ai_vision_tool.capture.time_lapse import TimeLapseCapture
```

- [ ] **Step 2: Update `tests/test_basic_augmentations.py`**

```python
# OLD
from ai_vision_tool.cli import load_profile_components
from ai_vision_tool.components.augmentations import (
    Blur, Brightness, CameraGain, Crop, Cutout, Exposure, Flip,
    Greyscale, Hue, MotionBlur, Mosaic, Noise, Rotate90, Rotation,
    Saturation, Shear,
)

# NEW
from ai_vision_tool.cli.main import load_profile_components
from ai_vision_tool.augmentation import (
    Blur, Brightness, CameraGain, Crop, Cutout, Exposure, Flip,
    Greyscale, Hue, MotionBlur, Mosaic, Noise, Rotate90, Rotation,
    Saturation, Shear,
)
```

- [ ] **Step 3: Update `tests/test_advanced_augmentations.py`**

```python
# OLD
from ai_vision_tool.components.augmentations import (...)

# NEW
from ai_vision_tool.augmentation import (...)
```
(Same pattern — replace `ai_vision_tool.components.augmentations` → `ai_vision_tool.augmentation`)

- [ ] **Step 4: Update `tests/test_preprocessing_components.py`**

```python
# OLD
from ai_vision_tool.components.preprocessing import (...)

# NEW
from ai_vision_tool.preprocessing import (...)
```

Also update `RemoveBackground` source if needed — it now comes from `classical_segmentation`:
```python
from ai_vision_tool.preprocessing import RemoveBackground  # still works via __init__.py
```

- [ ] **Step 5: Update `tests/test_capture_components.py`**

```python
# OLD
from ai_vision_tool.components.burst_picture_taker import BurstPictureTaker
from ai_vision_tool.components.frame_grabber import FrameGrabber
from ai_vision_tool.components.image_exporter import ImageExporter
from ai_vision_tool.components.picture_taker import PictureTaker
from ai_vision_tool.components.roi_capture import ROICapture
from ai_vision_tool.components.video_taker import VideoTaker

# NEW
from ai_vision_tool.capture.burst_image_capture import BurstPictureTaker
from ai_vision_tool.capture.frame_grabber import FrameGrabber
from ai_vision_tool.io.image_exporter import ImageExporter
from ai_vision_tool.capture.image_capture import PictureTaker
from ai_vision_tool.capture.roi_capture import ROICapture
from ai_vision_tool.capture.video_capture import VideoTaker
```

- [ ] **Step 6: Update `tests/test_labeler_components.py`**

```python
# OLD
from ai_vision_tool.components.darknet_auto_labeler import DarknetAutoLabeler
from ai_vision_tool.components.tensorflow_auto_labeler import TensorFlowAutoLabeler

# NEW
from ai_vision_tool.integrations.labeling.darknet_auto_labeler import DarknetAutoLabeler
from ai_vision_tool.integrations.labeling.tensorflow_auto_labeler import TensorFlowAutoLabeler
```

- [ ] **Step 7: Update `tests/test_api.py`**

Check current imports:
```bash
head -10 tests/test_api.py
```
If it imports from `ai_vision_tool.api` or `ai_vision_tool.api_service` (now in backup/), skip or stub this test file with a clear note:
```python
import pytest
pytest.skip("API module moved to backup/; skip until re-introduced via [api] extra", allow_module_level=True)
```

- [ ] **Step 8: Update `tests/test_cli_file_processing.py`**

```bash
head -10 tests/test_cli_file_processing.py
```
Update any `from ai_vision_tool.cli import ...` → `from ai_vision_tool.cli.main import ...`

- [ ] **Step 9: Verify all import-update scans are clean**

```bash
grep -rn "ai_vision_tool\.components\." tests --include="*.py"
```
Expected: no output.

- [ ] **Step 10: Commit**

```bash
git add tests/
git commit -m "test: update all test imports to new module paths after restructure"
```

---

## Task 17: Add `tests/test_imports.py`

**Files:**
- Create: `tests/test_imports.py`

- [ ] **Step 1: Write the lightweight base-install smoke test**

```python
"""Verify base install is lightweight — no heavy deps imported by default."""

import subprocess
import sys

import pytest


def test_import_ai_vision_tool_base_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import ai_vision_tool; print('ok')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert "onnxruntime" not in result.stderr.lower()
    assert "torch" not in result.stderr.lower()


def test_core_base_importable():
    from ai_vision_tool.core.base import AIVisionComponent
    assert AIVisionComponent is not None


def test_preprocessing_importable():
    from ai_vision_tool.preprocessing import Resize, CLAHE, LetterboxResize
    assert Resize is not None


def test_augmentation_importable():
    from ai_vision_tool.augmentation import Blur, Flip, Rotation
    assert Blur is not None


def test_enhancement_core_importable():
    from ai_vision_tool.enhancement.denoiser import Denoiser
    from ai_vision_tool.enhancement.low_light import LowLightEnhancer
    assert Denoiser is not None
    assert LowLightEnhancer is not None


def test_top_level_namespace_exports_core_classes():
    import ai_vision_tool as avt
    assert hasattr(avt, "AIVisionComponent")
    assert hasattr(avt, "AIVisionPipeline")
    assert hasattr(avt, "Resize")
    assert hasattr(avt, "Blur")
    assert hasattr(avt, "Denoiser")


@pytest.mark.parametrize("name", [
    "ObjectDetector", "ByteTracker", "SemanticSegmenter",
])
def test_optional_modules_not_in_top_level_namespace(name):
    import ai_vision_tool as avt
    assert not hasattr(avt, name), (
        f"{name} leaked into top-level namespace — must be imported from its submodule"
    )
```

- [ ] **Step 2: Run the new test**

```bash
pytest tests/test_imports.py -v
```
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_imports.py
git commit -m "test: add test_imports.py to verify base install boundary"
```

---

## Task 18: Delete dead directories + full test run

**Files:**
- Delete: `ai_vision_tool/components/` (everything remaining — the `__init__.py` files for components, augmentations, preprocessing)
- Delete: any empty `ai_vision_tool/pipeline/` remnants

- [ ] **Step 1: Verify `components/` only contains `__init__.py` files (all source files already moved)**

```bash
find ai_vision_tool/components -name "*.py" | grep -v __pycache__ | sort
```
Expected: only `__init__.py` files and empty subdirs. If any source `.py` remains, investigate before deleting.

- [ ] **Step 2: Remove `components/` directory**

```bash
git rm -r ai_vision_tool/components/
```

- [ ] **Step 3: Remove `pipeline/` directory (singular) if not already gone**

```bash
[ -d ai_vision_tool/pipeline ] && git rm -r ai_vision_tool/pipeline/ || echo "already removed"
```

- [ ] **Step 4: Verify no remaining broken references to deleted paths**

```bash
grep -rn "ai_vision_tool\.components\|from \.\.components\|from \.components" \
  ai_vision_tool tests --include="*.py" | grep -v __pycache__
```
Expected: no output.

- [ ] **Step 5: Run full test suite**

```bash
pytest -v 2>&1 | tee /tmp/test_output.txt | tail -40
```

- [ ] **Step 6: Fix any remaining failures**

Common failure patterns and fixes:

- `ModuleNotFoundError: No module named 'ai_vision_tool.components'` — grep for the import, update to new path
- `ModuleNotFoundError: No module named 'ai_vision_tool.pipeline'` — update to `ai_vision_tool.pipelines`
- `ImportError` from `ai_vision_tool.api_service` in cli — remove that import block (API is in backup/)
- Relative import errors in moved files — check if any `from ..` still references wrong parent

- [ ] **Step 7: Verify test count matches pre-restructure**

```bash
pytest --collect-only -q 2>&1 | tail -5
```
The number of collected tests should match what was present before (or more with the new `test_imports.py`).

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "feat!: restructure package layout — breaking change v0.2.0

- components/ removed; preprocessing, augmentation top-level
- enhancement/ split: cv2-only core, models/ for ONNX/DL
- integrations/ added for streaming, cloud, labeling connectors
- models/backends/ for runtime adapters (onnx, torch, tflite)
- pipeline/ (singular) merged into pipelines/
- cli.py moved to cli/main.py
- fastapi/uvicorn removed from base deps; pyyaml added
- All optional extras defined in pyproject.toml

Migrate: ai_vision_tool.components.* → ai_vision_tool.preprocessing.*
         ai_vision_tool.components.augmentations.* → ai_vision_tool.augmentation.*"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Target directory layout — Tasks 1–12
- ✅ Base install boundary — Task 15 (pyproject.toml) + Task 17 (test_imports.py)
- ✅ Optional extras — Task 15
- ✅ Import guard pattern — Task 9 (models/backends), Task 11 (cloud)
- ✅ `__init__.py` strategy with caching — Task 14
- ✅ Testing strategy — Tasks 16–17
- ✅ Migration execution order — Tasks 1–18 follow spec's 10-step order
- ✅ Breaking change — Task 18 commit message

**Risks:**
- `cloud_source.py` split (Task 11) requires manual copy-paste of class bodies — no git history for the split files. This is unavoidable when one file becomes two.
- `cli/main.py` imports `uvicorn` and `api_service` — removing those imports may break the `webcam` command if it invokes the API. Review CLI carefully in Task 13 before removing.
