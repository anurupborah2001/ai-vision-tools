# README.md and CLAUDE.md Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite README.md with correct `ai_vision_tool` imports, verbose per-module examples using `images/github/sample.jpg`, and create CLAUDE.md covering package identity, module architecture, design patterns, and dev workflow.

**Architecture:** Complete file replacement for README.md (existing file has stale `visionflow.*` imports throughout — all broken after rename to `ai_vision_tool`). New CLAUDE.md created from scratch. No code changes — documentation only. Split into one commit per major section to keep diffs reviewable.

**Tech Stack:** Markdown, Python code examples using `ai_vision_tool`, OpenCV (`cv2`), NumPy.

---

### Task 1: README — Header, Badges, and Installation

**Files:**
- Modify: `README.md` (replace entire file — start fresh to avoid stale content)

- [ ] **Step 1: Write README.md from scratch with header, badges, feature list, and installation**

Replace `README.md` with this content (Tasks 2–9 will append to it):

```markdown
# ai-vision-tool

[![PyPI version](https://img.shields.io/pypi/v/ai-vision-tool)](https://pypi.org/project/ai-vision-tool/)
[![Python](https://img.shields.io/pypi/pyversions/ai-vision-tool)](https://pypi.org/project/ai-vision-tool/)
[![License](https://img.shields.io/pypi/l/ai-vision-tool)](https://pypi.org/project/ai-vision-tool/)

**ai-vision-tool** is a modular computer-vision toolkit built around composable pipeline
components. It provides preprocessing, augmentation, webcam capture, dataset collection,
and an HTTP service layer — all usable from Python, the command line, or a FastAPI server.

![Sample](images/github/sample.jpg)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Preprocessing](#preprocessing)
- [Augmentation](#augmentation)
- [Pipeline](#pipeline)
- [Components](#components)
- [Capture Templates](#capture-templates)
- [FastAPI Service](#fastapi-service)
- [CLI Reference](#cli-reference)
- [Output Structure](#output-structure)
- [Testing](#testing)
- [Build and Publish](#build-and-publish)

## Features

- Composable pipeline via `AIVisionPipeline` — chain any mix of components with one interface
- 40+ preprocessing transforms: geometry, intensity, color space, quality checks, segmentation
- 70+ augmentation components: geometric, weather, blur, noise, dropout, multi-image composition
- Webcam capture, burst, ROI crop, time-lapse, and video recording helpers
- Dataset collection with label-aware folder structure
- FastAPI service layer for HTTP-based image processing
- CLI with live augmentation profile loading from JSON
- Optional TensorFlow and Darknet auto-labeler integrations

---

## Installation

### pip

```bash
pip install ai-vision-tool
```

With optional TensorFlow integration:

```bash
pip install "ai-vision-tool[tensorflow]"
```

### uv

```bash
uv add ai-vision-tool
```

With optional TensorFlow integration:

```bash
uv add "ai-vision-tool[tensorflow]"
```

### Poetry

```bash
poetry add ai-vision-tool
```

With optional TensorFlow integration:

```bash
poetry add --extras tensorflow ai-vision-tool
```

### Development Setup

Clone the repository and install all dev dependencies:

```bash
git clone https://github.com/your-org/ai-vision-tool.git
cd ai-vision-tool

# Using uv
uv sync --dev

# Using Poetry
poetry install --with dev
```

Install pre-commit hooks (required for formatting, linting, and commit validation):

```bash
pre-commit install
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg
```

Run the full hook suite manually:

```bash
pre-commit run --all-files
```
```

- [ ] **Step 2: Verify**

Confirm the file starts with `# ai-vision-tool`, badges use `ai-vision-tool` (not `ai-vision-flow`), and all three package managers are covered.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README header, badges, and installation section"
```

---

### Task 2: README — 30-Second Quickstart

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append the Quickstart section**

Append to `README.md`:

```markdown
---

## Quickstart

Install the package and run this to process your first image through a pipeline in under
30 seconds. The example loads a real image, runs four components in sequence, and prints
the output shape.

```python
import cv2
from ai_vision_tool.pipeline import AIVisionPipeline
from ai_vision_tool.components.preprocessing import AutoOrient, AutoAdjustContrast
from ai_vision_tool.components.augmentations import Flip, GaussianBlur

# Load image
image = cv2.imread("images/github/sample.jpg")

# Build a pipeline
pipeline = AIVisionPipeline()
pipeline.add(AutoOrient(rotation=90))
pipeline.add(AutoAdjustContrast(method="adaptive_equalization", clip_limit=2.0))
pipeline.add(Flip(horizontal=True))
pipeline.add(GaussianBlur(kernel_size=5, sigma_x=1.0))

# Execute
result = pipeline.execute(
    initial_data={"frame": image},
    global_config={},
)

output = result["frame"]
print(output.shape)  # (height, width, 3)
```

You can also import any component directly from the top-level namespace:

```python
from ai_vision_tool import AutoOrient, Flip, GaussianBlur, AIVisionPipeline
```

All imports use lazy loading, so only the modules you actually use are loaded.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add quickstart section to README"
```

---

### Task 3: README — Preprocessing

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append the Preprocessing section**

Append to `README.md`:

```markdown
---

## Preprocessing

Preprocessing transforms prepare raw images for downstream model inference, quality gating,
or dataset ingestion. Every component accepts either a NumPy array or a payload dictionary
`{"frame": ndarray, ...}`.

```python
import cv2
image = cv2.imread("images/github/sample.jpg")
```

### Import Path

```python
from ai_vision_tool.components.preprocessing import (
    AutoOrient,
    AutoAdjustContrast,
    Resize,
    LetterboxResize,
    CenterCrop,
    PadToSquare,
    Normalize,
    Standardize,
    RescalePixels,
    ConvertColorSpace,
    BGRToRGB,
    RGBToBGR,
    CLAHE,
    HistogramEqualization,
    GammaCorrection,
    WhiteBalance,
    Denoise,
    Sharpen,
    Deblur,
    RemoveBackground,
    Threshold,
    AdaptiveThreshold,
    EdgeDetection,
    ContourExtraction,
    PerspectiveCorrection,
    Deskew,
    AutoCrop,
    FaceAlign,
    ObjectCrop,
    BoundingBoxClamp,
    BoundingBoxNormalize,
    MaskResize,
    ImageQualityCheck,
    BlurDetection,
    BrightnessCheck,
    DuplicateImageCheck,
    CorruptImageCheck,
    AspectRatioFilter,
    MinSizeFilter,
    MaxSizeFilter,
)
```

---

### Geometry

Geometry transforms resize, crop, pad, and rectify images to a consistent spatial format.
They are typically the first stage in any preprocessing pipeline.

**`AutoOrient`** — Correct EXIF orientation metadata or apply an explicit rotation and flip.

```python
from ai_vision_tool.components.preprocessing import AutoOrient

# Rotate 90 degrees clockwise
result = AutoOrient(rotation=90).run(image)

# Flip horizontally without rotation
result = AutoOrient(flip_horizontal=True).run(image)

# Honour EXIF orientation tag stored in a payload dict
result = AutoOrient(use_exif=True, exif_key="exif_orientation").run(
    {"frame": image, "exif_orientation": 6}
)
```

**`Resize`** — Resize to an exact target size, ignoring aspect ratio.

```python
from ai_vision_tool.components.preprocessing import Resize

result = Resize(width=640, height=640).run(image)
```

**`LetterboxResize`** — Resize while preserving aspect ratio, padding the shorter axis.

```python
from ai_vision_tool.components.preprocessing import LetterboxResize

result = LetterboxResize(width=640, height=640, pad_value=(114, 114, 114)).run(image)
```

**`CenterCrop`** — Crop the centre region for classification inputs.

```python
from ai_vision_tool.components.preprocessing import CenterCrop

result = CenterCrop(width=224, height=224).run(image)
```

**`PadToSquare`** — Pad a rectangular image to a square canvas.

```python
from ai_vision_tool.components.preprocessing import PadToSquare

result = PadToSquare(pad_value=(0, 0, 0)).run(image)
```

**`PerspectiveCorrection`** — Rectify a quadrilateral document or planar surface.

```python
import numpy as np
from ai_vision_tool.components.preprocessing import PerspectiveCorrection

source_points = np.float32([[30, 20], [310, 10], [320, 240], [20, 250]])
result = PerspectiveCorrection(
    source_points=source_points,
    output_size=(300, 200),
).run(image)
```

**`Deskew`** — Rotate a document back to a levelled angle using skew estimation.

```python
from ai_vision_tool.components.preprocessing import Deskew

result = Deskew().run(image)
```

**`AutoCrop`** — Trim empty or near-black borders from around the active subject.

```python
from ai_vision_tool.components.preprocessing import AutoCrop

result = AutoCrop(threshold=10, padding=4).run(image)
```

**`FaceAlign`** — Align a face using eye landmark coordinates stored in a payload dict.

```python
from ai_vision_tool.components.preprocessing import FaceAlign

payload = {
    "frame": image,
    "metadata": {
        "left_eye": (40, 50),
        "right_eye": (90, 50),
    },
}
result = FaceAlign(output_size=(112, 112)).run(payload)
```

**`ObjectCrop`** — Crop the region described by bounding boxes in a payload dict.

```python
from ai_vision_tool.components.preprocessing import ObjectCrop

payload = {"frame": image, "bboxes": [(10, 20, 120, 80)]}
result = ObjectCrop().run(payload)
```

**`BoundingBoxClamp`** — Clamp bounding boxes that extend outside image boundaries.

```python
from ai_vision_tool.components.preprocessing import BoundingBoxClamp

payload = {"frame": image, "bboxes": [(-5, -5, 80, 90)]}
result = BoundingBoxClamp().run(payload)
```

**`BoundingBoxNormalize`** — Normalise absolute pixel bounding boxes to relative coordinates.

```python
from ai_vision_tool.components.preprocessing import BoundingBoxNormalize

payload = {"frame": image, "bboxes": [(10, 20, 120, 80)]}
result = BoundingBoxNormalize().run(payload)
```

**`MaskResize`** — Resize a payload mask to match a target spatial size.

```python
import numpy as np
from ai_vision_tool.components.preprocessing import MaskResize

mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
payload = {"frame": image, "mask": mask}
result = MaskResize(width=640, height=640).run(payload)
```

---

### Intensity and Color

Intensity and color transforms normalise pixel values, adjust exposure, and convert between
color spaces. Apply these after geometry but before feeding arrays into a model.

**`AutoAdjustContrast`** — Adaptive equalization, histogram equalization, or contrast stretching.

```python
from ai_vision_tool.components.preprocessing import AutoAdjustContrast

# Adaptive equalization (default)
result = AutoAdjustContrast(method="adaptive_equalization", clip_limit=2.0).run(image)

# Histogram equalization
result = AutoAdjustContrast(method="histogram_equalization").run(image)

# Contrast stretching between percentile bounds
result = AutoAdjustContrast(
    method="contrast_stretching",
    lower_percentile=2.0,
    upper_percentile=98.0,
    output_min=0,
    output_max=255,
).run(image)
```

**`Normalize`** — Map pixel values into [0, 1] (or a custom range).

```python
from ai_vision_tool.components.preprocessing import Normalize

result = Normalize().run(image)
```

**`Standardize`** — Standardise by per-channel mean and standard deviation (z-score style).

```python
from ai_vision_tool.components.preprocessing import Standardize

result = Standardize(per_channel=True).run(image)
```

**`RescalePixels`** — Explicit linear rescaling: `output = input * scale + offset`.

```python
from ai_vision_tool.components.preprocessing import RescalePixels

result = RescalePixels(scale=1.0 / 255.0, offset=0.0).run(image)
```

**`ConvertColorSpace`** — Convert between any two OpenCV-supported color spaces.

```python
from ai_vision_tool.components.preprocessing import ConvertColorSpace

result = ConvertColorSpace(source="BGR", target="RGB").run(image)
```

**`BGRToRGB`** / **`RGBToBGR`** — Shorthand converters for the most common swap.

```python
from ai_vision_tool.components.preprocessing import BGRToRGB, RGBToBGR

rgb_image = BGRToRGB().run(image)
bgr_image = RGBToBGR().run(rgb_image)
```

**`CLAHE`** — Contrast-Limited Adaptive Histogram Equalisation for local contrast boosting.

```python
from ai_vision_tool.components.preprocessing import CLAHE

result = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)).run(image)
```

**`HistogramEqualization`** — Global histogram equalisation on the luminance channel.

```python
from ai_vision_tool.components.preprocessing import HistogramEqualization

result = HistogramEqualization(color_space="ycrcb").run(image)
```

**`GammaCorrection`** — Gamma-based exposure tuning.

```python
from ai_vision_tool.components.preprocessing import GammaCorrection

result = GammaCorrection(gamma=1.4).run(image)  # brighten
result = GammaCorrection(gamma=0.7).run(image)  # darken
```

**`WhiteBalance`** — Correct per-channel colour casts.

```python
from ai_vision_tool.components.preprocessing import WhiteBalance

result = WhiteBalance(method="gray_world").run(image)
```

**`Denoise`** — Reduce sensor or compression noise.

```python
from ai_vision_tool.components.preprocessing import Denoise

result = Denoise(method="median", kernel_size=3).run(image)
```

**`Sharpen`** — Sharpen softened edges before downstream analysis.

```python
from ai_vision_tool.components.preprocessing import Sharpen

result = Sharpen(amount=1.0).run(image)
```

**`Deblur`** — Light deblurring via an unsharp-mask-style pass.

```python
from ai_vision_tool.components.preprocessing import Deblur

result = Deblur(amount=1.0).run(image)
```

**`Threshold`** — Binary threshold to create a mask.

```python
from ai_vision_tool.components.preprocessing import Threshold

result = Threshold(threshold=127, keep_channels=False).run(image)
```

**`AdaptiveThreshold`** — Local adaptive threshold using Gaussian or mean method.

```python
from ai_vision_tool.components.preprocessing import AdaptiveThreshold

result = AdaptiveThreshold(method="gaussian", block_size=11, keep_channels=False).run(image)
```

**`EdgeDetection`** — Extract edges via Canny, Sobel, or Laplacian.

```python
from ai_vision_tool.components.preprocessing import EdgeDetection

result = EdgeDetection(method="canny", threshold1=100, threshold2=200).run(image)
```

**`ContourExtraction`** — Populate contour metadata on a frame payload.

```python
from ai_vision_tool.components.preprocessing import ContourExtraction

payload = {"frame": image}
result = ContourExtraction().run(payload)
# result["contours"] contains the extracted contours
```

**`RemoveBackground`** — Mask or suppress the background region.

```python
from ai_vision_tool.components.preprocessing import RemoveBackground

payload = {"frame": image}
result = RemoveBackground(method="threshold", threshold=10, keep_mask=True).run(payload)
```

---

### Quality Checks

Quality components gate images before they enter a training pipeline or production flow.
They return a payload dict augmented with quality flags rather than a transformed image.

**`ImageQualityCheck`** — Compute blur and brightness quality flags in one pass.

```python
from ai_vision_tool.components.preprocessing import ImageQualityCheck

payload = {"frame": image}
result = ImageQualityCheck().run(payload)
# result["is_blurry"], result["brightness"] available in the returned payload
```

**`BlurDetection`** — Flag frames that fall below a Laplacian variance threshold.

```python
from ai_vision_tool.components.preprocessing import BlurDetection

payload = {"frame": image}
result = BlurDetection().run(payload)
```

**`BrightnessCheck`** — Validate that mean brightness falls within an acceptable range.

```python
from ai_vision_tool.components.preprocessing import BrightnessCheck

payload = {"frame": image}
result = BrightnessCheck(min_brightness=40.0, max_brightness=215.0).run(payload)
```

**`DuplicateImageCheck`** — Hash a frame and compare against a reference set.

```python
from ai_vision_tool.components.preprocessing import DuplicateImageCheck

payload = {"frame": image}
result = DuplicateImageCheck(reference_hashes=[]).run(payload)
```

**`CorruptImageCheck`** — Flag frames that are empty or structurally invalid.

```python
from ai_vision_tool.components.preprocessing import CorruptImageCheck

payload = {"frame": image}
result = CorruptImageCheck().run(payload)
```

**`AspectRatioFilter`** — Reject frames whose aspect ratio falls outside accepted bounds.

```python
from ai_vision_tool.components.preprocessing import AspectRatioFilter

payload = {"frame": image}
result = AspectRatioFilter(min_ratio=0.75, max_ratio=1.5).run(payload)
```

**`MinSizeFilter`** / **`MaxSizeFilter`** — Enforce minimum or maximum pixel dimensions.

```python
from ai_vision_tool.components.preprocessing import MinSizeFilter, MaxSizeFilter

payload = {"frame": image}
result = MinSizeFilter(min_width=320, min_height=320).run(payload)
result = MaxSizeFilter(max_width=2048, max_height=2048).run(payload)
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add preprocessing section with geometry, intensity, quality examples"
```

---

### Task 4: README — Augmentation (Geometric and Color/Weather)

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append augmentation intro and geometric/color sub-sections**

Append to `README.md`:

```markdown
---

## Augmentation

Augmentation components apply stochastic or deterministic transforms to build training-time
variation. Every component exposes the same `.run(input)` interface as preprocessing: pass
a NumPy array for simple transforms or a payload dict `{"frame": ndarray, ...}` for
annotation-aware components.

```python
import cv2
image = cv2.imread("images/github/sample.jpg")
```

### Import Path

```python
from ai_vision_tool.components.augmentations import (
    Flip, Rotate90, Crop, Rotation, Shear, Translate,
    RandomResize, RandomScale, RandomCrop, RandomResizedCrop, RandomPadding,
    AffineTransform, PerspectiveTransform, ElasticTransform,
    GridDistortion, OpticalDistortion,
    Brightness, Exposure, Hue, Saturation, Greyscale,
    ColorJitter, RandomGamma, RandomBrightnessContrast,
    RandomShadow, RandomSunFlare, RandomFog, RandomRain, RandomSnow,
    ChannelShuffle, RGBShift, HSVShift, ToSepia, InvertImage,
    Blur, GaussianBlur, MedianBlur, GlassBlur, DefocusBlur,
    ZoomBlur, MotionBlur, CameraGain,
    Emboss, Posterize, Solarize, Equalize,
    CompressionArtifacts, JPEGCompression, Downscale, Superpixel,
    Noise, ISONoise, MultiplicativeNoise, SaltPepperNoise,
    CoarseDropout, GridDropout, RandomErasing, PixelDropout, MaskDropout,
    Cutout, Mosaic, Mosaic9, MixUp, CutMix,
    CopyPaste, ObjectPaste, RandomOcclusion, BoundingBoxJitter,
)
```

---

### Geometric and Spatial

Geometric augmentations change the position, orientation, scale, or perspective of an image.
They are essential for training models that must be invariant to camera placement,
zoom level, and subject framing.

**`Flip`** — Mirror horizontally, vertically, or both.

```python
from ai_vision_tool.components.augmentations import Flip

result = Flip(horizontal=True).run(image)
result = Flip(vertical=True).run(image)
result = Flip(horizontal=True, vertical=True).run(image)
```

**`Rotate90`** — Rotate by multiples of 90 degrees. `k=1` is 90°, `k=2` is 180°, `k=3` is 270°.

```python
from ai_vision_tool.components.augmentations import Rotate90

result = Rotate90(k=1).run(image)
```

**`Crop`** — Extract a deterministic rectangular region.

```python
from ai_vision_tool.components.augmentations import Crop

result = Crop(x=20, y=20, width=200, height=200).run(image)
```

**`Rotation`** — Rotate by an arbitrary angle with configurable border handling.

```python
from ai_vision_tool.components.augmentations import Rotation

result = Rotation(angle=12.0, scale=1.0, expand=False, border_mode="constant").run(image)
```

**`Shear`** — Apply an affine shear along the X and/or Y axis.

```python
from ai_vision_tool.components.augmentations import Shear

result = Shear(shear_x=0.15, shear_y=0.0, border_mode="constant").run(image)
```

**`Translate`** — Shift the image spatially.

```python
from ai_vision_tool.components.augmentations import Translate

result = Translate(shift_x=12, shift_y=8, border_mode="constant").run(image)
```

**`RandomResize`** — Randomly resize within a width and height range.

```python
from ai_vision_tool.components.augmentations import RandomResize

result = RandomResize(min_width=320, max_width=640, min_height=320, max_height=640).run(image)
```

**`RandomScale`** — Randomly scale by a factor in [min_scale, max_scale].

```python
from ai_vision_tool.components.augmentations import RandomScale

result = RandomScale(min_scale=0.8, max_scale=1.2).run(image)
```

**`RandomCrop`** — Sample a random crop of fixed size.

```python
from ai_vision_tool.components.augmentations import RandomCrop

result = RandomCrop(crop_width=224, crop_height=224).run(image)
```

**`RandomResizedCrop`** — Random crop followed by resize, equivalent to `torchvision.RandomResizedCrop`.

```python
from ai_vision_tool.components.augmentations import RandomResizedCrop

result = RandomResizedCrop(
    output_width=224,
    output_height=224,
    scale_min=0.08,
    scale_max=1.0,
    ratio_min=0.75,
    ratio_max=1.3333,
).run(image)
```

**`RandomPadding`** — Add random amounts of padding to each edge.

```python
from ai_vision_tool.components.augmentations import RandomPadding

result = RandomPadding(
    max_top=16, max_bottom=16, max_left=16, max_right=16, pad_value=(0, 0, 0)
).run(image)
```

**`AffineTransform`** — Combined rotate/scale/translate/shear in one pass.

```python
from ai_vision_tool.components.augmentations import AffineTransform

result = AffineTransform(
    angle=8.0, scale=1.0, translate_x=10.0, translate_y=10.0,
    shear_x=0.05, shear_y=0.0, border_mode="constant",
).run(image)
```

**`PerspectiveTransform`** — Randomly perturb the four corners to simulate viewpoint change.

```python
from ai_vision_tool.components.augmentations import PerspectiveTransform

result = PerspectiveTransform(distortion_scale=0.15, border_mode="constant").run(image)
```

**`ElasticTransform`** — Apply elastic spatial distortion for medical imaging or OCR.

```python
from ai_vision_tool.components.augmentations import ElasticTransform

result = ElasticTransform(alpha=3.0, sigma=1.0, border_mode="reflect_101").run(image)
```

**`GridDistortion`** — Warp the image using a distorted sampling grid.

```python
from ai_vision_tool.components.augmentations import GridDistortion

result = GridDistortion(num_steps=5, distort_limit=0.2, border_mode="reflect_101").run(image)
```

**`OpticalDistortion`** — Simulate lens barrel or pincushion distortion.

```python
from ai_vision_tool.components.augmentations import OpticalDistortion

result = OpticalDistortion(k=0.00001, dx=0.0, dy=0.0, border_mode="constant").run(image)
```

---

### Lighting, Color, and Weather

These augmentations simulate real-world variation in lighting conditions, sensor
characteristics, and atmospheric effects. They are critical for models that must generalise
across times of day, environments, and camera types.

**`Brightness`** — Shift brightness by an additive offset.

```python
from ai_vision_tool.components.augmentations import Brightness

result = Brightness(beta=18).run(image)   # brighter
result = Brightness(beta=-18).run(image)  # darker
```

**`Exposure`** — Apply gamma-based exposure change.

```python
from ai_vision_tool.components.augmentations import Exposure

result = Exposure(gamma=1.3).run(image)
```

**`Hue`** — Shift the hue channel in HSV space.

```python
from ai_vision_tool.components.augmentations import Hue

result = Hue(delta=10).run(image)
```

**`Saturation`** — Scale saturation up or down.

```python
from ai_vision_tool.components.augmentations import Saturation

result = Saturation(scale=1.3).run(image)
```

**`Greyscale`** — Convert to grayscale, optionally preserving three output channels.

```python
from ai_vision_tool.components.augmentations import Greyscale

result = Greyscale(keep_channels=True).run(image)   # 3-channel gray
result = Greyscale(keep_channels=False).run(image)  # single channel
```

**`ColorJitter`** — Randomise brightness, contrast, saturation, and hue together.

```python
from ai_vision_tool.components.augmentations import ColorJitter

result = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=8).run(image)
```

**`RandomGamma`** — Randomise gamma within a range.

```python
from ai_vision_tool.components.augmentations import RandomGamma

result = RandomGamma(min_gamma=0.8, max_gamma=1.2).run(image)
```

**`RandomBrightnessContrast`** — Randomise brightness and contrast simultaneously.

```python
from ai_vision_tool.components.augmentations import RandomBrightnessContrast

result = RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2).run(image)
```

**`RandomShadow`** — Overlay a synthetic polygon shadow.

```python
from ai_vision_tool.components.augmentations import RandomShadow

result = RandomShadow(shadow_dimension=0.5, intensity=0.5).run(image)
```

**`RandomSunFlare`** — Overlay a sun flare hotspot.

```python
from ai_vision_tool.components.augmentations import RandomSunFlare

result = RandomSunFlare(radius=20, intensity=0.4).run(image)
```

**`RandomFog`** — Blend synthetic fog or haze into the frame.

```python
from ai_vision_tool.components.augmentations import RandomFog

result = RandomFog(alpha=0.2).run(image)
```

**`RandomRain`** — Render synthetic rain streaks.

```python
from ai_vision_tool.components.augmentations import RandomRain

result = RandomRain(drops=40, drop_length=12, intensity=0.25).run(image)
```

**`RandomSnow`** — Render synthetic snow particles.

```python
from ai_vision_tool.components.augmentations import RandomSnow

result = RandomSnow(intensity=0.1).run(image)
```

**`ChannelShuffle`** — Randomly permute the BGR/RGB channel order.

```python
from ai_vision_tool.components.augmentations import ChannelShuffle

result = ChannelShuffle().run(image)
```

**`RGBShift`** — Shift individual channels by independent offsets.

```python
from ai_vision_tool.components.augmentations import RGBShift

result = RGBShift(r_shift=10, g_shift=-5, b_shift=4).run(image)
```

**`HSVShift`** — Shift hue, saturation, and value directly in HSV space.

```python
from ai_vision_tool.components.augmentations import HSVShift

result = HSVShift(hue_shift=10, sat_shift=5, val_shift=5).run(image)
```

**`ToSepia`** — Apply a sepia tone effect.

```python
from ai_vision_tool.components.augmentations import ToSepia

result = ToSepia().run(image)
```

**`InvertImage`** — Invert all pixel values (`255 - pixel`).

```python
from ai_vision_tool.components.augmentations import InvertImage

result = InvertImage().run(image)
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add augmentation section — geometric and color/weather examples"
```

---

### Task 5: README — Augmentation (Blur/Compression, Noise/Dropout, Multi-Image, Batch, Profile)

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append blur/compression, noise/dropout, multi-image, batch, and profile sub-sections**

Append to `README.md`:

```markdown
---

### Blur, Compression, and Texture

These augmentations simulate degradation from camera optics, sensor quality, and lossy
compression. Use them to train models that must work reliably on mobile, CCTV, or
compressed-stream input.

**`Blur`** — Gaussian blur with configurable kernel size.

```python
from ai_vision_tool.components.augmentations import Blur

result = Blur(kernel_size=5, sigma_x=1.0).run(image)
```

**`GaussianBlur`** — Explicit Gaussian blur for augmentation pipelines.

```python
from ai_vision_tool.components.augmentations import GaussianBlur

result = GaussianBlur(kernel_size=5, sigma_x=1.0).run(image)
```

**`MedianBlur`** — Median blur, effective for salt-and-pepper-style noise.

```python
from ai_vision_tool.components.augmentations import MedianBlur

result = MedianBlur(kernel_size=5).run(image)
```

**`GlassBlur`** — Local pixel swap blur simulating glass refraction.

```python
from ai_vision_tool.components.augmentations import GlassBlur

result = GlassBlur(sigma=0.7, max_delta=2, iterations=1).run(image)
```

**`DefocusBlur`** — Simulate camera defocus using a disc kernel.

```python
from ai_vision_tool.components.augmentations import DefocusBlur

result = DefocusBlur(radius=5).run(image)
```

**`ZoomBlur`** — Simulate radial zoom-motion blur.

```python
from ai_vision_tool.components.augmentations import ZoomBlur

result = ZoomBlur(zoom_factor=1.2, steps=5).run(image)
```

**`MotionBlur`** — Directional blur simulating camera or subject motion.

```python
from ai_vision_tool.components.augmentations import MotionBlur

result = MotionBlur(kernel_size=11, angle=25.0).run(image)
```

**`CameraGain`** — Simulate sensor gain and black-level shifts.

```python
from ai_vision_tool.components.augmentations import CameraGain

result = CameraGain(gain=1.2, black_level=8.0).run(image)
```

**`Emboss`** — Create an embossed texture effect.

```python
from ai_vision_tool.components.augmentations import Emboss

result = Emboss(strength=1.0).run(image)
```

**`Posterize`** — Reduce tonal depth by quantising bit depth.

```python
from ai_vision_tool.components.augmentations import Posterize

result = Posterize(bits=4).run(image)
```

**`Solarize`** — Invert pixel values above a threshold (Sabattier effect).

```python
from ai_vision_tool.components.augmentations import Solarize

result = Solarize(threshold=128).run(image)
```

**`Equalize`** — Histogram equalisation as an augmentation.

```python
from ai_vision_tool.components.augmentations import Equalize

result = Equalize().run(image)
```

**`CompressionArtifacts`** — Simulate generic codec compression artifacts.

```python
from ai_vision_tool.components.augmentations import CompressionArtifacts

result = CompressionArtifacts(quality=40).run(image)
```

**`JPEGCompression`** — Round-trip through JPEG recompression.

```python
from ai_vision_tool.components.augmentations import JPEGCompression

result = JPEGCompression(quality=40).run(image)
```

**`Downscale`** — Downscale then restore to simulate low-resolution capture.

```python
from ai_vision_tool.components.augmentations import Downscale

result = Downscale(scale=0.5, interpolation="area").run(image)
```

**`Superpixel`** — Render an approximate superpixel abstraction.

```python
from ai_vision_tool.components.augmentations import Superpixel

result = Superpixel(region_size=10, ruler=10.0).run(image)
```

---

### Noise and Dropout

Noise and dropout augmentations improve robustness against sensor defects, transmission
loss, and occluded regions. They are particularly valuable for edge-device and CCTV
deployment scenarios.

**`Noise`** — Gaussian or salt-and-pepper noise.

```python
from ai_vision_tool.components.augmentations import Noise

result = Noise(mode="gaussian", mean=0.0, stddev=8.0).run(image)
result = Noise(mode="salt_pepper", amount=0.02, salt_vs_pepper=0.5).run(image)
```

**`ISONoise`** — Simulate camera ISO sensor noise (colour grain + luminance noise).

```python
from ai_vision_tool.components.augmentations import ISONoise

result = ISONoise(color_shift=0.01, intensity=0.5).run(image)
```

**`MultiplicativeNoise`** — Scale pixels by a random multiplier near 1.0.

```python
from ai_vision_tool.components.augmentations import MultiplicativeNoise

result = MultiplicativeNoise(multiplier_min=0.9, multiplier_max=1.1).run(image)
```

**`SaltPepperNoise`** — Classic salt-and-pepper pixel corruption.

```python
from ai_vision_tool.components.augmentations import SaltPepperNoise

result = SaltPepperNoise(amount=0.02, salt_vs_pepper=0.5).run(image)
```

**`CoarseDropout`** — Erase several rectangular patches (like CutOut but multiple holes).

```python
from ai_vision_tool.components.augmentations import CoarseDropout

result = CoarseDropout(holes=8, max_height=8, max_width=8, fill_value=0).run(image)
```

**`GridDropout`** — Drop pixels on a regular grid pattern.

```python
from ai_vision_tool.components.augmentations import GridDropout

result = GridDropout(ratio=0.5, unit_size=8, fill_value=0).run(image)
```

**`RandomErasing`** — Erase one random rectangular region.

```python
from ai_vision_tool.components.augmentations import RandomErasing

result = RandomErasing(scale=(0.02, 0.2), fill_value=0).run(image)
```

**`PixelDropout`** — Zero individual pixels at random.

```python
from ai_vision_tool.components.augmentations import PixelDropout

result = PixelDropout(dropout_prob=0.01, fill_value=0).run(image)
```

**`MaskDropout`** — Randomly zero out portions of a payload mask.

```python
import numpy as np
from ai_vision_tool.components.augmentations import MaskDropout

mask = (image[:, :, 0] > 128).astype("uint8")
payload = {"frame": image, "mask": mask}
result = MaskDropout(dropout_prob=0.1).run(payload)
```

---

### Multi-Image and Annotation-Aware

These augmentations combine two or more images or operate on bounding box and mask
annotations alongside the frame. They are most useful for detection and segmentation
training pipelines, and for long-tail class balancing via synthetic composition.

**`Cutout`** — Blank out a deterministic rectangular patch.

```python
from ai_vision_tool.components.augmentations import Cutout

result = Cutout(x=30, y=30, width=60, height=60, fill_value=(0, 0, 0)).run(image)
```

**`Mosaic`** — Compose a 2×2 mosaic from four images (YOLO-style data augmentation).

```python
from ai_vision_tool.components.augmentations import Mosaic

import cv2
image_a = cv2.imread("images/github/sample.jpg")
image_b = cv2.imread("images/github/sample.jpg")
image_c = cv2.imread("images/github/sample.jpg")

result = Mosaic(
    output_size=(640, 640),
    mosaic_images=[image_b, image_c, image_a],
).run(image_a)
```

**`Mosaic9`** — Compose a 3×3 mosaic from nine images.

```python
from ai_vision_tool.components.augmentations import Mosaic9

tiles = [image] * 8
result = Mosaic9(mosaic_images=tiles, output_size=(640, 640)).run(image)
```

**`MixUp`** — Blend a frame with a partner image at a given alpha.

```python
from ai_vision_tool.components.augmentations import MixUp

image_b = cv2.imread("images/github/sample.jpg")
payload = {"frame": image, "mix_image": image_b}
result = MixUp(alpha=0.5).run(payload)
```

**`CutMix`** — Replace a random patch in the frame with the same patch from a partner image.

```python
from ai_vision_tool.components.augmentations import CutMix

image_b = cv2.imread("images/github/sample.jpg")
payload = {"frame": image, "mix_image": image_b}
result = CutMix(alpha=0.5).run(payload)
```

**`CopyPaste`** — Paste an overlay image at a target position.

```python
from ai_vision_tool.components.augmentations import CopyPaste

overlay = cv2.imread("images/github/sample.jpg")
payload = {"frame": image, "overlay_image": overlay}
result = CopyPaste(x=10, y=10).run(payload)
```

**`ObjectPaste`** — Insert a cropped object image at a target position.

```python
from ai_vision_tool.components.augmentations import ObjectPaste

obj = cv2.imread("images/github/sample.jpg")
payload = {"frame": image, "object_image": obj}
result = ObjectPaste(x=20, y=30).run(payload)
```

**`RandomOcclusion`** — Hide a random rectangle to simulate partial obstruction.

```python
from ai_vision_tool.components.augmentations import RandomOcclusion

result = RandomOcclusion(max_width=20, max_height=20, fill_value=0).run(image)
```

**`BoundingBoxJitter`** — Perturb bounding box coordinates in a payload.

```python
from ai_vision_tool.components.augmentations import BoundingBoxJitter

payload = {"frame": image, "bboxes": [(10, 10, 100, 60)]}
result = BoundingBoxJitter(x_jitter=0.05, y_jitter=0.05, size_jitter=0.1).run(payload)
```

---

### Batch Processing

Every augmentation component supports batch execution. Pass a list of NumPy arrays and
receive a list of processed results:

```python
from ai_vision_tool.components.augmentations import Flip

import cv2
image_a = cv2.imread("images/github/sample.jpg")
image_b = cv2.imread("images/github/sample.jpg")
image_c = cv2.imread("images/github/sample.jpg")

augmenter = Flip(horizontal=True)
results = augmenter.run([image_a, image_b, image_c])
# results is a list of three flipped arrays
```

Chain multiple augmenters over a batch:

```python
from ai_vision_tool.components.augmentations import (
    ColorJitter, GaussianBlur, JPEGCompression, RandomResizedCrop,
)

augmenters = [
    RandomResizedCrop(output_width=256, output_height=256, scale_min=0.6, scale_max=1.0),
    ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=8),
    GaussianBlur(kernel_size=5, sigma_x=1.0),
    JPEGCompression(quality=45),
]

images = [cv2.imread("images/github/sample.jpg") for _ in range(8)]
for aug in augmenters:
    images = aug.run(images)
```

---

### Augmentation Profile (JSON)

Load augmentation pipelines from a JSON file without modifying Python code. Useful for
experiment tracking and CLI-driven workflows.

```json
[
  {
    "name": "RandomResizedCrop",
    "params": {
      "output_width": 256,
      "output_height": 256,
      "scale_min": 0.6,
      "scale_max": 1.0
    }
  },
  {
    "name": "ColorJitter",
    "params": {
      "brightness": 0.2,
      "contrast": 0.2,
      "saturation": 0.2,
      "hue": 8
    }
  },
  {
    "name": "GaussianBlur",
    "params": {
      "kernel_size": 5,
      "sigma_x": 1.0
    }
  }
]
```

Each entry supports:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | yes | Class name from `ai_vision_tool.components.augmentations` |
| `params` | no | Constructor keyword arguments |
| `module` | no | Fully-qualified module path if loading a custom component |

Use with the CLI:

```bash
ai-vision-tool --augmentation-config examples/augmentation_profile.json
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add augmentation section — blur, noise, multi-image, batch, profile"
```

---

### Task 6: README — Pipeline

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append Pipeline section**

Append to `README.md`:

```markdown
---

## Pipeline

`AIVisionPipeline` implements a Chain of Responsibility pattern. Each component receives
the output of the previous one as its input, allowing preprocessing, augmentation, and
runtime components to be composed freely in a single execution graph.

```python
from ai_vision_tool.pipeline import AIVisionPipeline
from ai_vision_tool.components.preprocessing import AutoOrient, AutoAdjustContrast, Resize
from ai_vision_tool.components.augmentations import Flip, GaussianBlur, ColorJitter
from ai_vision_tool.components import FrameAnnotator, MotionDetector

import cv2
image = cv2.imread("images/github/sample.jpg")

pipeline = AIVisionPipeline()

# Preprocessing stage
pipeline.add(AutoOrient(rotation=90))
pipeline.add(AutoAdjustContrast(method="adaptive_equalization", clip_limit=2.0))
pipeline.add(Resize(width=640, height=640))

# Augmentation stage
pipeline.add(Flip(horizontal=True))
pipeline.add(GaussianBlur(kernel_size=5, sigma_x=1.0))
pipeline.add(ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=5))

# Runtime components
pipeline.add(MotionDetector())
pipeline.add(FrameAnnotator())

# Execute
result = pipeline.execute(
    initial_data={"frame": image, "annotations": []},
    global_config={
        "min_area": 800,
        "draw_motion": True,
        "annotations": [{"type": "text", "text": "Pipeline Demo", "pos": (20, 30)}],
    },
)

output_frame = result["frame"]
print(output_frame.shape)
```

`pipeline.add()` returns `self` so calls can be chained:

```python
pipeline = (
    AIVisionPipeline()
    .add(AutoOrient(rotation=90))
    .add(Resize(width=640, height=640))
    .add(Flip(horizontal=True))
)
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add pipeline section with composite example"
```

---

### Task 7: README — Components

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append Components section**

Append to `README.md`:

```markdown
---

## Components

Components are higher-level building blocks that go beyond single-transform operations.
They manage state across frames (motion detection, time-lapse), coordinate I/O (capture,
export, recording), or wrap external model inference (auto-labeling).

```python
import cv2
image = cv2.imread("images/github/sample.jpg")
```

### Frame Processors

**`FrameEnhancer`** — Apply brightness, contrast, sharpening, denoising, and grayscale
conversion in a single pass. Intended for webcam and live-video pipelines.

```python
from ai_vision_tool.components import FrameEnhancer

enhancer = FrameEnhancer()
result = enhancer.run(
    {"frame": image},
    {
        "brightness": 10,
        "contrast": 1.15,
        "sharpen": True,
        "denoise": False,
        "grayscale": False,
    },
)
output = result["frame"]
```

**`FrameResizer`** — Resize frames using the classic component pipeline interface.

```python
from ai_vision_tool.components import FrameResizer

resizer = FrameResizer()
result = resizer.run(
    {"frame": image},
    {"size": (640, 480), "keep_aspect": True},
)
output = result["frame"]
```

**`MotionDetector`** — Detect motion regions across sequential frames using background
subtraction. Returns bounding boxes for regions above a minimum contour area.

```python
from ai_vision_tool.components import MotionDetector

detector = MotionDetector()
result = detector.run(
    {"frame": image},
    {"min_area": 800, "draw_motion": True},
)
# result["motion_boxes"] contains detected regions
```

**`FrameAnnotator`** — Render payload-driven annotations (text, rectangles) onto frames.

```python
from ai_vision_tool.components import FrameAnnotator

annotator = FrameAnnotator()
result = annotator.run(
    {
        "frame": image,
        "annotations": [
            {"type": "text", "text": "Demo Label", "pos": (20, 30)},
        ],
    },
    {},
)
output = result["frame"]
```

---

### Capture Helpers

**`PictureTaker`** — Interactive still-image capture from a webcam.

```python
from ai_vision_tool.components import PictureTaker

taker = PictureTaker()
taker.run(None, {"imgdir": "output/stills", "resolution": "1280x720", "camera_id": 0})
```

**`BurstPictureTaker`** — Capture a rapid sequence of frames.

```python
from ai_vision_tool.components import BurstPictureTaker

taker = BurstPictureTaker(burst_count=5, interval_seconds=0.2)
```

**`ROICapture`** — Save a crop defined by a region-of-interest rectangle.

```python
from ai_vision_tool.components import ROICapture

capture = ROICapture(roi=(100, 100, 300, 300))
```

**`VideoTaker`** — Interactive webcam recording.

```python
from ai_vision_tool.components import VideoTaker

taker = VideoTaker()
taker.run(None, {"viddir": "output/videos", "resolution": "1280x720", "camera_id": 0, "fps": 30.0})
```

**`FrameGrabber`** — Extract still frames from a video file at a configurable skip rate.

```python
from ai_vision_tool.components import FrameGrabber

grabber = FrameGrabber()
grabber.run("path/to/video.mp4", {"output_folder": "output/frames", "skip_frames": 90})
```

---

### Dataset and Export

**`DatasetCollector`** — Persist labelled dataset samples with metadata.

```python
from ai_vision_tool.components import DatasetCollector

collector = DatasetCollector()
collector.run(
    {"frame": image},
    {
        "save_sample": True,
        "output_dir": "output/dataset",
        "label": "forklift",
        "save_metadata": True,
        "metadata": {"source": "webcam"},
    },
)
```

**`TimeLapseCapture`** — Periodically save frames for time-lapse workflows.

```python
from ai_vision_tool.components import TimeLapseCapture

timelapse = TimeLapseCapture(output_dir="output/timelapse", interval_seconds=5)
timelapse.run({"frame": image}, {})
```

**`ImageExporter`** — Export grayscale and edge images from a payload frame.

```python
from ai_vision_tool.components import ImageExporter

exporter = ImageExporter(output_dir="output/exports")
exporter.run({"frame": image}, {"export_gray": True, "export_edges": True})
```

---

### Auto-Labeling

**`AutoLabeller`** — Base entrypoint for downstream auto-labeling integrations.

```python
from ai_vision_tool.components import AutoLabeller

labeller = AutoLabeller()
labeller.run(image)
```

**`DarknetAutoLabeler`** — Darknet/YOLO-powered labeling workflow.

```python
from ai_vision_tool.components import DarknetAutoLabeler

labeller = DarknetAutoLabeler()
labeller.run({"frame": image}, {"output_dir": "output/labels"})
```

**`TensorFlowAutoLabeler`** — TensorFlow Object Detection API labeling workflow.
Requires `pip install "ai-vision-tool[tensorflow]"`.

```python
from ai_vision_tool.components import TensorFlowAutoLabeler

labeller = TensorFlowAutoLabeler()
labeller.run({"frame": image}, {"output_dir": "output/labels"})
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add components section with frame processors, capture, dataset, labeling"
```

---

### Task 8: README — Capture Templates, FastAPI, and CLI

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append Capture Templates, FastAPI, and CLI sections**

Append to `README.md`:

```markdown
---

## Capture Templates

Capture templates are standalone helper functions for quick image display or live video
loops without building a full pipeline.

**`image_template`** — Display a still image with optional custom frame logic.

```python
from ai_vision_tool.capture.image_template import image_template

image_template(
    image_path="images/github/sample.jpg",
    custom_logic=lambda frame: frame,
    window_name="Preview",
    resolution=(1280, 720),
)
```

**`video_capture_template`** — Run a live webcam loop with custom per-frame logic.

```python
from ai_vision_tool.capture.video_template import video_capture_template

video_capture_template(
    video_source=0,
    custom_logic=lambda frame: frame,
    window_name="Live",
    resolution=(1280, 720),
    enable_recording=False,
    enable_screenshot=True,
)
```

**`save_screenshot`** — Save a frame to disk from within a template loop.

```python
from ai_vision_tool.capture.video_template import save_screenshot

save_screenshot(frame, output_dir="output/screenshots", prefix="capture")
```

---

## FastAPI Service

`ai-vision-tool` ships a FastAPI layer that exposes preprocessing, augmentation, and
component execution over HTTP. This is useful for integrating vision transforms into
microservice architectures or calling them from non-Python clients.

### Starting the Server

From Python:

```bash
python main.py --serve-api
```

With a custom host and port:

```bash
python main.py --serve-api --api-host 127.0.0.1 --api-port 8300
```

Using the installed CLI entrypoint directly:

```bash
ai-vision-tool-api
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/api/v1/catalog` | List all available components |
| `GET` | `/api/v1/catalog/{category}/{name}` | Describe one component |
| `POST` | `/api/v1/{category}/{name}` | Execute a component |

Supported categories: `preprocessing`, `augmentations`, `components`.

### Standard Image Request

Send a base64-encoded PNG and optional constructor/config arguments:

```json
{
  "image_base64": "<base64-encoded-png>",
  "init_args": {
    "rotation": 90
  },
  "config": {}
}
```

Example — rotate using `AutoOrient`:

```bash
curl -X POST http://localhost:8300/api/v1/preprocessing/AutoOrient \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64>", "init_args": {"rotation": 90}, "config": {}}'
```

### Payload-Style Request

For components that operate on payload dicts (bounding boxes, masks, metadata):

```json
{
  "payload": {
    "frame_base64": "<base64-encoded-png>",
    "bboxes": [[10, 20, 120, 80]]
  },
  "config": {}
}
```

Example — normalise bounding boxes:

```bash
curl -X POST http://localhost:8300/api/v1/preprocessing/BoundingBoxNormalize \
  -H "Content-Type: application/json" \
  -d '{"payload": {"frame_base64": "<base64>", "bboxes": [[10, 20, 120, 80]]}, "config": {}}'
```

---

## CLI Reference

### Process a Local Image File

The `--process-image-path` flag loads an image from disk, converts it to base64 internally,
and runs it through any registered component.

```bash
# Apply AutoOrient (preprocessing)
ai-vision-tool \
  --process-image-path \
  --component-category preprocessing \
  --component-name AutoOrient \
  --image-path images/github/sample.jpg \
  --init-args-json '{"rotation": 90}' \
  --save-output-image output/oriented.png

# Apply Flip (augmentation)
ai-vision-tool \
  --process-image-path \
  --component-category augmentations \
  --component-name Flip \
  --image-path images/github/sample.jpg \
  --init-args-json '{"horizontal": true}' \
  --save-output-image output/flipped.png

# Payload-aware component — BoundingBoxNormalize
ai-vision-tool \
  --process-image-path \
  --component-category preprocessing \
  --component-name BoundingBoxNormalize \
  --image-path images/github/sample.jpg \
  --payload-json '{"bboxes": [[10, 20, 120, 80]]}'
```

### Browse Built-In Examples

```bash
# List all examples
ai-vision-tool --show-examples

# Filter by category
ai-vision-tool --show-examples --example-category preprocessing
ai-vision-tool --show-examples --example-category augmentations
ai-vision-tool --show-examples --example-category components
ai-vision-tool --show-examples --example-category capture

# Filter by class name
ai-vision-tool --show-examples --example-name AutoOrient
ai-vision-tool --show-examples --example-name GaussianBlur
ai-vision-tool --show-examples --example-name FrameEnhancer
```

### Webcam Application

Run an interactive webcam session with a processing pipeline:

```bash
# Basic session
ai-vision-tool

# Enhancement + resize
ai-vision-tool \
  --enhance --brightness 12 --contrast 1.15 --sharpen \
  --resize --width 1280 --height 720 --keep-aspect

# Auto-orientation + contrast preprocessing
ai-vision-tool \
  --auto-orient --auto-orient-rotation 90 \
  --auto-adjust-contrast --contrast-method adaptive_equalization --clip-limit 2.5

# Augmentation flags
ai-vision-tool \
  --flip-horizontal \
  --rotation-angle 12 --rotation-expand \
  --blur --blur-kernel-size 7 \
  --noise --noise-mode gaussian --noise-stddev 8

# Motion detection + annotation
ai-vision-tool --motion --motion-area 1200 --annotate

# Time-lapse
ai-vision-tool --timelapse --timelapse-interval 10

# Dataset collection
ai-vision-tool --dataset --label forklift --enhance --resize

# Load augmentation profile from JSON
ai-vision-tool --augmentation-config examples/augmentation_profile.json
```

#### Webcam Hotkeys

| Key | Action |
|-----|--------|
| `p` | Capture a single processed frame to `output/captures` |
| `b` | Capture a burst of frames to `output/captures` |
| `r` | Start or stop video recording to `output/videos` |
| `d` | Save a dataset sample to `output/dataset/<label>` |
| `e` | Export grayscale and edge images to `output/exports` |
| `o` | Save the configured ROI crop to `output/captures` |
| `q` | Quit |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add capture templates, FastAPI, and CLI sections"
```

---

### Task 9: README — Component Index Tables, Output Structure, Testing, and Build

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append index tables, output structure, testing, and build sections**

Append to `README.md`:

```markdown
---

## Component Index

### Preprocessing

| Component | Purpose |
|-----------|---------|
| `AutoOrient` | EXIF or explicit rotation correction |
| `AutoAdjustContrast` | Adaptive, histogram, or stretch contrast |
| `Resize` | Exact spatial resize |
| `LetterboxResize` | Aspect-preserving resize with padding |
| `CenterCrop` | Centre crop for model inputs |
| `PadToSquare` | Square canvas padding |
| `Normalize` | Normalise pixel range |
| `Standardize` | z-score standardisation |
| `RescalePixels` | Explicit pixel scale and offset |
| `ConvertColorSpace` | Color-space conversion |
| `BGRToRGB` | OpenCV BGR to RGB |
| `RGBToBGR` | RGB to OpenCV BGR |
| `CLAHE` | Local contrast enhancement |
| `HistogramEqualization` | Histogram equalisation |
| `GammaCorrection` | Gamma-based exposure tuning |
| `WhiteBalance` | Colour cast correction |
| `Denoise` | Sensor or compression noise reduction |
| `Sharpen` | Edge sharpening |
| `Deblur` | Unsharp-mask deblur |
| `RemoveBackground` | Foreground isolation |
| `Threshold` | Binary thresholding |
| `AdaptiveThreshold` | Local adaptive thresholding |
| `EdgeDetection` | Edge extraction |
| `ContourExtraction` | Contour metadata generation |
| `PerspectiveCorrection` | Document or planar rectification |
| `Deskew` | Skew correction |
| `AutoCrop` | Trim empty borders |
| `FaceAlign` | Face normalisation from eye landmarks |
| `ObjectCrop` | Bounding-box crop extraction |
| `BoundingBoxClamp` | Clamp boxes to image bounds |
| `BoundingBoxNormalize` | Normalise bounding boxes |
| `MaskResize` | Payload mask resizing |
| `ImageQualityCheck` | Blur and brightness quality flags |
| `BlurDetection` | Blur threshold check |
| `BrightnessCheck` | Brightness range check |
| `DuplicateImageCheck` | Duplicate detection by hash |
| `CorruptImageCheck` | Corrupt or empty frame check |
| `AspectRatioFilter` | Aspect-ratio validation |
| `MinSizeFilter` | Minimum image-size validation |
| `MaxSizeFilter` | Maximum image-size validation |

### Augmentation

| Component | Purpose |
|-----------|---------|
| `Flip` | Mirror augmentation |
| `Rotate90` | 90-degree rotation |
| `Crop` | Deterministic crop |
| `Rotation` | Arbitrary-angle rotation |
| `Shear` | Affine shear |
| `Translate` | Spatial translation |
| `RandomResize` | Random size jitter |
| `RandomScale` | Random scale jitter |
| `RandomCrop` | Random crop |
| `RandomResizedCrop` | Random crop plus resize |
| `RandomPadding` | Random padding |
| `AffineTransform` | Combined affine transform |
| `PerspectiveTransform` | Perspective warp |
| `ElasticTransform` | Elastic distortion |
| `GridDistortion` | Grid warp |
| `OpticalDistortion` | Lens distortion |
| `Greyscale` | Grayscale augmentation |
| `Hue` | Hue shift |
| `Saturation` | Saturation scaling |
| `Brightness` | Brightness offset |
| `Exposure` | Gamma/exposure augmentation |
| `ColorJitter` | Compound color jitter |
| `RandomGamma` | Randomised gamma |
| `RandomBrightnessContrast` | Random brightness/contrast |
| `RandomShadow` | Synthetic shadows |
| `RandomSunFlare` | Flare overlay |
| `RandomFog` | Fog or haze overlay |
| `RandomRain` | Rain overlay |
| `RandomSnow` | Snow overlay |
| `ChannelShuffle` | Channel shuffle |
| `RGBShift` | Per-channel shift |
| `HSVShift` | HSV channel shift |
| `ToSepia` | Sepia colour effect |
| `InvertImage` | Image inversion |
| `Blur` | Gaussian blur |
| `GaussianBlur` | Explicit Gaussian blur |
| `MedianBlur` | Median blur |
| `GlassBlur` | Local glass blur |
| `DefocusBlur` | Defocus blur |
| `ZoomBlur` | Zoom blur |
| `MotionBlur` | Directional blur |
| `CameraGain` | Sensor gain simulation |
| `Emboss` | Emboss effect |
| `Posterize` | Reduce bit depth |
| `Solarize` | Highlight inversion |
| `Equalize` | Equalisation effect |
| `CompressionArtifacts` | Generic compression artifacts |
| `JPEGCompression` | JPEG recompression |
| `Downscale` | Low-resolution simulation |
| `Superpixel` | Superpixel rendering |
| `Noise` | Gaussian or salt-pepper noise |
| `ISONoise` | Sensor ISO noise |
| `MultiplicativeNoise` | Multiplicative noise |
| `SaltPepperNoise` | Salt-and-pepper noise |
| `CoarseDropout` | Block dropout |
| `GridDropout` | Grid dropout |
| `RandomErasing` | Random erasing |
| `PixelDropout` | Pixel-level dropout |
| `MaskDropout` | Mask dropout |
| `Cutout` | Deterministic rectangular masking |
| `Mosaic` | 2×2 mosaic composition |
| `Mosaic9` | 3×3 mosaic composition |
| `MixUp` | Image mixing |
| `CutMix` | Patch mixing |
| `CopyPaste` | Overlay paste |
| `ObjectPaste` | Object insertion |
| `RandomOcclusion` | Synthetic occlusion |
| `BoundingBoxJitter` | Bounding box perturbation |

---

## Output Structure

By default, the webcam application and CLI write all outputs under a single root:

```text
output/
├── captures/      — still images (p key, burst)
├── dataset/       — labelled training samples (d key)
├── exports/       — grayscale and edge exports (e key)
├── timelapse/     — periodic time-lapse frames
└── videos/        — recorded video files (r key)
```

Customise every directory with CLI flags:

```bash
ai-vision-tool \
  --output-root runs/session-01 \
  --output-dir stills \
  --video-dir recordings \
  --dataset-dir training-data \
  --export-dir processed
```

---

## Testing

```bash
# Run the full test suite
pytest

# Run individual suites
pytest tests/test_preprocessing_components.py
pytest tests/test_basic_augmentations.py
pytest tests/test_advanced_augmentations.py
pytest tests/test_capture_components.py
pytest tests/test_core_components.py
pytest tests/test_labeler_components.py
pytest tests/test_api.py
pytest tests/test_cli_file_processing.py
```

---

## Build and Publish

```bash
python -m pip install --upgrade build
python -m build
```

The wheel and source distribution are written to `dist/`.

See `PUBLISHING.md` for the release checklist and PyPI upload commands.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add component index tables, output structure, testing, and build sections"
```

---

### Task 10: Create CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write CLAUDE.md**

Create `CLAUDE.md` with this content:

```markdown
# CLAUDE.md — ai-vision-tool

This file provides context for AI assistants navigating or modifying this repository.

---

## Package Identity

| Property | Value |
|----------|-------|
| PyPI package | `ai-vision-tool` |
| Python import namespace | `ai_vision_tool` |
| CLI entrypoint (webcam app + image processing) | `ai-vision-tool` |
| CLI entrypoint (FastAPI server) | `ai-vision-tool-api` |
| Package version | `0.2.0` |
| Requires Python | `>=3.10,<4.0` |

> **Important:** The package was renamed from `ai-vision-flow` / `visionflow` to
> `ai-vision-tool` / `ai_vision_tool`. All imports must use `ai_vision_tool`.
> Never use `visionflow` or `ai_vision_flow`.

---

## Module Map

```
ai_vision_tool/
│
├── __init__.py              # Lazy import registry (_EXPORTS dict + __getattr__)
├── __main__.py              # python -m ai_vision_tool entrypoint
├── cli.py                   # argparse entrypoint, examples catalog, webcam loop
├── api.py                   # FastAPI app factory (create_app, app, run)
├── api_service.py           # encode_image_base64, decode_image_base64, execute_component
│
├── pipeline/
│   └── vision_pipeline.py   # AIVisionPipeline (Chain of Responsibility)
│
├── components/
│   ├── base.py              # AIVisionComponent base class
│   ├── _image_utils.py      # Shared image helper utilities
│   │
│   ├── preprocessing/       # Preprocessing transforms
│   │   ├── geometry.py      # Resize, LetterboxResize, CenterCrop, PadToSquare,
│   │   │                    # PerspectiveCorrection, Deskew, AutoCrop, FaceAlign,
│   │   │                    # ObjectCrop, BoundingBoxClamp, BoundingBoxNormalize, MaskResize
│   │   ├── intensity.py     # Normalize, Standardize, RescalePixels, ConvertColorSpace,
│   │   │                    # BGRToRGB, RGBToBGR, CLAHE, HistogramEqualization,
│   │   │                    # GammaCorrection, WhiteBalance, Denoise, Sharpen, Deblur,
│   │   │                    # Threshold, AdaptiveThreshold, EdgeDetection, ContourExtraction
│   │   ├── quality.py       # ImageQualityCheck, BlurDetection, BrightnessCheck,
│   │   │                    # DuplicateImageCheck, CorruptImageCheck, AspectRatioFilter,
│   │   │                    # MinSizeFilter, MaxSizeFilter
│   │   ├── segmentation.py  # RemoveBackground
│   │   ├── auto_orient.py   # AutoOrient
│   │   └── auto_adjust_contrast.py  # AutoAdjustContrast
│   │
│   ├── augmentations/       # Augmentation transforms
│   │   ├── blur.py          # Blur
│   │   ├── blur_artifact.py # GaussianBlur, MedianBlur, GlassBlur, DefocusBlur, ZoomBlur,
│   │   │                    # Emboss, Posterize, Solarize, Equalize, CompressionArtifacts,
│   │   │                    # JPEGCompression, Downscale, Superpixel, Sharpen (augmentation)
│   │   ├── brightness.py    # Brightness
│   │   ├── camera_gain.py   # CameraGain
│   │   ├── composite.py     # MixUp, CutMix, CopyPaste, ObjectPaste, RandomOcclusion,
│   │   │                    # BoundingBoxJitter, Mosaic9
│   │   ├── crop.py          # Crop
│   │   ├── cutout.py        # Cutout
│   │   ├── exposure.py      # Exposure
│   │   ├── flip.py          # Flip
│   │   ├── geometric_random.py  # RandomResize, RandomScale, RandomCrop, RandomResizedCrop,
│   │   │                        # RandomPadding, Translate, AffineTransform,
│   │   │                        # PerspectiveTransform, ElasticTransform,
│   │   │                        # GridDistortion, OpticalDistortion
│   │   ├── greyscale.py     # Greyscale
│   │   ├── hue.py           # Hue
│   │   ├── mosaic.py        # Mosaic
│   │   ├── motion_blur.py   # MotionBlur
│   │   ├── noise.py         # Noise
│   │   ├── noise_dropout.py # ISONoise, MultiplicativeNoise, SaltPepperNoise,
│   │   │                    # CoarseDropout, GridDropout, RandomErasing,
│   │   │                    # PixelDropout, MaskDropout
│   │   ├── rotate90.py      # Rotate90
│   │   ├── rotation.py      # Rotation
│   │   ├── saturation.py    # Saturation
│   │   ├── shear.py         # Shear
│   │   ├── weather_light.py # RandomShadow, RandomSunFlare, RandomFog, RandomRain,
│   │   │                    # RandomSnow, RandomGamma, ColorJitter, ChannelShuffle,
│   │   │                    # RGBShift, HSVShift, ToSepia, InvertImage,
│   │   │                    # RandomBrightnessContrast
│   │   └── common.py        # parse_component_profile (JSON profile loader)
│   │
│   ├── frame_enhancer.py    # FrameEnhancer
│   ├── frame_resizer.py     # FrameResizer
│   ├── frame_annotator.py   # FrameAnnotator
│   ├── frame_grabber.py     # FrameGrabber
│   ├── motion_detector.py   # MotionDetector
│   ├── picture_taker.py     # PictureTaker
│   ├── burst_picture_taker.py  # BurstPictureTaker
│   ├── roi_capture.py       # ROICapture
│   ├── video_taker.py       # VideoTaker
│   ├── time_lapse_capture.py   # TimeLapseCapture
│   ├── time_lapse.py        # TimeLapse
│   ├── dataset_collector.py # DatasetCollector
│   ├── image_exporter.py    # ImageExporter
│   ├── auto_labeller.py     # AutoLabeller
│   ├── darknet_auto_labeler.py   # DarknetAutoLabeler
│   └── tensorflow_auto_labeler.py  # TensorFlowAutoLabeler
│
├── capture/
│   ├── image_template.py    # image_template()
│   └── video_template.py    # video_capture_template(), save_screenshot()
│
├── config/                  # Configuration stubs (future phases)
├── core/                    # Core stubs (future phases)
├── detection/               # Stub (future phases)
├── enhancement/             # Stub (future phases)
├── io/                      # Stub (future phases)
├── models/                  # Stub (future phases)
├── pipelines/               # Stub (future phases)
├── segmentation/            # Stub (future phases)
├── streaming/               # Stub (future phases)
├── tracking/                # Stub (future phases)
├── utils/                   # Stub (future phases)
└── visualization/           # Stub (future phases)
```

---

## Design Patterns

### Lazy Imports

`__init__.py` uses a `_EXPORTS` dict and `__getattr__` to load modules only when first
accessed. When adding a new top-level export:

1. Add an entry to `_EXPORTS` in `ai_vision_tool/__init__.py`:
   ```python
   "MyNewClass": ("ai_vision_tool.components.my_module", "MyNewClass"),
   ```
2. Do **not** add a direct `from ... import ...` at the top of `__init__.py`.

### Payload Convention

Every component accepts either:
- A raw NumPy array: `component.run(image)`
- A payload dict: `component.run({"frame": image, "bboxes": [...], "mask": ..., ...})`

When a component returns a payload dict, the `"frame"` key always holds the processed
NumPy array. Downstream components receive the full dict as their input.

### Component Interface

All components subclass `AIVisionComponent` from `ai_vision_tool.components.base` and
implement:

```python
def run(self, data, config=None):
    ...
```

`config` is optional; some components only read constructor arguments.
`cleanup()` is available for components that hold resources (e.g., video writers).

### Pipeline Execution

`AIVisionPipeline.execute()` chains `processor.run(data, config)` calls. The output of
each processor becomes the `data` input of the next. `global_config` is passed unchanged
to every processor.

### Top-Level Namespace Exports

Only stable, commonly used classes are exported from `ai_vision_tool`. All classes are
also importable from their specific submodule paths.

---

## Dev Workflow

### Install Dependencies

```bash
# uv (recommended)
uv sync --dev

# Poetry
poetry install --with dev
```

### Run Tests

```bash
pytest                                          # all tests
pytest tests/test_preprocessing_components.py
pytest tests/test_basic_augmentations.py
pytest tests/test_advanced_augmentations.py
pytest tests/test_capture_components.py
pytest tests/test_core_components.py
pytest tests/test_labeler_components.py
pytest tests/test_api.py
pytest tests/test_cli_file_processing.py
```

### Lint and Format

```bash
ruff check .
black .
isort .
```

### Pre-Commit Hooks

```bash
# Install all hook types
pre-commit install
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg

# Run all hooks manually
pre-commit run --all-files
```

Hooks enforce: `ruff`, `isort`, `black`, `pre-commit-hooks`, Conventional Commits,
and `pytest` on pre-push.

### Commit Message Convention

This repository uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add fog and rain augmentation coverage
fix: correct augmentation profile parameter names
docs: update publishing workflow
test: add missing unit tests for basic augmentations
chore: rename template → capture module
```

---

## Release

See `PUBLISHING.md` for the full release checklist and PyPI upload commands.

Key steps:
1. Bump version in `pyproject.toml` (`[project].version` and `[tool.poetry].version`)
2. Update `__version__` in `ai_vision_tool/__init__.py`
3. Run `python -m build`
4. Run `twine upload dist/*`
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: create CLAUDE.md with package identity, module map, patterns, and dev workflow"
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| pip install `ai-vision-tool` | Task 1 |
| uv install | Task 1 |
| poetry install | Task 1 |
| TensorFlow optional extra | Task 1 |
| Dev setup + pre-commit | Task 1 |
| 30-second quickstart | Task 2 |
| Preprocessing — geometry | Task 3 |
| Preprocessing — intensity/color | Task 3 |
| Preprocessing — quality checks | Task 3 |
| Augmentation — geometric | Task 4 |
| Augmentation — lighting/color/weather | Task 4 |
| Augmentation — blur/compression | Task 5 |
| Augmentation — noise/dropout | Task 5 |
| Augmentation — multi-image | Task 5 |
| Batch example | Task 5 |
| Augmentation profile JSON | Task 5 |
| Pipeline section | Task 6 |
| Components — frame processors | Task 7 |
| Components — capture helpers | Task 7 |
| Components — dataset/export | Task 7 |
| Components — auto-labeling | Task 7 |
| Capture templates | Task 8 |
| FastAPI service | Task 8 |
| CLI — process-image-path | Task 8 |
| CLI — show-examples | Task 8 |
| CLI — webcam flags | Task 8 |
| Component index tables | Task 9 |
| Output structure | Task 9 |
| Testing commands | Task 9 |
| Build and publish | Task 9 |
| CLAUDE.md — package identity | Task 10 |
| CLAUDE.md — module map | Task 10 |
| CLAUDE.md — design patterns | Task 10 |
| CLAUDE.md — dev workflow | Task 10 |
| CLAUDE.md — release pointer | Task 10 |

All spec requirements covered. No TBDs or placeholders. Import paths consistent
(`ai_vision_tool` throughout). Image source `images/github/sample.jpg` used in every
Python example. Install command `ai-vision-tool` consistent in all sections.
