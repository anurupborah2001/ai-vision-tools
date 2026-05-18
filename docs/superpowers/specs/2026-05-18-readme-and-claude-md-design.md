# Design: README.md and CLAUDE.md Rewrite

## Context

Package renamed from `ai-vision-flow` / `visionflow` to `ai-vision-tool` / `ai_vision_tool`.
Existing README.md is stale: all import paths reference `visionflow.*` (broken).
CLAUDE.md does not exist.

## Audience

Both ML/CV researchers (training pipelines, augmentation, dataset collection) and application
developers (embedding vision processing in Python apps, running the FastAPI service).

## Sample Image

All Python examples use `image = cv2.imread("images/github/sample.jpg")` as the input source.

---

## README.md Structure (Option B — Quickstart-first)

### 1. Header + Badges

- Title: `ai-vision-tool`
- Badges: PyPI version, Python 3.10+, License
- One-line description from pyproject.toml
- Feature bullet list (pipeline, preprocessing, augmentation, capture, API, CLI)

### 2. Installation

Order: pip → uv → poetry. Each with optional TensorFlow extra below.

```
pip install ai-vision-tool
pip install "ai-vision-tool[tensorflow]"

uv add ai-vision-tool
uv add "ai-vision-tool[tensorflow]"

poetry add ai-vision-tool
poetry add ai-vision-tool --extras tensorflow
```

Dev setup section: clone → install dev deps → pre-commit install.

### 3. 30-Second Quickstart

Single self-contained pipeline example:
- Load `images/github/sample.jpg` with `cv2.imread`
- Build `AIVisionPipeline` with `AutoOrient`, `AutoAdjustContrast`, `Flip`, `GaussianBlur`
- Execute and inspect output shape

### 4. Preprocessing

Import path: `from ai_vision_tool.components.preprocessing import ...`
Also available top-level: `from ai_vision_tool import AutoOrient, Resize, ...`

Sub-sections with examples:

- **Geometry**: `AutoOrient`, `Resize`, `LetterboxResize`, `CenterCrop`, `PadToSquare`,
  `PerspectiveCorrection`, `Deskew`, `AutoCrop`, `FaceAlign`, `ObjectCrop`,
  `BoundingBoxClamp`, `BoundingBoxNormalize`, `MaskResize`
- **Intensity & Color**: `AutoAdjustContrast`, `Normalize`, `Standardize`, `RescalePixels`,
  `ConvertColorSpace`, `BGRToRGB`, `CLAHE`, `HistogramEqualization`, `GammaCorrection`,
  `WhiteBalance`, `Denoise`, `Sharpen`, `Deblur`, `Threshold`, `AdaptiveThreshold`,
  `EdgeDetection`, `ContourExtraction`, `RemoveBackground`
- **Quality Checks**: `ImageQualityCheck`, `BlurDetection`, `BrightnessCheck`,
  `DuplicateImageCheck`, `CorruptImageCheck`, `AspectRatioFilter`, `MinSizeFilter`, `MaxSizeFilter`

Each group: 2-sentence description, import line, runnable example with `sample.jpg`.

### 5. Augmentation

Import path: `from ai_vision_tool.components.augmentations import ...`

Sub-sections:

- **Geometric & Spatial**: `Flip`, `Rotate90`, `Crop`, `Rotation`, `Shear`, `Translate`,
  `RandomResize`, `RandomScale`, `RandomCrop`, `RandomResizedCrop`, `RandomPadding`,
  `AffineTransform`, `PerspectiveTransform`, `ElasticTransform`, `GridDistortion`, `OpticalDistortion`
- **Lighting, Color & Weather**: `Brightness`, `Exposure`, `Hue`, `Saturation`, `Greyscale`,
  `ColorJitter`, `RandomGamma`, `RandomBrightnessContrast`, `RandomShadow`, `RandomSunFlare`,
  `RandomFog`, `RandomRain`, `RandomSnow`, `ChannelShuffle`, `RGBShift`, `HSVShift`,
  `ToSepia`, `InvertImage`
- **Blur, Compression & Texture**: `Blur`, `GaussianBlur`, `MedianBlur`, `GlassBlur`,
  `DefocusBlur`, `ZoomBlur`, `MotionBlur`, `CameraGain`, `Emboss`, `Posterize`, `Solarize`,
  `Equalize`, `CompressionArtifacts`, `JPEGCompression`, `Downscale`, `Superpixel`
- **Noise & Dropout**: `Noise`, `ISONoise`, `MultiplicativeNoise`, `SaltPepperNoise`,
  `CoarseDropout`, `GridDropout`, `RandomErasing`, `PixelDropout`, `MaskDropout`
- **Multi-Image & Annotation-Aware**: `Cutout`, `Mosaic`, `Mosaic9`, `MixUp`, `CutMix`,
  `CopyPaste`, `ObjectPaste`, `RandomOcclusion`, `BoundingBoxJitter`

Includes: batch example, augmentation profile JSON example.

### 6. Pipeline

`from ai_vision_tool.pipeline import AIVisionPipeline`

Composite pipeline example chaining preprocessing + augmentation + components.
Shows `pipeline.add()` fluent interface and `pipeline.execute()`.

### 7. Components

`from ai_vision_tool.components import FrameEnhancer, MotionDetector, ...`

Sub-sections for: frame processors, capture helpers, dataset/export, auto-labeling.
Each with constructor + `.run()` example.

### 8. Capture Templates

`from ai_vision_tool.capture.image_template import image_template`
`from ai_vision_tool.capture.video_template import video_capture_template, save_screenshot`

Short examples for each template function.

### 9. FastAPI Service

How to run (`--serve-api`, `ai-vision-tool-api`), key endpoints, request/response JSON.
Includes payload-style request for metadata-aware components.

### 10. CLI Usage

`ai-vision-tool` entrypoint — `--process-image-path`, `--show-examples`, webcam flags.
Augmentation profile JSON usage.

### 11. Output Structure

Directory tree for `output/`.

### 12. Testing

`pytest` commands per test file group.

### 13. Build & Publish

`python -m build`, pointer to `PUBLISHING.md`.

---

## CLAUDE.md Structure

### 1. Package Identity

- PyPI: `ai-vision-tool`
- Import namespace: `ai_vision_tool`
- CLI entrypoints: `ai-vision-tool`, `ai-vision-tool-api`
- Version: `0.2.0`

### 2. Module Map

```
ai_vision_tool/
  components/
    preprocessing/   — geometry, intensity, quality, segmentation
    augmentations/   — geometric_random, weather_light, blur_artifact, noise_dropout, composite, ...
    base.py          — AIVisionComponent base class
    frame_enhancer.py, frame_resizer.py, motion_detector.py, ...
  pipeline/
    vision_pipeline.py — AIVisionPipeline (Chain of Responsibility)
  capture/
    image_template.py, video_template.py
  api.py            — FastAPI app factory
  api_service.py    — encode/decode/execute helpers
  cli.py            — argparse entrypoint + examples catalog
  __init__.py       — lazy import registry (_EXPORTS dict)
```

### 3. Design Patterns

- **Lazy imports**: `__init__.py` uses `__getattr__` + `_EXPORTS` dict. Add new top-level
  exports there, not with direct imports.
- **Payload convention**: components accept either a NumPy array OR a dict
  `{"frame": ndarray, ...}`. `run()` handles both.
- **Component interface**: all components subclass `AIVisionComponent` and implement `run(data, config)`.
- **Pipeline**: `AIVisionPipeline.execute()` chains component outputs. Each processor
  receives the previous processor's output as its input.

### 4. Dev Workflow

```bash
uv sync --dev          # or: poetry install --with dev
pre-commit install
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg

pytest                 # run all tests
pytest tests/test_preprocessing_components.py
pytest tests/test_basic_augmentations.py
pytest tests/test_advanced_augmentations.py

ruff check .
black .
isort .
```

Commit convention: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).

### 5. Release

See `PUBLISHING.md` for the release checklist and PyPI upload commands.

---

## Constraints

- All import paths must use `ai_vision_tool` (not `visionflow`).
- pip install command: `ai-vision-tool` (not `ai-vision-flow`).
- CLI entrypoints: `ai-vision-tool` and `ai-vision-tool-api`.
- Image source for examples: `cv2.imread("images/github/sample.jpg")`.
- No fabricated API URLs or badge URLs — use real PyPI shield pattern.
