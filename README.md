# AI Vision Flow

`ai-vision-flow` is a modular computer-vision toolkit built around composable pipeline components. It provides a reusable Python package for image and video processing, along with a webcam-driven `main.py` application that exposes the most common runtime workflows through command-line arguments and keyboard controls.

The project is designed for rapid experimentation with:

- frame enhancement
- frame resizing
- motion detection
- on-screen annotation
- time-lapse capture
- dataset collection
- export workflows
- webcam-driven capture and recording

## Feature Overview

- Composable pipeline architecture through `visionflow.pipeline.AIVisionPipeline`
- Image enhancement with brightness, contrast, sharpening, denoising, and grayscale conversion
- Frame resizing with optional aspect-ratio preservation
- Motion detection with configurable minimum contour area
- Overlay-based annotation support
- ROI preview and ROI crop capture
- Burst capture, still capture, video recording, and export helpers
- Dataset sample persistence with label-aware folder structure
- Optional TensorFlow and Darknet auto-labeler integrations

## Documentation Map

The documentation now has two complementary modes:

- technical reference in this `README.md`
- a polished KAMI-style one-pager in [docs/kami-overview-en.html](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/docs/kami-overview-en.html:1)

Use the one-pager when you want a fast editorial overview of the project. Use the README when you need the detailed API, CLI, testing, and release references.

## Installation

### Install From PyPI

```bash
pip install ai-vision-flow
```

Optional TensorFlow integration:

```bash
pip install "ai-vision-flow[tensorflow]"
```

### Development With Poetry

```bash
poetry install --with dev
poetry run pytest
```

Optional TensorFlow extra in the Poetry environment:

```bash
poetry install --with dev -E tensorflow
```

### Code Quality And Commit Hooks

The repository uses `pre-commit` for formatting, linting, import ordering, commit-message validation, and test enforcement.

Install the hooks:

```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
poetry run pre-commit install --hook-type commit-msg
```

Run them on demand:

```bash
poetry run pre-commit run --all-files
```

## Process Local Images From The CLI

The CLI can now take an image file path directly, convert the image to base64 internally, and process it through the same execution layer used by the FastAPI service.

Required arguments:

- `--process-image-path`
- `--component-category preprocessing|augmentations|components`
- `--component-name <ClassName>`
- `--image-path <path/to/image>`

Optional arguments:

- `--init-args-json` for constructor arguments
- `--config-json` for runtime configuration
- `--payload-json` for payload-aware methods that need extra metadata such as bounding boxes or masks
- `--data-json` for raw non-payload data
- `--batch-json` for batch execution
- `--save-output-image` to persist the returned image or returned payload frame

Examples:

Rotate an image with preprocessing:

```bash
python main.py \
  --process-image-path \
  --component-category preprocessing \
  --component-name AutoOrient \
  --image-path path/to/image.jpg \
  --init-args-json '{"rotation": 90}' \
  --save-output-image output/auto_oriented.png
```

Apply an augmentation:

```bash
python main.py \
  --process-image-path \
  --component-category augmentations \
  --component-name Flip \
  --image-path path/to/image.jpg \
  --init-args-json '{"horizontal": true}' \
  --save-output-image output/flipped.png
```

Run a payload-aware preprocessing method:

```bash
python main.py \
  --process-image-path \
  --component-category preprocessing \
  --component-name BoundingBoxNormalize \
  --image-path path/to/image.jpg \
  --payload-json '{"bboxes": [[10, 20, 120, 80]]}'
```

## FastAPI Service

The library now includes a FastAPI layer for exposing preprocessing, augmentation, and core component execution over HTTP.

Run the API from the existing CLI surface:

```bash
python main.py --serve-api
```

Custom host and port:

```bash
python main.py --serve-api --api-host 127.0.0.1 --api-port 8300
```

You can also use the console entrypoint directly:

```bash
ai-vision-flow-api
```

Key endpoints:

- `GET /health`
- `GET /api/v1/catalog`
- `GET /api/v1/catalog/{category}/{name}`
- `POST /api/v1/{category}/{name}`

Supported categories:

- `preprocessing`
- `augmentations`
- `components`

Example request:

```json
{
  "image_base64": "<base64-png>",
  "init_args": {
    "rotation": 90
  },
  "config": {}
}
```

Example payload-style request for metadata-aware components:

```json
{
  "payload": {
    "frame_base64": "<base64-png>",
    "bboxes": [[10, 20, 120, 80]]
  },
  "config": {}
}
```

Included checks:

- `ruff`
- `isort`
- `black`
- `pre-commit-hooks`
- `conventional-pre-commit` for Conventional Commit messages
- `pytest` on `pre-push`

Commit message examples:

- `feat: add fog and rain augmentation coverage`
- `fix: correct augmentation profile parameter names`
- `docs: update publishing workflow`
- `test: add missing unit tests for basic augmentations`
- `test: expand preprocessing branch coverage`

## KAMI Example Layout

This README now follows a KAMI-style reference layout for discoverability:

- `K`: key purpose of the method
- `A`: arguments or activation path
- `M`: method or component name
- `I`: invocation command for a runnable example

The fastest way to inspect usage examples from the CLI is:

```bash
python main.py --show-examples --example-category preprocessing
python main.py --show-examples --example-category augmentations
python main.py --show-examples --example-category components
python main.py --show-examples --example-category template
python main.py --show-examples --example-name AutoOrient
python main.py --show-examples --example-name video_capture_template
```

If you want a concise document-style summary instead of the long-form technical reference, open:

- [docs/kami-overview-en.html](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/docs/kami-overview-en.html:1)

### KAMI Preprocessing Index

| Method | KAMI Use | KAMI Invoke |
| --- | --- | --- |
| `AutoOrient` | EXIF or rotation correction | `python main.py --show-examples --example-name AutoOrient` |
| `AutoAdjustContrast` | automatic contrast enhancement | `python main.py --show-examples --example-name AutoAdjustContrast` |
| `Resize` | exact spatial resize | `python main.py --show-examples --example-name Resize` |
| `LetterboxResize` | aspect-preserving resize with padding | `python main.py --show-examples --example-name LetterboxResize` |
| `CenterCrop` | center crop for model inputs | `python main.py --show-examples --example-name CenterCrop` |
| `PadToSquare` | square canvas padding | `python main.py --show-examples --example-name PadToSquare` |
| `Normalize` | normalize pixel range | `python main.py --show-examples --example-name Normalize` |
| `Standardize` | z-score style standardization | `python main.py --show-examples --example-name Standardize` |
| `RescalePixels` | explicit pixel scaling and offset | `python main.py --show-examples --example-name RescalePixels` |
| `ConvertColorSpace` | color-space conversion | `python main.py --show-examples --example-name ConvertColorSpace` |
| `BGRToRGB` | OpenCV to RGB conversion | `python main.py --show-examples --example-name BGRToRGB` |
| `RGBToBGR` | RGB back to OpenCV BGR | `python main.py --show-examples --example-name RGBToBGR` |
| `CLAHE` | local contrast enhancement | `python main.py --show-examples --example-name CLAHE` |
| `HistogramEqualization` | histogram equalization | `python main.py --show-examples --example-name HistogramEqualization` |
| `GammaCorrection` | gamma-based exposure tuning | `python main.py --show-examples --example-name GammaCorrection` |
| `WhiteBalance` | color cast correction | `python main.py --show-examples --example-name WhiteBalance` |
| `Denoise` | denoising for noisy inputs | `python main.py --show-examples --example-name Denoise` |
| `Sharpen` | preprocessing sharpen pass | `python main.py --show-examples --example-name Sharpen` |
| `Deblur` | light deblurring pass | `python main.py --show-examples --example-name Deblur` |
| `RemoveBackground` | foreground isolation | `python main.py --show-examples --example-name RemoveBackground` |
| `Threshold` | binary thresholding | `python main.py --show-examples --example-name Threshold` |
| `AdaptiveThreshold` | local thresholding | `python main.py --show-examples --example-name AdaptiveThreshold` |
| `EdgeDetection` | edge extraction | `python main.py --show-examples --example-name EdgeDetection` |
| `ContourExtraction` | contour metadata generation | `python main.py --show-examples --example-name ContourExtraction` |
| `PerspectiveCorrection` | document or planar rectification | `python main.py --show-examples --example-name PerspectiveCorrection` |
| `Deskew` | skew correction | `python main.py --show-examples --example-name Deskew` |
| `AutoCrop` | trim empty borders | `python main.py --show-examples --example-name AutoCrop` |
| `FaceAlign` | face normalization from eye landmarks | `python main.py --show-examples --example-name FaceAlign` |
| `ObjectCrop` | bounding-box crop extraction | `python main.py --show-examples --example-name ObjectCrop` |
| `BoundingBoxClamp` | clamp boxes into image bounds | `python main.py --show-examples --example-name BoundingBoxClamp` |
| `BoundingBoxNormalize` | normalize bounding boxes | `python main.py --show-examples --example-name BoundingBoxNormalize` |
| `MaskResize` | payload mask resizing | `python main.py --show-examples --example-name MaskResize` |
| `ImageQualityCheck` | quality summary flags | `python main.py --show-examples --example-name ImageQualityCheck` |
| `BlurDetection` | blur threshold check | `python main.py --show-examples --example-name BlurDetection` |
| `BrightnessCheck` | brightness range check | `python main.py --show-examples --example-name BrightnessCheck` |
| `DuplicateImageCheck` | duplicate detection by hash | `python main.py --show-examples --example-name DuplicateImageCheck` |
| `CorruptImageCheck` | corrupt or empty frame check | `python main.py --show-examples --example-name CorruptImageCheck` |
| `AspectRatioFilter` | aspect-ratio validation | `python main.py --show-examples --example-name AspectRatioFilter` |
| `MinSizeFilter` | minimum image-size validation | `python main.py --show-examples --example-name MinSizeFilter` |
| `MaxSizeFilter` | maximum image-size validation | `python main.py --show-examples --example-name MaxSizeFilter` |

### KAMI Augmentation Index

| Method | KAMI Use | KAMI Invoke |
| --- | --- | --- |
| `Flip` | mirror augmentation | `python main.py --show-examples --example-name Flip` |
| `Rotate90` | 90-degree rotation | `python main.py --show-examples --example-name Rotate90` |
| `Crop` | deterministic crop | `python main.py --show-examples --example-name Crop` |
| `Rotation` | arbitrary-angle rotation | `python main.py --show-examples --example-name Rotation` |
| `Shear` | affine shear | `python main.py --show-examples --example-name Shear` |
| `Greyscale` | grayscale augmentation | `python main.py --show-examples --example-name Greyscale` |
| `Hue` | hue shift | `python main.py --show-examples --example-name Hue` |
| `Saturation` | saturation scaling | `python main.py --show-examples --example-name Saturation` |
| `Brightness` | brightness offset | `python main.py --show-examples --example-name Brightness` |
| `Exposure` | gamma/exposure augmentation | `python main.py --show-examples --example-name Exposure` |
| `Blur` | basic Gaussian blur | `python main.py --show-examples --example-name Blur` |
| `Noise` | Gaussian or salt-pepper noise | `python main.py --show-examples --example-name Noise` |
| `Cutout` | deterministic rectangular masking | `python main.py --show-examples --example-name Cutout` |
| `Mosaic` | 2x2 mosaic composition | `python main.py --show-examples --example-name Mosaic` |
| `MotionBlur` | directional blur | `python main.py --show-examples --example-name MotionBlur` |
| `CameraGain` | sensor gain simulation | `python main.py --show-examples --example-name CameraGain` |
| `RandomResize` | random size jitter | `python main.py --show-examples --example-name RandomResize` |
| `RandomScale` | random scale jitter | `python main.py --show-examples --example-name RandomScale` |
| `RandomCrop` | random crop | `python main.py --show-examples --example-name RandomCrop` |
| `RandomResizedCrop` | random crop plus resize | `python main.py --show-examples --example-name RandomResizedCrop` |
| `RandomPadding` | random padding | `python main.py --show-examples --example-name RandomPadding` |
| `Translate` | spatial translation | `python main.py --show-examples --example-name Translate` |
| `AffineTransform` | combined affine transform | `python main.py --show-examples --example-name AffineTransform` |
| `PerspectiveTransform` | perspective warp | `python main.py --show-examples --example-name PerspectiveTransform` |
| `ElasticTransform` | elastic distortion | `python main.py --show-examples --example-name ElasticTransform` |
| `GridDistortion` | grid warp | `python main.py --show-examples --example-name GridDistortion` |
| `OpticalDistortion` | lens distortion | `python main.py --show-examples --example-name OpticalDistortion` |
| `RandomShadow` | synthetic shadows | `python main.py --show-examples --example-name RandomShadow` |
| `RandomSunFlare` | flare overlay | `python main.py --show-examples --example-name RandomSunFlare` |
| `RandomFog` | fog or haze overlay | `python main.py --show-examples --example-name RandomFog` |
| `RandomRain` | rain overlay | `python main.py --show-examples --example-name RandomRain` |
| `RandomSnow` | snow overlay | `python main.py --show-examples --example-name RandomSnow` |
| `RandomGamma` | randomized gamma | `python main.py --show-examples --example-name RandomGamma` |
| `ColorJitter` | compound color jitter | `python main.py --show-examples --example-name ColorJitter` |
| `ChannelShuffle` | channel shuffle | `python main.py --show-examples --example-name ChannelShuffle` |
| `RGBShift` | per-channel shift | `python main.py --show-examples --example-name RGBShift` |
| `Posterize` | reduce bit depth | `python main.py --show-examples --example-name Posterize` |
| `Solarize` | highlight inversion | `python main.py --show-examples --example-name Solarize` |
| `Equalize` | equalization effect | `python main.py --show-examples --example-name Equalize` |
| `Emboss` | emboss effect | `python main.py --show-examples --example-name Emboss` |
| `GaussianBlur` | explicit Gaussian blur | `python main.py --show-examples --example-name GaussianBlur` |
| `MedianBlur` | median blur | `python main.py --show-examples --example-name MedianBlur` |
| `GlassBlur` | local glass blur | `python main.py --show-examples --example-name GlassBlur` |
| `DefocusBlur` | defocus blur | `python main.py --show-examples --example-name DefocusBlur` |
| `ZoomBlur` | zoom blur | `python main.py --show-examples --example-name ZoomBlur` |
| `ISONoise` | sensor ISO noise | `python main.py --show-examples --example-name ISONoise` |
| `MultiplicativeNoise` | multiplicative noise | `python main.py --show-examples --example-name MultiplicativeNoise` |
| `SaltPepperNoise` | salt-and-pepper noise | `python main.py --show-examples --example-name SaltPepperNoise` |
| `CoarseDropout` | block dropout | `python main.py --show-examples --example-name CoarseDropout` |
| `GridDropout` | grid dropout | `python main.py --show-examples --example-name GridDropout` |
| `RandomErasing` | random erasing | `python main.py --show-examples --example-name RandomErasing` |
| `MixUp` | image mixing | `python main.py --show-examples --example-name MixUp` |
| `CutMix` | patch mixing | `python main.py --show-examples --example-name CutMix` |
| `CopyPaste` | overlay paste | `python main.py --show-examples --example-name CopyPaste` |
| `RandomOcclusion` | synthetic occlusion | `python main.py --show-examples --example-name RandomOcclusion` |
| `ObjectPaste` | object insertion | `python main.py --show-examples --example-name ObjectPaste` |
| `BoundingBoxJitter` | bbox perturbation | `python main.py --show-examples --example-name BoundingBoxJitter` |
| `MaskDropout` | mask dropout | `python main.py --show-examples --example-name MaskDropout` |
| `Mosaic9` | 3x3 mosaic composition | `python main.py --show-examples --example-name Mosaic9` |
| `HSVShift` | HSV channel shift | `python main.py --show-examples --example-name HSVShift` |
| `ToSepia` | sepia color effect | `python main.py --show-examples --example-name ToSepia` |
| `InvertImage` | image inversion | `python main.py --show-examples --example-name InvertImage` |
| `CompressionArtifacts` | generic compression artifacts | `python main.py --show-examples --example-name CompressionArtifacts` |
| `JPEGCompression` | JPEG recompression | `python main.py --show-examples --example-name JPEGCompression` |
| `Downscale` | low-resolution simulation | `python main.py --show-examples --example-name Downscale` |
| `Superpixel` | superpixel rendering effect | `python main.py --show-examples --example-name Superpixel` |
| `PixelDropout` | pixel-level dropout | `python main.py --show-examples --example-name PixelDropout` |
| `RandomBrightnessContrast` | random brightness/contrast | `python main.py --show-examples --example-name RandomBrightnessContrast` |

### KAMI Components And Template Index

| Method | KAMI Use | KAMI Invoke |
| --- | --- | --- |
| `FrameEnhancer` | classic enhancement pipeline component | `python main.py --show-examples --example-name FrameEnhancer` |
| `FrameResizer` | classic resize component | `python main.py --show-examples --example-name FrameResizer` |
| `MotionDetector` | motion-box detection | `python main.py --show-examples --example-name MotionDetector` |
| `FrameAnnotator` | frame annotation overlay | `python main.py --show-examples --example-name FrameAnnotator` |
| `DatasetCollector` | dataset image persistence | `python main.py --show-examples --example-name DatasetCollector` |
| `TimeLapseCapture` | periodic frame capture | `python main.py --show-examples --example-name TimeLapseCapture` |
| `PictureTaker` | interactive still capture | `python main.py --show-examples --example-name PictureTaker` |
| `BurstPictureTaker` | interactive burst capture | `python main.py --show-examples --example-name BurstPictureTaker` |
| `ROICapture` | interactive ROI capture helper | `python main.py --show-examples --example-name ROICapture` |
| `ImageExporter` | grayscale and edge export | `python main.py --show-examples --example-name ImageExporter` |
| `FrameGrabber` | extract frames from video | `python main.py --show-examples --example-name FrameGrabber` |
| `VideoTaker` | interactive video recording | `python main.py --show-examples --example-name VideoTaker` |
| `AutoLabeller` | auto-labeling base entrypoint | `python main.py --show-examples --example-name AutoLabeller` |
| `DarknetAutoLabeler` | Darknet labeling flow | `python main.py --show-examples --example-name DarknetAutoLabeler` |
| `TensorFlowAutoLabeler` | TensorFlow labeling flow | `python main.py --show-examples --example-name TensorFlowAutoLabeler` |
| `image_template` | legacy still-image template | `python main.py --show-examples --example-name image_template` |
| `save_screenshot` | legacy screenshot helper | `python main.py --show-examples --example-name save_screenshot` |
| `video_capture_template` | legacy live video template | `python main.py --show-examples --example-name video_capture_template` |

## Package Usage

The library can be used directly in your own Python code.

```python
from visionflow.pipeline import AIVisionPipeline
from visionflow import FrameEnhancer, FrameResizer, MotionDetector

pipeline = AIVisionPipeline()
pipeline.add(FrameEnhancer())
pipeline.add(FrameResizer())
pipeline.add(MotionDetector())

result = pipeline.execute(
    initial_data={"frame": frame},
    global_config={
        "brightness": 10,
        "contrast": 1.1,
        "size": (1280, 720),
        "keep_aspect": True,
        "min_area": 1200,
    },
)
```

## Preprocessing And Augmentation Components

The library now includes dedicated preprocessing and augmentation classes organized under:

- `visionflow.components.preprocessing`
- `visionflow.components.augmentations`

They work with:

- a single image as a NumPy array
- a dictionary payload such as `{"frame": image, "metadata": {...}}`
- a batch of images because `AIVisionComponent.run()` already supports list input

### Preprocessing Components

- `AutoOrient(use_exif=True, exif_key="exif_orientation", rotation=None, flip_horizontal=False, flip_vertical=False)`
  Supports EXIF-style orientation correction, explicit 90/180/270 degree rotation, and optional flipping.

- `AutoAdjustContrast(method="adaptive_equalization", clip_limit=2.0, tile_grid_size=(8, 8), lower_percentile=2.0, upper_percentile=98.0, output_min=0, output_max=255)`
  Supports:
  - adaptive equalization
  - histogram equalization
  - contrast stretching

Implementation imports:

```python
from visionflow.components.preprocessing import AutoAdjustContrast, AutoOrient
```

### Preprocessing Test Coverage

The preprocessing suite covers both standard image transforms and important error and alternate-method branches.

- [tests/test_preprocessing_components.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/tests/test_preprocessing_components.py:1) covers:
- `AutoOrient` with EXIF-style orientation handling and explicit rotation and flip behavior
- `AutoAdjustContrast` across adaptive equalization, histogram equalization, contrast stretching, and invalid-method rejection
- geometry utilities such as `Resize`, `LetterboxResize`, `CenterCrop`, `PadToSquare`, `PerspectiveCorrection`, `Deskew`, `AutoCrop`, `FaceAlign`, `ObjectCrop`, `BoundingBoxClamp`, `BoundingBoxNormalize`, and `MaskResize`
- intensity and color utilities such as `Normalize`, `Standardize`, `RescalePixels`, `ConvertColorSpace`, `CLAHE`, `HistogramEqualization`, `GammaCorrection`, `WhiteBalance`, `Denoise`, `Sharpen`, `Deblur`, `Threshold`, `AdaptiveThreshold`, `EdgeDetection`, and `ContourExtraction`
- quality and validation utilities such as `ImageQualityCheck`, `BlurDetection`, `BrightnessCheck`, `DuplicateImageCheck`, `CorruptImageCheck`, `AspectRatioFilter`, `MinSizeFilter`, and `MaxSizeFilter`
- segmentation behavior in `RemoveBackground`, including mask retention and unsupported-mode validation

Run the preprocessing suite with:

```bash
poetry run pytest tests/test_preprocessing_components.py
```

### Augmentation Components

- `Flip(horizontal=False, vertical=False)`
- `Rotate90(k=1)`
- `Crop(x=0, y=0, width=None, height=None, clamp=True)`
- `Rotation(angle=0.0, scale=1.0, expand=False, border_mode="constant", border_value=0)`
- `Shear(shear_x=0.0, shear_y=0.0, border_mode="constant", border_value=0)`
- `Greyscale(keep_channels=True)`
- `Hue(delta=0)`
- `Saturation(scale=1.0)`
- `Brightness(beta=0)`
- `Exposure(gamma=1.0)`
- `Blur(kernel_size=5, sigma_x=0.0)`
- `Noise(mode="gaussian", mean=0.0, stddev=10.0, amount=0.02, salt_vs_pepper=0.5)`
- `Cutout(x=0, y=0, width=32, height=32, fill_value=(0, 0, 0))`
- `Mosaic(output_size=None, mosaic_images=None)`
- `MotionBlur(kernel_size=9, angle=0.0)`
- `CameraGain(gain=1.0, black_level=0.0)`

Implementation imports:

```python
from visionflow.components.augmentations import (
    Blur,
    Brightness,
    CameraGain,
    Crop,
    Cutout,
    Exposure,
    Flip,
    Greyscale,
    Hue,
    Mosaic,
    MotionBlur,
    Noise,
    Rotate90,
    Rotation,
    Saturation,
    Shear,
)
```

### Example

```python
from visionflow.components.augmentations import Blur, Brightness, Crop, Flip, MotionBlur
from visionflow.components.preprocessing import AutoAdjustContrast, AutoOrient

frame = image  # NumPy array

frame = AutoOrient(rotation=90).run(frame)
frame = AutoAdjustContrast(method="adaptive_equalization", clip_limit=2.5).run(frame)
frame = Flip(horizontal=True).run(frame)
frame = Crop(x=20, y=20, width=200, height=200).run(frame)
frame = Brightness(beta=12).run(frame)
frame = Blur(kernel_size=7, sigma_x=1.2).run(frame)
frame = MotionBlur(kernel_size=11, angle=25).run(frame)
```

### Batch Example

```python
from visionflow.components.augmentations import Flip

augmenter = Flip(horizontal=True)
batch_output = augmenter.run([image_1, image_2, image_3])
```

## Advanced Augmentation Components

The augmentation package also includes a broader training-oriented library for geometric transforms, lighting and weather simulation, blur and compression artifacts, dropout and noise injection, and multi-image composition.

Recommended import paths:

- use `from visionflow.components.augmentations import ...` for augmentation-specific classes
- use `from visionflow.components.preprocessing import ...` for preprocessing classes
- use top-level `from visionflow import ...` for commonly used stable exports
- if you need augmentation sharpening specifically, use `from visionflow import AugmentationSharpen` or import `Sharpen` from `visionflow.components.augmentations`

### Geometric And Spatial Augmentations

- `RandomResize(min_width, max_width, min_height=None, max_height=None, interpolation="linear")`
- `RandomScale(min_scale=0.8, max_scale=1.2, interpolation="linear")`
- `RandomCrop(crop_width, crop_height)`
- `RandomResizedCrop(output_width, output_height, scale_min=0.08, scale_max=1.0, ratio_min=0.75, ratio_max=1.3333)`
- `RandomPadding(max_top=10, max_bottom=10, max_left=10, max_right=10, pad_value=(0, 0, 0))`
- `Translate(shift_x=0, shift_y=0, border_mode="constant", border_value=0)`
- `AffineTransform(angle=0.0, scale=1.0, translate_x=0.0, translate_y=0.0, shear_x=0.0, shear_y=0.0, border_mode="constant", border_value=0)`
- `PerspectiveTransform(distortion_scale=0.15, border_mode="constant", border_value=0)`
- `ElasticTransform(alpha=1.0, sigma=10.0, alpha_affine=0.0, border_mode="reflect_101", border_value=0)`
- `GridDistortion(num_steps=5, distort_limit=0.3, border_mode="reflect_101", border_value=0)`
- `OpticalDistortion(k=0.0, dx=0.0, dy=0.0, border_mode="constant", border_value=0)`

Where these are commonly used:

- object detection and segmentation robustness
- OCR and document pipelines that need viewpoint tolerance
- classification pipelines that should generalize across framing, zoom, and camera placement

### Lighting, Color, And Weather Augmentations

- `RandomShadow(shadow_dimension=0.5, intensity=0.5)`
- `RandomSunFlare(center=None, radius=20, intensity=0.4)`
- `RandomFog(alpha=0.35)`
- `RandomRain(drops=40, drop_length=12, intensity=0.25)`
- `RandomSnow(intensity=0.1)`
- `RandomGamma(min_gamma=0.7, max_gamma=1.5)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=10)`
- `ChannelShuffle()`
- `RGBShift(r_shift=0, g_shift=0, b_shift=0)`
- `HSVShift(hue_shift=0, sat_shift=0, val_shift=0)`
- `ToSepia()`
- `InvertImage()`
- `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)`

Where these are commonly used:

- outdoor vision models that must tolerate time-of-day changes
- retail and industrial inspection with variable lighting
- domain generalization when training from limited datasets

### Blur, Compression, And Texture Augmentations

- `Posterize(bits=4)`
- `Solarize(threshold=128)`
- `Equalize()`
- `Emboss(strength=1.0)`
- `Sharpen(amount=1.0)` from `visionflow.components.augmentations`
- `GaussianBlur(kernel_size=5, sigma_x=0.0)`
- `MedianBlur(kernel_size=5)`
- `GlassBlur(sigma=0.7, max_delta=2, iterations=1)`
- `DefocusBlur(radius=5)`
- `ZoomBlur(zoom_factor=1.2, steps=5)`
- `CompressionArtifacts(quality=40)`
- `JPEGCompression(quality=50)`
- `Downscale(scale=0.5, interpolation="area")`
- `Superpixel(region_size=10, ruler=10.0)`

Where these are commonly used:

- mobile camera pipelines
- CCTV and low-quality ingest simulation
- robustness testing for blurred or compressed inputs

### Noise And Dropout Augmentations

- `ISONoise(color_shift=0.01, intensity=0.5)`
- `MultiplicativeNoise(multiplier_min=0.9, multiplier_max=1.1)`
- `SaltPepperNoise(amount=0.02, salt_vs_pepper=0.5)`
- `CoarseDropout(holes=8, max_height=8, max_width=8, fill_value=0)`
- `GridDropout(ratio=0.5, unit_size=8, fill_value=0)`
- `RandomErasing(scale=(0.02, 0.2), fill_value=0)`
- `PixelDropout(dropout_prob=0.01, fill_value=0)`
- `MaskDropout(dropout_prob=0.1)`

Where these are commonly used:

- occlusion tolerance for detection and classification
- segmentation robustness against missing mask regions
- sensor noise simulation for edge devices and embedded cameras

### Multi-Image And Annotation-Aware Augmentations

- `MixUp(alpha=0.5)`
- `CutMix(alpha=0.5)`
- `CopyPaste(x=0, y=0)`
- `RandomOcclusion(max_width=20, max_height=20, fill_value=0)`
- `ObjectPaste(x=0, y=0)`
- `BoundingBoxJitter(x_jitter=0.05, y_jitter=0.05, size_jitter=0.1)`
- `Mosaic9(mosaic_images=None, output_size=None)`

Where these are commonly used:

- detection training with bounding boxes
- segmentation datasets with masks and pasted objects
- long-tail balancing and synthetic data expansion

### Programmatic Example

```python
from visionflow.components.augmentations import (
    ColorJitter,
    GaussianBlur,
    JPEGCompression,
    RandomResizedCrop,
)

augmenters = [
    RandomResizedCrop(
        output_width=256,
        output_height=256,
        scale_min=0.6,
        scale_max=1.0,
    ),
    ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=8),
    GaussianBlur(kernel_size=5, sigma_x=1.0),
    JPEGCompression(quality=45),
]

augmented = image
for augmenter in augmenters:
    augmented = augmenter.run(augmented)
```

### Test Coverage

Augmentation coverage is split so the suite stays easier to maintain:

- [tests/test_preprocessing_components.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/tests/test_preprocessing_components.py:1) covers preprocessing components and their method-specific branches
- [tests/test_basic_augmentations.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/tests/test_basic_augmentations.py:1) covers the previously missing baseline augmentation classes and the profile loader used by `main.py`
- [tests/test_advanced_augmentations.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/tests/test_advanced_augmentations.py:1) covers the larger geometric, weather, blur, noise, dropout, and composition augmentations

Run the relevant test groups with:

```bash
poetry run pytest tests/test_basic_augmentations.py
poetry run pytest tests/test_advanced_augmentations.py
poetry run pytest tests/test_preprocessing_components.py
```

## Running The Webcam Application

The repository includes a runnable webcam application in [main.py](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/main.py:1).

### Basic Run

```bash
poetry run python main.py
```

Because the application is packaged as a library CLI, you can also run it as:

```bash
poetry run ai-vision-flow
```

or:

```bash
poetry run python -m visionflow
```

You can also run it with standard Python if your environment already has the dependencies installed:

```bash
python main.py
```

### Runtime Controls

Once the OpenCV preview window is open and focused, the following hotkeys are available:

- `p`: capture a single processed frame to `output/captures`
- `b`: capture a burst of processed frames to `output/captures`
- `r`: start or stop video recording to `output/videos`
- `d`: save a dataset sample to `output/dataset/<label>`
- `e`: export grayscale and edge images to `output/exports`
- `o`: save the configured ROI crop to `output/captures`
- `q`: quit the application

Important behavior note:

- the hotkeys above are always active while the app is running
- flags such as `--enhance`, `--resize`, `--motion`, `--annotate`, `--timelapse`, and `--dataset` control which processing components are active in the pipeline

## `main.py` Command-Line Arguments

### Pipeline Activation Flags

These arguments enable processing components inside the frame pipeline.

- `--auto-orient`
  Enables `AutoOrient`.
  Useful companion arguments:
  - `--auto-orient-rotation 90|180|270`
  - `--auto-orient-flip-horizontal`
  - `--auto-orient-flip-vertical`
  - `--auto-orient-exif-key <str>`
  - `--no-auto-orient-exif`

- `--auto-adjust-contrast`
  Enables `AutoAdjustContrast`.
  Useful companion arguments:
  - `--contrast-method adaptive_equalization|histogram_equalization|contrast_stretching`
  - `--clip-limit <float>`
  - `--tile-grid-width <int>`
  - `--tile-grid-height <int>`
  - `--lower-percentile <float>`
  - `--upper-percentile <float>`
  - `--output-min <int>`
  - `--output-max <int>`

- `--enhance`
  Enables `FrameEnhancer`. Use together with `--brightness`, `--contrast`, `--sharpen`, `--denoise`, and `--grayscale`.

- `--resize`
  Enables `FrameResizer`. Use together with `--width`, `--height`, and optionally `--keep-aspect`.

- `--motion`
  Enables `MotionDetector`. Use together with `--motion-area`.

- `--annotate`
  Enables `FrameAnnotator`. Adds the default on-screen text annotation in the preview.

- `--augmentation-config <path>`
  Loads a JSON augmentation profile and appends those components to the runtime pipeline.
  This is the recommended way to use the advanced augmentation classes from `main.py`.

Example:

```bash
poetry run python main.py --augmentation-config examples/augmentation_profile.json
```

Profile format:

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
  }
]
```

Each entry supports:

- `name`: class name from `visionflow.components.augmentations`
- `params`: constructor arguments for that augmentation
- `module`: optional fully qualified module path if you want to load a custom component outside the default augmentation package

- `--timelapse`
  Enables `TimeLapseCapture`. Frames are periodically saved to the configured time-lapse directory.

- `--dataset`
  Enables the dataset collector component in the pipeline. This is required for `d` key dataset persistence to use the collector-based implementation.

### Augmentation Activation Flags

These arguments enable augmentation components inside the pipeline.

- `--flip-horizontal`
  Enables horizontal flipping.

- `--flip-vertical`
  Enables vertical flipping.

- `--rotate90-k <int>`
  Applies 90-degree rotations in multiples of `k`.
  Example: `--rotate90-k 1` rotates 90 degrees clockwise-equivalent through `np.rot90`.

- `--crop`
  Enables the crop augmentation.
  Companion arguments:
  - `--crop-x <int>`
  - `--crop-y <int>`
  - `--crop-width <int>`
  - `--crop-height <int>`
  - `--no-crop-clamp`

- `--rotation-angle <float>`
  Enables arbitrary rotation when non-zero.
  Companion arguments:
  - `--rotation-scale <float>`
  - `--rotation-expand`
  - `--rotation-border-mode constant|replicate|reflect|reflect_101|wrap`
  - `--rotation-border-value <int>`

- `--shear-x <float>`
- `--shear-y <float>`
  Enable shear augmentation when either value is non-zero.
  Companion arguments:
  - `--shear-border-mode constant|replicate|reflect|reflect_101|wrap`
  - `--shear-border-value <int>`

- `--augment-greyscale`
  Enables the `Greyscale` augmentation.
  Companion argument:
  - `--greyscale-single-channel`

- `--hue-delta <int>`
  Enables hue shifting when non-zero.

- `--saturation-scale <float>`
  Enables saturation scaling when not `1.0`.

- `--brightness-beta <float>`
  Enables brightness shifting when non-zero.

- `--exposure-gamma <float>`
  Enables gamma/exposure adjustment when not `1.0`.

- `--blur`
  Enables Gaussian blur.
  Companion arguments:
  - `--blur-kernel-size <int>`
  - `--blur-sigma-x <float>`

- `--noise`
  Enables noise augmentation.
  Companion arguments:
  - `--noise-mode gaussian|salt_pepper`
  - `--noise-mean <float>`
  - `--noise-stddev <float>`
  - `--noise-amount <float>`
  - `--salt-vs-pepper <float>`

- `--cutout`
  Enables cutout masking.
  Companion arguments:
  - `--cutout-x <int>`
  - `--cutout-y <int>`
  - `--cutout-width <int>`
  - `--cutout-height <int>`
  - `--cutout-fill-value <b> <g> <r>`

- `--mosaic`
  Enables mosaic generation.
  Companion arguments:
  - `--mosaic-image <path>` and repeat this flag up to three times for external images
  - `--mosaic-output-width <int>`
  - `--mosaic-output-height <int>`

- `--motion-blur`
  Enables motion blur.
  Companion arguments:
  - `--motion-blur-kernel-size <int>`
  - `--motion-blur-angle <float>`

- `--camera-gain <float>`
  Enables camera gain amplification when not `1.0`.
  Companion argument:
  - `--camera-black-level <float>`

### Capture And Output Flags

These flags are accepted for workflow intent and output customization.

- `--picture`
  Declares picture-capture intent. The `p` hotkey works regardless, but this flag is useful for explicit run profiles and future workflow extensions.

- `--burst`
  Declares burst-capture intent. The `b` hotkey works regardless, but this flag is useful for explicit run profiles and future workflow extensions.

- `--roi`
  Declares ROI workflow usage. The ROI overlay is currently shown in the preview and the `o` hotkey saves the configured crop.

- `--video`
  Declares video workflow usage. Recording can still be toggled with `r`.

- `--export`
  Declares export workflow usage. Exporting can still be triggered with `e`.

### Camera And Frame Geometry Arguments

- `--camera-id <int>`
  Selects the webcam device index.
  Default: `0`

- `--width <int>`
  Target output width used by the resize component.
  Default: `640`

- `--height <int>`
  Target output height used by the resize component.
  Default: `480`

- `--keep-aspect`
  Preserves aspect ratio during resizing and pads to the target frame size.

### Enhancement Arguments

These are only meaningful when `--enhance` is enabled.

- `--brightness <int>`
  Adjusts brightness using OpenCV scale conversion.
  Default: `0`

- `--contrast <float>`
  Adjusts contrast using OpenCV scale conversion.
  Default: `1.0`

- `--sharpen`
  Enables a sharpen filter.

- `--denoise`
  Enables colored denoising.

- `--grayscale`
  Converts frames to grayscale while keeping a 3-channel output for pipeline compatibility.

### Motion Detection Arguments

These are used by `MotionDetector` when `--motion` is enabled.

- `--motion-area <int>`
  Minimum contour area for motion detection.
  Default: `800`

### Burst Capture Arguments

- `--burst-count <int>`
  Number of frames saved when the `b` hotkey is pressed.
  Default: `5`

### Time-Lapse Arguments

These are used when `--timelapse` is enabled.

- `--timelapse-interval <int>`
  Number of seconds between time-lapse captures.
  Default: `5`

- `--timelapse-dir <path>`
  Subdirectory under the output root where time-lapse images are stored.
  Default: `timelapse`

### Recording Arguments

- `--fps <int>`
  Frames per second for saved video recordings.
  Default: `20`

- `--video-dir <path>`
  Subdirectory under the output root where recordings are stored.
  Default: `videos`

### Dataset Arguments

These are most useful when `--dataset` is enabled and when using the `d` hotkey.

- `--label <str>`
  Label name used when saving dataset samples.
  Default: `unknown`

- `--dataset-dir <path>`
  Subdirectory under the output root where dataset samples are stored.
  Default: `dataset`

### ROI Arguments

These values define the ROI rectangle used for preview overlay and `o` key crop capture.

- `--roi-x <int>`
  ROI top-left x coordinate.
  Default: `100`

- `--roi-y <int>`
  ROI top-left y coordinate.
  Default: `100`

- `--roi-w <int>`
  ROI width.
  Default: `300`

- `--roi-h <int>`
  ROI height.
  Default: `300`

### Output Directory Arguments

All outputs are written beneath a single output root.

- `--output-root <path>`
  Root output directory.
  Default: `output`

- `--output-dir <path>`
  Capture image subdirectory beneath `output-root`.
  Default: `captures`

- `--export-dir <path>`
  Export image subdirectory beneath `output-root`.
  Default: `exports`

## Practical Examples

### 1. Run A Basic Webcam Session

```bash
poetry run python main.py
```

### 2. Run With Enhancement And Resizing

```bash
poetry run python main.py \
  --enhance \
  --brightness 12 \
  --contrast 1.15 \
  --sharpen \
  --resize \
  --width 1280 \
  --height 720 \
  --keep-aspect
```

### 3. Run Auto-Orientation And Contrast Preprocessing

```bash
poetry run python main.py \
  --auto-orient \
  --auto-orient-rotation 90 \
  --auto-adjust-contrast \
  --contrast-method adaptive_equalization \
  --clip-limit 2.5 \
  --tile-grid-width 8 \
  --tile-grid-height 8
```

### 4. Run Augmentations From The CLI

```bash
poetry run python main.py \
  --flip-horizontal \
  --rotation-angle 12 \
  --rotation-expand \
  --blur \
  --blur-kernel-size 7 \
  --noise \
  --noise-mode gaussian \
  --noise-stddev 8
```

### 5. Run Motion Detection With Annotation

```bash
poetry run python main.py \
  --motion \
  --motion-area 1200 \
  --annotate
```

### 6. Run Time-Lapse Capture

```bash
poetry run python main.py \
  --timelapse \
  --timelapse-interval 10
```

### 7. Prepare For Dataset Collection

```bash
poetry run python main.py \
  --dataset \
  --label forklift \
  --enhance \
  --resize
```

Then press `d` while the window is focused to save labeled samples.

### 8. Customize Output Locations

```bash
poetry run python main.py \
  --output-root runs/session-01 \
  --output-dir stills \
  --video-dir recordings \
  --dataset-dir training-data \
  --export-dir processed
```

### 9. Create A Mosaic From Supplemental Images

```bash
poetry run python main.py \
  --mosaic \
  --mosaic-image ./assets/extra_1.jpg \
  --mosaic-image ./assets/extra_2.jpg \
  --mosaic-output-width 1280 \
  --mosaic-output-height 720
```

### 10. Increase Burst Size

```bash
poetry run python main.py --burst-count 12
```

Then press `b` in the preview window.

## Output Structure

By default, the application creates the following structure:

```text
output/
├── captures/
├── dataset/
├── exports/
├── timelapse/
└── videos/
```

## Testing

```bash
poetry install --with dev
poetry run pytest
```

## Build

```bash
python -m pip install --upgrade build
python -m build
```

The generated wheel and source distribution will be written to `dist/`.

## Publish To PyPI

See `PUBLISHING.md` in the repository for the release checklist and upload commands.

## Package Notes

- `visionflow` uses lazy imports so optional heavyweight integrations are not imported until you explicitly use them.
- TensorFlow and Darknet-based components may require additional external runtimes, model files, or local toolchains beyond the base package installation.
- The webcam application is intended for local experimentation and interactive operation, while the package itself is suitable for reuse in other Python applications.
