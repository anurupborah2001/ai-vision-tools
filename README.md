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
- [Component Index](#component-index)
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
