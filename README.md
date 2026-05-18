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
