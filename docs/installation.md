# Installation

## Requirements

- Python **3.10, 3.11, or 3.12**
- `numpy >= 1.26`
- `opencv-python >= 4.8`
- `pyyaml >= 6.0`

---

## Base Install

The base install has no heavy ML dependencies. Only `numpy`, `opencv-python`, and `pyyaml` are required.

```bash
pip install ai-vision-tool
```

---

## Optional Extras

Install extras for specific capabilities:

| Extra | Command | Unlocks |
|-------|---------|---------|
| `onnx` | `pip install "ai-vision-tool[onnx]"` | ONNX model inference via `onnxruntime` |
| `torch` | `pip install "ai-vision-tool[torch]"` | PyTorch and TorchScript model backends |
| `tflite` | `pip install "ai-vision-tool[tflite]"` | TensorFlow Lite model inference |
| `detection` | `pip install "ai-vision-tool[detection]"` | YOLO (ultralytics) + MediaPipe detectors |
| `segmentation` | `pip install "ai-vision-tool[segmentation]"` | YOLO-seg + Segment Anything Model |
| `tracking` | `pip install "ai-vision-tool[tracking]"` | ONNX-backed ReID for DeepSORT |
| `cloud` | `pip install "ai-vision-tool[cloud]"` | S3 and GCS dataset sources |
| `websocket` | `pip install "ai-vision-tool[websocket]"` | WebSocket frame sink/source |
| `kafka` | `pip install "ai-vision-tool[kafka]"` | Kafka frame source/sink |
| `streaming` | `pip install "ai-vision-tool[streaming]"` | WebSocket + Kafka combined |
| `api` | `pip install "ai-vision-tool[api]"` | FastAPI + Uvicorn server |
| `all` | `pip install "ai-vision-tool[all]"` | Every optional dependency |

### Combining extras

```bash
pip install "ai-vision-tool[onnx,detection,cloud]"
```

---

## Development Install

```bash
# Clone
git clone https://github.com/anuborah/ai-vision-tools.git
cd ai-vision-tools

# Install with uv (recommended)
uv sync --dev

# Or with pip in editable mode
pip install -e ".[all]"
pip install black ruff isort pytest pre-commit
```

### Pre-commit hooks

```bash
pre-commit install
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg
```

---

## Verify Installation

```python
import ai_vision_tool
print(ai_vision_tool.__version__)  # 0.2.0

# Check a component loads
from ai_vision_tool import Resize
r = Resize(640, 480)
print(r)
```

---

## Hardware Acceleration

AI Vision Tool auto-detects the best available device:

| Environment | Device used |
|-------------|-------------|
| NVIDIA GPU + CUDA | `cuda:0` |
| Apple Silicon | `mps` |
| Everything else | `cpu` |

Override programmatically:

```python
from ai_vision_tool.core.device import Device

# Force CPU
device = Device(force="cpu")
```
