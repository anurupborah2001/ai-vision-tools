# Models

The `models` module handles model lifecycle: registry with JSON caching, automatic downloading with SHA-256 verification, benchmarking, and three inference backends (ONNX, TorchScript, TFLite).

```python
from ai_vision_tool.models import <ClassName>
```

---

## Model Registry

`ModelRegistry` maintains a local JSON catalog of downloaded models.

```python
from ai_vision_tool.models import ModelRegistry

registry = ModelRegistry(cache_dir="~/.ai_vision_tool/models")

# Register a local model
registry.register(
    name="yolov8n-det",
    path="/path/to/yolov8n.onnx",
    metadata={"task": "detection", "input_size": [640, 640]},
)

# Load a registered model path
path = registry.load("yolov8n-det")

# Load from HuggingFace hub
path = registry.from_huggingface(
    repo_id="Ultralytics/assets",
    filename="yolov8n.pt",
    cache_dir="~/.ai_vision_tool/models",
)
```

---

## Model Downloader

```python
from ai_vision_tool.models import ModelDownloader

downloader = ModelDownloader(cache_dir="~/.ai_vision_tool/models")

# Download from URL with SHA-256 verification
path = downloader.download(
    url="https://example.com/model.onnx",
    filename="model.onnx",
    sha256="abc123...",
)

# Download from HuggingFace
path = downloader.from_huggingface(repo_id="owner/model", filename="model.onnx")
```

---

## Model Benchmark

Profile inference latency and memory usage. Reports p50/p95/p99 percentiles in an ASCII table.

```python
from ai_vision_tool.models import ModelBenchmark
from ai_vision_tool.models.backends import ONNXModel

model = ONNXModel("model.onnx")

bench = ModelBenchmark(model=model, warmup=10, iterations=100)
bench.run(input_data=sample_image)
bench.report()
```

**Example output:**

```
┌─────────────────────────────────────┐
│  ModelBenchmark — model.onnx        │
├──────────┬────────┬────────┬────────┤
│  Metric  │  p50   │  p95   │  p99   │
├──────────┼────────┼────────┼────────┤
│ Latency  │ 12.3ms │ 14.1ms │ 18.2ms │
│ Memory   │ 142 MB │ 145 MB │ 148 MB │
└──────────┴────────┴────────┴────────┘
```

---

## ONNX Backend

> **Extra required:** `pip install "ai-vision-tool[onnx]"`

```python
from ai_vision_tool.models.backends import ONNXModel

model = ONNXModel(
    model_path="detector.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

import numpy as np
input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)
output = model.run({"images": input_tensor})
# output["model_output"] → numpy array
```

---

## TorchScript Backend

> **Extra required:** `pip install "ai-vision-tool[torch]"`

```python
from ai_vision_tool.models.backends import TorchModel

model = TorchModel(
    model_path="model.pt",           # TorchScript .pt file
    device="cuda",                   # auto-detected if None
    half_precision=True,             # FP16 on GPU
)

output = model.run(input_tensor)
```

---

## TFLite Backend

> **Extra required:** `pip install "ai-vision-tool[tflite]"`

```python
from ai_vision_tool.models.backends import TFLiteModel

model = TFLiteModel("model.tflite")
output = model.run(input_tensor)
```

Falls back to full TensorFlow if `tflite-runtime` is not installed.
