from __future__ import annotations

import json
from pathlib import Path


class ModelRegistry:
    """Central registry for AI model metadata and instantiation.

    Usage:
        reg = ModelRegistry.instance()
        reg.register("yolov8n", "path/to/model.pt", "torch")
        model = reg.load("yolov8n")
    """

    _singleton: ModelRegistry | None = None
    _CACHE_PATH = Path.home() / ".cache" / "ai_vision_tool" / "model_registry.json"

    def __init__(self):
        self._registry: dict[str, dict] = {}
        self._load_cache()

    @classmethod
    def instance(cls) -> ModelRegistry:
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def _load_cache(self):
        if self._CACHE_PATH.exists():
            try:
                self._registry = json.loads(self._CACHE_PATH.read_text())
            except Exception:
                self._registry = {}

    def _save_cache(self):
        self._CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._CACHE_PATH.write_text(json.dumps(self._registry, indent=2))

    def register(
        self, name: str, path_or_url: str, model_type: str, metadata: dict | None = None
    ) -> None:
        self._registry[name] = {
            "name": name,
            "path_or_url": path_or_url,
            "model_type": model_type,
            "metadata": metadata or {},
        }
        self._save_cache()

    def get_info(self, name: str) -> dict:
        if name not in self._registry:
            raise KeyError(
                f"Model {name!r} not registered. Available: {list(self._registry)}"
            )
        return dict(self._registry[name])

    def list(self) -> list[dict]:
        return list(self._registry.values())

    def load(self, name: str, **kwargs):
        info = self.get_info(name)
        mtype = info["model_type"]
        path = info["path_or_url"]
        if mtype == "onnx":
            from ai_vision_tool.models.onnx_model import ONNXModel

            return ONNXModel(model_path=path, **kwargs)
        if mtype == "torch":
            from ai_vision_tool.models.torch_model import TorchModel

            return TorchModel(model_path=path, **kwargs)
        if mtype == "tflite":
            from ai_vision_tool.models.tflite_model import TFLiteModel

            return TFLiteModel(model_path=path, **kwargs)
        raise ValueError(f"Unknown model_type {mtype!r}")

    @classmethod
    def from_huggingface(
        cls, repo_id: str, filename: str, model_type: str
    ) -> ModelRegistry:
        reg = cls.instance()
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        name = f"{repo_id.replace('/', '__')}_{filename}"
        reg.register(name, url, model_type, {"repo_id": repo_id, "filename": filename})
        return reg
