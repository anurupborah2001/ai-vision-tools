from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


class ONNXModel(AIVisionComponent):
    """Generic ONNX model inference component.

    Args:
        model_path: Path to .onnx file.
        input_name: Input tensor name (None = auto-detect from session).
        output_names: List of output tensor names (None = all outputs).
        providers: ONNX Runtime execution providers.
    """

    def __init__(self, model_path: str, input_name: str | None = None,
                 output_names: list[str] | None = None, providers: list[str] | None = None):
        super().__init__()
        self.model_path = model_path
        self.input_name = input_name
        self.output_names = output_names
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = None
        self._input_name: str = ""
        self._input_shape: tuple = ()
        self._output_names: list[str] = []

    def setup(self, config: dict):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Install with: pip install onnxruntime  # or onnxruntime-gpu")

        valid_providers = [p for p in self.providers if p in ort.get_available_providers()]
        if not valid_providers:
            valid_providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(self.model_path, providers=valid_providers)
        inp = self._session.get_inputs()[0]
        self._input_name = self.input_name or inp.name
        self._input_shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)
        self._output_names = self.output_names or [o.name for o in self._session.get_outputs()]
        super().setup(config)

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @property
    def output_shape(self) -> list[tuple]:
        return [tuple(o.shape) for o in self._session.get_outputs()]

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        _, _, h, w = self._input_shape if len(self._input_shape) == 4 else (1, 3, 640, 640)
        resized = cv2.resize(frame, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))[np.newaxis]

    def _execute(self, data, config):
        frame = extract_frame(data)
        tensor = self.preprocess_frame(frame)
        outputs = self._session.run(self._output_names, {self._input_name: tensor})
        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["model_output"] = outputs[0] if len(outputs) == 1 else outputs
        payload["model_name"] = Path(self.model_path).stem
        return payload
