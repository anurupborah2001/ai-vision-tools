from __future__ import annotations

import time

import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite  # noqa: F401
except ImportError:
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "TFLite backend requires: pip install ai-vision-tool[tflite]"
        ) from exc

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class TFLiteModel(AIVisionComponent):
    """TFLite model inference component. Supports tflite-runtime and tensorflow.

    Args:
        model_path: Path to .tflite file.
        num_threads: Number of CPU inference threads.
    """

    def __init__(self, model_path: str, num_threads: int = 4):
        super().__init__()
        self.model_path = model_path
        self.num_threads = num_threads
        self._interpreter = None
        self._input_details = []
        self._output_details = []

    def setup(self, config: dict):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError as e:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                raise ImportError("Install with: pip install tflite-runtime") from e
        self._interpreter = Interpreter(
            model_path=self.model_path, num_threads=self.num_threads
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        inp = self._input_details[0]
        h, w = inp["shape"][1], inp["shape"][2]
        resized = cv2.resize(frame, (w, h))
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        if inp["dtype"] == np.float32:
            tensor = resized.astype(np.float32)[np.newaxis] / 255.0
        else:
            tensor = resized.astype(np.uint8)[np.newaxis]

        self._interpreter.set_tensor(inp["index"], tensor)
        t0 = time.perf_counter()
        self._interpreter.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        outputs = [
            self._interpreter.get_tensor(d["index"]) for d in self._output_details
        ]
        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["model_output"] = outputs[0] if len(outputs) == 1 else outputs
        payload["inference_time_ms"] = round(elapsed_ms, 3)
        return payload
