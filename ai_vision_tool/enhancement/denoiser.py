from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


class Denoiser(AIVisionComponent):
    """Image denoising with multiple methods from fast bilateral to DnCNN.

    Args:
        method: "nlmeans", "bilateral", "gaussian", "median", "dncnn", or "auto".
        strength: Denoising strength (mapped to method-specific params).
        model_path: Path to DnCNN ONNX model (dncnn method only).
    """

    def __init__(self, method: str = "nlmeans", strength: float = 10.0, model_path: str | None = None):
        super().__init__()
        self.method = method
        self.strength = strength
        self.model_path = model_path
        self._session = None
        self._method_used = method

    def setup(self, config: dict):
        m = config.get("method", self.method)
        if m == "dncnn":
            if self.model_path:
                try:
                    import onnxruntime as ort
                    providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                                 if p in ort.get_available_providers()]
                    self._session = ort.InferenceSession(self.model_path, providers=providers)
                    self._method_used = "dncnn"
                except ImportError:
                    print("[Denoiser] onnxruntime not available, falling back to nlmeans")
                    self._method_used = "nlmeans"
            else:
                self._method_used = "nlmeans"
        else:
            self._method_used = m if m != "auto" else "nlmeans"
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        strength = float(config.get("strength", self.strength))
        method = config.get("method", self._method_used)

        if method == "nlmeans":
            h_param = int(strength)
            if frame.ndim == 3:
                result = cv2.fastNlMeansDenoisingColored(frame, None, h_param, h_param, 7, 21)
            else:
                result = cv2.fastNlMeansDenoising(frame, None, h_param, 7, 21)
        elif method == "bilateral":
            d = max(1, int(strength / 2)) * 2 + 1
            sigma = strength * 7
            result = cv2.bilateralFilter(frame, d, sigma, sigma)
        elif method == "gaussian":
            ks = max(1, int(strength / 3)) * 2 + 1
            result = cv2.GaussianBlur(frame, (ks, ks), 0)
        elif method == "median":
            ks = max(1, int(strength / 5)) * 2 + 1
            result = cv2.medianBlur(frame, ks)
        elif method == "dncnn" and self._session:
            inp = self._session.get_inputs()[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            tensor = gray[np.newaxis, np.newaxis]
            noise = self._session.run(None, {inp.name: tensor})[0]
            denoised = np.clip((gray - noise[0, 0]) * 255, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        else:
            result = cv2.fastNlMeansDenoisingColored(frame, None, int(strength), int(strength), 7, 21)

        payload = replace_frame(data, result)
        if isinstance(payload, dict):
            payload["denoise_method"] = method
        return payload
