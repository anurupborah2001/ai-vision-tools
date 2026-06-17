from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class SuperResolution(AIVisionComponent):
    """Upscales images using OpenCV DNN SuperRes, ONNX ESRGAN, or bicubic fallback.

    Args:
        scale: Upscale factor (2, 3, or 4).
        backend: "auto", "cv2_dnn_superres", "onnx", or "bicubic".
        model_path: Path to model file (required for cv2_dnn_superres/onnx).
    """

    def __init__(
        self, scale: int = 2, backend: str = "auto", model_path: str | None = None
    ):
        super().__init__()
        self.scale = scale
        self.backend = backend
        self.model_path = model_path
        self._sr = None
        self._session = None
        self._backend_used = "bicubic"

    def setup(self, config: dict):
        backend = config.get("backend", self.backend)
        if backend in ("cv2_dnn_superres", "auto") and self.model_path:
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(self.model_path)
                algo = "EDSR" if "edsr" in self.model_path.lower() else "FSRCNN"
                sr.setModel(algo.lower(), self.scale)
                self._sr = sr
                self._backend_used = "cv2_dnn_superres"
                super().setup(config)
                return
            except (cv2.error, AttributeError):
                pass
        if backend == "onnx" and self.model_path:
            try:
                import onnxruntime as ort

                providers = [
                    p
                    for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if p in ort.get_available_providers()
                ]
                self._session = ort.InferenceSession(
                    self.model_path, providers=providers
                )
                self._backend_used = "onnx"
                super().setup(config)
                return
            except ImportError as e:
                raise ImportError("Install with: pip install onnxruntime") from e
        self._backend_used = "bicubic"
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        scale = int(config.get("scale", self.scale))
        h, w = frame.shape[:2]

        if self._backend_used == "cv2_dnn_superres" and self._sr:
            upscaled = self._sr.upsample(frame)
        elif self._backend_used == "onnx" and self._session:
            inp = self._session.get_inputs()[0]
            tensor = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis]
                / 255.0
            )
            tensor = np.transpose(tensor, (0, 3, 1, 2))
            out = self._session.run(None, {inp.name: tensor})[0]
            out = np.clip(out[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            upscaled = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        else:
            upscaled = cv2.resize(
                frame, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
            )

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["frame"] = upscaled
        payload["sr_scale"] = upscaled.shape[1] / w
        payload["sr_backend"] = self._backend_used
        return payload
