from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


class Colorizer(AIVisionComponent):
    """Colorizes grayscale images using DNN, OpenCV colormaps, or ONNX models.

    Args:
        method: "opencv_dnn" (Zhang et al. 2016), "pseudo_color", "thermal", or "onnx".
        model_path: Path to Caffe colorization model prototxt (.prototxt for opencv_dnn).
        pts_path: Path to cluster center npy file (opencv_dnn).
    """

    def __init__(self, method: str = "opencv_dnn", model_path: str | None = None, pts_path: str | None = None):
        super().__init__()
        self.method = method
        self.model_path = model_path
        self.pts_path = pts_path
        self._net = None
        self._session = None
        self._method_used = method

    def setup(self, config: dict):
        if self.method == "opencv_dnn" and self.model_path:
            try:
                self._net = cv2.dnn.readNetFromCaffe(self.model_path.replace(".caffemodel", ".prototxt"),
                                                       self.model_path)
                if self.pts_path:
                    pts = np.load(self.pts_path).transpose().reshape(2, 313, 1, 1).astype(np.float32)
                    self._net.getLayer(self._net.getLayerId("class8_ab")).blobs = [pts]
                    self._net.getLayer(self._net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]
                self._method_used = "opencv_dnn"
            except Exception as e:
                print(f"[Colorizer] Failed to load DNN model: {e}. Falling back to pseudo_color.")
                self._method_used = "pseudo_color"
        elif self.method == "onnx" and self.model_path:
            try:
                import onnxruntime as ort
                providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if p in ort.get_available_providers()]
                self._session = ort.InferenceSession(self.model_path, providers=providers)
                self._method_used = "onnx"
            except ImportError:
                raise ImportError("Install with: pip install onnxruntime")
        else:
            self._method_used = self.method if self.method in ("pseudo_color", "thermal") else "pseudo_color"
        super().setup(config)

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _execute(self, data, config):
        frame = extract_frame(data)
        method = config.get("method", self._method_used)
        is_gray = frame.ndim == 2 or (frame.ndim == 3 and np.allclose(frame[:, :, 0], frame[:, :, 1]))
        gray = self._to_gray(frame)

        if method == "opencv_dnn" and self._net:
            gray_rs = cv2.resize(gray, (224, 224))
            lab = cv2.cvtColor(cv2.cvtColor(gray_rs, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0].astype(np.float32) - 50.0
            blob = cv2.dnn.blobFromImage(L)
            self._net.setInput(blob)
            ab = self._net.forward()[0, :, :, :].transpose(1, 2, 0)
            ab_rs = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
            L_full = cv2.resize(lab[:, :, 0], (frame.shape[1], frame.shape[0])).astype(np.float32)
            lab_full = np.zeros((frame.shape[0], frame.shape[1], 3), np.float32)
            lab_full[:, :, 0] = L_full
            lab_full[:, :, 1:] = ab_rs
            result = np.clip(cv2.cvtColor(lab_full.astype(np.uint8), cv2.COLOR_LAB2BGR), 0, 255).astype(np.uint8)
        elif method == "pseudo_color":
            result = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        elif method == "thermal":
            result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        elif method == "onnx" and self._session:
            inp = self._session.get_inputs()[0]
            ih, iw = inp.shape[2], inp.shape[3]
            gray_rs = cv2.resize(gray, (iw, ih)).astype(np.float32)[np.newaxis, np.newaxis] / 255.0
            out = self._session.run(None, {inp.name: gray_rs})[0]
            out = np.clip(out[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            result = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), (frame.shape[1], frame.shape[0]))
        else:
            result = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)

        payload = replace_frame(data, result)
        if isinstance(payload, dict):
            payload["colorizer_method"] = method
            payload["is_grayscale_input"] = is_gray
        return payload
