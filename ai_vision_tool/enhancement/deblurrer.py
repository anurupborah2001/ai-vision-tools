from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame, replace_frame


def _wiener_deconv(channel: np.ndarray, kernel_size: int, snr: float) -> np.ndarray:
    h, w = channel.shape
    psf = np.zeros((h, w), dtype=np.float64)
    k = kernel_size
    psf[:k, :k] = 1.0 / (k * k)
    psf = np.roll(psf, -k // 2, axis=0)
    psf = np.roll(psf, -k // 2, axis=1)
    PSF = np.fft.fft2(psf)
    IMG = np.fft.fft2(channel.astype(np.float64) / 255.0)
    conj = np.conj(PSF)
    denom = np.abs(PSF) ** 2 + 1.0 / snr
    result = np.real(np.fft.ifft2((conj / denom) * IMG))
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _richardson_lucy(channel: np.ndarray, kernel_size: int, iterations: int = 10) -> np.ndarray:
    img = channel.astype(np.float64) / 255.0 + 1e-8
    psf = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    estimate = img.copy()
    for _ in range(iterations):
        conv = cv2.filter2D(estimate, -1, psf)
        ratio = img / (conv + 1e-8)
        estimate *= cv2.filter2D(ratio, -1, np.flip(psf))
    return np.clip(estimate * 255, 0, 255).astype(np.uint8)


class Deblurrer(AIVisionComponent):
    """Image deblurring using Wiener, Richardson-Lucy, unsharp masking, or ONNX models.

    Args:
        method: "wiener", "laplacian_sharpen", "unsharp_mask", "richardson_lucy", "onnx_nafnet".
        kernel_size: Blur kernel size estimate.
        snr: Signal-to-noise ratio estimate (Wiener method).
        model_path: Path to ONNX NAFNet model (onnx_nafnet method).
    """

    def __init__(self, method: str = "wiener", kernel_size: int = 5,
                 snr: float = 100.0, model_path: str | None = None):
        super().__init__()
        self.method = method
        self.kernel_size = kernel_size
        self.snr = snr
        self.model_path = model_path
        self._session = None

    def setup(self, config: dict):
        if self.method == "onnx_nafnet" and self.model_path:
            try:
                import onnxruntime as ort
                providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if p in ort.get_available_providers()]
                self._session = ort.InferenceSession(self.model_path, providers=providers)
            except ImportError:
                raise ImportError("Install with: pip install onnxruntime")
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        method = config.get("method", self.method)
        ks = int(config.get("kernel_size", self.kernel_size))
        snr = float(config.get("snr", self.snr))
        ks = ks if ks % 2 == 1 else ks + 1

        if method == "wiener":
            channels = cv2.split(frame)
            result = cv2.merge([_wiener_deconv(c, ks, snr) for c in channels])
        elif method == "laplacian_sharpen":
            lap = cv2.Laplacian(frame, cv2.CV_64F)
            result = np.clip(frame.astype(np.float64) - lap, 0, 255).astype(np.uint8)
        elif method == "unsharp_mask":
            blur = cv2.GaussianBlur(frame, (ks, ks), 0)
            result = np.clip(1.5 * frame.astype(np.float64) - 0.5 * blur.astype(np.float64), 0, 255).astype(np.uint8)
        elif method == "richardson_lucy":
            channels = cv2.split(frame)
            result = cv2.merge([_richardson_lucy(c, ks) for c in channels])
        elif method == "onnx_nafnet" and self._session:
            inp = self._session.get_inputs()[0]
            ih, iw = inp.shape[2], inp.shape[3]
            resized = cv2.resize(frame, (iw, ih))
            tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis] / 255.0
            tensor = np.transpose(tensor, (0, 3, 1, 2))
            out = self._session.run(None, {inp.name: tensor})[0]
            out = np.clip(out[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            result = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), (frame.shape[1], frame.shape[0]))
        else:
            result = frame

        payload = replace_frame(data, result)
        if isinstance(payload, dict):
            payload["deblur_method"] = method
        return payload
