from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


def _retinex_single(channel: np.ndarray, sigma: float) -> np.ndarray:
    log_img = np.log1p(channel.astype(np.float64))
    blur = cv2.GaussianBlur(channel.astype(np.float64), (0, 0), sigma)
    log_blur = np.log1p(blur)
    retinex = log_img - log_blur
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-8)
    return (retinex * 255).astype(np.uint8)


def _retinex_multiscale(frame: np.ndarray, sigmas: tuple = (15, 80, 250)) -> np.ndarray:
    result = np.zeros_like(frame, dtype=np.float64)
    for sigma in sigmas:
        for i in range(3):
            result[:, :, i] += _retinex_single(frame[:, :, i], sigma).astype(np.float64)
    result /= len(sigmas)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


class LowLightEnhancer(AIVisionComponent):
    """Enhances low-light images using CLAHE, Retinex, gamma correction, or ONNX.

    Args:
        method: "clahe", "gamma", "histogram_stretch", "retinex_singlescale",
                "retinex_multiscale", "zero_dce_approx", or "onnx".
        clip_limit: CLAHE clip limit.
        tile_size: CLAHE tile grid size.
        gamma: Gamma correction value (< 1 brightens).
        model_path: Path to ONNX low-light model.
    """

    def __init__(
        self,
        method: str = "clahe",
        clip_limit: float = 2.0,
        tile_size: int = 8,
        gamma: float = 1.5,
        model_path: str | None = None,
    ):
        super().__init__()
        self.method = method
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.gamma = gamma
        self.model_path = model_path
        self._clahe = None
        self._session = None

    def setup(self, config: dict):
        self._clahe = cv2.createCLAHE(
            clipLimit=float(config.get("clip_limit", self.clip_limit)),
            tileGridSize=(int(config.get("tile_size", self.tile_size)),) * 2,
        )
        if self.method == "onnx" and self.model_path:
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
            except ImportError as e:
                raise ImportError("Install with: pip install onnxruntime") from e
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        method = config.get("method", self.method)
        gamma = float(config.get("gamma", self.gamma))

        if method == "clahe":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a, b = cv2.split(lab)
            l_enhanced = self._clahe.apply(l_ch)
            result = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

        elif method == "gamma":
            table = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            result = cv2.LUT(frame, table)

        elif method == "histogram_stretch":
            result = np.zeros_like(frame)
            for i in range(3):
                c = frame[:, :, i].astype(np.float64)
                lo, hi = np.percentile(c, 2), np.percentile(c, 98)
                result[:, :, i] = np.clip(
                    (c - lo) / (hi - lo + 1e-8) * 255, 0, 255
                ).astype(np.uint8)

        elif method == "retinex_singlescale":
            result = np.zeros_like(frame)
            for i in range(3):
                result[:, :, i] = _retinex_single(frame[:, :, i], sigma=80.0)

        elif method == "retinex_multiscale":
            result = _retinex_multiscale(frame)

        elif method == "zero_dce_approx":
            img = frame.astype(np.float32) / 255.0
            mean_l = img.mean()
            alpha = max(0.0, 0.5 - mean_l)
            enhanced = img + alpha * (img - img * img)
            result = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        elif method == "onnx" and self._session:
            inp = self._session.get_inputs()[0]
            ih, iw = inp.shape[2], inp.shape[3]
            resized = cv2.resize(frame, (iw, ih))
            tensor = (
                cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis]
                / 255.0
            )
            tensor = np.transpose(tensor, (0, 3, 1, 2))
            out = self._session.run(None, {inp.name: tensor})[0]
            out = np.clip(out[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            result = cv2.resize(
                cv2.cvtColor(out, cv2.COLOR_RGB2BGR), (frame.shape[1], frame.shape[0])
            )
        else:
            result = frame

        payload = replace_frame(data, result)
        if isinstance(payload, dict):
            payload["enhancement_method"] = method
        return payload
