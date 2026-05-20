from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame


class HeatmapRenderer(AIVisionComponent):
    """Generates and overlays spatial heatmaps from detections, anomaly maps, or optical flow.

    Args:
        source: "detections", "anomaly_map", "attention", or "motion".
        colormap: OpenCV colormap constant (default COLORMAP_JET).
        alpha: Overlay blend alpha.
        accumulate: Keep cumulative density (True) or per-frame (False).
        decay: Decay factor applied each frame (accumulate mode only).
    """

    def __init__(self, source: str = "detections", colormap: int = cv2.COLORMAP_JET,
                 alpha: float = 0.5, accumulate: bool = False, decay: float = 0.95):
        super().__init__()
        self.source = source
        self.colormap = colormap
        self.alpha = alpha
        self.accumulate = accumulate
        self.decay = decay
        self._density: np.ndarray | None = None
        self._prev_frame: np.ndarray | None = None

    def _gaussian_blob(self, density: np.ndarray, cx: int, cy: int, sigma: float) -> np.ndarray:
        h, w = density.shape
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        return density + blob

    def _execute(self, data, config):
        frame = extract_frame(data)
        h, w = frame.shape[:2]
        src = config.get("source", self.source)
        alpha = float(config.get("alpha", self.alpha))

        if self._density is None or self._density.shape != (h, w):
            self._density = np.zeros((h, w), dtype=np.float32)

        if self.accumulate:
            self._density *= float(config.get("decay", self.decay))

        if src == "detections" and isinstance(data, dict):
            bboxes = data.get("bboxes", data.get("tracks", []))
            for bb in bboxes:
                cx = int((bb["x1"] + bb["x2"]) / 2)
                cy = int((bb["y1"] + bb["y2"]) / 2)
                area = (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"])
                sigma = max(5.0, float(area) ** 0.5 / 2)
                self._density = self._gaussian_blob(self._density, cx, cy, sigma)

        elif src == "anomaly_map" and isinstance(data, dict):
            amap = data.get("anomaly_map")
            if amap is not None:
                amap_rs = cv2.resize(amap.astype(np.float32), (w, h))
                self._density = amap_rs if not self.accumulate else self._density + amap_rs

        elif src == "attention" and isinstance(data, dict):
            atten = data.get("attention_map")
            if atten is not None:
                self._density = cv2.resize(atten.astype(np.float32), (w, h))

        elif src == "motion":
            if self._prev_frame is not None:
                prev_gray = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                     0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
                self._density = mag.astype(np.float32)
            self._prev_frame = frame.copy()

        dmax = self._density.max()
        if dmax > 0:
            norm = (self._density / dmax * 255).astype(np.uint8)
        else:
            norm = np.zeros((h, w), np.uint8)

        heatmap = cv2.applyColorMap(norm, self.colormap)
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["heatmap"] = self._density.copy()
        payload["heatmap_overlay"] = overlay
        return payload
