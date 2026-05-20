from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


class TextDetector(AIVisionComponent):
    """Detects text regions using EasyOCR, PaddleOCR, or OpenCV EAST.

    Args:
        backend: "easyocr", "paddleocr", or "opencv_east".
        conf_threshold: Minimum detection confidence.
        nms_threshold: NMS IoU threshold.
        ocr: If True, also extract text content from detected regions.
        model_path: Path to EAST model (only for opencv_east backend).
    """

    def __init__(self, backend: str = "easyocr", conf_threshold: float = 0.5,
                 nms_threshold: float = 0.4, ocr: bool = False, model_path: str | None = None):
        super().__init__()
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.ocr = ocr
        self.model_path = model_path
        self._reader = None
        self._available = True

    def setup(self, config: dict):
        if self.backend == "easyocr":
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=False)
            except ImportError:
                raise ImportError("Install with: pip install easyocr")
        elif self.backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self._reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            except ImportError:
                raise ImportError("Install with: pip install paddleocr")
        elif self.backend == "opencv_east":
            if not self.model_path:
                print("[TextDetector] Warning: model_path not provided for opencv_east. Skipping detection.")
                self._available = False
            else:
                self._reader = cv2.dnn.readNet(self.model_path)
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        conf_thr = float(config.get("conf_threshold", self.conf_threshold))
        do_ocr = config.get("ocr", self.ocr)
        regions = []

        if not self._available:
            payload = data if isinstance(data, dict) else {"frame": frame}
            payload["text_regions"] = []
            payload["warning"] = "TextDetector not available (model_path missing)"
            return payload

        if self.backend == "easyocr":
            results = self._reader.readtext(frame)
            for (bbox_pts, text, conf) in results:
                if conf < conf_thr:
                    continue
                xs = [p[0] for p in bbox_pts]
                ys = [p[1] for p in bbox_pts]
                regions.append({"x1": int(min(xs)), "y1": int(min(ys)),
                                 "x2": int(max(xs)), "y2": int(max(ys)),
                                 "text": text if do_ocr else "", "conf": conf})

        elif self.backend == "paddleocr":
            result = self._reader.ocr(frame, cls=True)
            for line in (result[0] or []):
                pts, (text, conf) = line
                if conf < conf_thr:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                regions.append({"x1": int(min(xs)), "y1": int(min(ys)),
                                 "x2": int(max(xs)), "y2": int(max(ys)),
                                 "text": text if do_ocr else "", "conf": float(conf)})

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["text_regions"] = regions
        return payload
