from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Saturation(AIVisionComponent):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def _execute(self, data, config):
        frame = extract_frame(data)
        scale = float(config.get("scale", self.scale))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return replace_frame(data, output)
