from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Hue(AIVisionComponent):
    def __init__(self, delta=0):
        super().__init__()
        self.delta = delta

    def _execute(self, data, config):
        frame = extract_frame(data)
        delta = int(config.get("delta", self.delta))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return replace_frame(data, output)
