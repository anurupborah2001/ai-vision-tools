from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Exposure(AIVisionComponent):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def _execute(self, data, config):
        frame = extract_frame(data)
        gamma = float(config.get("gamma", self.gamma))
        gamma = max(gamma, 1e-6)
        table = np.array(
            [((index / 255.0) ** (1.0 / gamma)) * 255 for index in np.arange(256)],
            dtype=np.uint8,
        )
        output = cv2.LUT(frame, table)
        return replace_frame(data, output)
