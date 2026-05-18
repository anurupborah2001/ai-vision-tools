from __future__ import annotations

import cv2

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Brightness(AIVisionComponent):
    def __init__(self, beta=0):
        super().__init__()
        self.beta = beta

    def _execute(self, data, config):
        frame = extract_frame(data)
        beta = float(config.get("beta", self.beta))
        output = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
        return replace_frame(data, output)
