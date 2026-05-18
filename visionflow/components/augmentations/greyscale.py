from __future__ import annotations

import cv2

from .._image_utils import extract_frame, maybe_grayscale_to_bgr, replace_frame
from ..base import AIVisionComponent


class Greyscale(AIVisionComponent):
    def __init__(self, keep_channels=True):
        super().__init__()
        self.keep_channels = keep_channels

    def _execute(self, data, config):
        frame = extract_frame(data)
        keep_channels = config.get("keep_channels", self.keep_channels)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = maybe_grayscale_to_bgr(gray, keep_channels)
        return replace_frame(data, output)
