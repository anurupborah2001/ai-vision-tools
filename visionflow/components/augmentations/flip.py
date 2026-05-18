from __future__ import annotations

import cv2

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Flip(AIVisionComponent):
    def __init__(self, horizontal=False, vertical=False):
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical

    def _execute(self, data, config):
        frame = extract_frame(data)
        horizontal = config.get("horizontal", self.horizontal)
        vertical = config.get("vertical", self.vertical)

        if horizontal and vertical:
            output = cv2.flip(frame, -1)
        elif horizontal:
            output = cv2.flip(frame, 1)
        elif vertical:
            output = cv2.flip(frame, 0)
        else:
            output = frame.copy()

        return replace_frame(data, output)
