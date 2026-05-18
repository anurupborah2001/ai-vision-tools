from __future__ import annotations

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Crop(AIVisionComponent):
    def __init__(self, x=0, y=0, width=None, height=None, clamp=True):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.clamp = clamp

    def _execute(self, data, config):
        frame = extract_frame(data)
        x = int(config.get("crop_x", config.get("x", self.x)))
        y = int(config.get("crop_y", config.get("y", self.y)))
        width = config.get("crop_width", config.get("width", self.width))
        height = config.get("crop_height", config.get("height", self.height))
        clamp = config.get("crop_clamp", config.get("clamp", self.clamp))

        height = frame.shape[0] - y if height is None else int(height)
        width = frame.shape[1] - x if width is None else int(width)

        if clamp:
            x = max(0, min(x, frame.shape[1]))
            y = max(0, min(y, frame.shape[0]))
            width = max(1, min(width, frame.shape[1] - x))
            height = max(1, min(height, frame.shape[0] - y))

        output = frame[y : y + height, x : x + width].copy()
        return replace_frame(data, output)
