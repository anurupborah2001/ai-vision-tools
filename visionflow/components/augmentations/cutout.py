from __future__ import annotations

from .._image_utils import extract_frame, normalize_color_value, replace_frame
from ..base import AIVisionComponent


class Cutout(AIVisionComponent):
    def __init__(self, x=0, y=0, width=32, height=32, fill_value=(0, 0, 0)):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.fill_value = fill_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        x = int(config.get("cutout_x", config.get("x", self.x)))
        y = int(config.get("cutout_y", config.get("y", self.y)))
        width = int(config.get("cutout_width", config.get("width", self.width)))
        height = int(config.get("cutout_height", config.get("height", self.height)))
        fill_value = normalize_color_value(
            config.get("cutout_fill_value", config.get("fill_value", self.fill_value))
        )

        output = frame.copy()
        x2 = min(output.shape[1], max(0, x + width))
        y2 = min(output.shape[0], max(0, y + height))
        x = max(0, x)
        y = max(0, y)
        output[y:y2, x:x2] = fill_value
        return replace_frame(data, output)
