from __future__ import annotations

import cv2

from .._image_utils import (
    extract_frame,
    normalize_color_value,
    replace_frame,
    resolve_border_mode,
    rotate_bound,
)
from ..base import AIVisionComponent


class Rotation(AIVisionComponent):
    def __init__(
        self,
        angle=0.0,
        scale=1.0,
        expand=False,
        border_mode="constant",
        border_value=0,
    ):
        super().__init__()
        self.angle = angle
        self.scale = scale
        self.expand = expand
        self.border_mode = border_mode
        self.border_value = border_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        angle = float(config.get("angle", self.angle))
        scale = float(config.get("scale", self.scale))
        expand = config.get("expand", self.expand)
        border_mode = resolve_border_mode(config.get("border_mode", self.border_mode))
        border_value = config.get("border_value", self.border_value)

        if expand:
            output = rotate_bound(frame, angle, border_mode, border_value)
        else:
            height, width = frame.shape[:2]
            center = (width / 2.0, height / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, scale)
            output = cv2.warpAffine(
                frame,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
                borderValue=normalize_color_value(border_value),
            )

        return replace_frame(data, output)
