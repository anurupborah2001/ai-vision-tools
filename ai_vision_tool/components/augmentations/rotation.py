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
    """Rotates an image by an arbitrary angle, optionally expanding the canvas.

    Args:
        angle (float): Rotation angle in degrees (counter-clockwise). Default is 0.0.
        scale (float): Scale factor applied during rotation. Default is 1.0.
        expand (bool): If True, expands the output canvas to fit the full rotated image.
            If False, crops to the original dimensions. Default is False.
        border_mode (str): Border fill mode for exposed regions. Default is 'constant'.
        border_value (int or tuple): Fill color for constant border. Default is 0.
    """

    def __init__(
        self,
        angle=0.0,
        scale=1.0,
        expand=False,
        border_mode="constant",
        border_value=0,
    ):
        """Initializes Rotation with angle, scale, and border parameters.

        Args:
            angle (float): Rotation angle in degrees. Default is 0.0.
            scale (float): Scale multiplier. Default is 1.0.
            expand (bool): Expand canvas to fit rotated content. Default is False.
            border_mode (str): Border fill strategy. Default is 'constant'.
            border_value (int or tuple): Constant border fill value. Default is 0.
        """
        super().__init__()
        self.angle = angle
        self.scale = scale
        self.expand = expand
        self.border_mode = border_mode
        self.border_value = border_value

    def _execute(self, data, config):
        """Applies rotation to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'angle', 'scale', 'expand',
                'border_mode', 'border_value'.

        Returns:
            NumPy array or dict: Rotated image in the same format as input.
        """
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
