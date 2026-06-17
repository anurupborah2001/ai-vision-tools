from __future__ import annotations

import cv2
import numpy as np

from ..core.base import AIVisionComponent
from ..utils.image_utils import (
    extract_frame,
    normalize_color_value,
    replace_frame,
    resolve_border_mode,
)


class Shear(AIVisionComponent):
    """Applies an affine shear transformation to an image.

    Args:
        shear_x (float): Horizontal shear coefficient. Default is 0.0.
        shear_y (float): Vertical shear coefficient. Default is 0.0.
        border_mode (str): Border fill mode for exposed regions. Default is 'constant'.
        border_value (int or tuple): Fill color for constant border. Default is 0.
    """

    def __init__(
        self, shear_x=0.0, shear_y=0.0, border_mode="constant", border_value=0
    ):
        """Initializes Shear with shear coefficients and border parameters.

        Args:
            shear_x (float): Horizontal shear. Default is 0.0.
            shear_y (float): Vertical shear. Default is 0.0.
            border_mode (str): Border fill strategy. Default is 'constant'.
            border_value (int or tuple): Constant border fill value. Default is 0.
        """
        super().__init__()
        self.shear_x = shear_x
        self.shear_y = shear_y
        self.border_mode = border_mode
        self.border_value = border_value

    def _execute(self, data, config):
        """Applies shear transformation to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'shear_x', 'shear_y',
                'border_mode', 'border_value'.

        Returns:
            NumPy array or dict: Sheared image in the same format as input.
        """
        frame = extract_frame(data)
        shear_x = float(config.get("shear_x", self.shear_x))
        shear_y = float(config.get("shear_y", self.shear_y))
        border_mode = resolve_border_mode(config.get("border_mode", self.border_mode))
        border_value = config.get("border_value", self.border_value)

        height, width = frame.shape[:2]
        matrix = np.array([[1.0, shear_x, 0.0], [shear_y, 1.0, 0.0]], dtype=np.float32)
        output = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=normalize_color_value(border_value),
        )
        return replace_frame(data, output)
