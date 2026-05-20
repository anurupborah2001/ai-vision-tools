from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_utils import extract_frame, replace_frame
from ..core.base import AIVisionComponent


class Saturation(AIVisionComponent):
    """Scales the saturation channel of an image in HSV color space.

    Args:
        scale (float): Saturation multiplier. Values > 1.0 increase saturation;
            values < 1.0 decrease it. Default is 1.0.
    """

    def __init__(self, scale=1.0):
        """Initializes Saturation with a saturation scale factor.

        Args:
            scale (float): Saturation multiplier. Default is 1.0 (no change).
        """
        super().__init__()
        self.scale = scale

    def _execute(self, data, config):
        """Applies saturation scaling to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'scale'.

        Returns:
            NumPy array or dict: Saturation-adjusted image in BGR format, in the same format as input.
        """
        frame = extract_frame(data)
        scale = float(config.get("scale", self.scale))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return replace_frame(data, output)
