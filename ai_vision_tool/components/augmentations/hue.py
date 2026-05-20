from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_utils import extract_frame, replace_frame
from ..core.base import AIVisionComponent


class Hue(AIVisionComponent):
    """Shifts the hue channel of an image in HSV color space.

    Args:
        delta (int): Hue shift in degrees, applied modulo 180 (OpenCV hue range). Default is 0.
    """

    def __init__(self, delta=0):
        """Initializes Hue with a hue shift value.

        Args:
            delta (int): Degrees to shift the hue channel. Default is 0.
        """
        super().__init__()
        self.delta = delta

    def _execute(self, data, config):
        """Applies a hue shift to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'delta'.

        Returns:
            NumPy array or dict: Hue-shifted image in BGR format, in the same format as input.
        """
        frame = extract_frame(data)
        delta = int(config.get("delta", self.delta))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return replace_frame(data, output)
