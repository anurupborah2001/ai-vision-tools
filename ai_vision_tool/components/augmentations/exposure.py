from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Exposure(AIVisionComponent):
    """Adjusts image exposure using gamma correction via a lookup table.

    Args:
        gamma (float): Gamma value. Values > 1.0 brighten the image; values < 1.0 darken it. Default is 1.0.
    """

    def __init__(self, gamma=1.0):
        """Initializes Exposure with a gamma value.

        Args:
            gamma (float): Gamma correction factor. Default is 1.0 (no change).
        """
        super().__init__()
        self.gamma = gamma

    def _execute(self, data, config):
        """Applies gamma-based exposure adjustment to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'gamma'.

        Returns:
            NumPy array or dict: Exposure-adjusted image in the same format as input.
        """
        frame = extract_frame(data)
        gamma = float(config.get("gamma", self.gamma))
        gamma = max(gamma, 1e-6)
        table = np.array(
            [((index / 255.0) ** (1.0 / gamma)) * 255 for index in np.arange(256)],
            dtype=np.uint8,
        )
        output = cv2.LUT(frame, table)
        return replace_frame(data, output)
