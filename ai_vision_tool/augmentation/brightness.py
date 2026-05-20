from __future__ import annotations

import cv2

from ..utils.image_utils import extract_frame, replace_frame
from ..core.base import AIVisionComponent


class Brightness(AIVisionComponent):
    """Adjusts image brightness by adding a constant offset to all pixels.

    Args:
        beta (float): Brightness offset added to every pixel. Positive values brighten,
            negative values darken. Default is 0.
    """

    def __init__(self, beta=0):
        """Initializes Brightness with a pixel offset value.

        Args:
            beta (float): Brightness offset. Default is 0.
        """
        super().__init__()
        self.beta = beta

    def _execute(self, data, config):
        """Applies brightness adjustment to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'beta'.

        Returns:
            NumPy array or dict: Brightness-adjusted image in the same format as input.
        """
        frame = extract_frame(data)
        beta = float(config.get("beta", self.beta))
        output = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
        return replace_frame(data, output)
