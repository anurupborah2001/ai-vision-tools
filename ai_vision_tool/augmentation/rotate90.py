from __future__ import annotations

import numpy as np

from ..core.base import AIVisionComponent
from ..utils.image_utils import extract_frame, replace_frame


class Rotate90(AIVisionComponent):
    """Rotates an image by multiples of 90 degrees counter-clockwise.

    Args:
        k (int): Number of 90-degree rotations to apply (1=90°, 2=180°, 3=270°). Default is 1.
    """

    def __init__(self, k=1):
        """Initializes Rotate90 with a rotation count.

        Args:
            k (int): Number of 90-degree counter-clockwise rotations. Default is 1.
        """
        super().__init__()
        self.k = k

    def _execute(self, data, config):
        """Rotates the extracted frame by k * 90 degrees.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'k'.

        Returns:
            NumPy array or dict: Rotated image in the same format as input.
        """
        frame = extract_frame(data)
        k = int(config.get("k", self.k)) % 4
        output = np.rot90(frame, k).copy()
        return replace_frame(data, output)
