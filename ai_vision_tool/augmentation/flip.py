from __future__ import annotations

import cv2

from ..core.base import AIVisionComponent
from ..utils.image_utils import extract_frame, replace_frame


class Flip(AIVisionComponent):
    """Flips an image horizontally, vertically, or both.

    Args:
        horizontal (bool): If True, flips the image left-to-right. Default is False.
        vertical (bool): If True, flips the image top-to-bottom. Default is False.
    """

    def __init__(self, horizontal=False, vertical=False):
        """Initializes Flip with flip direction flags.

        Args:
            horizontal (bool): Flip left-to-right. Default is False.
            vertical (bool): Flip top-to-bottom. Default is False.
        """
        super().__init__()
        self.horizontal = horizontal
        self.vertical = vertical

    def _execute(self, data, config):
        """Applies the specified flip(s) to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'horizontal' and 'vertical'.

        Returns:
            NumPy array or dict: Flipped image in the same format as input.
                Returns a copy of the original if neither flip is enabled.
        """
        frame = extract_frame(data)
        horizontal = config.get("horizontal", self.horizontal)
        vertical = config.get("vertical", self.vertical)

        if horizontal and vertical:
            output = cv2.flip(frame, -1)
        elif horizontal:
            output = cv2.flip(frame, 1)
        elif vertical:
            output = cv2.flip(frame, 0)
        else:
            output = frame.copy()

        return replace_frame(data, output)
