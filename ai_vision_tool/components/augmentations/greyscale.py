from __future__ import annotations

import cv2

from ..utils.image_utils import extract_frame, maybe_grayscale_to_bgr, replace_frame
from ..core.base import AIVisionComponent


class Greyscale(AIVisionComponent):
    """Converts an image to greyscale, optionally retaining three output channels.

    Args:
        keep_channels (bool): If True, converts to BGR with all three channels set to the
            grey value, preserving downstream shape compatibility. Default is True.
    """

    def __init__(self, keep_channels=True):
        """Initializes Greyscale with channel-retention preference.

        Args:
            keep_channels (bool): Retain three channels in the output. Default is True.
        """
        super().__init__()
        self.keep_channels = keep_channels

    def _execute(self, data, config):
        """Converts the extracted frame to greyscale.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'keep_channels'.

        Returns:
            NumPy array or dict: Greyscale image (3-channel if keep_channels is True,
                otherwise single-channel), in the same format as input.
        """
        frame = extract_frame(data)
        keep_channels = config.get("keep_channels", self.keep_channels)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = maybe_grayscale_to_bgr(gray, keep_channels)
        return replace_frame(data, output)
