from __future__ import annotations

import cv2

from ..utils.image_utils import ensure_odd, extract_frame, replace_frame
from ..core.base import AIVisionComponent


class Blur(AIVisionComponent):
    """Applies Gaussian blur to an image frame.

    Args:
        kernel_size (int): Size of the Gaussian kernel. Must be odd. Default is 5.
        sigma_x (float): Standard deviation in the X direction. Default is 0.0 (auto-computed).
    """

    def __init__(self, kernel_size=5, sigma_x=0.0):
        """Initializes Blur with kernel size and sigma.

        Args:
            kernel_size (int): Size of the Gaussian kernel. Must be odd. Default is 5.
            sigma_x (float): Standard deviation in the X direction. Default is 0.0 (auto-computed).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def _execute(self, data, config):
        """Applies Gaussian blur to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'kernel_size' and 'sigma_x'.

        Returns:
            NumPy array or dict: Blurred image in the same format as input.
        """
        frame = extract_frame(data)
        kernel_size = ensure_odd(config.get("kernel_size", self.kernel_size))
        sigma_x = float(config.get("sigma_x", self.sigma_x))
        output = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma_x)
        return replace_frame(data, output)
