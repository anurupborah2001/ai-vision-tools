from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_utils import ensure_odd, extract_frame, replace_frame
from ..core.base import AIVisionComponent


class MotionBlur(AIVisionComponent):
    """Simulates linear motion blur by convolving with a rotated line kernel.

    Args:
        kernel_size (int): Length of the motion blur kernel. Must be odd. Default is 9.
        angle (float): Angle of motion in degrees (0 = horizontal). Default is 0.0.
    """

    def __init__(self, kernel_size=9, angle=0.0):
        """Initializes MotionBlur with kernel size and direction.

        Args:
            kernel_size (int): Kernel length. Must be odd. Default is 9.
            angle (float): Motion direction in degrees. Default is 0.0.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle

    def _execute(self, data, config):
        """Applies motion blur to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'kernel_size' and 'angle'.

        Returns:
            NumPy array or dict: Motion-blurred image in the same format as input.
        """
        frame = extract_frame(data)
        kernel_size = ensure_odd(config.get("kernel_size", self.kernel_size))
        angle = float(config.get("angle", self.angle))

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(
            (kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle, 1.0
        )
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        kernel_sum = kernel.sum()
        if kernel_sum != 0:
            kernel /= kernel_sum

        output = cv2.filter2D(frame, -1, kernel)
        return replace_frame(data, output)
