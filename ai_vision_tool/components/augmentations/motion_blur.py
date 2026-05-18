from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import ensure_odd, extract_frame, replace_frame
from ..base import AIVisionComponent


class MotionBlur(AIVisionComponent):
    def __init__(self, kernel_size=9, angle=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.angle = angle

    def _execute(self, data, config):
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
