from __future__ import annotations

import cv2

from .._image_utils import ensure_odd, extract_frame, replace_frame
from ..base import AIVisionComponent


class Blur(AIVisionComponent):
    def __init__(self, kernel_size=5, sigma_x=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def _execute(self, data, config):
        frame = extract_frame(data)
        kernel_size = ensure_odd(config.get("kernel_size", self.kernel_size))
        sigma_x = float(config.get("sigma_x", self.sigma_x))
        output = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma_x)
        return replace_frame(data, output)
