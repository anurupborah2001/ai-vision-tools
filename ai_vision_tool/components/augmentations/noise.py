from __future__ import annotations

import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent


class Noise(AIVisionComponent):
    def __init__(
        self,
        mode="gaussian",
        mean=0.0,
        stddev=10.0,
        amount=0.02,
        salt_vs_pepper=0.5,
    ):
        super().__init__()
        self.mode = mode
        self.mean = mean
        self.stddev = stddev
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def _execute(self, data, config):
        frame = extract_frame(data)
        mode = config.get("mode", self.mode).lower()
        mean = float(config.get("mean", self.mean))
        stddev = float(config.get("stddev", self.stddev))
        amount = float(config.get("amount", self.amount))
        salt_vs_pepper = float(config.get("salt_vs_pepper", self.salt_vs_pepper))

        if mode == "gaussian":
            noise = np.random.normal(mean, stddev, frame.shape)
            output = to_uint8(frame.astype(np.float32) + noise)
        elif mode == "salt_pepper":
            output = frame.copy()
            total_pixels = frame.shape[0] * frame.shape[1]
            salt = int(total_pixels * amount * salt_vs_pepper)
            pepper = int(total_pixels * amount * (1.0 - salt_vs_pepper))

            if salt > 0:
                coords = (
                    np.random.randint(0, frame.shape[0], salt),
                    np.random.randint(0, frame.shape[1], salt),
                )
                output[coords] = 255
            if pepper > 0:
                coords = (
                    np.random.randint(0, frame.shape[0], pepper),
                    np.random.randint(0, frame.shape[1], pepper),
                )
                output[coords] = 0
        else:
            raise ValueError("Unsupported noise mode. Use gaussian or salt_pepper.")

        return replace_frame(data, output)
