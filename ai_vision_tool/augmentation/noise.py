from __future__ import annotations

import numpy as np

from ..core.base import AIVisionComponent
from ..utils.image_utils import extract_frame, replace_frame, to_uint8


class Noise(AIVisionComponent):
    """Adds Gaussian or salt-and-pepper noise to an image.

    Args:
        mode (str): Noise type. Either 'gaussian' or 'salt_pepper'. Default is 'gaussian'.
        mean (float): Mean of the Gaussian distribution (used when mode='gaussian'). Default is 0.0.
        stddev (float): Standard deviation of the Gaussian distribution. Default is 10.0.
        amount (float): Fraction of pixels affected (used when mode='salt_pepper'). Default is 0.02.
        salt_vs_pepper (float): Ratio of salt to total noise pixels in [0, 1]. Default is 0.5.
    """

    def __init__(
        self,
        mode="gaussian",
        mean=0.0,
        stddev=10.0,
        amount=0.02,
        salt_vs_pepper=0.5,
    ):
        """Initializes Noise with noise mode and distribution parameters.

        Args:
            mode (str): Noise mode ('gaussian' or 'salt_pepper'). Default is 'gaussian'.
            mean (float): Gaussian mean. Default is 0.0.
            stddev (float): Gaussian standard deviation. Default is 10.0.
            amount (float): Fraction of pixels corrupted (salt_pepper mode). Default is 0.02.
            salt_vs_pepper (float): Salt fraction among noise pixels. Default is 0.5.
        """
        super().__init__()
        self.mode = mode
        self.mean = mean
        self.stddev = stddev
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def _execute(self, data, config):
        """Applies the configured noise to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'mode', 'mean', 'stddev',
                'amount', 'salt_vs_pepper'.

        Returns:
            NumPy array or dict: Noisy image in the same format as input.

        Raises:
            ValueError: If mode is not 'gaussian' or 'salt_pepper'.
        """
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
