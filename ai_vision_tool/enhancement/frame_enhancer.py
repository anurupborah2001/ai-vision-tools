import cv2
import numpy as np
from ai_vision_tool.core.base import AIVisionComponent


class FrameEnhancer(AIVisionComponent):
    """Adjusts visual properties of a frame (brightness, contrast, noise, etc.)."""

    def _execute(self, data, config):
        """Applies brightness, contrast, denoising, sharpening, and grayscale conversion.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime parameters. Supports:
                - 'brightness' (int): Additive brightness offset applied via
                  cv2.convertScaleAbs beta. Default is 0.
                - 'contrast' (float): Multiplicative contrast scale applied via
                  cv2.convertScaleAbs alpha. Default is 1.0.
                - 'sharpen' (bool): If True, applies a 3x3 Laplacian sharpening kernel.
                  Default is False.
                - 'denoise' (bool): If True, applies fastNlMeansDenoisingColored.
                  Default is False.
                - 'grayscale' (bool): If True, converts to grayscale then back to BGR
                  to maintain 3-channel pipeline compatibility. Default is False.

        Returns:
            numpy.ndarray or dict: Enhanced image in the same format as input.
        """
        frame = data["frame"] if isinstance(data, dict) else data
        output = frame.copy()

        brightness = config.get('brightness', 0)
        contrast = config.get('contrast', 1.0)
        sharpen = config.get('sharpen', False)
        denoise = config.get('denoise', False)
        grayscale = config.get('grayscale', False)

        output = cv2.convertScaleAbs(output, alpha=contrast, beta=brightness)

        if denoise:
            output = cv2.fastNlMeansDenoisingColored(output, None, 10, 10, 7, 21)

        if sharpen:
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ])
            output = cv2.filter2D(output, -1, kernel)

        if grayscale:
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if isinstance(data, dict):
            data["frame"] = output
            return data
        return output
