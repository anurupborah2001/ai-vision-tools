import cv2
import numpy as np
from .base import AIVisionComponent

class FrameEnhancer(AIVisionComponent):
    """Adjusts visual properties of a frame (brightness, contrast, noise, etc.)."""
    
    def _execute(self, data, config):
        # Allow data to be a frame or a dictionary containing a frame
        frame = data["frame"] if isinstance(data, dict) else data
        output = frame.copy()

        # Extract parameters from config with safe defaults
        brightness = config.get('brightness', 0)
        contrast = config.get('contrast', 1.0)
        sharpen = config.get('sharpen', False)
        denoise = config.get('denoise', False)
        grayscale = config.get('grayscale', False)

        # Apply Brightness & Contrast
        output = cv2.convertScaleAbs(output, alpha=contrast, beta=brightness)

        # Apply Denoise
        if denoise:
            output = cv2.fastNlMeansDenoisingColored(output, None, 10, 10, 7, 21)

        # Apply Sharpen
        if sharpen:
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ])
            output = cv2.filter2D(output, -1, kernel)

        # Apply Grayscale (returns as 3-channel to maintain pipeline compatibility)
        if grayscale:
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Reconstruct output based on input type
        if isinstance(data, dict):
            data["frame"] = output
            return data
        return output
