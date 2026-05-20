from __future__ import annotations

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Crop(AIVisionComponent):
    """Crops a rectangular region from an image frame.

    Args:
        x (int): Left edge of the crop region in pixels. Default is 0.
        y (int): Top edge of the crop region in pixels. Default is 0.
        width (int or None): Width of the crop region. None means extend to right edge. Default is None.
        height (int or None): Height of the crop region. None means extend to bottom edge. Default is None.
        clamp (bool): If True, clamps the crop region to valid image boundaries. Default is True.
    """

    def __init__(self, x=0, y=0, width=None, height=None, clamp=True):
        """Initializes Crop with position and size parameters.

        Args:
            x (int): Horizontal start of the crop. Default is 0.
            y (int): Vertical start of the crop. Default is 0.
            width (int or None): Crop width in pixels. Default is None (full width from x).
            height (int or None): Crop height in pixels. Default is None (full height from y).
            clamp (bool): Clamp crop region to image bounds. Default is True.
        """
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.clamp = clamp

    def _execute(self, data, config):
        """Crops the extracted frame to the specified region.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'crop_x' / 'x', 'crop_y' / 'y',
                'crop_width' / 'width', 'crop_height' / 'height', 'crop_clamp' / 'clamp'.

        Returns:
            NumPy array or dict: Cropped image in the same format as input.
        """
        frame = extract_frame(data)
        x = int(config.get("crop_x", config.get("x", self.x)))
        y = int(config.get("crop_y", config.get("y", self.y)))
        width = config.get("crop_width", config.get("width", self.width))
        height = config.get("crop_height", config.get("height", self.height))
        clamp = config.get("crop_clamp", config.get("clamp", self.clamp))

        height = frame.shape[0] - y if height is None else int(height)
        width = frame.shape[1] - x if width is None else int(width)

        if clamp:
            x = max(0, min(x, frame.shape[1]))
            y = max(0, min(y, frame.shape[0]))
            width = max(1, min(width, frame.shape[1] - x))
            height = max(1, min(height, frame.shape[0] - y))

        output = frame[y : y + height, x : x + width].copy()
        return replace_frame(data, output)
