from __future__ import annotations

from .._image_utils import extract_frame, normalize_color_value, replace_frame
from ..base import AIVisionComponent


class Cutout(AIVisionComponent):
    """Fills a rectangular region of an image with a constant color (cutout regularization).

    Args:
        x (int): Left edge of the cutout region. Default is 0.
        y (int): Top edge of the cutout region. Default is 0.
        width (int): Width of the cutout region in pixels. Default is 32.
        height (int): Height of the cutout region in pixels. Default is 32.
        fill_value (tuple[int, int, int] or int): BGR fill color. Default is (0, 0, 0).
    """

    def __init__(self, x=0, y=0, width=32, height=32, fill_value=(0, 0, 0)):
        """Initializes Cutout with position, size, and fill color.

        Args:
            x (int): Horizontal start of the cutout. Default is 0.
            y (int): Vertical start of the cutout. Default is 0.
            width (int): Cutout width. Default is 32.
            height (int): Cutout height. Default is 32.
            fill_value (tuple[int, int, int] or int): Fill color in BGR. Default is (0, 0, 0).
        """
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.fill_value = fill_value

    def _execute(self, data, config):
        """Applies a filled rectangular mask to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'cutout_x' / 'x', 'cutout_y' / 'y',
                'cutout_width' / 'width', 'cutout_height' / 'height',
                'cutout_fill_value' / 'fill_value'.

        Returns:
            NumPy array or dict: Image with the cutout region filled, in the same format as input.
        """
        frame = extract_frame(data)
        x = int(config.get("cutout_x", config.get("x", self.x)))
        y = int(config.get("cutout_y", config.get("y", self.y)))
        width = int(config.get("cutout_width", config.get("width", self.width)))
        height = int(config.get("cutout_height", config.get("height", self.height)))
        fill_value = normalize_color_value(
            config.get("cutout_fill_value", config.get("fill_value", self.fill_value))
        )

        output = frame.copy()
        x2 = min(output.shape[1], max(0, x + width))
        y2 = min(output.shape[0], max(0, y + height))
        x = max(0, x)
        y = max(0, y)
        output[y:y2, x:x2] = fill_value
        return replace_frame(data, output)
