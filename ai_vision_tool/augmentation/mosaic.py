from __future__ import annotations

import cv2
import numpy as np

from ..core.base import AIVisionComponent
from ..utils.image_utils import extract_frame, replace_frame


class Mosaic(AIVisionComponent):
    """Tiles four images in a 2x2 grid mosaic.

    Uses the primary frame to fill any missing slots when fewer than three partner images
    are provided.

    Args:
        output_size (tuple[int, int] or None): (width, height) of the mosaic output.
            Default is None (2x the input frame dimensions).
        mosaic_images (list or None): Up to three additional images for the mosaic grid.
            Default is None.
    """

    def __init__(self, output_size=None, mosaic_images=None):
        """Initializes Mosaic with optional output size and partner images.

        Args:
            output_size (tuple[int, int] or None): Output (width, height). Default is None.
            mosaic_images (list or None): Up to 3 additional NumPy images. Default is None.
        """
        super().__init__()
        self.output_size = output_size
        self.mosaic_images = mosaic_images or []

    def _execute(self, data, config):
        """Assembles a 2x2 mosaic from the primary frame and up to three partner images.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'output_size' and 'mosaic_images'.

        Returns:
            NumPy array or dict: 2x2 mosaic image in the same format as input.
        """
        frame = extract_frame(data)
        output_size = config.get("output_size", self.output_size)
        extra_images = config.get("mosaic_images", self.mosaic_images)

        tiles = [frame]
        tiles.extend(extra_images[:3])
        while len(tiles) < 4:
            tiles.append(frame)

        height, width = frame.shape[:2]
        if output_size is None:
            output_width, output_height = width * 2, height * 2
        else:
            output_width, output_height = output_size

        cell_width = output_width // 2
        cell_height = output_height // 2

        resized_tiles = [
            cv2.resize(tile, (cell_width, cell_height), interpolation=cv2.INTER_LINEAR)
            for tile in tiles[:4]
        ]
        top = np.hstack((resized_tiles[0], resized_tiles[1]))
        bottom = np.hstack((resized_tiles[2], resized_tiles[3]))
        output = np.vstack((top, bottom))

        return replace_frame(data, output)
