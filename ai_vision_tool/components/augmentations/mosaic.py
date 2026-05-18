from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Mosaic(AIVisionComponent):
    def __init__(self, output_size=None, mosaic_images=None):
        super().__init__()
        self.output_size = output_size
        self.mosaic_images = mosaic_images or []

    def _execute(self, data, config):
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
