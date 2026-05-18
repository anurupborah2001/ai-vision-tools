from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent


class AutoAdjustContrast(AIVisionComponent):
    """Automatically adjusts contrast using one of several enhancement strategies."""

    def __init__(
        self,
        method="adaptive_equalization",
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        lower_percentile=2.0,
        upper_percentile=98.0,
        output_min=0,
        output_max=255,
    ):
        super().__init__()
        self.method = method
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.output_min = output_min
        self.output_max = output_max

    def _execute(self, data, config):
        frame = extract_frame(data)
        output = frame.copy()

        method = config.get("contrast_method", config.get("method", self.method)).lower()
        clip_limit = config.get("clip_limit", self.clip_limit)
        tile_grid_size = tuple(config.get("tile_grid_size", self.tile_grid_size))
        lower_percentile = config.get("lower_percentile", self.lower_percentile)
        upper_percentile = config.get("upper_percentile", self.upper_percentile)
        output_min = config.get("output_min", self.output_min)
        output_max = config.get("output_max", self.output_max)

        if method == "adaptive_equalization":
            output = self._adaptive_equalization(output, clip_limit, tile_grid_size)
        elif method == "histogram_equalization":
            output = self._histogram_equalization(output)
        elif method == "contrast_stretching":
            output = self._contrast_stretching(
                output, lower_percentile, upper_percentile, output_min, output_max
            )
        else:
            raise ValueError(
                "Unsupported contrast method. Use adaptive_equalization, "
                "histogram_equalization, or contrast_stretching."
            )

        return replace_frame(data, output)

    @staticmethod
    def _adaptive_equalization(frame, clip_limit, tile_grid_size):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _histogram_equalization(frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        equalized_y = cv2.equalizeHist(y_channel)
        merged = cv2.merge((equalized_y, cr_channel, cb_channel))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    @staticmethod
    def _contrast_stretching(
        frame, lower_percentile, upper_percentile, output_min, output_max
    ):
        lower = np.percentile(frame, lower_percentile)
        upper = np.percentile(frame, upper_percentile)
        if upper <= lower:
            return frame

        stretched = (frame.astype(np.float32) - lower) * (
            (output_max - output_min) / (upper - lower)
        ) + output_min
        return to_uint8(stretched)
