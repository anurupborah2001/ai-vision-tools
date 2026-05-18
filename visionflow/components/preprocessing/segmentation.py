from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent


class RemoveBackground(AIVisionComponent):
    def __init__(self, method="threshold", threshold=10, rect=None, keep_mask=False, background_value=(0, 0, 0)):
        super().__init__()
        self.method = method
        self.threshold = threshold
        self.rect = rect
        self.keep_mask = keep_mask
        self.background_value = background_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        method = config.get("remove_background_method", config.get("method", self.method)).lower()
        background_value = config.get("background_value", self.background_value)

        if method == "threshold":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(
                gray,
                float(config.get("threshold", self.threshold)),
                255,
                cv2.THRESH_BINARY,
            )
        elif method == "grabcut":
            rect = tuple(config.get("rect", self.rect or (1, 1, frame.shape[1] - 2, frame.shape[0] - 2)))
            mask = np.zeros(frame.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
            mask = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype("uint8")
        else:
            raise ValueError("RemoveBackground supports threshold or grabcut.")

        output = frame.copy()
        output[mask == 0] = background_value
        if isinstance(data, dict) and config.get("keep_mask", self.keep_mask):
            data["mask"] = mask
        return replace_frame(data, output)
