from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_utils import extract_frame, replace_frame, to_uint8
from ..core.base import AIVisionComponent


class RemoveBackground(AIVisionComponent):
    """Removes the background from an image using thresholding or GrabCut.

    Supports two methods:
    - 'threshold': simple binary threshold on grayscale.
    - 'grabcut': iterative foreground/background segmentation using GrabCut.

    Pixels classified as background are replaced with background_value.

    Args:
        method (str): Segmentation method ('threshold' or 'grabcut'). Default is 'threshold'.
        threshold (float): Threshold value used with 'threshold' method. Default is 10.
        rect (tuple[int, int, int, int] or None): (x, y, w, h) seed rectangle for GrabCut.
            Default is None (uses near-full image bounds).
        keep_mask (bool): If True and input is a dict, saves the binary mask under 'mask'. Default is False.
        background_value (tuple[int, int, int] or int): Replacement color for background pixels. Default is (0, 0, 0).
    """

    def __init__(self, method="threshold", threshold=10, rect=None, keep_mask=False, background_value=(0, 0, 0)):
        """Initializes RemoveBackground with segmentation parameters.

        Args:
            method (str): Segmentation method. Default is 'threshold'.
            threshold (float): Pixel threshold for binary segmentation. Default is 10.
            rect (tuple[int, int, int, int] or None): GrabCut seed rect. Default is None.
            keep_mask (bool): Store binary mask in payload. Default is False.
            background_value (tuple[int, int, int] or int): Background fill value. Default is (0, 0, 0).
        """
        super().__init__()
        self.method = method
        self.threshold = threshold
        self.rect = rect
        self.keep_mask = keep_mask
        self.background_value = background_value

    def _execute(self, data, config):
        """Applies background removal to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'remove_background_method' / 'method',
                'threshold', 'rect', 'keep_mask', 'background_value'.

        Returns:
            NumPy array or dict: Image with background pixels replaced, in the same format as input.

        Raises:
            ValueError: If method is not 'threshold' or 'grabcut'.
        """
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
