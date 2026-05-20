from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class MaskPostProcessor(AIVisionComponent):
    """Applies morphological and filtering operations to masks in the pipeline payload.

    Args:
        operations: List of operations to apply in order.
            Choices: "erode", "dilate", "open", "close", "smooth",
                     "fill_holes", "largest_only", "remove_small".
        kernel_size: Morphological kernel size.
    """

    def __init__(self, operations: list[str] | None = None, kernel_size: int = 5):
        super().__init__()
        self.operations = operations or ["erode", "dilate", "smooth"]
        self.kernel_size = kernel_size

    def _apply_op(self, mask: np.ndarray, op: str, config: dict) -> np.ndarray:
        ks = int(config.get("kernel_size", self.kernel_size))
        ks = ks if ks % 2 == 1 else ks + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        m = mask.astype(np.uint8)

        if op == "erode":
            return cv2.erode(m, kernel) > 0
        if op == "dilate":
            return cv2.dilate(m, kernel) > 0
        if op == "open":
            return cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel) > 0
        if op == "close":
            return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel) > 0
        if op == "smooth":
            return cv2.GaussianBlur(m.astype(np.float32), (ks, ks), 0) > 0.5
        if op == "fill_holes":
            flooded = m.copy()
            h, w = m.shape
            flood = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flooded, flood, (0, 0), 1)
            return (m | (1 - flooded)) > 0
        if op == "largest_only":
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
            if n <= 1:
                return m > 0
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            return (labels == largest)
        if op == "remove_small":
            min_area = int(config.get("min_area", 100))
            n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
            result = np.zeros_like(m)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    result[labels == i] = 1
            return result > 0
        return mask

    def _execute(self, data, config):
        frame = extract_frame(data)
        masks = data.get("masks", []) if isinstance(data, dict) else []
        ops = config.get("operations", self.operations)

        updated = []
        for m_info in masks:
            mask = m_info.get("mask")
            if mask is None:
                updated.append(m_info)
                continue
            for op in ops:
                mask = self._apply_op(mask, op, config)
            updated.append({**m_info, "mask": mask})

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["masks"] = updated
        return payload

    @staticmethod
    def mask_to_polygon(mask: np.ndarray) -> list[list[int]]:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        c = max(contours, key=cv2.contourArea)
        return c.reshape(-1, 2).tolist()

    @staticmethod
    def polygon_to_mask(polygon: list, shape: tuple) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        return mask > 0
