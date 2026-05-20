from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame
from ai_vision_tool.utils.color_palette import ColorPalette


class DrawUtils(AIVisionComponent):
    """Draws bboxes, masks, and keypoints onto frames from pipeline payload.

    Args:
        font_scale: OpenCV font scale for label text.
        thickness: Line/border thickness in pixels.
        alpha: Mask overlay blend alpha (0=transparent, 1=opaque).
    """

    def __init__(self, font_scale: float = 0.5, thickness: int = 1, alpha: float = 0.4):
        super().__init__()
        self.font_scale = font_scale
        self.thickness = thickness
        self.alpha = alpha
        self._palette = ColorPalette()

    def _draw_bboxes(self, frame: np.ndarray, bboxes: list, color_map: dict, show_conf: bool) -> np.ndarray:
        for det in bboxes:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            label = det.get("label", "")
            conf = det.get("conf", 1.0)
            track_id = det.get("track_id", -1)
            color = color_map.get(label, self._palette.get(label))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            text = f"#{track_id} {label}" if track_id >= 0 else label
            if show_conf and conf < 1.0:
                text += f" {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(frame, text, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _draw_masks(self, frame: np.ndarray, masks: list) -> np.ndarray:
        overlay = frame.copy()
        for mask_info in masks:
            mask = mask_info.get("mask")
            label = mask_info.get("label", "")
            if mask is None:
                continue
            color = self._palette.get(label)
            colored = np.zeros_like(frame)
            colored[mask.astype(bool)] = color
            overlay = cv2.addWeighted(overlay, 1 - self.alpha, colored, self.alpha, 0)
        return overlay

    def _draw_keypoints(self, frame: np.ndarray, poses: list) -> np.ndarray:
        for pose in poses:
            for kp in pose.get("keypoints", []):
                x, y = int(kp.get("x", 0)), int(kp.get("y", 0))
                vis = kp.get("visibility", 1.0)
                if vis > 0.5:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        return frame

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        color_map = config.get("color_map", {})
        show_conf = config.get("show_conf", True)

        if config.get("draw_bboxes", True) and isinstance(data, dict):
            bboxes = data.get("bboxes", data.get("tracks", []))
            if bboxes:
                frame = self._draw_bboxes(frame, bboxes, color_map, show_conf)

        if config.get("draw_masks", False) and isinstance(data, dict):
            masks = data.get("masks", [])
            if masks:
                frame = self._draw_masks(frame, masks)

        if config.get("draw_keypoints", False) and isinstance(data, dict):
            poses = data.get("poses", [])
            if poses:
                frame = self._draw_keypoints(frame, poses)

        return replace_frame(data, frame)
