from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.color_palette import ColorPalette
from ai_vision_tool.utils.image_utils import extract_frame


class BBoxRenderer(AIVisionComponent):
    """Renders bounding boxes and track annotations onto frames.

    Args:
        color_map: Dict mapping label → BGR tuple.
        thickness: Rectangle border thickness.
        font_scale: Label text font scale.
        show_conf: Show confidence score in label.
        show_label: Show class label.
        show_track_id: Show track ID (from "track_id" key in bbox).
        alpha: Semi-transparent fill alpha (0 = no fill).
    """

    def __init__(
        self,
        color_map: dict | None = None,
        thickness: int = 2,
        font_scale: float = 0.5,
        show_conf: bool = True,
        show_label: bool = True,
        show_track_id: bool = True,
        alpha: float = 0.0,
    ):
        super().__init__()
        self.color_map = color_map or {}
        self.thickness = thickness
        self.font_scale = font_scale
        self.show_conf = show_conf
        self.show_label = show_label
        self.show_track_id = show_track_id
        self.alpha = alpha
        self._palette = ColorPalette()

    def _get_color(self, label: str) -> tuple:
        return self.color_map.get(label, self._palette.get(label))

    def _render(self, frame: np.ndarray, items: list, config: dict) -> np.ndarray:
        thickness = int(config.get("thickness", self.thickness))
        font_scale = float(config.get("font_scale", self.font_scale))
        show_conf = config.get("show_conf", self.show_conf)
        show_label = config.get("show_label", self.show_label)
        show_tid = config.get("show_track_id", self.show_track_id)
        alpha = float(config.get("alpha", self.alpha))

        overlay = frame.copy() if alpha > 0 else None

        for item in items:
            x1, y1, x2, y2 = (
                int(item["x1"]),
                int(item["y1"]),
                int(item["x2"]),
                int(item["y2"]),
            )
            label = item.get("label", "")
            conf = item.get("conf", 1.0)
            track_id = item.get("track_id", -1)
            color = self._get_color(label)

            if alpha > 0 and overlay is not None:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            parts = []
            if show_tid and track_id >= 0:
                parts.append(f"#{track_id}")
            if show_label and label:
                parts.append(label)
            if show_conf and conf < 1.0:
                parts.append(f"{conf:.2f}")
            text = " ".join(parts)

            if text:
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame,
                    text,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if alpha > 0 and overlay is not None:
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        items = []
        if isinstance(data, dict):
            items = data.get("bboxes", data.get("tracks", []))
        rendered = self._render(frame, items, config)
        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["rendered_frame"] = rendered
        payload["frame"] = rendered
        return payload
