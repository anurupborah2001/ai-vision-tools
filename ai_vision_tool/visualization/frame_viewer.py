from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class FrameViewer(AIVisionComponent):
    """Displays frames in an OpenCV window with optional FPS and info overlay.

    Args:
        window_name: Display window title.
        scale: Display scale factor.
        wait_ms: cv2.waitKey delay (1 = near-real-time, 0 = pause).
        show_fps: Overlay rolling FPS counter.
        show_info: Overlay info dict from payload["info"].
    """

    def __init__(self, window_name: str = "AI Vision Tool", scale: float = 1.0,
                 wait_ms: int = 1, show_fps: bool = True, show_info: bool = True):
        super().__init__()
        self.window_name = window_name
        self.scale = scale
        self.wait_ms = wait_ms
        self.show_fps = show_fps
        self.show_info = show_info
        self._timestamps: deque = deque(maxlen=30)
        self._headless_warned = False

    def _fps(self) -> float:
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def _execute(self, data, config):
        frame = extract_frame(data)
        display = frame.copy()

        scale = float(config.get("scale", self.scale))
        if scale != 1.0:
            h, w = display.shape[:2]
            display = cv2.resize(display, (int(w * scale), int(h * scale)))

        fps = self._fps()
        if config.get("show_fps", self.show_fps):
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if config.get("show_info", self.show_info) and isinstance(data, dict):
            info = data.get("info", {})
            y = 55
            for k, v in info.items():
                cv2.putText(display, f"{k}: {v}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                y += 20

        try:
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(int(config.get("wait_ms", self.wait_ms))) & 0xFF
            if key == ord("q"):
                if isinstance(data, dict):
                    data["stop"] = True
        except cv2.error:
            if not self._headless_warned:
                print(f"[FrameViewer] No display available (headless environment)")
                self._headless_warned = True

        return data

    def cleanup(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
