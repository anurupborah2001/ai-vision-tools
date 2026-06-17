from __future__ import annotations

import time

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent


class CameraSource(AIVisionComponent):
    """Reads frames from a camera, file, or network stream via OpenCV.

    Args:
        source: Webcam index (int), file path, RTSP URL, or HTTP MJPEG URL.
        width: Requested frame width.
        height: Requested frame height.
        fps: Requested frame rate.
        buffer_size: cv2 capture buffer size (1 = latest frame only).
    """

    def __init__(
        self,
        source=0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        buffer_size: int = 1,
    ):
        super().__init__()
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0

    def setup(self, config: dict):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise OSError(f"CameraSource: cannot open {self.source!r}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        super().setup(config)

    def _execute(self, data, config):
        ret, frame = self._cap.read()
        if not ret:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if config.get("flip_horizontal", False):
            frame = cv2.flip(frame, 1)
        if config.get("flip_vertical", False):
            frame = cv2.flip(frame, 0)

        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_id += 1
        return {
            "frame": frame,
            "frame_id": self._frame_id,
            "timestamp_ms": time.time() * 1000.0,
            "source": str(self.source),
            "fps_actual": actual_fps,
        }

    def cleanup(self):
        if self._cap:
            self._cap.release()
            self._cap = None
