import time
from pathlib import Path

import cv2

from .base import AIVisionComponent


class TimeLapseCapture(AIVisionComponent):
    """Periodically saves frames while passing the input through unchanged."""

    def __init__(self, output_dir="captures", interval_seconds=5, prefix="timelapse"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.interval_seconds = interval_seconds
        self.prefix = prefix
        self._last_capture_at = 0.0

    def setup(self, config):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_initialized = True

    def _execute(self, data, config):
        frame = data["frame"] if isinstance(data, dict) else data
        now = time.time()

        if now - self._last_capture_at >= self.interval_seconds:
            timestamp = int(now * 1000)
            image_path = self.output_dir / f"{self.prefix}_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
            self._last_capture_at = now
            print(f"[{self.__class__.__name__}] Saved frame to {image_path}")

        return data
