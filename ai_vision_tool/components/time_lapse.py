import time
from pathlib import Path

import cv2

from .base import AIVisionComponent


class TimeLapseCapture(AIVisionComponent):
    """Periodically saves frames while passing the input through unchanged.

    Captures one frame to disk every ``interval_seconds`` and returns data
    unmodified so downstream pipeline components continue processing.

    Args:
        output_dir (str): Directory where captured frames are saved. Default is 'captures'.
        interval_seconds (float): Minimum time between consecutive saves. Default is 5.
        prefix (str): Filename prefix for saved frames. Default is 'timelapse'.
    """

    def __init__(self, output_dir="captures", interval_seconds=5, prefix="timelapse"):
        """Initializes TimeLapseCapture with capture interval and output settings.

        Args:
            output_dir (str): Directory for saved frames. Default is 'captures'.
            interval_seconds (float): Seconds between captures. Default is 5.
            prefix (str): Filename prefix. Default is 'timelapse'.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.interval_seconds = interval_seconds
        self.prefix = prefix
        self._last_capture_at = 0.0

    def setup(self, config):
        """Creates the output directory on first use.

        Args:
            config (dict): Unused. Present for interface compatibility.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_initialized = True

    def _execute(self, data, config):
        """Saves the current frame if the capture interval has elapsed.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Unused. Present for interface compatibility.

        Returns:
            Same as input data, unchanged.
        """
        frame = data["frame"] if isinstance(data, dict) else data
        now = time.time()

        if now - self._last_capture_at >= self.interval_seconds:
            timestamp = int(now * 1000)
            image_path = self.output_dir / f"{self.prefix}_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
            self._last_capture_at = now
            print(f"[{self.__class__.__name__}] Saved frame to {image_path}")

        return data
