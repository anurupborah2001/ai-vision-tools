from __future__ import annotations

import queue
import time

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame


class BufferedStream(AIVisionComponent):
    """Ring-buffer component that smooths bursty frame input.

    Args:
        buffer_size: Maximum number of frames to buffer.
        drop_policy: "oldest" (drop oldest on full), "newest" (drop incoming), "block".
        emit_rate: Optional target output FPS (token bucket throttle).
    """

    def __init__(self, buffer_size: int = 30, drop_policy: str = "oldest",
                 emit_rate: float | None = None):
        super().__init__()
        self.buffer_size = buffer_size
        self.drop_policy = drop_policy
        self.emit_rate = emit_rate
        self._q: queue.Queue | None = None
        self._last_emit: float = 0.0

    def setup(self, config: dict):
        self._q = queue.Queue(maxsize=self.buffer_size)
        super().setup(config)

    def _execute(self, data, config):
        policy = config.get("drop_policy", self.drop_policy)
        rate = config.get("emit_rate", self.emit_rate)

        if policy == "oldest":
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(data)
        elif policy == "newest":
            if not self._q.full():
                self._q.put_nowait(data)
        elif policy == "block":
            self._q.put(data)

        if rate is not None:
            interval = 1.0 / rate
            now = time.monotonic()
            wait = interval - (now - self._last_emit)
            if wait > 0:
                time.sleep(wait)
            self._last_emit = time.monotonic()

        try:
            return self._q.get_nowait()
        except queue.Empty:
            return data

    def __len__(self) -> int:
        return self._q.qsize() if self._q else 0

    def flush(self) -> None:
        if self._q:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break


class SlidingWindowBuffer:
    """Accumulates frames and yields fixed-size windows with optional overlap.

    Args:
        window: Number of frames per window.
        overlap: Number of frames to keep from previous window.
    """

    def __init__(self, window: int, overlap: int = 0):
        self.window = window
        self.overlap = overlap
        self._buffer: list = []

    def add(self, frame) -> list | None:
        self._buffer.append(frame)
        if len(self._buffer) >= self.window:
            result = list(self._buffer)
            self._buffer = self._buffer[self.window - self.overlap:]
            return result
        return None

    def reset(self) -> None:
        self._buffer = []

    def __len__(self) -> int:
        return len(self._buffer)
