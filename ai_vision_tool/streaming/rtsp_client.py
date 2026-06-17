from __future__ import annotations

import queue
import threading
import time
from urllib.parse import urlparse

import cv2

from ai_vision_tool.core.base import AIVisionComponent


class RTSPClient(AIVisionComponent):
    """Reads frames from an RTSP or network stream with automatic reconnection.

    Args:
        url: RTSP/HTTP stream URL.
        reconnect: Retry on failure.
        reconnect_delay: Seconds to wait between reconnect attempts.
        buffer_size: OpenCV capture buffer size.
        timeout: Connection timeout in seconds.
        threaded: If True, capture frames in a background thread.
    """

    def __init__(
        self,
        url: str,
        reconnect: bool = True,
        reconnect_delay: float = 2.0,
        buffer_size: int = 1,
        timeout: float = 10.0,
        threaded: bool = False,
    ):
        super().__init__()
        self.url = url
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.threaded = threaded
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def _open_cap(self) -> bool:
        self._cap = cv2.VideoCapture(self.url)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return self._cap.isOpened()

    def _threaded_reader(self):
        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._queue.put(frame)
                else:
                    time.sleep(0.01)
            else:
                time.sleep(self.reconnect_delay)

    def setup(self, config: dict):
        if not self._open_cap():
            raise OSError(f"RTSPClient: cannot connect to {self.url!r}")
        if self.threaded:
            self._queue = queue.Queue(maxsize=self.buffer_size * 2)
            self._running = True
            self._thread = threading.Thread(target=self._threaded_reader, daemon=True)
            self._thread.start()
        super().setup(config)

    def _execute(self, data, config):
        self._frame_id += 1
        ts = time.time() * 1000.0

        if self.threaded and self._queue:
            try:
                frame = self._queue.get(timeout=self.timeout)
                return {
                    "frame": frame,
                    "frame_id": self._frame_id,
                    "timestamp_ms": ts,
                    "url": self.url,
                    "connected": True,
                }
            except queue.Empty:
                return {
                    "frame": None,
                    "frame_id": self._frame_id,
                    "timestamp_ms": ts,
                    "url": self.url,
                    "connected": False,
                }

        for _attempt in range(3):
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    return {
                        "frame": frame,
                        "frame_id": self._frame_id,
                        "timestamp_ms": ts,
                        "url": self.url,
                        "connected": True,
                    }
            if self.reconnect:
                time.sleep(self.reconnect_delay)
                self._open_cap()
        return {
            "frame": None,
            "frame_id": self._frame_id,
            "timestamp_ms": ts,
            "url": self.url,
            "connected": False,
        }

    def cleanup(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None


class RTSPServer:
    """Utility for parsing and probing RTSP stream metadata."""

    @staticmethod
    def parse_url(url: str) -> dict:
        p = urlparse(url)
        return {
            "scheme": p.scheme,
            "host": p.hostname,
            "port": p.port,
            "path": p.path,
            "user": p.username,
            "password": p.password,
        }

    @staticmethod
    def probe(url: str, timeout: float = 5.0) -> dict:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return {"available": False}
        result = {
            "available": True,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
        cap.release()
        return result
