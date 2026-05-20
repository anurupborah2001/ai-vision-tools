from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


class FrameStream:
    """Generator-based frame iterator for webcams, files, URLs, or frame lists.

    Args:
        source: int (webcam), str (file/RTSP/HTTP), list of numpy arrays, or generator.
        max_frames: Stop after this many frames (None = no limit).
        skip_empty: Skip frames where cap.read() returns False.
    """

    def __init__(self, source, max_frames: int | None = None, skip_empty: bool = True):
        self.source = source
        self.max_frames = max_frames
        self.skip_empty = skip_empty
        self._cap: cv2.VideoCapture | None = None
        self._list_iter = None
        self._frame_id = 0

    def _open(self):
        if isinstance(self.source, (int, str)):
            self._cap = cv2.VideoCapture(self.source)
        elif isinstance(self.source, list):
            self._list_iter = iter(self.source)

    def _close(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self._close()

    def __iter__(self):
        if self._cap is None and self._list_iter is None:
            self._open()
        return self

    def __next__(self) -> dict:
        if self.max_frames and self._frame_id >= self.max_frames:
            self._close()
            raise StopIteration

        ts = time.time() * 1000.0

        if self._cap is not None:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    if self.skip_empty:
                        self._close()
                        raise StopIteration
                    frame = np.zeros((480, 640, 3), np.uint8)
                break
        elif self._list_iter is not None:
            try:
                item = next(self._list_iter)
                frame = item["frame"] if isinstance(item, dict) else item
            except StopIteration:
                raise StopIteration
        else:
            raise StopIteration

        self._frame_id += 1
        return {"frame": frame, "frame_id": self._frame_id, "timestamp_ms": ts,
                "source": str(self.source)}


class DirectoryStream(FrameStream):
    """Streams all images from a directory in sorted order.

    Args:
        path: Directory path.
        extensions: Image file extensions to include.
        max_frames: Maximum frames to yield.
    """

    def __init__(self, path: str, extensions: tuple = (".jpg", ".png", ".jpeg", ".bmp"),
                 max_frames: int | None = None):
        self._dir = Path(path)
        self._files = sorted(p for p in self._dir.iterdir() if p.suffix.lower() in extensions)
        super().__init__(source=self._files, max_frames=max_frames)
        self._file_iter = iter(self._files)
        self._list_iter = None

    def __next__(self) -> dict:
        if self.max_frames and self._frame_id >= self.max_frames:
            raise StopIteration
        try:
            file_path = next(self._file_iter)
        except StopIteration:
            raise StopIteration
        frame = cv2.imread(str(file_path))
        if frame is None:
            return self.__next__()
        self._frame_id += 1
        return {"frame": frame, "frame_id": self._frame_id,
                "timestamp_ms": time.time() * 1000.0,
                "path": str(file_path), "source": str(self._dir)}
