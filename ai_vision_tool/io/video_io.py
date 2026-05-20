from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame, replace_frame

_CODECS = {"mp4v": cv2.VideoWriter_fourcc(*"mp4v"), "avc1": cv2.VideoWriter_fourcc(*"avc1"),
           "xvid": cv2.VideoWriter_fourcc(*"XVID"), "h264": cv2.VideoWriter_fourcc(*"avc1")}


class VideoReader(AIVisionComponent):
    """Reads frames sequentially from a video file.

    Args:
        path: Path to video file.
        start_frame: First frame index to read.
        end_frame: Last frame index (inclusive), None = read to end.
        step: Read every nth frame.
    """

    def __init__(self, path: str, start_frame: int = 0, end_frame: int | None = None, step: int = 1):
        super().__init__()
        self.path = path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step = step
        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0
        self.fps = 0.0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.duration_s = 0.0

    def setup(self, config: dict):
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"VideoReader: cannot open {self.path!r}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_s = self.frame_count / self.fps if self.fps else 0.0
        if self.start_frame > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self._frame_id = self.start_frame
        super().setup(config)

    def seek(self, frame_id: int) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self._frame_id = frame_id

    def _execute(self, data, config):
        step = int(config.get("step", self.step))
        end = self.end_frame

        for _ in range(step - 1):
            self._cap.grab()
            self._frame_id += 1

        ret, frame = self._cap.read()
        eof = not ret or (end is not None and self._frame_id > end)
        ts = self._frame_id / self.fps * 1000.0 if self.fps else 0.0
        payload = {"frame": frame if ret else np.zeros((self.height, self.width, 3), np.uint8),
                   "frame_id": self._frame_id, "timestamp_ms": ts,
                   "fps": self.fps, "eof": eof}
        self._frame_id += 1
        return payload

    def read_all(self) -> list:
        results = []
        while True:
            result = self.run()
            if result.get("eof"):
                break
            results.append(result)
        return results

    def cleanup(self):
        if self._cap:
            self._cap.release()
            self._cap = None


class VideoWriter(AIVisionComponent):
    """Writes frames from the pipeline payload to a video file.

    Args:
        output_path: Destination video file path.
        fps: Output frame rate.
        codec: FourCC codec string ("mp4v", "avc1", "xvid").
        width: Frame width. None = infer from first frame.
        height: Frame height. None = infer from first frame.
    """

    def __init__(self, output_path: str, fps: float = 30.0, codec: str = "mp4v",
                 width: int | None = None, height: int | None = None):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.width = width
        self.height = height
        self._writer: cv2.VideoWriter | None = None

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            return
        h = self.height or frame.shape[0]
        w = self.width or frame.shape[1]
        fourcc = _CODECS.get(self.codec.lower(), cv2.VideoWriter_fourcc(*"mp4v"))
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

    def _execute(self, data, config):
        frame = extract_frame(data)
        self._ensure_writer(frame)
        self._writer.write(frame)
        return data

    def cleanup(self):
        if self._writer:
            self._writer.release()
            self._writer = None
            print(f"[VideoWriter] Saved: {self.output_path}")
