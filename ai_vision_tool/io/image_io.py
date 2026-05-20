from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame, replace_frame

_COLOR_MODES = {"bgr": cv2.IMREAD_COLOR, "rgb": cv2.IMREAD_COLOR, "gray": cv2.IMREAD_GRAYSCALE}


class ImageReader(AIVisionComponent):
    """Reads images from disk into the pipeline payload.

    Args:
        path: Default file path to read when data does not provide one.
        color_mode: "bgr" (default), "rgb", or "gray".
    """

    def __init__(self, path: str | None = None, color_mode: str = "bgr"):
        super().__init__()
        self.path = path
        self.color_mode = color_mode

    def _execute(self, data, config):
        mode = config.get("color_mode", self.color_mode)
        flag = _COLOR_MODES.get(mode, cv2.IMREAD_COLOR)

        if isinstance(data, str):
            src = data
        elif isinstance(data, dict) and "path" in data:
            src = data["path"]
        else:
            src = config.get("read_path", self.path)

        if not src:
            raise ValueError("ImageReader: no path provided")

        frame = cv2.imread(src, flag)
        if frame is None:
            raise IOError(f"ImageReader: could not read {src!r}")
        if mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        payload = data if isinstance(data, dict) else {}
        payload["frame"] = frame
        payload["path"] = src
        payload["source"] = "file"
        return payload


class ImageWriter(AIVisionComponent):
    """Writes frames from the pipeline payload to disk.

    Args:
        output_dir: Directory to save images.
        filename_pattern: Filename pattern with {index}, {timestamp}, {label} placeholders.
        quality: JPEG quality (1-100).
    """

    def __init__(self, output_dir: str = "./output", filename_pattern: str = "{index:06d}.jpg", quality: int = 95):
        super().__init__()
        self.output_dir = output_dir
        self.filename_pattern = filename_pattern
        self.quality = quality
        self._index = 0

    def setup(self, config: dict):
        out = config.get("output_dir", self.output_dir)
        Path(out).mkdir(parents=True, exist_ok=True)
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        out_dir = config.get("output_dir", self.output_dir)
        quality = int(config.get("quality", self.quality))

        label = data.get("label", "") if isinstance(data, dict) else ""
        filename = config.get("filename") or self.filename_pattern.format(
            index=self._index, timestamp=int(time.time()), label=label
        )
        dest = Path(out_dir) / filename
        ext = dest.suffix.lower()

        if ext in (".jpg", ".jpeg"):
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == ".webp":
            cv2.imwrite(str(dest), frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
        else:
            cv2.imwrite(str(dest), frame)

        self._index += 1
        if isinstance(data, dict):
            data["saved_path"] = str(dest)
        return data
