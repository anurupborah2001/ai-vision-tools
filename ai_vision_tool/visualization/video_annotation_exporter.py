from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.color_palette import ColorPalette
from ai_vision_tool.utils.image_utils import extract_frame

_CODECS = {
    "mp4v": cv2.VideoWriter_fourcc(*"mp4v"),
    "avc1": cv2.VideoWriter_fourcc(*"avc1"),
    "xvid": cv2.VideoWriter_fourcc(*"XVID"),
}


class VideoAnnotationExporter(AIVisionComponent):
    """Writes annotated video with optional JSON sidecar annotation file.

    Args:
        output_path: Destination video file path.
        fps: Output frame rate.
        codec: FourCC codec string.
        burn_annotations: Render bboxes/tracks onto frames before writing.
        export_json: Write annotation JSON sidecar file.
    """

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        codec: str = "mp4v",
        burn_annotations: bool = True,
        export_json: bool = True,
    ):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
        self.burn_annotations = burn_annotations
        self.export_json = export_json
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0
        self._annotations: list = []
        self._start_time = 0.0
        self._palette = ColorPalette()

    def setup(self, config: dict):
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        super().setup(config)

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is not None:
            return
        h, w = frame.shape[:2]
        fourcc = _CODECS.get(self.codec.lower(), cv2.VideoWriter_fourcc(*"mp4v"))
        self._writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))

    def _burn(self, frame: np.ndarray, data: dict) -> np.ndarray:
        out = frame.copy()
        items = data.get("bboxes", data.get("tracks", []))
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
            color = self._palette.get(label)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            parts = [f"#{track_id}" if track_id >= 0 else "", label, f"{conf:.2f}"]
            text = " ".join(p for p in parts if p)
            if text:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    out,
                    text,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return out

    def _execute(self, data, config):
        frame = extract_frame(data)
        self._ensure_writer(frame)
        self._frame_count += 1
        ts_ms = (time.time() - self._start_time) * 1000.0

        write_frame = (
            self._burn(frame, data)
            if self.burn_annotations and isinstance(data, dict)
            else frame
        )
        self._writer.write(write_frame)

        if self.export_json and isinstance(data, dict):
            self._annotations.append(
                {
                    "frame_id": self._frame_count,
                    "timestamp_ms": round(ts_ms, 1),
                    "bboxes": data.get("bboxes", []),
                    "tracks": data.get("tracks", []),
                    "metadata": {
                        k: v
                        for k, v in data.items()
                        if k not in ("frame", "bboxes", "tracks", "masks", "seg_map")
                    },
                }
            )
        return data

    def finalize(self):
        if self._writer:
            self._writer.release()
            self._writer = None
            print(f"[VideoAnnotationExporter] Video saved: {self.output_path}")

        if self.export_json and self._annotations:
            stem = Path(self.output_path).stem
            json_path = Path(self.output_path).parent / f"{stem}_annotations.json"
            json_path.write_text(
                json.dumps(
                    {
                        "frames": self._annotations,
                        "fps": self.fps,
                        "frame_count": self._frame_count,
                    },
                    indent=2,
                )
            )
            print(f"[VideoAnnotationExporter] Annotations saved: {json_path}")

        print(f"[VideoAnnotationExporter] Total frames: {self._frame_count}")

    def cleanup(self):
        self.finalize()
