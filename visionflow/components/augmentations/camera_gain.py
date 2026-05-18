from __future__ import annotations

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent


class CameraGain(AIVisionComponent):
    def __init__(self, gain=1.0, black_level=0.0):
        super().__init__()
        self.gain = gain
        self.black_level = black_level

    def _execute(self, data, config):
        frame = extract_frame(data)
        gain = float(config.get("gain", self.gain))
        black_level = float(config.get("black_level", self.black_level))
        adjusted = frame.astype("float32") * gain + black_level
        output = to_uint8(adjusted)
        return replace_frame(data, output)
