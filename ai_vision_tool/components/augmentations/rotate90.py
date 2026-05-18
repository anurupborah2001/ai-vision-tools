from __future__ import annotations

import numpy as np

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class Rotate90(AIVisionComponent):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def _execute(self, data, config):
        frame = extract_frame(data)
        k = int(config.get("k", self.k)) % 4
        output = np.rot90(frame, k).copy()
        return replace_frame(data, output)
