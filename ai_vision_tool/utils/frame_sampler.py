from __future__ import annotations

import time

from ai_vision_tool.core.base import AIVisionComponent


class FrameSampler(AIVisionComponent):
    """Samples frames based on count, target FPS, or random probability.

    Args:
        every_n: Keep every nth frame (count mode).
        mode: "count", "fps", or "random".
        target_fps: Target output FPS (fps mode).
        prob: Keep probability in [0,1] (random mode).
    """

    def __init__(
        self,
        every_n: int = 1,
        mode: str = "count",
        target_fps: float = 10.0,
        prob: float = 0.5,
    ):
        super().__init__()
        self.every_n = every_n
        self.mode = mode
        self.target_fps = target_fps
        self.prob = prob
        self._counter = 0
        self._last_emit_time: float | None = None

    def _execute(self, data, config):
        import random

        mode = config.get("sample_mode", self.mode)
        keep = False

        if mode == "count":
            n = int(config.get("sample_every_n", self.every_n))
            keep = self._counter % n == 0

        elif mode == "fps":
            fps = float(config.get("sample_every_n", self.target_fps))
            now = time.monotonic()
            interval = 1.0 / fps if fps > 0 else 0.0
            if self._last_emit_time is None or (now - self._last_emit_time) >= interval:
                keep = True
                self._last_emit_time = now

        elif mode == "random":
            p = float(config.get("sample_prob", self.prob))
            keep = random.random() < p

        self._counter += 1

        if isinstance(data, dict):
            data["skip"] = not keep
            return data

        return {"frame": data, "skip": not keep}
