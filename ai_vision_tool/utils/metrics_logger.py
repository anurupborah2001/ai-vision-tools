from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


class MetricsLogger:
    """Thread-safe rolling-window FPS and latency tracker."""

    def __init__(self, window: int = 30):
        self._window = window
        self._timestamps: deque[float] = deque(maxlen=window)
        self._latencies: deque[float] = deque(maxlen=window)
        self._frame_count = 0
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self._timestamps.append(time.perf_counter())
            self._frame_count += 1

    def log_latency(self, ms: float) -> None:
        with self._lock:
            self._latencies.append(ms)

    def fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def mean_latency(self) -> float:
        with self._lock:
            return float(np.mean(self._latencies)) if self._latencies else 0.0

    def report(self) -> dict:
        with self._lock:
            lats = list(self._latencies)
        fps_val = self.fps()
        return {
            "fps": round(fps_val, 2),
            "mean_latency_ms": round(float(np.mean(lats)) if lats else 0.0, 3),
            "p95_latency_ms": round(float(np.percentile(lats, 95)) if lats else 0.0, 3),
            "frame_count": self._frame_count,
        }


class MetricsLoggerComponent(AIVisionComponent):
    """Pipeline component that passes frames through and attaches metrics to payload."""

    def __init__(self, window: int = 30):
        super().__init__()
        self._logger = MetricsLogger(window=window)

    def _execute(self, data, config):
        self._logger.tick()
        latency = config.get("latency_ms")
        if latency is not None:
            self._logger.log_latency(latency)
        if isinstance(data, dict):
            data["metrics"] = self._logger.report()
            return data
        return {"frame": data, "metrics": self._logger.report()}
