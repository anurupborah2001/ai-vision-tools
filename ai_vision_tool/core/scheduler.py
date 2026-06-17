from __future__ import annotations

import threading
import time

from ai_vision_tool.core.base import AIVisionComponent


class RateLimiter:
    """Token bucket rate limiter. Thread-safe.

    Args:
        calls_per_second: Maximum call rate.
    """

    def __init__(self, calls_per_second: float):
        self._rate = calls_per_second
        self._tokens = calls_per_second
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, block: bool = True, timeout: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            if not block:
                return False
            wait = (1.0 - self._tokens) / self._rate
            if wait > timeout:
                return False
        time.sleep(wait)
        return self.acquire(block=False)


class Scheduler(AIVisionComponent):
    """Rate-limits frame throughput via token bucket.

    Args:
        target_fps: Target output frame rate.
        drop_policy: "skip" (mark frame skippable) or "block" (sleep to hit rate).
    """

    def __init__(self, target_fps: float = 30.0, drop_policy: str = "skip"):
        super().__init__()
        self.target_fps = target_fps
        self.drop_policy = drop_policy
        self._limiter: RateLimiter | None = None

    def setup(self, config: dict):
        fps = float(config.get("target_fps", self.target_fps))
        self._limiter = RateLimiter(fps)
        super().setup(config)

    def _execute(self, data, config):
        policy = config.get("drop_policy", self.drop_policy)
        block = policy == "block"
        acquired = self._limiter.acquire(block=block, timeout=2.0 / self.target_fps)

        payload = data if isinstance(data, dict) else {"frame": data}
        payload["scheduled_at"] = time.time()
        if not acquired:
            payload["skip"] = True
        return payload
