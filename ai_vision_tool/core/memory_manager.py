from __future__ import annotations

import queue
from contextlib import contextmanager

import numpy as np


class MemoryManager:
    """Pre-allocated numpy frame buffer pool to minimize allocation overhead.

    Args:
        pool_size: Number of pre-allocated frame buffers.
        shape: Frame shape (H, W, C).
        dtype: Frame dtype.
    """

    def __init__(
        self, pool_size: int = 10, shape: tuple = (720, 1280, 3), dtype=np.uint8
    ):
        self._pool: queue.Queue = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            self._pool.put(np.empty(shape, dtype=dtype))

    def acquire(self, timeout: float = 1.0) -> np.ndarray:
        try:
            return self._pool.get(timeout=timeout)
        except queue.Empty as e:
            raise RuntimeError(
                "MemoryManager pool exhausted — increase pool_size or release frames faster"
            ) from e

    def release(self, frame: np.ndarray) -> None:
        try:
            self._pool.put_nowait(frame)
        except queue.Full:
            pass

    @contextmanager
    def context(self):
        frame = self.acquire()
        try:
            yield frame
        finally:
            self.release(frame)

    @property
    def available(self) -> int:
        return self._pool.qsize()


class GPUMemoryTracker:
    """Tracks GPU memory usage via PyTorch. No-op if torch/CUDA unavailable."""

    def __init__(self):
        try:
            import torch

            self._torch = torch
            self._available = torch.cuda.is_available()
        except ImportError:
            self._torch = None
            self._available = False

    def used_mb(self) -> float:
        if not self._available:
            return 0.0
        return self._torch.cuda.memory_allocated() / 1024**2

    def free_mb(self) -> float:
        if not self._available:
            return 0.0
        props = self._torch.cuda.get_device_properties(0)
        return (props.total_memory - self._torch.cuda.memory_allocated()) / 1024**2

    def log_stats(self) -> None:
        if not self._available:
            print("[GPUMemoryTracker] CUDA not available")
            return
        print(
            f"[GPUMemoryTracker] Used: {self.used_mb():.1f} MB | Free: {self.free_mb():.1f} MB"
        )
