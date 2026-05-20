from __future__ import annotations

import platform
import sys


class Device:
    """Hardware device abstraction for CPU/CUDA/MPS selection.

    Args:
        device: "auto", "cpu", "cuda", "cuda:N", or "mps".
    """

    _instance: Device | None = None

    def __init__(self, device: str = "auto"):
        self._requested = device
        self._resolved = self._resolve(device)

    @staticmethod
    def _resolve(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def device_str(self) -> str:
        return self._resolved

    @property
    def is_cuda(self) -> bool:
        return self._resolved.startswith("cuda")

    @property
    def is_mps(self) -> bool:
        return self._resolved == "mps"

    @property
    def is_cpu(self) -> bool:
        return self._resolved == "cpu"

    def to_torch(self):
        try:
            import torch
            return torch.device(self._resolved)
        except ImportError:
            raise ImportError("Install with: pip install torch")

    def to_cv_backend(self) -> int:
        import cv2
        sys_name = platform.system().lower()
        if sys_name == "linux":
            return cv2.CAP_V4L2
        if sys_name == "darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_DSHOW

    @classmethod
    def default(cls) -> Device:
        if cls._instance is None:
            cls._instance = cls("auto")
        return cls._instance

    def __repr__(self) -> str:
        return f"Device({self._resolved!r})"
