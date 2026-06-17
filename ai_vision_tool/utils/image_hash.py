from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class ImageHash(AIVisionComponent):
    """Perceptual hash component for deduplication and frame similarity detection.

    Args:
        method: "phash" (DCT-based), "ahash" (average), or "dhash" (difference).
        hash_size: Hash grid dimension (default 8 → 64-bit hash).
        threshold: Hamming distance below which frames are marked as duplicates.
    """

    def __init__(self, method: str = "phash", hash_size: int = 8, threshold: int = 10):
        super().__init__()
        self.method = method
        self.hash_size = hash_size
        self.threshold = threshold
        self._prev_hash: str | None = None
        self._seen: set[str] = set()

    def _phash(self, gray: np.ndarray) -> str:
        size = self.hash_size * 4
        img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        dct = cv2.dct(img)
        dct_low = dct[: self.hash_size, : self.hash_size]
        median = np.median(dct_low)
        bits = (dct_low > median).flatten()
        val = int(np.packbits(bits).tobytes().hex(), 16)
        return format(val, f"0{self.hash_size * self.hash_size // 4}x")

    def _ahash(self, gray: np.ndarray) -> str:
        img = cv2.resize(
            gray, (self.hash_size, self.hash_size), interpolation=cv2.INTER_AREA
        )
        avg = img.mean()
        bits = img.flatten() > avg
        val = int(np.packbits(bits).tobytes().hex(), 16)
        return format(val, f"0{self.hash_size * self.hash_size // 4}x")

    def _dhash(self, gray: np.ndarray) -> str:
        img = cv2.resize(
            gray, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA
        )
        bits = (img[:, :-1] > img[:, 1:]).flatten()
        val = int(np.packbits(bits).tobytes().hex(), 16)
        return format(val, f"0{self.hash_size * self.hash_size // 4}x")

    @staticmethod
    def _hamming(a: str, b: str) -> int:
        x = int(a, 16) ^ int(b, 16)
        return bin(x).count("1")

    def _compute(self, gray: np.ndarray) -> str:
        if self.method == "phash":
            return self._phash(gray)
        if self.method == "ahash":
            return self._ahash(gray)
        return self._dhash(gray)

    def _execute(self, data, config):
        frame = extract_frame(data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h = self._compute(gray)
        distance = self._hamming(h, self._prev_hash) if self._prev_hash else None
        is_dup = (h in self._seen) or (
            distance is not None and distance < self.threshold
        )
        self._seen.add(h)
        self._prev_hash = h
        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["hash"] = h
        payload["hash_distance"] = distance
        payload["is_duplicate"] = is_dup
        return payload
