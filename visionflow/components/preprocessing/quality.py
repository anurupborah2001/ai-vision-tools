from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .._image_utils import extract_frame
from ..base import AIVisionComponent


class ImageQualityCheck(AIVisionComponent):
    def __init__(self, blur_threshold=100.0, min_brightness=40.0, max_brightness=215.0):
        super().__init__()
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def _execute(self, data, config):
        frame = extract_frame(data)
        result = {
            "blur_score": _blur_score(frame),
            "brightness_mean": float(frame.mean()),
        }
        result["is_blurry"] = result["blur_score"] < float(
            config.get("blur_threshold", self.blur_threshold)
        )
        result["brightness_ok"] = float(config.get("min_brightness", self.min_brightness)) <= result[
            "brightness_mean"
        ] <= float(config.get("max_brightness", self.max_brightness))

        if isinstance(data, dict):
            data["quality"] = result
            return data
        return frame


class BlurDetection(AIVisionComponent):
    def __init__(self, threshold=100.0):
        super().__init__()
        self.threshold = threshold

    def _execute(self, data, config):
        frame = extract_frame(data)
        score = _blur_score(frame)
        if isinstance(data, dict):
            data["blur_score"] = score
            data["is_blurry"] = score < float(config.get("threshold", self.threshold))
            return data
        return frame


class BrightnessCheck(AIVisionComponent):
    def __init__(self, min_brightness=40.0, max_brightness=215.0):
        super().__init__()
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def _execute(self, data, config):
        frame = extract_frame(data)
        mean_value = float(frame.mean())
        if isinstance(data, dict):
            data["brightness_mean"] = mean_value
            data["brightness_ok"] = float(config.get("min_brightness", self.min_brightness)) <= mean_value <= float(
                config.get("max_brightness", self.max_brightness)
            )
            return data
        return frame


class DuplicateImageCheck(AIVisionComponent):
    def __init__(self, reference_hashes=None):
        super().__init__()
        self.reference_hashes = set(reference_hashes or [])

    def _execute(self, data, config):
        frame = extract_frame(data)
        reference_hashes = set(config.get("reference_hashes", self.reference_hashes))
        image_hash = _average_hash(frame)
        if isinstance(data, dict):
            data["image_hash"] = image_hash
            data["is_duplicate"] = image_hash in reference_hashes
            return data
        return frame


class CorruptImageCheck(AIVisionComponent):
    def _execute(self, data, config):
        frame = extract_frame(data)
        corrupt = not isinstance(frame, np.ndarray) or frame.size == 0
        if isinstance(data, dict):
            data["is_corrupt"] = corrupt
            return data
        return frame


class AspectRatioFilter(AIVisionComponent):
    def __init__(self, min_ratio=0.0, max_ratio=float("inf")):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        ratio = width / max(height, 1)
        if isinstance(data, dict):
            data["aspect_ratio"] = ratio
            data["aspect_ratio_ok"] = float(config.get("min_ratio", self.min_ratio)) <= ratio <= float(
                config.get("max_ratio", self.max_ratio)
            )
            return data
        return frame


class MinSizeFilter(AIVisionComponent):
    def __init__(self, min_width=1, min_height=1):
        super().__init__()
        self.min_width = min_width
        self.min_height = min_height

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        if isinstance(data, dict):
            data["min_size_ok"] = width >= int(config.get("min_width", self.min_width)) and height >= int(
                config.get("min_height", self.min_height)
            )
            return data
        return frame


class MaxSizeFilter(AIVisionComponent):
    def __init__(self, max_width=4096, max_height=4096):
        super().__init__()
        self.max_width = max_width
        self.max_height = max_height

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        if isinstance(data, dict):
            data["max_size_ok"] = width <= int(config.get("max_width", self.max_width)) and height <= int(
                config.get("max_height", self.max_height)
            )
            return data
        return frame


def _blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _average_hash(frame, hash_size=8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_value = resized.mean()
    bits = "".join("1" if pixel > mean_value else "0" for pixel in resized.flatten())
    return hashlib.sha256(bits.encode("utf-8")).hexdigest()
