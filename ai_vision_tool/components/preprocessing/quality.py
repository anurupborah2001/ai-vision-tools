from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .._image_utils import extract_frame
from ..base import AIVisionComponent


class ImageQualityCheck(AIVisionComponent):
    """Runs a combined blur and brightness quality check and annotates the payload.

    Adds 'blur_score', 'is_blurry', 'brightness_mean', and 'brightness_ok' keys to
    the payload dict when the input is a dict.

    Args:
        blur_threshold (float): Laplacian variance below which the image is considered blurry. Default is 100.0.
        min_brightness (float): Minimum acceptable mean pixel value. Default is 40.0.
        max_brightness (float): Maximum acceptable mean pixel value. Default is 215.0.
    """

    def __init__(self, blur_threshold=100.0, min_brightness=40.0, max_brightness=215.0):
        """Initializes ImageQualityCheck with blur and brightness thresholds.

        Args:
            blur_threshold (float): Blur detection threshold. Default is 100.0.
            min_brightness (float): Minimum mean brightness. Default is 40.0.
            max_brightness (float): Maximum mean brightness. Default is 215.0.
        """
        super().__init__()
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def _execute(self, data, config):
        """Computes quality metrics and annotates the payload dict.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'blur_threshold',
                'min_brightness', 'max_brightness'.

        Returns:
            dict or numpy.ndarray: Payload with 'quality' key added if input was a dict,
                otherwise the original frame unchanged.
        """
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
    """Detects image blur using the Laplacian variance method.

    Adds 'blur_score' and 'is_blurry' keys to the payload dict.

    Args:
        threshold (float): Laplacian variance below which the image is classified as blurry. Default is 100.0.
    """

    def __init__(self, threshold=100.0):
        """Initializes BlurDetection with a blur threshold.

        Args:
            threshold (float): Variance threshold for blur detection. Default is 100.0.
        """
        super().__init__()
        self.threshold = threshold

    def _execute(self, data, config):
        """Computes blur score and annotates the payload dict.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'threshold'.

        Returns:
            dict or numpy.ndarray: Payload with 'blur_score' and 'is_blurry' added if dict,
                otherwise the original frame unchanged.
        """
        frame = extract_frame(data)
        score = _blur_score(frame)
        if isinstance(data, dict):
            data["blur_score"] = score
            data["is_blurry"] = score < float(config.get("threshold", self.threshold))
            return data
        return frame


class BrightnessCheck(AIVisionComponent):
    """Checks whether the mean image brightness falls within an acceptable range.

    Adds 'brightness_mean' and 'brightness_ok' keys to the payload dict.

    Args:
        min_brightness (float): Minimum acceptable mean pixel value. Default is 40.0.
        max_brightness (float): Maximum acceptable mean pixel value. Default is 215.0.
    """

    def __init__(self, min_brightness=40.0, max_brightness=215.0):
        """Initializes BrightnessCheck with acceptable brightness range.

        Args:
            min_brightness (float): Minimum mean brightness. Default is 40.0.
            max_brightness (float): Maximum mean brightness. Default is 215.0.
        """
        super().__init__()
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def _execute(self, data, config):
        """Computes mean brightness and annotates the payload dict.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'min_brightness' and 'max_brightness'.

        Returns:
            dict or numpy.ndarray: Payload with 'brightness_mean' and 'brightness_ok' added
                if dict, otherwise the original frame unchanged.
        """
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
    """Detects duplicate images by comparing average hash against a reference set.

    Adds 'image_hash' and 'is_duplicate' keys to the payload dict.

    Args:
        reference_hashes (list[str] or None): Known hashes to check against. Default is None (empty set).
    """

    def __init__(self, reference_hashes=None):
        """Initializes DuplicateImageCheck with a set of known image hashes.

        Args:
            reference_hashes (list[str] or None): Pre-computed hash strings. Default is None.
        """
        super().__init__()
        self.reference_hashes = set(reference_hashes or [])

    def _execute(self, data, config):
        """Computes the average hash and checks for duplicates.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'reference_hashes'.

        Returns:
            dict or numpy.ndarray: Payload with 'image_hash' and 'is_duplicate' added
                if dict, otherwise the original frame unchanged.
        """
        frame = extract_frame(data)
        reference_hashes = set(config.get("reference_hashes", self.reference_hashes))
        image_hash = _average_hash(frame)
        if isinstance(data, dict):
            data["image_hash"] = image_hash
            data["is_duplicate"] = image_hash in reference_hashes
            return data
        return frame


class CorruptImageCheck(AIVisionComponent):
    """Checks whether the image is a valid non-empty NumPy array.

    Adds 'is_corrupt' key to the payload dict.
    """

    def _execute(self, data, config):
        """Validates that the extracted frame is a non-empty NumPy array.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Unused. Present for interface compatibility.

        Returns:
            dict or numpy.ndarray: Payload with 'is_corrupt' added if dict,
                otherwise the original frame unchanged.
        """
        frame = extract_frame(data)
        corrupt = not isinstance(frame, np.ndarray) or frame.size == 0
        if isinstance(data, dict):
            data["is_corrupt"] = corrupt
            return data
        return frame


class AspectRatioFilter(AIVisionComponent):
    """Checks whether the image aspect ratio (width / height) falls within acceptable bounds.

    Adds 'aspect_ratio' and 'aspect_ratio_ok' keys to the payload dict.

    Args:
        min_ratio (float): Minimum acceptable aspect ratio. Default is 0.0.
        max_ratio (float): Maximum acceptable aspect ratio. Default is infinity.
    """

    def __init__(self, min_ratio=0.0, max_ratio=float("inf")):
        """Initializes AspectRatioFilter with min and max ratio bounds.

        Args:
            min_ratio (float): Lower bound. Default is 0.0.
            max_ratio (float): Upper bound. Default is infinity.
        """
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def _execute(self, data, config):
        """Computes the aspect ratio and checks bounds.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'min_ratio' and 'max_ratio'.

        Returns:
            dict or numpy.ndarray: Payload with 'aspect_ratio' and 'aspect_ratio_ok' added
                if dict, otherwise the original frame unchanged.
        """
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
    """Checks whether the image meets minimum width and height requirements.

    Adds 'min_size_ok' key to the payload dict.

    Args:
        min_width (int): Minimum required width in pixels. Default is 1.
        min_height (int): Minimum required height in pixels. Default is 1.
    """

    def __init__(self, min_width=1, min_height=1):
        """Initializes MinSizeFilter with minimum dimension thresholds.

        Args:
            min_width (int): Minimum width. Default is 1.
            min_height (int): Minimum height. Default is 1.
        """
        super().__init__()
        self.min_width = min_width
        self.min_height = min_height

    def _execute(self, data, config):
        """Checks if the frame meets minimum size requirements.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'min_width' and 'min_height'.

        Returns:
            dict or numpy.ndarray: Payload with 'min_size_ok' added if dict,
                otherwise the original frame unchanged.
        """
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        if isinstance(data, dict):
            data["min_size_ok"] = width >= int(config.get("min_width", self.min_width)) and height >= int(
                config.get("min_height", self.min_height)
            )
            return data
        return frame


class MaxSizeFilter(AIVisionComponent):
    """Checks whether the image does not exceed maximum width and height limits.

    Adds 'max_size_ok' key to the payload dict.

    Args:
        max_width (int): Maximum allowed width in pixels. Default is 4096.
        max_height (int): Maximum allowed height in pixels. Default is 4096.
    """

    def __init__(self, max_width=4096, max_height=4096):
        """Initializes MaxSizeFilter with maximum dimension limits.

        Args:
            max_width (int): Maximum width. Default is 4096.
            max_height (int): Maximum height. Default is 4096.
        """
        super().__init__()
        self.max_width = max_width
        self.max_height = max_height

    def _execute(self, data, config):
        """Checks if the frame is within the maximum size limits.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'max_width' and 'max_height'.

        Returns:
            dict or numpy.ndarray: Payload with 'max_size_ok' added if dict,
                otherwise the original frame unchanged.
        """
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        if isinstance(data, dict):
            data["max_size_ok"] = width <= int(config.get("max_width", self.max_width)) and height <= int(
                config.get("max_height", self.max_height)
            )
            return data
        return frame


def _blur_score(frame):
    """Computes an image sharpness score using the Laplacian variance method.

    Higher scores indicate sharper images.

    Args:
        frame (numpy.ndarray): Input image (BGR or grayscale).

    Returns:
        float: Variance of the Laplacian response.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _average_hash(frame, hash_size=8):
    """Computes an average hash fingerprint for duplicate detection.

    Args:
        frame (numpy.ndarray): Input image (BGR or grayscale).
        hash_size (int): Dimension of the downscaled hash image. Default is 8.

    Returns:
        str: SHA-256 hex digest of the binary hash bit string.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_value = resized.mean()
    bits = "".join("1" if pixel > mean_value else "0" for pixel in resized.flatten())
    return hashlib.sha256(bits.encode("utf-8")).hexdigest()
