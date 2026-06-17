from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def extract_frame(data):
    """Extracts the image array from a payload dict or returns the raw array.

    Args:
        data: NumPy array or payload dict containing a 'frame' key.

    Returns:
        numpy.ndarray: The image array.
    """
    return data["frame"] if isinstance(data, dict) else data


def replace_frame(data, frame):
    """Inserts a processed frame back into a payload dict or returns it directly.

    Args:
        data: Original input — either a NumPy array or a payload dict.
        frame (numpy.ndarray): Processed image to store.

    Returns:
        dict or numpy.ndarray: Updated payload dict with 'frame' replaced,
            or the raw frame if the input was not a dict.
    """
    if isinstance(data, dict):
        data["frame"] = frame
        return data
    return frame


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Clips pixel values to [0, 255] and converts to uint8.

    Args:
        frame (numpy.ndarray): Input image array of any numeric dtype.

    Returns:
        numpy.ndarray: Image as uint8. Returns input unchanged if already uint8.
    """
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame, 0, 255).astype(np.uint8)


def maybe_grayscale_to_bgr(frame: np.ndarray, keep_channels: bool) -> np.ndarray:
    """Converts a 2D grayscale array to 3-channel BGR if keep_channels is True.

    Args:
        frame (numpy.ndarray): Input image, either 2D (grayscale) or 3D.
        keep_channels (bool): If True, expand a grayscale frame to BGR.

    Returns:
        numpy.ndarray: BGR image if conversion applied, otherwise the original frame.
    """
    if keep_channels and frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def resolve_border_mode(name: str) -> int:
    """Maps a border mode name string to its OpenCV integer constant.

    Args:
        name (str): Border mode name. One of 'constant', 'replicate', 'reflect',
            'reflect_101', or 'wrap'. Case-insensitive.

    Returns:
        int: Corresponding cv2.BORDER_* constant. Defaults to cv2.BORDER_CONSTANT
            if the name is unrecognised.
    """
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "reflect_101": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP,
    }
    return border_modes.get(name.lower(), cv2.BORDER_CONSTANT)


def normalize_color_value(value):
    """Normalises a color value to a tuple if it is a non-string sequence.

    Args:
        value: A scalar, tuple, list, or other sequence representing a color.

    Returns:
        tuple or scalar: Tuple if value is a sequence, otherwise the original scalar.
    """
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(value)
    return value


def ensure_odd(value: int) -> int:
    """Ensures a kernel size value is a positive odd integer.

    Args:
        value (int): Desired kernel size.

    Returns:
        int: The input clamped to at least 1 and incremented by 1 if even.
    """
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def rotate_bound(frame: np.ndarray, angle: float, border_mode: int, border_value):
    """Rotates an image by the given angle, expanding the canvas to avoid cropping.

    Args:
        frame (numpy.ndarray): Input BGR image.
        angle (float): Counter-clockwise rotation angle in degrees.
        border_mode (int): OpenCV border mode constant for fill strategy.
        border_value: Fill color for constant border mode. Scalar or BGR tuple.

    Returns:
        numpy.ndarray: Rotated image on an expanded canvas.
    """
    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    bound_w = int((height * sin) + (width * cos))
    bound_h = int((height * cos) + (width * sin))

    matrix[0, 2] += (bound_w / 2) - center[0]
    matrix[1, 2] += (bound_h / 2) - center[1]

    return cv2.warpAffine(
        frame,
        matrix,
        (bound_w, bound_h),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=normalize_color_value(border_value),
    )
