from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def extract_frame(data):
    return data["frame"] if isinstance(data, dict) else data


def replace_frame(data, frame):
    if isinstance(data, dict):
        data["frame"] = frame
        return data
    return frame


def to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    return np.clip(frame, 0, 255).astype(np.uint8)


def maybe_grayscale_to_bgr(frame: np.ndarray, keep_channels: bool) -> np.ndarray:
    if keep_channels and frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def resolve_border_mode(name: str) -> int:
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "reflect_101": cv2.BORDER_REFLECT_101,
        "wrap": cv2.BORDER_WRAP,
    }
    return border_modes.get(name.lower(), cv2.BORDER_CONSTANT)


def normalize_color_value(value):
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return value


def ensure_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def rotate_bound(frame: np.ndarray, angle: float, border_mode: int, border_value):
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
