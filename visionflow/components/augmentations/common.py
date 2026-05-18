from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from .._image_utils import normalize_color_value, resolve_border_mode, to_uint8


def random_uniform(config, key_min, key_max, fallback_min, fallback_max):
    min_value = float(config.get(key_min, fallback_min))
    max_value = float(config.get(key_max, fallback_max))
    return random.uniform(min_value, max_value)


def maybe_get_partner_frame(data, config, key):
    if key in config:
        return config[key]
    if isinstance(data, dict):
        return data.get(key)
    return None


def apply_affine(frame, matrix, output_size=None, border_mode="constant", border_value=0):
    height, width = frame.shape[:2]
    out_size = output_size or (width, height)
    return cv2.warpAffine(
        frame,
        matrix,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderMode=resolve_border_mode(border_mode),
        borderValue=normalize_color_value(border_value),
    )


def build_translation_matrix(translate_x, translate_y):
    return np.array([[1.0, 0.0, translate_x], [0.0, 1.0, translate_y]], dtype=np.float32)


def apply_perspective(frame, src_points, dst_points, border_mode="constant", border_value=0):
    height, width = frame.shape[:2]
    matrix = cv2.getPerspectiveTransform(
        np.array(src_points, dtype=np.float32),
        np.array(dst_points, dtype=np.float32),
    )
    return cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        borderMode=resolve_border_mode(border_mode),
        borderValue=normalize_color_value(border_value),
    )


def additive_noise(frame, noise):
    return to_uint8(frame.astype(np.float32) + noise)


def ensure_color(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def parse_component_profile(path):
    with open(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)
