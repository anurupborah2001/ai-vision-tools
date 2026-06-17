from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from ..utils.image_utils import normalize_color_value, resolve_border_mode, to_uint8


def random_uniform(config, key_min, key_max, fallback_min, fallback_max):
    """Samples a uniform random float from a range defined in config or fallback values.

    Args:
        config (dict): Runtime config that may contain explicit min/max values.
        key_min (str): Config key for the minimum bound.
        key_max (str): Config key for the maximum bound.
        fallback_min (float): Default minimum if key_min not in config.
        fallback_max (float): Default maximum if key_max not in config.

    Returns:
        float: Random value in [min_value, max_value].
    """
    min_value = float(config.get(key_min, fallback_min))
    max_value = float(config.get(key_max, fallback_max))
    return random.uniform(min_value, max_value)


def maybe_get_partner_frame(data, config, key):
    """Retrieves a partner image from config or payload dict.

    Args:
        data: Current pipeline data (NumPy array or dict).
        config (dict): Runtime config that may contain the partner frame under key.
        key (str): Key to look up in config first, then in data if data is a dict.

    Returns:
        NumPy array or None: Partner frame if found, otherwise None.
    """
    if key in config:
        return config[key]
    if isinstance(data, dict):
        return data.get(key)
    return None


def apply_affine(
    frame, matrix, output_size=None, border_mode="constant", border_value=0
):
    """Applies an affine transformation matrix to a frame.

    Args:
        frame (numpy.ndarray): Input image to transform.
        matrix (numpy.ndarray): 2x3 affine transformation matrix.
        output_size (tuple[int, int] or None): (width, height) of output. Defaults to input size.
        border_mode (str): Border fill mode ('constant', 'reflect', 'replicate', etc.). Default is 'constant'.
        border_value (int or tuple): Fill value for constant border mode. Default is 0.

    Returns:
        numpy.ndarray: Affine-transformed image.
    """
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
    """Builds a 2x3 affine translation matrix.

    Args:
        translate_x (float): Horizontal translation in pixels.
        translate_y (float): Vertical translation in pixels.

    Returns:
        numpy.ndarray: 2x3 float32 affine translation matrix.
    """
    return np.array(
        [[1.0, 0.0, translate_x], [0.0, 1.0, translate_y]], dtype=np.float32
    )


def apply_perspective(
    frame, src_points, dst_points, border_mode="constant", border_value=0
):
    """Applies a perspective (homography) transformation to a frame.

    Args:
        frame (numpy.ndarray): Input image to transform.
        src_points (list[list[float]]): Four source corner points as [[x, y], ...].
        dst_points (list[list[float]]): Four destination corner points as [[x, y], ...].
        border_mode (str): Border fill mode. Default is 'constant'.
        border_value (int or tuple): Fill value for constant border mode. Default is 0.

    Returns:
        numpy.ndarray: Perspective-transformed image at the same spatial size as input.
    """
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
    """Adds a noise array to a frame and clips the result to uint8 range.

    Args:
        frame (numpy.ndarray): Input image.
        noise (numpy.ndarray): Noise array with the same shape as frame.

    Returns:
        numpy.ndarray: Noisy image clipped to [0, 255] as uint8.
    """
    return to_uint8(frame.astype(np.float32) + noise)


def ensure_color(frame):
    """Converts a grayscale frame to BGR if it is single-channel.

    Args:
        frame (numpy.ndarray): Input image. May be 2D (grayscale) or 3D (color).

    Returns:
        numpy.ndarray: BGR image. Returns input unchanged if already 3-channel.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def parse_component_profile(path):
    """Loads a JSON component profile from disk.

    Args:
        path (str or Path): Path to the JSON profile file.

    Returns:
        dict: Parsed JSON content as a Python dict.
    """
    with open(Path(path), encoding="utf-8") as handle:
        return json.load(handle)
