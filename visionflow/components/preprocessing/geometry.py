from __future__ import annotations

import math

import cv2
import numpy as np

from .._image_utils import (
    extract_frame,
    normalize_color_value,
    replace_frame,
    resolve_border_mode,
    rotate_bound,
)
from ..base import AIVisionComponent


def _get_bboxes(data, config):
    return config.get("bboxes", data.get("bboxes", [])) if isinstance(data, dict) else config.get("bboxes", [])


class Resize(AIVisionComponent):
    def __init__(self, width, height, interpolation="linear"):
        super().__init__()
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def _execute(self, data, config):
        frame = extract_frame(data)
        width = int(config.get("resize_width", config.get("width", self.width)))
        height = int(config.get("resize_height", config.get("height", self.height)))
        interpolation = _resolve_interpolation(
            config.get("resize_interpolation", config.get("interpolation", self.interpolation))
        )
        output = cv2.resize(frame, (width, height), interpolation=interpolation)
        return replace_frame(data, output)


class LetterboxResize(AIVisionComponent):
    def __init__(
        self,
        width,
        height,
        pad_value=(0, 0, 0),
        interpolation="linear",
        center=True,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.pad_value = pad_value
        self.interpolation = interpolation
        self.center = center

    def _execute(self, data, config):
        frame = extract_frame(data)
        target_w = int(config.get("letterbox_width", config.get("width", self.width)))
        target_h = int(config.get("letterbox_height", config.get("height", self.height)))
        pad_value = normalize_color_value(config.get("pad_value", self.pad_value))
        interpolation = _resolve_interpolation(config.get("interpolation", self.interpolation))
        center = config.get("center", self.center)

        height, width = frame.shape[:2]
        scale = min(target_w / width, target_h / height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

        canvas = np.full((target_h, target_w, frame.shape[2]), pad_value, dtype=frame.dtype)
        if center:
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
        else:
            x = 0
            y = 0

        canvas[y : y + new_h, x : x + new_w] = resized
        return replace_frame(data, canvas)


class CenterCrop(AIVisionComponent):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height

    def _execute(self, data, config):
        frame = extract_frame(data)
        crop_w = int(config.get("center_crop_width", config.get("width", self.width)))
        crop_h = int(config.get("center_crop_height", config.get("height", self.height)))
        height, width = frame.shape[:2]
        crop_w = min(crop_w, width)
        crop_h = min(crop_h, height)
        x = max(0, (width - crop_w) // 2)
        y = max(0, (height - crop_h) // 2)
        output = frame[y : y + crop_h, x : x + crop_w].copy()
        return replace_frame(data, output)


class PadToSquare(AIVisionComponent):
    def __init__(self, pad_value=(0, 0, 0), size=None):
        super().__init__()
        self.pad_value = pad_value
        self.size = size

    def _execute(self, data, config):
        frame = extract_frame(data)
        target = config.get("square_size", self.size)
        pad_value = normalize_color_value(config.get("pad_value", self.pad_value))
        height, width = frame.shape[:2]
        side = max(height, width) if target is None else int(target)
        canvas = np.full((side, side, frame.shape[2]), pad_value, dtype=frame.dtype)
        x = (side - width) // 2
        y = (side - height) // 2
        canvas[y : y + height, x : x + width] = frame
        return replace_frame(data, canvas)


class PerspectiveCorrection(AIVisionComponent):
    def __init__(self, source_points=None, output_size=None, border_mode="constant", border_value=0):
        super().__init__()
        self.source_points = source_points
        self.output_size = output_size
        self.border_mode = border_mode
        self.border_value = border_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        src_points = np.array(
            config.get("source_points", self.source_points),
            dtype=np.float32,
        )
        if src_points.shape != (4, 2):
            raise ValueError("source_points must contain four (x, y) pairs.")

        output_size = config.get("output_size", self.output_size)
        if output_size is None:
            width_a = np.linalg.norm(src_points[2] - src_points[3])
            width_b = np.linalg.norm(src_points[1] - src_points[0])
            height_a = np.linalg.norm(src_points[1] - src_points[2])
            height_b = np.linalg.norm(src_points[0] - src_points[3])
            output_size = (int(max(width_a, width_b)), int(max(height_a, height_b)))

        out_w, out_h = output_size
        dst_points = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        output = cv2.warpPerspective(
            frame,
            matrix,
            (out_w, out_h),
            borderMode=resolve_border_mode(config.get("border_mode", self.border_mode)),
            borderValue=normalize_color_value(config.get("border_value", self.border_value)),
        )
        return replace_frame(data, output)


class Deskew(AIVisionComponent):
    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = threshold

    def _execute(self, data, config):
        frame = extract_frame(data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        thresh = int(config.get("deskew_threshold", config.get("threshold", self.threshold)))
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        points = cv2.findNonZero(binary)
        if points is None:
            return replace_frame(data, frame.copy())

        angle = cv2.minAreaRect(points)[-1]
        if angle < -45:
            angle = 90 + angle
        output = rotate_bound(frame, angle, cv2.BORDER_REPLICATE, 0)
        return replace_frame(data, output)


class AutoCrop(AIVisionComponent):
    def __init__(self, threshold=0, padding=0):
        super().__init__()
        self.threshold = threshold
        self.padding = padding

    def _execute(self, data, config):
        frame = extract_frame(data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        threshold = int(config.get("auto_crop_threshold", config.get("threshold", self.threshold)))
        padding = int(config.get("padding", self.padding))
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return replace_frame(data, frame.copy())

        x, y, w, h = cv2.boundingRect(coords)
        x = max(0, x - padding)
        y = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + (padding * 2))
        y2 = min(frame.shape[0], y + h + (padding * 2))
        output = frame[y:y2, x:x2].copy()
        return replace_frame(data, output)


class FaceAlign(AIVisionComponent):
    def __init__(
        self,
        left_eye=None,
        right_eye=None,
        desired_left_eye=(0.35, 0.35),
        output_size=(112, 112),
    ):
        super().__init__()
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.desired_left_eye = desired_left_eye
        self.output_size = output_size

    def _execute(self, data, config):
        frame = extract_frame(data)
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        left_eye = config.get("left_eye", self.left_eye or metadata.get("left_eye"))
        right_eye = config.get("right_eye", self.right_eye or metadata.get("right_eye"))
        if left_eye is None or right_eye is None:
            return replace_frame(data, frame.copy())

        output_size = tuple(config.get("output_size", self.output_size))
        desired_left_eye = tuple(config.get("desired_left_eye", self.desired_left_eye))
        desired_face_width, desired_face_height = output_size

        d_y = right_eye[1] - left_eye[1]
        d_x = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(d_y, d_x))

        desired_right_eye_x = 1.0 - desired_left_eye[0]
        dist = math.hypot(d_x, d_y)
        desired_dist = (desired_right_eye_x - desired_left_eye[0]) * desired_face_width
        scale = desired_dist / max(dist, 1e-6)

        eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        t_x = desired_face_width * 0.5
        t_y = desired_face_height * desired_left_eye[1]
        matrix[0, 2] += t_x - eyes_center[0]
        matrix[1, 2] += t_y - eyes_center[1]

        output = cv2.warpAffine(frame, matrix, output_size, flags=cv2.INTER_CUBIC)
        return replace_frame(data, output)


class ObjectCrop(AIVisionComponent):
    def __init__(self, bbox=None, bbox_index=0, clamp=True):
        super().__init__()
        self.bbox = bbox
        self.bbox_index = bbox_index
        self.clamp = clamp

    def _execute(self, data, config):
        frame = extract_frame(data)
        bbox = config.get("bbox", self.bbox)
        if bbox is None and isinstance(data, dict):
            bboxes = _get_bboxes(data, config)
            if bboxes:
                bbox = bboxes[int(config.get("bbox_index", self.bbox_index))]

        if bbox is None:
            return replace_frame(data, frame.copy())

        x, y, w, h = map(int, bbox)
        if config.get("clamp", self.clamp):
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

        output = frame[y : y + h, x : x + w].copy()
        return replace_frame(data, output)


class BoundingBoxClamp(AIVisionComponent):
    def _execute(self, data, config):
        frame = extract_frame(data)
        if not isinstance(data, dict):
            return data

        clamped = []
        height, width = frame.shape[:2]
        for bbox in _get_bboxes(data, config):
            x, y, w, h = map(float, bbox)
            x = min(max(0.0, x), width)
            y = min(max(0.0, y), height)
            w = max(0.0, min(w, width - x))
            h = max(0.0, min(h, height - y))
            clamped.append((x, y, w, h))

        data["bboxes"] = clamped
        return data


class BoundingBoxNormalize(AIVisionComponent):
    def __init__(self, mode="xywh", inverse=False):
        super().__init__()
        self.mode = mode
        self.inverse = inverse

    def _execute(self, data, config):
        frame = extract_frame(data)
        if not isinstance(data, dict):
            return data

        mode = config.get("bbox_mode", self.mode)
        inverse = config.get("inverse", self.inverse)
        height, width = frame.shape[:2]

        normalized = []
        for bbox in _get_bboxes(data, config):
            if mode != "xywh":
                raise ValueError("Only xywh mode is currently supported.")
            x, y, w, h = map(float, bbox)
            if inverse:
                normalized.append((x * width, y * height, w * width, h * height))
            else:
                normalized.append((x / width, y / height, w / width, h / height))

        data["bboxes"] = normalized
        return data


class MaskResize(AIVisionComponent):
    def __init__(self, width=None, height=None, interpolation="nearest"):
        super().__init__()
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def _execute(self, data, config):
        if not isinstance(data, dict) or "mask" not in data:
            return data

        mask = data["mask"]
        frame = extract_frame(data)
        width = config.get("mask_width", self.width or frame.shape[1])
        height = config.get("mask_height", self.height or frame.shape[0])
        interpolation = _resolve_interpolation(
            config.get("mask_interpolation", self.interpolation)
        )
        data["mask"] = cv2.resize(mask, (int(width), int(height)), interpolation=interpolation)
        return data


def _resolve_interpolation(name):
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return mapping.get(str(name).lower(), cv2.INTER_LINEAR)
