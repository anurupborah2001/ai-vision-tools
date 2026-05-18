from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, normalize_color_value, replace_frame, resolve_border_mode
from ..base import AIVisionComponent
from .common import apply_affine, apply_perspective, build_translation_matrix, random_uniform


class RandomResize(AIVisionComponent):
    def __init__(self, min_width, max_width, min_height=None, max_height=None, interpolation="linear"):
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height if min_height is not None else min_width
        self.max_height = max_height if max_height is not None else max_width
        self.interpolation = interpolation

    def _execute(self, data, config):
        frame = extract_frame(data)
        width = int(random_uniform(config, "min_width", "max_width", self.min_width, self.max_width))
        height = int(
            random_uniform(config, "min_height", "max_height", self.min_height, self.max_height)
        )
        interpolation = _resolve_interpolation(config.get("interpolation", self.interpolation))
        return replace_frame(data, cv2.resize(frame, (width, height), interpolation=interpolation))


class RandomScale(AIVisionComponent):
    def __init__(self, min_scale=0.8, max_scale=1.2, interpolation="linear"):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.interpolation = interpolation

    def _execute(self, data, config):
        frame = extract_frame(data)
        scale = random_uniform(config, "min_scale", "max_scale", self.min_scale, self.max_scale)
        height, width = frame.shape[:2]
        out_w = max(1, int(round(width * scale)))
        out_h = max(1, int(round(height * scale)))
        interpolation = _resolve_interpolation(config.get("interpolation", self.interpolation))
        return replace_frame(data, cv2.resize(frame, (out_w, out_h), interpolation=interpolation))


class RandomCrop(AIVisionComponent):
    def __init__(self, crop_width, crop_height):
        super().__init__()
        self.crop_width = crop_width
        self.crop_height = crop_height

    def _execute(self, data, config):
        frame = extract_frame(data)
        crop_w = int(config.get("crop_width", self.crop_width))
        crop_h = int(config.get("crop_height", self.crop_height))
        height, width = frame.shape[:2]
        crop_w = min(crop_w, width)
        crop_h = min(crop_h, height)
        x = np.random.randint(0, max(1, width - crop_w + 1))
        y = np.random.randint(0, max(1, height - crop_h + 1))
        return replace_frame(data, frame[y : y + crop_h, x : x + crop_w].copy())


class RandomResizedCrop(AIVisionComponent):
    def __init__(
        self,
        output_width,
        output_height,
        scale_min=0.08,
        scale_max=1.0,
        ratio_min=0.75,
        ratio_max=1.3333,
    ):
        super().__init__()
        self.output_width = output_width
        self.output_height = output_height
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        area = width * height
        scale = random_uniform(config, "scale_min", "scale_max", self.scale_min, self.scale_max)
        ratio = random_uniform(config, "ratio_min", "ratio_max", self.ratio_min, self.ratio_max)
        target_area = area * scale
        crop_w = int(round((target_area * ratio) ** 0.5))
        crop_h = int(round((target_area / ratio) ** 0.5))
        crop_w = min(max(1, crop_w), width)
        crop_h = min(max(1, crop_h), height)
        x = np.random.randint(0, max(1, width - crop_w + 1))
        y = np.random.randint(0, max(1, height - crop_h + 1))
        cropped = frame[y : y + crop_h, x : x + crop_w]
        output = cv2.resize(
            cropped,
            (int(config.get("output_width", self.output_width)), int(config.get("output_height", self.output_height))),
            interpolation=cv2.INTER_LINEAR,
        )
        return replace_frame(data, output)


class RandomPadding(AIVisionComponent):
    def __init__(self, max_top=10, max_bottom=10, max_left=10, max_right=10, pad_value=(0, 0, 0)):
        super().__init__()
        self.max_top = max_top
        self.max_bottom = max_bottom
        self.max_left = max_left
        self.max_right = max_right
        self.pad_value = pad_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        top = np.random.randint(0, int(config.get("max_top", self.max_top)) + 1)
        bottom = np.random.randint(0, int(config.get("max_bottom", self.max_bottom)) + 1)
        left = np.random.randint(0, int(config.get("max_left", self.max_left)) + 1)
        right = np.random.randint(0, int(config.get("max_right", self.max_right)) + 1)
        output = cv2.copyMakeBorder(
            frame,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=normalize_color_value(config.get("pad_value", self.pad_value)),
        )
        return replace_frame(data, output)


class Translate(AIVisionComponent):
    def __init__(self, shift_x=0, shift_y=0, border_mode="constant", border_value=0):
        super().__init__()
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.border_mode = border_mode
        self.border_value = border_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        matrix = build_translation_matrix(
            float(config.get("shift_x", self.shift_x)),
            float(config.get("shift_y", self.shift_y)),
        )
        output = apply_affine(
            frame,
            matrix,
            border_mode=config.get("border_mode", self.border_mode),
            border_value=config.get("border_value", self.border_value),
        )
        return replace_frame(data, output)


class AffineTransform(AIVisionComponent):
    def __init__(
        self,
        angle=0.0,
        scale=1.0,
        translate_x=0.0,
        translate_y=0.0,
        shear_x=0.0,
        shear_y=0.0,
        border_mode="constant",
        border_value=0,
    ):
        super().__init__()
        self.angle = angle
        self.scale = scale
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.shear_x = shear_x
        self.shear_y = shear_y
        self.border_mode = border_mode
        self.border_value = border_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(
            center,
            float(config.get("angle", self.angle)),
            float(config.get("scale", self.scale)),
        )
        matrix[0, 1] += float(config.get("shear_x", self.shear_x))
        matrix[1, 0] += float(config.get("shear_y", self.shear_y))
        matrix[0, 2] += float(config.get("translate_x", self.translate_x))
        matrix[1, 2] += float(config.get("translate_y", self.translate_y))
        output = apply_affine(
            frame,
            matrix,
            border_mode=config.get("border_mode", self.border_mode),
            border_value=config.get("border_value", self.border_value),
        )
        return replace_frame(data, output)


class PerspectiveTransform(AIVisionComponent):
    def __init__(self, distortion_scale=0.15, border_mode="constant", border_value=0):
        super().__init__()
        self.distortion_scale = distortion_scale
        self.border_mode = border_mode
        self.border_value = border_value

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        scale = float(config.get("distortion_scale", self.distortion_scale))
        dx = width * scale
        dy = height * scale
        src = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        dst = [
            [np.random.uniform(0, dx), np.random.uniform(0, dy)],
            [width - 1 - np.random.uniform(0, dx), np.random.uniform(0, dy)],
            [width - 1 - np.random.uniform(0, dx), height - 1 - np.random.uniform(0, dy)],
            [np.random.uniform(0, dx), height - 1 - np.random.uniform(0, dy)],
        ]
        output = apply_perspective(
            frame,
            src,
            dst,
            border_mode=config.get("border_mode", self.border_mode),
            border_value=config.get("border_value", self.border_value),
        )
        return replace_frame(data, output)


class ElasticTransform(AIVisionComponent):
    def __init__(self, alpha=20.0, sigma=4.0, alpha_affine=0.0):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine

    def _execute(self, data, config):
        frame = extract_frame(data)
        alpha = float(config.get("alpha", self.alpha))
        sigma = float(config.get("sigma", self.sigma))
        random_state = np.random.RandomState(config.get("seed", None))
        shape = frame.shape[:2]
        dx = cv2.GaussianBlur(
            (random_state.rand(*shape).astype(np.float32) * 2 - 1),
            (0, 0),
            sigma,
        ) * alpha
        dy = cv2.GaussianBlur(
            (random_state.rand(*shape).astype(np.float32) * 2 - 1),
            (0, 0),
            sigma,
        ) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        output = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return replace_frame(data, output)


class GridDistortion(AIVisionComponent):
    def __init__(self, num_steps=5, distort_limit=0.3):
        super().__init__()
        self.num_steps = num_steps
        self.distort_limit = distort_limit

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        steps = int(config.get("num_steps", self.num_steps))
        distort_limit = float(config.get("distort_limit", self.distort_limit))
        xsteps = [1 + np.random.uniform(-distort_limit, distort_limit) for _ in range(steps + 1)]
        ysteps = [1 + np.random.uniform(-distort_limit, distort_limit) for _ in range(steps + 1)]
        xx = np.zeros(width, np.float32)
        yy = np.zeros(height, np.float32)
        x_segment = width / steps
        y_segment = height / steps
        prev = 0
        prev_index = 0
        for idx in range(steps):
            cur = prev + x_segment * xsteps[idx]
            start = int(round(idx * x_segment))
            end = int(round((idx + 1) * x_segment))
            xx[start:end] = np.linspace(prev_index, cur, max(1, end - start), endpoint=False)
            prev_index = cur
            prev = cur
        xx[end:] = np.linspace(prev_index, width - 1, width - end)
        prev = 0
        prev_index = 0
        for idx in range(steps):
            cur = prev + y_segment * ysteps[idx]
            start = int(round(idx * y_segment))
            end = int(round((idx + 1) * y_segment))
            yy[start:end] = np.linspace(prev_index, cur, max(1, end - start), endpoint=False)
            prev_index = cur
            prev = cur
        yy[end:] = np.linspace(prev_index, height - 1, height - end)
        map_x, map_y = np.meshgrid(xx, yy)
        output = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return replace_frame(data, output)


class OpticalDistortion(AIVisionComponent):
    def __init__(self, k=0.00001, dx=0.0, dy=0.0):
        super().__init__()
        self.k = k
        self.dx = dx
        self.dy = dy

    def _execute(self, data, config):
        frame = extract_frame(data)
        height, width = frame.shape[:2]
        k = float(config.get("k", self.k))
        dx = float(config.get("dx", self.dx))
        dy = float(config.get("dy", self.dy))
        fx = width
        fy = height
        cx = width / 2.0 + dx
        cy = height / 2.0 + dy
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist = np.array([k, k, 0, 0], dtype=np.float32)
        output = cv2.undistort(frame, camera_matrix, dist)
        return replace_frame(data, output)


def _resolve_interpolation(name):
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return mapping.get(str(name).lower(), cv2.INTER_LINEAR)
