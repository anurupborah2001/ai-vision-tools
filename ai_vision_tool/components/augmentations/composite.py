from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent
from .common import maybe_get_partner_frame


class MixUp(AIVisionComponent):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def _execute(self, data, config):
        frame = extract_frame(data)
        mix_image = maybe_get_partner_frame(data, config, "mix_image")
        if mix_image is None:
            return replace_frame(data, frame.copy())
        alpha = float(config.get("alpha", self.alpha))
        mixed = cv2.resize(mix_image, (frame.shape[1], frame.shape[0]))
        output = cv2.addWeighted(frame, alpha, mixed, 1.0 - alpha, 0)
        return replace_frame(data, output)


class CutMix(AIVisionComponent):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        mix_image = maybe_get_partner_frame(data, config, "mix_image")
        if mix_image is None:
            return replace_frame(data, frame)
        mix_image = cv2.resize(mix_image, (frame.shape[1], frame.shape[0]))
        height, width = frame.shape[:2]
        cut_w = max(1, int(width * float(config.get("alpha", self.alpha)) * 0.5))
        cut_h = max(1, int(height * float(config.get("alpha", self.alpha)) * 0.5))
        x = np.random.randint(0, max(1, width - cut_w + 1))
        y = np.random.randint(0, max(1, height - cut_h + 1))
        frame[y : y + cut_h, x : x + cut_w] = mix_image[y : y + cut_h, x : x + cut_w]
        return replace_frame(data, frame)


class CopyPaste(AIVisionComponent):
    def __init__(self, x=0, y=0):
        super().__init__()
        self.x = x
        self.y = y

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        overlay = maybe_get_partner_frame(data, config, "overlay_image")
        if overlay is None:
            return replace_frame(data, frame)
        x = int(config.get("x", self.x))
        y = int(config.get("y", self.y))
        overlay_h, overlay_w = overlay.shape[:2]
        x2 = min(frame.shape[1], x + overlay_w)
        y2 = min(frame.shape[0], y + overlay_h)
        frame[y:y2, x:x2] = overlay[: y2 - y, : x2 - x]
        return replace_frame(data, frame)


class RandomOcclusion(AIVisionComponent):
    def __init__(self, max_width=20, max_height=20, fill_value=0):
        super().__init__()
        self.max_width = max_width
        self.max_height = max_height
        self.fill_value = fill_value

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        width = np.random.randint(1, int(config.get("max_width", self.max_width)) + 1)
        height = np.random.randint(1, int(config.get("max_height", self.max_height)) + 1)
        x = np.random.randint(0, max(1, frame.shape[1] - width + 1))
        y = np.random.randint(0, max(1, frame.shape[0] - height + 1))
        frame[y : y + height, x : x + width] = config.get("fill_value", self.fill_value)
        return replace_frame(data, frame)


class ObjectPaste(AIVisionComponent):
    def __init__(self, x=0, y=0):
        super().__init__()
        self.x = x
        self.y = y

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        obj = maybe_get_partner_frame(data, config, "object_image")
        if obj is None:
            return replace_frame(data, frame)
        x = int(config.get("x", self.x))
        y = int(config.get("y", self.y))
        obj_h, obj_w = obj.shape[:2]
        x2 = min(frame.shape[1], x + obj_w)
        y2 = min(frame.shape[0], y + obj_h)
        frame[y:y2, x:x2] = obj[: y2 - y, : x2 - x]
        return replace_frame(data, frame)


class BoundingBoxJitter(AIVisionComponent):
    def __init__(self, x_jitter=0.05, y_jitter=0.05, size_jitter=0.1):
        super().__init__()
        self.x_jitter = x_jitter
        self.y_jitter = y_jitter
        self.size_jitter = size_jitter

    def _execute(self, data, config):
        if not isinstance(data, dict):
            return data
        bboxes = data.get("bboxes", config.get("bboxes", []))
        jittered = []
        for x, y, w, h in bboxes:
            jx = w * float(config.get("x_jitter", self.x_jitter))
            jy = h * float(config.get("y_jitter", self.y_jitter))
            js = float(config.get("size_jitter", self.size_jitter))
            nx = x + np.random.uniform(-jx, jx)
            ny = y + np.random.uniform(-jy, jy)
            nw = w * (1 + np.random.uniform(-js, js))
            nh = h * (1 + np.random.uniform(-js, js))
            jittered.append((nx, ny, nw, nh))
        data["bboxes"] = jittered
        return data


class Mosaic9(AIVisionComponent):
    def __init__(self, mosaic_images=None, output_size=None):
        super().__init__()
        self.mosaic_images = mosaic_images or []
        self.output_size = output_size

    def _execute(self, data, config):
        frame = extract_frame(data)
        images = [frame]
        images.extend(config.get("mosaic_images", self.mosaic_images)[:8])
        while len(images) < 9:
            images.append(frame)
        height, width = frame.shape[:2]
        output_size = config.get("output_size", self.output_size) or (width * 3, height * 3)
        cell_w = output_size[0] // 3
        cell_h = output_size[1] // 3
        rows = []
        for row in range(3):
            row_images = [
                cv2.resize(images[row * 3 + col], (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
                for col in range(3)
            ]
            rows.append(np.hstack(row_images))
        output = np.vstack(rows)
        return replace_frame(data, output)
