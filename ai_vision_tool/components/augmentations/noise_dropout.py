from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent
from .common import additive_noise


class ISONoise(AIVisionComponent):
    def __init__(self, color_shift=0.01, intensity=0.5):
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity

    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        intensity = float(config.get("intensity", self.intensity))
        color_shift = float(config.get("color_shift", self.color_shift))
        noise = np.random.normal(0, 255 * intensity * 0.05, frame.shape)
        chroma = np.random.normal(0, 255 * color_shift, frame.shape)
        output = to_uint8(frame + noise + chroma)
        return replace_frame(data, output)


class MultiplicativeNoise(AIVisionComponent):
    def __init__(self, multiplier_min=0.9, multiplier_max=1.1):
        super().__init__()
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max

    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        multiplier = np.random.uniform(
            float(config.get("multiplier_min", self.multiplier_min)),
            float(config.get("multiplier_max", self.multiplier_max)),
            size=frame.shape,
        )
        return replace_frame(data, to_uint8(frame * multiplier))


class SaltPepperNoise(AIVisionComponent):
    def __init__(self, amount=0.02, salt_vs_pepper=0.5):
        super().__init__()
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        amount = float(config.get("amount", self.amount))
        salt_vs_pepper = float(config.get("salt_vs_pepper", self.salt_vs_pepper))
        total_pixels = frame.shape[0] * frame.shape[1]
        salt = int(total_pixels * amount * salt_vs_pepper)
        pepper = int(total_pixels * amount * (1.0 - salt_vs_pepper))
        if salt > 0:
            coords = (np.random.randint(0, frame.shape[0], salt), np.random.randint(0, frame.shape[1], salt))
            frame[coords] = 255
        if pepper > 0:
            coords = (np.random.randint(0, frame.shape[0], pepper), np.random.randint(0, frame.shape[1], pepper))
            frame[coords] = 0
        return replace_frame(data, frame)


class CoarseDropout(AIVisionComponent):
    def __init__(self, holes=8, max_height=8, max_width=8, fill_value=0):
        super().__init__()
        self.holes = holes
        self.max_height = max_height
        self.max_width = max_width
        self.fill_value = fill_value

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        holes = int(config.get("holes", self.holes))
        max_height = int(config.get("max_height", self.max_height))
        max_width = int(config.get("max_width", self.max_width))
        fill_value = config.get("fill_value", self.fill_value)
        for _ in range(holes):
            h = np.random.randint(1, max_height + 1)
            w = np.random.randint(1, max_width + 1)
            y = np.random.randint(0, max(1, frame.shape[0] - h + 1))
            x = np.random.randint(0, max(1, frame.shape[1] - w + 1))
            frame[y : y + h, x : x + w] = fill_value
        return replace_frame(data, frame)


class GridDropout(AIVisionComponent):
    def __init__(self, ratio=0.5, unit_size=8, fill_value=0):
        super().__init__()
        self.ratio = ratio
        self.unit_size = unit_size
        self.fill_value = fill_value

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        ratio = float(config.get("ratio", self.ratio))
        unit_size = int(config.get("unit_size", self.unit_size))
        cut = max(1, int(unit_size * ratio))
        for y in range(0, frame.shape[0], unit_size):
            for x in range(0, frame.shape[1], unit_size):
                frame[y : y + cut, x : x + cut] = config.get("fill_value", self.fill_value)
        return replace_frame(data, frame)


class RandomErasing(AIVisionComponent):
    def __init__(self, scale=(0.02, 0.2), fill_value=0):
        super().__init__()
        self.scale = scale
        self.fill_value = fill_value

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        area = frame.shape[0] * frame.shape[1]
        scale_min, scale_max = config.get("scale", self.scale)
        erase_area = np.random.uniform(scale_min, scale_max) * area
        erase_w = int(np.sqrt(erase_area))
        erase_h = int(np.sqrt(erase_area))
        x = np.random.randint(0, max(1, frame.shape[1] - erase_w + 1))
        y = np.random.randint(0, max(1, frame.shape[0] - erase_h + 1))
        frame[y : y + erase_h, x : x + erase_w] = config.get("fill_value", self.fill_value)
        return replace_frame(data, frame)


class PixelDropout(AIVisionComponent):
    def __init__(self, dropout_prob=0.01, fill_value=0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.fill_value = fill_value

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        prob = float(config.get("dropout_prob", self.dropout_prob))
        mask = np.random.rand(*frame.shape[:2]) < prob
        frame[mask] = config.get("fill_value", self.fill_value)
        return replace_frame(data, frame)


class MaskDropout(AIVisionComponent):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob

    def _execute(self, data, config):
        if not isinstance(data, dict) or "mask" not in data:
            return data
        mask = data["mask"].copy()
        prob = float(config.get("dropout_prob", self.dropout_prob))
        drop = np.random.rand(*mask.shape[:2]) < prob
        mask[drop] = 0
        data["mask"] = mask
        return data
