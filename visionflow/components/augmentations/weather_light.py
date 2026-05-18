from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent
from .common import ensure_color, random_uniform


class RandomShadow(AIVisionComponent):
    def __init__(self, shadow_dimension=0.5, intensity=0.5):
        super().__init__()
        self.shadow_dimension = shadow_dimension
        self.intensity = intensity

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        height, width = frame.shape[:2]
        shadow_w = int(width * float(config.get("shadow_dimension", self.shadow_dimension)))
        x1 = np.random.randint(0, max(1, width - shadow_w))
        overlay = frame.copy()
        overlay[:, x1 : x1 + shadow_w] = (overlay[:, x1 : x1 + shadow_w] * float(config.get("intensity", self.intensity))).astype(frame.dtype)
        return replace_frame(data, overlay)


class RandomSunFlare(AIVisionComponent):
    def __init__(self, center=None, radius=20, intensity=0.4):
        super().__init__()
        self.center = center
        self.radius = radius
        self.intensity = intensity

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        height, width = frame.shape[:2]
        center = config.get("center", self.center) or (width // 2, height // 4)
        radius = int(config.get("radius", self.radius))
        intensity = float(config.get("intensity", self.intensity))
        overlay = frame.copy()
        cv2.circle(overlay, tuple(center), radius, (255, 255, 255), -1)
        output = cv2.addWeighted(overlay, intensity, frame, 1.0, 0.0)
        return replace_frame(data, output)


class RandomFog(AIVisionComponent):
    def __init__(self, alpha=0.35):
        super().__init__()
        self.alpha = alpha

    def _execute(self, data, config):
        frame = extract_frame(data)
        alpha = float(config.get("alpha", self.alpha))
        fog = np.full_like(frame, 255)
        output = cv2.addWeighted(frame, 1.0 - alpha, fog, alpha, 0)
        return replace_frame(data, output)


class RandomRain(AIVisionComponent):
    def __init__(self, drops=40, drop_length=12, intensity=0.25):
        super().__init__()
        self.drops = drops
        self.drop_length = drop_length
        self.intensity = intensity

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        overlay = frame.copy()
        height, width = frame.shape[:2]
        for _ in range(int(config.get("drops", self.drops))):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            length = int(config.get("drop_length", self.drop_length))
            cv2.line(overlay, (x, y), (x + 3, min(height - 1, y + length)), (180, 180, 180), 1)
        output = cv2.addWeighted(overlay, float(config.get("intensity", self.intensity)), frame, 1.0 - float(config.get("intensity", self.intensity)), 0)
        return replace_frame(data, output)


class RandomSnow(AIVisionComponent):
    def __init__(self, intensity=0.1):
        super().__init__()
        self.intensity = intensity

    def _execute(self, data, config):
        frame = extract_frame(data)
        intensity = float(config.get("intensity", self.intensity))
        noise = np.random.normal(loc=255 * intensity, scale=255 * intensity * 0.5, size=frame.shape)
        output = to_uint8(frame.astype(np.float32) + noise)
        return replace_frame(data, output)


class RandomGamma(AIVisionComponent):
    def __init__(self, min_gamma=0.7, max_gamma=1.5):
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def _execute(self, data, config):
        frame = extract_frame(data)
        gamma = random_uniform(config, "min_gamma", "max_gamma", self.min_gamma, self.max_gamma)
        table = np.array([((i / 255.0) ** (1.0 / max(gamma, 1e-6))) * 255 for i in range(256)], dtype=np.uint8)
        output = cv2.LUT(frame, table)
        return replace_frame(data, output)


class ColorJitter(AIVisionComponent):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=10):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        brightness = random_uniform(config, "brightness_min", "brightness_max", -self.brightness, self.brightness)
        contrast = random_uniform(config, "contrast_min", "contrast_max", 1 - self.contrast, 1 + self.contrast)
        saturation = random_uniform(config, "saturation_min", "saturation_max", 1 - self.saturation, 1 + self.saturation)
        hue_delta = random_uniform(config, "hue_min", "hue_max", -self.hue, self.hue)

        jittered = np.clip(frame * contrast + brightness * 255, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(jittered, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return replace_frame(data, output)


class ChannelShuffle(AIVisionComponent):
    def _execute(self, data, config):
        frame = extract_frame(data)
        order = np.random.permutation(3)
        output = frame[:, :, order]
        return replace_frame(data, output)


class RGBShift(AIVisionComponent):
    def __init__(self, r_shift=0, g_shift=0, b_shift=0):
        super().__init__()
        self.r_shift = r_shift
        self.g_shift = g_shift
        self.b_shift = b_shift

    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        shifts = np.array(
            [
                float(config.get("b_shift", self.b_shift)),
                float(config.get("g_shift", self.g_shift)),
                float(config.get("r_shift", self.r_shift)),
            ]
        )
        output = to_uint8(frame + shifts)
        return replace_frame(data, output)


class HSVShift(AIVisionComponent):
    def __init__(self, hue_shift=0, sat_shift=0, val_shift=0):
        super().__init__()
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift

    def _execute(self, data, config):
        frame = extract_frame(data)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(config.get("hue_shift", self.hue_shift))) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(config.get("sat_shift", self.sat_shift)), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(config.get("val_shift", self.val_shift)), 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return replace_frame(data, output)


class ToSepia(AIVisionComponent):
    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]],
            dtype=np.float32,
        )
        output = to_uint8(frame @ kernel.T)
        return replace_frame(data, output)


class InvertImage(AIVisionComponent):
    def _execute(self, data, config):
        frame = extract_frame(data)
        return replace_frame(data, 255 - frame)


class RandomBrightnessContrast(AIVisionComponent):
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2):
        super().__init__()
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def _execute(self, data, config):
        frame = extract_frame(data)
        brightness = random_uniform(
            config,
            "brightness_min",
            "brightness_max",
            -self.brightness_limit,
            self.brightness_limit,
        )
        contrast = random_uniform(
            config,
            "contrast_min",
            "contrast_max",
            1 - self.contrast_limit,
            1 + self.contrast_limit,
        )
        output = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness * 255)
        return replace_frame(data, output)
