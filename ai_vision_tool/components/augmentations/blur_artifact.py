from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import ensure_odd, extract_frame, maybe_grayscale_to_bgr, replace_frame, to_uint8
from ..base import AIVisionComponent


class Posterize(AIVisionComponent):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits

    def _execute(self, data, config):
        frame = extract_frame(data)
        bits = int(config.get("bits", self.bits))
        shift = max(0, 8 - bits)
        output = np.left_shift(np.right_shift(frame, shift), shift)
        return replace_frame(data, output)


class Solarize(AIVisionComponent):
    def __init__(self, threshold=128):
        super().__init__()
        self.threshold = threshold

    def _execute(self, data, config):
        frame = extract_frame(data)
        threshold = int(config.get("threshold", self.threshold))
        output = np.where(frame >= threshold, 255 - frame, frame).astype(frame.dtype)
        return replace_frame(data, output)


class Equalize(AIVisionComponent):
    def _execute(self, data, config):
        frame = extract_frame(data)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        output = cv2.cvtColor(cv2.merge((cv2.equalizeHist(y), cr, cb)), cv2.COLOR_YCrCb2BGR)
        return replace_frame(data, output)


class Emboss(AIVisionComponent):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = strength

    def _execute(self, data, config):
        frame = extract_frame(data)
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        strength = float(config.get("strength", self.strength))
        output = cv2.filter2D(frame, -1, kernel * strength) + 128
        return replace_frame(data, to_uint8(output))


class Sharpen(AIVisionComponent):
    def __init__(self, amount=1.0):
        super().__init__()
        self.amount = amount

    def _execute(self, data, config):
        frame = extract_frame(data)
        amount = float(config.get("amount", self.amount))
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        output = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        return replace_frame(data, output)


class GaussianBlur(AIVisionComponent):
    def __init__(self, kernel_size=5, sigma_x=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def _execute(self, data, config):
        frame = extract_frame(data)
        kernel = ensure_odd(config.get("kernel_size", self.kernel_size))
        sigma_x = float(config.get("sigma_x", self.sigma_x))
        return replace_frame(data, cv2.GaussianBlur(frame, (kernel, kernel), sigma_x))


class MedianBlur(AIVisionComponent):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size

    def _execute(self, data, config):
        frame = extract_frame(data)
        kernel = ensure_odd(config.get("kernel_size", self.kernel_size))
        return replace_frame(data, cv2.medianBlur(frame, kernel))


class GlassBlur(AIVisionComponent):
    def __init__(self, sigma=0.7, max_delta=2, iterations=1):
        super().__init__()
        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations

    def _execute(self, data, config):
        frame = extract_frame(data).copy()
        sigma = float(config.get("sigma", self.sigma))
        max_delta = int(config.get("max_delta", self.max_delta))
        iterations = int(config.get("iterations", self.iterations))
        output = cv2.GaussianBlur(frame, (0, 0), sigma)
        height, width = output.shape[:2]
        for _ in range(iterations):
            for y in range(max_delta, height - max_delta):
                for x in range(max_delta, width - max_delta):
                    dx = np.random.randint(-max_delta, max_delta + 1)
                    dy = np.random.randint(-max_delta, max_delta + 1)
                    output[y, x], output[y + dy, x + dx] = output[y + dy, x + dx].copy(), output[y, x].copy()
        return replace_frame(data, output)


class DefocusBlur(AIVisionComponent):
    def __init__(self, radius=5):
        super().__init__()
        self.radius = radius

    def _execute(self, data, config):
        frame = extract_frame(data)
        radius = max(1, int(config.get("radius", self.radius)))
        kernel_size = radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        kernel /= kernel.sum()
        output = cv2.filter2D(frame, -1, kernel)
        return replace_frame(data, output)


class ZoomBlur(AIVisionComponent):
    def __init__(self, zoom_factor=1.2, steps=5):
        super().__init__()
        self.zoom_factor = zoom_factor
        self.steps = steps

    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        height, width = frame.shape[:2]
        zoom_factor = float(config.get("zoom_factor", self.zoom_factor))
        steps = int(config.get("steps", self.steps))
        accum = np.zeros_like(frame, dtype=np.float32)
        for i in range(steps):
            factor = 1.0 + (zoom_factor - 1.0) * (i / max(steps - 1, 1))
            zoomed = cv2.resize(frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
            zh, zw = zoomed.shape[:2]
            x1 = max(0, (zw - width) // 2)
            y1 = max(0, (zh - height) // 2)
            cropped = zoomed[y1 : y1 + height, x1 : x1 + width]
            if cropped.shape[:2] != (height, width):
                cropped = cv2.resize(cropped, (width, height))
            accum += cropped
        output = to_uint8(accum / steps)
        return replace_frame(data, output)


class CompressionArtifacts(AIVisionComponent):
    def __init__(self, quality=40):
        super().__init__()
        self.quality = quality

    def _execute(self, data, config):
        return JPEGCompression(quality=config.get("quality", self.quality)).run(data, config)


class JPEGCompression(AIVisionComponent):
    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality

    def _execute(self, data, config):
        frame = extract_frame(data)
        quality = int(config.get("quality", self.quality))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return replace_frame(data, frame.copy())
        output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return replace_frame(data, output)


class Downscale(AIVisionComponent):
    def __init__(self, scale=0.5, interpolation="area"):
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation

    def _execute(self, data, config):
        frame = extract_frame(data)
        scale = float(config.get("scale", self.scale))
        interpolation = _resolve_interpolation(config.get("interpolation", self.interpolation))
        height, width = frame.shape[:2]
        down = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=interpolation)
        output = cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)
        return replace_frame(data, output)


class Superpixel(AIVisionComponent):
    def __init__(self, region_size=10, ruler=10.0):
        super().__init__()
        self.region_size = region_size
        self.ruler = ruler

    def _execute(self, data, config):
        frame = extract_frame(data)
        if not hasattr(cv2, "ximgproc"):
            return replace_frame(data, frame.copy())
        slic = cv2.ximgproc.createSuperpixelSLIC(
            frame,
            algorithm=cv2.ximgproc.SLICO,
            region_size=int(config.get("region_size", self.region_size)),
            ruler=float(config.get("ruler", self.ruler)),
        )
        slic.iterate(5)
        labels = slic.getLabels()
        output = np.zeros_like(frame)
        for label in np.unique(labels):
            mask = labels == label
            output[mask] = frame[mask].mean(axis=0)
        return replace_frame(data, to_uint8(output))


def _resolve_interpolation(name):
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    return mapping.get(str(name).lower(), cv2.INTER_AREA)
