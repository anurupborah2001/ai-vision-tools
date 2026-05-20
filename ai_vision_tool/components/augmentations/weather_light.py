from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent
from .common import ensure_color, random_uniform


class RandomShadow(AIVisionComponent):
    """Applies a vertical shadow strip of random position and width to an image.

    Args:
        shadow_dimension (float): Width of the shadow as a fraction of image width. Default is 0.5.
        intensity (float): Multiplier applied to pixels inside the shadow (< 1.0 = darker). Default is 0.5.
    """

    def __init__(self, shadow_dimension=0.5, intensity=0.5):
        """Initializes RandomShadow with shadow size and intensity.

        Args:
            shadow_dimension (float): Shadow width fraction. Default is 0.5.
            intensity (float): Darkening multiplier. Default is 0.5.
        """
        super().__init__()
        self.shadow_dimension = shadow_dimension
        self.intensity = intensity

    def _execute(self, data, config):
        """Applies a random shadow strip to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'shadow_dimension' and 'intensity'.

        Returns:
            NumPy array or dict: Image with shadow applied.
        """
        frame = extract_frame(data).copy()
        height, width = frame.shape[:2]
        shadow_w = int(width * float(config.get("shadow_dimension", self.shadow_dimension)))
        x1 = np.random.randint(0, max(1, width - shadow_w))
        overlay = frame.copy()
        overlay[:, x1 : x1 + shadow_w] = (overlay[:, x1 : x1 + shadow_w] * float(config.get("intensity", self.intensity))).astype(frame.dtype)
        return replace_frame(data, overlay)


class RandomSunFlare(AIVisionComponent):
    """Simulates a sun flare by blending a bright circle onto the image.

    Args:
        center (tuple[int, int] or None): (x, y) center of the flare. Default is None (image center-top).
        radius (int): Radius of the flare circle in pixels. Default is 20.
        intensity (float): Blend weight of the flare overlay. Default is 0.4.
    """

    def __init__(self, center=None, radius=20, intensity=0.4):
        """Initializes RandomSunFlare with flare position, size, and intensity.

        Args:
            center (tuple[int, int] or None): Flare center coordinates. Default is None.
            radius (int): Flare circle radius. Default is 20.
            intensity (float): Blend weight. Default is 0.4.
        """
        super().__init__()
        self.center = center
        self.radius = radius
        self.intensity = intensity

    def _execute(self, data, config):
        """Applies a sun flare effect to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'center', 'radius', 'intensity'.

        Returns:
            NumPy array or dict: Image with flare applied.
        """
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
    """Blends the image with a white layer to simulate fog.

    Args:
        alpha (float): Fog blend weight in [0, 1]. Higher values produce denser fog. Default is 0.35.
    """

    def __init__(self, alpha=0.35):
        """Initializes RandomFog with a fog density.

        Args:
            alpha (float): Fog blend weight. Default is 0.35.
        """
        super().__init__()
        self.alpha = alpha

    def _execute(self, data, config):
        """Applies fog overlay to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'alpha'.

        Returns:
            NumPy array or dict: Fogged image in the same format as input.
        """
        frame = extract_frame(data)
        alpha = float(config.get("alpha", self.alpha))
        fog = np.full_like(frame, 255)
        output = cv2.addWeighted(frame, 1.0 - alpha, fog, alpha, 0)
        return replace_frame(data, output)


class RandomRain(AIVisionComponent):
    """Draws random rain streaks onto the image.

    Args:
        drops (int): Number of rain streaks to draw. Default is 40.
        drop_length (int): Length of each streak in pixels. Default is 12.
        intensity (float): Blend weight of the rain overlay. Default is 0.25.
    """

    def __init__(self, drops=40, drop_length=12, intensity=0.25):
        """Initializes RandomRain with rain parameters.

        Args:
            drops (int): Number of rain lines. Default is 40.
            drop_length (int): Length of each drop in pixels. Default is 12.
            intensity (float): Blend weight of rain overlay. Default is 0.25.
        """
        super().__init__()
        self.drops = drops
        self.drop_length = drop_length
        self.intensity = intensity

    def _execute(self, data, config):
        """Draws rain streaks on the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'drops', 'drop_length', 'intensity'.

        Returns:
            NumPy array or dict: Image with rain effect applied.
        """
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
    """Adds Gaussian-distributed bright noise to simulate falling snow.

    Args:
        intensity (float): Controls the mean and spread of snowflake brightness. Default is 0.1.
    """

    def __init__(self, intensity=0.1):
        """Initializes RandomSnow with a snow intensity parameter.

        Args:
            intensity (float): Snowflake brightness scale. Default is 0.1.
        """
        super().__init__()
        self.intensity = intensity

    def _execute(self, data, config):
        """Applies snow noise to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'intensity'.

        Returns:
            NumPy array or dict: Image with snow effect applied.
        """
        frame = extract_frame(data)
        intensity = float(config.get("intensity", self.intensity))
        noise = np.random.normal(loc=255 * intensity, scale=255 * intensity * 0.5, size=frame.shape)
        output = to_uint8(frame.astype(np.float32) + noise)
        return replace_frame(data, output)


class RandomGamma(AIVisionComponent):
    """Applies gamma correction with a randomly sampled gamma value.

    Args:
        min_gamma (float): Minimum gamma value. Default is 0.7.
        max_gamma (float): Maximum gamma value. Default is 1.5.
    """

    def __init__(self, min_gamma=0.7, max_gamma=1.5):
        """Initializes RandomGamma with gamma range bounds.

        Args:
            min_gamma (float): Lower gamma bound. Default is 0.7.
            max_gamma (float): Upper gamma bound. Default is 1.5.
        """
        super().__init__()
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def _execute(self, data, config):
        """Applies random gamma correction to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'min_gamma' and 'max_gamma'.

        Returns:
            NumPy array or dict: Gamma-corrected image in the same format as input.
        """
        frame = extract_frame(data)
        gamma = random_uniform(config, "min_gamma", "max_gamma", self.min_gamma, self.max_gamma)
        table = np.array([((i / 255.0) ** (1.0 / max(gamma, 1e-6))) * 255 for i in range(256)], dtype=np.uint8)
        output = cv2.LUT(frame, table)
        return replace_frame(data, output)


class ColorJitter(AIVisionComponent):
    """Randomly jitters brightness, contrast, saturation, and hue.

    Args:
        brightness (float): Maximum absolute brightness change as a fraction of 255. Default is 0.2.
        contrast (float): Maximum contrast change (factor range becomes [1-contrast, 1+contrast]). Default is 0.2.
        saturation (float): Maximum saturation scale change. Default is 0.2.
        hue (float): Maximum hue shift in degrees. Default is 10.
    """

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=10):
        """Initializes ColorJitter with per-channel jitter amounts.

        Args:
            brightness (float): Brightness jitter magnitude. Default is 0.2.
            contrast (float): Contrast jitter magnitude. Default is 0.2.
            saturation (float): Saturation jitter magnitude. Default is 0.2.
            hue (float): Hue jitter in degrees. Default is 10.
        """
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def _execute(self, data, config):
        """Applies random color jitter to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'brightness_min/max', 'contrast_min/max',
                'saturation_min/max', 'hue_min/max'.

        Returns:
            NumPy array or dict: Color-jittered image in the same format as input.
        """
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
    """Randomly permutes the order of the three color channels."""

    def _execute(self, data, config):
        """Randomly shuffles the BGR channels of the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Unused. Present for interface compatibility.

        Returns:
            NumPy array or dict: Channel-shuffled image in the same format as input.
        """
        frame = extract_frame(data)
        order = np.random.permutation(3)
        output = frame[:, :, order]
        return replace_frame(data, output)


class RGBShift(AIVisionComponent):
    """Shifts each RGB channel by a fixed additive offset.

    Args:
        r_shift (float): Offset added to the red channel. Default is 0.
        g_shift (float): Offset added to the green channel. Default is 0.
        b_shift (float): Offset added to the blue channel. Default is 0.
    """

    def __init__(self, r_shift=0, g_shift=0, b_shift=0):
        """Initializes RGBShift with per-channel offsets.

        Args:
            r_shift (float): Red channel offset. Default is 0.
            g_shift (float): Green channel offset. Default is 0.
            b_shift (float): Blue channel offset. Default is 0.
        """
        super().__init__()
        self.r_shift = r_shift
        self.g_shift = g_shift
        self.b_shift = b_shift

    def _execute(self, data, config):
        """Applies per-channel additive shifts to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'r_shift', 'g_shift', 'b_shift'.

        Returns:
            NumPy array or dict: Channel-shifted image clipped to uint8 range.
        """
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
    """Shifts the hue, saturation, and value channels in HSV color space.

    Args:
        hue_shift (int): Hue shift in degrees (modulo 180). Default is 0.
        sat_shift (int): Saturation additive offset. Default is 0.
        val_shift (int): Value (brightness) additive offset. Default is 0.
    """

    def __init__(self, hue_shift=0, sat_shift=0, val_shift=0):
        """Initializes HSVShift with per-channel shift values.

        Args:
            hue_shift (int): Hue shift. Default is 0.
            sat_shift (int): Saturation shift. Default is 0.
            val_shift (int): Value shift. Default is 0.
        """
        super().__init__()
        self.hue_shift = hue_shift
        self.sat_shift = sat_shift
        self.val_shift = val_shift

    def _execute(self, data, config):
        """Applies HSV channel shifts to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'hue_shift', 'sat_shift', 'val_shift'.

        Returns:
            NumPy array or dict: HSV-shifted image in BGR format.
        """
        frame = extract_frame(data)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(config.get("hue_shift", self.hue_shift))) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + int(config.get("sat_shift", self.sat_shift)), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(config.get("val_shift", self.val_shift)), 0, 255)
        output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return replace_frame(data, output)


class ToSepia(AIVisionComponent):
    """Applies a sepia tone filter using a fixed color transformation matrix."""

    def _execute(self, data, config):
        """Converts the extracted frame to sepia tones.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Unused. Present for interface compatibility.

        Returns:
            NumPy array or dict: Sepia-toned image clipped to uint8 range.
        """
        frame = extract_frame(data).astype(np.float32)
        kernel = np.array(
            [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]],
            dtype=np.float32,
        )
        output = to_uint8(frame @ kernel.T)
        return replace_frame(data, output)


class InvertImage(AIVisionComponent):
    """Inverts all pixel values in an image (photographic negative)."""

    def _execute(self, data, config):
        """Inverts pixel values of the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Unused. Present for interface compatibility.

        Returns:
            NumPy array or dict: Pixel-inverted image (255 - original).
        """
        frame = extract_frame(data)
        return replace_frame(data, 255 - frame)


class RandomBrightnessContrast(AIVisionComponent):
    """Randomly adjusts brightness and contrast within bounded ranges.

    Args:
        brightness_limit (float): Maximum absolute brightness change as a fraction of 255. Default is 0.2.
        contrast_limit (float): Maximum contrast factor deviation from 1.0. Default is 0.2.
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2):
        """Initializes RandomBrightnessContrast with adjustment limits.

        Args:
            brightness_limit (float): Brightness adjustment bound. Default is 0.2.
            contrast_limit (float): Contrast adjustment bound. Default is 0.2.
        """
        super().__init__()
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def _execute(self, data, config):
        """Applies random brightness and contrast adjustment to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'brightness_min/max' and 'contrast_min/max'.

        Returns:
            NumPy array or dict: Brightness- and contrast-adjusted image.
        """
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
