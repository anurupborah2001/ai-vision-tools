from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, replace_frame, to_uint8
from ..base import AIVisionComponent
from .common import additive_noise


class ISONoise(AIVisionComponent):
    """Simulates ISO camera sensor noise with luminance and chroma components.

    Args:
        color_shift (float): Standard deviation scale for per-channel chroma noise. Default is 0.01.
        intensity (float): Standard deviation scale for luminance noise. Default is 0.5.
    """

    def __init__(self, color_shift=0.01, intensity=0.5):
        """Initializes ISONoise with noise intensity and color shift parameters.

        Args:
            color_shift (float): Chroma noise scale. Default is 0.01.
            intensity (float): Luminance noise scale. Default is 0.5.
        """
        super().__init__()
        self.color_shift = color_shift
        self.intensity = intensity

    def _execute(self, data, config):
        """Applies ISO-style sensor noise to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'intensity' and 'color_shift'.

        Returns:
            NumPy array or dict: Noisy image in the same format as input.
        """
        frame = extract_frame(data).astype(np.float32)
        intensity = float(config.get("intensity", self.intensity))
        color_shift = float(config.get("color_shift", self.color_shift))
        noise = np.random.normal(0, 255 * intensity * 0.05, frame.shape)
        chroma = np.random.normal(0, 255 * color_shift, frame.shape)
        output = to_uint8(frame + noise + chroma)
        return replace_frame(data, output)


class MultiplicativeNoise(AIVisionComponent):
    """Applies pixel-wise multiplicative noise sampled from a uniform distribution.

    Args:
        multiplier_min (float): Minimum multiplier value. Default is 0.9.
        multiplier_max (float): Maximum multiplier value. Default is 1.1.
    """

    def __init__(self, multiplier_min=0.9, multiplier_max=1.1):
        """Initializes MultiplicativeNoise with multiplier bounds.

        Args:
            multiplier_min (float): Lower bound of the multiplier range. Default is 0.9.
            multiplier_max (float): Upper bound of the multiplier range. Default is 1.1.
        """
        super().__init__()
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max

    def _execute(self, data, config):
        """Applies pixel-wise multiplicative noise to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'multiplier_min' and 'multiplier_max'.

        Returns:
            NumPy array or dict: Noise-multiplied image in the same format as input.
        """
        frame = extract_frame(data).astype(np.float32)
        multiplier = np.random.uniform(
            float(config.get("multiplier_min", self.multiplier_min)),
            float(config.get("multiplier_max", self.multiplier_max)),
            size=frame.shape,
        )
        return replace_frame(data, to_uint8(frame * multiplier))


class SaltPepperNoise(AIVisionComponent):
    """Adds salt-and-pepper noise by setting random pixels to 255 or 0.

    Args:
        amount (float): Fraction of total pixels to corrupt. Default is 0.02.
        salt_vs_pepper (float): Ratio of white (salt) pixels among corrupted pixels. Default is 0.5.
    """

    def __init__(self, amount=0.02, salt_vs_pepper=0.5):
        """Initializes SaltPepperNoise with corruption rate and salt ratio.

        Args:
            amount (float): Fraction of pixels to corrupt. Default is 0.02.
            salt_vs_pepper (float): Salt pixel fraction. Default is 0.5.
        """
        super().__init__()
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def _execute(self, data, config):
        """Applies salt-and-pepper noise to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'amount' and 'salt_vs_pepper'.

        Returns:
            NumPy array or dict: Noisy image in the same format as input.
        """
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
    """Randomly fills rectangular holes in the image with a constant value.

    Args:
        holes (int): Number of rectangular holes to apply. Default is 8.
        max_height (int): Maximum height of each hole in pixels. Default is 8.
        max_width (int): Maximum width of each hole in pixels. Default is 8.
        fill_value (int or tuple): Fill color or intensity for holes. Default is 0.
    """

    def __init__(self, holes=8, max_height=8, max_width=8, fill_value=0):
        """Initializes CoarseDropout with hole count, maximum dimensions, and fill value.

        Args:
            holes (int): Number of holes. Default is 8.
            max_height (int): Max hole height. Default is 8.
            max_width (int): Max hole width. Default is 8.
            fill_value (int or tuple): Hole fill value. Default is 0.
        """
        super().__init__()
        self.holes = holes
        self.max_height = max_height
        self.max_width = max_width
        self.fill_value = fill_value

    def _execute(self, data, config):
        """Applies coarse dropout holes to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'holes', 'max_height',
                'max_width', 'fill_value'.

        Returns:
            NumPy array or dict: Image with dropout holes applied.
        """
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
    """Drops a square sub-region from each cell of a regular grid.

    Args:
        ratio (float): Fraction of each grid cell to drop. Default is 0.5.
        unit_size (int): Size of each grid cell in pixels. Default is 8.
        fill_value (int or tuple): Fill color for dropped regions. Default is 0.
    """

    def __init__(self, ratio=0.5, unit_size=8, fill_value=0):
        """Initializes GridDropout with grid cell size, drop ratio, and fill value.

        Args:
            ratio (float): Drop fraction per cell. Default is 0.5.
            unit_size (int): Grid cell size in pixels. Default is 8.
            fill_value (int or tuple): Fill value. Default is 0.
        """
        super().__init__()
        self.ratio = ratio
        self.unit_size = unit_size
        self.fill_value = fill_value

    def _execute(self, data, config):
        """Applies grid dropout to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'ratio', 'unit_size', 'fill_value'.

        Returns:
            NumPy array or dict: Image with grid dropout applied.
        """
        frame = extract_frame(data).copy()
        ratio = float(config.get("ratio", self.ratio))
        unit_size = int(config.get("unit_size", self.unit_size))
        cut = max(1, int(unit_size * ratio))
        for y in range(0, frame.shape[0], unit_size):
            for x in range(0, frame.shape[1], unit_size):
                frame[y : y + cut, x : x + cut] = config.get("fill_value", self.fill_value)
        return replace_frame(data, frame)


class RandomErasing(AIVisionComponent):
    """Randomly erases a rectangular region from the image (Random Erasing augmentation).

    Args:
        scale (tuple[float, float]): Min and max fraction of total image area to erase. Default is (0.02, 0.2).
        fill_value (int or tuple): Fill value for the erased region. Default is 0.
    """

    def __init__(self, scale=(0.02, 0.2), fill_value=0):
        """Initializes RandomErasing with an area scale range and fill value.

        Args:
            scale (tuple[float, float]): (min_area_fraction, max_area_fraction). Default is (0.02, 0.2).
            fill_value (int or tuple): Fill value for the erased region. Default is 0.
        """
        super().__init__()
        self.scale = scale
        self.fill_value = fill_value

    def _execute(self, data, config):
        """Erases a randomly sized and positioned rectangle from the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'scale' and 'fill_value'.

        Returns:
            NumPy array or dict: Image with erased region filled.
        """
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
    """Randomly sets individual pixels to a constant value.

    Args:
        dropout_prob (float): Probability of each pixel being dropped. Default is 0.01.
        fill_value (int or tuple): Replacement value for dropped pixels. Default is 0.
    """

    def __init__(self, dropout_prob=0.01, fill_value=0):
        """Initializes PixelDropout with a per-pixel dropout probability.

        Args:
            dropout_prob (float): Drop probability per pixel. Default is 0.01.
            fill_value (int or tuple): Fill value for dropped pixels. Default is 0.
        """
        super().__init__()
        self.dropout_prob = dropout_prob
        self.fill_value = fill_value

    def _execute(self, data, config):
        """Applies pixel-level dropout to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'dropout_prob' and 'fill_value'.

        Returns:
            NumPy array or dict: Image with random pixels dropped.
        """
        frame = extract_frame(data).copy()
        prob = float(config.get("dropout_prob", self.dropout_prob))
        mask = np.random.rand(*frame.shape[:2]) < prob
        frame[mask] = config.get("fill_value", self.fill_value)
        return replace_frame(data, frame)


class MaskDropout(AIVisionComponent):
    """Randomly zeroes regions of the segmentation mask in a payload dict.

    Operates only on payload dicts that contain a 'mask' key.

    Args:
        dropout_prob (float): Probability of each mask pixel being set to 0. Default is 0.1.
    """

    def __init__(self, dropout_prob=0.1):
        """Initializes MaskDropout with a per-pixel dropout probability for the mask.

        Args:
            dropout_prob (float): Mask dropout probability. Default is 0.1.
        """
        super().__init__()
        self.dropout_prob = dropout_prob

    def _execute(self, data, config):
        """Applies dropout to the 'mask' field in the payload dict.

        Args:
            data: Payload dict with 'mask' key. Returns data unchanged if not a dict or
                'mask' key is absent.
            config (dict): Runtime overrides. Supports 'dropout_prob'.

        Returns:
            dict: Updated payload with the mask partially zeroed, or original data unchanged.
        """
        if not isinstance(data, dict) or "mask" not in data:
            return data
        mask = data["mask"].copy()
        prob = float(config.get("dropout_prob", self.dropout_prob))
        drop = np.random.rand(*mask.shape[:2]) < prob
        mask[drop] = 0
        data["mask"] = mask
        return data
