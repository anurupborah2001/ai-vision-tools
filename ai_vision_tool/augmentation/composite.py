from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_utils import extract_frame, replace_frame, to_uint8
from ..core.base import AIVisionComponent
from .common import maybe_get_partner_frame


class MixUp(AIVisionComponent):
    """Blends two images using a weighted linear combination (MixUp augmentation).

    Requires a 'mix_image' partner frame available via config or the input payload dict.

    Args:
        alpha (float): Weight of the primary image. The partner image is weighted (1 - alpha). Default is 0.5.
    """

    def __init__(self, alpha=0.5):
        """Initializes MixUp with a blending weight.

        Args:
            alpha (float): Primary image weight in [0.0, 1.0]. Default is 0.5.
        """
        super().__init__()
        self.alpha = alpha

    def _execute(self, data, config):
        """Blends the primary frame with a partner image.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' and optionally 'mix_image'.
            config (dict): Runtime overrides. Supports 'alpha' and 'mix_image'.

        Returns:
            NumPy array or dict: Blended image, or copy of input if no partner is available.
        """
        frame = extract_frame(data)
        mix_image = maybe_get_partner_frame(data, config, "mix_image")
        if mix_image is None:
            return replace_frame(data, frame.copy())
        alpha = float(config.get("alpha", self.alpha))
        mixed = cv2.resize(mix_image, (frame.shape[1], frame.shape[0]))
        output = cv2.addWeighted(frame, alpha, mixed, 1.0 - alpha, 0)
        return replace_frame(data, output)


class CutMix(AIVisionComponent):
    """Pastes a rectangular patch from a partner image onto the primary frame (CutMix augmentation).

    Requires a 'mix_image' partner frame available via config or the input payload dict.

    Args:
        alpha (float): Controls the relative patch size (patch dimensions proportional to alpha * 0.5). Default is 0.5.
    """

    def __init__(self, alpha=0.5):
        """Initializes CutMix with a patch size factor.

        Args:
            alpha (float): Patch size factor. Default is 0.5.
        """
        super().__init__()
        self.alpha = alpha

    def _execute(self, data, config):
        """Pastes a randomly placed patch from the partner image into the primary frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' and optionally 'mix_image'.
            config (dict): Runtime overrides. Supports 'alpha' and 'mix_image'.

        Returns:
            NumPy array or dict: Frame with the patch applied, or original if no partner available.
        """
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
    """Pastes an overlay image onto the primary frame at a specified position.

    Requires an 'overlay_image' available via config or the input payload dict.

    Args:
        x (int): Horizontal position of the overlay's top-left corner. Default is 0.
        y (int): Vertical position of the overlay's top-left corner. Default is 0.
    """

    def __init__(self, x=0, y=0):
        """Initializes CopyPaste with a paste position.

        Args:
            x (int): Horizontal offset. Default is 0.
            y (int): Vertical offset. Default is 0.
        """
        super().__init__()
        self.x = x
        self.y = y

    def _execute(self, data, config):
        """Pastes the overlay image onto the primary frame at the configured position.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' and optionally 'overlay_image'.
            config (dict): Runtime overrides. Supports 'x', 'y', and 'overlay_image'.

        Returns:
            NumPy array or dict: Frame with overlay applied, or original if no overlay available.
        """
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
    """Fills a randomly positioned rectangle with a constant value to simulate occlusion.

    Args:
        max_width (int): Maximum width of the occlusion rectangle. Default is 20.
        max_height (int): Maximum height of the occlusion rectangle. Default is 20.
        fill_value (int or tuple): Fill color or intensity for the occluded region. Default is 0.
    """

    def __init__(self, max_width=20, max_height=20, fill_value=0):
        """Initializes RandomOcclusion with occlusion region bounds.

        Args:
            max_width (int): Max occlusion rectangle width. Default is 20.
            max_height (int): Max occlusion rectangle height. Default is 20.
            fill_value (int or tuple): Fill value for the occluded pixels. Default is 0.
        """
        super().__init__()
        self.max_width = max_width
        self.max_height = max_height
        self.fill_value = fill_value

    def _execute(self, data, config):
        """Applies a random occlusion patch to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'max_width', 'max_height', 'fill_value'.

        Returns:
            NumPy array or dict: Frame with random occlusion applied.
        """
        frame = extract_frame(data).copy()
        width = np.random.randint(1, int(config.get("max_width", self.max_width)) + 1)
        height = np.random.randint(1, int(config.get("max_height", self.max_height)) + 1)
        x = np.random.randint(0, max(1, frame.shape[1] - width + 1))
        y = np.random.randint(0, max(1, frame.shape[0] - height + 1))
        frame[y : y + height, x : x + width] = config.get("fill_value", self.fill_value)
        return replace_frame(data, frame)


class ObjectPaste(AIVisionComponent):
    """Pastes an object image patch onto the primary frame at a fixed position.

    Requires an 'object_image' available via config or the input payload dict.

    Args:
        x (int): Horizontal position of the object's top-left corner. Default is 0.
        y (int): Vertical position of the object's top-left corner. Default is 0.
    """

    def __init__(self, x=0, y=0):
        """Initializes ObjectPaste with a paste position.

        Args:
            x (int): Horizontal offset. Default is 0.
            y (int): Vertical offset. Default is 0.
        """
        super().__init__()
        self.x = x
        self.y = y

    def _execute(self, data, config):
        """Pastes the object image onto the primary frame at the configured position.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' and optionally 'object_image'.
            config (dict): Runtime overrides. Supports 'x', 'y', and 'object_image'.

        Returns:
            NumPy array or dict: Frame with object pasted, or original if no object available.
        """
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
    """Applies random jitter to bounding box coordinates in a payload dict.

    Perturbs x, y position and width/height of each bounding box independently.

    Args:
        x_jitter (float): Maximum horizontal jitter as a fraction of box width. Default is 0.05.
        y_jitter (float): Maximum vertical jitter as a fraction of box height. Default is 0.05.
        size_jitter (float): Maximum size perturbation as a fraction of box dimensions. Default is 0.1.
    """

    def __init__(self, x_jitter=0.05, y_jitter=0.05, size_jitter=0.1):
        """Initializes BoundingBoxJitter with jitter fractions.

        Args:
            x_jitter (float): Horizontal jitter fraction. Default is 0.05.
            y_jitter (float): Vertical jitter fraction. Default is 0.05.
            size_jitter (float): Size jitter fraction. Default is 0.1.
        """
        super().__init__()
        self.x_jitter = x_jitter
        self.y_jitter = y_jitter
        self.size_jitter = size_jitter

    def _execute(self, data, config):
        """Jitters bounding boxes in the payload dict.

        Args:
            data: Payload dict with 'bboxes' key containing list of (x, y, w, h) tuples.
                Returns data unchanged if not a dict.
            config (dict): Runtime overrides. Supports 'x_jitter', 'y_jitter', 'size_jitter', 'bboxes'.

        Returns:
            dict: Updated payload dict with jittered 'bboxes', or original data if not a dict.
        """
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
    """Tiles up to nine images in a 3x3 grid mosaic.

    Uses the primary frame to fill any missing mosaic slots.

    Args:
        mosaic_images (list or None): Up to eight additional images for the mosaic grid. Default is None.
        output_size (tuple[int, int] or None): (width, height) of the output mosaic. Default is None (3x input size).
    """

    def __init__(self, mosaic_images=None, output_size=None):
        """Initializes Mosaic9 with partner images and output size.

        Args:
            mosaic_images (list or None): Up to 8 additional NumPy images. Default is None.
            output_size (tuple[int, int] or None): Output (width, height). Default is None.
        """
        super().__init__()
        self.mosaic_images = mosaic_images or []
        self.output_size = output_size

    def _execute(self, data, config):
        """Assembles a 3x3 mosaic from the primary frame and up to eight partner images.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'mosaic_images' and 'output_size'.

        Returns:
            NumPy array or dict: 3x3 mosaic image in the same format as input.
        """
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
