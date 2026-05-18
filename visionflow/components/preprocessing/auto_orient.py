from __future__ import annotations

import cv2

from .._image_utils import extract_frame, replace_frame
from ..base import AIVisionComponent


class AutoOrient(AIVisionComponent):
    """Applies EXIF-style orientation or explicit rotation/flip transforms."""

    def __init__(
        self,
        use_exif=True,
        exif_key="exif_orientation",
        rotation=None,
        flip_horizontal=False,
        flip_vertical=False,
    ):
        super().__init__()
        self.use_exif = use_exif
        self.exif_key = exif_key
        self.rotation = rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    def _execute(self, data, config):
        frame = extract_frame(data)
        output = frame.copy()

        use_exif = config.get("use_exif", self.use_exif)
        exif_key = config.get("exif_key", self.exif_key)
        rotation = config.get("rotation", self.rotation)
        flip_horizontal = config.get("flip_horizontal", self.flip_horizontal)
        flip_vertical = config.get("flip_vertical", self.flip_vertical)

        exif_orientation = None
        if use_exif and isinstance(data, dict):
            metadata = data.get("metadata", {})
            exif_orientation = metadata.get(exif_key)

        if exif_orientation is not None:
            output = self._apply_exif_orientation(output, exif_orientation)
        elif rotation is not None:
            normalized = int(rotation) % 360
            if normalized == 90:
                output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
            elif normalized == 180:
                output = cv2.rotate(output, cv2.ROTATE_180)
            elif normalized == 270:
                output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if flip_horizontal:
            output = cv2.flip(output, 1)
        if flip_vertical:
            output = cv2.flip(output, 0)

        return replace_frame(data, output)

    @staticmethod
    def _apply_exif_orientation(frame, orientation):
        mapping = {
            2: lambda img: cv2.flip(img, 1),
            3: lambda img: cv2.rotate(img, cv2.ROTATE_180),
            4: lambda img: cv2.flip(img, 0),
            5: lambda img: cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1),
            6: lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            7: lambda img: cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1),
            8: lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        return mapping.get(int(orientation), lambda img: img)(frame)
