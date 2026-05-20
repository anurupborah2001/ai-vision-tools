from __future__ import annotations

import cv2

from ..utils.image_utils import extract_frame, replace_frame
from ..core.base import AIVisionComponent


class AutoOrient(AIVisionComponent):
    """Applies EXIF-style orientation or explicit rotation/flip transforms.

    When EXIF orientation metadata is present in the payload, applies the
    corresponding transform. Otherwise falls back to the explicit rotation
    and flip parameters.

    Args:
        use_exif (bool): Whether to use EXIF orientation metadata if available.
            Default is True.
        exif_key (str): Key in the payload metadata dict where EXIF orientation
            is stored. Default is 'exif_orientation'.
        rotation (int or None): Explicit rotation in degrees (90, 180, 270) applied
            when EXIF is not used. Default is None (no rotation).
        flip_horizontal (bool): Whether to flip the image horizontally. Default is False.
        flip_vertical (bool): Whether to flip the image vertically. Default is False.
    """

    def __init__(
        self,
        use_exif=True,
        exif_key="exif_orientation",
        rotation=None,
        flip_horizontal=False,
        flip_vertical=False,
    ):
        """Initializes AutoOrient with orientation correction parameters.

        Args:
            use_exif (bool): Read orientation from EXIF metadata. Default is True.
            exif_key (str): Metadata dict key for EXIF orientation value. Default is
                'exif_orientation'.
            rotation (int or None): Explicit clockwise rotation in degrees (90, 180, 270).
                Default is None.
            flip_horizontal (bool): Flip left-right. Default is False.
            flip_vertical (bool): Flip top-bottom. Default is False.
        """
        super().__init__()
        self.use_exif = use_exif
        self.exif_key = exif_key
        self.rotation = rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    def _execute(self, data, config):
        """Applies orientation correction to the extracted frame.

        Reads EXIF orientation from ``data['metadata'][exif_key]`` when available
        and use_exif is True. Falls back to explicit rotation if EXIF is absent.
        Horizontal and vertical flips are applied after rotation.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' and
                optional 'metadata' keys.
            config (dict): Runtime overrides. Supports 'use_exif', 'exif_key',
                'rotation', 'flip_horizontal', 'flip_vertical'.

        Returns:
            numpy.ndarray or dict: Reoriented image in the same format as input.
        """
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
        """Rotates or flips a frame to match the given EXIF orientation value.

        Maps standard EXIF orientation codes (1–8) to the corresponding
        OpenCV transform. Code 1 (normal) is a no-op.

        Args:
            frame (numpy.ndarray): Input BGR image.
            orientation (int): EXIF orientation tag value (1–8).

        Returns:
            numpy.ndarray: Corrected image, or the original if orientation is
                unrecognised.
        """
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
