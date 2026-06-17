import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent


class FrameResizer(AIVisionComponent):
    """Resizes a frame, optionally maintaining aspect ratio."""

    def _execute(self, data, config):
        """Resizes the input frame to the target dimensions.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime parameters. Supports:
                - 'size' (tuple[int, int]): Target (width, height). Default is (640, 480).
                - 'keep_aspect' (bool): If True, resize to fit inside the target size
                  with letterboxing on a black canvas. Default is False.

        Returns:
            numpy.ndarray or dict: Resized image in the same format as input.
        """
        frame = data["frame"] if isinstance(data, dict) else data

        target_size = config.get("size", (640, 480))
        keep_aspect = config.get("keep_aspect", False)

        if keep_aspect:
            output = self._resize_keep_aspect(frame, target_size)
        else:
            output = cv2.resize(frame, target_size)

        if isinstance(data, dict):
            data["frame"] = output
            return data
        return output

    def _resize_keep_aspect(self, frame, size):
        """Resizes a frame to fit inside the target size while preserving aspect ratio.

        The resized frame is centered on a zero-filled (black) canvas of the target size.

        Args:
            frame (numpy.ndarray): Input BGR image.
            size (tuple[int, int]): Target (width, height).

        Returns:
            numpy.ndarray: Letterboxed BGR image of exactly the target size.
        """
        target_w, target_h = size
        h, w = frame.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2

        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas
