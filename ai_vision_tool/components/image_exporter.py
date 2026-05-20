from .base import AIVisionComponent
import cv2
from pathlib import Path


class ImageExporter(AIVisionComponent):
    """Exports processed versions of frames (grayscale, edges) to disk.

    Args:
        output_dir (str): Directory where exported images are saved. Default is 'exports'.
    """

    def __init__(self, output_dir="exports"):
        """Initializes ImageExporter and creates the output directory.

        Args:
            output_dir (str): Target export directory path. Default is 'exports'.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_grayscale(self, frame, name="gray.jpg"):
        """Converts a frame to grayscale and saves it to the output directory.

        Args:
            frame (numpy.ndarray): BGR input image.
            name (str): Output filename. Default is 'gray.jpg'.

        Returns:
            str: Absolute path of the saved grayscale image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        path = self.output_dir / name
        cv2.imwrite(str(path), gray)
        return str(path)

    def export_edges(self, frame, name="edges.jpg", t1=100, t2=200):
        """Applies Canny edge detection and saves the result to the output directory.

        Args:
            frame (numpy.ndarray): BGR input image.
            name (str): Output filename. Default is 'edges.jpg'.
            t1 (int): Lower hysteresis threshold for Canny. Default is 100.
            t2 (int): Upper hysteresis threshold for Canny. Default is 200.

        Returns:
            str: Absolute path of the saved edge image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t1, t2)
        path = self.output_dir / name
        cv2.imwrite(str(path), edges)
        return str(path)

    def _execute(self, data, config):
        """Conditionally exports grayscale and/or edge images based on config flags.

        Acts as a passthrough — returns data unchanged so the pipeline continues.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime parameters. Supports:
                - 'export_gray' (bool): If True, saves a grayscale export. Default is False.
                - 'export_edges' (bool): If True, saves a Canny edge export. Default is False.

        Returns:
            Same as input data, unchanged.
        """
        frame = data["frame"] if isinstance(data, dict) else data

        if config.get("export_gray", False):
            self.export_grayscale(frame)

        if config.get("export_edges", False):
            self.export_edges(frame)

        return data
