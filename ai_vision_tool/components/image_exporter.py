from .base import AIVisionComponent
import cv2
from pathlib import Path

class ImageExporter(AIVisionComponent):
    def __init__(self, output_dir="exports"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_grayscale(self, frame, name="gray.jpg"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        path = self.output_dir / name
        cv2.imwrite(str(path), gray)
        return str(path)

    def export_edges(self, frame, name="edges.jpg", t1=100, t2=200):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t1, t2)
        path = self.output_dir / name
        cv2.imwrite(str(path), edges)
        return str(path)

    def _execute(self, data, config):
        frame = data["frame"] if isinstance(data, dict) else data

        if config.get("export_gray", False):
            self.export_grayscale(frame)

        if config.get("export_edges", False):
            self.export_edges(frame)

        return data
