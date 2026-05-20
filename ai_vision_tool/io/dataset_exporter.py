from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame


class DatasetExporter(AIVisionComponent):
    """Exports annotated images to YOLO, COCO, or Pascal VOC dataset format.

    Args:
        output_dir: Root output directory.
        format: "yolo", "coco", or "voc".
        split: Dataset split name ("train", "val", "test").
        class_names: List of class names (required for YOLO/VOC index mapping).
    """

    def __init__(self, output_dir: str, format: str = "yolo", split: str = "train",
                 class_names: list[str] | None = None):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.format = format
        self.split = split
        self.class_names = class_names or []
        self._counter = 0
        self._coco_images: list = []
        self._coco_annotations: list = []
        self._ann_id = 0

    def setup(self, config: dict):
        fmt = self.format
        if fmt == "yolo":
            (self.output_dir / "images" / self.split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / self.split).mkdir(parents=True, exist_ok=True)
        elif fmt == "coco":
            (self.output_dir / "images" / self.split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)
        elif fmt == "voc":
            (self.output_dir / "Annotations").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "JPEGImages").mkdir(parents=True, exist_ok=True)
        super().setup(config)

    def _class_id(self, label: str) -> int:
        if label in self.class_names:
            return self.class_names.index(label)
        self.class_names.append(label)
        return len(self.class_names) - 1

    def _execute(self, data, config):
        frame = extract_frame(data)
        bboxes = data.get("bboxes", []) if isinstance(data, dict) else []
        h, w = frame.shape[:2]
        stem = f"{self._counter:06d}"
        self._counter += 1

        if self.format == "yolo":
            img_path = self.output_dir / "images" / self.split / f"{stem}.jpg"
            cv2.imwrite(str(img_path), frame)
            lines = []
            for bb in bboxes:
                cid = self._class_id(bb.get("label", "object"))
                cx = ((bb["x1"] + bb["x2"]) / 2) / w
                cy = ((bb["y1"] + bb["y2"]) / 2) / h
                bw = (bb["x2"] - bb["x1"]) / w
                bh = (bb["y2"] - bb["y1"]) / h
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lbl_path = self.output_dir / "labels" / self.split / f"{stem}.txt"
            lbl_path.write_text("\n".join(lines))

        elif self.format == "coco":
            img_path = self.output_dir / "images" / self.split / f"{stem}.jpg"
            cv2.imwrite(str(img_path), frame)
            img_id = self._counter
            self._coco_images.append({"id": img_id, "file_name": f"{stem}.jpg", "width": w, "height": h})
            for bb in bboxes:
                self._ann_id += 1
                self._coco_annotations.append({
                    "id": self._ann_id, "image_id": img_id,
                    "category_id": self._class_id(bb.get("label", "object")) + 1,
                    "bbox": [bb["x1"], bb["y1"], bb["x2"] - bb["x1"], bb["y2"] - bb["y1"]],
                    "area": (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"]),
                    "iscrowd": 0, "score": bb.get("conf", 1.0),
                })

        elif self.format == "voc":
            img_path = self.output_dir / "JPEGImages" / f"{stem}.jpg"
            cv2.imwrite(str(img_path), frame)
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = f"{stem}.jpg"
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(w)
            ET.SubElement(size, "height").text = str(h)
            ET.SubElement(size, "depth").text = str(frame.shape[2] if frame.ndim == 3 else 1)
            for bb in bboxes:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = bb.get("label", "object")
                ET.SubElement(obj, "difficult").text = "0"
                bndbox = ET.SubElement(obj, "bndbox")
                for key, val in [("xmin", bb["x1"]), ("ymin", bb["y1"]),
                                   ("xmax", bb["x2"]), ("ymax", bb["y2"])]:
                    ET.SubElement(bndbox, key).text = str(int(val))
            ET.ElementTree(root).write(str(self.output_dir / "Annotations" / f"{stem}.xml"))

        return data

    def finalize(self) -> None:
        if self.format == "coco":
            categories = [{"id": i + 1, "name": n} for i, n in enumerate(self.class_names)]
            ann = {"images": self._coco_images, "annotations": self._coco_annotations, "categories": categories}
            out = self.output_dir / "annotations" / f"instances_{self.split}.json"
            out.write_text(json.dumps(ann, indent=2))
            print(f"[DatasetExporter] COCO annotations: {out}")
        if self.format == "yolo" and self.class_names:
            (self.output_dir / "classes.txt").write_text("\n".join(self.class_names))
        print(f"[DatasetExporter] Exported {self._counter} images ({self.format})")

    def cleanup(self):
        self.finalize()
