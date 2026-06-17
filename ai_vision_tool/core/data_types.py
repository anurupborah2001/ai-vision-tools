from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    conf: float = 1.0
    track_id: int = -1

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def clip(self, w: float, h: float) -> BBox:
        return BBox(
            max(0.0, min(self.x1, w)),
            max(0.0, min(self.y1, h)),
            max(0.0, min(self.x2, w)),
            max(0.0, min(self.y2, h)),
            self.label,
            self.conf,
            self.track_id,
        )

    def as_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "label": self.label,
            "conf": self.conf,
            "track_id": self.track_id,
        }


@dataclass
class Detection:
    bboxes: list[BBox] = field(default_factory=list)
    frame_id: int = 0
    source: str = ""


@dataclass
class Keypoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    name: str = ""


@dataclass
class Pose:
    keypoints: list[Keypoint] = field(default_factory=list)
    bbox: BBox | None = None
    person_id: int = -1


@dataclass
class Mask:
    array: np.ndarray
    label: str = ""
    conf: float = 1.0

    def area(self) -> int:
        return int(self.array.astype(bool).sum())

    def to_polygon(self) -> list[list[int]]:
        import cv2

        contours, _ = cv2.findContours(
            self.array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []
        c = max(contours, key=cv2.contourArea)
        return c.reshape(-1, 2).tolist()


@dataclass
class SegmentationResult:
    masks: list[Mask] = field(default_factory=list)
    frame_id: int = 0


@dataclass
class Track:
    track_id: int
    bbox: BBox
    age: int = 0
    hits: int = 0
    state: str = "tentative"

    def as_dict(self) -> dict:
        d = self.bbox.as_dict()
        d.update(
            {
                "track_id": self.track_id,
                "age": self.age,
                "hits": self.hits,
                "state": self.state,
            }
        )
        return d
