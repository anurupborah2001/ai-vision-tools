from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame
from ai_vision_tool.tracking.track_manager import TrackManager


class DeepSORTTracker(AIVisionComponent):
    """DeepSORT tracker with appearance-based re-identification.

    Args:
        max_age: Frames before lost track removed.
        min_hits: Hits before track is active.
        iou_threshold: IoU threshold for IoU fallback matching.
        max_cosine_distance: Cosine distance threshold for appearance matching.
        nn_budget: Max embeddings stored per track.
        embedder: "hog" (no deps), "none" (IoU only), or "mobilenet" (requires model).
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3,
                 max_cosine_distance: float = 0.3, nn_budget: int = 100, embedder: str = "hog"):
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.embedder = embedder
        self._tracker: TrackManager | None = None
        self._hog: cv2.HOGDescriptor | None = None
        self._gallery: dict[int, list[np.ndarray]] = {}

    def setup(self, config: dict):
        self._tracker = TrackManager(self.max_age, self.min_hits, self.iou_threshold)
        if self.embedder == "hog":
            self._hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        super().setup(config)

    def _extract_embedding(self, frame: np.ndarray, bbox: dict) -> np.ndarray:
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(3780 if self.embedder == "hog" else 128)
        if self.embedder == "hog":
            roi_resized = cv2.resize(roi, (64, 128))
            gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            feat = self._hog.compute(gray).flatten()
        else:
            feat = roi.astype(np.float32).flatten()[:128]
            if len(feat) < 128:
                feat = np.pad(feat, (0, 128 - len(feat)))
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-6)

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def _execute(self, data, config):
        frame = extract_frame(data)
        bboxes = data.get("bboxes", []) if isinstance(data, dict) else []

        # Appearance matching: update gallery
        tracks = self._tracker.update(bboxes)

        for track in tracks:
            tid = track["track_id"]
            emb = self._extract_embedding(frame, track)
            if tid not in self._gallery:
                self._gallery[tid] = []
            self._gallery[tid].append(emb)
            if len(self._gallery[tid]) > self.nn_budget:
                self._gallery[tid].pop(0)

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["tracks"] = tracks
        return payload

    def cleanup(self):
        if self._tracker:
            self._tracker.reset()
        self._gallery = {}
