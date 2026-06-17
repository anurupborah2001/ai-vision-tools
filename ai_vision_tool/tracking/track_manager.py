from __future__ import annotations

import numpy as np

from ai_vision_tool.tracking.kalman_filter import KalmanFilter


class TrackManager:
    """Manages multi-object track lifecycle with IoU-based Hungarian assignment.

    Args:
        max_age: Frames a track can be lost before removal.
        min_hits: Minimum hits before track becomes active.
        iou_threshold: IoU threshold for assignment.
    """

    def __init__(
        self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._kf = KalmanFilter()
        self.tracks: list[dict] = []
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    @staticmethod
    def _iou_matrix(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return np.zeros((len(track_boxes), len(det_boxes)))
        tx1, ty1, tx2, ty2 = (
            track_boxes[:, 0],
            track_boxes[:, 1],
            track_boxes[:, 2],
            track_boxes[:, 3],
        )
        dx1, dy1, dx2, dy2 = (
            det_boxes[:, 0],
            det_boxes[:, 1],
            det_boxes[:, 2],
            det_boxes[:, 3],
        )
        t_areas = (tx2 - tx1) * (ty2 - ty1)
        d_areas = (dx2 - dx1) * (dy2 - dy1)
        ix1 = np.maximum(tx1[:, None], dx1[None, :])
        iy1 = np.maximum(ty1[:, None], dy1[None, :])
        ix2 = np.minimum(tx2[:, None], dx2[None, :])
        iy2 = np.minimum(ty2[:, None], dy2[None, :])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = t_areas[:, None] + d_areas[None, :] - inter
        return inter / (union + 1e-6)

    def _hungarian(self, cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            from scipy.optimize import linear_sum_assignment

            r, c = linear_sum_assignment(-cost_matrix)
            return r, c
        except ImportError:
            pass
        # Greedy fallback
        rows, cols = [], []
        used_r, used_c = set(), set()
        flat = cost_matrix.flatten()
        for idx in np.argsort(-flat):
            r, c = divmod(idx, cost_matrix.shape[1])
            if r not in used_r and c not in used_c:
                rows.append(r)
                cols.append(c)
                used_r.add(r)
                used_c.add(c)
        return np.array(rows), np.array(cols)

    def update(self, detections: list[dict]) -> list[dict]:
        # Predict all tracks
        for t in self.tracks:
            t["mean"], t["cov"] = self._kf.predict(t["mean"], t["cov"])

        det_boxes = (
            np.array(
                [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections],
                dtype=np.float64,
            )
            if detections
            else np.zeros((0, 4))
        )

        track_boxes = (
            np.array(
                [self._kf.to_xyxy(t["mean"]) for t in self.tracks], dtype=np.float64
            )
            if self.tracks
            else np.zeros((0, 4))
        )

        iou_mat = self._iou_matrix(track_boxes, det_boxes)
        matched_t, matched_d = (
            self._hungarian(iou_mat) if iou_mat.size else (np.array([]), np.array([]))
        )

        valid_pairs = [
            (int(r), int(c))
            for r, c in zip(matched_t, matched_d, strict=False)
            if iou_mat[r, c] >= self.iou_threshold
        ]
        matched_track_ids = {r for r, _ in valid_pairs}
        matched_det_ids = {c for _, c in valid_pairs}

        # Update matched tracks
        for r, c in valid_pairs:
            det = detections[c]
            bbox = np.array(
                [det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float64
            )
            self.tracks[r]["mean"], self.tracks[r]["cov"] = self._kf.update(
                self.tracks[r]["mean"], self.tracks[r]["cov"], bbox
            )
            self.tracks[r]["hits"] += 1
            self.tracks[r]["age"] = 0
            self.tracks[r]["label"] = det.get("label", "")
            self.tracks[r]["conf"] = det.get("conf", 1.0)
            if self.tracks[r]["hits"] >= self.min_hits:
                self.tracks[r]["state"] = "active"

        # Missed tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_track_ids:
                t["age"] += 1
                if t["state"] == "active":
                    t["state"] = "lost"

        # New tracks
        for j, det in enumerate(detections):
            if j not in matched_det_ids:
                bbox = np.array(
                    [det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float64
                )
                mean, cov = self._kf.initiate(bbox)
                self.tracks.append(
                    {
                        "track_id": self._next_id,
                        "mean": mean,
                        "cov": cov,
                        "hits": 1,
                        "age": 0,
                        "state": "tentative",
                        "label": det.get("label", ""),
                        "conf": det.get("conf", 1.0),
                    }
                )
                self._next_id += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        return self.get_active_tracks()

    def get_active_tracks(self) -> list[dict]:
        result = []
        for t in self.tracks:
            if t["state"] in ("active", "tentative"):
                xyxy = self._kf.to_xyxy(t["mean"])
                result.append(
                    {
                        "track_id": t["track_id"],
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                        "label": t["label"],
                        "conf": t["conf"],
                        "age": t["age"],
                        "hits": t["hits"],
                        "state": t["state"],
                    }
                )
        return result
