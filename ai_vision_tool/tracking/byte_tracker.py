from __future__ import annotations

import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame
from ai_vision_tool.tracking.kalman_filter import KalmanFilter
from ai_vision_tool.tracking.track_manager import TrackManager


class ByteTracker(AIVisionComponent):
    """ByteTrack multi-object tracker (Zhang et al. 2022).

    Two-stage association: high-confidence dets first, then low-confidence dets
    matched against remaining unmatched tracks.

    Args:
        track_thresh: High-confidence detection threshold (stage 1).
        track_buffer: Max frames before lost track is removed.
        match_thresh: IoU threshold for stage-1 assignment.
        min_box_area: Minimum bbox area to consider.
    """

    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, min_box_area: float = 10.0):
        super().__init__()
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self._active: list[dict] = []
        self._lost: list[dict] = []
        self._next_id = 1
        self._kf = KalmanFilter()
        self._frame_id = 0

    def setup(self, config: dict):
        self._active = []
        self._lost = []
        self._next_id = 1
        self._frame_id = 0
        super().setup(config)

    @staticmethod
    def _iou_batch(tracks: list[dict], dets: list[dict]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.array([[t["x1"], t["y1"], t["x2"], t["y2"]] for t in tracks])
        db = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
        ta = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
        da = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
        ix1 = np.maximum(tb[:, 0:1], db[:, 0])
        iy1 = np.maximum(tb[:, 1:2], db[:, 1])
        ix2 = np.minimum(tb[:, 2:3], db[:, 2])
        iy2 = np.minimum(tb[:, 3:4], db[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = ta[:, None] + da[None, :] - inter
        return inter / (union + 1e-6)

    def _assign(self, tracks, dets, iou_thresh):
        iou = self._iou_batch(tracks, dets)
        try:
            from scipy.optimize import linear_sum_assignment
            row, col = linear_sum_assignment(-iou)
        except ImportError:
            row, col = [], []
            used_r, used_c = set(), set()
            for idx in np.argsort(-iou.flatten()):
                r, c = divmod(idx, iou.shape[1])
                if r not in used_r and c not in used_c:
                    row.append(r); col.append(c)
                    used_r.add(r); used_c.add(c)
            row, col = np.array(row), np.array(col)
        matched_t, matched_d, unmatched_t, unmatched_d = [], [], [], []
        for r, c in zip(row, col):
            if iou[r, c] >= iou_thresh:
                matched_t.append(r); matched_d.append(c)
            else:
                unmatched_t.append(r); unmatched_d.append(c)
        unmatched_t += [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d += [j for j in range(len(dets)) if j not in matched_d]
        return list(zip(matched_t, matched_d)), list(set(unmatched_t)), list(set(unmatched_d))

    def _predict_all(self, tracks):
        for t in tracks:
            t["mean"], t["cov"] = self._kf.predict(t["mean"], t["cov"])
            xyxy = self._kf.to_xyxy(t["mean"])
            t["x1"], t["y1"], t["x2"], t["y2"] = xyxy

    def _execute(self, data, config):
        self._frame_id += 1
        bboxes = data.get("bboxes", []) if isinstance(data, dict) else []
        bboxes = [b for b in bboxes if (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) >= self.min_box_area]

        high_dets = [b for b in bboxes if b.get("conf", 1.0) >= self.track_thresh]
        low_dets = [b for b in bboxes if b.get("conf", 1.0) < self.track_thresh]

        self._predict_all(self._active)
        self._predict_all(self._lost)

        # Stage 1: high-conf dets vs active tracks
        pairs1, unm_t1, unm_d1 = self._assign(self._active, high_dets, self.match_thresh)
        for ti, di in pairs1:
            det = high_dets[di]
            bbox = np.array([det["x1"], det["y1"], det["x2"], det["y2"]])
            self._active[ti]["mean"], self._active[ti]["cov"] = self._kf.update(self._active[ti]["mean"], self._active[ti]["cov"], bbox)
            self._active[ti].update({"hits": self._active[ti]["hits"]+1, "age": 0, **{k: det[k] for k in ("x1","y1","x2","y2","label","conf")}})

        # Stage 2: low-conf dets vs unmatched active
        rem_tracks = [self._active[i] for i in unm_t1]
        pairs2, unm_t2, unm_d2 = self._assign(rem_tracks, low_dets, 0.5)
        for ti, di in pairs2:
            det = low_dets[di]
            rem_tracks[ti].update({"age": 0, **{k: det[k] for k in ("x1","y1","x2","y2","label","conf")}})

        # Mark still-unmatched as lost
        for i in unm_t2:
            rem_tracks[i]["age"] = rem_tracks[i].get("age", 0) + 1
            if rem_tracks[i]["age"] > self.track_buffer:
                self._active = [t for t in self._active if t["track_id"] != rem_tracks[i]["track_id"]]

        # New detections
        for j in unm_d1:
            det = high_dets[j]
            bbox = np.array([det["x1"], det["y1"], det["x2"], det["y2"]])
            mean, cov = self._kf.initiate(bbox)
            self._active.append({"track_id": self._next_id, "mean": mean, "cov": cov,
                                  "hits": 1, "age": 0, "state": "tentative", **det})
            self._next_id += 1

        self._active = [t for t in self._active if t.get("age", 0) <= self.track_buffer]

        tracks_out = [{"track_id": t["track_id"], "x1": t["x1"], "y1": t["y1"],
                        "x2": t["x2"], "y2": t["y2"], "label": t.get("label", ""),
                        "conf": t.get("conf", 1.0), "hits": t["hits"]} for t in self._active]

        payload = data if isinstance(data, dict) else {"frame": data}
        payload["tracks"] = tracks_out
        return payload

    def cleanup(self):
        self._active = []
        self._lost = []
        self._frame_id = 0
