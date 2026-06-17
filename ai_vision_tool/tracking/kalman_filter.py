from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Kalman filter for bounding box tracking (SORT/ByteTrack formulation).

    State: [cx, cy, s, r, vx, vy, vs]
    where s=scale(area), r=aspect_ratio, vx,vy,vs=velocities.
    Measurement: [cx, cy, s, r]
    """

    def __init__(self):
        dt = 1.0
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt

        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

        self.P0 = np.eye(7, dtype=np.float64)
        self.P0[4:, 4:] *= 1000.0

    @staticmethod
    def _xyxy_to_state(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / (y2 - y1 + 1e-6)
        return np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def _state_to_xyxy(mean: np.ndarray) -> np.ndarray:
        cx, cy, s, r = mean[:4]
        w = np.sqrt(max(s * r, 0.0))
        h = np.sqrt(max(s / (r + 1e-6), 0.0))
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def initiate(self, bbox_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = self._xyxy_to_state(bbox_xyxy)
        covariance = self.P0.copy()
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, bbox_xyxy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        z = self._xyxy_to_state(bbox_xyxy)[:4]
        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ mean
        mean = mean + K @ y
        covariance = (np.eye(7) - K @ self.H) @ covariance
        return mean, covariance

    def to_xyxy(self, mean: np.ndarray) -> np.ndarray:
        return self._state_to_xyxy(mean)

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T
        diff = measurements - projected_mean
        try:
            inv_S = np.linalg.inv(projected_cov + self.R)
        except np.linalg.LinAlgError:
            inv_S = np.eye(4)
        return np.einsum("ni,ij,nj->n", diff, inv_S, diff)
