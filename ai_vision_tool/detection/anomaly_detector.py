from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


class AnomalyDetector(AIVisionComponent):
    """Frame-level anomaly detection using statistical, patchcore, or PCA-based methods.

    Args:
        method: "statistical" (z-score on histogram), "patchcore" (NN on HOG patches),
                or "autoencoder_approx" (PCA on patches).
        window: Warmup frames used to build reference distribution.
        threshold: Anomaly score threshold.
        patch_size: Patch size for patchcore/autoencoder methods.
    """

    def __init__(self, method: str = "statistical", window: int = 30, threshold: float = 2.0,
                 patch_size: int = 64):
        super().__init__()
        self.method = method
        self.window = window
        self.threshold = threshold
        self.patch_size = patch_size
        self._warmup_frames: list = []
        self._ready = False
        self._ref_mean: np.ndarray | None = None
        self._ref_std: np.ndarray | None = None
        self._nn = None
        self._pca = None
        self._hog = None

    def setup(self, config: dict):
        if self.method == "patchcore":
            self._hog = cv2.HOGDescriptor((self.patch_size, self.patch_size), (16, 16), (8, 8), (8, 8), 9)
        super().setup(config)

    def _hist_features(self, frame: np.ndarray) -> np.ndarray:
        feats = []
        for c in range(3):
            hist = cv2.calcHist([frame], [c], None, [64], [0, 256]).flatten()
            feats.append(hist / (hist.sum() + 1e-6))
        return np.concatenate(feats)

    def _extract_patches(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        patches = []
        for y in range(0, h - self.patch_size, self.patch_size // 2):
            for x in range(0, w - self.patch_size, self.patch_size // 2):
                patch = cv2.resize(frame[y:y+self.patch_size, x:x+self.patch_size], (self.patch_size, self.patch_size))
                if self.method == "patchcore" and self._hog:
                    feat = self._hog.compute(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)).flatten()
                else:
                    feat = patch.astype(np.float32).flatten() / 255.0
                patches.append(feat)
        return np.array(patches) if patches else np.zeros((1, self.patch_size * self.patch_size * 3))

    def _build_reference(self):
        if self.method == "statistical":
            feats = np.array([self._hist_features(f) for f in self._warmup_frames])
            self._ref_mean = feats.mean(axis=0)
            self._ref_std = feats.std(axis=0) + 1e-6
        elif self.method == "patchcore":
            try:
                from sklearn.neighbors import NearestNeighbors
                all_patches = np.concatenate([self._extract_patches(f) for f in self._warmup_frames])
                self._nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
                self._nn.fit(all_patches)
            except ImportError:
                print("[AnomalyDetector] sklearn not available; falling back to statistical method")
                self.method = "statistical"
                self._build_reference()
        elif self.method == "autoencoder_approx":
            try:
                from sklearn.decomposition import PCA
                all_patches = np.concatenate([self._extract_patches(f) for f in self._warmup_frames])
                n_comp = min(50, all_patches.shape[0], all_patches.shape[1])
                self._pca = PCA(n_components=n_comp)
                self._pca.fit(all_patches)
            except ImportError:
                print("[AnomalyDetector] sklearn not available; falling back to statistical method")
                self.method = "statistical"
                self._build_reference()
        self._ready = True

    def _score(self, frame: np.ndarray) -> tuple[float, np.ndarray | None]:
        if self.method == "statistical":
            feat = self._hist_features(frame)
            z = np.abs((feat - self._ref_mean) / self._ref_std)
            return float(z.max()), None
        elif self.method == "patchcore" and self._nn:
            patches = self._extract_patches(frame)
            dists, _ = self._nn.kneighbors(patches)
            score = float(dists.mean())
            h, w = frame.shape[:2]
            amap = np.full((h, w), score, dtype=np.float32)
            return score, amap
        elif self.method == "autoencoder_approx" and self._pca:
            patches = self._extract_patches(frame)
            reconstructed = self._pca.inverse_transform(self._pca.transform(patches))
            errors = np.mean((patches - reconstructed) ** 2, axis=1)
            return float(errors.max()), None
        return 0.0, None

    def _execute(self, data, config):
        frame = extract_frame(data)
        threshold = float(config.get("threshold", self.threshold))

        if not self._ready:
            self._warmup_frames.append(frame.copy())
            if len(self._warmup_frames) >= self.window:
                self._build_reference()
            payload = data if isinstance(data, dict) else {"frame": frame}
            payload["anomaly_score"] = 0.0
            payload["is_anomaly"] = False
            payload["anomaly_map"] = None
            return payload

        score, amap = self._score(frame)
        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["anomaly_score"] = round(score, 4)
        payload["is_anomaly"] = score > threshold
        payload["anomaly_map"] = amap
        return payload
