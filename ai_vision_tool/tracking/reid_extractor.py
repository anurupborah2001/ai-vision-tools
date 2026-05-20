from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class ReIDExtractor(AIVisionComponent):
    """Extracts appearance embeddings for re-identification.

    Args:
        method: "hog" (always available), "osnet" (ONNX), or "clip_roi" (CLIP).
        model_path: Path to ONNX model (osnet method).
        embedding_dim: Embedding dimension (osnet/clip).
    """

    def __init__(self, method: str = "hog", model_path: str | None = None, embedding_dim: int = 128):
        super().__init__()
        self.method = method
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self._hog: cv2.HOGDescriptor | None = None
        self._session = None
        self._clip = None

    def setup(self, config: dict):
        if self.method == "hog":
            self._hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        elif self.method == "osnet":
            if not self.model_path:
                raise ValueError("ReIDExtractor: model_path required for osnet method")
            try:
                import onnxruntime as ort
                providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if p in ort.get_available_providers()]
                self._session = ort.InferenceSession(self.model_path, providers=providers)
            except ImportError:
                raise ImportError("Install with: pip install onnxruntime")
        elif self.method == "clip_roi":
            try:
                from transformers import CLIPProcessor, CLIPModel
                self._clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip.eval()
            except ImportError:
                raise ImportError("Install with: pip install transformers torch")
        super().setup(config)

    def _embed(self, frame: np.ndarray, bbox: dict) -> np.ndarray:
        x1, y1, x2, y2 = max(0, int(bbox["x1"])), max(0, int(bbox["y1"])), \
                          min(frame.shape[1], int(bbox["x2"])), min(frame.shape[0], int(bbox["y2"]))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(self.embedding_dim)

        if self.method == "hog":
            roi_r = cv2.resize(roi, (64, 128))
            gray = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
            feat = self._hog.compute(gray).flatten()
        elif self.method == "osnet" and self._session:
            inp = self._session.get_inputs()[0]
            ih, iw = inp.shape[2], inp.shape[3]
            roi_r = cv2.resize(roi, (iw, ih))
            tensor = cv2.cvtColor(roi_r, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis] / 255.0
            tensor = np.transpose(tensor, (0, 3, 1, 2))
            feat = self._session.run(None, {inp.name: tensor})[0].flatten()
        elif self.method == "clip_roi" and self._clip:
            import torch
            roi_rgb = cv2.cvtColor(cv2.resize(roi, (224, 224)), cv2.COLOR_BGR2RGB)
            from PIL import Image
            inputs = self._clip_proc(images=Image.fromarray(roi_rgb), return_tensors="pt")
            with torch.no_grad():
                feat = self._clip.get_image_features(**inputs).squeeze().numpy()
        else:
            feat = roi.astype(np.float32).flatten()

        norm = np.linalg.norm(feat)
        feat = feat / (norm + 1e-6)
        if len(feat) > self.embedding_dim:
            feat = feat[:self.embedding_dim]
        elif len(feat) < self.embedding_dim:
            feat = np.pad(feat, (0, self.embedding_dim - len(feat)))
        return feat

    def _execute(self, data, config):
        frame = extract_frame(data)
        sources = data.get("tracks", data.get("bboxes", [])) if isinstance(data, dict) else []
        embeddings = []
        for item in sources:
            emb = self._embed(frame, item)
            embeddings.append({"track_id": item.get("track_id", -1), "embedding": emb})

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["embeddings"] = embeddings
        return payload

    @classmethod
    def cosine_similarity(cls, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    @staticmethod
    def build_gallery(embeddings: list[dict]) -> dict[int, np.ndarray]:
        gallery: dict[int, list] = {}
        for e in embeddings:
            tid = e.get("track_id", -1)
            emb = e.get("embedding")
            if emb is not None:
                gallery.setdefault(tid, []).append(emb)
        return {tid: np.mean(embs, axis=0) for tid, embs in gallery.items()}
