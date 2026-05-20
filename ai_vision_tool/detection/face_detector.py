from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame, replace_frame


class FaceDetector(AIVisionComponent):
    """Detects faces using OpenCV Haar cascades or MediaPipe.

    Args:
        backend: "opencv" (Haar cascade) or "mediapipe".
        conf_threshold: Minimum detection confidence (mediapipe).
        min_face_size: Minimum face size in pixels (opencv).
    """

    def __init__(self, backend: str = "opencv", conf_threshold: float = 0.5, min_face_size: int = 20):
        super().__init__()
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.min_face_size = min_face_size
        self._detector = None

    def setup(self, config: dict):
        if self.backend == "mediapipe":
            try:
                import mediapipe as mp
                self._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=self.conf_threshold
                )
            except ImportError:
                raise ImportError("Install with: pip install mediapipe")
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade_path)
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        h, w = frame.shape[:2]
        conf_thr = float(config.get("conf_threshold", self.conf_threshold))
        min_sz = int(config.get("min_face_size", self.min_face_size))
        faces = []

        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._detector.process(rgb)
            if result.detections:
                for det in result.detections:
                    bb = det.location_data.relative_bounding_box
                    x1 = int(bb.xmin * w)
                    y1 = int(bb.ymin * h)
                    x2 = int((bb.xmin + bb.width) * w)
                    y2 = int((bb.ymin + bb.height) * h)
                    conf = det.score[0]
                    faces.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "landmarks": []})
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_sz, min_sz)
            )
            for (x, y, fw, fh) in (rects if len(rects) > 0 else []):
                faces.append({"x1": int(x), "y1": int(y), "x2": int(x + fw), "y2": int(y + fh),
                               "conf": 1.0, "landmarks": []})

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["faces"] = faces
        payload["bboxes"] = [dict(f, label="face") for f in faces]
        return payload

    def cleanup(self):
        if self.backend == "mediapipe" and self._detector:
            self._detector.close()
            self._detector = None
