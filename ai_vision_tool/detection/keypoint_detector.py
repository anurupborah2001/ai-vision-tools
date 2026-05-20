from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame

_MP_POSE_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]


class KeypointDetector(AIVisionComponent):
    """Detects human pose keypoints using MediaPipe or YOLO-pose.

    Args:
        backend: "mediapipe" or "yolo".
        model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy).
    """

    def __init__(self, backend: str = "mediapipe", model_complexity: int = 1):
        super().__init__()
        self.backend = backend
        self.model_complexity = model_complexity
        self._detector = None
        self._model = None

    def setup(self, config: dict):
        static = config.get("static_image_mode", True)
        if self.backend == "mediapipe":
            try:
                import mediapipe as mp
                self._detector = mp.solutions.pose.Pose(
                    model_complexity=self.model_complexity,
                    static_image_mode=static,
                    enable_segmentation=False,
                )
            except ImportError:
                raise ImportError("Install with: pip install mediapipe")
        elif self.backend == "yolo":
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8n-pose.pt")
            except ImportError:
                raise ImportError("Install with: pip install ultralytics")
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        h, w = frame.shape[:2]
        poses = []

        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._detector.process(rgb)
            if result.pose_landmarks:
                kps = []
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    kps.append({
                        "name": _MP_POSE_NAMES[i] if i < len(_MP_POSE_NAMES) else f"kp_{i}",
                        "x": lm.x * w, "y": lm.y * h, "z": lm.z,
                        "visibility": lm.visibility,
                    })
                poses.append({"keypoints": kps, "bbox": None})

        elif self.backend == "yolo" and self._model:
            results = self._model(frame, verbose=False)
            for r in results:
                if r.keypoints is None:
                    continue
                for kp_set, box in zip(r.keypoints.xy, r.boxes.xyxy):
                    kps = [{"name": f"kp_{i}", "x": float(k[0]), "y": float(k[1]),
                             "z": 0.0, "visibility": 1.0} for i, k in enumerate(kp_set)]
                    x1, y1, x2, y2 = box.tolist()
                    poses.append({"keypoints": kps, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["poses"] = poses
        return payload

    def cleanup(self):
        if self.backend == "mediapipe" and self._detector:
            self._detector.close()
