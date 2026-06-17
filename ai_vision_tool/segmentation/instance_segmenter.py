from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.color_palette import ColorPalette
from ai_vision_tool.utils.image_utils import extract_frame


class InstanceSegmenter(AIVisionComponent):
    """Instance segmentation using YOLO (ultralytics) or ONNX backends.

    Args:
        model_path: Path to model (.pt for YOLO *-seg model, .onnx for ONNX).
        backend: "yolo" or "onnx".
        conf_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        class_names: Class names list.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "yolo",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: list[str] | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.backend = backend
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []
        self._model = None
        self._palette = ColorPalette()

    def setup(self, config: dict):
        if not self.model_path:
            super().setup(config)
            return
        if self.backend == "yolo":
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)
                if not self.class_names and hasattr(self._model, "names"):
                    self.class_names = list(self._model.names.values())
            except ImportError as e:
                raise ImportError("Install with: pip install ultralytics") from e
        elif self.backend == "onnx":
            try:
                import onnxruntime as ort

                providers = [
                    p
                    for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if p in ort.get_available_providers()
                ]
                self._model = ort.InferenceSession(self.model_path, providers=providers)
            except ImportError as e:
                raise ImportError("Install with: pip install onnxruntime") from e
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        h, w = frame.shape[:2]
        conf_thr = float(config.get("conf_threshold", self.conf_threshold))
        iou_thr = float(config.get("iou_threshold", self.iou_threshold))
        masks, bboxes = [], []

        if self._model is None:
            payload = data if isinstance(data, dict) else {"frame": frame}
            payload["masks"] = []
            payload["bboxes"] = []
            return payload

        if self.backend == "yolo":
            results = self._model(frame, conf=conf_thr, iou=iou_thr, verbose=False)
            for r in results:
                if r.masks is None:
                    continue
                for _i, (mask_data, box) in enumerate(
                    zip(r.masks.data, r.boxes, strict=False)
                ):
                    mask_np = mask_data.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w, h)) > 0.5
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    label = (
                        self.class_names[cls]
                        if cls < len(self.class_names)
                        else str(cls)
                    )
                    conf = float(box.conf[0])
                    masks.append(
                        {
                            "mask": mask_resized,
                            "label": label,
                            "conf": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )
                    bboxes.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "label": label,
                            "conf": conf,
                        }
                    )

        # Build overlay
        overlay = frame.copy()
        for _i, m in enumerate(masks):
            label = m.get("label", "")
            color = self._palette.get(label)
            colored = np.zeros_like(frame)
            colored[m["mask"]] = color
            overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["masks"] = masks
        payload["bboxes"] = bboxes
        payload["instance_overlay"] = overlay
        return payload
