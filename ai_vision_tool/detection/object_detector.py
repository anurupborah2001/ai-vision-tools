from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame, replace_frame


class ObjectDetector(AIVisionComponent):
    """Object detection component supporting YOLO (ultralytics) and ONNX backends.

    Args:
        model_path: Path to model weights (.pt for YOLO, .onnx for ONNX backend).
        conf_threshold: Minimum confidence score.
        iou_threshold: NMS IoU threshold.
        class_names: List of class names for ONNX backend.
        backend: "yolo", "onnx", or "auto".
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 class_names: list[str] | None = None, backend: str = "auto"):
        super().__init__()
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []
        self.backend = backend
        self._model = None
        self._backend_used = ""

    def setup(self, config: dict):
        backend = config.get("backend", self.backend)
        if backend in ("yolo", "auto"):
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                if not self.class_names and hasattr(self._model, "names"):
                    self.class_names = list(self._model.names.values())
                self._backend_used = "yolo"
                super().setup(config)
                return
            except ImportError:
                if backend == "yolo":
                    raise ImportError("Install with: pip install ultralytics")
        try:
            import onnxruntime as ort
            providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if p in ort.get_available_providers()]
            self._model = ort.InferenceSession(self.model_path, providers=providers)
            self._backend_used = "onnx"
        except ImportError:
            raise ImportError("Install with: pip install onnxruntime  # or ultralytics")
        super().setup(config)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[1:][iou <= iou_threshold]
        return keep

    def _parse_yolo(self, result) -> list[dict]:
        dets = []
        for r in result:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.class_names[cls] if cls < len(self.class_names) else str(cls)
                dets.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label, "conf": conf})
        return dets

    def _execute(self, data, config):
        frame = extract_frame(data)
        conf_thr = float(config.get("conf_threshold", self.conf_threshold))
        iou_thr = float(config.get("iou_threshold", self.iou_threshold))

        if self._backend_used == "yolo":
            result = self._model(frame, conf=conf_thr, iou=iou_thr, verbose=False)
            dets = self._parse_yolo(result)
        else:
            inp = self._model.get_inputs()[0]
            h, w = inp.shape[2], inp.shape[3]
            resized = cv2.resize(frame, (w, h))
            tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis] / 255.0
            tensor = np.transpose(tensor, (0, 3, 1, 2))
            outputs = self._model.run(None, {inp.name: tensor})[0]
            dets = self._parse_onnx_output(outputs, frame.shape, conf_thr, iou_thr)

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["bboxes"] = dets
        payload["detection_count"] = len(dets)
        return payload

    def _parse_onnx_output(self, output, orig_shape, conf_thr, iou_thr) -> list[dict]:
        oh, ow = orig_shape[:2]
        dets = []
        if output.ndim == 3:
            output = output[0]
        if output.shape[-1] > 6:
            output = output.T
        boxes, scores_all = output[:, :4], output[:, 4:]
        class_ids = scores_all.argmax(axis=1)
        confs = scores_all.max(axis=1)
        mask = confs >= conf_thr
        boxes, class_ids, confs = boxes[mask], class_ids[mask], confs[mask]
        if len(boxes) == 0:
            return []
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        keep = self._nms(xyxy, confs, iou_thr)
        for k in keep:
            label = self.class_names[class_ids[k]] if class_ids[k] < len(self.class_names) else str(class_ids[k])
            dets.append({"x1": float(xyxy[k, 0] * ow), "y1": float(xyxy[k, 1] * oh),
                         "x2": float(xyxy[k, 2] * ow), "y2": float(xyxy[k, 3] * oh),
                         "label": label, "conf": float(confs[k])})
        return dets
