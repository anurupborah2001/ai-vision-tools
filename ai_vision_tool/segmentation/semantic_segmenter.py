from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame
from ai_vision_tool.utils.color_palette import ColorPalette

_VOC21 = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
          "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
          "sheep","sofa","train","tvmonitor"]


class SemanticSegmenter(AIVisionComponent):
    """Semantic segmentation component supporting ONNX, OpenCV DNN, and TorchScript backends.

    Args:
        model_path: Path to model file.
        backend: "onnx", "opencv_dnn", or "torchscript".
        num_classes: Number of segmentation classes.
        class_names: Class name list (defaults to VOC21 if num_classes=21).
        conf_threshold: Confidence threshold for mask inclusion.
    """

    def __init__(self, model_path: str | None = None, backend: str = "onnx",
                 num_classes: int = 21, class_names: list[str] | None = None,
                 conf_threshold: float = 0.5):
        super().__init__()
        self.model_path = model_path
        self.backend = backend
        self.num_classes = num_classes
        self.class_names = class_names or (_VOC21 if num_classes == 21 else [f"class_{i}" for i in range(num_classes)])
        self.conf_threshold = conf_threshold
        self._model = None
        self._palette = ColorPalette(num_classes)

    def setup(self, config: dict):
        if not self.model_path:
            super().setup(config)
            return
        if self.backend == "onnx":
            try:
                import onnxruntime as ort
                providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                             if p in ort.get_available_providers()]
                self._model = ort.InferenceSession(self.model_path, providers=providers)
            except ImportError:
                raise ImportError("Install with: pip install onnxruntime")
        elif self.backend == "opencv_dnn":
            self._model = cv2.dnn.readNetFromONNX(self.model_path)
        elif self.backend == "torchscript":
            try:
                import torch
                self._model = torch.jit.load(self.model_path)
                self._model.eval()
            except ImportError:
                raise ImportError("Install with: pip install torch")
        super().setup(config)

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        inp_size = (512, 512)
        resized = cv2.resize(frame, inp_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis]

        if self.backend == "onnx" and self._model:
            inp_name = self._model.get_inputs()[0].name
            out = self._model.run(None, {inp_name: tensor})[0]
        elif self.backend == "opencv_dnn" and self._model:
            blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, inp_size, swapRB=True)
            self._model.setInput(blob)
            out = self._model.forward()
        elif self.backend == "torchscript" and self._model:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(tensor)
                out = self._model(t).numpy()
        else:
            return np.zeros((h, w), dtype=np.uint8)

        seg_map = np.argmax(out[0], axis=0).astype(np.uint8)
        return cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)

    def _execute(self, data, config):
        frame = extract_frame(data)
        h, w = frame.shape[:2]
        overlay_alpha = float(config.get("overlay_alpha", 0.5))

        seg_map = self._infer(frame)

        # Build colorized overlay
        color_mask = np.zeros_like(frame)
        for cls_id in np.unique(seg_map):
            color_mask[seg_map == cls_id] = self._palette[int(cls_id)]
        overlay = cv2.addWeighted(frame, 1 - overlay_alpha, color_mask, overlay_alpha, 0)

        # Class pixel counts
        class_counts = {self.class_names[i]: int((seg_map == i).sum())
                        for i in np.unique(seg_map) if i < len(self.class_names)}

        # Build mask list
        masks = []
        for cls_id in np.unique(seg_map):
            if cls_id == 0:
                continue
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            masks.append({"mask": (seg_map == cls_id), "label": label, "conf": 1.0})

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["seg_map"] = seg_map
        payload["seg_overlay"] = overlay
        payload["class_counts"] = class_counts
        payload["masks"] = masks
        return payload
