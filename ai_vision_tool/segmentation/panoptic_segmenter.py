from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.color_palette import ColorPalette
from ai_vision_tool.utils.image_utils import extract_frame


class PanopticSegmenter(AIVisionComponent):
    """Panoptic segmentation combining semantic (stuff) and instance (thing) predictions.

    Args:
        model_path: Path to ONNX panoptic model.
        backend: "onnx" (only backend currently supported).
        stuff_classes: Semantic class names (background, sky, road...).
        thing_classes: Instance class names (person, car...).
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "onnx",
        stuff_classes: list[str] | None = None,
        thing_classes: list[str] | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.backend = backend
        self.stuff_classes = stuff_classes or []
        self.thing_classes = thing_classes or []
        self._model = None
        self._palette = ColorPalette(200)

    def setup(self, config: dict):
        if not self.model_path:
            super().setup(config)
            return
        if self.backend == "onnx":
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

        if self._model is None:
            payload = data if isinstance(data, dict) else {"frame": frame}
            payload["panoptic_seg"] = np.zeros((h, w), dtype=np.int32)
            payload["segments_info"] = []
            payload["panoptic_overlay"] = frame.copy()
            payload["stuff_masks"] = []
            payload["thing_masks"] = []
            payload["warning"] = "PanopticSegmenter: no model loaded"
            return payload

        inp = self._model.get_inputs()[0]
        ih, iw = inp.shape[2], inp.shape[3]
        resized = cv2.resize(frame, (iw, ih))
        tensor = (
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis]
            / 255.0
        )
        tensor = np.transpose(tensor, (0, 3, 1, 2))
        outputs = self._model.run(None, {inp.name: tensor})

        seg_out = outputs[0][0] if outputs[0].ndim == 4 else outputs[0]
        panoptic_seg = np.argmax(seg_out, axis=0).astype(np.int32)
        panoptic_seg = cv2.resize(
            panoptic_seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

        unique_ids = np.unique(panoptic_seg)
        segments_info = []
        for seg_id in unique_ids:
            is_thing = seg_id < len(self.thing_classes)
            label = (
                self.thing_classes[seg_id]
                if is_thing and seg_id < len(self.thing_classes)
                else (
                    self.stuff_classes[seg_id - len(self.thing_classes)]
                    if seg_id - len(self.thing_classes) < len(self.stuff_classes)
                    else f"class_{seg_id}"
                )
            )
            segments_info.append(
                {"id": int(seg_id), "label": label, "is_thing": is_thing, "conf": 1.0}
            )

        overlay = np.zeros_like(frame)
        for seg in segments_info:
            overlay[panoptic_seg == seg["id"]] = self._palette[seg["id"]]
        panoptic_overlay = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        stuff_masks = [
            {"mask": panoptic_seg == s["id"], "label": s["label"]}
            for s in segments_info
            if not s["is_thing"]
        ]
        thing_masks = [
            {"mask": panoptic_seg == s["id"], "label": s["label"]}
            for s in segments_info
            if s["is_thing"]
        ]

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload.update(
            {
                "panoptic_seg": panoptic_seg,
                "segments_info": segments_info,
                "panoptic_overlay": panoptic_overlay,
                "stuff_masks": stuff_masks,
                "thing_masks": thing_masks,
            }
        )
        return payload
