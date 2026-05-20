from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame
from ai_vision_tool.utils.color_palette import ColorPalette


class SAMSegmenter(AIVisionComponent):
    """Segment Anything Model (Kirillov et al., Meta 2023) wrapper.

    Args:
        model_path: Path to SAM checkpoint file.
        model_type: "vit_b", "vit_l", or "vit_h".
        device: "auto", "cpu", "cuda", or "mps".
        points_per_side: Grid resolution for automatic mask generation.
        pred_iou_thresh: Predicted IoU threshold.
        stability_score_thresh: Stability score threshold.
    """

    def __init__(self, model_path: str | None = None, model_type: str = "vit_b",
                 device: str = "auto", points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88, stability_score_thresh: float = 0.95):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self._predictor = None
        self._mask_gen = None
        self._available = False
        self._palette = ColorPalette()

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def setup(self, config: dict):
        if not self.model_path:
            super().setup(config)
            return
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
            device = self._resolve_device()
            sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
            sam.to(device)
            self._predictor = SamPredictor(sam)
            self._mask_gen = SamAutomaticMaskGenerator(
                sam, points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
            )
            self._available = True
        except ImportError:
            raise ImportError("Install with: pip install segment-anything")
        super().setup(config)

    def _execute(self, data, config):
        frame = extract_frame(data)
        pred_iou = float(config.get("pred_iou_thresh", self.pred_iou_thresh))
        stab = float(config.get("stability_score_thresh", self.stability_score_thresh))

        if not self._available:
            payload = data if isinstance(data, dict) else {"frame": frame}
            payload["masks"] = []
            payload["warning"] = "SAMSegmenter: model not loaded (model_path required)"
            return payload

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prompts = data.get("prompts") if isinstance(data, dict) else None

        if prompts:
            self._predictor.set_image(rgb)
            points, labels, boxes = [], [], None
            for p in prompts:
                if p["type"] == "point":
                    points.append(p["coords"])
                    labels.append(p.get("label", 1))
                elif p["type"] == "box":
                    boxes = np.array(p["bbox"])
            import torch
            pt_arr = np.array(points) if points else None
            lb_arr = np.array(labels) if labels else None
            masks_np, scores, _ = self._predictor.predict(
                point_coords=pt_arr, point_labels=lb_arr,
                box=boxes[np.newaxis] if boxes is not None else None,
                multimask_output=True,
            )
            sam_masks = [{"mask": m, "label": "object", "conf": float(s)}
                         for m, s in zip(masks_np, scores)]
        else:
            raw = self._mask_gen.generate(rgb)
            sam_masks = [{"mask": m["segmentation"], "label": "object",
                           "conf": float(m["predicted_iou"]),
                           "bbox": m["bbox"]} for m in raw]

        overlay = frame.copy()
        for i, m in enumerate(sam_masks):
            color = self._palette[i]
            colored = np.zeros_like(frame)
            colored[m["mask"]] = color
            overlay = cv2.addWeighted(overlay, 0.7, colored, 0.3, 0)

        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["masks"] = sam_masks
        payload["sam_overlay"] = overlay
        return payload
