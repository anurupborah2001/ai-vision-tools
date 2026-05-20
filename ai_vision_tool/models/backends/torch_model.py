from __future__ import annotations

import cv2
import numpy as np

try:
    import torch  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "PyTorch backend requires: pip install ai-vision-tool[torch]"
    ) from exc

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame, replace_frame


class TorchModel(AIVisionComponent):
    """Generic TorchScript model inference component.

    Args:
        model_path: Path to TorchScript (.pt) file.
        device: "auto", "cpu", "cuda", "cuda:N", or "mps".
        half_precision: Use FP16 inference (CUDA only).
    """

    def __init__(self, model_path: str, device: str = "auto", half_precision: bool = False):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.half_precision = half_precision
        self._model = None
        self._device = None

    def _resolve_device(self):
        try:
            import torch
            if self.device == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda:0")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
                return torch.device("cpu")
            return torch.device(self.device)
        except ImportError:
            raise ImportError("Install with: pip install torch torchvision")

    def setup(self, config: dict):
        try:
            import torch
        except ImportError:
            raise ImportError("Install with: pip install torch torchvision")
        self._device = self._resolve_device()
        try:
            self._model = torch.jit.load(self.model_path, map_location=self._device)
        except Exception:
            self._model = torch.load(self.model_path, map_location=self._device)
        self._model.eval()
        if self.half_precision and str(self._device).startswith("cuda"):
            self._model = self._model.half()
        super().setup(config)

    def _execute(self, data, config):
        import torch
        frame = extract_frame(data)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(self._device)
        if self.half_precision and str(self._device).startswith("cuda"):
            tensor = tensor.half()
        with torch.no_grad():
            output = self._model(tensor)
        if isinstance(output, torch.Tensor):
            out_np = output.cpu().float().numpy()
        else:
            out_np = [o.cpu().float().numpy() for o in output]
        payload = data if isinstance(data, dict) else {"frame": frame}
        payload["model_output"] = out_np
        payload["device"] = str(self._device)
        return payload

    def cleanup(self):
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
