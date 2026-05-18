from __future__ import annotations

import base64
import copy
import importlib
from typing import Any

import cv2
import numpy as np


COMPONENT_MODULES = {
    "preprocessing": "visionflow.components.preprocessing",
    "augmentations": "visionflow.components.augmentations",
}

DIRECT_COMPONENTS = {
    "components": {
        "FrameEnhancer": "visionflow.components.frame_enhancer",
        "FrameResizer": "visionflow.components.frame_resizer",
        "MotionDetector": "visionflow.components.motion_detector",
        "FrameAnnotator": "visionflow.components.frame_annotator",
        "DatasetCollector": "visionflow.components.dataset_collector",
        "TimeLapseCapture": "visionflow.components.time_lapse_capture",
        "PictureTaker": "visionflow.components.picture_taker",
        "BurstPictureTaker": "visionflow.components.burst_picture_taker",
        "ROICapture": "visionflow.components.roi_capture",
        "ImageExporter": "visionflow.components.image_exporter",
        "FrameGrabber": "visionflow.components.frame_grabber",
        "VideoTaker": "visionflow.components.video_taker",
        "AutoLabeller": "visionflow.components.auto_labeller",
        "DarknetAutoLabeler": "visionflow.components.darknet_auto_labeler",
        "TensorFlowAutoLabeler": "visionflow.components.tensorflow_auto_labeler",
    }
}

INTERACTIVE_COMPONENTS = {
    "PictureTaker",
    "BurstPictureTaker",
    "ROICapture",
    "VideoTaker",
}


def _category_exports(category: str) -> dict[str, str]:
    if category in COMPONENT_MODULES:
        module_name = COMPONENT_MODULES[category]
        module = importlib.import_module(module_name)
        return {name: module_name for name in getattr(module, "__all__", [])}
    if category in DIRECT_COMPONENTS:
        return DIRECT_COMPONENTS[category]
    raise KeyError(f"Unknown category: {category}")


def list_components() -> dict[str, list[dict[str, Any]]]:
    categories = {}
    for category in ("preprocessing", "augmentations", "components"):
        items = []
        for name, module_name in sorted(_category_exports(category).items()):
            module = importlib.import_module(module_name)
            cls = getattr(module, name)
            items.append(
                {
                    "name": name,
                    "module": module_name,
                    "description": (cls.__doc__ or "").strip(),
                    "interactive": name in INTERACTIVE_COMPONENTS,
                }
            )
        categories[category] = items
    return categories


def get_component(category: str, name: str):
    exports = _category_exports(category.lower())
    if name not in exports:
        raise KeyError(f"Unknown component '{name}' in category '{category}'")
    module = importlib.import_module(exports[name])
    return getattr(module, name)


def instantiate_component(category: str, name: str, init_args: dict[str, Any] | None = None):
    component_cls = get_component(category, name)
    return component_cls(**(init_args or {}))


def decode_image_base64(encoded: str, grayscale: bool = False) -> np.ndarray:
    content = encoded.split(",", 1)[1] if "," in encoded else encoded
    data = base64.b64decode(content)
    arr = np.frombuffer(data, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(arr, flag)
    if image is None:
        raise ValueError("Unable to decode base64 image content.")
    return image


def encode_image_base64(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Unable to encode image output.")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _materialize_dict(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[key] = _materialize_dict(value)
        elif isinstance(value, list):
            result[key] = [_materialize_value(item) for item in value]
        elif key.endswith("_image_base64") and isinstance(value, str):
            result[key[: -len("_base64")]] = decode_image_base64(value)
        elif key.endswith("_mask_base64") and isinstance(value, str):
            result[key[: -len("_base64")]] = decode_image_base64(value, grayscale=True)
        elif key == "frame_base64" and isinstance(value, str):
            result["frame"] = decode_image_base64(value)
        elif key == "mask_base64" and isinstance(value, str):
            result["mask"] = decode_image_base64(value, grayscale=True)
        else:
            result[key] = value
    return result


def _materialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _materialize_dict(value)
    if isinstance(value, list):
        return [_materialize_value(item) for item in value]
    return value


def materialize_request_data(
    *,
    image_base64: str | None = None,
    payload: dict[str, Any] | None = None,
    data: Any = None,
    batch: list[Any] | None = None,
) -> Any:
    if batch is not None:
        return [_materialize_value(item) for item in batch]

    if payload is not None:
        materialized = _materialize_dict(copy.deepcopy(payload))
        if image_base64 and "frame" not in materialized:
            materialized["frame"] = decode_image_base64(image_base64)
        return materialized

    if image_base64 is not None:
        return {"frame": decode_image_base64(image_base64)}

    if data is not None:
        return _materialize_value(copy.deepcopy(data))

    return None


def serialize_result(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "type": "image" if value.ndim in (2, 3) else "array",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "base64": encode_image_base64(value) if value.ndim in (2, 3) else None,
            "values": value.tolist() if value.ndim < 2 else None,
        }
    if isinstance(value, dict):
        return {key: serialize_result(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_result(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_result(item) for item in value]
    return value


def execute_component(
    *,
    category: str,
    name: str,
    init_args: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    image_base64: str | None = None,
    payload: dict[str, Any] | None = None,
    data: Any = None,
    batch: list[Any] | None = None,
) -> dict[str, Any]:
    component = instantiate_component(category, name, init_args)
    request_data = materialize_request_data(
        image_base64=image_base64,
        payload=payload,
        data=data,
        batch=batch,
    )
    result = component.run(request_data, config or {})
    return {
        "category": category,
        "name": name,
        "interactive": name in INTERACTIVE_COMPONENTS,
        "result": serialize_result(result),
    }
