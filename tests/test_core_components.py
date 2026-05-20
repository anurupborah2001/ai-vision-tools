import json

import cv2
import numpy as np

from ai_vision_tool.components.auto_labeller import AutoLabeller
from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.components.dataset_collector import DatasetCollector
from ai_vision_tool.components.frame_annotator import FrameAnnotator
from ai_vision_tool.components.frame_enhancer import FrameEnhancer
from ai_vision_tool.components.frame_resizer import FrameResizer
from ai_vision_tool.components.motion_detector import MotionDetector
from ai_vision_tool.components.time_lapse import TimeLapseCapture


class RecordingComponent(AIVisionComponent):
    def __init__(self):
        super().__init__()
        self.calls = []

    def preprocess(self, data, config):
        self.calls.append(("pre", data, config.copy()))
        return data + 1

    def _execute(self, data, config):
        self.calls.append(("exec", data, config.copy()))
        return data * 2

    def postprocess(self, result, config):
        self.calls.append(("post", result, config.copy()))
        return result - 3


def test_base_component_runs_single_item_and_initializes_once():
    component = RecordingComponent()

    result = component.run(4, {"flag": True})

    assert result == 7
    assert component.is_initialized is True
    assert [entry[0] for entry in component.calls] == ["pre", "exec", "post"]


def test_base_component_processes_batches():
    component = RecordingComponent()

    result = component.run([1, 2], {"mode": "batch"})

    assert result == [1, 3]
    assert [entry[0] for entry in component.calls] == ["pre", "exec", "post", "pre", "exec", "post"]


def test_auto_labeller_setup_downloads_weights(monkeypatch):
    component = AutoLabeller()
    called = {}

    def fake_download(url):
        called["url"] = url
        return True

    monkeypatch.setattr(component, "download_model_weights", fake_download)
    component.setup({"model_url": "https://example.com/model.bin"})

    assert component.is_initialized is True
    assert called["url"] == "https://example.com/model.bin"


def test_auto_labeller_execute_passthrough():
    component = AutoLabeller()

    assert component.run("frame-001", {}) == "frame-001"


def test_frame_enhancer_updates_dict_frame(sample_frame):
    component = FrameEnhancer()
    payload = {"frame": sample_frame.copy()}

    result = component.run(
        payload,
        {
            "brightness": 15,
            "contrast": 1.2,
            "sharpen": True,
            "denoise": False,
            "grayscale": True,
        },
    )

    assert result is payload
    assert result["frame"].shape == sample_frame.shape
    assert np.array_equal(result["frame"][:, :, 0], result["frame"][:, :, 1])


def test_frame_enhancer_handles_plain_frame(sample_frame):
    component = FrameEnhancer()

    result = component.run(sample_frame.copy(), {"brightness": 10})

    assert result.shape == sample_frame.shape
    assert not np.array_equal(result, sample_frame)


def test_frame_resizer_plain_frame(sample_frame):
    component = FrameResizer()

    result = component.run(sample_frame.copy(), {"size": (20, 10), "keep_aspect": False})

    assert result.shape == (10, 20, 3)


def test_frame_resizer_keep_aspect_returns_canvas(sample_frame):
    component = FrameResizer()
    payload = {"frame": sample_frame.copy()}

    result = component.run(payload, {"size": (100, 100), "keep_aspect": True})

    assert result["frame"].shape == (100, 100, 3)
    assert np.count_nonzero(result["frame"]) > 0


def test_frame_annotator_uses_payload_annotations(sample_frame):
    component = FrameAnnotator()
    payload = {
        "frame": sample_frame.copy(),
        "annotations": [
            {"type": "text", "text": "demo", "pos": (5, 10)},
            {"type": "box", "box": (5, 5, 10, 10)},
            {"type": "point", "point": (3, 3)},
        ],
    }

    result = component.run(payload, {})

    assert result is payload
    assert np.count_nonzero(result["frame"]) > np.count_nonzero(sample_frame)


def test_frame_annotator_uses_config_annotations_for_plain_frame(sample_frame):
    component = FrameAnnotator()

    result = component.run(
        sample_frame.copy(),
        {"annotations": [{"type": "text", "text": "cfg", "pos": (5, 15)}]},
    )

    assert result.shape == sample_frame.shape
    assert np.count_nonzero(result) > np.count_nonzero(sample_frame)


def test_motion_detector_first_frame_sets_baseline_without_boxes(sample_frame):
    component = MotionDetector()

    result = component.run({"frame": sample_frame.copy()}, {"min_area": 5})

    assert result["motion_boxes"] == []
    assert component.prev_gray is not None


def test_motion_detector_detects_motion_between_frames(sample_frame):
    component = MotionDetector()
    first = np.zeros_like(sample_frame)
    second = first.copy()
    second[10:25, 10:25] = 255

    component.run({"frame": first}, {"min_area": 5, "draw_motion": True})
    result = component.run({"frame": second}, {"min_area": 5, "draw_motion": True})

    assert result["motion_boxes"]
    x, y, w, h = result["motion_boxes"][0]
    assert w > 0 and h > 0


def test_motion_detector_can_skip_drawing(sample_frame):
    component = MotionDetector()
    first = np.zeros_like(sample_frame)
    second = first.copy()
    second[5:20, 5:20] = 255

    component.run({"frame": first}, {"min_area": 5, "draw_motion": False})
    result = component.run({"frame": second}, {"min_area": 5, "draw_motion": False})

    assert result["motion_boxes"]
    assert np.array_equal(result["frame"], second)


def test_dataset_collector_saves_image_and_metadata(tmp_path, sample_frame, stub_imwrite):
    component = DatasetCollector()
    payload = {"frame": sample_frame.copy()}

    component.run(
        payload,
        {
            "save_sample": True,
            "output_dir": str(tmp_path / "dataset"),
            "save_metadata": True,
            "label": "fox",
            "metadata": {"source": "unit-test"},
        },
    )

    metadata_path = tmp_path / "dataset" / "metadata.jsonl"
    label_dir = tmp_path / "dataset" / "fox"

    assert label_dir.exists()
    assert metadata_path.exists()
    lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["label"] == "fox"
    assert record["metadata"]["source"] == "unit-test"


def test_dataset_collector_passthrough_when_not_saving(tmp_path, sample_frame):
    component = DatasetCollector()
    payload = {"frame": sample_frame.copy()}

    result = component.run(payload, {"save_sample": False, "output_dir": str(tmp_path / "dataset")})

    assert result is payload
    assert not (tmp_path / "dataset").exists()


def test_time_lapse_capture_saves_on_interval(monkeypatch, tmp_path, sample_frame, created_files, stub_imwrite):
    times = iter([100.0, 102.0, 108.5])
    monkeypatch.setattr("ai_vision_tool.components.time_lapse.time.time", lambda: next(times))
    component = TimeLapseCapture(output_dir=tmp_path / "timelapse", interval_seconds=5, prefix="snap")

    payload = {"frame": sample_frame.copy()}
    component.run(payload, {})
    component.run(payload, {})
    component.run(payload, {})

    assert len(created_files) == 2
    assert all(path.parent == tmp_path / "timelapse" for path in created_files)
    assert all(path.name.startswith("snap_") for path in created_files)
