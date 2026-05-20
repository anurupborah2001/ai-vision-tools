from pathlib import Path

from ai_vision_tool.integrations.labeling.darknet_auto_labeler import DarknetAutoLabeler
from ai_vision_tool.integrations.labeling.tensorflow_auto_labeler import TensorFlowAutoLabeler


def test_darknet_auto_labeler_missing_file_returns_error():
    component = DarknetAutoLabeler()
    component.darknet_width = 416
    component.darknet_height = 416

    result = component._execute("missing.jpg", {})

    assert result["status"] == "error - file not found"


def test_darknet_auto_labeler_min_max_and_iou_helpers():
    component = DarknetAutoLabeler()

    coords = component._get_min_max([10, 10, 6, 4])
    assert coords == [7, 8, 13, 12]

    iou = component._calculate_iou(
        ["a", 0.9, (10, 10, 8, 8)],
        ["b", 0.8, (12, 10, 8, 8)],
    )
    assert 0 < iou < 1


def test_darknet_auto_labeler_filters_overlapping_boxes():
    component = DarknetAutoLabeler()

    detections = [
        ["cat", 0.95, (10, 10, 10, 10)],
        ["cat", 0.50, (11, 10, 10, 10)],
        ["dog", 0.90, (50, 50, 5, 5)],
    ]

    filtered = component._filter_iou(detections, 0.2)

    assert len(filtered) == 2
    assert filtered[0][0] == "cat"
    assert filtered[1][0] == "dog"


def test_darknet_auto_labeler_creates_xml(tmp_path):
    component = DarknetAutoLabeler()
    component.folder_name = "images"
    component.labeled_dir = str(tmp_path)
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"img")

    component._create_xml(str(image_path), [["fox", [1, 2, 3, 4]]], 100, 200)

    xml_path = tmp_path / "frame.xml"
    assert xml_path.exists()
    xml_text = xml_path.read_text(encoding="utf-8")
    assert "<name>fox</name>" in xml_text
    assert "<width>200</width>" in xml_text


def test_tensorflow_auto_labeler_create_xml_and_cleanup(tmp_path):
    component = TensorFlowAutoLabeler()
    component.folder_name = "images"
    component.labeled_dir = str(tmp_path)
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"img")

    component._create_xml(str(image_path), [["car", [10, 20, 30, 40]]], 480, 640)

    xml_path = tmp_path / "scene.xml"
    assert xml_path.exists()
    assert "<name>car</name>" in xml_path.read_text(encoding="utf-8")

    class FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    component.sess = FakeSession()
    component.cleanup()
    assert component.sess.closed is True
