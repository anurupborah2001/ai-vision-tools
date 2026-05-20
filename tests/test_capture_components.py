from pathlib import Path

import cv2
import numpy as np

from ai_vision_tool.capture.burst_image_capture import BurstPictureTaker
from ai_vision_tool.capture.frame_grabber import FrameGrabber
from ai_vision_tool.io.image_exporter import ImageExporter
from ai_vision_tool.capture.image_capture import PictureTaker
from ai_vision_tool.capture.roi_capture import ROICapture
from ai_vision_tool.capture.video_capture import VideoTaker


def test_frame_grabber_returns_empty_for_missing_video(capsys):
    component = FrameGrabber()

    result = component.run("missing-video.mp4", {"output_folder": "frames"})

    assert result == []
    assert "Video file not found" in capsys.readouterr().out


def test_frame_grabber_extracts_frames(monkeypatch, tmp_cwd, sample_frame, fake_video_capture_cls, created_files, stub_imwrite):
    component = FrameGrabber()
    video_path = tmp_cwd / "demo.mp4"
    video_path.write_bytes(b"video")
    fake_capture = fake_video_capture_cls([sample_frame, sample_frame, sample_frame])

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: fake_capture)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 0)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    result = component.run(str(video_path), {"output_folder": "frames", "skip_frames": 1, "resize_factor": 1.0})

    assert len(result) == 3
    assert fake_capture.released is True
    assert all(Path(path).exists() for path in result)
    assert created_files


def test_picture_taker_captures_until_quit(monkeypatch, tmp_cwd, sample_frame, fake_video_capture_cls, created_files, stub_imwrite):
    component = PictureTaker()
    fake_capture = fake_video_capture_cls([sample_frame, sample_frame])
    keys = iter([ord("p"), ord("q")])

    monkeypatch.setattr(cv2, "VideoCapture", lambda camera_id: fake_capture)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "moveWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    result = component.run(None, {"imgdir": "captures", "resolution": "60x40", "camera_id": 7})

    assert len(result) == 1
    assert Path(result[0]).exists()
    assert fake_capture.released is True
    assert created_files[0].parent == tmp_cwd / "captures"


def test_burst_picture_taker_stops_when_capture_fails(monkeypatch, sample_frame):
    component = BurstPictureTaker(burst_count=3, interval_seconds=0.01)
    saved = []

    class FakeBurstCap:
        def __init__(self):
            self.frames = [sample_frame.copy(), sample_frame.copy()]

        def read(self):
            if self.frames:
                return True, self.frames.pop(0)
            return False, None

    monkeypatch.setattr(component, "save_frame", lambda frame: saved.append(frame.copy()) or f"saved-{len(saved)}", raising=False)
    monkeypatch.setattr("ai_vision_tool.capture.burst_image_capture.time.sleep", lambda seconds: None)

    result = component.capture_burst(FakeBurstCap())

    assert result == ["saved-1", "saved-2"]
    assert len(saved) == 2


def test_roi_capture_extracts_expected_region(sample_frame):
    component = ROICapture(roi=(10, 5, 20, 15), draw_roi=False)

    roi = component.extract_roi(sample_frame)

    assert roi.shape == (15, 20, 3)
    assert np.array_equal(roi, sample_frame[5:20, 10:30])


def test_roi_capture_process_draws_and_saves(monkeypatch, sample_frame):
    component = ROICapture(roi=(5, 5, 10, 10), draw_roi=True)
    saved = {}

    monkeypatch.setattr(component, "save_frame", lambda frame: saved.setdefault("frame", frame.copy()), raising=False)

    result = component.process(sample_frame.copy(), capture_roi=True)

    assert "frame" in saved
    assert saved["frame"].shape == (10, 10, 3)
    assert np.count_nonzero(result) >= np.count_nonzero(sample_frame)


def test_image_exporter_writes_gray_and_edges(tmp_path, sample_frame, created_files, stub_imwrite):
    component = ImageExporter(output_dir=tmp_path / "exports")

    gray_path = component.export_grayscale(sample_frame, "gray.jpg")
    edge_path = component.export_edges(sample_frame, "edges.jpg")

    assert Path(gray_path).exists()
    assert Path(edge_path).exists()
    assert len(created_files) == 2


def test_image_exporter_execute_passthrough_dict(tmp_path, sample_frame, created_files, stub_imwrite):
    component = ImageExporter(output_dir=tmp_path / "exports")
    payload = {"frame": sample_frame.copy()}

    result = component.run(payload, {"export_gray": True, "export_edges": True})

    assert result is payload
    assert len(created_files) == 2


def test_video_taker_records_and_returns_saved_video(monkeypatch, tmp_cwd, sample_frame, fake_video_capture_cls, fake_video_writer_cls):
    component = VideoTaker()
    fake_capture = fake_video_capture_cls([sample_frame, sample_frame, sample_frame])
    writers = []
    keys = iter([ord("r"), ord("r"), ord("q")])

    monkeypatch.setattr(cv2, "VideoCapture", lambda camera_id: fake_capture)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: "FOURCC")
    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        lambda path, fourcc, fps, size: writers.append(fake_video_writer_cls(path, fourcc, fps, size)) or writers[-1],
    )
    monkeypatch.setattr(cv2, "namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    result = component.run(None, {"viddir": "videos", "resolution": "60x40", "camera_id": 0, "fps": 12})

    assert len(result) == 1
    assert result[0].endswith(".avi")
    assert fake_capture.released is True
    assert writers[0].released is True
    assert writers[0].frames
