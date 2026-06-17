from pathlib import Path

import cv2
import numpy as np

from ai_vision_tool.capture.burst_image_capture import BurstPictureTaker
from ai_vision_tool.capture.frame_grabber import FrameGrabber
from ai_vision_tool.io.image_exporter import ImageExporter
from ai_vision_tool.capture.image_capture import PictureTaker
from ai_vision_tool.capture.roi_capture import ROICapture
from ai_vision_tool.capture.video_capture import VideoTaker
import ai_vision_tool.capture.video_template as _vt
from ai_vision_tool.capture.video_template import (
    video_capture_template,
    save_screenshot,
    KeyEventManager,
)


def _noop(*a, **kw): ...


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


# ── save_screenshot ──────────────────────────────────────────────────────────


def test_save_screenshot_writes_png(tmp_cwd, sample_frame, stub_imwrite):
    path = save_screenshot(sample_frame, output_dir="shots", prefix="test")
    assert path.endswith(".png")
    assert "test_" in path


def test_save_screenshot_creates_directory(tmp_cwd, sample_frame, stub_imwrite):
    save_screenshot(sample_frame, output_dir="new_dir/nested", prefix="cap")
    assert (tmp_cwd / "new_dir" / "nested").is_dir()


# ── video_capture_template — unit tests ──────────────────────────────────────


def test_vct_returns_when_source_fails_to_open(monkeypatch, fake_video_capture_cls, capsys):
    fake_cap = fake_video_capture_cls([])  # isOpened() → False
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)

    video_capture_template(video_source=99, show_window=False, draw_fps=False)

    assert "Error" in capsys.readouterr().out


def test_vct_exits_on_esc(monkeypatch, sample_frame, fake_video_capture_cls):
    fake_cap = fake_video_capture_cls([sample_frame, sample_frame])
    keys = iter([27])
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)

    video_capture_template(video_source=0, loop_forever=False, show_window=False, draw_fps=False)

    assert fake_cap.released is True


def test_vct_exits_on_end_of_stream(monkeypatch, sample_frame, fake_video_capture_cls, capsys):
    fake_cap = fake_video_capture_cls([sample_frame])
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: 0)
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)

    video_capture_template(video_source="clip.mp4", loop_forever=False, show_window=False, draw_fps=False)

    assert fake_cap.released is True
    assert "End of video" in capsys.readouterr().out


def test_vct_applies_custom_logic_to_each_frame(monkeypatch, sample_frame, fake_video_capture_cls):
    processed = []
    fake_cap = fake_video_capture_cls([sample_frame, sample_frame])
    keys = iter([0, 27])
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        custom_logic=lambda frame: processed.append(True) or frame,
    )

    assert len(processed) == 2


def test_vct_key_handler_called_for_matching_key(monkeypatch, sample_frame, fake_video_capture_cls):
    calls = []
    km = KeyEventManager()
    km.register(ord("x"), lambda frame, state: calls.append(True))

    fake_cap = fake_video_capture_cls([sample_frame, sample_frame])
    keys = iter([ord("x"), 27])
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        key_manager=km,
    )

    assert calls == [True]


def test_vct_screenshot_on_s_key(monkeypatch, sample_frame, fake_video_capture_cls):
    shots = []
    fake_cap = fake_video_capture_cls([sample_frame, sample_frame])
    keys = iter([ord("s"), 27])
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(_vt, "save_screenshot", lambda f, **kw: shots.append(True) or "p.png")

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_screenshot=True,
    )

    assert shots == [True]


def test_vct_no_window_created_when_hidden(monkeypatch, sample_frame, fake_video_capture_cls):
    window_calls = []
    fake_cap = fake_video_capture_cls([sample_frame])
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: 27)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **kw: window_calls.append(True))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)

    video_capture_template(video_source=0, loop_forever=False, show_window=False, draw_fps=False)

    assert window_calls == []


# ── VideoTaker ───────────────────────────────────────────────────────────────


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
