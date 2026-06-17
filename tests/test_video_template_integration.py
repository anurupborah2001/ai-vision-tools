"""
Integration tests for video_capture_template.

These tests exercise complete multi-feature flows — auto/manual recording
cycles, timed screenshot logic, and end-of-stream behaviour — with the
camera and VideoRecorder replaced by lightweight fakes.
"""
import cv2
import numpy as np
import pytest

import ai_vision_tool.capture.video_template as _vt
from ai_vision_tool.capture.video_template import video_capture_template, KeyEventManager


# ── shared fakes ─────────────────────────────────────────────────────────────


class FakeVideoCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.released = False

    def isOpened(self):
        return bool(self.frames)

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def set(self, *a): ...

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.index
        return 0

    def release(self):
        self.released = True


class FakeVideoRecorder:
    def __init__(self, output_format="mp4", fps=15):
        self.format = output_format
        self.fps = fps
        self.started = False
        self.stopped = False
        self.frames_written = 0

    def start(self, shape):
        self.started = True

    def write(self, frame):
        self.frames_written += 1

    def stop(self):
        self.stopped = True


def _noop(*a, **kw): ...


@pytest.fixture
def small_frame():
    return np.zeros((40, 60, 3), dtype=np.uint8)


# ── auto screenshot ───────────────────────────────────────────────────────────


def test_auto_screenshot_fires_exactly_once(monkeypatch, small_frame):
    fake_cap = FakeVideoCapture([small_frame] * 3)
    keys = iter([0, 0, 27])
    shots = []

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(_vt, "save_screenshot", lambda f, **kw: shots.append(True) or "shot.png")

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_screenshot=True,
        auto_screenshot_after_seconds=0,
        auto_screenshot_repeat=False,
    )

    assert len(shots) == 1


def test_auto_screenshot_repeats_every_interval(monkeypatch, small_frame):
    # interval=0 → every frame satisfies (now - last >= 0)
    fake_cap = FakeVideoCapture([small_frame] * 5)
    keys = iter([0, 0, 0, 0, 27])
    shots = []

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(_vt, "save_screenshot", lambda f, **kw: shots.append(True) or "shot.png")

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_screenshot=True,
        auto_screenshot_after_seconds=0,
        auto_screenshot_repeat=True,
    )

    assert len(shots) >= 2


def test_auto_screenshot_not_taken_before_interval(monkeypatch, small_frame):
    fake_cap = FakeVideoCapture([small_frame] * 3)
    keys = iter([0, 0, 27])
    shots = []

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(_vt, "save_screenshot", lambda f, **kw: shots.append(True) or "shot.png")

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_screenshot=True,
        auto_screenshot_after_seconds=9999,
        auto_screenshot_repeat=False,
    )

    assert shots == []


# ── auto recording ────────────────────────────────────────────────────────────


def test_auto_recording_starts_writes_and_stops(monkeypatch, small_frame):
    fake_cap = FakeVideoCapture([small_frame, small_frame])
    keys = iter([0, 27])
    recorder_instances = []

    def make_recorder(output_format="mp4", fps=15):
        r = FakeVideoRecorder(output_format=output_format, fps=fps)
        recorder_instances.append(r)
        return r

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(cv2, "putText", _noop)
    monkeypatch.setattr(_vt, "_make_recorder", make_recorder)

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_auto_recording=True,
        record_format="mp4",
    )

    assert len(recorder_instances) == 1
    rec = recorder_instances[0]
    assert rec.started is True
    assert rec.frames_written == 2
    assert rec.stopped is True


def test_auto_recording_uses_gif_format(monkeypatch, small_frame):
    fake_cap = FakeVideoCapture([small_frame])
    recorder_instances = []

    def make_recorder(output_format="mp4", fps=15):
        r = FakeVideoRecorder(output_format=output_format, fps=fps)
        recorder_instances.append(r)
        return r

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: 27)
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(cv2, "putText", _noop)
    monkeypatch.setattr(_vt, "_make_recorder", make_recorder)

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_auto_recording=True,
        record_format="gif",
    )

    assert recorder_instances[0].format == "gif"


# ── manual recording ──────────────────────────────────────────────────────────


def test_manual_recording_start_stop_via_r_key(monkeypatch, small_frame):
    # Frame sequence: r=start, 0=record frame, r=stop, ESC
    fake_cap = FakeVideoCapture([small_frame] * 4)
    keys = iter([ord("r"), 0, ord("r"), 27])
    recorder_instances = []

    def make_recorder(output_format="mp4", fps=15):
        r = FakeVideoRecorder(output_format=output_format, fps=fps)
        recorder_instances.append(r)
        return r

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(cv2, "putText", _noop)
    monkeypatch.setattr(_vt, "_make_recorder", make_recorder)

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_manual_recording=True,
        record_format="mp4",
    )

    assert len(recorder_instances) == 1
    rec = recorder_instances[0]
    assert rec.started is True
    assert rec.frames_written >= 1
    assert rec.stopped is True


def test_manual_recording_not_active_without_r_key(monkeypatch, small_frame):
    fake_cap = FakeVideoCapture([small_frame, small_frame])
    keys = iter([0, 27])
    recorder_instances = []

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(_vt, "_make_recorder",
                        lambda **kw: recorder_instances.append(FakeVideoRecorder()) or recorder_instances[-1])

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        enable_manual_recording=True,
    )

    assert recorder_instances == []


# ── end-of-stream ─────────────────────────────────────────────────────────────


def test_end_of_stream_exits_cleanly(monkeypatch, small_frame, capsys):
    fake_cap = FakeVideoCapture([small_frame])
    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: 0)
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)

    video_capture_template(video_source="clip.mp4", loop_forever=False, show_window=False, draw_fps=False)

    assert fake_cap.released is True
    assert "End of video" in capsys.readouterr().out


# ── combined features ─────────────────────────────────────────────────────────


def test_custom_logic_plus_key_manager_plus_screenshot(monkeypatch, small_frame):
    processed = []
    handled = []
    shots = []

    km = KeyEventManager()
    km.register(ord("x"), lambda frame, state: handled.append(True))

    fake_cap = FakeVideoCapture([small_frame] * 3)
    keys = iter([ord("x"), ord("s"), 27])

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: fake_cap)
    monkeypatch.setattr(cv2, "waitKey", lambda _: next(keys, 27))
    monkeypatch.setattr(cv2, "destroyAllWindows", _noop)
    monkeypatch.setattr(_vt, "save_screenshot", lambda f, **kw: shots.append(True) or "p.png")

    video_capture_template(
        video_source=0,
        loop_forever=False,
        show_window=False,
        draw_fps=False,
        custom_logic=lambda f: processed.append(True) or f,
        key_manager=km,
        enable_screenshot=True,
    )

    assert len(processed) == 3
    assert handled == [True]
    assert shots == [True]
