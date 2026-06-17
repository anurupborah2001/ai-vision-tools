import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def sample_frame():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[10:30, 15:45] = (25, 125, 225)
    return frame


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class FakeVideoCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.settings = []
        self.released = False

    def isOpened(self):
        return self.index < len(self.frames)

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def set(self, prop, value):
        self.settings.append((prop, value))

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.index
        return 0

    def release(self):
        self.released = True


class FakeVideoWriter:
    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


@pytest.fixture
def fake_video_capture_cls():
    def factory(frames):
        return FakeVideoCapture(frames)

    return factory


@pytest.fixture
def fake_video_writer_cls():
    return FakeVideoWriter


@pytest.fixture
def created_files():
    return []


@pytest.fixture
def stub_imwrite(monkeypatch, created_files):
    import cv2

    def fake_imwrite(path, frame):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"written")
        created_files.append(path)
        return True

    monkeypatch.setattr(cv2, "imwrite", fake_imwrite)
    return fake_imwrite
