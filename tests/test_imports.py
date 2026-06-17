import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.parent)


def test_import_ai_vision_tool_base_is_lightweight():
    env = {**os.environ, "PYTHONPATH": _PROJECT_ROOT}
    result = subprocess.run(
        [sys.executable, "-c", "import ai_vision_tool; print('ok')"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert "onnxruntime" not in result.stderr.lower()
    assert "torch" not in result.stderr.lower()


def test_core_base_importable():
    from ai_vision_tool.core.base import AIVisionComponent

    assert AIVisionComponent is not None


def test_preprocessing_importable():
    from ai_vision_tool.preprocessing import Resize

    assert Resize is not None


def test_augmentation_importable():
    from ai_vision_tool.augmentation import Blur

    assert Blur is not None


def test_enhancement_core_importable():
    from ai_vision_tool.enhancement.denoiser import Denoiser
    from ai_vision_tool.enhancement.low_light import LowLightEnhancer

    assert Denoiser is not None
    assert LowLightEnhancer is not None


def test_top_level_namespace_exports_core_classes():
    import ai_vision_tool as avt

    assert hasattr(avt, "AIVisionComponent")
    assert hasattr(avt, "AIVisionPipeline")
    assert hasattr(avt, "Resize")
    assert hasattr(avt, "Blur")
    assert hasattr(avt, "Denoiser")


def test_heavy_dep_classes_lazy_not_eager():
    # Verify that simply importing ai_vision_tool does NOT pull in heavy optional deps.
    # The subprocess approach is authoritative; this test double-checks via sys.modules.
    import ai_vision_tool  # noqa: F401

    assert "onnxruntime" not in sys.modules
    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules
    assert "mediapipe" not in sys.modules
