import json

import numpy as np
import pytest

from ai_vision_tool.augmentation import (
    Blur,
    Brightness,
    CameraGain,
    Crop,
    Cutout,
    Exposure,
    Flip,
    Greyscale,
    Hue,
    Mosaic,
    MotionBlur,
    Noise,
    Rotate90,
    Rotation,
    Saturation,
    Shear,
)
from ai_vision_tool.cli.main import load_profile_components


def _frame(width=8, height=6):
    base = np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3)
    return base


def test_flip_supports_horizontal_vertical_both_and_noop():
    frame = _frame(4, 3)

    horizontal = Flip(horizontal=True).run(frame, {})
    vertical = Flip(vertical=True).run(frame, {})
    both = Flip(horizontal=True, vertical=True).run(frame, {})
    unchanged = Flip().run(frame, {})

    assert np.array_equal(horizontal, np.fliplr(frame))
    assert np.array_equal(vertical, np.flipud(frame))
    assert np.array_equal(both, np.flipud(np.fliplr(frame)))
    assert np.array_equal(unchanged, frame)


def test_rotate90_respects_k_modulo_and_swaps_dimensions():
    frame = _frame(5, 3)

    rotated = Rotate90(k=1).run(frame, {})
    wrapped = Rotate90(k=5).run(frame, {})

    assert rotated.shape == (5, 3, 3)
    assert np.array_equal(rotated, wrapped)


def test_crop_uses_prefixed_config_and_clamps_to_frame():
    frame = _frame(10, 8)

    cropped = Crop(x=0, y=0, width=2, height=2).run(
        frame,
        {"crop_x": 7, "crop_y": 5, "crop_width": 20, "crop_height": 20},
    )

    assert cropped.shape == (3, 3, 3)
    assert np.array_equal(cropped, frame[5:8, 7:10])


def test_rotation_expand_and_shear_keep_expected_behavior():
    frame = _frame(10, 6)

    rotated_same = Rotation(angle=30).run(frame, {})
    rotated_expand = Rotation(angle=30, expand=True).run(frame, {})
    sheared = Shear(shear_x=0.2, shear_y=0.1).run(frame, {})

    assert rotated_same.shape == frame.shape
    assert rotated_expand.shape[0] >= frame.shape[0]
    assert rotated_expand.shape[1] >= frame.shape[1]
    assert sheared.shape == frame.shape


def test_greyscale_supports_bgr_and_single_channel_output():
    frame = _frame()

    keep_channels = Greyscale(keep_channels=True).run(frame, {})
    single_channel = Greyscale(keep_channels=False).run(frame, {})

    assert keep_channels.shape == frame.shape
    assert single_channel.ndim == 2
    assert np.array_equal(keep_channels[:, :, 0], keep_channels[:, :, 1])
    assert np.array_equal(keep_channels[:, :, 1], keep_channels[:, :, 2])


def test_color_intensity_augmentations_modify_pixels():
    frame = np.full((6, 8, 3), 90, dtype=np.uint8)

    hue = Hue(delta=15).run(frame, {})
    saturation = Saturation(scale=1.5).run(frame, {})
    brighter = Brightness(beta=25).run(frame, {})
    exposure = Exposure(gamma=1.4).run(frame, {})
    gained = CameraGain(gain=1.2, black_level=5).run(frame, {})

    for result in (hue, saturation, brighter, exposure, gained):
        assert result.shape == frame.shape

    assert not np.array_equal(brighter, frame)
    assert not np.array_equal(exposure, frame)
    assert not np.array_equal(gained, frame)


def test_blur_motion_blur_and_mosaic_return_expected_shapes():
    frame = _frame(12, 10)
    other = np.full_like(frame, 255)

    blurred = Blur(kernel_size=4, sigma_x=1.0).run(frame, {})
    motion = MotionBlur(kernel_size=6, angle=20).run(frame, {})
    mosaic = Mosaic(output_size=(20, 16), mosaic_images=[other]).run(frame, {})

    assert blurred.shape == frame.shape
    assert motion.shape == frame.shape
    assert mosaic.shape == (16, 20, 3)


def test_noise_supports_gaussian_and_salt_pepper_and_rejects_invalid_mode():
    np.random.seed(1)
    frame = _frame(12, 10)

    gaussian = Noise(mode="gaussian", stddev=12).run(frame, {})
    salt_pepper = Noise(mode="salt_pepper", amount=0.1).run(frame, {})

    assert gaussian.shape == frame.shape
    assert salt_pepper.shape == frame.shape
    assert np.count_nonzero(gaussian != frame) > 0
    assert np.count_nonzero(salt_pepper != frame) > 0

    with pytest.raises(ValueError, match="Unsupported noise mode"):
        Noise(mode="unknown").run(frame, {})


def test_cutout_uses_prefixed_config_over_generic_values():
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)

    output = Cutout(x=0, y=0, width=1, height=1, fill_value=(0, 0, 0)).run(
        frame,
        {
            "x": 0,
            "y": 0,
            "width": 2,
            "height": 2,
            "fill_value": (1, 1, 1),
            "cutout_x": 3,
            "cutout_y": 2,
            "cutout_width": 4,
            "cutout_height": 3,
            "cutout_fill_value": (9, 8, 7),
        },
    )

    assert np.all(output[2:5, 3:7] == np.array([9, 8, 7], dtype=np.uint8))
    assert np.all(output[:2, :2] == 200)


def test_load_profile_components_creates_augmentation_instances(tmp_path):
    profile_path = tmp_path / "augmentations.json"
    profile_path.write_text(
        json.dumps(
            [
                {"name": "Flip", "params": {"horizontal": True}},
                {"name": "Brightness", "params": {"beta": 12}},
            ]
        ),
        encoding="utf-8",
    )

    components = load_profile_components(str(profile_path))

    assert [component.__class__.__name__ for component in components] == [
        "Flip",
        "Brightness",
    ]
