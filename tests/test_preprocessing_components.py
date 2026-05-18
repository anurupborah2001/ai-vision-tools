import numpy as np
import pytest

from visionflow.components.preprocessing import (
    AdaptiveThreshold,
    AspectRatioFilter,
    AutoAdjustContrast,
    AutoCrop,
    AutoOrient,
    BGRToRGB,
    BlurDetection,
    BoundingBoxClamp,
    BoundingBoxNormalize,
    BrightnessCheck,
    CenterCrop,
    CLAHE,
    ContourExtraction,
    ConvertColorSpace,
    CorruptImageCheck,
    Deblur,
    Denoise,
    Deskew,
    DuplicateImageCheck,
    EdgeDetection,
    FaceAlign,
    GammaCorrection,
    HistogramEqualization,
    ImageQualityCheck,
    LetterboxResize,
    MaskResize,
    MaxSizeFilter,
    MinSizeFilter,
    Normalize,
    ObjectCrop,
    PadToSquare,
    PerspectiveCorrection,
    RemoveBackground,
    RescalePixels,
    Resize,
    RGBToBGR,
    Sharpen,
    Standardize,
    Threshold,
    WhiteBalance,
)


def _gradient_frame(width=80, height=60):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    return np.dstack((xv, yv, np.full_like(xv, 100)))


def test_auto_orient_uses_exif_metadata_when_available():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    payload = {"frame": frame.copy(), "metadata": {"exif_orientation": 6}}

    result = AutoOrient().run(payload, {})

    assert result["frame"].shape == (3, 2, 3)
    assert np.array_equal(result["frame"][0, 1], np.array([255, 0, 0], dtype=np.uint8))


def test_auto_orient_can_use_explicit_rotation_and_flips():
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    frame[0, 0] = (0, 255, 0)

    result = AutoOrient(rotation=90, flip_horizontal=True, use_exif=False).run(frame.copy(), {})

    assert result.shape == (4, 3, 3)
    assert result.sum() == frame.sum()


def test_auto_adjust_contrast_supports_all_methods_and_invalid_mode():
    frame = _gradient_frame()

    adaptive = AutoAdjustContrast(method="adaptive_equalization", clip_limit=1.5).run(
        frame.copy(), {}
    )
    histogram = AutoAdjustContrast(method="histogram_equalization").run(frame.copy(), {})
    stretched = AutoAdjustContrast(
        method="contrast_stretching",
        lower_percentile=5,
        upper_percentile=95,
    ).run(frame.copy(), {})

    for result in (adaptive, histogram, stretched):
        assert result.shape == frame.shape

    with pytest.raises(ValueError, match="Unsupported contrast method"):
        AutoAdjustContrast(method="unknown").run(frame.copy(), {})


def test_resize_changes_dimensions(sample_frame):
    result = Resize(width=20, height=10).run(sample_frame.copy(), {})
    assert result.shape == (10, 20, 3)


def test_letterbox_resize_preserves_target_canvas(sample_frame):
    result = LetterboxResize(width=100, height=100, pad_value=(5, 5, 5)).run(sample_frame.copy(), {})
    assert result.shape == (100, 100, 3)
    assert (result[0, 0] == np.array([5, 5, 5])).all()


def test_center_crop_returns_center_region(sample_frame):
    result = CenterCrop(width=20, height=10).run(sample_frame.copy(), {})
    assert result.shape == (10, 20, 3)


def test_pad_to_square_makes_square(sample_frame):
    result = PadToSquare(pad_value=(1, 2, 3)).run(sample_frame.copy(), {})
    assert result.shape[0] == result.shape[1]


def test_normalize_returns_float_range(sample_frame):
    result = Normalize().run(sample_frame.copy(), {})
    assert result.dtype.kind == "f"
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_standardize_centers_values(sample_frame):
    result = Standardize().run(sample_frame.copy(), {})
    assert result.dtype.kind == "f"
    assert abs(float(result.mean())) < 1.0


def test_rescale_pixels_applies_scale_and_offset(sample_frame):
    result = RescalePixels(scale=0.5, offset=1.0).run(sample_frame.copy(), {})
    assert result.dtype.kind == "f"
    assert np.isclose(result[0, 0, 0], sample_frame[0, 0, 0] * 0.5 + 1.0)


def test_convert_color_space_and_wrappers(sample_frame):
    rgb = ConvertColorSpace(source="BGR", target="RGB").run(sample_frame.copy(), {})
    back = RGBToBGR().run(rgb.copy(), {})
    direct = BGRToRGB().run(sample_frame.copy(), {})
    assert np.array_equal(direct, rgb)
    assert np.array_equal(back, sample_frame)


def test_clahe_and_histogram_equalization_keep_shape():
    frame = _gradient_frame()
    clahe = CLAHE().run(frame.copy(), {})
    hist = HistogramEqualization().run(frame.copy(), {})
    assert clahe.shape == frame.shape
    assert hist.shape == frame.shape


def test_histogram_equalization_supports_gray_and_rejects_invalid_color_space():
    frame = _gradient_frame()

    gray_equalized = HistogramEqualization(color_space="gray").run(frame.copy(), {})

    assert gray_equalized.shape == frame.shape

    with pytest.raises(ValueError, match="HistogramEqualization supports ycrcb or gray"):
        HistogramEqualization(color_space="lab").run(frame.copy(), {})


def test_gamma_correction_changes_image(sample_frame):
    result = GammaCorrection(gamma=1.8).run(sample_frame.copy(), {})
    assert result.shape == sample_frame.shape
    assert not np.array_equal(result, sample_frame)


def test_white_balance_rebalances_color_channels():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[:, :] = (40, 90, 180)
    result = WhiteBalance(method="gray_world").run(frame.copy(), {})
    assert result.shape == frame.shape
    assert not np.array_equal(result, frame)


def test_white_balance_supports_white_patch_and_rejects_invalid_method():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[5:15, 5:15] = (60, 120, 200)

    result = WhiteBalance(method="white_patch", percentile=90).run(frame.copy(), {})

    assert result.shape == frame.shape

    with pytest.raises(ValueError, match="WhiteBalance supports gray_world or white_patch"):
        WhiteBalance(method="invalid").run(frame.copy(), {})


def test_denoise_sharpen_and_deblur_keep_shape(sample_frame):
    denoised = Denoise(method="median", kernel_size=3).run(sample_frame.copy(), {})
    sharpened = Sharpen(amount=1.2).run(sample_frame.copy(), {})
    deblurred = Deblur(amount=1.1).run(sample_frame.copy(), {})
    assert denoised.shape == sample_frame.shape
    assert sharpened.shape == sample_frame.shape
    assert deblurred.shape == sample_frame.shape


def test_denoise_supports_bilateral_and_rejects_invalid_method(sample_frame):
    bilateral = Denoise(method="bilateral", kernel_size=5, sigma_color=50, sigma_space=50).run(
        sample_frame.copy(), {}
    )

    assert bilateral.shape == sample_frame.shape

    with pytest.raises(ValueError, match="Denoise supports nlm, median, or bilateral"):
        Denoise(method="invalid").run(sample_frame.copy(), {})


def test_remove_background_threshold_sets_background():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[5:15, 5:15] = 255
    result = RemoveBackground(method="threshold", threshold=10).run(frame.copy(), {})
    assert result[0, 0].tolist() == [0, 0, 0]
    assert result[10, 10].tolist() == [255, 255, 255]


def test_remove_background_keeps_mask_and_rejects_invalid_method():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[4:16, 4:16] = 255
    payload = {"frame": frame.copy()}

    result = RemoveBackground(method="threshold", threshold=10, keep_mask=True).run(payload, {})

    assert "mask" in result
    assert result["mask"].shape == frame.shape[:2]

    with pytest.raises(ValueError, match="RemoveBackground supports threshold or grabcut"):
        RemoveBackground(method="invalid").run(frame.copy(), {})


def test_threshold_and_adaptive_threshold_produce_masks(sample_frame):
    thresholded = Threshold(threshold=10, keep_channels=False).run(sample_frame.copy(), {})
    adaptive = AdaptiveThreshold(keep_channels=False).run(sample_frame.copy(), {})
    assert thresholded.ndim == 2
    assert adaptive.ndim == 2


def test_edge_detection_produces_edges(sample_frame):
    result = EdgeDetection(method="canny").run(sample_frame.copy(), {})
    assert result.shape == sample_frame.shape


def test_contour_extraction_populates_metadata(sample_frame):
    payload = {"frame": sample_frame.copy()}
    result = ContourExtraction().run(payload, {})
    assert "contours" in result
    assert result["frame"].shape == sample_frame.shape


def test_perspective_correction_warps_to_requested_size(sample_frame):
    points = np.array([[0, 0], [59, 0], [59, 39], [0, 39]], dtype=np.float32)
    result = PerspectiveCorrection(source_points=points, output_size=(30, 20)).run(
        sample_frame.copy(), {}
    )
    assert result.shape == (20, 30, 3)


def test_perspective_correction_rejects_invalid_source_points(sample_frame):
    with pytest.raises(ValueError, match="source_points must contain four"):
        PerspectiveCorrection(source_points=np.array([[0, 0], [1, 1]], dtype=np.float32)).run(
            sample_frame.copy(), {}
        )


def test_deskew_returns_image(sample_frame):
    result = Deskew().run(sample_frame.copy(), {})
    assert result.ndim == 3


def test_auto_crop_reduces_frame():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[10:30, 15:35] = 255
    result = AutoCrop(threshold=1).run(frame.copy(), {})
    assert result.shape[0] < frame.shape[0]
    assert result.shape[1] < frame.shape[1]


def test_face_align_uses_eye_coordinates(sample_frame):
    payload = {"frame": sample_frame.copy(), "metadata": {"left_eye": (15, 15), "right_eye": (35, 15)}}
    result = FaceAlign(output_size=(64, 64)).run(payload, {})
    assert result["frame"].shape == (64, 64, 3)


def test_object_crop_uses_bbox_from_payload(sample_frame):
    payload = {"frame": sample_frame.copy(), "bboxes": [(10, 5, 20, 15)]}
    result = ObjectCrop().run(payload, {})
    assert result["frame"].shape == (15, 20, 3)


def test_bounding_box_clamp_and_normalize(sample_frame):
    payload = {"frame": sample_frame.copy(), "bboxes": [(-5, -5, 80, 80)]}
    clamped = BoundingBoxClamp().run(payload, {})
    assert clamped["bboxes"][0][0] == 0.0
    normalized = BoundingBoxNormalize().run(clamped, {})
    x, y, w, h = normalized["bboxes"][0]
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert 0.0 <= w <= 1.0
    assert 0.0 <= h <= 1.0


def test_bounding_box_normalize_supports_inverse_and_rejects_invalid_mode(sample_frame):
    payload = {"frame": sample_frame.copy(), "bboxes": [(0.25, 0.25, 0.5, 0.5)]}

    denormalized = BoundingBoxNormalize(inverse=True).run(payload, {})
    x, y, w, h = denormalized["bboxes"][0]
    assert x == sample_frame.shape[1] * 0.25
    assert y == sample_frame.shape[0] * 0.25
    assert w == sample_frame.shape[1] * 0.5
    assert h == sample_frame.shape[0] * 0.5

    with pytest.raises(ValueError, match="Only xywh mode is currently supported"):
        BoundingBoxNormalize(mode="xyxy").run(
            {"frame": sample_frame.copy(), "bboxes": [(1, 1, 2, 2)]},
            {},
        )


def test_mask_resize_uses_frame_dimensions(sample_frame):
    payload = {"frame": sample_frame.copy(), "mask": np.zeros((10, 10), dtype=np.uint8)}
    result = MaskResize().run(payload, {})
    assert result["mask"].shape == sample_frame.shape[:2]


def test_mask_resize_respects_explicit_size(sample_frame):
    payload = {"frame": sample_frame.copy(), "mask": np.zeros((10, 10), dtype=np.uint8)}
    result = MaskResize(width=16, height=12).run(payload, {})
    assert result["mask"].shape == (12, 16)


def test_image_quality_check_sets_flags(sample_frame):
    payload = {"frame": sample_frame.copy()}
    result = ImageQualityCheck().run(payload, {})
    assert "quality" in result
    assert "blur_score" in result["quality"]
    assert "brightness_ok" in result["quality"]


def test_blur_detection_and_brightness_check(sample_frame):
    payload = {"frame": sample_frame.copy()}
    blur_result = BlurDetection().run(payload, {})
    bright_result = BrightnessCheck().run(blur_result, {})
    assert "is_blurry" in bright_result
    assert "brightness_ok" in bright_result


def test_duplicate_and_corrupt_checks(sample_frame):
    payload = {"frame": sample_frame.copy()}
    hashed = DuplicateImageCheck(reference_hashes=[]).run(payload, {})
    dup = DuplicateImageCheck(reference_hashes=[hashed["image_hash"]]).run(payload, {})
    assert dup["is_duplicate"] is True

    corrupt = CorruptImageCheck().run({"frame": np.array([])}, {})
    assert corrupt["is_corrupt"] is True


def test_ratio_and_size_filters(sample_frame):
    payload = {"frame": sample_frame.copy()}
    ratio = AspectRatioFilter(min_ratio=1.0, max_ratio=2.0).run(payload, {})
    minimum = MinSizeFilter(min_width=10, min_height=10).run(ratio, {})
    maximum = MaxSizeFilter(max_width=200, max_height=200).run(minimum, {})
    assert maximum["aspect_ratio_ok"] is True
    assert maximum["min_size_ok"] is True
    assert maximum["max_size_ok"] is True


def test_quality_filters_can_fail_on_extreme_inputs():
    frame = np.zeros((20, 200, 3), dtype=np.uint8)
    payload = {"frame": frame}

    ratio = AspectRatioFilter(min_ratio=0.5, max_ratio=2.0).run(payload, {})
    minimum = MinSizeFilter(min_width=300, min_height=30).run(ratio, {})
    maximum = MaxSizeFilter(max_width=100, max_height=10).run(minimum, {})

    assert maximum["aspect_ratio_ok"] is False
    assert maximum["min_size_ok"] is False
    assert maximum["max_size_ok"] is False
