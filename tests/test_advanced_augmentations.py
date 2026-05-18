import numpy as np

from visionflow.components.augmentations import (
    AffineTransform,
    ChannelShuffle,
    CoarseDropout,
    ColorJitter,
    CompressionArtifacts,
    CopyPaste,
    CutMix,
    DefocusBlur,
    Downscale,
    ElasticTransform,
    Emboss,
    Equalize,
    GaussianBlur,
    GlassBlur,
    GridDistortion,
    GridDropout,
    HSVShift,
    ISONoise,
    InvertImage,
    JPEGCompression,
    MaskDropout,
    MedianBlur,
    MixUp,
    Mosaic9,
    MultiplicativeNoise,
    ObjectPaste,
    OpticalDistortion,
    PerspectiveTransform,
    PixelDropout,
    Posterize,
    RandomBrightnessContrast,
    RandomCrop,
    RandomErasing,
    RandomFog,
    RandomGamma,
    RandomOcclusion,
    RandomPadding,
    RandomRain,
    RandomResize,
    RandomResizedCrop,
    RandomScale,
    RandomShadow,
    RandomSnow,
    RandomSunFlare,
    RGBShift,
    SaltPepperNoise,
    Sharpen,
    Solarize,
    Superpixel,
    ToSepia,
    Translate,
    ZoomBlur,
)


def _frame(width=80, height=60):
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    return np.dstack((xv, yv, np.full_like(xv, 120)))


def test_random_resize_changes_size():
    np.random.seed(1)
    frame = _frame()
    result = RandomResize(40, 50, 30, 40).run(frame, {})
    assert result.shape[0] in range(30, 41)
    assert result.shape[1] in range(40, 51)


def test_random_scale_resizes_frame():
    np.random.seed(1)
    frame = _frame()
    result = RandomScale(0.5, 0.6).run(frame, {})
    assert result.shape[0] < frame.shape[0]
    assert result.shape[1] < frame.shape[1]


def test_random_crop_and_random_resized_crop():
    np.random.seed(1)
    frame = _frame()
    cropped = RandomCrop(20, 10).run(frame, {})
    resized = RandomResizedCrop(32, 32).run(frame, {})
    assert cropped.shape == (10, 20, 3)
    assert resized.shape == (32, 32, 3)


def test_random_padding_expands_canvas():
    np.random.seed(1)
    frame = _frame()
    result = RandomPadding(5, 5, 5, 5).run(frame, {})
    assert result.shape[0] >= frame.shape[0]
    assert result.shape[1] >= frame.shape[1]


def test_translate_affine_perspective_keep_shape():
    frame = _frame()
    translated = Translate(shift_x=3, shift_y=4).run(frame, {})
    affine = AffineTransform(angle=5, translate_x=2, translate_y=1).run(frame, {})
    perspective = PerspectiveTransform(distortion_scale=0.05).run(frame, {})
    assert translated.shape == frame.shape
    assert affine.shape == frame.shape
    assert perspective.shape == frame.shape


def test_elastic_grid_and_optical_distortion_keep_shape():
    np.random.seed(1)
    frame = _frame()
    elastic = ElasticTransform(alpha=3, sigma=1).run(frame, {"seed": 1})
    grid = GridDistortion(num_steps=4, distort_limit=0.1).run(frame, {})
    optical = OpticalDistortion(k=0.00001).run(frame, {})
    assert elastic.shape == frame.shape
    assert grid.shape == frame.shape
    assert optical.shape == frame.shape


def test_weather_effects_keep_shape():
    np.random.seed(1)
    frame = _frame()
    assert RandomShadow().run(frame, {}).shape == frame.shape
    assert RandomSunFlare().run(frame, {}).shape == frame.shape
    assert RandomFog().run(frame, {}).shape == frame.shape
    assert RandomRain().run(frame, {}).shape == frame.shape
    assert RandomSnow().run(frame, {}).shape == frame.shape


def test_color_transforms_modify_image():
    frame = _frame()
    gamma = RandomGamma(0.8, 1.2).run(frame, {})
    jitter = ColorJitter().run(frame, {})
    shuffled = ChannelShuffle().run(frame, {})
    shifted = RGBShift(10, 5, -5).run(frame, {})
    hsv = HSVShift(10, 5, 5).run(frame, {})
    sepia = ToSepia().run(frame, {})
    inverted = InvertImage().run(frame, {})
    rbc = RandomBrightnessContrast().run(frame, {})
    for result in (gamma, jitter, shuffled, shifted, hsv, sepia, inverted, rbc):
        assert result.shape == frame.shape
    assert not np.array_equal(inverted, frame)


def test_posterize_solarize_equalize_emboss_sharpen():
    frame = _frame()
    outputs = [
        Posterize(bits=3).run(frame, {}),
        Solarize(threshold=100).run(frame, {}),
        Equalize().run(frame, {}),
        Emboss().run(frame, {}),
        Sharpen().run(frame, {}),
    ]
    for result in outputs:
        assert result.shape == frame.shape


def test_blur_family_keep_shape():
    frame = _frame()
    outputs = [
        GaussianBlur().run(frame, {}),
        MedianBlur().run(frame, {}),
        GlassBlur(max_delta=1).run(frame, {}),
        DefocusBlur(radius=3).run(frame, {}),
        ZoomBlur(zoom_factor=1.1, steps=3).run(frame, {}),
    ]
    for result in outputs:
        assert result.shape == frame.shape


def test_noise_family_keep_shape():
    np.random.seed(1)
    frame = _frame()
    outputs = [
        ISONoise().run(frame, {}),
        MultiplicativeNoise().run(frame, {}),
        SaltPepperNoise(amount=0.05).run(frame, {}),
    ]
    for result in outputs:
        assert result.shape == frame.shape


def test_dropout_family_modifies_pixels():
    np.random.seed(1)
    frame = _frame()
    coarse = CoarseDropout(holes=2, max_height=4, max_width=4).run(frame, {})
    grid = GridDropout(ratio=0.5, unit_size=6).run(frame, {})
    erase = RandomErasing(scale=(0.05, 0.05)).run(frame, {})
    pixel = PixelDropout(dropout_prob=0.2).run(frame, {})
    for result in (coarse, grid, erase, pixel):
        assert result.shape == frame.shape
        assert np.count_nonzero(result != frame) > 0


def test_mixup_cutmix_and_copy_paste():
    frame = _frame()
    other = np.full_like(frame, 255)
    payload = {"frame": frame.copy(), "mix_image": other}
    mixup = MixUp(alpha=0.4).run(payload.copy(), {})
    cutmix = CutMix(alpha=0.5).run(payload.copy(), {})
    copied = CopyPaste(x=5, y=5).run({"frame": frame.copy(), "overlay_image": other[:10, :10]}, {})
    assert mixup["frame"].shape == frame.shape
    assert cutmix["frame"].shape == frame.shape
    assert copied["frame"].shape == frame.shape


def test_random_occlusion_and_object_paste():
    frame = _frame()
    obj = np.zeros((10, 10, 3), dtype=np.uint8)
    occluded = RandomOcclusion(max_width=5, max_height=5).run(frame.copy(), {})
    pasted = ObjectPaste(x=3, y=4).run({"frame": frame.copy(), "object_image": obj}, {})
    assert occluded.shape == frame.shape
    assert pasted["frame"].shape == frame.shape


def test_bounding_box_jitter_and_mask_dropout(sample_frame):
    payload = {"frame": sample_frame.copy(), "bboxes": [(10, 10, 20, 20)], "mask": np.ones(sample_frame.shape[:2], dtype=np.uint8)}
    jittered = BoundingBoxJitter().run(payload, {})
    dropped = MaskDropout(dropout_prob=0.5).run(jittered, {})
    assert len(dropped["bboxes"]) == 1
    assert dropped["mask"].shape == sample_frame.shape[:2]


def test_mosaic9_builds_larger_canvas():
    frame = _frame(20, 20)
    output = Mosaic9(mosaic_images=[frame] * 8).run(frame, {})
    assert output.shape[0] == 60
    assert output.shape[1] == 60


def test_compression_downscale_and_superpixel():
    frame = _frame()
    compressed = CompressionArtifacts(quality=30).run(frame, {})
    jpeg = JPEGCompression(quality=25).run(frame, {})
    downscaled = Downscale(scale=0.5).run(frame, {})
    superpixel = Superpixel(region_size=5).run(frame, {})
    assert compressed.shape == frame.shape
    assert jpeg.shape == frame.shape
    assert downscaled.shape == frame.shape
    assert superpixel.shape == frame.shape
