import argparse
import importlib
import json
import time
from pathlib import Path

import cv2
import uvicorn

from ai_vision_tool.components import (
    AutoAdjustContrast,
    AutoOrient,
    Blur,
    Brightness,
    CameraGain,
    Crop,
    Cutout,
    DatasetCollector,
    Exposure,
    Flip,
    FrameAnnotator,
    FrameEnhancer,
    FrameResizer,
    Greyscale,
    Hue,
    Mosaic,
    MotionDetector,
    MotionBlur,
    Noise,
    Rotate90,
    Rotation,
    Saturation,
    Shear,
    TimeLapseCapture,
)
from ai_vision_tool.components.augmentations.common import parse_component_profile
from ai_vision_tool.pipelines import AIVisionPipeline
from ai_vision_tool.api_service import (
    decode_image_base64,
    encode_image_base64,
    execute_component,
)

EXAMPLE_CATEGORY_CHOICES = ["all", "preprocessing", "augmentations", "components", "capture"]


def _example_entry(category, name, summary, python_example, runtime_example=None):
    """Constructs a single entry dict for the examples catalog.

    Args:
        category (str): Component category ('preprocessing', 'augmentations', etc.).
        name (str): Component or helper name.
        summary (str): One-line description of what the component does.
        python_example (str): Python code snippet demonstrating usage.
        runtime_example (str or None): Optional CLI / main.py invocation example.

    Returns:
        dict: Catalog entry with 'category', 'name', 'summary', 'python', and
            'runtime' keys.
    """
    return {
        "category": category,
        "name": name,
        "summary": summary,
        "python": python_example.strip(),
        "runtime": runtime_example,
    }


def _run_example(import_from, name, constructor, target="image", result_name="result"):
    """Generates a two-line Python import-and-run code snippet string.

    Args:
        import_from (str): Module path to import the component from.
        name (str): Component class name.
        constructor (str): Constructor argument string (e.g. 'rotation=90').
        target (str): Variable name passed to ``.run()``. Default is 'image'.
        result_name (str): Variable name assigned the result. Default is 'result'.

    Returns:
        str: Formatted two-line Python code snippet.
    """
    return (
        f"from {import_from} import {name}\n"
        f"{result_name} = {name}({constructor}).run({target})"
    )


def build_examples_catalog():
    """Builds the full examples catalog for all component categories.

    Returns:
        list[dict]: List of example entry dicts covering preprocessing,
            augmentations, components, and capture helpers.
    """
    catalog = []
    lookup_prefix = "python main.py --show-examples --example-name"

    preprocessing_entries = [
        ("AutoOrient", "rotation=90, flip_horizontal=False", "Fix orientation metadata or rotate frames.", "image", "python main.py --auto-orient --auto-orient-rotation 90"),
        ("AutoAdjustContrast", 'method="adaptive_equalization", clip_limit=2.0', "Apply adaptive equalization or histogram-based contrast recovery.", "image", 'python main.py --auto-adjust-contrast --contrast-method adaptive_equalization --clip-limit 2.0'),
        ("Resize", "width=640, height=640", "Resize to an exact output size.", "image", None),
        ("LetterboxResize", "width=640, height=640, pad_value=(114, 114, 114)", "Resize with padding while preserving aspect ratio.", "image", None),
        ("CenterCrop", "width=224, height=224", "Crop the center patch for classification inputs.", "image", None),
        ("PadToSquare", "pad_value=(0, 0, 0)", "Pad rectangular frames to a square canvas.", "image", None),
        ("Normalize", "", "Map pixel intensities into a normalized numeric range.", "image", None),
        ("Standardize", "per_channel=True", "Standardize by mean and standard deviation.", "image", None),
        ("RescalePixels", "scale=1.0 / 255.0, offset=0.0", "Rescale raw pixel values for model ingestion.", "image", None),
        ("ConvertColorSpace", 'source="BGR", target="RGB"', "Convert between OpenCV color spaces.", "image", None),
        ("BGRToRGB", "", "Convert BGR frames to RGB.", "image", None),
        ("RGBToBGR", "", "Convert RGB frames back to BGR.", "image", None),
        ("CLAHE", "clip_limit=2.0, tile_grid_size=(8, 8)", "Boost local contrast with CLAHE.", "image", None),
        ("HistogramEqualization", 'color_space="ycrcb"', "Equalize luminance histograms for flatter lighting.", "image", None),
        ("GammaCorrection", "gamma=1.4", "Apply gamma correction to brighten or darken frames.", "image", None),
        ("WhiteBalance", 'method="gray_world"', "Correct channel color casts.", "image", None),
        ("Denoise", 'method="median", kernel_size=3', "Reduce sensor or compression noise.", "image", None),
        ("Sharpen", "amount=1.0", "Sharpen softened edges before downstream analysis.", "image", None),
        ("Deblur", "amount=1.0", "Recover edges with an unsharp-style deblur pass.", "image", None),
        ("RemoveBackground", 'method="threshold", threshold=10, keep_mask=True', "Mask or suppress the background region.", '{"frame": image}', None),
        ("Threshold", "threshold=127, keep_channels=False", "Create a binary threshold mask.", "image", None),
        ("AdaptiveThreshold", 'method="gaussian", block_size=11, keep_channels=False', "Create a local adaptive threshold mask.", "image", None),
        ("EdgeDetection", 'method="canny", threshold1=100, threshold2=200', "Extract edges for feature or contour workflows.", "image", None),
        ("ContourExtraction", "", "Populate contour metadata on a frame payload.", '{"frame": image}', None),
        ("PerspectiveCorrection", "source_points=points, output_size=(600, 400)", "Rectify a quadrilateral document or planar region.", "image", None),
        ("Deskew", "", "Rotate a document or scene back to a leveled angle.", "image", None),
        ("AutoCrop", "threshold=10, padding=4", "Trim empty borders around the active subject.", "image", None),
        ("FaceAlign", "output_size=(112, 112)", "Align a face payload using eye coordinates.", '{"frame": image, "metadata": {"left_eye": (40, 50), "right_eye": (90, 50)}}', None),
        ("ObjectCrop", "", "Crop a region from payload bounding boxes.", '{"frame": image, "bboxes": [(10, 20, 120, 80)]}', None),
        ("BoundingBoxClamp", "", "Clamp payload bounding boxes to image boundaries.", '{"frame": image, "bboxes": [(-5, -5, 80, 90)]}', None),
        ("BoundingBoxNormalize", "", "Normalize payload bounding boxes to relative coordinates.", '{"frame": image, "bboxes": [(10, 20, 120, 80)]}', None),
        ("MaskResize", "width=640, height=640", "Resize a payload mask to a target size.", '{"frame": image, "mask": mask}', None),
        ("ImageQualityCheck", "", "Compute blur and brightness quality flags.", '{"frame": image}', None),
        ("BlurDetection", "", "Flag blurry payloads using Laplacian variance.", '{"frame": image}', None),
        ("BrightnessCheck", "min_brightness=40.0, max_brightness=215.0", "Validate payload brightness bounds.", '{"frame": image}', None),
        ("DuplicateImageCheck", "reference_hashes=[]", "Hash a payload to detect duplicates.", '{"frame": image}', None),
        ("CorruptImageCheck", "", "Flag empty or invalid payload frames.", '{"frame": image}', None),
        ("AspectRatioFilter", "min_ratio=0.75, max_ratio=1.5", "Filter payloads by aspect ratio.", '{"frame": image}', None),
        ("MinSizeFilter", "min_width=320, min_height=320", "Require a minimum payload size.", '{"frame": image}', None),
        ("MaxSizeFilter", "max_width=2048, max_height=2048", "Reject overly large payloads.", '{"frame": image}', None),
    ]

    for name, constructor, summary, target, runtime_example in preprocessing_entries:
        catalog.append(
            _example_entry(
                "preprocessing",
                name,
                summary,
                _run_example("ai_vision_tool.components.preprocessing", name, constructor, target=target),
                runtime_example,
            )
        )

    augmentation_entries = [
        ("Flip", "horizontal=True", "Mirror a frame horizontally or vertically.", "image", "python main.py --flip-horizontal"),
        ("Rotate90", "k=1", "Rotate a frame by 90-degree increments.", "image", "python main.py --rotate90-k 1"),
        ("Crop", "x=20, y=20, width=200, height=200", "Extract a rectangular region.", "image", "python main.py --crop --crop-x 20 --crop-y 20 --crop-width 200 --crop-height 200"),
        ("Rotation", "angle=12.0, scale=1.0", "Rotate with arbitrary angles.", "image", "python main.py --rotation-angle 12"),
        ("Shear", "shear_x=0.15, shear_y=0.0", "Apply an affine shear transform.", "image", "python main.py --shear-x 0.15"),
        ("Greyscale", "keep_channels=True", "Convert to grayscale while keeping 3 channels if desired.", "image", "python main.py --augment-greyscale"),
        ("Hue", "delta=10", "Shift hue values in HSV space.", "image", "python main.py --hue-delta 10"),
        ("Saturation", "scale=1.2", "Increase or reduce saturation.", "image", "python main.py --saturation-scale 1.2"),
        ("Brightness", "beta=18", "Add or subtract brightness.", "image", "python main.py --brightness-beta 18"),
        ("Exposure", "gamma=1.3", "Apply gamma-based exposure changes.", "image", "python main.py --exposure-gamma 1.3"),
        ("Blur", "kernel_size=5, sigma_x=1.0", "Apply Gaussian blur.", "image", "python main.py --blur --blur-kernel-size 5 --blur-sigma-x 1.0"),
        ("Noise", 'mode="gaussian", stddev=8.0', "Inject Gaussian or salt-and-pepper noise.", "image", "python main.py --noise --noise-mode gaussian --noise-stddev 8"),
        ("Cutout", "x=30, y=30, width=60, height=60", "Blank out a rectangular patch.", "image", "python main.py --cutout --cutout-x 30 --cutout-y 30 --cutout-width 60 --cutout-height 60"),
        ("Mosaic", "output_size=(640, 640), mosaic_images=[image, image, image]", "Create a 2x2 mosaic composition.", "image", "python main.py --mosaic --mosaic-image path/to/a.jpg --mosaic-image path/to/b.jpg"),
        ("MotionBlur", "kernel_size=11, angle=25", "Simulate camera or subject motion blur.", "image", "python main.py --motion-blur --motion-blur-kernel-size 11 --motion-blur-angle 25"),
        ("CameraGain", "gain=1.2, black_level=8.0", "Simulate sensor gain and black-level shifts.", "image", "python main.py --camera-gain 1.2 --camera-black-level 8"),
        ("RandomResize", "min_width=320, max_width=640, min_height=320, max_height=640", "Randomize output size for training variation.", "image", f"{lookup_prefix} RandomResize"),
        ("RandomScale", "min_scale=0.8, max_scale=1.2", "Randomize image scale.", "image", f"{lookup_prefix} RandomScale"),
        ("RandomCrop", "crop_width=224, crop_height=224", "Sample a random crop.", "image", f"{lookup_prefix} RandomCrop"),
        ("RandomResizedCrop", "output_width=224, output_height=224", "Random crop plus resize for model inputs.", "image", f"{lookup_prefix} RandomResizedCrop"),
        ("RandomPadding", "max_top=16, max_bottom=16, max_left=16, max_right=16", "Add random padding to vary context.", "image", f"{lookup_prefix} RandomPadding"),
        ("Translate", "shift_x=12, shift_y=8", "Translate a frame in X/Y.", "image", f"{lookup_prefix} Translate"),
        ("AffineTransform", "angle=8, scale=1.0, translate_x=10, translate_y=10", "Apply combined rotate/scale/translate/shear.", "image", f"{lookup_prefix} AffineTransform"),
        ("PerspectiveTransform", "distortion_scale=0.15", "Perturb perspective corners.", "image", f"{lookup_prefix} PerspectiveTransform"),
        ("ElasticTransform", "alpha=3.0, sigma=1.0", "Apply elastic spatial distortion.", "image", f"{lookup_prefix} ElasticTransform"),
        ("GridDistortion", "num_steps=5, distort_limit=0.2", "Warp using a distorted sampling grid.", "image", f"{lookup_prefix} GridDistortion"),
        ("OpticalDistortion", "k=0.00001", "Simulate lens distortion.", "image", f"{lookup_prefix} OpticalDistortion"),
        ("RandomShadow", "shadow_dimension=0.5, intensity=0.5", "Overlay synthetic shadows.", "image", f"{lookup_prefix} RandomShadow"),
        ("RandomSunFlare", "radius=20, intensity=0.4", "Overlay a sun flare or hotspot.", "image", f"{lookup_prefix} RandomSunFlare"),
        ("RandomFog", "alpha=0.2", "Blend fog or haze into the frame.", "image", f"{lookup_prefix} RandomFog"),
        ("RandomRain", "drops=40, drop_length=12, intensity=0.25", "Render synthetic rain streaks.", "image", f"{lookup_prefix} RandomRain"),
        ("RandomSnow", "intensity=0.1", "Render synthetic snow.", "image", f"{lookup_prefix} RandomSnow"),
        ("RandomGamma", "min_gamma=0.8, max_gamma=1.2", "Randomize gamma for exposure drift.", "image", f"{lookup_prefix} RandomGamma"),
        ("ColorJitter", "brightness=0.2, contrast=0.2, saturation=0.2, hue=8", "Randomize multiple color properties together.", "image", f"{lookup_prefix} ColorJitter"),
        ("ChannelShuffle", "", "Randomize channel order.", "image", f"{lookup_prefix} ChannelShuffle"),
        ("RGBShift", "r_shift=10, g_shift=-5, b_shift=4", "Shift individual RGB-style channels.", "image", f"{lookup_prefix} RGBShift"),
        ("Posterize", "bits=4", "Reduce tonal depth.", "image", f"{lookup_prefix} Posterize"),
        ("Solarize", "threshold=128", "Invert highlights above a threshold.", "image", f"{lookup_prefix} Solarize"),
        ("Equalize", "", "Equalize image intensity channels.", "image", f"{lookup_prefix} Equalize"),
        ("Emboss", "strength=1.0", "Create an embossed texture effect.", "image", f"{lookup_prefix} Emboss"),
        ("GaussianBlur", "kernel_size=5, sigma_x=1.0", "Apply an explicit Gaussian blur augmentation.", "image", f"{lookup_prefix} GaussianBlur"),
        ("MedianBlur", "kernel_size=5", "Apply a median blur.", "image", f"{lookup_prefix} MedianBlur"),
        ("GlassBlur", "sigma=0.7, max_delta=2, iterations=1", "Apply glass blur style local pixel swaps.", "image", f"{lookup_prefix} GlassBlur"),
        ("DefocusBlur", "radius=5", "Simulate defocus blur.", "image", f"{lookup_prefix} DefocusBlur"),
        ("ZoomBlur", "zoom_factor=1.2, steps=5", "Simulate zoom-style motion blur.", "image", f"{lookup_prefix} ZoomBlur"),
        ("ISONoise", "color_shift=0.01, intensity=0.5", "Simulate ISO sensor noise.", "image", f"{lookup_prefix} ISONoise"),
        ("MultiplicativeNoise", "multiplier_min=0.9, multiplier_max=1.1", "Scale pixels with multiplicative noise.", "image", f"{lookup_prefix} MultiplicativeNoise"),
        ("SaltPepperNoise", "amount=0.02, salt_vs_pepper=0.5", "Add salt-and-pepper noise.", "image", f"{lookup_prefix} SaltPepperNoise"),
        ("CoarseDropout", "holes=8, max_height=8, max_width=8", "Erase several rectangular patches.", "image", f"{lookup_prefix} CoarseDropout"),
        ("GridDropout", "ratio=0.5, unit_size=8", "Drop pixels on a grid pattern.", "image", f"{lookup_prefix} GridDropout"),
        ("RandomErasing", "scale=(0.02, 0.2)", "Erase one random region.", "image", f"{lookup_prefix} RandomErasing"),
        ("MixUp", "alpha=0.5", "Mix the payload frame with a partner image.", '{"frame": image, "mix_image": image_b}', f"{lookup_prefix} MixUp"),
        ("CutMix", "alpha=0.5", "Replace a patch with a partner image.", '{"frame": image, "mix_image": image_b}', f"{lookup_prefix} CutMix"),
        ("CopyPaste", "x=10, y=10", "Paste an overlay image into the frame.", '{"frame": image, "overlay_image": overlay}', f"{lookup_prefix} CopyPaste"),
        ("RandomOcclusion", "max_width=20, max_height=20", "Hide a random rectangle.", "image", f"{lookup_prefix} RandomOcclusion"),
        ("ObjectPaste", "x=20, y=30", "Paste an object image into the frame.", '{"frame": image, "object_image": object_image}', f"{lookup_prefix} ObjectPaste"),
        ("BoundingBoxJitter", "x_jitter=0.05, y_jitter=0.05, size_jitter=0.1", "Perturb payload bounding boxes.", '{"frame": image, "bboxes": [(10, 10, 100, 60)]}', f"{lookup_prefix} BoundingBoxJitter"),
        ("MaskDropout", "dropout_prob=0.1", "Randomly zero a payload mask.", '{"frame": image, "mask": mask}', f"{lookup_prefix} MaskDropout"),
        ("Mosaic9", "mosaic_images=[image] * 8", "Create a 3x3 mosaic.", "image", f"{lookup_prefix} Mosaic9"),
        ("HSVShift", "hue_shift=10, sat_shift=5, val_shift=5", "Shift hue, saturation, and value directly.", "image", f"{lookup_prefix} HSVShift"),
        ("ToSepia", "", "Apply a sepia color effect.", "image", f"{lookup_prefix} ToSepia"),
        ("InvertImage", "", "Invert all pixel values.", "image", f"{lookup_prefix} InvertImage"),
        ("CompressionArtifacts", "quality=40", "Simulate generic compression artifacts.", "image", f"{lookup_prefix} CompressionArtifacts"),
        ("JPEGCompression", "quality=40", "Round-trip through JPEG compression.", "image", f"{lookup_prefix} JPEGCompression"),
        ("Downscale", 'scale=0.5, interpolation="area"', "Downscale and restore to simulate low resolution.", "image", f"{lookup_prefix} Downscale"),
        ("Superpixel", "region_size=10, ruler=10.0", "Approximate a superpixel rendering effect.", "image", f"{lookup_prefix} Superpixel"),
        ("PixelDropout", "dropout_prob=0.01", "Randomly zero individual pixels.", "image", f"{lookup_prefix} PixelDropout"),
        ("RandomBrightnessContrast", "brightness_limit=0.2, contrast_limit=0.2", "Randomize brightness and contrast together.", "image", f"{lookup_prefix} RandomBrightnessContrast"),
    ]

    for name, constructor, summary, target, runtime_example in augmentation_entries:
        catalog.append(
            _example_entry(
                "augmentations",
                name,
                summary,
                _run_example("ai_vision_tool.components.augmentations", name, constructor, target=target),
                runtime_example,
            )
        )

    component_entries = [
        ("FrameEnhancer", 'FrameEnhancer().run({"frame": image}, {"brightness": 10, "contrast": 1.1, "sharpen": True})', "Apply brightness, contrast, sharpen, denoise, and grayscale settings.", "python main.py --enhance --brightness 10 --contrast 1.1 --sharpen"),
        ("FrameResizer", 'FrameResizer().run({"frame": image}, {"size": (640, 480), "keep_aspect": True})', "Resize frames using the classic component pipeline.", "python main.py --resize --width 640 --height 480 --keep-aspect"),
        ("MotionDetector", 'MotionDetector().run({"frame": image}, {"min_area": 800, "draw_motion": True})', "Detect motion regions across sequential frames.", "python main.py --motion --motion-area 800"),
        ("FrameAnnotator", 'FrameAnnotator().run({"frame": image, "annotations": [{"type": "text", "text": "Demo", "pos": (20, 30)}]}, {})', "Overlay payload-driven annotations onto frames.", "python main.py --annotate"),
        ("DatasetCollector", 'DatasetCollector().run({"frame": image}, {"save_sample": True, "output_dir": "output/dataset", "label": "demo"})', "Persist dataset samples and metadata.", "python main.py --dataset --label demo"),
        ("TimeLapseCapture", 'TimeLapseCapture(output_dir="output/timelapse", interval_seconds=5).run({"frame": image}, {})', "Persist periodic frames for time-lapse workflows.", "python main.py --timelapse --timelapse-interval 5"),
        ("PictureTaker", 'PictureTaker().run(None, {"imgdir": "Pics", "resolution": "1280x720", "camera_id": 0})', "Interactive still-image capture from a webcam.", "python main.py --picture"),
        ("BurstPictureTaker", 'BurstPictureTaker(burst_count=5, interval_seconds=0.2)', "Interactive burst capture helper.", "python main.py --burst --burst-count 5"),
        ("ROICapture", 'ROICapture(roi=(100, 100, 300, 300))', "Interactive region-of-interest capture helper.", "python main.py --roi --roi-x 100 --roi-y 100 --roi-w 300 --roi-h 300"),
        ("ImageExporter", 'ImageExporter(output_dir="output/exports").run({"frame": image}, {"export_gray": True, "export_edges": True})', "Export grayscale and edge images.", "python main.py --export"),
        ("FrameGrabber", 'FrameGrabber().run("sample.mp4", {"output_folder": "extracted_pics", "skip_frames": 90})', "Extract still frames from a video file.", None),
        ("VideoTaker", 'VideoTaker().run(None, {"viddir": "Videos", "resolution": "1280x720", "camera_id": 0, "fps": 30.0})', "Interactive webcam recording helper.", "python main.py --video --fps 20"),
        ("AutoLabeller", 'AutoLabeller().run(image)', "Base auto-labeling entrypoint for downstream integrations.", None),
        ("DarknetAutoLabeler", 'DarknetAutoLabeler().run({"frame": image}, {"output_dir": "labels"})', "Darknet-powered labeling workflow.", None),
        ("TensorFlowAutoLabeler", 'TensorFlowAutoLabeler().run({"frame": image}, {"output_dir": "labels"})', "TensorFlow-powered labeling workflow.", None),
    ]

    for name, python_call, summary, runtime_example in component_entries:
        catalog.append(
            _example_entry(
                "components",
                name,
                summary,
                f"from ai_vision_tool.components import {name}\n{python_call}",
                runtime_example,
            )
        )

    template_entries = [
        (
            "image_template",
            "Display a still image with optional custom logic.",
            "from ai_vision_tool.capture.image_template import image_template\nimage_template(\n    image_path=\"path/to/image.jpg\",\n    custom_logic=lambda frame: frame,\n    window_name=\"KAMI Demo\",\n    resolution=(1280, 720),\n)",
        ),
        (
            "save_screenshot",
            "Save a screenshot frame to disk from the template workflow.",
            "from ai_vision_tool.capture.video_template import save_screenshot\nsave_screenshot(frame, output_dir=\"screenshots\", prefix=\"capture\")",
        ),
        (
            "video_capture_template",
            "Run the legacy OpenCV video template with custom frame logic.",
            "from ai_vision_tool.capture.video_template import video_capture_template\nvideo_capture_template(\n    video_source=0,\n    custom_logic=lambda frame: frame,\n    window_name=\"KAMI Live\",\n    resolution=(1280, 720),\n    enable_recording=False,\n    enable_screenshot=True,\n)",
        ),
    ]

    for name, summary, python_example in template_entries:
        catalog.append(_example_entry("capture", name, summary, python_example, None))

    return catalog


EXAMPLES_CATALOG = build_examples_catalog()


def format_examples(category="all", name_filter=None):
    """Formats and returns a Markdown string of filtered example entries.

    Args:
        category (str): Category to filter by, or 'all' for no category filter.
            Default is 'all'.
        name_filter (str or None): Case-insensitive substring to match against
            component names. Default is None (no name filter).

    Returns:
        str: Formatted Markdown string with matching examples, or a 'no results'
            message listing all available names.
    """
    category = (category or "all").lower()
    name_filter = (name_filter or "").strip().lower()
    filtered = []
    for item in EXAMPLES_CATALOG:
        if category != "all" and item["category"] != category:
            continue
        if name_filter and name_filter not in item["name"].lower():
            continue
        filtered.append(item)

    if not filtered:
        available = ", ".join(sorted({item["name"] for item in EXAMPLES_CATALOG}))
        return (
            "[ai-vision-tool] No matching examples found.\n\n"
            f"Available example names: {available}\n"
        )

    lines = ["# AI Vision Flow Example Catalog", ""]
    current_category = None
    for item in filtered:
        if item["category"] != current_category:
            current_category = item["category"]
            lines.extend([f"## {current_category.title()}", ""])
        lines.append(f"### {item['name']}")
        lines.append(item["summary"])
        lines.append("")
        lines.append("Python:")
        lines.append("```python")
        lines.append(item["python"])
        lines.append("```")
        if item["runtime"]:
            lines.append("")
            lines.append("CLI / main.py:")
            lines.append("```bash")
            lines.append(item["runtime"])
            lines.append("```")
        lines.append("")
        lines.append(
            "Lookup command: "
            f"`python main.py --show-examples --example-name {item['name']}`"
        )
        lines.append("")
    return "\n".join(lines)


def parse_json_argument(value, argument_name):
    """Parses a JSON string CLI argument, returning None when the value is None.

    Args:
        value (str or None): Raw JSON string from argparse, or None.
        argument_name (str): The CLI flag name used in error messages.

    Returns:
        Any: Parsed Python object, or None if value is None.

    Raises:
        ValueError: If value is not valid JSON.
    """
    if value is None:
        return None

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON passed to {argument_name}: {exc.msg}"
        ) from exc


def process_component_from_image_path(args):
    """Loads an image file, executes the specified component, and prints the result.

    Reads ``args.image_path``, base64-encodes it, and calls ``execute_component``
    with the parsed CLI arguments. Optionally saves the processed image to disk.

    Args:
        args (argparse.Namespace): Parsed CLI arguments. Must include
            ``image_path``, ``component_category``, ``component_name``,
            and optional ``init_args_json``, ``config_json``, ``payload_json``,
            ``data_json``, ``batch_json``, ``save_output_image``.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image file: {image_path}")

    init_args = parse_json_argument(args.init_args_json, "--init-args-json") or {}
    config = parse_json_argument(args.config_json, "--config-json") or {}
    payload = parse_json_argument(args.payload_json, "--payload-json")
    data = parse_json_argument(args.data_json, "--data-json")
    batch = parse_json_argument(args.batch_json, "--batch-json")
    image_base64 = encode_image_base64(image)

    result = execute_component(
        category=args.component_category,
        name=args.component_name,
        init_args=init_args,
        config=config,
        image_base64=image_base64,
        payload=payload,
        data=data,
        batch=batch,
    )

    if args.save_output_image:
        serialized = result.get("result")
        if isinstance(serialized, dict) and serialized.get("type") == "image" and serialized.get("base64"):
            output_image = decode_image_base64(serialized["base64"])
            output_path = Path(args.save_output_image)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), output_image)
            print(f"[ai-vision-tool] Saved processed image to {output_path}")
        elif (
            isinstance(serialized, dict)
            and isinstance(serialized.get("frame"), dict)
            and serialized["frame"].get("type") == "image"
            and serialized["frame"].get("base64")
        ):
            output_image = decode_image_base64(serialized["frame"]["base64"])
            output_path = Path(args.save_output_image)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), output_image)
            print(f"[ai-vision-tool] Saved processed frame to {output_path}")
        else:
            print(
                "[ai-vision-tool] --save-output-image was ignored because the component "
                "result did not contain a serializable image."
            )

    print(json.dumps(result, indent=2))


def build_pipeline(args):
    """Constructs an AIVisionPipeline from parsed CLI arguments.

    Adds components in a fixed order based on which flags are set in args.

    Args:
        args (argparse.Namespace): Parsed CLI arguments controlling which
            pipeline stages are active.

    Returns:
        AIVisionPipeline: Configured pipeline instance.
    """
    pipeline = AIVisionPipeline()

    if args.auto_orient:
        pipeline.add(
            AutoOrient(
                use_exif=not args.no_auto_orient_exif,
                exif_key=args.auto_orient_exif_key,
                rotation=args.auto_orient_rotation,
                flip_horizontal=args.auto_orient_flip_horizontal,
                flip_vertical=args.auto_orient_flip_vertical,
            )
        )

    if args.auto_adjust_contrast:
        pipeline.add(
            AutoAdjustContrast(
                method=args.contrast_method,
                clip_limit=args.clip_limit,
                tile_grid_size=(args.tile_grid_width, args.tile_grid_height),
                lower_percentile=args.lower_percentile,
                upper_percentile=args.upper_percentile,
                output_min=args.output_min,
                output_max=args.output_max,
            )
        )

    if args.enhance:
        pipeline.add(FrameEnhancer())

    if args.resize:
        pipeline.add(FrameResizer())

    if args.flip_horizontal or args.flip_vertical:
        pipeline.add(Flip(horizontal=args.flip_horizontal, vertical=args.flip_vertical))

    if args.rotate90_k % 4:
        pipeline.add(Rotate90(k=args.rotate90_k))

    if args.crop:
        pipeline.add(
            Crop(
                x=args.crop_x,
                y=args.crop_y,
                width=args.crop_width,
                height=args.crop_height,
                clamp=not args.no_crop_clamp,
            )
        )

    if args.rotation_angle or args.rotation_expand or args.rotation_scale != 1.0:
        pipeline.add(
            Rotation(
                angle=args.rotation_angle,
                scale=args.rotation_scale,
                expand=args.rotation_expand,
                border_mode=args.rotation_border_mode,
                border_value=args.rotation_border_value,
            )
        )

    if args.shear_x or args.shear_y:
        pipeline.add(
            Shear(
                shear_x=args.shear_x,
                shear_y=args.shear_y,
                border_mode=args.shear_border_mode,
                border_value=args.shear_border_value,
            )
        )

    if args.augment_greyscale:
        pipeline.add(Greyscale(keep_channels=not args.greyscale_single_channel))

    if args.hue_delta:
        pipeline.add(Hue(delta=args.hue_delta))

    if args.saturation_scale != 1.0:
        pipeline.add(Saturation(scale=args.saturation_scale))

    if args.brightness_beta:
        pipeline.add(Brightness(beta=args.brightness_beta))

    if args.exposure_gamma != 1.0:
        pipeline.add(Exposure(gamma=args.exposure_gamma))

    if args.blur:
        pipeline.add(Blur(kernel_size=args.blur_kernel_size, sigma_x=args.blur_sigma_x))

    if args.noise:
        pipeline.add(
            Noise(
                mode=args.noise_mode,
                mean=args.noise_mean,
                stddev=args.noise_stddev,
                amount=args.noise_amount,
                salt_vs_pepper=args.salt_vs_pepper,
            )
        )

    if args.cutout:
        pipeline.add(
            Cutout(
                x=args.cutout_x,
                y=args.cutout_y,
                width=args.cutout_width,
                height=args.cutout_height,
                fill_value=tuple(args.cutout_fill_value),
            )
        )

    if args.mosaic:
        pipeline.add(
            Mosaic(
                output_size=(args.mosaic_output_width, args.mosaic_output_height)
                if args.mosaic_output_width and args.mosaic_output_height
                else None,
                mosaic_images=load_mosaic_images(args.mosaic_images),
            )
        )

    if args.motion_blur:
        pipeline.add(
            MotionBlur(
                kernel_size=args.motion_blur_kernel_size,
                angle=args.motion_blur_angle,
            )
        )

    if args.camera_gain != 1.0 or args.camera_black_level != 0.0:
        pipeline.add(
            CameraGain(
                gain=args.camera_gain,
                black_level=args.camera_black_level,
            )
        )

    for component in load_profile_components(args.augmentation_config):
        pipeline.add(component)

    if args.motion:
        pipeline.add(MotionDetector())

    if args.timelapse:
        pipeline.add(
            TimeLapseCapture(
                output_dir=str(resolve_output_dirs(args)["timelapse"]),
                interval_seconds=args.timelapse_interval,
            )
        )

    if args.dataset:
        pipeline.add(DatasetCollector())

    if args.annotate:
        pipeline.add(FrameAnnotator())

    return pipeline


def build_runtime_config(args, save_sample=False, annotations=None):
    """Builds the per-frame runtime config dict from parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        save_sample (bool): If True, sets 'save_sample' to trigger dataset saving.
            Default is False.
        annotations (list or None): Annotation entries for FrameAnnotator.
            Default is None (empty list).

    Returns:
        dict: Runtime configuration dict passed to every pipeline processor.
    """
    output_dirs = resolve_output_dirs(args)
    return {
        "brightness": args.brightness,
        "contrast": args.contrast,
        "sharpen": args.sharpen,
        "denoise": args.denoise,
        "grayscale": args.grayscale,
        "size": (args.width, args.height),
        "keep_aspect": args.keep_aspect,
        "min_area": args.motion_area,
        "draw_motion": True,
        "save_sample": save_sample,
        "output_dir": str(output_dirs["dataset"]),
        "save_metadata": True,
        "label": args.label,
        "metadata": {"source": "webcam"},
        "annotations": annotations or [],
    }


def resolve_output_dirs(args):
    """Resolves all output subdirectory paths relative to the output root.

    Args:
        args (argparse.Namespace): Parsed CLI arguments containing output path flags.

    Returns:
        dict[str, Path]: Mapping of directory names to Path objects:
            'root', 'captures', 'timelapse', 'videos', 'dataset', 'exports'.
    """
    output_root = Path(args.output_root)
    return {
        "root": output_root,
        "captures": output_root / args.output_dir,
        "timelapse": output_root / args.timelapse_dir,
        "videos": output_root / args.video_dir,
        "dataset": output_root / args.dataset_dir,
        "exports": output_root / args.export_dir,
    }


def ensure_output_dirs(output_dirs):
    """Creates all output directories, making parent directories as needed.

    Args:
        output_dirs (dict[str, Path]): Output directory mapping as returned by
            ``resolve_output_dirs``.
    """
    output_dirs["root"].mkdir(parents=True, exist_ok=True)
    output_dirs["captures"].mkdir(parents=True, exist_ok=True)
    output_dirs["timelapse"].mkdir(parents=True, exist_ok=True)
    output_dirs["videos"].mkdir(parents=True, exist_ok=True)
    output_dirs["dataset"].mkdir(parents=True, exist_ok=True)
    output_dirs["exports"].mkdir(parents=True, exist_ok=True)


def load_profile_components(config_path):
    """Loads and instantiates components from a JSON augmentation profile file.

    Each entry in the profile must contain 'name' and optional 'params' and
    'module' keys. When 'module' is absent, the component is looked up in
    ``ai_vision_tool.components.augmentations``.

    Args:
        config_path (str or None): Path to the JSON profile file, or None to
            return an empty list.

    Returns:
        list[AIVisionComponent]: Instantiated components in profile order.
    """
    if not config_path:
        return []

    profile = parse_component_profile(config_path)
    components = []
    for item in profile:
        class_name = item["name"]
        params = item.get("params", {})
        module_name = item.get("module")
        if module_name:
            module = importlib.import_module(module_name)
            component_cls = getattr(module, class_name)
        else:
            module = importlib.import_module("ai_vision_tool.components.augmentations")
            component_cls = getattr(module, class_name)
        components.append(component_cls(**params))
    return components


def load_mosaic_images(paths):
    """Loads a list of image files for use as mosaic tiles.

    Args:
        paths (list[str]): File paths of images to load.

    Returns:
        list[numpy.ndarray]: BGR images in the same order as paths.

    Raises:
        ValueError: If any image file cannot be loaded.
    """
    images = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Unable to load mosaic image: {path}")
        images.append(image)
    return images


def run_pipeline(pipeline, frame, config, annotations=None):
    """Wraps a frame in a payload dict and runs it through the pipeline manually.

    Bypasses ``AIVisionPipeline.execute`` to allow injecting annotations into
    the payload at the call site.

    Args:
        pipeline (AIVisionPipeline): Configured pipeline instance.
        frame (numpy.ndarray): Current BGR camera frame.
        config (dict): Runtime configuration dict.
        annotations (list or None): Annotation entries for FrameAnnotator.
            Default is None (empty list).

    Returns:
        dict or numpy.ndarray: Final processed payload from the last pipeline stage.
    """
    current = {"frame": frame, "annotations": annotations or []}

    for processor in pipeline.processors:
        current = processor.run(current, config)

    return current


def next_output_path(output_dir, prefix, extension):
    """Returns a timestamped output file path, creating the directory if needed.

    Args:
        output_dir (Path): Directory to write the file into.
        prefix (str): Filename prefix before the timestamp.
        extension (str): File extension without a leading dot.

    Returns:
        Path: Full output path in the form ``{output_dir}/{prefix}_{timestamp}.{extension}``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    return output_dir / f"{prefix}_{timestamp}.{extension}"


def save_capture(frame, output_dir, prefix="capture"):
    """Saves a single frame as a timestamped JPEG to the given directory.

    Args:
        frame (numpy.ndarray): BGR image to save.
        output_dir (Path): Directory to write the file into.
        prefix (str): Filename prefix. Default is 'capture'.
    """
    image_path = next_output_path(output_dir, prefix, "jpg")
    cv2.imwrite(str(image_path), frame)
    print(f"[ai-vision-tool] Saved capture to {image_path}")


def save_roi(frame, output_dir, roi):
    """Crops the ROI region from a frame and saves it as a JPEG.

    Skips saving if the ROI is entirely outside the frame bounds.

    Args:
        frame (numpy.ndarray): BGR image to crop from.
        output_dir (Path): Directory to write the file into.
        roi (tuple[int, int, int, int]): Region of interest as (x, y, w, h).
    """
    x, y, w, h = roi
    roi_frame = frame[y:y + h, x:x + w]
    if roi_frame.size == 0:
        print("[ai-vision-tool] ROI capture skipped because the selected area is outside the frame.")
        return

    image_path = next_output_path(output_dir, "roi", "jpg")
    cv2.imwrite(str(image_path), roi_frame)
    print(f"[ai-vision-tool] Saved ROI capture to {image_path}")


def save_exports(frame, export_dir):
    """Saves grayscale and Canny edge images of the frame to the export directory.

    Args:
        frame (numpy.ndarray): BGR image to export.
        export_dir (Path): Directory to write the exported files into.
    """
    gray_path = next_output_path(export_dir, "gray", "jpg")
    edges_path = next_output_path(export_dir, "edges", "jpg")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    cv2.imwrite(str(gray_path), gray)
    cv2.imwrite(str(edges_path), edges)
    print(f"[ai-vision-tool] Exported grayscale image to {gray_path}")
    print(f"[ai-vision-tool] Exported edge image to {edges_path}")


def build_video_writer(frame, video_dir, fps):
    """Creates an XVID VideoWriter for recording to a timestamped AVI file.

    Args:
        frame (numpy.ndarray): Frame used to determine output dimensions.
        video_dir (Path): Directory to write the video file into.
        fps (int or float): Output frame rate.

    Returns:
        tuple[cv2.VideoWriter, Path]: The VideoWriter instance and the output path.
    """
    height, width = frame.shape[:2]
    video_path = next_output_path(video_dir, "recording", "avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    return writer, video_path


def add_default_annotations():
    """Returns the default annotation list for the live webcam overlay.

    Returns:
        list[dict]: List with one text annotation showing 'AI Vision Pipeline'.
    """
    return [
        {"type": "text", "text": "AI Vision Pipeline", "pos": (20, 30)},
    ]


def draw_roi_overlay(frame, roi):
    """Draws a green ROI rectangle and 'ROI' label on the frame.

    Args:
        frame (numpy.ndarray): BGR frame to annotate in place.
        roi (tuple[int, int, int, int]): Region as (x, y, w, h).

    Returns:
        numpy.ndarray: Annotated frame (same object, modified in place).
    """
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "ROI",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return frame


def create_parser():
    """Builds and returns the argument parser for the ai-vision-tool CLI.

    Returns:
        argparse.ArgumentParser: Fully configured parser covering all flags
            for the API server, image processing, webcam pipeline, and
            individual augmentation parameters.
    """
    parser = argparse.ArgumentParser(prog="ai-vision-flow")

    parser.add_argument("--serve-api", action="store_true")
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8300)
    parser.add_argument("--api-reload", action="store_true")
    parser.add_argument(
        "--process-image-path",
        action="store_true",
        help="Process a single image file path through a preprocessing, augmentation, or component class.",
    )
    parser.add_argument(
        "--component-category",
        choices=["preprocessing", "augmentations", "components"],
        help="Category for --process-image-path execution.",
    )
    parser.add_argument(
        "--component-name",
        help="Class name for --process-image-path execution, for example AutoOrient or Flip.",
    )
    parser.add_argument(
        "--image-path",
        help="Input image file path for --process-image-path. The CLI loads the file and converts it to base64 automatically.",
    )
    parser.add_argument(
        "--init-args-json",
        default=None,
        help="JSON object of constructor arguments for --process-image-path, for example '{\"rotation\": 90}'.",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="JSON object of runtime config arguments for --process-image-path.",
    )
    parser.add_argument(
        "--payload-json",
        default=None,
        help="Optional JSON payload merged with the image frame for payload-aware components.",
    )
    parser.add_argument(
        "--data-json",
        default=None,
        help="Optional raw JSON data for non-payload component execution.",
    )
    parser.add_argument(
        "--batch-json",
        default=None,
        help="Optional JSON array for batch execution. Use image or mask *_base64 keys inside the array items when needed.",
    )
    parser.add_argument(
        "--save-output-image",
        default=None,
        help="Optional output file path to save the returned processed image or payload frame.",
    )
    parser.add_argument("--show-examples", action="store_true")
    parser.add_argument(
        "--example-category",
        choices=EXAMPLE_CATEGORY_CHOICES,
        default="all",
        help="Print example usage for preprocessing, augmentations, components, capture, or all.",
    )
    parser.add_argument(
        "--example-name",
        default=None,
        help="Filter printed examples by class or helper name, for example AutoOrient or video_capture_template.",
    )

    parser.add_argument("--picture", action="store_true")
    parser.add_argument("--burst", action="store_true")
    parser.add_argument("--timelapse", action="store_true")
    parser.add_argument("--roi", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--auto-orient", action="store_true")
    parser.add_argument("--no-auto-orient-exif", action="store_true")
    parser.add_argument("--auto-orient-exif-key", default="exif_orientation")
    parser.add_argument("--auto-orient-rotation", type=int, choices=[90, 180, 270], default=None)
    parser.add_argument("--auto-orient-flip-horizontal", action="store_true")
    parser.add_argument("--auto-orient-flip-vertical", action="store_true")
    parser.add_argument("--auto-adjust-contrast", action="store_true")
    parser.add_argument(
        "--contrast-method",
        choices=["adaptive_equalization", "histogram_equalization", "contrast_stretching"],
        default="adaptive_equalization",
    )
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid-width", type=int, default=8)
    parser.add_argument("--tile-grid-height", type=int, default=8)
    parser.add_argument("--lower-percentile", type=float, default=2.0)
    parser.add_argument("--upper-percentile", type=float, default=98.0)
    parser.add_argument("--output-min", type=int, default=0)
    parser.add_argument("--output-max", type=int, default=255)
    parser.add_argument("--flip-horizontal", action="store_true")
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--rotate90-k", type=int, default=0)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--crop-x", type=int, default=0)
    parser.add_argument("--crop-y", type=int, default=0)
    parser.add_argument("--crop-width", type=int, default=None)
    parser.add_argument("--crop-height", type=int, default=None)
    parser.add_argument("--no-crop-clamp", action="store_true")
    parser.add_argument("--rotation-angle", type=float, default=0.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-expand", action="store_true")
    parser.add_argument(
        "--rotation-border-mode",
        choices=["constant", "replicate", "reflect", "reflect_101", "wrap"],
        default="constant",
    )
    parser.add_argument("--rotation-border-value", type=int, default=0)
    parser.add_argument("--shear-x", type=float, default=0.0)
    parser.add_argument("--shear-y", type=float, default=0.0)
    parser.add_argument(
        "--shear-border-mode",
        choices=["constant", "replicate", "reflect", "reflect_101", "wrap"],
        default="constant",
    )
    parser.add_argument("--shear-border-value", type=int, default=0)
    parser.add_argument("--augment-greyscale", action="store_true")
    parser.add_argument("--greyscale-single-channel", action="store_true")
    parser.add_argument("--hue-delta", type=int, default=0)
    parser.add_argument("--saturation-scale", type=float, default=1.0)
    parser.add_argument("--brightness-beta", type=float, default=0.0)
    parser.add_argument("--exposure-gamma", type=float, default=1.0)
    parser.add_argument("--blur", action="store_true")
    parser.add_argument("--blur-kernel-size", type=int, default=5)
    parser.add_argument("--blur-sigma-x", type=float, default=0.0)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--noise-mode", choices=["gaussian", "salt_pepper"], default="gaussian")
    parser.add_argument("--noise-mean", type=float, default=0.0)
    parser.add_argument("--noise-stddev", type=float, default=10.0)
    parser.add_argument("--noise-amount", type=float, default=0.02)
    parser.add_argument("--salt-vs-pepper", type=float, default=0.5)
    parser.add_argument("--cutout", action="store_true")
    parser.add_argument("--cutout-x", type=int, default=0)
    parser.add_argument("--cutout-y", type=int, default=0)
    parser.add_argument("--cutout-width", type=int, default=32)
    parser.add_argument("--cutout-height", type=int, default=32)
    parser.add_argument("--cutout-fill-value", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--mosaic", action="store_true")
    parser.add_argument("--mosaic-image", dest="mosaic_images", action="append", default=[])
    parser.add_argument("--mosaic-output-width", type=int, default=None)
    parser.add_argument("--mosaic-output-height", type=int, default=None)
    parser.add_argument("--motion-blur", action="store_true")
    parser.add_argument("--motion-blur-kernel-size", type=int, default=9)
    parser.add_argument("--motion-blur-angle", type=float, default=0.0)
    parser.add_argument("--camera-gain", type=float, default=1.0)
    parser.add_argument("--camera-black-level", type=float, default=0.0)
    parser.add_argument(
        "--augmentation-config",
        default=None,
        help=(
            "Path to a JSON augmentation profile. Each entry should contain "
            "'name', optional 'params', and optional 'module'."
        ),
    )

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--keep-aspect", action="store_true")
    parser.add_argument("--camera-id", type=int, default=0)

    parser.add_argument("--brightness", type=int, default=0)
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--sharpen", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--grayscale", action="store_true")

    parser.add_argument("--output-root", default="output")
    parser.add_argument("--output-dir", default="captures")
    parser.add_argument("--timelapse-dir", default="timelapse")
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--fps", type=int, default=20)

    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--label", default="unknown")

    parser.add_argument("--export-dir", default="exports")

    parser.add_argument("--burst-count", type=int, default=5)
    parser.add_argument("--timelapse-interval", type=int, default=5)

    parser.add_argument("--roi-x", type=int, default=100)
    parser.add_argument("--roi-y", type=int, default=100)
    parser.add_argument("--roi-w", type=int, default=300)
    parser.add_argument("--roi-h", type=int, default=300)

    parser.add_argument("--motion-area", type=int, default=800)
    return parser


def parse_args(argv=None):
    """Parses command-line arguments using the ai-vision-tool parser.

    Args:
        argv (list[str] or None): Argument list to parse. Defaults to
            ``sys.argv[1:]`` when None.

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    return create_parser().parse_args(argv)


def main(argv=None):
    """Entry point for the ai-vision-tool CLI.

    Dispatches to one of four modes based on the parsed flags:
    - ``--serve-api``: Starts the FastAPI Uvicorn server.
    - ``--process-image-path``: Processes a single image file through a component.
    - ``--show-examples`` / ``--example-name``: Prints the examples catalog.
    - Default: Opens the webcam pipeline loop with the configured pipeline stages.

    Args:
        argv (list[str] or None): CLI arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)

    if args.serve_api:
        uvicorn.run(
            "ai_vision_tool.api:app",
            host=args.api_host,
            port=args.api_port,
            reload=args.api_reload,
        )
        return

    if args.process_image_path:
        if not args.component_category:
            raise ValueError("--component-category is required with --process-image-path")
        if not args.component_name:
            raise ValueError("--component-name is required with --process-image-path")
        if not args.image_path:
            raise ValueError("--image-path is required with --process-image-path")
        process_component_from_image_path(args)
        return

    if args.show_examples or args.example_name:
        print(format_examples(args.example_category, args.example_name))
        return

    output_dirs = resolve_output_dirs(args)
    ensure_output_dirs(output_dirs)
    pipeline = build_pipeline(args)

    capture_dir = output_dirs["captures"]
    video_dir = output_dirs["videos"]
    export_dir = output_dirs["exports"]
    roi = (args.roi_x, args.roi_y, args.roi_w, args.roi_h)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera_id}")

    print("\nControls:")
    print("  p -> capture image")
    print("  b -> capture burst")
    print("  r -> toggle recording")
    print("  d -> save dataset sample")
    print("  e -> export grayscale/edges")
    print("  o -> capture ROI")
    print("  q -> quit\n")
    print(f"[ai-vision-tool] Output root: {output_dirs['root'].resolve()}")

    video_writer = None
    video_path = None
    recording = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ai-vision-tool] Camera frame read failed.")
                break

            annotations = add_default_annotations() if args.annotate else []
            config = build_runtime_config(args, save_sample=False, annotations=annotations)
            processed_payload = run_pipeline(pipeline, frame, config, annotations=annotations)
            processed_frame = (
                processed_payload["frame"]
                if isinstance(processed_payload, dict)
                else processed_payload
            )

            processed_frame = draw_roi_overlay(processed_frame, roi)

            if recording and video_writer is not None:
                recording_frame = processed_frame.copy()
                cv2.circle(recording_frame, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(
                    recording_frame,
                    "REC",
                    (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                video_writer.write(recording_frame)
                display_frame = recording_frame
            else:
                display_frame = processed_frame

            cv2.imshow("AI Vision Pipeline", display_frame)
            key = cv2.waitKey(1) & 0xFF

            capture_requested = key == ord("p")
            burst_requested = key == ord("b")
            roi_requested = key == ord("o")
            export_requested = key == ord("e")
            save_sample = key == ord("d")

            if key == ord("r"):
                recording = not recording
                if recording:
                    video_writer, video_path = build_video_writer(
                        processed_frame, video_dir, args.fps
                    )
                    print(f"[ai-vision-tool] Started recording to {video_path}")
                elif video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print(f"[ai-vision-tool] Stopped recording. Saved to {video_path}")
                    video_path = None

            if capture_requested:
                save_capture(processed_frame, capture_dir)

            if burst_requested:
                for _ in range(args.burst_count):
                    save_capture(processed_frame, capture_dir, prefix="burst")
                    time.sleep(0.2)

            if roi_requested:
                save_roi(processed_frame, capture_dir, roi)

            if export_requested:
                save_exports(processed_frame, export_dir)

            if save_sample:
                dataset_config = build_runtime_config(
                    args, save_sample=True, annotations=annotations
                )
                run_pipeline(pipeline, frame, dataset_config, annotations=annotations)

            if key == ord("q"):
                break
    finally:
        for processor in pipeline.processors:
            processor.cleanup()

        if video_writer is not None:
            video_writer.release()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
