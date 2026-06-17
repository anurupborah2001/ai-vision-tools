# CLAUDE.md — ai-vision-tool

This file provides context for AI assistants navigating or modifying this repository.

---

## Package Identity

| Property | Value |
|----------|-------|
| PyPI package | `ai-vision-tool` |
| Python import namespace | `ai_vision_tool` |
| CLI entrypoint (webcam app + image processing) | `ai-vision-tool` |
| Package version | `0.4.1` |
| Requires Python | `>=3.10,<4.0` |

> **Important:** The package was renamed from `ai-vision-flow` / `visionflow` to
> `ai-vision-tool` / `ai_vision_tool`. All imports must use `ai_vision_tool`.
> Never use `visionflow` or `ai_vision_flow`.

---

## Module Map

```
ai_vision_tool/
│
├── __init__.py              # Lazy import registry (_EXPORTS dict + __getattr__)
├── __main__.py              # python -m ai_vision_tool entrypoint
│
├── cli/
│   └── main.py              # argparse entrypoint, webcam loop, --process-image-path
│
├── core/
│   ├── base.py              # AIVisionComponent base class
│   ├── data_types.py        # BBox, Detection, Keypoint, Pose, Mask, Track (dataclasses)
│   ├── device.py            # Device (auto CUDA/MPS/CPU, singleton default())
│   ├── batch_processor.py   # BatchProcessor (ThreadPoolExecutor, process_directory)
│   ├── scheduler.py         # Scheduler (token bucket), RateLimiter
│   └── memory_manager.py    # MemoryManager (buffer pool), GPUMemoryTracker
│
├── preprocessing/           # import: from ai_vision_tool.preprocessing import X
│   ├── auto_orient.py       # AutoOrient
│   ├── auto_adjust_contrast.py  # AutoAdjustContrast
│   ├── classical_segmentation.py  # RemoveBackground
│   ├── frame_resizer.py     # FrameResizer
│   ├── geometry.py          # Resize, LetterboxResize, CenterCrop, PadToSquare,
│   │                        # PerspectiveCorrection, Deskew, AutoCrop, FaceAlign,
│   │                        # ObjectCrop, BoundingBoxClamp, BoundingBoxNormalize, MaskResize
│   ├── intensity.py         # Normalize, Standardize, RescalePixels, ConvertColorSpace,
│   │                        # BGRToRGB, RGBToBGR, CLAHE, HistogramEqualization,
│   │                        # GammaCorrection, WhiteBalance, Denoise, Sharpen, Deblur,
│   │                        # Threshold, AdaptiveThreshold, EdgeDetection, ContourExtraction
│   └── quality.py           # ImageQualityCheck, BlurDetection, BrightnessCheck,
│                            # DuplicateImageCheck, CorruptImageCheck, AspectRatioFilter,
│                            # MinSizeFilter, MaxSizeFilter
│
├── augmentation/            # import: from ai_vision_tool.augmentation import X
│   ├── blur.py              # Blur
│   ├── blur_artifact.py     # GaussianBlur, MedianBlur, GlassBlur, DefocusBlur, ZoomBlur,
│   │                        # Emboss, Posterize, Solarize, Equalize, CompressionArtifacts,
│   │                        # JPEGCompression, Downscale, Superpixel, Sharpen
│   ├── brightness.py        # Brightness
│   ├── camera_gain.py       # CameraGain
│   ├── common.py            # parse_component_profile (JSON profile loader)
│   ├── composite.py         # MixUp, CutMix, CopyPaste, ObjectPaste, RandomOcclusion,
│   │                        # BoundingBoxJitter, Mosaic9
│   ├── crop.py              # Crop
│   ├── cutout.py            # Cutout
│   ├── exposure.py          # Exposure
│   ├── flip.py              # Flip
│   ├── geometric_random.py  # RandomResize, RandomScale, RandomCrop, RandomResizedCrop,
│   │                        # RandomPadding, Translate, AffineTransform,
│   │                        # PerspectiveTransform, ElasticTransform,
│   │                        # GridDistortion, OpticalDistortion
│   ├── grayscale.py         # Greyscale
│   ├── hue.py               # Hue
│   ├── mosaic.py            # Mosaic
│   ├── motion_blur.py       # MotionBlur
│   ├── noise.py             # Noise
│   ├── noise_dropout.py     # ISONoise, MultiplicativeNoise, SaltPepperNoise,
│   │                        # CoarseDropout, GridDropout, RandomErasing,
│   │                        # PixelDropout, MaskDropout
│   ├── rotate90.py          # Rotate90
│   ├── rotation.py          # Rotation
│   ├── saturation.py        # Saturation
│   ├── shear.py             # Shear
│   └── weather_light.py     # RandomShadow, RandomSunFlare, RandomFog, RandomRain,
│                            # RandomSnow, RandomGamma, ColorJitter, ChannelShuffle,
│                            # RGBShift, HSVShift, ToSepia, InvertImage,
│                            # RandomBrightnessContrast
│
├── capture/                 # import: from ai_vision_tool.capture import X
│   ├── image_capture.py     # PictureTaker
│   ├── burst_image_capture.py  # BurstPictureTaker
│   ├── video_capture.py     # VideoTaker
│   ├── video_recorder.py    # VideoRecorder
│   ├── frame_grabber.py     # FrameGrabber
│   ├── roi_capture.py       # ROICapture
│   ├── motion_detector.py   # MotionDetector
│   ├── screen_capture.py    # ScreenCapture
│   ├── time_lapse_capture.py   # TimeLapseCapture
│   ├── time_lapse.py        # TimeLapse
│   ├── image_template.py    # image_template()
│   └── video_template.py    # video_capture_template(key_manager, state, …), save_screenshot(),
│                            # KeyEventManager (register/handle), _make_recorder (patchable factory)
│
├── enhancement/             # import: from ai_vision_tool.enhancement import X
│   ├── denoiser.py          # Denoiser (nlmeans/bilateral/gaussian/DnCNN-ONNX)
│   ├── frame_enhancer.py    # FrameEnhancer (brightness/contrast/sharpen pass)
│   ├── low_light.py         # LowLightEnhancer (CLAHE/gamma/MSR/Zero-DCE/ONNX)
│   └── models/              # DL-backed enhancement (requires onnx/torch extra)
│       ├── colorization.py  # Colorizer (Zhang 2016 LAB-AB, pseudo_color, ONNX)
│       ├── deblurring.py    # Deblurrer (Wiener FFT, Richardson-Lucy, NAFNet-ONNX)
│       └── super_resolution.py  # SuperResolution (cv2_dnn_superres/ONNX/bicubic)
│
├── io/                      # import: from ai_vision_tool.io import X
│   ├── image_io.py          # ImageReader, ImageWriter (pattern filenames)
│   ├── video_io.py          # VideoReader (seek, read_all), VideoWriter
│   ├── camera_source.py     # CameraSource (webcam/RTSP/HTTP, auto-reconnect)
│   ├── dataset_collector.py # DatasetCollector
│   ├── dataset_exporter.py  # DatasetExporter (YOLO/COCO/VOC formats)
│   └── image_exporter.py    # ImageExporter
│
├── detection/               # import: from ai_vision_tool.detection import X
│   ├── object_detector.py   # ObjectDetector (ultralytics YOLO or ONNX + greedy NMS)
│   ├── face_detector.py     # FaceDetector (OpenCV Haar or MediaPipe)
│   ├── keypoint_detector.py # KeypointDetector (MediaPipe pose, YOLO-pose)
│   ├── text_detector.py     # TextDetector (EasyOCR, PaddleOCR, EAST)
│   └── anomaly_detector.py  # AnomalyDetector (statistical/patchcore/pca)
│
├── tracking/                # import: from ai_vision_tool.tracking import X
│   ├── kalman_filter.py     # KalmanFilter (7-state SORT formulation)
│   ├── track_manager.py     # TrackManager (IoU Hungarian, tentative/active/lost)
│   ├── byte_tracker.py      # ByteTracker (two-stage high/low-conf association)
│   ├── deepsort_tracker.py  # DeepSORTTracker (HOG embedding, cosine distance)
│   └── reid_extractor.py    # ReIDExtractor (HOG/OSNet-ONNX, build_gallery)
│
├── segmentation/            # import: from ai_vision_tool.segmentation import X
│   ├── semantic_segmenter.py    # SemanticSegmenter (ONNX/dnn/torch, VOC21)
│   ├── instance_segmenter.py    # InstanceSegmenter (YOLO-seg masks)
│   ├── panoptic_segmenter.py    # PanopticSegmenter (stuff/thing separation)
│   ├── sam_segmenter.py         # SAMSegmenter (point/box/auto-everything)
│   └── mask_post_processor.py   # MaskPostProcessor (erode/dilate/fill/largest_only)
│
├── models/                  # import: from ai_vision_tool.models import X
│   ├── registry.py          # ModelRegistry (JSON cache, load, from_huggingface)
│   ├── downloader.py        # ModelDownloader (urllib, SHA256, HuggingFace)
│   ├── benchmark.py         # ModelBenchmark (p50/p95/p99, tracemalloc, ASCII)
│   └── backends/
│       ├── onnx_model.py    # ONNXModel (onnxruntime, data["model_output"])
│       ├── torch_model.py   # TorchModel (TorchScript, device auto, half precision)
│       └── tflite_model.py  # TFLiteModel (tflite-runtime/tensorflow fallback)
│
├── pipelines/               # import: from ai_vision_tool.pipelines import X
│   ├── vision_pipeline.py   # AIVisionPipeline (Chain of Responsibility)
│   ├── prebuilt.py          # PrebuiltPipelines (detection/augmentation/tracking/…)
│   ├── serializer.py        # PipelineSerializer (to_dict/from_dict, YAML/JSON save)
│   ├── async_pipeline.py    # AsyncPipeline, AsyncComponent (asyncio executor)
│   └── parallel_pipeline.py # ParallelPipeline, FanOutPipeline (ThreadPoolExecutor)
│
├── streaming/               # import: from ai_vision_tool.streaming import X
│   ├── frame_stream.py      # FrameStream, DirectoryStream (context manager, iterator)
│   ├── rtsp_client.py       # RTSPClient (background reader, reconnect)
│   └── buffered_stream.py   # BufferedStream (drop policy), SlidingWindowBuffer
│
├── visualization/           # import: from ai_vision_tool.visualization import X
│   ├── frame_viewer.py              # FrameViewer (FPS overlay, headless-safe)
│   ├── frame_annotator.py           # FrameAnnotator (text/box/line overlays)
│   ├── bbox_renderer.py             # BBoxRenderer (alpha fill, ColorPalette)
│   ├── heatmap_renderer.py          # HeatmapRenderer (Gaussian blob, motion, anomaly)
│   ├── dashboard_view.py            # DashboardSink (Gradio or MJPEG HTTP)
│   └── video_annotation_exporter.py # VideoAnnotationExporter (burn + JSON sidecar)
│
├── integrations/
│   ├── cloud/               # import: from ai_vision_tool.integrations.cloud import X
│   │   ├── s3_source.py     # S3Source (boto3) — requires [cloud] extra
│   │   └── gcs_source.py    # GCSSource (google-cloud-storage) — requires [cloud] extra
│   ├── labeling/            # import: from ai_vision_tool.integrations.labeling import X
│   │   ├── auto_labeller.py          # AutoLabeller
│   │   ├── darknet_auto_labeler.py   # DarknetAutoLabeler
│   │   └── tensorflow_auto_labeler.py  # TensorFlowAutoLabeler
│   └── streaming/           # import: from ai_vision_tool.integrations.streaming import X
│       ├── websocket_sink.py  # WebSocketSink, WebSocketSource — requires [websocket] extra
│       └── kafka_io.py        # KafkaSource, KafkaSink — requires [kafka] extra
│
├── config/                  # import: from ai_vision_tool.config import X
│   ├── yaml_config.py       # YAMLConfig (dot-notation get, merge, validate, reload)
│   ├── json_config.py       # JSONConfig (same interface + save, from_dict)
│   ├── registry.py          # ComponentRegistry (singleton, register decorator, build)
│   ├── profile_loader.py    # ProfileLoader (search paths, load_pipeline)
│   └── env_config.py        # EnvConfig (prefix-based env vars, cast, require)
│
└── utils/                   # import: from ai_vision_tool.utils import X
    ├── color_palette.py     # ColorPalette (golden-ratio hue, get/as_dict)
    ├── metrics_logger.py    # MetricsLogger, MetricsLoggerComponent
    ├── frame_sampler.py     # FrameSampler (count/fps/random modes)
    ├── image_hash.py        # ImageHash (phash/ahash/dhash, duplicate detection)
    └── draw_utils.py        # DrawUtils (bboxes, masks, keypoints rendering)
```

---

## Design Patterns

### Lazy Imports

`__init__.py` uses a `_EXPORTS` dict and `__getattr__` to load modules only when first
accessed. When adding a new top-level export:

1. Add an entry to `_EXPORTS` in `ai_vision_tool/__init__.py`:
   ```python
   "MyNewClass": ("ai_vision_tool.my_domain.my_module", "MyNewClass"),
   ```
2. Do **not** add a direct `from ... import ...` at the top of `__init__.py`.
3. `__getattr__` caches the resolved value into `globals()` so subsequent accesses are instant.

### Payload Convention

Every component accepts either:
- A raw NumPy array: `component.run(image)`
- A payload dict: `component.run({"frame": image, "bboxes": [...], "mask": ..., ...})`

When a component returns a payload dict, the `"frame"` key always holds the processed
NumPy array. Downstream components receive the full dict as their input.

### Component Interface

All components subclass `AIVisionComponent` from `ai_vision_tool.core.base` and
implement:

```python
def run(self, data, config=None):
    ...
```

`config` is optional; some components only read constructor arguments.
`cleanup()` is available for components that hold resources (e.g., video writers).

### Pipeline Execution

`AIVisionPipeline.execute()` chains `processor.run(data, config)` calls. The output of
each processor becomes the `data` input of the next. `global_config` is passed unchanged
to every processor.

### Top-Level Namespace Exports

Only stable, commonly used classes are exported from `ai_vision_tool`. All classes are
also importable from their specific submodule paths.

The `_EXPORTS` dict in `__init__.py` registers 130+ classes across all modules. Adding a
new export follows the same two-step pattern: add entry to `_EXPORTS`, then import via
the lazy `__getattr__`.

### Optional Heavy Dependencies

Heavy deps (onnxruntime, torch, mediapipe, ultralytics, boto3, etc.) are imported inside
`setup()` with `ImportError` + install hint if missing. Every class that can use a heavy
dep also has a pure cv2/numpy fallback so the library works with no extras installed.

---

## Dev Workflow

### Install Dependencies

```bash
make install-dev   # uv sync --dev
```

### Common Tasks (Makefile)

```bash
make help          # list all targets
make lint          # ruff + isort + black (check only)
make format        # auto-fix with ruff + isort + black
make test          # full pytest suite
make test-fast     # stop on first failure (-x)
make build         # sdist + wheel → dist/
make clean         # remove dist/ build/
```

### Run Tests

```bash
make test
# or individually:
uv run pytest tests/test_imports.py
uv run pytest tests/test_preprocessing_components.py
uv run pytest tests/test_basic_augmentations.py
uv run pytest tests/test_advanced_augmentations.py
uv run pytest tests/test_capture_components.py
uv run pytest tests/test_core_components.py
uv run pytest tests/test_labeler_components.py
uv run pytest tests/test_cli_file_processing.py
```

> `tests/test_api.py` is intentionally skipped — the API module moved to `backup/` and
> will be re-introduced as a proper `[api]` extra in a future release.

### Lint and Format

```bash
make lint     # check only
make format   # auto-fix
```

### Pre-Commit Hooks

Hooks are managed via [pre-commit](https://pre-commit.com/). Install once:

```bash
make hooks
# equivalent to:
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
uv run pre-commit install --hook-type pre-push
```

Active hooks:
- **pre-commit**: `check-yaml`, `check-toml`, `check-merge-conflict`, `end-of-file-fixer`,
  `trailing-whitespace`, `ruff`, `isort`, `black`
- **commit-msg**: commitizen validates message against `cz_conventional_commits` schema
- **pre-push**: `pytest` full suite

### Commit Message Convention

Commit messages are validated by [commitizen](https://github.com/commitizen-tools/commitizen).
Use the interactive prompt to build a valid message:

```bash
uv run cz commit
```

Or write manually following [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add fog and rain augmentation pipeline stage
fix: correct augmentation profile parameter names
docs: update publishing workflow
test: add missing unit tests for basic augmentations
chore: bump dev dependency versions
```

| Prefix | Triggers bump |
|--------|--------------|
| `fix:`, `perf:`, `refactor:` | patch |
| `feat:` | minor |
| `BREAKING CHANGE:` / `feat!:` / `fix!:` | major |

---

## Release

Releases are fully automated. Push qualifying conventional commits to `master`;
`semantic-versioning.yml` bumps the version, creates a tag, publishes a GitHub Release,
and publishes to PyPI via OIDC trusted publishing.

```bash
# Local version bump (CI does this automatically — only run locally to preview)
make bump          # auto from commits
make bump-patch    # force patch
make bump-minor    # force minor

# Trigger CI release manually
gh workflow run semantic-versioning.yml -f bump=patch
```

Version files updated automatically by commitizen:
- `pyproject.toml` → `[project].version` and `[tool.commitizen].version`
- `ai_vision_tool/__init__.py` → `__version__`

See `PUBLISHING.md` for the full automated flow and PyPI trusted publisher setup.
