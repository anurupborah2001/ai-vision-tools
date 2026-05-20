# CLAUDE.md — ai-vision-tool

This file provides context for AI assistants navigating or modifying this repository.

---

## Package Identity

| Property | Value |
|----------|-------|
| PyPI package | `ai-vision-tool` |
| Python import namespace | `ai_vision_tool` |
| CLI entrypoint (webcam app + image processing) | `ai-vision-tool` |
| CLI entrypoint (FastAPI server) | `ai-vision-tool-api` |
| Package version | `0.2.0` |
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
├── cli.py                   # argparse entrypoint, examples catalog, webcam loop
├── backup/api.py            # FastAPI app factory (moved to backup/)
├── backup/api_service.py    # encode_image_base64, decode_image_base64 (moved to backup/)
│
├── pipeline/
│   └── vision_pipeline.py   # AIVisionPipeline (Chain of Responsibility)
│
├── components/
│   ├── base.py              # AIVisionComponent base class
│   ├── _image_utils.py      # Shared image helper utilities
│   │
│   ├── preprocessing/       # Preprocessing transforms
│   │   ├── geometry.py      # Resize, LetterboxResize, CenterCrop, PadToSquare,
│   │   │                    # PerspectiveCorrection, Deskew, AutoCrop, FaceAlign,
│   │   │                    # ObjectCrop, BoundingBoxClamp, BoundingBoxNormalize, MaskResize
│   │   ├── intensity.py     # Normalize, Standardize, RescalePixels, ConvertColorSpace,
│   │   │                    # BGRToRGB, RGBToBGR, CLAHE, HistogramEqualization,
│   │   │                    # GammaCorrection, WhiteBalance, Denoise, Sharpen, Deblur,
│   │   │                    # Threshold, AdaptiveThreshold, EdgeDetection, ContourExtraction
│   │   ├── quality.py       # ImageQualityCheck, BlurDetection, BrightnessCheck,
│   │   │                    # DuplicateImageCheck, CorruptImageCheck, AspectRatioFilter,
│   │   │                    # MinSizeFilter, MaxSizeFilter
│   │   ├── segmentation.py  # RemoveBackground
│   │   ├── auto_orient.py   # AutoOrient
│   │   └── auto_adjust_contrast.py  # AutoAdjustContrast
│   │
│   ├── augmentations/       # Augmentation transforms
│   │   ├── blur.py          # Blur
│   │   ├── blur_artifact.py # GaussianBlur, MedianBlur, GlassBlur, DefocusBlur, ZoomBlur,
│   │   │                    # Emboss, Posterize, Solarize, Equalize, CompressionArtifacts,
│   │   │                    # JPEGCompression, Downscale, Superpixel, Sharpen (augmentation)
│   │   ├── brightness.py    # Brightness
│   │   ├── camera_gain.py   # CameraGain
│   │   ├── composite.py     # MixUp, CutMix, CopyPaste, ObjectPaste, RandomOcclusion,
│   │   │                    # BoundingBoxJitter, Mosaic9
│   │   ├── crop.py          # Crop
│   │   ├── cutout.py        # Cutout
│   │   ├── exposure.py      # Exposure
│   │   ├── flip.py          # Flip
│   │   ├── geometric_random.py  # RandomResize, RandomScale, RandomCrop, RandomResizedCrop,
│   │   │                        # RandomPadding, Translate, AffineTransform,
│   │   │                        # PerspectiveTransform, ElasticTransform,
│   │   │                        # GridDistortion, OpticalDistortion
│   │   ├── greyscale.py     # Greyscale
│   │   ├── hue.py           # Hue
│   │   ├── mosaic.py        # Mosaic
│   │   ├── motion_blur.py   # MotionBlur
│   │   ├── noise.py         # Noise
│   │   ├── noise_dropout.py # ISONoise, MultiplicativeNoise, SaltPepperNoise,
│   │   │                    # CoarseDropout, GridDropout, RandomErasing,
│   │   │                    # PixelDropout, MaskDropout
│   │   ├── rotate90.py      # Rotate90
│   │   ├── rotation.py      # Rotation
│   │   ├── saturation.py    # Saturation
│   │   ├── shear.py         # Shear
│   │   ├── weather_light.py # RandomShadow, RandomSunFlare, RandomFog, RandomRain,
│   │   │                    # RandomSnow, RandomGamma, ColorJitter, ChannelShuffle,
│   │   │                    # RGBShift, HSVShift, ToSepia, InvertImage,
│   │   │                    # RandomBrightnessContrast
│   │   └── common.py        # parse_component_profile (JSON profile loader)
│   │
│   ├── frame_enhancer.py    # FrameEnhancer
│   ├── frame_resizer.py     # FrameResizer
│   ├── frame_annotator.py   # FrameAnnotator
│   ├── frame_grabber.py     # FrameGrabber
│   ├── motion_detector.py   # MotionDetector
│   ├── picture_taker.py     # PictureTaker
│   ├── burst_picture_taker.py  # BurstPictureTaker
│   ├── roi_capture.py       # ROICapture
│   ├── video_taker.py       # VideoTaker
│   ├── time_lapse_capture.py   # TimeLapseCapture
│   ├── time_lapse.py        # TimeLapse
│   ├── dataset_collector.py # DatasetCollector
│   ├── image_exporter.py    # ImageExporter
│   ├── auto_labeller.py     # AutoLabeller
│   ├── darknet_auto_labeler.py   # DarknetAutoLabeler
│   └── tensorflow_auto_labeler.py  # TensorFlowAutoLabeler
│
├── capture/
│   ├── image_template.py    # image_template()
│   └── video_template.py    # video_capture_template(), save_screenshot()
│
├── utils/
│   ├── color_palette.py     # ColorPalette (golden-ratio hue, get/as_dict)
│   ├── metrics_logger.py    # MetricsLogger, MetricsLoggerComponent
│   ├── frame_sampler.py     # FrameSampler (count/fps/random modes)
│   ├── image_hash.py        # ImageHash (phash/ahash/dhash, duplicate detection)
│   └── draw_utils.py        # DrawUtils (bboxes, masks, keypoints rendering)
│
├── core/
│   ├── device.py            # Device (auto CUDA/MPS/CPU, singleton default())
│   ├── data_types.py        # BBox, Detection, Keypoint, Pose, Mask,
│   │                        # SegmentationResult, Track (typed dataclasses)
│   ├── batch_processor.py   # BatchProcessor (ThreadPoolExecutor, process_directory)
│   ├── scheduler.py         # Scheduler (token bucket), RateLimiter
│   └── memory_manager.py    # MemoryManager (buffer pool), GPUMemoryTracker
│
├── config/
│   ├── yaml_config.py       # YAMLConfig (dot-notation get, merge, validate, reload)
│   ├── json_config.py       # JSONConfig (same interface + save, from_dict)
│   ├── registry.py          # ComponentRegistry (singleton, register decorator, build)
│   ├── profile_loader.py    # ProfileLoader (search paths, load_pipeline)
│   └── env_config.py        # EnvConfig (prefix-based env vars, cast, require)
│
├── io/
│   ├── image_io.py          # ImageReader, ImageWriter (pattern filenames)
│   ├── video_io.py          # VideoReader (seek, read_all), VideoWriter
│   ├── camera_source.py     # CameraSource (webcam/RTSP/HTTP, auto-reconnect)
│   ├── cloud_source.py      # S3Source (boto3), GCSSource (google-cloud-storage)
│   └── dataset_exporter.py  # DatasetExporter (YOLO/COCO/VOC formats)
│
├── models/
│   ├── registry.py          # ModelRegistry (JSON cache, load, from_huggingface)
│   ├── onnx_model.py        # ONNXModel (onnxruntime, preprocess, data["model_output"])
│   ├── torch_model.py       # TorchModel (TorchScript, device auto, half precision)
│   ├── tflite_model.py      # TFLiteModel (tflite-runtime/tensorflow fallback)
│   ├── downloader.py        # ModelDownloader (urllib, SHA256, HuggingFace)
│   └── benchmark.py         # ModelBenchmark (p50/p95/p99, tracemalloc, ASCII report)
│
├── detection/
│   ├── object_detector.py   # ObjectDetector (ultralytics YOLO or ONNX + greedy NMS)
│   ├── face_detector.py     # FaceDetector (OpenCV Haar or MediaPipe)
│   ├── keypoint_detector.py # KeypointDetector (MediaPipe pose, YOLO-pose)
│   ├── text_detector.py     # TextDetector (EasyOCR, PaddleOCR, EAST)
│   └── anomaly_detector.py  # AnomalyDetector (statistical/patchcore/pca)
│
├── tracking/
│   ├── kalman_filter.py     # KalmanFilter (7-state SORT formulation)
│   ├── track_manager.py     # TrackManager (IoU Hungarian, tentative/active/lost states)
│   ├── byte_tracker.py      # ByteTracker (two-stage high/low-conf association)
│   ├── deepsort_tracker.py  # DeepSORTTracker (HOG embedding, cosine distance)
│   └── reid_extractor.py    # ReIDExtractor (HOG/OSNet-ONNX/CLIP, build_gallery)
│
├── segmentation/
│   ├── semantic_segmenter.py    # SemanticSegmenter (ONNX/dnn/torch, VOC21 defaults)
│   ├── instance_segmenter.py    # InstanceSegmenter (YOLO-seg masks)
│   ├── panoptic_segmenter.py    # PanopticSegmenter (stuff/thing separation)
│   ├── sam_segmenter.py         # SAMSegmenter (SAM point/box/auto-everything)
│   └── mask_post_processor.py   # MaskPostProcessor (erode/dilate/fill/largest_only)
│
├── enhancement/
│   ├── super_resolution.py  # SuperResolution (cv2_dnn_superres/ONNX/bicubic)
│   ├── denoiser.py          # Denoiser (nlmeans/bilateral/gaussian/DnCNN-ONNX)
│   ├── deblurrer.py         # Deblurrer (Wiener FFT, Richardson-Lucy, NAFNet-ONNX)
│   ├── low_light_enhancer.py # LowLightEnhancer (CLAHE/gamma/MSR/Zero-DCE/ONNX)
│   └── colorizer.py         # Colorizer (Zhang 2016 LAB-AB, pseudo_color, ONNX)
│
├── pipelines/
│   ├── prebuilt.py          # PrebuiltPipelines (detection/augmentation/tracking/…)
│   ├── serializer.py        # PipelineSerializer (to_dict/from_dict, YAML/JSON save)
│   ├── async_pipeline.py    # AsyncPipeline, AsyncComponent (asyncio executor)
│   └── parallel_pipeline.py # ParallelPipeline, FanOutPipeline (ThreadPoolExecutor)
│
├── streaming/
│   ├── frame_stream.py      # FrameStream, DirectoryStream (context manager, iterator)
│   ├── rtsp_client.py       # RTSPClient (background reader, reconnect), RTSPServer
│   ├── websocket_sink.py    # WebSocketSink, WebSocketSource (websockets/MJPEG fallback)
│   ├── kafka_io.py          # KafkaSource, KafkaSink (confluent-kafka/kafka-python)
│   └── buffered_stream.py   # BufferedStream (drop policy), SlidingWindowBuffer
│
└── visualization/
    ├── frame_viewer.py              # FrameViewer (FPS overlay, headless-safe)
    ├── bbox_renderer.py             # BBoxRenderer (alpha fill, ColorPalette)
    ├── heatmap_renderer.py          # HeatmapRenderer (Gaussian blob, motion, anomaly)
    ├── dashboard_sink.py            # DashboardSink (Gradio or MJPEG HTTP)
    └── video_annotation_exporter.py # VideoAnnotationExporter (burn + JSON sidecar)
```

---

## Design Patterns

### Lazy Imports

`__init__.py` uses a `_EXPORTS` dict and `__getattr__` to load modules only when first
accessed. When adding a new top-level export:

1. Add an entry to `_EXPORTS` in `ai_vision_tool/__init__.py`:
   ```python
   "MyNewClass": ("ai_vision_tool.components.my_module", "MyNewClass"),
   ```
2. Do **not** add a direct `from ... import ...` at the top of `__init__.py`.

### Payload Convention

Every component accepts either:
- A raw NumPy array: `component.run(image)`
- A payload dict: `component.run({"frame": image, "bboxes": [...], "mask": ..., ...})`

When a component returns a payload dict, the `"frame"` key always holds the processed
NumPy array. Downstream components receive the full dict as their input.

### Component Interface

All components subclass `AIVisionComponent` from `ai_vision_tool.components.base` and
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
# uv (recommended)
uv sync --dev

# Poetry
poetry install --with dev
```

### Run Tests

```bash
pytest                                          # all tests
pytest tests/test_preprocessing_components.py
pytest tests/test_basic_augmentations.py
pytest tests/test_advanced_augmentations.py
pytest tests/test_capture_components.py
pytest tests/test_core_components.py
pytest tests/test_labeler_components.py
pytest tests/test_api.py
pytest tests/test_cli_file_processing.py
```

### Lint and Format

```bash
ruff check .
black .
isort .
```

### Pre-Commit Hooks

```bash
# Install all hook types
pre-commit install
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg

# Run all hooks manually
pre-commit run --all-files
```

Hooks enforce: `ruff`, `isort`, `black`, `pre-commit-hooks`, Conventional Commits,
and `pytest` on pre-push.

### Commit Message Convention

This repository uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add fog and rain augmentation coverage
fix: correct augmentation profile parameter names
docs: update publishing workflow
test: add missing unit tests for basic augmentations
chore: rename template → capture module
```

---

## Release

See `PUBLISHING.md` for the full release checklist and PyPI upload commands.

Key steps:
1. Bump version in `pyproject.toml` (`[project].version` and `[tool.poetry].version`)
2. Update `__version__` in `ai_vision_tool/__init__.py`
3. Run `python -m build`
4. Run `twine upload dist/*`
