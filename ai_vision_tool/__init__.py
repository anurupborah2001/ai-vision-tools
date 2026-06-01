"""Top-level package for AI Vision Tool."""

import importlib

__version__ = "0.4.0"

_EXPORTS = {
    # core
    "AIVisionComponent": ("ai_vision_tool.core.base",                   "AIVisionComponent"),
    "AIVisionPipeline":  ("ai_vision_tool.pipelines.vision_pipeline",   "AIVisionPipeline"),

    # preprocessing
    "AdaptiveThreshold":        ("ai_vision_tool.preprocessing.intensity",              "AdaptiveThreshold"),
    "AspectRatioFilter":        ("ai_vision_tool.preprocessing.quality",                "AspectRatioFilter"),
    "AutoAdjustContrast":       ("ai_vision_tool.preprocessing.auto_adjust_contrast",   "AutoAdjustContrast"),
    "AutoCrop":                 ("ai_vision_tool.preprocessing.geometry",               "AutoCrop"),
    "AutoOrient":               ("ai_vision_tool.preprocessing.auto_orient",            "AutoOrient"),
    "BGRToRGB":                 ("ai_vision_tool.preprocessing.intensity",              "BGRToRGB"),
    "BlurDetection":            ("ai_vision_tool.preprocessing.quality",                "BlurDetection"),
    "BoundingBoxClamp":         ("ai_vision_tool.preprocessing.geometry",               "BoundingBoxClamp"),
    "BoundingBoxNormalize":     ("ai_vision_tool.preprocessing.geometry",               "BoundingBoxNormalize"),
    "BrightnessCheck":          ("ai_vision_tool.preprocessing.quality",                "BrightnessCheck"),
    "CenterCrop":               ("ai_vision_tool.preprocessing.geometry",               "CenterCrop"),
    "CLAHE":                    ("ai_vision_tool.preprocessing.intensity",              "CLAHE"),
    "ContourExtraction":        ("ai_vision_tool.preprocessing.intensity",              "ContourExtraction"),
    "ConvertColorSpace":        ("ai_vision_tool.preprocessing.intensity",              "ConvertColorSpace"),
    "CorruptImageCheck":        ("ai_vision_tool.preprocessing.quality",                "CorruptImageCheck"),
    "Deblur":                   ("ai_vision_tool.preprocessing.intensity",              "Deblur"),
    "Denoise":                  ("ai_vision_tool.preprocessing.intensity",              "Denoise"),
    "Deskew":                   ("ai_vision_tool.preprocessing.geometry",               "Deskew"),
    "DuplicateImageCheck":      ("ai_vision_tool.preprocessing.quality",                "DuplicateImageCheck"),
    "EdgeDetection":            ("ai_vision_tool.preprocessing.intensity",              "EdgeDetection"),
    "FaceAlign":                ("ai_vision_tool.preprocessing.geometry",               "FaceAlign"),
    "FrameResizer":             ("ai_vision_tool.preprocessing.frame_resizer",          "FrameResizer"),
    "GammaCorrection":          ("ai_vision_tool.preprocessing.intensity",              "GammaCorrection"),
    "HistogramEqualization":    ("ai_vision_tool.preprocessing.intensity",              "HistogramEqualization"),
    "ImageQualityCheck":        ("ai_vision_tool.preprocessing.quality",                "ImageQualityCheck"),
    "LetterboxResize":          ("ai_vision_tool.preprocessing.geometry",               "LetterboxResize"),
    "MaskResize":               ("ai_vision_tool.preprocessing.geometry",               "MaskResize"),
    "MaxSizeFilter":            ("ai_vision_tool.preprocessing.quality",                "MaxSizeFilter"),
    "MinSizeFilter":            ("ai_vision_tool.preprocessing.quality",                "MinSizeFilter"),
    "Normalize":                ("ai_vision_tool.preprocessing.intensity",              "Normalize"),
    "ObjectCrop":               ("ai_vision_tool.preprocessing.geometry",               "ObjectCrop"),
    "PadToSquare":              ("ai_vision_tool.preprocessing.geometry",               "PadToSquare"),
    "PerspectiveCorrection":    ("ai_vision_tool.preprocessing.geometry",               "PerspectiveCorrection"),
    "RemoveBackground":         ("ai_vision_tool.preprocessing.classical_segmentation", "RemoveBackground"),
    "RescalePixels":            ("ai_vision_tool.preprocessing.intensity",              "RescalePixels"),
    "Resize":                   ("ai_vision_tool.preprocessing.geometry",               "Resize"),
    "RGBToBGR":                 ("ai_vision_tool.preprocessing.intensity",              "RGBToBGR"),
    "Sharpen":                  ("ai_vision_tool.preprocessing.intensity",              "Sharpen"),
    "Standardize":              ("ai_vision_tool.preprocessing.intensity",              "Standardize"),
    "Threshold":                ("ai_vision_tool.preprocessing.intensity",              "Threshold"),
    "WhiteBalance":             ("ai_vision_tool.preprocessing.intensity",              "WhiteBalance"),

    # augmentation
    "AffineTransform":          ("ai_vision_tool.augmentation.geometric_random",  "AffineTransform"),
    "AugmentationSharpen":      ("ai_vision_tool.augmentation.blur_artifact",     "Sharpen"),
    "Blur":                     ("ai_vision_tool.augmentation.blur",              "Blur"),
    "BoundingBoxJitter":        ("ai_vision_tool.augmentation.composite",         "BoundingBoxJitter"),
    "Brightness":               ("ai_vision_tool.augmentation.brightness",        "Brightness"),
    "CameraGain":               ("ai_vision_tool.augmentation.camera_gain",       "CameraGain"),
    "ChannelShuffle":           ("ai_vision_tool.augmentation.weather_light",     "ChannelShuffle"),
    "CoarseDropout":            ("ai_vision_tool.augmentation.noise_dropout",     "CoarseDropout"),
    "ColorJitter":              ("ai_vision_tool.augmentation.weather_light",     "ColorJitter"),
    "CompressionArtifacts":     ("ai_vision_tool.augmentation.blur_artifact",     "CompressionArtifacts"),
    "CopyPaste":                ("ai_vision_tool.augmentation.composite",         "CopyPaste"),
    "Crop":                     ("ai_vision_tool.augmentation.crop",              "Crop"),
    "CutMix":                   ("ai_vision_tool.augmentation.composite",         "CutMix"),
    "Cutout":                   ("ai_vision_tool.augmentation.cutout",            "Cutout"),
    "DefocusBlur":              ("ai_vision_tool.augmentation.blur_artifact",     "DefocusBlur"),
    "Downscale":                ("ai_vision_tool.augmentation.blur_artifact",     "Downscale"),
    "ElasticTransform":         ("ai_vision_tool.augmentation.geometric_random",  "ElasticTransform"),
    "Emboss":                   ("ai_vision_tool.augmentation.blur_artifact",     "Emboss"),
    "Equalize":                 ("ai_vision_tool.augmentation.blur_artifact",     "Equalize"),
    "Exposure":                 ("ai_vision_tool.augmentation.exposure",          "Exposure"),
    "Flip":                     ("ai_vision_tool.augmentation.flip",              "Flip"),
    "GaussianBlur":             ("ai_vision_tool.augmentation.blur_artifact",     "GaussianBlur"),
    "GlassBlur":                ("ai_vision_tool.augmentation.blur_artifact",     "GlassBlur"),
    "Greyscale":                ("ai_vision_tool.augmentation.grayscale",         "Greyscale"),
    "GridDistortion":           ("ai_vision_tool.augmentation.geometric_random",  "GridDistortion"),
    "GridDropout":              ("ai_vision_tool.augmentation.noise_dropout",     "GridDropout"),
    "HSVShift":                 ("ai_vision_tool.augmentation.weather_light",     "HSVShift"),
    "Hue":                      ("ai_vision_tool.augmentation.hue",               "Hue"),
    "ISONoise":                 ("ai_vision_tool.augmentation.noise_dropout",     "ISONoise"),
    "InvertImage":              ("ai_vision_tool.augmentation.weather_light",     "InvertImage"),
    "JPEGCompression":          ("ai_vision_tool.augmentation.blur_artifact",     "JPEGCompression"),
    "MaskDropout":              ("ai_vision_tool.augmentation.noise_dropout",     "MaskDropout"),
    "MedianBlur":               ("ai_vision_tool.augmentation.blur_artifact",     "MedianBlur"),
    "MixUp":                    ("ai_vision_tool.augmentation.composite",         "MixUp"),
    "Mosaic":                   ("ai_vision_tool.augmentation.mosaic",            "Mosaic"),
    "Mosaic9":                  ("ai_vision_tool.augmentation.composite",         "Mosaic9"),
    "MotionBlur":               ("ai_vision_tool.augmentation.motion_blur",       "MotionBlur"),
    "MultiplicativeNoise":      ("ai_vision_tool.augmentation.noise_dropout",     "MultiplicativeNoise"),
    "Noise":                    ("ai_vision_tool.augmentation.noise",             "Noise"),
    "ObjectPaste":              ("ai_vision_tool.augmentation.composite",         "ObjectPaste"),
    "OpticalDistortion":        ("ai_vision_tool.augmentation.geometric_random",  "OpticalDistortion"),
    "PerspectiveTransform":     ("ai_vision_tool.augmentation.geometric_random",  "PerspectiveTransform"),
    "PixelDropout":             ("ai_vision_tool.augmentation.noise_dropout",     "PixelDropout"),
    "Posterize":                ("ai_vision_tool.augmentation.blur_artifact",     "Posterize"),
    "RandomBrightnessContrast": ("ai_vision_tool.augmentation.weather_light",     "RandomBrightnessContrast"),
    "RandomCrop":               ("ai_vision_tool.augmentation.geometric_random",  "RandomCrop"),
    "RandomErasing":            ("ai_vision_tool.augmentation.noise_dropout",     "RandomErasing"),
    "RandomFog":                ("ai_vision_tool.augmentation.weather_light",     "RandomFog"),
    "RandomGamma":              ("ai_vision_tool.augmentation.weather_light",     "RandomGamma"),
    "RandomOcclusion":          ("ai_vision_tool.augmentation.composite",         "RandomOcclusion"),
    "RandomPadding":            ("ai_vision_tool.augmentation.geometric_random",  "RandomPadding"),
    "RandomRain":               ("ai_vision_tool.augmentation.weather_light",     "RandomRain"),
    "RandomResize":             ("ai_vision_tool.augmentation.geometric_random",  "RandomResize"),
    "RandomResizedCrop":        ("ai_vision_tool.augmentation.geometric_random",  "RandomResizedCrop"),
    "RandomScale":              ("ai_vision_tool.augmentation.geometric_random",  "RandomScale"),
    "RandomShadow":             ("ai_vision_tool.augmentation.weather_light",     "RandomShadow"),
    "RandomSnow":               ("ai_vision_tool.augmentation.weather_light",     "RandomSnow"),
    "RandomSunFlare":           ("ai_vision_tool.augmentation.weather_light",     "RandomSunFlare"),
    "RGBShift":                 ("ai_vision_tool.augmentation.weather_light",     "RGBShift"),
    "Rotate90":                 ("ai_vision_tool.augmentation.rotate90",          "Rotate90"),
    "Rotation":                 ("ai_vision_tool.augmentation.rotation",          "Rotation"),
    "Saturation":               ("ai_vision_tool.augmentation.saturation",        "Saturation"),
    "SaltPepperNoise":          ("ai_vision_tool.augmentation.noise_dropout",     "SaltPepperNoise"),
    "Shear":                    ("ai_vision_tool.augmentation.shear",             "Shear"),
    "Solarize":                 ("ai_vision_tool.augmentation.blur_artifact",     "Solarize"),
    "Superpixel":               ("ai_vision_tool.augmentation.blur_artifact",     "Superpixel"),
    "ToSepia":                  ("ai_vision_tool.augmentation.weather_light",     "ToSepia"),
    "Translate":                ("ai_vision_tool.augmentation.geometric_random",  "Translate"),
    "ZoomBlur":                 ("ai_vision_tool.augmentation.blur_artifact",     "ZoomBlur"),

    # enhancement (cv2-only core)
    "Denoiser":         ("ai_vision_tool.enhancement.denoiser",       "Denoiser"),
    "FrameEnhancer":    ("ai_vision_tool.enhancement.frame_enhancer", "FrameEnhancer"),
    "LowLightEnhancer": ("ai_vision_tool.enhancement.low_light",      "LowLightEnhancer"),

    # enhancement (DL-backed — require [onnx] or [torch])
    "Colorizer":        ("ai_vision_tool.enhancement.models.colorization",  "Colorizer"),
    "Deblurrer":        ("ai_vision_tool.enhancement.models.deblurring",    "Deblurrer"),
    "SuperResolution":  ("ai_vision_tool.enhancement.models.super_resolution", "SuperResolution"),

    # capture
    "BurstPictureTaker": ("ai_vision_tool.capture.burst_image_capture", "BurstPictureTaker"),
    "FrameGrabber":      ("ai_vision_tool.capture.frame_grabber",       "FrameGrabber"),
    "MotionDetector":    ("ai_vision_tool.capture.motion_detector",     "MotionDetector"),
    "PictureTaker":      ("ai_vision_tool.capture.image_capture",       "PictureTaker"),
    "ROICapture":        ("ai_vision_tool.capture.roi_capture",         "ROICapture"),
    "TimeLapseCapture":  ("ai_vision_tool.capture.time_lapse_capture",  "TimeLapseCapture"),
    "VideoTaker":        ("ai_vision_tool.capture.video_capture",       "VideoTaker"),

    # io
    "DatasetCollector":  ("ai_vision_tool.io.dataset_collector",  "DatasetCollector"),
    "DatasetExporter":   ("ai_vision_tool.io.dataset_exporter",   "DatasetExporter"),
    "ImageExporter":     ("ai_vision_tool.io.image_exporter",     "ImageExporter"),
    "ImageReader":       ("ai_vision_tool.io.image_io",           "ImageReader"),
    "ImageWriter":       ("ai_vision_tool.io.image_io",           "ImageWriter"),
    "VideoReader":       ("ai_vision_tool.io.video_io",           "VideoReader"),
    "VideoWriter":       ("ai_vision_tool.io.video_io",           "VideoWriter"),
    "CameraSource":      ("ai_vision_tool.io.camera_source",      "CameraSource"),

    # integrations — cloud
    "GCSSource": ("ai_vision_tool.integrations.cloud.gcs_source", "GCSSource"),
    "S3Source":  ("ai_vision_tool.integrations.cloud.s3_source",  "S3Source"),

    # integrations — labeling
    "AutoLabeller":          ("ai_vision_tool.integrations.labeling.auto_labeller",          "AutoLabeller"),
    "DarknetAutoLabeler":    ("ai_vision_tool.integrations.labeling.darknet_auto_labeler",   "DarknetAutoLabeler"),
    "TensorFlowAutoLabeler": ("ai_vision_tool.integrations.labeling.tensorflow_auto_labeler","TensorFlowAutoLabeler"),

    # models
    "ModelBenchmark":  ("ai_vision_tool.models.benchmark",          "ModelBenchmark"),
    "ModelDownloader": ("ai_vision_tool.models.downloader",          "ModelDownloader"),
    "ModelRegistry":   ("ai_vision_tool.models.registry",            "ModelRegistry"),
    "ONNXModel":       ("ai_vision_tool.models.backends.onnx_model", "ONNXModel"),
    "TFLiteModel":     ("ai_vision_tool.models.backends.tflite_model","TFLiteModel"),
    "TorchModel":      ("ai_vision_tool.models.backends.torch_model","TorchModel"),

    # detection
    "AnomalyDetector":   ("ai_vision_tool.detection.anomaly_detector",  "AnomalyDetector"),
    "FaceDetector":      ("ai_vision_tool.detection.face_detector",      "FaceDetector"),
    "KeypointDetector":  ("ai_vision_tool.detection.keypoint_detector",  "KeypointDetector"),
    "ObjectDetector":    ("ai_vision_tool.detection.object_detector",    "ObjectDetector"),
    "TextDetector":      ("ai_vision_tool.detection.text_detector",      "TextDetector"),

    # tracking
    "ByteTracker":      ("ai_vision_tool.tracking.byte_tracker",    "ByteTracker"),
    "DeepSORTTracker":  ("ai_vision_tool.tracking.deepsort_tracker","DeepSORTTracker"),
    "KalmanFilter":     ("ai_vision_tool.tracking.kalman_filter",   "KalmanFilter"),
    "ReIDExtractor":    ("ai_vision_tool.tracking.reid_extractor",  "ReIDExtractor"),
    "TrackManager":     ("ai_vision_tool.tracking.track_manager",   "TrackManager"),

    # segmentation
    "InstanceSegmenter":  ("ai_vision_tool.segmentation.instance_segmenter",  "InstanceSegmenter"),
    "MaskPostProcessor":  ("ai_vision_tool.segmentation.mask_post_processor",  "MaskPostProcessor"),
    "PanopticSegmenter":  ("ai_vision_tool.segmentation.panoptic_segmenter",   "PanopticSegmenter"),
    "SAMSegmenter":       ("ai_vision_tool.segmentation.sam_segmenter",        "SAMSegmenter"),
    "SemanticSegmenter":  ("ai_vision_tool.segmentation.semantic_segmenter",   "SemanticSegmenter"),

    # pipelines
    "AsyncComponent":    ("ai_vision_tool.pipelines.async_pipeline",    "AsyncComponent"),
    "AsyncPipeline":     ("ai_vision_tool.pipelines.async_pipeline",    "AsyncPipeline"),
    "FanOutPipeline":    ("ai_vision_tool.pipelines.parallel_pipeline", "FanOutPipeline"),
    "ParallelPipeline":  ("ai_vision_tool.pipelines.parallel_pipeline", "ParallelPipeline"),
    "PipelineSerializer":("ai_vision_tool.pipelines.serializer",        "PipelineSerializer"),
    "PrebuiltPipelines": ("ai_vision_tool.pipelines.prebuilt",          "PrebuiltPipelines"),

    # streaming (local primitives)
    "BufferedStream":      ("ai_vision_tool.streaming.buffered_stream", "BufferedStream"),
    "DirectoryStream":     ("ai_vision_tool.streaming.frame_stream",    "DirectoryStream"),
    "FrameStream":         ("ai_vision_tool.streaming.frame_stream",    "FrameStream"),
    "RTSPClient":          ("ai_vision_tool.streaming.rtsp_client",     "RTSPClient"),
    "RTSPServer":          ("ai_vision_tool.streaming.rtsp_client",     "RTSPServer"),
    "SlidingWindowBuffer": ("ai_vision_tool.streaming.buffered_stream", "SlidingWindowBuffer"),

    # integrations — streaming (require [websocket] or [kafka])
    "KafkaSink":       ("ai_vision_tool.integrations.streaming.kafka_io",       "KafkaSink"),
    "KafkaSource":     ("ai_vision_tool.integrations.streaming.kafka_io",       "KafkaSource"),
    "WebSocketSink":   ("ai_vision_tool.integrations.streaming.websocket_sink", "WebSocketSink"),
    "WebSocketSource": ("ai_vision_tool.integrations.streaming.websocket_sink", "WebSocketSource"),

    # visualization
    "BBoxRenderer":            ("ai_vision_tool.visualization.bbox_renderer",            "BBoxRenderer"),
    "DashboardSink":           ("ai_vision_tool.visualization.dashboard_view",           "DashboardSink"),
    "FrameAnnotator":          ("ai_vision_tool.visualization.frame_annotator",          "FrameAnnotator"),
    "FrameViewer":             ("ai_vision_tool.visualization.frame_viewer",             "FrameViewer"),
    "HeatmapRenderer":         ("ai_vision_tool.visualization.heatmap_renderer",         "HeatmapRenderer"),
    "VideoAnnotationExporter": ("ai_vision_tool.visualization.video_annotation_exporter","VideoAnnotationExporter"),

    # utils
    "ColorPalette":          ("ai_vision_tool.utils.color_palette",  "ColorPalette"),
    "DrawUtils":             ("ai_vision_tool.utils.draw_utils",      "DrawUtils"),
    "FrameSampler":          ("ai_vision_tool.utils.frame_sampler",   "FrameSampler"),
    "ImageHash":             ("ai_vision_tool.utils.image_hash",      "ImageHash"),
    "MetricsLogger":         ("ai_vision_tool.utils.metrics_logger",  "MetricsLogger"),
    "MetricsLoggerComponent":("ai_vision_tool.utils.metrics_logger",  "MetricsLoggerComponent"),

    # core data types and utilities
    "BatchProcessor":    ("ai_vision_tool.core.batch_processor", "BatchProcessor"),
    "BBox":              ("ai_vision_tool.core.data_types",       "BBox"),
    "Detection":         ("ai_vision_tool.core.data_types",       "Detection"),
    "Device":            ("ai_vision_tool.core.device",           "Device"),
    "GPUMemoryTracker":  ("ai_vision_tool.core.memory_manager",   "GPUMemoryTracker"),
    "Keypoint":          ("ai_vision_tool.core.data_types",       "Keypoint"),
    "Mask":              ("ai_vision_tool.core.data_types",       "Mask"),
    "MemoryManager":     ("ai_vision_tool.core.memory_manager",   "MemoryManager"),
    "Pose":              ("ai_vision_tool.core.data_types",       "Pose"),
    "RateLimiter":       ("ai_vision_tool.core.scheduler",        "RateLimiter"),
    "Scheduler":         ("ai_vision_tool.core.scheduler",        "Scheduler"),
    "SegmentationResult":("ai_vision_tool.core.data_types",       "SegmentationResult"),
    "Track":             ("ai_vision_tool.core.data_types",       "Track"),

    # config
    "ComponentRegistry": ("ai_vision_tool.config.registry",       "ComponentRegistry"),
    "EnvConfig":         ("ai_vision_tool.config.env_config",     "EnvConfig"),
    "JSONConfig":        ("ai_vision_tool.config.json_config",    "JSONConfig"),
    "ProfileLoader":     ("ai_vision_tool.config.profile_loader", "ProfileLoader"),
    "YAMLConfig":        ("ai_vision_tool.config.yaml_config",    "YAMLConfig"),
}

__all__ = sorted(_EXPORTS.keys()) + ["__version__"]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ai_vision_tool' has no attribute {name!r}")
    module_path, attr = _EXPORTS[name]
    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value  # cache — import once, reuse for process lifetime
    return value
