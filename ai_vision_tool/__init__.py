"""Top-level package for AI Vision Tool."""

from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
    "AIVisionComponent": ("ai_vision_tool.core.base", "AIVisionComponent"),
    "AutoAdjustContrast": (
        "ai_vision_tool.components.preprocessing.auto_adjust_contrast",
        "AutoAdjustContrast",
    ),
    "AdaptiveThreshold": ("ai_vision_tool.components.preprocessing.intensity", "AdaptiveThreshold"),
    "app": ("ai_vision_tool.api", "app"),
    "AspectRatioFilter": ("ai_vision_tool.components.preprocessing.quality", "AspectRatioFilter"),
    "AutoCrop": ("ai_vision_tool.components.preprocessing.geometry", "AutoCrop"),
    "AutoOrient": ("ai_vision_tool.components.preprocessing.auto_orient", "AutoOrient"),
    "AffineTransform": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "AffineTransform",
    ),
    "AugmentationSharpen": (
        "ai_vision_tool.components.augmentations.blur_artifact",
        "Sharpen",
    ),
    "BGRToRGB": ("ai_vision_tool.components.preprocessing.intensity", "BGRToRGB"),
    "Blur": ("ai_vision_tool.components.augmentations.blur", "Blur"),
    "BlurDetection": ("ai_vision_tool.components.preprocessing.quality", "BlurDetection"),
    "BoundingBoxClamp": (
        "ai_vision_tool.components.preprocessing.geometry",
        "BoundingBoxClamp",
    ),
    "BoundingBoxNormalize": (
        "ai_vision_tool.components.preprocessing.geometry",
        "BoundingBoxNormalize",
    ),
    "Brightness": ("ai_vision_tool.components.augmentations.brightness", "Brightness"),
    "BrightnessCheck": ("ai_vision_tool.components.preprocessing.quality", "BrightnessCheck"),
    "FrameEnhancer": ("ai_vision_tool.components.frame_enhancer", "FrameEnhancer"),
    "FrameResizer": ("ai_vision_tool.components.frame_resizer", "FrameResizer"),
    "FrameGrabber": ("ai_vision_tool.components.frame_grabber", "FrameGrabber"),
    "CameraGain": ("ai_vision_tool.components.augmentations.camera_gain", "CameraGain"),
    "ChannelShuffle": (
        "ai_vision_tool.components.augmentations.weather_light",
        "ChannelShuffle",
    ),
    "CenterCrop": ("ai_vision_tool.components.preprocessing.geometry", "CenterCrop"),
    "CLAHE": ("ai_vision_tool.components.preprocessing.intensity", "CLAHE"),
    "CoarseDropout": (
        "ai_vision_tool.components.augmentations.noise_dropout",
        "CoarseDropout",
    ),
    "ColorJitter": (
        "ai_vision_tool.components.augmentations.weather_light",
        "ColorJitter",
    ),
    "CompressionArtifacts": (
        "ai_vision_tool.components.augmentations.blur_artifact",
        "CompressionArtifacts",
    ),
    "ContourExtraction": (
        "ai_vision_tool.components.preprocessing.intensity",
        "ContourExtraction",
    ),
    "ConvertColorSpace": (
        "ai_vision_tool.components.preprocessing.intensity",
        "ConvertColorSpace",
    ),
    "CorruptImageCheck": (
        "ai_vision_tool.components.preprocessing.quality",
        "CorruptImageCheck",
    ),
    "CopyPaste": ("ai_vision_tool.components.augmentations.composite", "CopyPaste"),
    "Crop": ("ai_vision_tool.components.augmentations.crop", "Crop"),
    "CutMix": ("ai_vision_tool.components.augmentations.composite", "CutMix"),
    "Cutout": ("ai_vision_tool.components.augmentations.cutout", "Cutout"),
    "Deblur": ("ai_vision_tool.components.preprocessing.intensity", "Deblur"),
    "DefocusBlur": (
        "ai_vision_tool.components.augmentations.blur_artifact",
        "DefocusBlur",
    ),
    "Denoise": ("ai_vision_tool.components.preprocessing.intensity", "Denoise"),
    "Deskew": ("ai_vision_tool.components.preprocessing.geometry", "Deskew"),
    "Downscale": ("ai_vision_tool.components.augmentations.blur_artifact", "Downscale"),
    "DuplicateImageCheck": (
        "ai_vision_tool.components.preprocessing.quality",
        "DuplicateImageCheck",
    ),
    "ElasticTransform": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "ElasticTransform",
    ),
    "EdgeDetection": ("ai_vision_tool.components.preprocessing.intensity", "EdgeDetection"),
    "Emboss": ("ai_vision_tool.components.augmentations.blur_artifact", "Emboss"),
    "Equalize": ("ai_vision_tool.components.augmentations.blur_artifact", "Equalize"),
    "FaceAlign": ("ai_vision_tool.components.preprocessing.geometry", "FaceAlign"),
    "GammaCorrection": (
        "ai_vision_tool.components.preprocessing.intensity",
        "GammaCorrection",
    ),
    "GaussianBlur": (
        "ai_vision_tool.components.augmentations.blur_artifact",
        "GaussianBlur",
    ),
    "GlassBlur": ("ai_vision_tool.components.augmentations.blur_artifact", "GlassBlur"),
    "HistogramEqualization": (
        "ai_vision_tool.components.preprocessing.intensity",
        "HistogramEqualization",
    ),
    "GridDistortion": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "GridDistortion",
    ),
    "GridDropout": (
        "ai_vision_tool.components.augmentations.noise_dropout",
        "GridDropout",
    ),
    "HSVShift": ("ai_vision_tool.components.augmentations.weather_light", "HSVShift"),
    "ImageExporter": ("ai_vision_tool.components.image_exporter", "ImageExporter"),
    "ImageQualityCheck": (
        "ai_vision_tool.components.preprocessing.quality",
        "ImageQualityCheck",
    ),
    "ISONoise": ("ai_vision_tool.components.augmentations.noise_dropout", "ISONoise"),
    "InvertImage": (
        "ai_vision_tool.components.augmentations.weather_light",
        "InvertImage",
    ),
    "JPEGCompression": (
        "ai_vision_tool.components.augmentations.blur_artifact",
        "JPEGCompression",
    ),
    "LetterboxResize": (
        "ai_vision_tool.components.preprocessing.geometry",
        "LetterboxResize",
    ),
    "MaskResize": ("ai_vision_tool.components.preprocessing.geometry", "MaskResize"),
    "MaskDropout": ("ai_vision_tool.components.augmentations.noise_dropout", "MaskDropout"),
    "MaxSizeFilter": ("ai_vision_tool.components.preprocessing.quality", "MaxSizeFilter"),
    "MedianBlur": ("ai_vision_tool.components.augmentations.blur_artifact", "MedianBlur"),
    "MinSizeFilter": ("ai_vision_tool.components.preprocessing.quality", "MinSizeFilter"),
    "MixUp": ("ai_vision_tool.components.augmentations.composite", "MixUp"),
    "Mosaic9": ("ai_vision_tool.components.augmentations.composite", "Mosaic9"),
    "MultiplicativeNoise": (
        "ai_vision_tool.components.augmentations.noise_dropout",
        "MultiplicativeNoise",
    ),
    "VideoTaker": ("ai_vision_tool.components.video_taker", "VideoTaker"),
    "TensorFlowAutoLabeler": ("ai_vision_tool.components.tensorflow_auto_labeler", "TensorFlowAutoLabeler"),
    "DarknetAutoLabeler": ("ai_vision_tool.components.darknet_auto_labeler", "DarknetAutoLabeler"),
    "AutoLabeller": ("ai_vision_tool.components.auto_labeller", "AutoLabeller"),
    "Exposure": ("ai_vision_tool.components.augmentations.exposure", "Exposure"),
    "Flip": ("ai_vision_tool.components.augmentations.flip", "Flip"),
    "Greyscale": ("ai_vision_tool.components.augmentations.greyscale", "Greyscale"),
    "Hue": ("ai_vision_tool.components.augmentations.hue", "Hue"),
    "Mosaic": ("ai_vision_tool.components.augmentations.mosaic", "Mosaic"),
    "MotionBlur": ("ai_vision_tool.components.augmentations.motion_blur", "MotionBlur"),
    "PictureTaker": ("ai_vision_tool.components.picture_taker", "PictureTaker"),
    "BurstPictureTaker": ("ai_vision_tool.components.burst_picture_taker", "BurstPictureTaker"),
    "Normalize": ("ai_vision_tool.components.preprocessing.intensity", "Normalize"),
    "Noise": ("ai_vision_tool.components.augmentations.noise", "Noise"),
    "ObjectCrop": ("ai_vision_tool.components.preprocessing.geometry", "ObjectCrop"),
    "ObjectPaste": ("ai_vision_tool.components.augmentations.composite", "ObjectPaste"),
    "OpticalDistortion": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "OpticalDistortion",
    ),
    "PadToSquare": ("ai_vision_tool.components.preprocessing.geometry", "PadToSquare"),
    "PerspectiveTransform": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "PerspectiveTransform",
    ),
    "PixelDropout": ("ai_vision_tool.components.augmentations.noise_dropout", "PixelDropout"),
    "Posterize": ("ai_vision_tool.components.augmentations.blur_artifact", "Posterize"),
    "PerspectiveCorrection": (
        "ai_vision_tool.components.preprocessing.geometry",
        "PerspectiveCorrection",
    ),
    "RandomBrightnessContrast": (
        "ai_vision_tool.components.augmentations.weather_light",
        "RandomBrightnessContrast",
    ),
    "RandomCrop": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "RandomCrop",
    ),
    "RandomErasing": (
        "ai_vision_tool.components.augmentations.noise_dropout",
        "RandomErasing",
    ),
    "RandomFog": ("ai_vision_tool.components.augmentations.weather_light", "RandomFog"),
    "RandomGamma": (
        "ai_vision_tool.components.augmentations.weather_light",
        "RandomGamma",
    ),
    "RandomOcclusion": (
        "ai_vision_tool.components.augmentations.composite",
        "RandomOcclusion",
    ),
    "RandomPadding": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "RandomPadding",
    ),
    "RandomRain": (
        "ai_vision_tool.components.augmentations.weather_light",
        "RandomRain",
    ),
    "RandomResize": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "RandomResize",
    ),
    "RandomResizedCrop": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "RandomResizedCrop",
    ),
    "RandomScale": (
        "ai_vision_tool.components.augmentations.geometric_random",
        "RandomScale",
    ),
    "RandomShadow": (
        "ai_vision_tool.components.augmentations.weather_light",
        "RandomShadow",
    ),
    "RandomSnow": (
        "ai_vision_tool.components.augmentations.weather_light",
        "RandomSnow",
    ),
    "RandomSunFlare": (
        "ai_vision_tool.components.augmentations.weather_light",
        "RandomSunFlare",
    ),
    "RemoveBackground": (
        "ai_vision_tool.components.preprocessing.segmentation",
        "RemoveBackground",
    ),
    "RescalePixels": ("ai_vision_tool.components.preprocessing.intensity", "RescalePixels"),
    "Resize": ("ai_vision_tool.components.preprocessing.geometry", "Resize"),
    "RGBShift": ("ai_vision_tool.components.augmentations.weather_light", "RGBShift"),
    "RGBToBGR": ("ai_vision_tool.components.preprocessing.intensity", "RGBToBGR"),
    "Rotate90": ("ai_vision_tool.components.augmentations.rotate90", "Rotate90"),
    "Rotation": ("ai_vision_tool.components.augmentations.rotation", "Rotation"),
    "TimeLapseCapture": ("ai_vision_tool.components.time_lapse_capture", "TimeLapseCapture"),
    "ROICapture": ("ai_vision_tool.components.roi_capture", "ROICapture"),
    "Saturation": ("ai_vision_tool.components.augmentations.saturation", "Saturation"),
    "SaltPepperNoise": (
        "ai_vision_tool.components.augmentations.noise_dropout",
        "SaltPepperNoise",
    ),
    "Sharpen": ("ai_vision_tool.components.preprocessing.intensity", "Sharpen"),
    "Shear": ("ai_vision_tool.components.augmentations.shear", "Shear"),
    "Solarize": ("ai_vision_tool.components.augmentations.blur_artifact", "Solarize"),
    "Standardize": ("ai_vision_tool.components.preprocessing.intensity", "Standardize"),
    "Superpixel": ("ai_vision_tool.components.augmentations.blur_artifact", "Superpixel"),
    "Threshold": ("ai_vision_tool.components.preprocessing.intensity", "Threshold"),
    "ToSepia": ("ai_vision_tool.components.augmentations.weather_light", "ToSepia"),
    "Translate": ("ai_vision_tool.components.augmentations.geometric_random", "Translate"),
    "WhiteBalance": ("ai_vision_tool.components.preprocessing.intensity", "WhiteBalance"),
    "MotionDetector": ("ai_vision_tool.components.motion_detector", "MotionDetector"),
    "FrameAnnotator": ("ai_vision_tool.components.frame_annotator", "FrameAnnotator"),
    "DatasetCollector": ("ai_vision_tool.components.dataset_collector", "DatasetCollector"),
    "create_app": ("ai_vision_tool.api", "create_app"),
    "ZoomBlur": ("ai_vision_tool.components.augmentations.blur_artifact", "ZoomBlur"),
    # utils
    "ColorPalette": ("ai_vision_tool.utils.color_palette", "ColorPalette"),
    "MetricsLogger": ("ai_vision_tool.utils.metrics_logger", "MetricsLogger"),
    "MetricsLoggerComponent": ("ai_vision_tool.utils.metrics_logger", "MetricsLoggerComponent"),
    "FrameSampler": ("ai_vision_tool.utils.frame_sampler", "FrameSampler"),
    "ImageHash": ("ai_vision_tool.utils.image_hash", "ImageHash"),
    "DrawUtils": ("ai_vision_tool.utils.draw_utils", "DrawUtils"),
    # core
    "Device": ("ai_vision_tool.core.device", "Device"),
    "BBox": ("ai_vision_tool.core.data_types", "BBox"),
    "Detection": ("ai_vision_tool.core.data_types", "Detection"),
    "Keypoint": ("ai_vision_tool.core.data_types", "Keypoint"),
    "Pose": ("ai_vision_tool.core.data_types", "Pose"),
    "Mask": ("ai_vision_tool.core.data_types", "Mask"),
    "SegmentationResult": ("ai_vision_tool.core.data_types", "SegmentationResult"),
    "Track": ("ai_vision_tool.core.data_types", "Track"),
    "BatchProcessor": ("ai_vision_tool.core.batch_processor", "BatchProcessor"),
    "Scheduler": ("ai_vision_tool.core.scheduler", "Scheduler"),
    "RateLimiter": ("ai_vision_tool.core.scheduler", "RateLimiter"),
    "MemoryManager": ("ai_vision_tool.core.memory_manager", "MemoryManager"),
    "GPUMemoryTracker": ("ai_vision_tool.core.memory_manager", "GPUMemoryTracker"),
    # config
    "YAMLConfig": ("ai_vision_tool.config.yaml_config", "YAMLConfig"),
    "JSONConfig": ("ai_vision_tool.config.json_config", "JSONConfig"),
    "ComponentRegistry": ("ai_vision_tool.config.registry", "ComponentRegistry"),
    "ProfileLoader": ("ai_vision_tool.config.profile_loader", "ProfileLoader"),
    "EnvConfig": ("ai_vision_tool.config.env_config", "EnvConfig"),
    # io
    "ImageReader": ("ai_vision_tool.io.image_io", "ImageReader"),
    "ImageWriter": ("ai_vision_tool.io.image_io", "ImageWriter"),
    "VideoReader": ("ai_vision_tool.io.video_io", "VideoReader"),
    "VideoWriter": ("ai_vision_tool.io.video_io", "VideoWriter"),
    "CameraSource": ("ai_vision_tool.io.camera_source", "CameraSource"),
    "S3Source": ("ai_vision_tool.io.cloud_source", "S3Source"),
    "GCSSource": ("ai_vision_tool.io.cloud_source", "GCSSource"),
    "DatasetExporter": ("ai_vision_tool.io.dataset_exporter", "DatasetExporter"),
    # models
    "ModelRegistry": ("ai_vision_tool.models.registry", "ModelRegistry"),
    "ONNXModel": ("ai_vision_tool.models.onnx_model", "ONNXModel"),
    "TorchModel": ("ai_vision_tool.models.torch_model", "TorchModel"),
    "TFLiteModel": ("ai_vision_tool.models.tflite_model", "TFLiteModel"),
    "ModelDownloader": ("ai_vision_tool.models.downloader", "ModelDownloader"),
    "ModelBenchmark": ("ai_vision_tool.models.benchmark", "ModelBenchmark"),
    # detection
    "ObjectDetector": ("ai_vision_tool.detection.object_detector", "ObjectDetector"),
    "FaceDetector": ("ai_vision_tool.detection.face_detector", "FaceDetector"),
    "KeypointDetector": ("ai_vision_tool.detection.keypoint_detector", "KeypointDetector"),
    "TextDetector": ("ai_vision_tool.detection.text_detector", "TextDetector"),
    "AnomalyDetector": ("ai_vision_tool.detection.anomaly_detector", "AnomalyDetector"),
    # tracking
    "KalmanFilter": ("ai_vision_tool.tracking.kalman_filter", "KalmanFilter"),
    "TrackManager": ("ai_vision_tool.tracking.track_manager", "TrackManager"),
    "ByteTracker": ("ai_vision_tool.tracking.byte_tracker", "ByteTracker"),
    "DeepSORTTracker": ("ai_vision_tool.tracking.deepsort_tracker", "DeepSORTTracker"),
    "ReIDExtractor": ("ai_vision_tool.tracking.reid_extractor", "ReIDExtractor"),
    # segmentation
    "SemanticSegmenter": ("ai_vision_tool.segmentation.semantic_segmenter", "SemanticSegmenter"),
    "InstanceSegmenter": ("ai_vision_tool.segmentation.instance_segmenter", "InstanceSegmenter"),
    "PanopticSegmenter": ("ai_vision_tool.segmentation.panoptic_segmenter", "PanopticSegmenter"),
    "SAMSegmenter": ("ai_vision_tool.segmentation.sam_segmenter", "SAMSegmenter"),
    "MaskPostProcessor": ("ai_vision_tool.segmentation.mask_post_processor", "MaskPostProcessor"),
    # enhancement
    "SuperResolution": ("ai_vision_tool.enhancement.super_resolution", "SuperResolution"),
    "Denoiser": ("ai_vision_tool.enhancement.denoiser", "Denoiser"),
    "Deblurrer": ("ai_vision_tool.enhancement.deblurrer", "Deblurrer"),
    "LowLightEnhancer": ("ai_vision_tool.enhancement.low_light_enhancer", "LowLightEnhancer"),
    "Colorizer": ("ai_vision_tool.enhancement.colorizer", "Colorizer"),
    # pipelines (plural — prebuilt / orchestration)
    "PrebuiltPipelines": ("ai_vision_tool.pipelines.prebuilt", "PrebuiltPipelines"),
    "PipelineSerializer": ("ai_vision_tool.pipelines.serializer", "PipelineSerializer"),
    "AsyncPipeline": ("ai_vision_tool.pipelines.async_pipeline", "AsyncPipeline"),
    "AsyncComponent": ("ai_vision_tool.pipelines.async_pipeline", "AsyncComponent"),
    "ParallelPipeline": ("ai_vision_tool.pipelines.parallel_pipeline", "ParallelPipeline"),
    "FanOutPipeline": ("ai_vision_tool.pipelines.parallel_pipeline", "FanOutPipeline"),
    # streaming
    "FrameStream": ("ai_vision_tool.streaming.frame_stream", "FrameStream"),
    "DirectoryStream": ("ai_vision_tool.streaming.frame_stream", "DirectoryStream"),
    "RTSPClient": ("ai_vision_tool.streaming.rtsp_client", "RTSPClient"),
    "RTSPServer": ("ai_vision_tool.streaming.rtsp_client", "RTSPServer"),
    "WebSocketSink": ("ai_vision_tool.streaming.websocket_sink", "WebSocketSink"),
    "WebSocketSource": ("ai_vision_tool.streaming.websocket_sink", "WebSocketSource"),
    "KafkaSource": ("ai_vision_tool.streaming.kafka_io", "KafkaSource"),
    "KafkaSink": ("ai_vision_tool.streaming.kafka_io", "KafkaSink"),
    "BufferedStream": ("ai_vision_tool.streaming.buffered_stream", "BufferedStream"),
    "SlidingWindowBuffer": ("ai_vision_tool.streaming.buffered_stream", "SlidingWindowBuffer"),
    # visualization
    "FrameViewer": ("ai_vision_tool.visualization.frame_viewer", "FrameViewer"),
    "DashboardSink": ("ai_vision_tool.visualization.dashboard_sink", "DashboardSink"),
    "BBoxRenderer": ("ai_vision_tool.visualization.bbox_renderer", "BBoxRenderer"),
    "HeatmapRenderer": ("ai_vision_tool.visualization.heatmap_renderer", "HeatmapRenderer"),
    "VideoAnnotationExporter": ("ai_vision_tool.visualization.video_annotation_exporter", "VideoAnnotationExporter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ai_vision_tool' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
