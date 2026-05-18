"""Component exports for AI Vision Flow."""

from importlib import import_module

_EXPORTS = {
    "AIVisionComponent": ("ai_vision_tool.components.base", "AIVisionComponent"),
    "AutoAdjustContrast": (
        "ai_vision_tool.components.preprocessing.auto_adjust_contrast",
        "AutoAdjustContrast",
    ),
    "AdaptiveThreshold": ("ai_vision_tool.components.preprocessing.intensity", "AdaptiveThreshold"),
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
    "ZoomBlur": ("ai_vision_tool.components.augmentations.blur_artifact", "ZoomBlur"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'ai_vision_tool.components' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
