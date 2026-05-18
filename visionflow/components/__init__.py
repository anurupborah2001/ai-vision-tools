"""Component exports for AI Vision Flow."""

from importlib import import_module

_EXPORTS = {
    "AIVisionComponent": ("visionflow.components.base", "AIVisionComponent"),
    "AutoAdjustContrast": (
        "visionflow.components.preprocessing.auto_adjust_contrast",
        "AutoAdjustContrast",
    ),
    "AdaptiveThreshold": ("visionflow.components.preprocessing.intensity", "AdaptiveThreshold"),
    "AspectRatioFilter": ("visionflow.components.preprocessing.quality", "AspectRatioFilter"),
    "AutoCrop": ("visionflow.components.preprocessing.geometry", "AutoCrop"),
    "AutoOrient": ("visionflow.components.preprocessing.auto_orient", "AutoOrient"),
    "AffineTransform": (
        "visionflow.components.augmentations.geometric_random",
        "AffineTransform",
    ),
    "AugmentationSharpen": (
        "visionflow.components.augmentations.blur_artifact",
        "Sharpen",
    ),
    "BGRToRGB": ("visionflow.components.preprocessing.intensity", "BGRToRGB"),
    "Blur": ("visionflow.components.augmentations.blur", "Blur"),
    "BlurDetection": ("visionflow.components.preprocessing.quality", "BlurDetection"),
    "BoundingBoxClamp": (
        "visionflow.components.preprocessing.geometry",
        "BoundingBoxClamp",
    ),
    "BoundingBoxNormalize": (
        "visionflow.components.preprocessing.geometry",
        "BoundingBoxNormalize",
    ),
    "Brightness": ("visionflow.components.augmentations.brightness", "Brightness"),
    "BrightnessCheck": ("visionflow.components.preprocessing.quality", "BrightnessCheck"),
    "FrameEnhancer": ("visionflow.components.frame_enhancer", "FrameEnhancer"),
    "FrameResizer": ("visionflow.components.frame_resizer", "FrameResizer"),
    "FrameGrabber": ("visionflow.components.frame_grabber", "FrameGrabber"),
    "CameraGain": ("visionflow.components.augmentations.camera_gain", "CameraGain"),
    "ChannelShuffle": (
        "visionflow.components.augmentations.weather_light",
        "ChannelShuffle",
    ),
    "CenterCrop": ("visionflow.components.preprocessing.geometry", "CenterCrop"),
    "CLAHE": ("visionflow.components.preprocessing.intensity", "CLAHE"),
    "CoarseDropout": (
        "visionflow.components.augmentations.noise_dropout",
        "CoarseDropout",
    ),
    "ColorJitter": (
        "visionflow.components.augmentations.weather_light",
        "ColorJitter",
    ),
    "CompressionArtifacts": (
        "visionflow.components.augmentations.blur_artifact",
        "CompressionArtifacts",
    ),
    "ContourExtraction": (
        "visionflow.components.preprocessing.intensity",
        "ContourExtraction",
    ),
    "ConvertColorSpace": (
        "visionflow.components.preprocessing.intensity",
        "ConvertColorSpace",
    ),
    "CorruptImageCheck": (
        "visionflow.components.preprocessing.quality",
        "CorruptImageCheck",
    ),
    "CopyPaste": ("visionflow.components.augmentations.composite", "CopyPaste"),
    "Crop": ("visionflow.components.augmentations.crop", "Crop"),
    "CutMix": ("visionflow.components.augmentations.composite", "CutMix"),
    "Cutout": ("visionflow.components.augmentations.cutout", "Cutout"),
    "Deblur": ("visionflow.components.preprocessing.intensity", "Deblur"),
    "DefocusBlur": (
        "visionflow.components.augmentations.blur_artifact",
        "DefocusBlur",
    ),
    "Denoise": ("visionflow.components.preprocessing.intensity", "Denoise"),
    "Deskew": ("visionflow.components.preprocessing.geometry", "Deskew"),
    "Downscale": ("visionflow.components.augmentations.blur_artifact", "Downscale"),
    "DuplicateImageCheck": (
        "visionflow.components.preprocessing.quality",
        "DuplicateImageCheck",
    ),
    "ElasticTransform": (
        "visionflow.components.augmentations.geometric_random",
        "ElasticTransform",
    ),
    "EdgeDetection": ("visionflow.components.preprocessing.intensity", "EdgeDetection"),
    "Emboss": ("visionflow.components.augmentations.blur_artifact", "Emboss"),
    "Equalize": ("visionflow.components.augmentations.blur_artifact", "Equalize"),
    "FaceAlign": ("visionflow.components.preprocessing.geometry", "FaceAlign"),
    "GammaCorrection": (
        "visionflow.components.preprocessing.intensity",
        "GammaCorrection",
    ),
    "GaussianBlur": (
        "visionflow.components.augmentations.blur_artifact",
        "GaussianBlur",
    ),
    "GlassBlur": ("visionflow.components.augmentations.blur_artifact", "GlassBlur"),
    "HistogramEqualization": (
        "visionflow.components.preprocessing.intensity",
        "HistogramEqualization",
    ),
    "GridDistortion": (
        "visionflow.components.augmentations.geometric_random",
        "GridDistortion",
    ),
    "GridDropout": (
        "visionflow.components.augmentations.noise_dropout",
        "GridDropout",
    ),
    "HSVShift": ("visionflow.components.augmentations.weather_light", "HSVShift"),
    "ImageExporter": ("visionflow.components.image_exporter", "ImageExporter"),
    "ImageQualityCheck": (
        "visionflow.components.preprocessing.quality",
        "ImageQualityCheck",
    ),
    "ISONoise": ("visionflow.components.augmentations.noise_dropout", "ISONoise"),
    "InvertImage": (
        "visionflow.components.augmentations.weather_light",
        "InvertImage",
    ),
    "JPEGCompression": (
        "visionflow.components.augmentations.blur_artifact",
        "JPEGCompression",
    ),
    "LetterboxResize": (
        "visionflow.components.preprocessing.geometry",
        "LetterboxResize",
    ),
    "MaskResize": ("visionflow.components.preprocessing.geometry", "MaskResize"),
    "MaskDropout": ("visionflow.components.augmentations.noise_dropout", "MaskDropout"),
    "MaxSizeFilter": ("visionflow.components.preprocessing.quality", "MaxSizeFilter"),
    "MedianBlur": ("visionflow.components.augmentations.blur_artifact", "MedianBlur"),
    "MinSizeFilter": ("visionflow.components.preprocessing.quality", "MinSizeFilter"),
    "MixUp": ("visionflow.components.augmentations.composite", "MixUp"),
    "Mosaic9": ("visionflow.components.augmentations.composite", "Mosaic9"),
    "MultiplicativeNoise": (
        "visionflow.components.augmentations.noise_dropout",
        "MultiplicativeNoise",
    ),
    "VideoTaker": ("visionflow.components.video_taker", "VideoTaker"),
    "TensorFlowAutoLabeler": ("visionflow.components.tensorflow_auto_labeler", "TensorFlowAutoLabeler"),
    "DarknetAutoLabeler": ("visionflow.components.darknet_auto_labeler", "DarknetAutoLabeler"),
    "AutoLabeller": ("visionflow.components.auto_labeller", "AutoLabeller"),
    "Exposure": ("visionflow.components.augmentations.exposure", "Exposure"),
    "Flip": ("visionflow.components.augmentations.flip", "Flip"),
    "Greyscale": ("visionflow.components.augmentations.greyscale", "Greyscale"),
    "Hue": ("visionflow.components.augmentations.hue", "Hue"),
    "Mosaic": ("visionflow.components.augmentations.mosaic", "Mosaic"),
    "MotionBlur": ("visionflow.components.augmentations.motion_blur", "MotionBlur"),
    "PictureTaker": ("visionflow.components.picture_taker", "PictureTaker"),
    "BurstPictureTaker": ("visionflow.components.burst_picture_taker", "BurstPictureTaker"),
    "Normalize": ("visionflow.components.preprocessing.intensity", "Normalize"),
    "Noise": ("visionflow.components.augmentations.noise", "Noise"),
    "ObjectCrop": ("visionflow.components.preprocessing.geometry", "ObjectCrop"),
    "ObjectPaste": ("visionflow.components.augmentations.composite", "ObjectPaste"),
    "OpticalDistortion": (
        "visionflow.components.augmentations.geometric_random",
        "OpticalDistortion",
    ),
    "PadToSquare": ("visionflow.components.preprocessing.geometry", "PadToSquare"),
    "PerspectiveTransform": (
        "visionflow.components.augmentations.geometric_random",
        "PerspectiveTransform",
    ),
    "PixelDropout": ("visionflow.components.augmentations.noise_dropout", "PixelDropout"),
    "Posterize": ("visionflow.components.augmentations.blur_artifact", "Posterize"),
    "PerspectiveCorrection": (
        "visionflow.components.preprocessing.geometry",
        "PerspectiveCorrection",
    ),
    "RandomBrightnessContrast": (
        "visionflow.components.augmentations.weather_light",
        "RandomBrightnessContrast",
    ),
    "RandomCrop": (
        "visionflow.components.augmentations.geometric_random",
        "RandomCrop",
    ),
    "RandomErasing": (
        "visionflow.components.augmentations.noise_dropout",
        "RandomErasing",
    ),
    "RandomFog": ("visionflow.components.augmentations.weather_light", "RandomFog"),
    "RandomGamma": (
        "visionflow.components.augmentations.weather_light",
        "RandomGamma",
    ),
    "RandomOcclusion": (
        "visionflow.components.augmentations.composite",
        "RandomOcclusion",
    ),
    "RandomPadding": (
        "visionflow.components.augmentations.geometric_random",
        "RandomPadding",
    ),
    "RandomRain": (
        "visionflow.components.augmentations.weather_light",
        "RandomRain",
    ),
    "RandomResize": (
        "visionflow.components.augmentations.geometric_random",
        "RandomResize",
    ),
    "RandomResizedCrop": (
        "visionflow.components.augmentations.geometric_random",
        "RandomResizedCrop",
    ),
    "RandomScale": (
        "visionflow.components.augmentations.geometric_random",
        "RandomScale",
    ),
    "RandomShadow": (
        "visionflow.components.augmentations.weather_light",
        "RandomShadow",
    ),
    "RandomSnow": (
        "visionflow.components.augmentations.weather_light",
        "RandomSnow",
    ),
    "RandomSunFlare": (
        "visionflow.components.augmentations.weather_light",
        "RandomSunFlare",
    ),
    "RemoveBackground": (
        "visionflow.components.preprocessing.segmentation",
        "RemoveBackground",
    ),
    "RescalePixels": ("visionflow.components.preprocessing.intensity", "RescalePixels"),
    "Resize": ("visionflow.components.preprocessing.geometry", "Resize"),
    "RGBShift": ("visionflow.components.augmentations.weather_light", "RGBShift"),
    "RGBToBGR": ("visionflow.components.preprocessing.intensity", "RGBToBGR"),
    "Rotate90": ("visionflow.components.augmentations.rotate90", "Rotate90"),
    "Rotation": ("visionflow.components.augmentations.rotation", "Rotation"),
    "TimeLapseCapture": ("visionflow.components.time_lapse_capture", "TimeLapseCapture"),
    "ROICapture": ("visionflow.components.roi_capture", "ROICapture"),
    "Saturation": ("visionflow.components.augmentations.saturation", "Saturation"),
    "SaltPepperNoise": (
        "visionflow.components.augmentations.noise_dropout",
        "SaltPepperNoise",
    ),
    "Sharpen": ("visionflow.components.preprocessing.intensity", "Sharpen"),
    "Shear": ("visionflow.components.augmentations.shear", "Shear"),
    "Solarize": ("visionflow.components.augmentations.blur_artifact", "Solarize"),
    "Standardize": ("visionflow.components.preprocessing.intensity", "Standardize"),
    "Superpixel": ("visionflow.components.augmentations.blur_artifact", "Superpixel"),
    "Threshold": ("visionflow.components.preprocessing.intensity", "Threshold"),
    "ToSepia": ("visionflow.components.augmentations.weather_light", "ToSepia"),
    "Translate": ("visionflow.components.augmentations.geometric_random", "Translate"),
    "WhiteBalance": ("visionflow.components.preprocessing.intensity", "WhiteBalance"),
    "MotionDetector": ("visionflow.components.motion_detector", "MotionDetector"),
    "FrameAnnotator": ("visionflow.components.frame_annotator", "FrameAnnotator"),
    "DatasetCollector": ("visionflow.components.dataset_collector", "DatasetCollector"),
    "ZoomBlur": ("visionflow.components.augmentations.blur_artifact", "ZoomBlur"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'visionflow.components' has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
