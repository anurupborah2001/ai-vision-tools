"""Compatibility wrapper for running the packaged CLI from the repository root.

This entrypoint supports both the standard CLI flags and profile-driven
augmentations via ``--augmentation-config <path/to/profile.json>``.

Example discovery is also available directly from ``main.py``:

- ``python main.py --show-examples --example-category preprocessing``
- ``python main.py --show-examples --example-category augmentations``
- ``python main.py --show-examples --example-name AutoOrient``

Quick start:

- Run the webcam app:
  ``python main.py``
- Run the FastAPI server:
  ``python main.py --serve-api``
- Process a local image file through a component:
  ``python main.py --process-image-path --component-category preprocessing --component-name AutoOrient --image-path path/to/image.jpg --init-args-json '{"rotation": 90}'``
- Show all preprocessing examples:
  ``python main.py --show-examples --example-category preprocessing``
- Show all augmentation examples:
  ``python main.py --show-examples --example-category augmentations``
- Show core component examples:
  ``python main.py --show-examples --example-category components``
- Show template helper examples:
  ``python main.py --show-examples --example-category template``

How to call any specific method:

- Use:
  ``python main.py --show-examples --example-name <MethodName>``
- Example:
  ``python main.py --show-examples --example-name RandomResizedCrop``

How to process a local image file:

- The CLI loads the image from ``--image-path``, converts it to base64 internally,
  and forwards it through the same execution path used by the FastAPI service.
- Required flags:
  ``--process-image-path --component-category <preprocessing|augmentations|components> --component-name <ClassName> --image-path <file>``
- Optional flags:
  ``--init-args-json``, ``--config-json``, ``--payload-json``, and ``--save-output-image``

Preprocessing method lookups:

- ``python main.py --show-examples --example-name AutoOrient``
- ``python main.py --show-examples --example-name AutoAdjustContrast``
- ``python main.py --show-examples --example-name Resize``
- ``python main.py --show-examples --example-name LetterboxResize``
- ``python main.py --show-examples --example-name CenterCrop``
- ``python main.py --show-examples --example-name PadToSquare``
- ``python main.py --show-examples --example-name Normalize``
- ``python main.py --show-examples --example-name Standardize``
- ``python main.py --show-examples --example-name RescalePixels``
- ``python main.py --show-examples --example-name ConvertColorSpace``
- ``python main.py --show-examples --example-name BGRToRGB``
- ``python main.py --show-examples --example-name RGBToBGR``
- ``python main.py --show-examples --example-name CLAHE``
- ``python main.py --show-examples --example-name HistogramEqualization``
- ``python main.py --show-examples --example-name GammaCorrection``
- ``python main.py --show-examples --example-name WhiteBalance``
- ``python main.py --show-examples --example-name Denoise``
- ``python main.py --show-examples --example-name Sharpen``
- ``python main.py --show-examples --example-name Deblur``
- ``python main.py --show-examples --example-name RemoveBackground``
- ``python main.py --show-examples --example-name Threshold``
- ``python main.py --show-examples --example-name AdaptiveThreshold``
- ``python main.py --show-examples --example-name EdgeDetection``
- ``python main.py --show-examples --example-name ContourExtraction``
- ``python main.py --show-examples --example-name PerspectiveCorrection``
- ``python main.py --show-examples --example-name Deskew``
- ``python main.py --show-examples --example-name AutoCrop``
- ``python main.py --show-examples --example-name FaceAlign``
- ``python main.py --show-examples --example-name ObjectCrop``
- ``python main.py --show-examples --example-name BoundingBoxClamp``
- ``python main.py --show-examples --example-name BoundingBoxNormalize``
- ``python main.py --show-examples --example-name MaskResize``
- ``python main.py --show-examples --example-name ImageQualityCheck``
- ``python main.py --show-examples --example-name BlurDetection``
- ``python main.py --show-examples --example-name BrightnessCheck``
- ``python main.py --show-examples --example-name DuplicateImageCheck``
- ``python main.py --show-examples --example-name CorruptImageCheck``
- ``python main.py --show-examples --example-name AspectRatioFilter``
- ``python main.py --show-examples --example-name MinSizeFilter``
- ``python main.py --show-examples --example-name MaxSizeFilter``

Augmentation method lookups:

- ``python main.py --show-examples --example-name Flip``
- ``python main.py --show-examples --example-name Rotate90``
- ``python main.py --show-examples --example-name Crop``
- ``python main.py --show-examples --example-name Rotation``
- ``python main.py --show-examples --example-name Shear``
- ``python main.py --show-examples --example-name Greyscale``
- ``python main.py --show-examples --example-name Hue``
- ``python main.py --show-examples --example-name Saturation``
- ``python main.py --show-examples --example-name Brightness``
- ``python main.py --show-examples --example-name Exposure``
- ``python main.py --show-examples --example-name Blur``
- ``python main.py --show-examples --example-name Noise``
- ``python main.py --show-examples --example-name Cutout``
- ``python main.py --show-examples --example-name Mosaic``
- ``python main.py --show-examples --example-name MotionBlur``
- ``python main.py --show-examples --example-name CameraGain``
- ``python main.py --show-examples --example-name RandomResize``
- ``python main.py --show-examples --example-name RandomScale``
- ``python main.py --show-examples --example-name RandomCrop``
- ``python main.py --show-examples --example-name RandomResizedCrop``
- ``python main.py --show-examples --example-name RandomPadding``
- ``python main.py --show-examples --example-name Translate``
- ``python main.py --show-examples --example-name AffineTransform``
- ``python main.py --show-examples --example-name PerspectiveTransform``
- ``python main.py --show-examples --example-name ElasticTransform``
- ``python main.py --show-examples --example-name GridDistortion``
- ``python main.py --show-examples --example-name OpticalDistortion``
- ``python main.py --show-examples --example-name RandomShadow``
- ``python main.py --show-examples --example-name RandomSunFlare``
- ``python main.py --show-examples --example-name RandomFog``
- ``python main.py --show-examples --example-name RandomRain``
- ``python main.py --show-examples --example-name RandomSnow``
- ``python main.py --show-examples --example-name RandomGamma``
- ``python main.py --show-examples --example-name ColorJitter``
- ``python main.py --show-examples --example-name ChannelShuffle``
- ``python main.py --show-examples --example-name RGBShift``
- ``python main.py --show-examples --example-name Posterize``
- ``python main.py --show-examples --example-name Solarize``
- ``python main.py --show-examples --example-name Equalize``
- ``python main.py --show-examples --example-name Emboss``
- ``python main.py --show-examples --example-name GaussianBlur``
- ``python main.py --show-examples --example-name MedianBlur``
- ``python main.py --show-examples --example-name GlassBlur``
- ``python main.py --show-examples --example-name DefocusBlur``
- ``python main.py --show-examples --example-name ZoomBlur``
- ``python main.py --show-examples --example-name ISONoise``
- ``python main.py --show-examples --example-name MultiplicativeNoise``
- ``python main.py --show-examples --example-name SaltPepperNoise``
- ``python main.py --show-examples --example-name CoarseDropout``
- ``python main.py --show-examples --example-name GridDropout``
- ``python main.py --show-examples --example-name RandomErasing``
- ``python main.py --show-examples --example-name MixUp``
- ``python main.py --show-examples --example-name CutMix``
- ``python main.py --show-examples --example-name CopyPaste``
- ``python main.py --show-examples --example-name RandomOcclusion``
- ``python main.py --show-examples --example-name ObjectPaste``
- ``python main.py --show-examples --example-name BoundingBoxJitter``
- ``python main.py --show-examples --example-name MaskDropout``
- ``python main.py --show-examples --example-name Mosaic9``
- ``python main.py --show-examples --example-name HSVShift``
- ``python main.py --show-examples --example-name ToSepia``
- ``python main.py --show-examples --example-name InvertImage``
- ``python main.py --show-examples --example-name CompressionArtifacts``
- ``python main.py --show-examples --example-name JPEGCompression``
- ``python main.py --show-examples --example-name Downscale``
- ``python main.py --show-examples --example-name Superpixel``
- ``python main.py --show-examples --example-name PixelDropout``
- ``python main.py --show-examples --example-name RandomBrightnessContrast``

Core component and template lookups:

- ``python main.py --show-examples --example-name FrameEnhancer``
- ``python main.py --show-examples --example-name FrameResizer``
- ``python main.py --show-examples --example-name MotionDetector``
- ``python main.py --show-examples --example-name FrameAnnotator``
- ``python main.py --show-examples --example-name DatasetCollector``
- ``python main.py --show-examples --example-name TimeLapseCapture``
- ``python main.py --show-examples --example-name PictureTaker``
- ``python main.py --show-examples --example-name BurstPictureTaker``
- ``python main.py --show-examples --example-name ROICapture``
- ``python main.py --show-examples --example-name ImageExporter``
- ``python main.py --show-examples --example-name FrameGrabber``
- ``python main.py --show-examples --example-name VideoTaker``
- ``python main.py --show-examples --example-name AutoLabeller``
- ``python main.py --show-examples --example-name DarknetAutoLabeler``
- ``python main.py --show-examples --example-name TensorFlowAutoLabeler``
- ``python main.py --show-examples --example-name image_template``
- ``python main.py --show-examples --example-name save_screenshot``
- ``python main.py --show-examples --example-name video_capture_template``
"""

# The actual CLI implementation lives inside the packaged library.
# Keeping this wrapper thin makes the project runnable both from source
# and after installation via the console script entrypoint.
from ai_vision_tool.cli import main


# Delegate directly to the packaged CLI.
# This preserves a single source of truth for runtime flags, examples,
# webcam behavior, and profile-driven augmentation loading.
if __name__ == "__main__":
    main()
