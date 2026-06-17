from __future__ import annotations


def _make_pipeline(components: list):
    from ai_vision_tool.pipelines.vision_pipeline import AIVisionPipeline

    p = AIVisionPipeline()
    for c in components:
        p.add(c)
    return p


class PrebuiltPipelines:
    """Factory class for common pipeline configurations."""

    @classmethod
    def detection_pipeline(
        cls,
        model_path: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        draw: bool = True,
    ):
        from ai_vision_tool.preprocessing.geometry import LetterboxResize

        comps = [LetterboxResize(640, 640)]
        if model_path:
            from ai_vision_tool.detection.object_detector import ObjectDetector

            comps.append(
                ObjectDetector(
                    model_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                )
            )
        if draw:
            from ai_vision_tool.utils.draw_utils import DrawUtils

            comps.append(DrawUtils())
        return _make_pipeline(comps)

    @classmethod
    def augmentation_pipeline(cls, profile: str = "standard"):
        from ai_vision_tool.augmentation.brightness import Brightness
        from ai_vision_tool.augmentation.cutout import Cutout
        from ai_vision_tool.augmentation.flip import Flip
        from ai_vision_tool.augmentation.geometric_random import PerspectiveTransform
        from ai_vision_tool.augmentation.noise import Noise
        from ai_vision_tool.augmentation.weather_light import (
            RandomBrightnessContrast,
            RandomFog,
        )

        if profile == "minimal":
            return _make_pipeline([Flip()])
        if profile == "standard":
            return _make_pipeline(
                [Flip(), Brightness(factor_range=(0.7, 1.3)), Noise(prob=0.3)]
            )
        if profile == "heavy":
            return _make_pipeline(
                [
                    Flip(),
                    Brightness(factor_range=(0.5, 1.5)),
                    Noise(prob=0.5),
                    PerspectiveTransform(prob=0.3),
                    Cutout(prob=0.3),
                    RandomFog(prob=0.2),
                    RandomBrightnessContrast(prob=0.3),
                ]
            )
        if profile == "detection":
            from ai_vision_tool.augmentation.geometric_random import RandomScale
            from ai_vision_tool.augmentation.mosaic import Mosaic

            return _make_pipeline([Mosaic(), Flip(), RandomScale()])
        return _make_pipeline([Flip()])

    @classmethod
    def preprocessing_pipeline(
        cls,
        target_size: tuple = (640, 640),
        normalize: bool = True,
        quality_check: bool = True,
    ):
        from ai_vision_tool.preprocessing.geometry import LetterboxResize
        from ai_vision_tool.preprocessing.intensity import Normalize
        from ai_vision_tool.preprocessing.quality import ImageQualityCheck

        comps = [LetterboxResize(*target_size)]
        if normalize:
            comps.append(Normalize())
        if quality_check:
            comps.append(ImageQualityCheck())
        return _make_pipeline(comps)

    @classmethod
    def quality_check_pipeline(cls):
        from ai_vision_tool.preprocessing.quality import (
            BlurDetection,
            BrightnessCheck,
            CorruptImageCheck,
            MinSizeFilter,
        )

        return _make_pipeline(
            [CorruptImageCheck(), BlurDetection(), BrightnessCheck(), MinSizeFilter()]
        )

    @classmethod
    def tracking_pipeline(
        cls, model_path: str | None = None, conf_threshold: float = 0.25
    ):
        from ai_vision_tool.preprocessing.geometry import LetterboxResize
        from ai_vision_tool.tracking.byte_tracker import ByteTracker
        from ai_vision_tool.utils.draw_utils import DrawUtils

        comps = [LetterboxResize(640, 640)]
        if model_path:
            from ai_vision_tool.detection.object_detector import ObjectDetector

            comps.append(ObjectDetector(model_path, conf_threshold=conf_threshold))
        comps.extend([ByteTracker(), DrawUtils()])
        return _make_pipeline(comps)

    @classmethod
    def enhancement_pipeline(cls, method: str = "clahe"):
        from ai_vision_tool.enhancement.denoiser import Denoiser
        from ai_vision_tool.enhancement.low_light import LowLightEnhancer

        return _make_pipeline([LowLightEnhancer(method=method), Denoiser()])
