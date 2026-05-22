# Augmentation

The `augmentation` module provides 50+ transforms for generating training data. All components follow the same `run(data, config)` interface and can be chained in a pipeline.

```python
from ai_vision_tool.augmentation import <ClassName>
```

---

## Basic Transforms

### Flip

```python
from ai_vision_tool.augmentation import Flip

# Horizontal flip
output = Flip(direction="horizontal", probability=0.5).run(image)

# Vertical flip
output = Flip(direction="vertical", probability=0.5).run(image)

# Both axes
output = Flip(direction="both").run(image)
```

---

### Rotation / Rotate90

```python
from ai_vision_tool.augmentation import Rotation, Rotate90

# Arbitrary rotation (degrees, random range)
output = Rotation(angle=15, probability=0.5).run(image)
output = Rotation(min_angle=-30, max_angle=30).run(image)

# 90-degree multiples only (fast, lossless)
output = Rotate90(times=1).run(image)     # 90°
output = Rotate90(times=2).run(image)     # 180°
output = Rotate90(times=3).run(image)     # 270°
```

---

### Crop

```python
from ai_vision_tool.augmentation import Crop

output = Crop(x=100, y=50, width=400, height=300).run(image)
```

---

### Shear

```python
from ai_vision_tool.augmentation import Shear

output = Shear(shear_x=0.2, shear_y=0.0).run(image)
```

---

## Color and Brightness

### Brightness / Exposure / Hue / Saturation / CameraGain

```python
from ai_vision_tool.augmentation import Brightness, Exposure, Hue, Saturation, CameraGain

output = Brightness(factor=1.2).run(image)          # 20% brighter
output = Exposure(ev=0.5).run(image)                # +0.5 EV
output = Hue(shift=15).run(image)                   # hue shift in degrees
output = Saturation(factor=1.3).run(image)          # more vivid
output = CameraGain(gain=1.1).run(image)            # simulate sensor gain
```

---

### Greyscale

```python
from ai_vision_tool.augmentation import Greyscale

output = Greyscale(probability=0.3).run(image)      # 30% chance to convert
```

---

## Blur and Artifact Transforms

### Blur / MotionBlur

```python
from ai_vision_tool.augmentation import Blur, MotionBlur

output = Blur(kernel_size=5).run(image)
output = MotionBlur(kernel_size=15, angle=45).run(image)
```

### Advanced Blur Variants

```python
from ai_vision_tool.augmentation import (
    GaussianBlur, MedianBlur, GlassBlur, DefocusBlur, ZoomBlur
)

output = GaussianBlur(sigma_limit=(0.1, 2.0)).run(image)
output = MedianBlur(blur_limit=7).run(image)
output = GlassBlur(sigma=0.7, max_delta=4).run(image)
output = DefocusBlur(radius=3).run(image)
output = ZoomBlur(max_factor=1.3).run(image)
```

---

## Noise and Dropout

### Noise

```python
from ai_vision_tool.augmentation import Noise

output = Noise(intensity=0.05, noise_type="gaussian").run(image)
output = Noise(intensity=0.02, noise_type="salt_pepper").run(image)
```

### Advanced Noise

```python
from ai_vision_tool.augmentation import ISONoise, MultiplicativeNoise, SaltPepperNoise

output = ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)).run(image)
output = MultiplicativeNoise(multiplier=(0.9, 1.1)).run(image)
output = SaltPepperNoise(salt_vs_pepper=0.5, amount=0.02).run(image)
```

### Dropout / Erasing

```python
from ai_vision_tool.augmentation import (
    Cutout, CoarseDropout, GridDropout, RandomErasing, PixelDropout, MaskDropout
)

output = Cutout(num_holes=4, hole_size=32).run(image)
output = CoarseDropout(max_holes=8, max_height=32, max_width=32).run(image)
output = GridDropout(ratio=0.5).run(image)
output = RandomErasing(probability=0.5, scale=(0.02, 0.33)).run(image)
output = PixelDropout(dropout_prob=0.01).run(image)
```

---

## Geometric Random Transforms

```python
from ai_vision_tool.augmentation import (
    RandomResize, RandomScale, RandomCrop, RandomResizedCrop,
    RandomPadding, Translate, AffineTransform,
    PerspectiveTransform, ElasticTransform,
    GridDistortion, OpticalDistortion,
)

output = RandomResize(scale_range=(0.8, 1.2)).run(image)
output = RandomCrop(width=512, height=512).run(image)
output = RandomResizedCrop(width=640, height=640, scale=(0.08, 1.0)).run(image)
output = Translate(tx=0.1, ty=0.05).run(image)           # fractional shift
output = AffineTransform(rotate=15, scale=1.1, shear=5).run(image)
output = PerspectiveTransform(scale=0.05).run(image)
output = ElasticTransform(alpha=120, sigma=6).run(image)
output = GridDistortion(num_steps=5, distort_limit=0.3).run(image)
output = OpticalDistortion(distort_limit=0.05, shift_limit=0.05).run(image)
```

---

## Composite / Mosaic Transforms

### Mosaic

```python
from ai_vision_tool.augmentation import Mosaic

# Combines 4 images into one mosaic tile
mosaic = Mosaic(output_size=(640, 640))
images = [img1, img2, img3, img4]
output = mosaic.run(images)
```

### Mosaic9

Nine-image mosaic variant:

```python
from ai_vision_tool.augmentation import Mosaic9

mosaic9 = Mosaic9(output_size=(1280, 1280))
output = mosaic9.run(nine_images)
```

### MixUp / CutMix / CopyPaste

```python
from ai_vision_tool.augmentation import MixUp, CutMix, CopyPaste

payload = {"frame": img1, "mix_image": img2, "labels": labels1, "mix_labels": labels2}

output = MixUp(alpha=0.4).run(payload)
output = CutMix(alpha=1.0).run(payload)
output = CopyPaste(probability=0.5).run(payload)
```

### ObjectPaste / RandomOcclusion / BoundingBoxJitter

```python
from ai_vision_tool.augmentation import ObjectPaste, RandomOcclusion, BoundingBoxJitter

# Paste objects from one image into another
output = ObjectPaste(probability=0.5).run(payload)

# Random rectangular occlusion over detected objects
output = RandomOcclusion(max_occlusion=0.4).run(payload)

# Slightly jitter bounding box coordinates for label noise
output = BoundingBoxJitter(jitter=0.05).run(payload)
```

---

## Weather and Light Effects

```python
from ai_vision_tool.augmentation import (
    RandomShadow, RandomSunFlare, RandomFog,
    RandomRain, RandomSnow, RandomGamma,
    ColorJitter, ChannelShuffle, RGBShift, HSVShift,
    ToSepia, InvertImage, RandomBrightnessContrast,
)

output = RandomShadow(num_shadows=2).run(image)
output = RandomSunFlare(flare_roi=(0, 0, 1, 0.5)).run(image)
output = RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6).run(image)
output = RandomRain(slant_lower=-10, slant_upper=10).run(image)
output = RandomSnow(snow_point_lower=0.1).run(image)
output = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1).run(image)
output = ChannelShuffle(probability=0.5).run(image)
output = RGBShift(r_shift=(-20, 20), g_shift=(-20, 20), b_shift=(-20, 20)).run(image)
output = HSVShift(hue_shift=10, sat_shift=20, val_shift=10).run(image)
output = ToSepia().run(image)
output = InvertImage().run(image)
output = RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2).run(image)
```

---

## Other Artifact Transforms

```python
from ai_vision_tool.augmentation import (
    Emboss, Posterize, Solarize, Equalize,
    CompressionArtifacts, JPEGCompression,
    Downscale, Superpixel,
)

output = Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7)).run(image)
output = Posterize(num_bits=4).run(image)
output = Solarize(threshold=128).run(image)
output = Equalize(mode="cv").run(image)
output = JPEGCompression(quality_lower=50, quality_upper=90).run(image)
output = Downscale(scale_min=0.25, scale_max=0.5).run(image)
output = Superpixel(p_replace=0.1, n_segments=100).run(image)
```

---

## JSON Augmentation Profiles

Define a pipeline as a JSON file for CLI or programmatic use:

```json
{
  "components": [
    { "name": "Flip",     "params": { "direction": "horizontal", "probability": 0.5 } },
    { "name": "Rotation", "params": { "min_angle": -15, "max_angle": 15 } },
    { "name": "Noise",    "params": { "intensity": 0.03 } },
    { "name": "CLAHE",    "params": { "clip_limit": 2.0 } }
  ]
}
```

```python
from ai_vision_tool.augmentation.common import parse_component_profile

components = parse_component_profile("augment.json")
```

Or from CLI:

```bash
ai-vision-tool --process-image-path ./images/ --profile augment.json
```
