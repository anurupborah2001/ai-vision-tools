# Preprocessing

The `preprocessing` module provides stateless components for geometry transforms, pixel-level intensity adjustments, and image quality filtering. All components follow the standard `AIVisionComponent` interface.

```python
from ai_vision_tool.preprocessing import <ClassName>
```

---

## Geometry Transforms

### Resize

Resize to exact pixel dimensions with configurable interpolation.

```python
from ai_vision_tool.preprocessing import Resize

transform = Resize(width=640, height=480, interpolation="linear")
output = transform.run(image)
```

**Interpolation options:** `linear` (default), `cubic`, `nearest`, `area`, `lanczos`

---

### LetterboxResize

Resize while preserving aspect ratio, padding with a fill color.

```python
from ai_vision_tool.preprocessing import LetterboxResize

transform = LetterboxResize(width=640, height=640, fill_color=(114, 114, 114))
output = transform.run(image)
```

Useful as a YOLO pre-processing step — maintains object proportions without distortion.

---

### CenterCrop

Crop a region from the center of the image.

```python
from ai_vision_tool.preprocessing import CenterCrop

transform = CenterCrop(width=512, height=512)
output = transform.run(image)
```

---

### PadToSquare

Pad shortest dimension to make the image square.

```python
from ai_vision_tool.preprocessing import PadToSquare

transform = PadToSquare(fill_color=(0, 0, 0))
output = transform.run(image)
```

---

### AutoCrop

Auto-detect and crop to the non-background content region.

```python
from ai_vision_tool.preprocessing import AutoCrop

transform = AutoCrop(threshold=10)
output = transform.run(image)
```

---

### PerspectiveCorrection

Correct perspective distortion given four corner points.

```python
from ai_vision_tool.preprocessing import PerspectiveCorrection

src_points = [(50, 50), (600, 30), (620, 450), (30, 470)]
transform = PerspectiveCorrection(src_points=src_points, output_size=(640, 480))
output = transform.run(image)
```

---

### Deskew

Detect and correct skew angle (useful for scanned documents).

```python
from ai_vision_tool.preprocessing import Deskew

transform = Deskew(max_angle=45)
output = transform.run(image)
```

---

### BoundingBoxClamp / BoundingBoxNormalize

Clamp bounding boxes to image boundaries and normalize coordinates.

```python
from ai_vision_tool.preprocessing import BoundingBoxClamp, BoundingBoxNormalize

payload = {"frame": image, "bboxes": [[10, 20, 700, 500]]}  # may go out of bounds

payload = BoundingBoxClamp().run(payload)          # clamp to image
payload = BoundingBoxNormalize().run(payload)      # scale 0–1
```

---

### FaceAlign / ObjectCrop / MaskResize

```python
from ai_vision_tool.preprocessing import FaceAlign, ObjectCrop, MaskResize

# Align face using landmark points
payload = FaceAlign(output_size=(112, 112)).run({"frame": image, "landmarks": pts})

# Crop tightest bounding box around each detected object
payload = ObjectCrop(padding=10).run({"frame": image, "bboxes": boxes})

# Resize segmentation masks to match a new frame size
payload = MaskResize(width=640, height=480).run({"frame": image, "masks": masks})
```

---

## Intensity Adjustments

### Normalize / Standardize / RescalePixels

```python
from ai_vision_tool.preprocessing import Normalize, Standardize, RescalePixels

# Normalize to [0, 1]
output = Normalize().run(image)

# Zero-mean, unit-std per channel
output = Standardize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).run(image)

# Rescale pixel range
output = RescalePixels(in_range=(0, 255), out_range=(0, 1)).run(image)
```

---

### Color Space Conversion

```python
from ai_vision_tool.preprocessing import BGRToRGB, RGBToBGR, ConvertColorSpace

# Common conversions
rgb_image = BGRToRGB().run(bgr_image)          # OpenCV → ML frameworks
bgr_image = RGBToBGR().run(rgb_image)

# General conversion
gray = ConvertColorSpace(code="bgr2gray").run(image)
hsv  = ConvertColorSpace(code="bgr2hsv").run(image)
```

---

### CLAHE

Contrast Limited Adaptive Histogram Equalization — improves local contrast without amplifying noise.

```python
from ai_vision_tool.preprocessing import CLAHE

transform = CLAHE(clip_limit=3.0, tile_grid_size=(8, 8))
output = transform.run(image)
```

---

### GammaCorrection / WhiteBalance

```python
from ai_vision_tool.preprocessing import GammaCorrection, WhiteBalance

# Gamma correction (< 1 brightens, > 1 darkens)
output = GammaCorrection(gamma=0.7).run(image)

# Auto white balance
output = WhiteBalance(method="gray_world").run(image)
```

---

### Denoise / Sharpen / Deblur

```python
from ai_vision_tool.preprocessing import Denoise, Sharpen, Deblur

output = Denoise(method="bilateral", diameter=9).run(image)
output = Sharpen(strength=1.5).run(image)
output = Deblur(method="wiener").run(image)
```

---

### Edge and Threshold

```python
from ai_vision_tool.preprocessing import EdgeDetection, Threshold, AdaptiveThreshold, ContourExtraction

edges = EdgeDetection(method="canny", low=50, high=150).run(image)
binary = Threshold(thresh=127, method="binary").run(image)
adaptive = AdaptiveThreshold(block_size=11, C=2).run(image)
contours = ContourExtraction(min_area=100).run(image)
```

---

## Quality Filters

Quality components return the original image/payload if it passes, or raise / return `None` if it fails.

```python
from ai_vision_tool.preprocessing import (
    ImageQualityCheck,
    BlurDetection,
    BrightnessCheck,
    DuplicateImageCheck,
    CorruptImageCheck,
    AspectRatioFilter,
    MinSizeFilter,
    MaxSizeFilter,
)

# Reject blurry images (Laplacian variance threshold)
check = BlurDetection(threshold=100)
result = check.run(image)  # raises or returns None if too blurry

# Reject images outside brightness range
check = BrightnessCheck(min_brightness=30, max_brightness=220)

# Reject duplicates (perceptual hash)
check = DuplicateImageCheck(hash_size=8)

# Size constraints
check = MinSizeFilter(min_width=100, min_height=100)
check = MaxSizeFilter(max_width=4096, max_height=4096)

# Aspect ratio (e.g. only landscape images)
check = AspectRatioFilter(min_ratio=1.0, max_ratio=3.0)
```

---

## AutoOrient / AutoAdjustContrast / RemoveBackground

```python
from ai_vision_tool.preprocessing import AutoOrient, AutoAdjustContrast, RemoveBackground

# Correct EXIF-based rotation
output = AutoOrient().run(image)

# Auto contrast stretch
output = AutoAdjustContrast(percentile=2).run(image)

# Classical background removal via contour + mask
output = RemoveBackground(threshold=10).run(image)
```

---

## FrameResizer

Convenience wrapper — resizes to a target long edge, keeps aspect ratio.

```python
from ai_vision_tool.preprocessing import FrameResizer

transform = FrameResizer(max_size=1280)
output = transform.run(image)
```
