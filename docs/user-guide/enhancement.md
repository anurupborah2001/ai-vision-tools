# Enhancement

The `enhancement` module improves image quality: denoising, brightness correction for dark scenes, super-resolution upscaling, colorization, and deblurring.

```python
from ai_vision_tool.enhancement import <ClassName>
```

Heavy model backends (ONNX, PyTorch) are auto-loaded if available, with pure cv2 fallbacks.

---

## Denoising

`Denoiser` supports multiple methods, auto-selects cv2 fallback if no extra is installed.

```python
from ai_vision_tool.enhancement import Denoiser

# Non-local means (best quality, slow)
denoiser = Denoiser(method="nlmeans", h=10)
output = denoiser.run(noisy_image)

# Bilateral filter (edge-preserving, fast)
denoiser = Denoiser(method="bilateral", diameter=9, sigma_color=75, sigma_space=75)

# Gaussian (fastest, least quality)
denoiser = Denoiser(method="gaussian", kernel_size=5)

# DnCNN via ONNX (requires [onnx] extra)
denoiser = Denoiser(method="dncnn", model_path="dncnn.onnx")
output = denoiser.run(noisy_image)
```

---

## Low-Light Enhancement

`LowLightEnhancer` recovers detail in dark or underexposed images.

```python
from ai_vision_tool.enhancement import LowLightEnhancer

# CLAHE-based (fast, no extra required)
enhancer = LowLightEnhancer(method="clahe")
output = enhancer.run(dark_image)

# Gamma correction
enhancer = LowLightEnhancer(method="gamma", gamma=0.5)

# Multi-Scale Retinex (MSR)
enhancer = LowLightEnhancer(method="msr", sigmas=[15, 80, 250])

# Zero-DCE via ONNX (requires [onnx] extra)
enhancer = LowLightEnhancer(method="zerodce", model_path="zero_dce.onnx")
output = enhancer.run(dark_image)
```

---

## Frame Enhancer

`FrameEnhancer` applies a brightness → contrast → sharpness pass in a single component.

```python
from ai_vision_tool.enhancement import FrameEnhancer

enhancer = FrameEnhancer(brightness=1.1, contrast=1.2, sharpness=1.3)
output = enhancer.run(image)
```

---

## Super Resolution

`SuperResolution` upscales images beyond their native resolution.

> **Extra recommended:** `pip install "ai-vision-tool[onnx]"` or `[torch]` for DL-backed models.

```python
from ai_vision_tool.enhancement.models import SuperResolution

# OpenCV DNN super-res (EDSR, ESPCN, FSRCNN, LapSRN)
sr = SuperResolution(method="cv2_dnn_superres", model_path="EDSR_x4.pb", scale=4)
output = sr.run(low_res_image)

# ONNX model
sr = SuperResolution(method="onnx", model_path="real_esrgan_x4.onnx", scale=4)
output = sr.run(low_res_image)

# Bicubic fallback (no extra required)
sr = SuperResolution(method="bicubic", scale=2)
output = sr.run(low_res_image)
```

---

## Colorization

`Colorizer` converts grayscale images to plausible color using Zhang 2016 LAB-AB prediction.

> **Extra required:** `pip install "ai-vision-tool[onnx]"` or `[torch]`.

```python
from ai_vision_tool.enhancement.models import Colorizer

# ONNX backend
colorizer = Colorizer(method="onnx", model_path="colorizer.onnx")
output = colorizer.run(gray_image)     # returns BGR color image

# Pseudo-color (deterministic, heuristic-based)
colorizer = Colorizer(method="pseudo_color")
output = colorizer.run(gray_image)
```

---

## Deblurring

`Deblurrer` restores images blurred by camera shake or focus errors.

```python
from ai_vision_tool.enhancement.models import Deblurrer

# Wiener filter (frequency domain, fast)
deblurrer = Deblurrer(method="wiener", kernel_size=15, noise_power=0.01)
output = deblurrer.run(blurry_image)

# Richardson-Lucy iterative deconvolution
deblurrer = Deblurrer(method="rl", iterations=30, psf_size=15)
output = deblurrer.run(blurry_image)

# NAFNet via ONNX (best quality, requires [onnx] extra)
deblurrer = Deblurrer(method="nafnet", model_path="nafnet.onnx")
output = deblurrer.run(blurry_image)
```

---

## Pipeline Integration

```python
from ai_vision_tool import AIVisionPipeline
from ai_vision_tool.enhancement import LowLightEnhancer, Denoiser
from ai_vision_tool.enhancement.models import SuperResolution

pipeline = AIVisionPipeline([
    LowLightEnhancer(method="clahe"),
    Denoiser(method="bilateral"),
    SuperResolution(method="bicubic", scale=2),
])

result = pipeline.execute(dark_noisy_image)
```
