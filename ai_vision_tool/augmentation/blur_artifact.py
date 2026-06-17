from __future__ import annotations

import cv2
import numpy as np

from ..core.base import AIVisionComponent
from ..utils.image_utils import ensure_odd, extract_frame, replace_frame, to_uint8


class Posterize(AIVisionComponent):
    """Reduces color depth by retaining only the most significant bits per channel.

    Args:
        bits (int): Number of bits to keep per channel. Lower values produce stronger posterization. Default is 4.
    """

    def __init__(self, bits=4):
        """Initializes Posterize with a bit depth.

        Args:
            bits (int): Number of bits retained per channel. Default is 4.
        """
        super().__init__()
        self.bits = bits

    def _execute(self, data, config):
        """Applies posterization to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'bits'.

        Returns:
            NumPy array or dict: Posterized image in the same format as input.
        """
        frame = extract_frame(data)
        bits = int(config.get("bits", self.bits))
        shift = max(0, 8 - bits)
        output = np.left_shift(np.right_shift(frame, shift), shift)
        return replace_frame(data, output)


class Solarize(AIVisionComponent):
    """Inverts pixel values that meet or exceed a threshold, simulating a solarize effect.

    Args:
        threshold (int): Pixel intensity threshold in [0, 255]. Pixels >= threshold are inverted. Default is 128.
    """

    def __init__(self, threshold=128):
        """Initializes Solarize with an inversion threshold.

        Args:
            threshold (int): Intensity threshold above which pixels are inverted. Default is 128.
        """
        super().__init__()
        self.threshold = threshold

    def _execute(self, data, config):
        """Applies solarization to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'threshold'.

        Returns:
            NumPy array or dict: Solarized image in the same format as input.
        """
        frame = extract_frame(data)
        threshold = int(config.get("threshold", self.threshold))
        output = np.where(frame >= threshold, 255 - frame, frame).astype(frame.dtype)
        return replace_frame(data, output)


class Equalize(AIVisionComponent):
    """Equalizes the histogram of the luminance channel (Y in YCrCb) to enhance contrast."""

    def _execute(self, data, config):
        """Applies histogram equalization to the Y channel of the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Unused. Present for interface compatibility.

        Returns:
            NumPy array or dict: Contrast-equalized image in the same format as input.
        """
        frame = extract_frame(data)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        output = cv2.cvtColor(
            cv2.merge((cv2.equalizeHist(y), cr, cb)), cv2.COLOR_YCrCb2BGR
        )
        return replace_frame(data, output)


class Emboss(AIVisionComponent):
    """Applies an emboss convolution filter to create a raised-relief effect.

    Args:
        strength (float): Multiplier applied to the emboss kernel. Default is 1.0.
    """

    def __init__(self, strength=1.0):
        """Initializes Emboss with a kernel strength multiplier.

        Args:
            strength (float): Kernel multiplier. Default is 1.0.
        """
        super().__init__()
        self.strength = strength

    def _execute(self, data, config):
        """Applies the emboss filter to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'strength'.

        Returns:
            NumPy array or dict: Embossed image in the same format as input.
        """
        frame = extract_frame(data)
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        strength = float(config.get("strength", self.strength))
        output = cv2.filter2D(frame, -1, kernel * strength) + 128
        return replace_frame(data, to_uint8(output))


class Sharpen(AIVisionComponent):
    """Sharpens an image using unsharp masking.

    Args:
        amount (float): Sharpening strength. Higher values produce stronger edges. Default is 1.0.
    """

    def __init__(self, amount=1.0):
        """Initializes Sharpen with an unsharp mask amount.

        Args:
            amount (float): Sharpening intensity. Default is 1.0.
        """
        super().__init__()
        self.amount = amount

    def _execute(self, data, config):
        """Applies unsharp masking to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'amount'.

        Returns:
            NumPy array or dict: Sharpened image in the same format as input.
        """
        frame = extract_frame(data)
        amount = float(config.get("amount", self.amount))
        blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
        output = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        return replace_frame(data, output)


class GaussianBlur(AIVisionComponent):
    """Applies Gaussian blur to an image frame.

    Args:
        kernel_size (int): Size of the Gaussian kernel. Must be odd. Default is 5.
        sigma_x (float): Standard deviation in the X direction. Default is 0.0 (auto-computed).
    """

    def __init__(self, kernel_size=5, sigma_x=0.0):
        """Initializes GaussianBlur with kernel size and sigma.

        Args:
            kernel_size (int): Kernel size. Must be odd. Default is 5.
            sigma_x (float): Standard deviation in X. Default is 0.0 (auto).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def _execute(self, data, config):
        """Applies Gaussian blur to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'kernel_size' and 'sigma_x'.

        Returns:
            NumPy array or dict: Blurred image in the same format as input.
        """
        frame = extract_frame(data)
        kernel = ensure_odd(config.get("kernel_size", self.kernel_size))
        sigma_x = float(config.get("sigma_x", self.sigma_x))
        return replace_frame(data, cv2.GaussianBlur(frame, (kernel, kernel), sigma_x))


class MedianBlur(AIVisionComponent):
    """Applies median blur to reduce salt-and-pepper noise.

    Args:
        kernel_size (int): Size of the median filter kernel. Must be odd. Default is 5.
    """

    def __init__(self, kernel_size=5):
        """Initializes MedianBlur with a kernel size.

        Args:
            kernel_size (int): Kernel size. Must be odd. Default is 5.
        """
        super().__init__()
        self.kernel_size = kernel_size

    def _execute(self, data, config):
        """Applies median blur to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'kernel_size'.

        Returns:
            NumPy array or dict: Median-blurred image in the same format as input.
        """
        frame = extract_frame(data)
        kernel = ensure_odd(config.get("kernel_size", self.kernel_size))
        return replace_frame(data, cv2.medianBlur(frame, kernel))


class GlassBlur(AIVisionComponent):
    """Simulates the look of frosted glass by randomly swapping neighboring pixels.

    Args:
        sigma (float): Standard deviation for the initial Gaussian smoothing pass. Default is 0.7.
        max_delta (int): Maximum pixel displacement for neighbor swaps. Default is 2.
        iterations (int): Number of swap passes. Default is 1.
    """

    def __init__(self, sigma=0.7, max_delta=2, iterations=1):
        """Initializes GlassBlur with blur parameters.

        Args:
            sigma (float): Gaussian sigma for pre-smoothing. Default is 0.7.
            max_delta (int): Max pixel offset for swaps. Default is 2.
            iterations (int): Number of swap iterations. Default is 1.
        """
        super().__init__()
        self.sigma = sigma
        self.max_delta = max_delta
        self.iterations = iterations

    def _execute(self, data, config):
        """Applies glass blur to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'sigma', 'max_delta', 'iterations'.

        Returns:
            NumPy array or dict: Glass-blurred image in the same format as input.
        """
        frame = extract_frame(data).copy()
        sigma = float(config.get("sigma", self.sigma))
        max_delta = int(config.get("max_delta", self.max_delta))
        iterations = int(config.get("iterations", self.iterations))
        output = cv2.GaussianBlur(frame, (0, 0), sigma)
        height, width = output.shape[:2]
        for _ in range(iterations):
            for y in range(max_delta, height - max_delta):
                for x in range(max_delta, width - max_delta):
                    dx = np.random.randint(-max_delta, max_delta + 1)
                    dy = np.random.randint(-max_delta, max_delta + 1)
                    output[y, x], output[y + dy, x + dx] = (
                        output[y + dy, x + dx].copy(),
                        output[y, x].copy(),
                    )
        return replace_frame(data, output)


class DefocusBlur(AIVisionComponent):
    """Simulates lens defocus using a circular (disk) convolution kernel.

    Args:
        radius (int): Radius of the disk kernel in pixels. Default is 5.
    """

    def __init__(self, radius=5):
        """Initializes DefocusBlur with a disk radius.

        Args:
            radius (int): Disk kernel radius. Default is 5.
        """
        super().__init__()
        self.radius = radius

    def _execute(self, data, config):
        """Applies defocus blur to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'radius'.

        Returns:
            NumPy array or dict: Defocus-blurred image in the same format as input.
        """
        frame = extract_frame(data)
        radius = max(1, int(config.get("radius", self.radius)))
        kernel_size = radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        kernel /= kernel.sum()
        output = cv2.filter2D(frame, -1, kernel)
        return replace_frame(data, output)


class ZoomBlur(AIVisionComponent):
    """Simulates zoom-motion blur by averaging progressively zoomed frames.

    Args:
        zoom_factor (float): Maximum zoom multiplier applied across all steps. Default is 1.2.
        steps (int): Number of zoom levels averaged together. Default is 5.
    """

    def __init__(self, zoom_factor=1.2, steps=5):
        """Initializes ZoomBlur with zoom factor and step count.

        Args:
            zoom_factor (float): Maximum zoom scale. Default is 1.2.
            steps (int): Number of intermediate zoom levels. Default is 5.
        """
        super().__init__()
        self.zoom_factor = zoom_factor
        self.steps = steps

    def _execute(self, data, config):
        """Applies zoom blur to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'zoom_factor' and 'steps'.

        Returns:
            NumPy array or dict: Zoom-blurred image in the same format as input.
        """
        frame = extract_frame(data).astype(np.float32)
        height, width = frame.shape[:2]
        zoom_factor = float(config.get("zoom_factor", self.zoom_factor))
        steps = int(config.get("steps", self.steps))
        accum = np.zeros_like(frame, dtype=np.float32)
        for i in range(steps):
            factor = 1.0 + (zoom_factor - 1.0) * (i / max(steps - 1, 1))
            zoomed = cv2.resize(
                frame, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR
            )
            zh, zw = zoomed.shape[:2]
            x1 = max(0, (zw - width) // 2)
            y1 = max(0, (zh - height) // 2)
            cropped = zoomed[y1 : y1 + height, x1 : x1 + width]
            if cropped.shape[:2] != (height, width):
                cropped = cv2.resize(cropped, (width, height))
            accum += cropped
        output = to_uint8(accum / steps)
        return replace_frame(data, output)


class CompressionArtifacts(AIVisionComponent):
    """Introduces JPEG compression artifacts at a specified quality level.

    Delegates to JPEGCompression internally.

    Args:
        quality (int): JPEG quality in [1, 100]. Lower values introduce stronger artifacts. Default is 40.
    """

    def __init__(self, quality=40):
        """Initializes CompressionArtifacts with a JPEG quality level.

        Args:
            quality (int): JPEG quality in [1, 100]. Default is 40.
        """
        super().__init__()
        self.quality = quality

    def _execute(self, data, config):
        """Applies JPEG compression artifacts to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'quality'.

        Returns:
            NumPy array or dict: Artifact-compressed image in the same format as input.
        """
        return JPEGCompression(quality=config.get("quality", self.quality)).run(
            data, config
        )


class JPEGCompression(AIVisionComponent):
    """Encodes and decodes a frame as JPEG to simulate compression loss.

    Args:
        quality (int): JPEG quality in [1, 100]. Lower values produce more compression artifacts. Default is 50.
    """

    def __init__(self, quality=50):
        """Initializes JPEGCompression with a quality level.

        Args:
            quality (int): JPEG quality factor. Default is 50.
        """
        super().__init__()
        self.quality = quality

    def _execute(self, data, config):
        """Encodes the frame to JPEG and decodes it back to introduce compression artifacts.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'quality'.

        Returns:
            NumPy array or dict: JPEG-compressed image in the same format as input.
            Returns a copy of the original frame if encoding fails.
        """
        frame = extract_frame(data)
        quality = int(config.get("quality", self.quality))
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return replace_frame(data, frame.copy())
        output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return replace_frame(data, output)


class Downscale(AIVisionComponent):
    """Downscales an image and upscales it back to simulate low-resolution artifacts.

    Args:
        scale (float): Downscale factor in (0, 1]. Default is 0.5.
        interpolation (str): Interpolation method for downscaling ('nearest', 'linear', 'cubic', 'area'). Default is 'area'.
    """

    def __init__(self, scale=0.5, interpolation="area"):
        """Initializes Downscale with a scale factor and interpolation mode.

        Args:
            scale (float): Fraction of original size to downscale to. Default is 0.5.
            interpolation (str): Interpolation method for downscaling. Default is 'area'.
        """
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation

    def _execute(self, data, config):
        """Downscales and upscales the extracted frame to simulate resolution loss.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'scale' and 'interpolation'.

        Returns:
            NumPy array or dict: Degraded image restored to original size, in the same format as input.
        """
        frame = extract_frame(data)
        scale = float(config.get("scale", self.scale))
        interpolation = _resolve_interpolation(
            config.get("interpolation", self.interpolation)
        )
        height, width = frame.shape[:2]
        down = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=interpolation,
        )
        output = cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)
        return replace_frame(data, output)


class Superpixel(AIVisionComponent):
    """Segments an image into superpixels and replaces each region with its mean color.

    Requires opencv-contrib-python for cv2.ximgproc. Falls back to identity if unavailable.

    Args:
        region_size (int): Approximate superpixel region size in pixels. Default is 10.
        ruler (float): Compactness parameter for the SLIC algorithm. Default is 10.0.
    """

    def __init__(self, region_size=10, ruler=10.0):
        """Initializes Superpixel with region size and compactness.

        Args:
            region_size (int): Target superpixel region size. Default is 10.
            ruler (float): SLIC compactness. Default is 10.0.
        """
        super().__init__()
        self.region_size = region_size
        self.ruler = ruler

    def _execute(self, data, config):
        """Applies superpixel segmentation and mean-color fill to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'region_size' and 'ruler'.

        Returns:
            NumPy array or dict: Superpixel-processed image, or original frame if ximgproc is unavailable.
        """
        frame = extract_frame(data)
        if not hasattr(cv2, "ximgproc"):
            return replace_frame(data, frame.copy())
        slic = cv2.ximgproc.createSuperpixelSLIC(
            frame,
            algorithm=cv2.ximgproc.SLICO,
            region_size=int(config.get("region_size", self.region_size)),
            ruler=float(config.get("ruler", self.ruler)),
        )
        slic.iterate(5)
        labels = slic.getLabels()
        output = np.zeros_like(frame)
        for label in np.unique(labels):
            mask = labels == label
            output[mask] = frame[mask].mean(axis=0)
        return replace_frame(data, to_uint8(output))


def _resolve_interpolation(name):
    """Maps an interpolation name string to the corresponding OpenCV flag.

    Args:
        name (str): Interpolation method name ('nearest', 'linear', 'cubic', 'area').

    Returns:
        int: OpenCV interpolation flag. Defaults to cv2.INTER_AREA for unknown names.
    """
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    return mapping.get(str(name).lower(), cv2.INTER_AREA)
