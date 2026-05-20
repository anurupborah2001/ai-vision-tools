from __future__ import annotations

import cv2
import numpy as np

from ..utils.image_utils import extract_frame, maybe_grayscale_to_bgr, replace_frame, to_uint8
from ..core.base import AIVisionComponent


class Normalize(AIVisionComponent):
    """Normalizes pixel values to a specified float range using min-max normalization.

    Args:
        output_min (float): Minimum output value. Default is 0.0.
        output_max (float): Maximum output value. Default is 1.0.
    """

    def __init__(self, output_min=0.0, output_max=1.0):
        """Initializes Normalize with output range bounds.

        Args:
            output_min (float): Minimum normalized value. Default is 0.0.
            output_max (float): Maximum normalized value. Default is 1.0.
        """
        super().__init__()
        self.output_min = output_min
        self.output_max = output_max

    def _execute(self, data, config):
        """Applies min-max normalization to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'output_min' and 'output_max'.

        Returns:
            NumPy array or dict: Float32 normalized image in the same format as input.
        """
        frame = extract_frame(data).astype(np.float32)
        output_min = float(config.get("output_min", self.output_min))
        output_max = float(config.get("output_max", self.output_max))
        normalized = cv2.normalize(frame, None, alpha=output_min, beta=output_max, norm_type=cv2.NORM_MINMAX)
        return replace_frame(data, normalized)


class Standardize(AIVisionComponent):
    """Standardizes pixel values by subtracting mean and dividing by standard deviation.

    Args:
        mean (list[float] or None): Per-channel mean. Default is None (computed from image).
        std (list[float] or None): Per-channel standard deviation. Default is None (computed from image).
        per_channel (bool): If True, computes statistics per channel. Default is True.
        epsilon (float): Small value added to denominator to avoid division by zero. Default is 1e-6.
    """

    def __init__(self, mean=None, std=None, per_channel=True, epsilon=1e-6):
        """Initializes Standardize with optional mean and std.

        Args:
            mean (list[float] or None): Channel means. Default is None (auto-computed).
            std (list[float] or None): Channel stds. Default is None (auto-computed).
            per_channel (bool): Compute stats per channel. Default is True.
            epsilon (float): Numerical stability floor. Default is 1e-6.
        """
        super().__init__()
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        self.epsilon = epsilon

    def _execute(self, data, config):
        """Standardizes the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'per_channel', 'epsilon',
                'mean', 'std'.

        Returns:
            NumPy array or dict: Standardized float32 image in the same format as input.
        """
        frame = extract_frame(data).astype(np.float32)
        per_channel = config.get("per_channel", self.per_channel)
        epsilon = float(config.get("epsilon", self.epsilon))

        if self.mean is None:
            mean = frame.mean(axis=(0, 1), keepdims=True) if per_channel else frame.mean()
        else:
            mean = np.array(config.get("mean", self.mean), dtype=np.float32)

        if self.std is None:
            std = frame.std(axis=(0, 1), keepdims=True) if per_channel else frame.std()
        else:
            std = np.array(config.get("std", self.std), dtype=np.float32)

        standardized = (frame - mean) / np.maximum(std, epsilon)
        return replace_frame(data, standardized)


class RescalePixels(AIVisionComponent):
    """Rescales pixel values by a multiplicative scale and additive offset.

    Args:
        scale (float): Multiplier applied to each pixel. Default is 1.0/255.0.
        offset (float): Additive offset applied after scaling. Default is 0.0.
        clip (bool): If True, clips output to [0.0, 1.0]. Default is False.
    """

    def __init__(self, scale=1.0 / 255.0, offset=0.0, clip=False):
        """Initializes RescalePixels with scale, offset, and clip flag.

        Args:
            scale (float): Pixel multiplier. Default is 1/255.
            offset (float): Additive offset. Default is 0.0.
            clip (bool): Clip to [0, 1]. Default is False.
        """
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.clip = clip

    def _execute(self, data, config):
        """Applies linear rescaling to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'pixel_scale' / 'scale',
                'pixel_offset' / 'offset', 'clip'.

        Returns:
            NumPy array or dict: Rescaled float32 image in the same format as input.
        """
        frame = extract_frame(data).astype(np.float32)
        scale = float(config.get("pixel_scale", config.get("scale", self.scale)))
        offset = float(config.get("pixel_offset", config.get("offset", self.offset)))
        clip = config.get("clip", self.clip)
        output = frame * scale + offset
        if clip:
            output = np.clip(output, 0.0, 1.0)
        return replace_frame(data, output)


class ConvertColorSpace(AIVisionComponent):
    """Converts an image from one color space to another.

    Supported conversions include BGR↔RGB, BGR/RGB→GRAY, BGR↔HSV, BGR↔LAB, BGR↔YCrCb.

    Args:
        source (str): Source color space name (e.g., 'BGR'). Default is 'BGR'.
        target (str): Target color space name (e.g., 'RGB'). Default is 'RGB'.
        keep_channels (bool): If converting to grayscale, expand back to 3 channels. Default is True.
    """

    def __init__(self, source="BGR", target="RGB", keep_channels=True):
        """Initializes ConvertColorSpace with source/target space names.

        Args:
            source (str): Source color space. Default is 'BGR'.
            target (str): Target color space. Default is 'RGB'.
            keep_channels (bool): Expand grayscale to 3 channels. Default is True.
        """
        super().__init__()
        self.source = source
        self.target = target
        self.keep_channels = keep_channels

    def _execute(self, data, config):
        """Converts the extracted frame to the target color space.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'source', 'target', 'keep_channels'.

        Returns:
            NumPy array or dict: Color-converted image in the same format as input.
        """
        frame = extract_frame(data)
        source = config.get("source", self.source).upper()
        target = config.get("target", self.target).upper()
        keep_channels = config.get("keep_channels", self.keep_channels)
        code = _resolve_color_code(source, target)
        output = cv2.cvtColor(frame, code)
        output = maybe_grayscale_to_bgr(output, keep_channels)
        return replace_frame(data, output)


class BGRToRGB(ConvertColorSpace):
    """Converts an image from BGR to RGB channel order."""

    def __init__(self):
        """Initializes BGRToRGB as a BGR→RGB ConvertColorSpace."""
        super().__init__(source="BGR", target="RGB")


class RGBToBGR(ConvertColorSpace):
    """Converts an image from RGB to BGR channel order."""

    def __init__(self):
        """Initializes RGBToBGR as a RGB→BGR ConvertColorSpace."""
        super().__init__(source="RGB", target="BGR")


class CLAHE(AIVisionComponent):
    """Applies Contrast Limited Adaptive Histogram Equalization to the L-channel in LAB space.

    Args:
        clip_limit (float): CLAHE clip limit. Default is 2.0.
        tile_grid_size (tuple[int, int]): CLAHE tile grid size. Default is (8, 8).
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Initializes CLAHE with clip limit and tile grid size.

        Args:
            clip_limit (float): Clip limit for contrast limiting. Default is 2.0.
            tile_grid_size (tuple[int, int]): Grid size for local histogram computation. Default is (8, 8).
        """
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def _execute(self, data, config):
        """Applies CLAHE to the L-channel of the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'clip_limit' and 'tile_grid_size'.

        Returns:
            NumPy array or dict: CLAHE-enhanced image in the same format as input.
        """
        frame = extract_frame(data)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(config.get("clip_limit", self.clip_limit)),
            tileGridSize=tuple(config.get("tile_grid_size", self.tile_grid_size)),
        )
        enhanced = clahe.apply(l_channel)
        output = cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        return replace_frame(data, output)


class HistogramEqualization(AIVisionComponent):
    """Equalizes the histogram of an image in YCrCb or grayscale color space.

    Args:
        color_space (str): Target color space for equalization ('ycrcb' or 'gray'). Default is 'ycrcb'.
    """

    def __init__(self, color_space="ycrcb"):
        """Initializes HistogramEqualization with a color space selector.

        Args:
            color_space (str): 'ycrcb' or 'gray'. Default is 'ycrcb'.
        """
        super().__init__()
        self.color_space = color_space

    def _execute(self, data, config):
        """Applies histogram equalization to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'color_space'.

        Returns:
            NumPy array or dict: Histogram-equalized image in the same format as input.

        Raises:
            ValueError: If color_space is not 'ycrcb' or 'gray'.
        """
        frame = extract_frame(data)
        color_space = config.get("color_space", self.color_space).lower()
        if color_space == "ycrcb":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(converted)
            equalized = cv2.equalizeHist(y)
            output = cv2.cvtColor(cv2.merge((equalized, cr, cb)), cv2.COLOR_YCrCb2BGR)
        elif color_space == "gray":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output = maybe_grayscale_to_bgr(cv2.equalizeHist(gray), True)
        else:
            raise ValueError("HistogramEqualization supports ycrcb or gray.")
        return replace_frame(data, output)


class GammaCorrection(AIVisionComponent):
    """Applies gamma correction via a precomputed lookup table.

    Args:
        gamma (float): Gamma value. Default is 1.0 (no change).
    """

    def __init__(self, gamma=1.0):
        """Initializes GammaCorrection with a gamma value.

        Args:
            gamma (float): Gamma correction factor. Default is 1.0.
        """
        super().__init__()
        self.gamma = gamma

    def _execute(self, data, config):
        """Applies gamma correction to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'gamma'.

        Returns:
            NumPy array or dict: Gamma-corrected image in the same format as input.
        """
        frame = extract_frame(data)
        gamma = max(float(config.get("gamma", self.gamma)), 1e-6)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        output = cv2.LUT(frame, table)
        return replace_frame(data, output)


class WhiteBalance(AIVisionComponent):
    """Applies white balance correction using gray-world or white-patch assumption.

    Args:
        method (str): White balance algorithm ('gray_world' or 'white_patch'). Default is 'gray_world'.
        percentile (float): Percentile used to identify highlights in 'white_patch' mode. Default is 95.
    """

    def __init__(self, method="gray_world", percentile=95):
        """Initializes WhiteBalance with method and percentile.

        Args:
            method (str): Balance algorithm. Default is 'gray_world'.
            percentile (float): Highlight percentile for white_patch. Default is 95.
        """
        super().__init__()
        self.method = method
        self.percentile = percentile

    def _execute(self, data, config):
        """Applies white balance to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'white_balance_method' / 'method',
                'percentile'.

        Returns:
            NumPy array or dict: White-balanced image in the same format as input.

        Raises:
            ValueError: If method is not 'gray_world' or 'white_patch'.
        """
        frame = extract_frame(data).astype(np.float32)
        method = config.get("white_balance_method", config.get("method", self.method)).lower()
        if method == "gray_world":
            channel_means = frame.mean(axis=(0, 1))
            scale = channel_means.mean() / np.maximum(channel_means, 1e-6)
            output = frame * scale
        elif method == "white_patch":
            percentile = float(config.get("percentile", self.percentile))
            reference = np.percentile(frame, percentile, axis=(0, 1))
            scale = 255.0 / np.maximum(reference, 1e-6)
            output = frame * scale
        else:
            raise ValueError("WhiteBalance supports gray_world or white_patch.")
        return replace_frame(data, to_uint8(output))


class Denoise(AIVisionComponent):
    """Removes noise from an image using non-local means, median, or bilateral filtering.

    Args:
        method (str): Denoising algorithm ('nlm', 'median', or 'bilateral'). Default is 'nlm'.
        strength (float): Denoising strength (used by 'nlm'). Default is 10.
        kernel_size (int): Kernel size (used by 'median' and 'bilateral'). Default is 5.
        sigma_color (float): Color sigma for bilateral filter. Default is 75.
        sigma_space (float): Spatial sigma for bilateral filter. Default is 75.
    """

    def __init__(self, method="nlm", strength=10, kernel_size=5, sigma_color=75, sigma_space=75):
        """Initializes Denoise with denoising method and parameters.

        Args:
            method (str): Denoising method. Default is 'nlm'.
            strength (float): NLM denoising strength. Default is 10.
            kernel_size (int): Filter kernel size. Default is 5.
            sigma_color (float): Bilateral color sigma. Default is 75.
            sigma_space (float): Bilateral spatial sigma. Default is 75.
        """
        super().__init__()
        self.method = method
        self.strength = strength
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def _execute(self, data, config):
        """Applies denoising to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'denoise_method' / 'method',
                'strength', 'kernel_size', 'sigma_color', 'sigma_space'.

        Returns:
            NumPy array or dict: Denoised image in the same format as input.

        Raises:
            ValueError: If method is not 'nlm', 'median', or 'bilateral'.
        """
        frame = extract_frame(data)
        method = config.get("denoise_method", config.get("method", self.method)).lower()
        if method == "nlm":
            output = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                float(config.get("strength", self.strength)),
                float(config.get("strength", self.strength)),
                7,
                21,
            )
        elif method == "median":
            output = cv2.medianBlur(frame, int(config.get("kernel_size", self.kernel_size)) | 1)
        elif method == "bilateral":
            output = cv2.bilateralFilter(
                frame,
                int(config.get("kernel_size", self.kernel_size)),
                float(config.get("sigma_color", self.sigma_color)),
                float(config.get("sigma_space", self.sigma_space)),
            )
        else:
            raise ValueError("Denoise supports nlm, median, or bilateral.")
        return replace_frame(data, output)


class Sharpen(AIVisionComponent):
    """Sharpens an image using unsharp masking.

    Args:
        amount (float): Sharpening strength. Default is 1.0.
        sigma (float): Gaussian blur sigma for the mask. Default is 1.0.
    """

    def __init__(self, amount=1.0, sigma=1.0):
        """Initializes Sharpen with amount and sigma.

        Args:
            amount (float): Sharpening intensity. Default is 1.0.
            sigma (float): Gaussian sigma for mask. Default is 1.0.
        """
        super().__init__()
        self.amount = amount
        self.sigma = sigma

    def _execute(self, data, config):
        """Applies unsharp masking to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'amount' and 'sigma'.

        Returns:
            NumPy array or dict: Sharpened image in the same format as input.
        """
        frame = extract_frame(data)
        amount = float(config.get("amount", self.amount))
        sigma = float(config.get("sigma", self.sigma))
        blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
        output = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        return replace_frame(data, output)


class Deblur(AIVisionComponent):
    """Reduces blur artifacts using unsharp masking.

    Args:
        amount (float): Deblur strength. Default is 1.0.
        sigma (float): Gaussian blur sigma for the mask. Default is 1.0.
    """

    def __init__(self, amount=1.0, sigma=1.0):
        """Initializes Deblur with amount and sigma.

        Args:
            amount (float): Deblurring strength. Default is 1.0.
            sigma (float): Gaussian sigma for mask. Default is 1.0.
        """
        super().__init__()
        self.amount = amount
        self.sigma = sigma

    def _execute(self, data, config):
        """Applies deblurring to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'amount' and 'sigma'.

        Returns:
            NumPy array or dict: Deblurred image in the same format as input.
        """
        frame = extract_frame(data)
        amount = float(config.get("amount", self.amount))
        sigma = float(config.get("sigma", self.sigma))
        blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
        output = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        return replace_frame(data, output)


class Threshold(AIVisionComponent):
    """Applies a fixed-level binary threshold to convert an image to a binary mask.

    Args:
        threshold (float): Pixel intensity threshold. Default is 127.
        max_value (float): Value assigned to pixels above the threshold. Default is 255.
        mode (str): Threshold type ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv'). Default is 'binary'.
        keep_channels (bool): If True, expands the single-channel result to 3 channels. Default is False.
    """

    def __init__(self, threshold=127, max_value=255, mode="binary", keep_channels=False):
        """Initializes Threshold with threshold value and mode.

        Args:
            threshold (float): Pixel threshold. Default is 127.
            max_value (float): Output max value. Default is 255.
            mode (str): Threshold mode. Default is 'binary'.
            keep_channels (bool): Expand to 3 channels. Default is False.
        """
        super().__init__()
        self.threshold = threshold
        self.max_value = max_value
        self.mode = mode
        self.keep_channels = keep_channels

    def _execute(self, data, config):
        """Applies thresholding to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'threshold_mode' / 'mode',
                'threshold', 'max_value', 'keep_channels'.

        Returns:
            NumPy array or dict: Binary mask image in the same format as input.
        """
        frame = extract_frame(data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        mode = _resolve_threshold_mode(config.get("threshold_mode", config.get("mode", self.mode)))
        _, output = cv2.threshold(
            gray,
            float(config.get("threshold", self.threshold)),
            float(config.get("max_value", self.max_value)),
            mode,
        )
        output = maybe_grayscale_to_bgr(output, config.get("keep_channels", self.keep_channels))
        return replace_frame(data, output)


class AdaptiveThreshold(AIVisionComponent):
    """Applies adaptive thresholding, computing the threshold locally per region.

    Args:
        max_value (float): Output value for pixels above the threshold. Default is 255.
        method (str): Adaptive method ('gaussian' or 'mean'). Default is 'gaussian'.
        threshold_type (str): Threshold type ('binary' or 'binary_inv'). Default is 'binary'.
        block_size (int): Size of the local neighborhood window. Must be odd. Default is 11.
        c (float): Constant subtracted from the computed mean or weighted sum. Default is 2.
        keep_channels (bool): Expand output to 3 channels. Default is False.
    """

    def __init__(
        self,
        max_value=255,
        method="gaussian",
        threshold_type="binary",
        block_size=11,
        c=2,
        keep_channels=False,
    ):
        """Initializes AdaptiveThreshold with adaptive parameters.

        Args:
            max_value (float): Max output value. Default is 255.
            method (str): Adaptive method. Default is 'gaussian'.
            threshold_type (str): Binary type. Default is 'binary'.
            block_size (int): Neighborhood window size (odd). Default is 11.
            c (float): Constant offset. Default is 2.
            keep_channels (bool): Expand to 3 channels. Default is False.
        """
        super().__init__()
        self.max_value = max_value
        self.method = method
        self.threshold_type = threshold_type
        self.block_size = block_size
        self.c = c
        self.keep_channels = keep_channels

    def _execute(self, data, config):
        """Applies adaptive thresholding to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'method', 'threshold_type',
                'block_size', 'c', 'max_value', 'keep_channels'.

        Returns:
            NumPy array or dict: Adaptive binary mask in the same format as input.
        """
        frame = extract_frame(data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if config.get("method", self.method).lower() == "mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        threshold_type = (
            cv2.THRESH_BINARY_INV
            if config.get("threshold_type", self.threshold_type).lower() == "binary_inv"
            else cv2.THRESH_BINARY
        )
        block_size = int(config.get("block_size", self.block_size))
        if block_size % 2 == 0:
            block_size += 1
        output = cv2.adaptiveThreshold(
            gray,
            float(config.get("max_value", self.max_value)),
            adaptive_method,
            threshold_type,
            block_size,
            float(config.get("c", self.c)),
        )
        output = maybe_grayscale_to_bgr(output, config.get("keep_channels", self.keep_channels))
        return replace_frame(data, output)


class EdgeDetection(AIVisionComponent):
    """Detects edges in an image using Canny, Sobel, or Laplacian methods.

    Args:
        method (str): Edge detection algorithm ('canny', 'sobel', or 'laplacian'). Default is 'canny'.
        threshold1 (float): First Canny threshold. Default is 100.
        threshold2 (float): Second Canny threshold. Default is 200.
        aperture_size (int): Sobel aperture size for Canny. Default is 3.
        keep_channels (bool): Expand single-channel output to 3 channels. Default is True.
    """

    def __init__(self, method="canny", threshold1=100, threshold2=200, aperture_size=3, keep_channels=True):
        """Initializes EdgeDetection with method and parameters.

        Args:
            method (str): Algorithm name. Default is 'canny'.
            threshold1 (float): Lower Canny threshold. Default is 100.
            threshold2 (float): Upper Canny threshold. Default is 200.
            aperture_size (int): Aperture for Canny Sobel kernels. Default is 3.
            keep_channels (bool): Expand to 3 channels. Default is True.
        """
        super().__init__()
        self.method = method
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.keep_channels = keep_channels

    def _execute(self, data, config):
        """Applies edge detection to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'edge_method' / 'method',
                'threshold1', 'threshold2', 'aperture_size', 'keep_channels'.

        Returns:
            NumPy array or dict: Edge map in the same format as input.

        Raises:
            ValueError: If method is not 'canny', 'sobel', or 'laplacian'.
        """
        frame = extract_frame(data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        method = config.get("edge_method", config.get("method", self.method)).lower()
        if method == "canny":
            output = cv2.Canny(
                gray,
                float(config.get("threshold1", self.threshold1)),
                float(config.get("threshold2", self.threshold2)),
                apertureSize=int(config.get("aperture_size", self.aperture_size)),
            )
        elif method == "sobel":
            dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            output = cv2.convertScaleAbs(cv2.magnitude(dx, dy))
        elif method == "laplacian":
            output = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
        else:
            raise ValueError("EdgeDetection supports canny, sobel, or laplacian.")
        output = maybe_grayscale_to_bgr(output, config.get("keep_channels", self.keep_channels))
        return replace_frame(data, output)


class ContourExtraction(AIVisionComponent):
    """Extracts contours from a binary image and optionally draws them.

    Adds 'contours' and 'hierarchy' keys to the payload dict.

    Args:
        retrieval_mode (str): Contour retrieval mode ('external', 'list', 'tree', 'ccomp'). Default is 'external'.
        approximation (str): Contour approximation method ('simple', 'none', 'tc89_l1', 'tc89_kcos'). Default is 'simple'.
        draw (bool): If True, draws contours onto the frame. Default is True.
        color (tuple[int, int, int]): BGR contour drawing color. Default is (0, 255, 0).
        thickness (int): Contour line thickness. Default is 2.
    """

    def __init__(self, retrieval_mode="external", approximation="simple", draw=True, color=(0, 255, 0), thickness=2):
        """Initializes ContourExtraction with retrieval and drawing parameters.

        Args:
            retrieval_mode (str): Contour retrieval hierarchy. Default is 'external'.
            approximation (str): Approximation method. Default is 'simple'.
            draw (bool): Draw contours on frame. Default is True.
            color (tuple[int, int, int]): Drawing color. Default is (0, 255, 0).
            thickness (int): Line thickness. Default is 2.
        """
        super().__init__()
        self.retrieval_mode = retrieval_mode
        self.approximation = approximation
        self.draw = draw
        self.color = color
        self.thickness = thickness

    def _execute(self, data, config):
        """Extracts contours from the extracted frame and optionally draws them.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'retrieval_mode', 'approximation',
                'draw', 'color', 'thickness'.

        Returns:
            dict: Payload with 'frame', 'contours', and 'hierarchy' keys if input was dict.
            numpy.ndarray: Frame with contours drawn if input was a raw array.
        """
        frame = extract_frame(data)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(
            binary,
            _resolve_retrieval_mode(config.get("retrieval_mode", self.retrieval_mode)),
            _resolve_approximation_mode(config.get("approximation", self.approximation)),
        )
        output = frame.copy()
        if config.get("draw", self.draw):
            cv2.drawContours(
                output,
                contours,
                -1,
                config.get("color", self.color),
                int(config.get("thickness", self.thickness)),
            )
        if isinstance(data, dict):
            data["frame"] = output
            data["contours"] = contours
            data["hierarchy"] = hierarchy
            return data
        return output


def _resolve_color_code(source, target):
    """Maps a (source, target) color space pair to the corresponding OpenCV conversion code.

    Args:
        source (str): Source color space name (e.g., 'BGR').
        target (str): Target color space name (e.g., 'RGB').

    Returns:
        int: OpenCV color conversion code.

    Raises:
        ValueError: If the (source, target) pair is not in the supported mapping.
    """
    mapping = {
        ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
        ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
        ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
        ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
        ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
        ("HSV", "BGR"): cv2.COLOR_HSV2BGR,
        ("BGR", "LAB"): cv2.COLOR_BGR2LAB,
        ("LAB", "BGR"): cv2.COLOR_LAB2BGR,
        ("BGR", "YCRCB"): cv2.COLOR_BGR2YCrCb,
        ("YCRCB", "BGR"): cv2.COLOR_YCrCb2BGR,
    }
    key = (source.upper(), target.upper())
    if key not in mapping:
        raise ValueError(f"Unsupported color conversion: {source} -> {target}")
    return mapping[key]


def _resolve_threshold_mode(name):
    """Maps a threshold mode name to the corresponding OpenCV flag.

    Args:
        name (str): Mode name ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv').

    Returns:
        int: OpenCV threshold flag. Defaults to cv2.THRESH_BINARY for unknown names.
    """
    mapping = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    return mapping.get(str(name).lower(), cv2.THRESH_BINARY)


def _resolve_retrieval_mode(name):
    """Maps a contour retrieval mode name to the corresponding OpenCV flag.

    Args:
        name (str): Retrieval mode name ('external', 'list', 'tree', 'ccomp').

    Returns:
        int: OpenCV retrieval flag. Defaults to cv2.RETR_EXTERNAL for unknown names.
    """
    mapping = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE,
        "ccomp": cv2.RETR_CCOMP,
    }
    return mapping.get(str(name).lower(), cv2.RETR_EXTERNAL)


def _resolve_approximation_mode(name):
    """Maps a contour approximation mode name to the corresponding OpenCV flag.

    Args:
        name (str): Approximation mode name ('simple', 'none', 'tc89_l1', 'tc89_kcos').

    Returns:
        int: OpenCV approximation flag. Defaults to cv2.CHAIN_APPROX_SIMPLE.
    """
    mapping = {
        "simple": cv2.CHAIN_APPROX_SIMPLE,
        "none": cv2.CHAIN_APPROX_NONE,
        "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
        "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
    }
    return mapping.get(str(name).lower(), cv2.CHAIN_APPROX_SIMPLE)
