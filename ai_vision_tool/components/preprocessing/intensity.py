from __future__ import annotations

import cv2
import numpy as np

from .._image_utils import extract_frame, maybe_grayscale_to_bgr, replace_frame, to_uint8
from ..base import AIVisionComponent


class Normalize(AIVisionComponent):
    def __init__(self, output_min=0.0, output_max=1.0):
        super().__init__()
        self.output_min = output_min
        self.output_max = output_max

    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        output_min = float(config.get("output_min", self.output_min))
        output_max = float(config.get("output_max", self.output_max))
        normalized = cv2.normalize(frame, None, alpha=output_min, beta=output_max, norm_type=cv2.NORM_MINMAX)
        return replace_frame(data, normalized)


class Standardize(AIVisionComponent):
    def __init__(self, mean=None, std=None, per_channel=True, epsilon=1e-6):
        super().__init__()
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        self.epsilon = epsilon

    def _execute(self, data, config):
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
    def __init__(self, scale=1.0 / 255.0, offset=0.0, clip=False):
        super().__init__()
        self.scale = scale
        self.offset = offset
        self.clip = clip

    def _execute(self, data, config):
        frame = extract_frame(data).astype(np.float32)
        scale = float(config.get("pixel_scale", config.get("scale", self.scale)))
        offset = float(config.get("pixel_offset", config.get("offset", self.offset)))
        clip = config.get("clip", self.clip)
        output = frame * scale + offset
        if clip:
            output = np.clip(output, 0.0, 1.0)
        return replace_frame(data, output)


class ConvertColorSpace(AIVisionComponent):
    def __init__(self, source="BGR", target="RGB", keep_channels=True):
        super().__init__()
        self.source = source
        self.target = target
        self.keep_channels = keep_channels

    def _execute(self, data, config):
        frame = extract_frame(data)
        source = config.get("source", self.source).upper()
        target = config.get("target", self.target).upper()
        keep_channels = config.get("keep_channels", self.keep_channels)
        code = _resolve_color_code(source, target)
        output = cv2.cvtColor(frame, code)
        output = maybe_grayscale_to_bgr(output, keep_channels)
        return replace_frame(data, output)


class BGRToRGB(ConvertColorSpace):
    def __init__(self):
        super().__init__(source="BGR", target="RGB")


class RGBToBGR(ConvertColorSpace):
    def __init__(self):
        super().__init__(source="RGB", target="BGR")


class CLAHE(AIVisionComponent):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def _execute(self, data, config):
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
    def __init__(self, color_space="ycrcb"):
        super().__init__()
        self.color_space = color_space

    def _execute(self, data, config):
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
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def _execute(self, data, config):
        frame = extract_frame(data)
        gamma = max(float(config.get("gamma", self.gamma)), 1e-6)
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
        output = cv2.LUT(frame, table)
        return replace_frame(data, output)


class WhiteBalance(AIVisionComponent):
    def __init__(self, method="gray_world", percentile=95):
        super().__init__()
        self.method = method
        self.percentile = percentile

    def _execute(self, data, config):
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
    def __init__(self, method="nlm", strength=10, kernel_size=5, sigma_color=75, sigma_space=75):
        super().__init__()
        self.method = method
        self.strength = strength
        self.kernel_size = kernel_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def _execute(self, data, config):
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
    def __init__(self, amount=1.0, sigma=1.0):
        super().__init__()
        self.amount = amount
        self.sigma = sigma

    def _execute(self, data, config):
        frame = extract_frame(data)
        amount = float(config.get("amount", self.amount))
        sigma = float(config.get("sigma", self.sigma))
        blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
        output = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        return replace_frame(data, output)


class Deblur(AIVisionComponent):
    def __init__(self, amount=1.0, sigma=1.0):
        super().__init__()
        self.amount = amount
        self.sigma = sigma

    def _execute(self, data, config):
        frame = extract_frame(data)
        amount = float(config.get("amount", self.amount))
        sigma = float(config.get("sigma", self.sigma))
        blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
        output = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
        return replace_frame(data, output)


class Threshold(AIVisionComponent):
    def __init__(self, threshold=127, max_value=255, mode="binary", keep_channels=False):
        super().__init__()
        self.threshold = threshold
        self.max_value = max_value
        self.mode = mode
        self.keep_channels = keep_channels

    def _execute(self, data, config):
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
    def __init__(
        self,
        max_value=255,
        method="gaussian",
        threshold_type="binary",
        block_size=11,
        c=2,
        keep_channels=False,
    ):
        super().__init__()
        self.max_value = max_value
        self.method = method
        self.threshold_type = threshold_type
        self.block_size = block_size
        self.c = c
        self.keep_channels = keep_channels

    def _execute(self, data, config):
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
    def __init__(self, method="canny", threshold1=100, threshold2=200, aperture_size=3, keep_channels=True):
        super().__init__()
        self.method = method
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.aperture_size = aperture_size
        self.keep_channels = keep_channels

    def _execute(self, data, config):
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
    def __init__(self, retrieval_mode="external", approximation="simple", draw=True, color=(0, 255, 0), thickness=2):
        super().__init__()
        self.retrieval_mode = retrieval_mode
        self.approximation = approximation
        self.draw = draw
        self.color = color
        self.thickness = thickness

    def _execute(self, data, config):
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
    mapping = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    return mapping.get(str(name).lower(), cv2.THRESH_BINARY)


def _resolve_retrieval_mode(name):
    mapping = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE,
        "ccomp": cv2.RETR_CCOMP,
    }
    return mapping.get(str(name).lower(), cv2.RETR_EXTERNAL)


def _resolve_approximation_mode(name):
    mapping = {
        "simple": cv2.CHAIN_APPROX_SIMPLE,
        "none": cv2.CHAIN_APPROX_NONE,
        "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
        "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
    }
    return mapping.get(str(name).lower(), cv2.CHAIN_APPROX_SIMPLE)
