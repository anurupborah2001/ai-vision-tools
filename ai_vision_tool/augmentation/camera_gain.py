from __future__ import annotations

from ..core.base import AIVisionComponent
from ..utils.image_utils import extract_frame, replace_frame, to_uint8


class CameraGain(AIVisionComponent):
    """Simulates camera sensor gain and black-level offset.

    Multiplies pixel values by a gain factor and adds a black-level offset,
    mimicking analog sensor amplification.

    Args:
        gain (float): Multiplicative gain applied to all pixel channels. Default is 1.0.
        black_level (float): Additive offset applied after gain. Default is 0.0.
    """

    def __init__(self, gain=1.0, black_level=0.0):
        """Initializes CameraGain with gain and black-level parameters.

        Args:
            gain (float): Multiplicative gain. Default is 1.0.
            black_level (float): Additive black-level offset. Default is 0.0.
        """
        super().__init__()
        self.gain = gain
        self.black_level = black_level

    def _execute(self, data, config):
        """Applies gain and black-level adjustment to the extracted frame.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime overrides. Supports 'gain' and 'black_level'.

        Returns:
            NumPy array or dict: Adjusted image clipped to uint8 range, in the same format as input.
        """
        frame = extract_frame(data)
        gain = float(config.get("gain", self.gain))
        black_level = float(config.get("black_level", self.black_level))
        adjusted = frame.astype("float32") * gain + black_level
        output = to_uint8(adjusted)
        return replace_frame(data, output)
