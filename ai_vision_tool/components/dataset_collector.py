from ai_vision_tool.core.base import AIVisionComponent
import cv2
import json
import time
from pathlib import Path


class DatasetCollector(AIVisionComponent):
    """Saves frames and metadata to disk for ML training."""

    def setup(self, config):
        """Initializes output directories and metadata file path.

        Args:
            config (dict): Setup parameters. Supports:
                - 'output_dir' (str): Root directory for the dataset. Default is 'dataset'.
                - 'save_metadata' (bool): If True, appends JSON records to a metadata
                  JSONL file alongside saved images. Default is True.
        """
        self.output_dir = Path(config.get('output_dir', 'dataset'))
        self.save_metadata = config.get('save_metadata', True)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.is_initialized = True

    def _execute(self, data, config):
        """Optionally saves the current frame as a labeled training sample.

        Acts as a passthrough — returns data unchanged so the pipeline
        continues unmodified. A sample is only written when 'save_sample'
        is True in config.

        Args:
            data: Input image as NumPy array or payload dict with 'frame' key.
            config (dict): Runtime parameters. Supports:
                - 'save_sample' (bool): If True, saves this frame to disk. Default is False.
                - 'label' (str): Class label subdirectory name. Default is 'unknown'.
                - 'metadata' (dict): Arbitrary metadata appended to the JSONL record.
                  Default is {}.

        Returns:
            Same as input data, unchanged.
        """
        frame = data["frame"] if isinstance(data, dict) else data

        if config.get("save_sample", False):
            label = config.get("label", "unknown")
            metadata = config.get("metadata", {})
            self._save_sample(frame, label, metadata)

        return data

    def _save_sample(self, frame, label, metadata):
        """Writes a frame to disk and appends a metadata record to the JSONL file.

        Args:
            frame (numpy.ndarray): BGR image to save.
            label (str): Class label used as the subdirectory name.
            metadata (dict): Arbitrary key-value pairs to include in the JSONL record.
        """
        label_dir = self.output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time() * 1000)
        filename = f"{label}_{timestamp}.jpg"
        image_path = label_dir / filename

        cv2.imwrite(str(image_path), frame)

        if self.save_metadata:
            record = {
                "image_path": str(image_path),
                "label": label,
                "timestamp": timestamp,
                "metadata": metadata,
            }
            with open(self.metadata_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        print(f"[{self.__class__.__name__}] Saved sample to {image_path}")
