from .base import AIVisionComponent
import cv2
import json
import time
from pathlib import Path

class DatasetCollector(AIVisionComponent):
    """Saves frames and metadata to disk for ML training."""
    
    def setup(self, config):
        self.output_dir = Path(config.get('output_dir', 'dataset'))
        self.save_metadata = config.get('save_metadata', True)
        self.metadata_path = self.output_dir / "metadata.jsonl"
        self.is_initialized = True

    def _execute(self, data, config):
        frame = data["frame"] if isinstance(data, dict) else data
        
        # Determine if we should save this specific iteration
        if config.get("save_sample", False):
            label = config.get("label", "unknown")
            metadata = config.get("metadata", {})
            self._save_sample(frame, label, metadata)
            
        # Acts as a passthrough so the pipeline can continue unmodified
        return data

    def _save_sample(self, frame, label, metadata):
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