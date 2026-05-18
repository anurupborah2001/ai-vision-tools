from abc import ABC, abstractmethod

class AIVisionComponent(ABC):
    """
    Abstract base class for vision tasks. 
    Now supports runtime configurations and native batch processing.
    """
    def __init__(self):
        self.is_initialized = False

    def setup(self, config: dict):
        """Hardware/Model initialization based on config."""
        self.is_initialized = True

    def run(self, data=None, config=None):
        """
        The Template Method. Natively handles both single items and batches (lists).
        """
        config = config or {}
        
        # Lazy initialization
        if not self.is_initialized:
            self.setup(config)

        # Batch Processing Logic
        if isinstance(data, list):
            print(f"[{self.__class__.__name__}] Processing batch of {len(data)} items...")
            return [self._process_single(item, config) for item in data]
        
        # Single File Processing Logic
        return self._process_single(data, config)

    def _process_single(self, item, config):
        """Internal pipeline for a single data item."""
        processed_item = self.preprocess(item, config)
        result = self._execute(processed_item, config)
        return self.postprocess(result, config)

    @abstractmethod
    def _execute(self, data, config):
        """Child classes implement specific logic here."""
        pass

    def preprocess(self, data, config):
        return data

    def postprocess(self, result, config):
        return result

    def cleanup(self):
        """Release resources."""
        pass