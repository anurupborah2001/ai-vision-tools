from abc import ABC, abstractmethod


class AIVisionComponent(ABC):
    """Abstract base class for all vision pipeline components.

    Implements the Template Method pattern. Subclasses override ``_execute``
    to provide component-specific logic. The ``run`` method handles both single
    items and lists, performing lazy initialization and delegating to
    ``preprocess`` → ``_execute`` → ``postprocess``.
    """

    def __init__(self):
        """Initializes the component in an uninitialized state."""
        self.is_initialized = False

    def setup(self, config: dict):
        """Performs hardware or model initialization before first use.

        Called lazily the first time ``run`` is invoked. Override to load
        model weights, open device handles, or allocate GPU resources.

        Args:
            config (dict): Configuration dict passed to ``run``.
        """
        self.is_initialized = True

    def run(self, data=None, config=None):
        """Executes the component on a single item or a batch.

        Lazily calls ``setup`` on first invocation, then routes to
        ``_process_single`` for each item.

        Args:
            data: Input data — a single NumPy array, payload dict, or a list
                of either for batch processing.
            config (dict or None): Runtime configuration overrides. Defaults
                to an empty dict if not provided.

        Returns:
            Processed result in the same structure as input: a single output
            for a single input, or a list of outputs for a batch.
        """
        config = config or {}

        if not self.is_initialized:
            self.setup(config)

        if isinstance(data, list):
            print(f"[{self.__class__.__name__}] Processing batch of {len(data)} items...")
            return [self._process_single(item, config) for item in data]

        return self._process_single(data, config)

    def _process_single(self, item, config):
        """Runs the preprocess → execute → postprocess pipeline for one item.

        Args:
            item: A single input (NumPy array or payload dict).
            config (dict): Runtime configuration dict.

        Returns:
            Output of ``postprocess``, which defaults to the result of ``_execute``.
        """
        processed_item = self.preprocess(item, config)
        result = self._execute(processed_item, config)
        return self.postprocess(result, config)

    @abstractmethod
    def _execute(self, data, config):
        """Component-specific processing logic. Must be implemented by subclasses.

        Args:
            data: Pre-processed input (NumPy array or payload dict).
            config (dict): Runtime configuration overrides.

        Returns:
            Processed result (NumPy array or payload dict).
        """
        pass

    def preprocess(self, data, config):
        """Optional hook to transform input before ``_execute``.

        Args:
            data: Raw input passed to ``run``.
            config (dict): Runtime configuration dict.

        Returns:
            Transformed input. Default implementation returns data unchanged.
        """
        return data

    def postprocess(self, result, config):
        """Optional hook to transform output after ``_execute``.

        Args:
            result: Output returned by ``_execute``.
            config (dict): Runtime configuration dict.

        Returns:
            Final result. Default implementation returns result unchanged.
        """
        return result

    def cleanup(self):
        """Releases any resources held by the component.

        Override to close file handles, release video writers, or free GPU memory.
        """
        pass
