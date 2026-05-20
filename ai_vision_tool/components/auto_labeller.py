from .base import AIVisionComponent


class AutoLabeller(AIVisionComponent):
    """Downloads model weights and labels input data using the loaded model."""

    def download_model_weights(self, url):
        """Downloads model weights from the specified URL.

        Args:
            url (str): Remote URL of the model weights file.

        Returns:
            bool: True when the download completes successfully.
        """
        print(f"[{self.__class__.__name__}] Downloading weights from {url}...")
        return True

    def setup(self, config):
        """Initializes the component by downloading model weights.

        Args:
            config (dict): Setup parameters. Supports:
                - 'model_url' (str): URL to fetch model weights from.
                  Default is 'http://default-model-url.com'.
        """
        model_url = config.get('model_url', 'http://default-model-url.com')
        self.download_model_weights(model_url)
        self.is_initialized = True

    def _execute(self, data, config):
        """Applies auto-labeling to the input data.

        Args:
            data: Input to label (file path string, image array, or payload dict).
            config (dict): Runtime configuration (unused by default implementation).

        Returns:
            Same as input data, unchanged by the stub implementation.
        """
        print(f"[{self.__class__.__name__}] Labeling {data}...")
        return data
