from .base import AIVisionComponent

class AutoLabeller(AIVisionComponent):
  # 1. Define the unique method
    def download_model_weights(self, url):
        print(f"[{self.__class__.__name__}] Downloading weights from {url}...")
        # Download logic here...
        return True

    # 2. Execute it during the setup phase
    def setup(self, config):
        model_url = config.get('model_url', 'http://default-model-url.com')
        
        # Call the specific method internally
        self.download_model_weights(model_url)
        
        self.is_initialized = True

    def _execute(self, data, config):
        print(f"[{self.__class__.__name__}] Labeling {data}...")
        return data