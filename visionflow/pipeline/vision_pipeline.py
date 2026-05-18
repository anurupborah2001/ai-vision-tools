from ..components.base import AIVisionComponent

class AIVisionPipeline:
    """
    Implements a Fluent Builder and Chain of Responsibility pattern.
    Passes the output of one processor as the input to the next.
    """
    def __init__(self):
        self.processors = []

    def add(self, processor: AIVisionComponent):
        """Adds a processor to the pipeline chain."""
        self.processors.append(processor)
        return self

    def execute(self, initial_data=None, global_config=None):
        """Runs the data sequentially through all added processors."""
        global_config = global_config or {}
        current_data = initial_data

        print("\n=== Starting Vision Pipeline ===")
        for processor in self.processors:
            current_data = processor.run(data=current_data, config=global_config)
            
        print("=== Pipeline Complete ===\n")
        return current_data