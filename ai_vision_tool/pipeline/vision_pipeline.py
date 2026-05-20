from ..core.base import AIVisionComponent


class AIVisionPipeline:
    """Chains vision components using a Fluent Builder and Chain of Responsibility pattern.

    Each call to ``add`` appends a processor. ``execute`` runs them in order,
    feeding the output of one processor as the input to the next.
    """

    def __init__(self):
        """Initializes an empty pipeline with no processors."""
        self.processors = []

    def add(self, processor: AIVisionComponent):
        """Appends a component to the end of the pipeline chain.

        Args:
            processor (AIVisionComponent): Component to add.

        Returns:
            AIVisionPipeline: This pipeline instance, for method chaining.
        """
        self.processors.append(processor)
        return self

    def execute(self, initial_data=None, global_config=None):
        """Runs the input data sequentially through all added processors.

        Each processor receives the output of the previous one. The same
        ``global_config`` dict is passed to every processor unchanged.

        Args:
            initial_data: Starting input — NumPy array, payload dict, list,
                or any value accepted by the first processor.
            global_config (dict or None): Configuration dict forwarded to every
                processor's ``run`` call. Defaults to an empty dict if not provided.

        Returns:
            Output of the last processor in the chain, or ``initial_data`` if
            the pipeline contains no processors.
        """
        global_config = global_config or {}
        current_data = initial_data

        print("\n=== Starting Vision Pipeline ===")
        for processor in self.processors:
            current_data = processor.run(data=current_data, config=global_config)

        print("=== Pipeline Complete ===\n")
        return current_data
