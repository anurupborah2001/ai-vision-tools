from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor


class AsyncPipeline:
    """Asyncio-based vision pipeline for non-blocking frame processing.

    Args:
        components: List of AIVisionComponent instances.
        global_config: Config dict passed to every component.
    """

    def __init__(
        self, components: list | None = None, global_config: dict | None = None
    ):
        self._components = list(components or [])
        self.global_config = global_config or {}
        self._executor = ThreadPoolExecutor()

    def add(self, component) -> AsyncPipeline:
        self._components.append(component)
        return self

    async def execute(self, data) -> object:
        loop = asyncio.get_event_loop()
        config = self.global_config
        for comp in self._components:
            data = await loop.run_in_executor(self._executor, comp.run, data, config)
        return data

    async def execute_batch(self, items: list) -> list:
        tasks = [self.execute(item) for item in items]
        return await asyncio.gather(*tasks)

    async def stream(self, source_iter) -> AsyncGenerator:
        """Process frames from an async iterable and yield results."""
        async for frame in source_iter:
            result = await self.execute(frame)
            yield result

    def cleanup(self):
        for comp in self._components:
            if hasattr(comp, "cleanup"):
                comp.cleanup()
        self._executor.shutdown(wait=False)


class AsyncComponent:
    """Mixin for components that support async execution."""

    async def arun(self, data, config: dict | None = None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, data, config or {})
