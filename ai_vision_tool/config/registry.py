from __future__ import annotations

import inspect


class ComponentRegistry:
    """Central registry mapping string names to AIVisionComponent classes.

    Usage:
        registry = ComponentRegistry.instance()
        registry.register("Resize", Resize)
        comp = registry.build("Resize", width=640, height=640)
    """

    _singleton: ComponentRegistry | None = None

    def __init__(self):
        self._registry: dict[str, type] = {}

    @classmethod
    def instance(cls) -> ComponentRegistry:
        if cls._singleton is None:
            cls._singleton = cls()
            cls._singleton._auto_register()
        return cls._singleton

    def _auto_register(self):
        try:
            import ai_vision_tool as avt

            for name in avt.__all__:
                try:
                    obj = getattr(avt, name)
                    if inspect.isclass(obj):
                        self._registry.setdefault(name, obj)
                except Exception:
                    pass
        except Exception:
            pass

    def register(self, name: str, cls: type = None, override: bool = False):
        if cls is None:

            def decorator(c):
                self.register(name, c, override=override)
                return c

            return decorator
        if name in self._registry and not override:
            raise ValueError(
                f"Component {name!r} already registered. Use override=True to replace."
            )
        self._registry[name] = cls

    def get(self, name: str) -> type:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Component {name!r} not found. Available: {available}")
        return self._registry[name]

    def build(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

    def build_from_config(self, config: dict):
        cfg = dict(config)
        name = cfg.pop("type")
        return self.build(name, **cfg)

    def list_names(self) -> list[str]:
        return sorted(self._registry)
