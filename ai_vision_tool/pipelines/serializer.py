from __future__ import annotations

import inspect
import json
from pathlib import Path


class PipelineSerializer:
    """Serializes and deserializes AIVisionPipeline to/from YAML or JSON.

    Pipeline format:
        version: "1.0"
        name: "my_pipeline"
        global_config: {}
        steps:
          - type: "Resize"
            args:
              width: 640
              height: 640
    """

    def to_dict(self, pipeline) -> dict:
        steps = []
        components = getattr(pipeline, "processors", getattr(pipeline, "_components", []))
        for comp in components:
            name = comp.__class__.__name__
            args = self._get_component_args(comp)
            steps.append({"type": name, "args": args})
        return {
            "version": "1.0",
            "name": getattr(pipeline, "name", "pipeline"),
            "global_config": getattr(pipeline, "global_config", {}),
            "steps": steps,
        }

    def _get_component_args(self, component) -> dict:
        try:
            sig = inspect.signature(component.__class__.__init__)
            params = {k: v for k, v in sig.parameters.items() if k != "self"}
            args = {}
            for name, param in params.items():
                if hasattr(component, name):
                    val = getattr(component, name)
                    try:
                        json.dumps(val)
                        args[name] = val
                    except (TypeError, ValueError):
                        args[name] = str(val)
            return args
        except Exception:
            return {}

    def from_dict(self, d: dict):
        from ai_vision_tool.pipelines.vision_pipeline import AIVisionPipeline
        from ai_vision_tool.config.registry import ComponentRegistry
        registry = ComponentRegistry.instance()
        pipeline = AIVisionPipeline()
        for step in d.get("steps", []):
            cfg = dict(step)
            name = cfg.pop("type")
            args = cfg.pop("args", {})
            try:
                comp = registry.build(name, **args)
                pipeline.add(comp)
            except Exception as e:
                print(f"[PipelineSerializer] Warning: could not build {name!r}: {e}")
        return pipeline

    def save(self, pipeline, path: str, format: str = "yaml") -> None:
        d = self.to_dict(pipeline)
        p = Path(path)
        if format == "yaml":
            try:
                import yaml
                p.write_text(yaml.dump(d, default_flow_style=False))
            except ImportError:
                raise ImportError("Install with: pip install pyyaml")
        else:
            p.write_text(json.dumps(d, indent=2))

    def load(self, path: str):
        p = Path(path)
        if p.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
                d = yaml.safe_load(p.read_text())
            except ImportError:
                raise ImportError("Install with: pip install pyyaml")
        else:
            d = json.loads(p.read_text())
        return self.from_dict(d)
