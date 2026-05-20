from __future__ import annotations

import json
import os
from pathlib import Path


class ProfileLoader:
    """Loads JSON/YAML augmentation/pipeline profiles from configurable search paths.

    Args:
        search_paths: List of directories to search for profiles.
    """

    def __init__(self, search_paths: list[str] | None = None):
        self._paths = [Path(p) for p in (search_paths or [])]
        self._paths.append(Path("./profiles"))

    def load(self, name_or_path: str) -> dict:
        p = Path(name_or_path)
        if p.is_absolute() or p.exists():
            return self._read(p)
        for base in self._paths:
            for ext in (".json", ".yaml", ".yml"):
                candidate = base / (name_or_path + ext)
                if candidate.exists():
                    return self._read(candidate)
                candidate2 = base / name_or_path
                if candidate2.exists():
                    return self._read(candidate2)
        raise FileNotFoundError(f"Profile {name_or_path!r} not found in {[str(p) for p in self._paths]}")

    def _read(self, path: Path) -> dict:
        suffix = path.suffix.lower()
        with open(path) as f:
            if suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    return yaml.safe_load(f) or {}
                except ImportError:
                    raise ImportError("Install with: pip install pyyaml")
            return json.load(f)

    def load_pipeline(self, profile: dict):
        from ai_vision_tool.config.registry import ComponentRegistry
        registry = ComponentRegistry.instance()
        components = []
        for step in profile.get("steps", []):
            cfg = dict(step)
            name = cfg.pop("type")
            args = cfg.pop("args", cfg)
            try:
                comp = registry.build(name, **args)
                components.append(comp)
            except Exception as e:
                print(f"[ProfileLoader] Warning: could not build {name!r}: {e}")
        return components

    def list_profiles(self) -> list[str]:
        found = []
        for base in self._paths:
            if base.exists():
                for f in base.iterdir():
                    if f.suffix.lower() in (".json", ".yaml", ".yml"):
                        found.append(f.stem)
        return sorted(set(found))

    def save_profile(self, name: str, config: dict) -> None:
        for base in self._paths:
            try:
                base.mkdir(parents=True, exist_ok=True)
                dest = base / (name + ".json")
                with open(dest, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"[ProfileLoader] Saved to {dest}")
                return
            except OSError:
                continue
        raise OSError("No writable search path found")
