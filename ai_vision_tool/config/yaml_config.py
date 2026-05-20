from __future__ import annotations

from pathlib import Path


class YAMLConfig:
    """YAML configuration file loader with dot-notation access and deep merge.

    Args:
        path: Path to YAML file.
    """

    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}
        self._load()

    def _load(self):
        try:
            import yaml
        except ImportError:
            raise ImportError("Install with: pip install pyyaml")
        with open(self._path) as f:
            self._data = yaml.safe_load(f) or {}

    def reload(self) -> None:
        self._load()

    def get(self, key: str, default=None):
        parts = key.split(".")
        node = self._data
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            elif isinstance(node, list):
                try:
                    node = node[int(p)]
                except (ValueError, IndexError):
                    return default
            else:
                return default
        return node

    def __getitem__(self, key: str):
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def as_dict(self) -> dict:
        return dict(self._data)

    def merge(self, other: dict) -> None:
        self._data = _deep_merge(self._data, other)

    def validate(self, schema: dict) -> None:
        for key, expected_type in schema.items():
            val = self.get(key)
            if val is None:
                raise ValueError(f"Missing required config key: {key!r}")
            if not isinstance(val, expected_type):
                raise TypeError(f"Config key {key!r}: expected {expected_type.__name__}, got {type(val).__name__}")


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
