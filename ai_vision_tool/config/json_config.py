from __future__ import annotations

import json


class JSONConfig:
    """JSON configuration loader with dot-notation access and save support.

    Args:
        path: Path to JSON file (or None when using from_dict).
    """

    def __init__(self, path: str | None = None):
        self._path = path
        self._data: dict = {}
        if path:
            self._load()

    def _load(self):
        with open(self._path) as f:
            self._data = json.load(f)

    def reload(self) -> None:
        if self._path:
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
                raise TypeError(
                    f"Config key {key!r}: expected {expected_type.__name__}, got {type(val).__name__}"
                )

    def save(self, path: str | None = None) -> None:
        target = path or self._path
        if not target:
            raise ValueError("No path specified for save")
        with open(target, "w") as f:
            json.dump(self._data, f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> JSONConfig:
        cfg = cls()
        cfg._data = dict(d)
        return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
