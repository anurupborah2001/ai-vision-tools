from __future__ import annotations

import os
from pathlib import Path


class EnvConfig:
    """Reads configuration from environment variables with a given prefix.

    Args:
        prefix: Env var prefix (default "AI_VISION"). Reads AI_VISION_{KEY}.
    """

    def __init__(self, prefix: str = "AI_VISION"):
        self._prefix = prefix.upper().rstrip("_")

    def _env_key(self, key: str) -> str:
        return f"{self._prefix}_{key.upper()}"

    def get(self, key: str, default=None, cast: type = str):
        val = os.environ.get(self._env_key(key))
        if val is None:
            return default
        return self._cast(val, cast)

    def require(self, key: str, cast: type = str):
        val = os.environ.get(self._env_key(key))
        if val is None:
            raise ValueError(f"Required env var {self._env_key(key)!r} is not set")
        return self._cast(val, cast)

    @staticmethod
    def _cast(val: str, cast: type):
        if cast is bool:
            return val.lower() in ("true", "1", "yes", "on")
        if cast is list:
            return [v.strip() for v in val.split(",") if v.strip()]
        return cast(val)

    def as_dict(self) -> dict[str, str]:
        prefix = self._prefix + "_"
        return {
            k[len(prefix):].lower(): v
            for k, v in os.environ.items()
            if k.startswith(prefix)
        }

    # Common keys with defaults
    @property
    def device(self) -> str:
        return self.get("device", "auto")

    @property
    def log_level(self) -> str:
        return self.get("log_level", "INFO")

    @property
    def model_cache_dir(self) -> str:
        return self.get("model_cache_dir", str(Path.home() / ".cache" / "ai_vision_tool" / "models"))

    @property
    def api_host(self) -> str:
        return self.get("api_host", "0.0.0.0")

    @property
    def api_port(self) -> int:
        return self.get("api_port", 8000, cast=int)

    @property
    def max_workers(self) -> int:
        return self.get("max_workers", 4, cast=int)
