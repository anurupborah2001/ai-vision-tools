# Configuration

The `config` module provides YAML/JSON config loaders with dot-notation access, a component registry with decorator-based registration, environment variable loading, and profile-based pipeline loading.

```python
from ai_vision_tool.config import <ClassName>
```

---

## YAMLConfig

Load, merge, validate, and hot-reload YAML configuration files.

```python
from ai_vision_tool.config import YAMLConfig

config = YAMLConfig("pipeline.yaml")

# Dot-notation access
model_path = config.get("detection.model_path")
threshold = config.get("detection.threshold", default=0.5)

# Merge another config (values in other override self)
config.merge(YAMLConfig("overrides.yaml"))

# Validate required keys
config.validate(required=["detection.model_path", "output.dir"])

# Hot-reload (reads file again)
config.reload()
```

---

## JSONConfig

Same interface as `YAMLConfig` plus `save()` and `from_dict()`.

```python
from ai_vision_tool.config import JSONConfig

# From file
config = JSONConfig("settings.json")

# From dict
config = JSONConfig.from_dict({"threshold": 0.5, "max_det": 100})

# Save back to disk
config.save("settings_updated.json")
```

---

## EnvConfig

Load configuration from environment variables with automatic type casting.

```python
from ai_vision_tool.config import EnvConfig

# All env vars with prefix "VISION_" are loaded
env = EnvConfig(prefix="VISION_")

# VISION_THRESHOLD=0.5 → float
threshold = env.get("THRESHOLD", cast=float, default=0.5)

# VISION_DEVICE=cuda → str
device = env.get("DEVICE", default="cpu")

# Require a variable — raises if not set
api_key = env.require("API_KEY")
```

---

## ComponentRegistry

A singleton registry for discovering and instantiating components by name.

```python
from ai_vision_tool.config import ComponentRegistry

registry = ComponentRegistry.instance()

# Register a custom component
@registry.register("my_vignette")
class Vignette(AIVisionComponent):
    ...

# Build from name + params
component = registry.build("Resize", {"width": 640, "height": 480})
component = registry.build("my_vignette", {"strength": 0.5})

# List all registered names
print(registry.list_all())
```

---

## ProfileLoader

Load and instantiate pipeline configurations from YAML/JSON profiles.

```python
from ai_vision_tool.config import ProfileLoader

loader = ProfileLoader(
    search_paths=["./profiles", "~/.ai_vision_tool/profiles"],
)

# Load a pipeline from a profile file
pipeline = loader.load_pipeline("augment_v2")   # looks for augment_v2.yaml / .json

# Load from explicit path
pipeline = loader.load_pipeline("/path/to/profile.yaml")
```

**Profile format:**

```yaml
components:
  - class: Resize
    params: { width: 640, height: 640 }
  - class: CLAHE
    params: { clip_limit: 3.0 }
  - class: Flip
    params: { direction: horizontal, probability: 0.5 }
```
