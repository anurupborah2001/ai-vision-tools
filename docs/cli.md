# CLI Reference

The `ai-vision-tool` CLI provides two modes: **live webcam processing** and **batch image processing**.

```bash
ai-vision-tool [OPTIONS]
```

---

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--webcam` | flag | — | Launch live webcam processing loop |
| `--process-image-path PATH` | path | — | Process a single image or directory of images |
| `--profile PATH` | path | — | JSON augmentation profile (component config) |
| `--example CATEGORY` | str | — | Print Python usage examples for a category |
| `--list-examples` | flag | — | List all available example categories |

---

## Live Webcam Mode

```bash
ai-vision-tool --webcam
```

Opens the default camera (index 0) and applies the active augmentation profile in real time. Press `q` to quit.

**With a custom profile:**

```bash
ai-vision-tool --webcam --profile ./my_profile.json
```

---

## Batch Image Processing

```bash
# Single file
ai-vision-tool --process-image-path ./photo.jpg

# Directory (processes all images recursively)
ai-vision-tool --process-image-path ./dataset/images/
```

Results are saved alongside originals with a `_processed` suffix by default.

---

## Augmentation Profiles

A profile is a JSON file that configures which components to apply and with what parameters:

```json
{
  "components": [
    {
      "name": "Resize",
      "params": { "width": 640, "height": 640 }
    },
    {
      "name": "CLAHE",
      "params": { "clip_limit": 3.0, "tile_grid_size": [8, 8] }
    },
    {
      "name": "Flip",
      "params": { "direction": "horizontal", "probability": 0.5 }
    }
  ]
}
```

Load a profile in Python:

```python
from ai_vision_tool.augmentation.common import parse_component_profile

components = parse_component_profile("my_profile.json")
```

---

## Usage Examples

List all available example categories:

```bash
ai-vision-tool --list-examples
```

Print Python examples for a category:

```bash
ai-vision-tool --example preprocessing
ai-vision-tool --example augmentations
ai-vision-tool --example capture
ai-vision-tool --example components
ai-vision-tool --example all
```

---

## Examples

```bash
# Quick test with webcam and default settings
ai-vision-tool --webcam

# Process a folder with a profile
ai-vision-tool --process-image-path ./raw/ --profile augment.json

# Show all preprocessing examples
ai-vision-tool --example preprocessing
```
