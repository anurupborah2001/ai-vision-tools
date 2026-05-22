# Contributing

## Development Setup

```bash
git clone https://github.com/anuborah/ai-vision-tools.git
cd ai-vision-tools

# Install with uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[all]"
pip install black ruff isort pytest pre-commit
```

## Pre-commit Hooks

```bash
pre-commit install
pre-commit install --hook-type pre-push
pre-commit install --hook-type commit-msg
```

Hooks enforce: `ruff`, `isort`, `black`, pre-commit standard hooks, Conventional Commits, and `pytest` on pre-push.

## Running Tests

```bash
pytest                                          # all tests
pytest tests/test_imports.py                   # base install boundary
pytest tests/test_preprocessing_components.py
pytest tests/test_basic_augmentations.py
pytest tests/test_advanced_augmentations.py
pytest tests/test_core_components.py
pytest tests/test_labeler_components.py
pytest tests/test_cli_file_processing.py
```

## Code Style

```bash
ruff check .     # lint
black .          # format
isort .          # import order
```

## Commit Convention

This repository uses [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Use for |
|------|---------|
| `feat:` | New component or feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | New or updated tests |
| `chore:` | Build, CI, tooling |
| `refactor:` | Code change without feature/fix |
| `perf:` | Performance improvement |

```bash
git commit -m "feat: add GridDistortion augmentation component"
git commit -m "fix: correct LetterboxResize aspect ratio for portrait images"
```

## Adding a New Component

1. Create the module file under the appropriate domain directory.
2. Subclass `AIVisionComponent` from `ai_vision_tool.core.base`.
3. Implement `_execute(self, data, config)`.
4. Add an entry to `_EXPORTS` in `ai_vision_tool/__init__.py`:
   ```python
   "MyNewClass": ("ai_vision_tool.my_domain.my_module", "MyNewClass"),
   ```
5. Write tests in `tests/`.
6. Add usage docs to the relevant `docs/user-guide/` page.
7. Add an `:::` directive to the relevant `docs/api/` page.

## Documenting with Docstrings

Use Google-style docstrings:

```python
class MyComponent(AIVisionComponent):
    """One-line summary.

    Longer description of what the component does, when to use it,
    and any important caveats.

    Args:
        param1 (int): Description of param1.
        param2 (str): Description of param2. Defaults to 'value'.
    """

    def _execute(self, data, config):
        """Run the component logic.

        Args:
            data: Input image (NumPy array) or payload dict with 'frame' key.
            config (dict): Runtime overrides.

        Returns:
            Processed result as NumPy array or payload dict.
        """
```

## Building Docs Locally

```bash
pip install "mkdocs-material>=9.5" "mkdocstrings[python]>=0.27" \
            "mkdocs-git-revision-date-localized-plugin>=1.2" \
            "mkdocs-minify-plugin>=0.8"

mkdocs serve     # live preview at http://localhost:8000
mkdocs build     # build static site to ./site/
```

## Release Process

See [PUBLISHING.md](https://github.com/anuborah/ai-vision-tools/blob/master/PUBLISHING.md) for the full release checklist.
