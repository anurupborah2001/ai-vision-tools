# Phase A Design: Rename & Restructure `ai-vision-flow` → `ai-vision-tool`

**Date:** 2026-05-18
**Status:** Approved
**Scope:** Rename only — no logic changes, no new features
**Next phase:** Phase B (CLI rewrite: argparse → Typer + Rich)

---

## Context

The existing package is `ai-vision-flow` with Python module `visionflow`. This phase renames it to `ai-vision-tool` / `ai_vision_tool` in-place, renames the `template/` submodule to `capture/`, fixes broken imports in `capture/video_template.py`, and scaffolds empty stub directories for future phases.

---

## Module Layout

### Before

```
visionflow/
├── __init__.py
├── __main__.py
├── api.py
├── api_service.py
├── cli.py
├── components/
│   ├── base.py
│   ├── augmentations/
│   ├── preprocessing/
│   └── [all component files]
├── pipeline/
└── template/
    ├── __init__.py
    ├── image_template.py
    ├── video_template.py       ← broken imports
    ├── screen_capture.py
    └── video_recorder.py
```

### After

```
ai_vision_tool/
├── __init__.py                  (lazy exports preserved, paths updated)
├── __main__.py
├── api.py
├── api_service.py
├── cli.py
├── components/                  (preserved as-is)
│   ├── base.py
│   ├── augmentations/
│   ├── preprocessing/
│   └── [all component files]
├── pipeline/
├── capture/                     ← renamed from template/
│   ├── __init__.py
│   ├── image_template.py
│   ├── video_template.py        ← broken imports fixed
│   ├── screen_capture.py
│   └── video_recorder.py
├── core/                        ← stub
├── detection/                   ← stub
├── segmentation/                ← stub
├── enhancement/                 ← stub
├── tracking/                    ← stub
├── streaming/                   ← stub
├── io/                          ← stub
├── visualization/               ← stub
├── models/                      ← stub
├── utils/                       ← stub
├── config/                      ← stub
└── pipelines/                   ← stub
```

---

## Broken Imports Fixed

`capture/video_template.py` currently has two broken imports that reference packages outside the library:

| Broken import | Fix |
|---|---|
| `from module.fps_counter import FPSCounter` | Inline FPS counter using `cv2.getTickCount()` / `cv2.getTickFrequency()` |
| `from template.video_recorder import VideoRecorder` | `from ai_vision_tool.capture.video_recorder import VideoRecorder` |

No other files have broken imports. The `template/` references in `cli.py` are string literals in the examples catalog (not imports) and must be updated to reference `ai_vision_tool.capture`.

---

## Packaging Changes

| Field | Before | After |
|---|---|---|
| `[project] name` | `ai-vision-flow` | `ai-vision-tool` |
| `[tool.poetry] name` | `ai-vision-flow` | `ai-vision-tool` |
| `packages` | `[{ include = "visionflow" }]` | `[{ include = "ai_vision_tool" }]` |
| `[project.scripts]` key | `ai-vision-flow` | `ai-vision-tool` |
| script entrypoint | `visionflow.cli:main` | `ai_vision_tool.cli:main` |
| api entrypoint | `visionflow.api:run` | `ai_vision_tool.api:run` |
| `version` | `0.1.0` | `0.2.0` |
| `__version__` in `__init__.py` | `"0.1.0"` | `"0.2.0"` |

`uv.lock` is regenerated after rename. GitHub Actions workflows need no changes (package-name-agnostic).

---

## Commit Sequence

### Commit 1 — `chore: rename package ai-vision-flow → ai-vision-tool`

- Rename directory `visionflow/` → `ai_vision_tool/`
- Update `pyproject.toml` (name, packages, scripts, version → 0.2.0)
- Update `main.py` root wrapper import
- Update `__version__` in `__init__.py`

**State after:** package metadata correct; tests fail (imports still say `visionflow`)

---

### Commit 2 — `chore: update all internal imports to ai_vision_tool`

Files touched:
- All `*.py` inside `ai_vision_tool/` — replace `from visionflow` / `import visionflow`
- `tests/*.py` — replace all `visionflow` references
- `main.py` — already updated in commit 1

**State after:** `pytest` passes; CLI entrypoint resolves correctly

---

### Commit 3 — `chore: rename template → capture, fix broken imports`

- Rename `ai_vision_tool/template/` → `ai_vision_tool/capture/`
- Fix `capture/video_template.py`:
  - Remove `from module.fps_counter import FPSCounter`; inline FPS logic
  - Fix `from template.video_recorder import VideoRecorder` → `from ai_vision_tool.capture.video_recorder import VideoRecorder`
- Update `cli.py` examples catalog: 3 template entries reference `visionflow.template` — update to `ai_vision_tool.capture`
- Update `tests/test_capture_components.py` if any paths reference `template`

**State after:** `from ai_vision_tool.capture import VideoRecorder` works; tests pass

---

### Commit 4 — `chore: scaffold stub modules for future phases`

Create `ai_vision_tool/{core,detection,segmentation,enhancement,tracking,streaming,io,visualization,models,utils,config,pipelines}/`.

Each stub `__init__.py` contains only a one-line module docstring:

```python
"""[Module name] — future phase placeholder."""
```

**State after:** Full target layout exists; tests unaffected

---

## Acceptance Criteria

- `pip install .` succeeds with package name `ai-vision-tool`
- `ai-vision-tool --help` runs correctly
- `ai-vision-tool-api` entrypoint resolves
- `from ai_vision_tool import Flip` works (lazy import preserved)
- `from ai_vision_tool.capture import VideoRecorder` works
- `from ai_vision_tool.capture.video_template import video_capture_template` works
- `pytest` passes (all existing tests green)
- No references to `visionflow` remain in any non-git file
- All 12 stub modules exist with `__init__.py`

---

## Out of Scope for Phase A

- CLI rewrite (Typer + Rich) → Phase B
- Streaming/capture module expansion → Phase C
- Augmentation pipeline DX → Phase D
- New AI feature interfaces (detection, segmentation, etc.) → future phases
- README rewrite → after Phase B
- New tests → each feature phase adds its own
