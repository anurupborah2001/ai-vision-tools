# Publishing Guide

Releases are fully automated via GitHub Actions using commitizen semantic versioning.
Manual steps are only needed on first-time setup.

---

## Automated Release Flow

Every push to `master` triggers `semantic-versioning.yml`:

1. Analyses conventional commits since the last `vX.Y.Z` tag
2. Bumps version in `pyproject.toml` and `ai_vision_tool/__init__.py`
3. Commits the bump and pushes an annotated `vX.Y.Z` tag to `master`
4. Creates a GitHub Release with changelog notes and distribution assets
5. Publishes to PyPI via OIDC trusted publishing (no token needed)

No version is created if there are no releasable commits (`feat:`, `fix:`, `perf:`,
`refactor:`, or `BREAKING CHANGE:`) since the last tag.

### Version bump rules

| Commit prefix | Bump |
|---------------|------|
| `fix:`, `perf:`, `refactor:` | patch (0.4.x) |
| `feat:` | minor (0.x.0) |
| `BREAKING CHANGE:` footer or `feat!:`/`fix!:` | major (x.0.0) |

---

## One-Time Setup: PyPI Trusted Publisher

PyPI trusted publishing uses OIDC — no API token or `.pypirc` required.
Configure once per project on pypi.org:

1. Go to **https://pypi.org/manage/account/publishing/**
2. Add a **pending trusted publisher** with:

   | Field | Value |
   |-------|-------|
   | Project name | `ai-vision-tool` |
   | Owner | `anurupborah2001` |
   | Repository | `ai-vision-tools` |
   | Workflow filename | `semantic-versioning.yml` |
   | Environment | `pypi` |

3. In the GitHub repository settings, create a **`pypi` environment**
   under **Settings → Environments** (no secrets needed — OIDC handles auth).

For TestPyPI rehearsals add a second publisher with workflow filename
`publish-pypi.yml` and environment `testpypi`, then trigger:

```bash
gh workflow run publish-pypi.yml \
  -f tag=vX.Y.Z \
  -f repository_url=https://test.pypi.org/legacy/
```

---

## Manual Release Trigger

Force a release without waiting for a qualifying commit:

```bash
gh workflow run semantic-versioning.yml -f bump=patch   # or minor / major
```

---

## Commit Message Convention

This repository enforces [Conventional Commits](https://www.conventionalcommits.org/)
via `pre-commit` at the `commit-msg` stage.

```
feat: add fog and rain augmentation pipeline stage
fix: clamp crop bounds to prevent negative slice indices
perf: replace nested loop with vectorised numpy op in mosaic
docs: update CLI usage examples in README
test: add coverage for weather augmentation components
chore: bump dev dependency versions
```

Install hooks locally:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
uv run pre-commit install --hook-type commit-msg
```

---

## Local Quality Gates

Run before opening a PR:

```bash
uv run pre-commit run --all-files
uv run pytest
```

Smoke-test the package in a clean environment:

```bash
uv build
uv venv /tmp/avt-check && source /tmp/avt-check/bin/activate
pip install dist/*.whl
python -c "import ai_vision_tool; print(ai_vision_tool.__version__)"
ai-vision-tool --help
deactivate
```

---

## Hotfix / Out-of-Band Tag

If you need to publish from an existing tag without a new bump:

```bash
gh workflow run publish-pypi.yml -f tag=vX.Y.Z
```

This re-runs the publish job against the specified tag without touching the version files.
