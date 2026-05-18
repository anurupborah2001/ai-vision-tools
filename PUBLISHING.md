# Publishing Guide

This project is packaged with Poetry and published as a PyPI distribution. The release flow below keeps local validation, commit conventions, and PyPI upload steps aligned.

## 1. Prepare the release

Review these fields before publishing:

- `version` in [pyproject.toml](/Users/anuborah@sphnet.com.sg/IdeaProjects/ai-vision-flow/pyproject.toml:1)
- `authors`
- description, keywords, and classifiers
- optional project URLs if you want homepage, docs, or issue tracker links on PyPI

Install the project with development tooling:

```bash
poetry install --with dev
```

## 2. Follow commit policy

This repository now enforces Conventional Commits through `pre-commit` at the `commit-msg` stage.

Examples:

- `feat: add advanced weather augmentations`
- `fix: clamp crop bounds in augmentation component`
- `docs: expand publishing guide`
- `test: add coverage for basic augmentation classes`
- `test: expand preprocessing component coverage`

Install the hooks locally if you have not already:

```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
poetry run pre-commit install --hook-type commit-msg
```

## 3. Run local quality gates

Run the hooks and tests before building:

```bash
poetry run pre-commit run --all-files
poetry run pytest
```

Recommended focused checks before a release:

- verify `README.md` examples still match the CLI and import paths
- verify `main.py --help` reflects the current flags
- verify the package imports cleanly with `poetry run python -c "import visionflow"`
- run the preprocessing-focused suite with `poetry run pytest tests/test_preprocessing_components.py`

## 4. Build distributions

```bash
poetry run python -m build
```

This creates:

- `dist/*.tar.gz`
- `dist/*.whl`

## 5. Validate the distributions

```bash
poetry run python -m twine check dist/*
```

Recommended additional smoke tests:

```bash
python -m venv /tmp/ai-vision-flow-release-check
source /tmp/ai-vision-flow-release-check/bin/activate
pip install dist/*.whl
python -c "import visionflow; print(visionflow.__version__)"
ai-vision-flow --help
deactivate
```

## 6. Upload to TestPyPI

```bash
poetry run python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment if you want a full rehearsal before production release.

## 7. Upload to PyPI

```bash
poetry run python -m twine upload dist/*
```

## 8. Tag and announce the release

After the PyPI upload succeeds:

- create a git tag for the released version
- push the tag
- update any changelog or release notes you maintain externally

## Environment variables

Use API tokens instead of passwords:

```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-..."
```

## Notes

- Do not commit `.pypirc` with secrets.
- If you change dependencies, package exports, or package layout, rebuild before uploading.
- If you publish optional extras, confirm they are documented clearly in `README.md`.
