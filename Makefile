.DEFAULT_GOAL := help
.PHONY: help install install-dev hooks lint format test test-fast \
        bump bump-patch bump-minor bump-major changelog \
        build clean release

PYTHON := uv run python
CZ     := uv run cz

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────

install: ## Install runtime dependencies only
	uv sync

install-dev: ## Install all dependencies including dev tools
	uv sync --dev

hooks: ## Install pre-commit hooks (commit-msg, pre-push, pre-commit)
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg
	uv run pre-commit install --hook-type pre-push

# ── Code quality ──────────────────────────────────────────────────────────────

lint: ## Run ruff, isort, and black in check mode (no writes)
	uv run ruff check .
	uv run isort --check-only .
	uv run black --check .

format: ## Auto-fix with ruff, isort, and black
	uv run ruff check --fix .
	uv run isort .
	uv run black .

# ── Tests ─────────────────────────────────────────────────────────────────────

test: ## Run full test suite
	uv run pytest

test-fast: ## Run tests, stop on first failure
	uv run pytest -x

# ── Versioning (commitizen) ───────────────────────────────────────────────────

bump: ## Auto-bump version from conventional commits
	$(CZ) bump --yes

bump-patch: ## Force patch bump (0.0.x)
	$(CZ) bump --increment PATCH --yes

bump-minor: ## Force minor bump (0.x.0)
	$(CZ) bump --increment MINOR --yes

bump-major: ## Force major bump (x.0.0)
	$(CZ) bump --increment MAJOR --yes

changelog: ## Regenerate CHANGELOG.md from git history
	$(CZ) changelog

# ── Build & release ───────────────────────────────────────────────────────────

build: ## Build sdist + wheel into dist/
	uv build

clean: ## Remove build artefacts
	rm -rf dist/ build/ *.egg-info

release: lint test bump build ## Lint → test → bump → build (local dry-run)
	@echo "Local release artefacts in dist/. Push to master to trigger CI publish."
