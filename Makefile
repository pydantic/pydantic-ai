.DEFAULT_GOAL := all

.PHONY: .uv  # Check that uv is installed
.uv:
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit  # Check that pre-commit is installed
.pre-commit:
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install  # Install the package, dependencies, and pre-commit for local development
install: .uv .pre-commit
	uv sync --frozen --all-extras --group lint --group docs
	pre-commit install --install-hooks

.PHONY: sync  # Update local packages and uv.lock
sync: .uv
	uv sync --all-extras --group lint --group docs

.PHONY: format  # Format the code
format:
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint  # Lint the code
lint:
	uv run ruff format --check
	uv run ruff check

.PHONY: typecheck-pyright
typecheck-pyright:
	uv run pyright

.PHONY: typecheck-mypy
typecheck-mypy:
	uv run mypy --strict tests/typed_agent.py

.PHONY: typecheck  # Run static type checking
typecheck: typecheck-pyright

.PHONY: typecheck-both  # Run static type checking with both Pyright and Mypy
typecheck-both: typecheck-pyright typecheck-mypy

.PHONY: test  # Run tests and collect coverage data
test:
	uv run coverage run -m pytest
	@uv run coverage report

.PHONY: test-all-python  # Run tests on Python 3.9 to 3.13
test-all-python:
	UV_PROJECT_ENVIRONMENT=.venv39 uv run --python 3.9 --all-extras coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv310 uv run --python 3.10 --all-extras coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv311 uv run --python 3.11 --all-extras coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv312 uv run --python 3.12 --all-extras coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv313 uv run --python 3.13 --all-extras coverage run -p -m pytest
	@uv run coverage combine
	@uv run coverage report

.PHONY: testcov  # Run tests and generate a coverage report
testcov: test
	@echo "building coverage html"
	@uv run coverage html

# `--no-strict` so you can build the docs without insiders packages
.PHONY: docs  # Build the documentation
docs:
	uv run mkdocs build --no-strict

# `--no-strict` so you can build the docs without insiders packages
.PHONY: docs-serve  # Build and serve the documentation
docs-serve:
	uv run mkdocs serve --no-strict

.PHONY: .docs-insiders-install # install insiders packages for docs if necessary
.docs-insiders-install:
ifeq ($(shell uv pip show mkdocs-material | grep -q insiders && echo 'installed'), installed)
	@echo 'insiders packages already installed'
else ifeq ($(PPPR_TOKEN),)
	@echo "Error: PPPR_TOKEN is not set, can't install insiders packages"
	@exit 1
else
	@echo 'installing insiders packages...'
	@uv pip install -U mkdocs-material mkdocstrings-python \
		--extra-index-url https://pydantic:${PPPR_TOKEN}@pppr.pydantic.dev/simple/
endif

.PHONY: docs-insiders  # Build the documentation using insiders packages
docs-insiders: .docs-insiders-install
	uv run --no-sync mkdocs build -f mkdocs.insiders.yml

.PHONY: docs-serve-insiders  # Build and serve the documentation using insiders packages
docs-serve-insiders: .docs-insiders-install
	uv run --no-sync mkdocs serve -f mkdocs.insiders.yml

.PHONY: cf-pages-build  # Install uv, install dependencies and build the docs, used on CloudFlare Pages
cf-pages-build:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.12
	uv sync --python 3.12 --frozen --group docs
	uv pip install -U \
		--extra-index-url https://pydantic:$(PPPR_TOKEN)@pppr.pydantic.dev/simple/ \
		mkdocs-material mkdocstrings-python
	uv pip freeze
	uv run --no-sync mkdocs build

.PHONY: all
all: format lint typecheck testcov
