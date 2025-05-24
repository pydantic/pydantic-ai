.DEFAULT_GOAL := all

.PHONY: .uv
.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: .deno
.deno: ## Check that deno is installed
	@deno --version > /dev/null 2>&1 || (printf "\033[0;31m✖ Error: deno is not installed, but is needed for mcp-run-python\033[0m\n    Please install deno: https://docs.deno.com/runtime/getting_started/installation/\n" && exit 1)

.PHONY: install
install: .uv .pre-commit .deno ## Install the package, dependencies, and pre-commit for local development
	uv sync --frozen --all-extras --all-packages --group lint --group docs
	pre-commit install --install-hooks

.PHONY: install-all-python
install-all-python: ## Install and synchronize an interpreter for every python version
	UV_PROJECT_ENVIRONMENT=.venv39 uv sync --python 3.9 --frozen --all-extras --all-packages --group lint --group docs
	UV_PROJECT_ENVIRONMENT=.venv310 uv sync --python 3.10 --frozen --all-extras --all-packages --group lint --group docs
	UV_PROJECT_ENVIRONMENT=.venv311 uv sync --python 3.11 --frozen --all-extras --all-packages --group lint --group docs
	UV_PROJECT_ENVIRONMENT=.venv312 uv sync --python 3.12 --frozen --all-extras --all-packages --group lint --group docs
	UV_PROJECT_ENVIRONMENT=.venv313 uv sync --python 3.13 --frozen --all-extras --all-packages --group lint --group docs

.PHONY: sync
sync: .uv ## Update local packages and uv.lock
	uv sync --all-extras --all-packages --group lint --group docs

.PHONY: format
format: ## Format the code
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint: ## Lint the code
	uv run ruff format --check
	uv run ruff check

.PHONY: lint-js
lint-js: ## Lint JS and TS code
	cd mcp-run-python && deno task lint-format

.PHONY: typecheck-pyright
typecheck-pyright:
	@# PYRIGHT_PYTHON_IGNORE_WARNINGS avoids the overhead of making a request to github on every invocation
	PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright

.PHONY: typecheck-mypy
typecheck-mypy:
	uv run mypy

.PHONY: typecheck
typecheck: typecheck-pyright ## Run static type checking

.PHONY: typecheck-both  ## Run static type checking with both Pyright and Mypy
typecheck-both: typecheck-pyright typecheck-mypy

.PHONY: test
test: ## Run tests and collect coverage data
	uv run coverage run -m pytest
	@uv run coverage report

.PHONY: test-fast
test-fast: ## Same as test except no coverage. ~1/4th the time depending on hardware.
	uv run pytest -n auto --dist=loadgroup

.PHONY: test-all-python
test-all-python: ## Run tests on Python 3.9 to 3.13
	UV_PROJECT_ENVIRONMENT=.venv39 uv run --python 3.9 --all-extras --all-packages coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv310 uv run --python 3.10 --all-extras --all-packages coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv311 uv run --python 3.11 --all-extras --all-packages coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv312 uv run --python 3.12 --all-extras --all-packages coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv313 uv run --python 3.13 --all-extras --all-packages coverage run -p -m pytest
	@uv run coverage combine
	@uv run coverage report

.PHONY: testcov
testcov: test ## Run tests and generate a coverage report
	@echo "building coverage html"
	@uv run coverage html

.PHONY: test-mrp
test-mrp: ## Build and  tests of mcp-run-python
	cd mcp-run-python && deno task build
	uv run --package mcp-run-python pytest mcp-run-python -v

.PHONY: update-examples
update-examples: ## Update documentation examples
	uv run -m pytest --update-examples tests/test_examples.py

.PHONY: update-vcr-tests
update-vcr-tests: ## Update tests using VCR that hit LLM APIs; note you'll need to set API keys as appropriate
	uv run -m pytest --record-mode=rewrite tests

# `--no-strict` so you can build the docs without insiders packages
.PHONY: docs
docs: ## Build the documentation
	uv run mkdocs build --no-strict

# `--no-strict` so you can build the docs without insiders packages
.PHONY: docs-serve
docs-serve: ## Build and serve the documentation
	uv run mkdocs serve --no-strict

.PHONY: .docs-insiders-install
.docs-insiders-install: ## Install insiders packages for docs if necessary
ifeq ($(shell uv pip show mkdocs-material | grep -q insiders && echo 'installed'), installed)
	@echo 'insiders packages already installed'
else ifeq ($(PPPR_TOKEN),)
	@echo "Error: PPPR_TOKEN is not set, can't install insiders packages"
	@exit 1
else
	@echo 'installing insiders packages...'
	@uv pip install --reinstall --no-deps \
		--extra-index-url https://pydantic:${PPPR_TOKEN}@pppr.pydantic.dev/simple/ \
		mkdocs-material mkdocstrings-python
endif

.PHONY: docs-insiders
docs-insiders: .docs-insiders-install ## Build the documentation using insiders packages
	uv run --no-sync mkdocs build -f mkdocs.insiders.yml

.PHONY: docs-serve-insiders
docs-serve-insiders: .docs-insiders-install ## Build and serve the documentation using insiders packages
	uv run --no-sync mkdocs serve -f mkdocs.insiders.yml

.PHONY: cf-pages-build
cf-pages-build: ## Install uv, install dependencies and build the docs, used on CloudFlare Pages
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.12
	uv sync --python 3.12 --frozen --group docs
	uv pip install --reinstall --no-deps \
		--extra-index-url https://pydantic:${PPPR_TOKEN}@pppr.pydantic.dev/simple/ \
		mkdocs-material mkdocstrings-python
	uv pip freeze
	uv run --no-sync mkdocs build -f mkdocs.insiders.yml

.PHONY: all
all: format lint typecheck testcov ## Run code formatting, linting, static type checks, and tests with coverage report generation

.PHONY: help
help: ## Show this help (usage: make help)
	@echo "Usage: make [recipe]"
	@echo "Recipes:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
